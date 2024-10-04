import Mathlib
import Mathlib.Algebra.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Prime
import Mathlib.Algebra.Quotient
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.FalconerSeries
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.Trigonometry.Tan
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Cycle
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Base
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Log
import Mathlib.Geometry.Euclidean.Angle
import Mathlib.Geometry.Euclidean.Circumcenter
import Mathlib.Polynomial
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Tactics.Basic
import Mathlib.Topology.Affine.Bases

namespace spotted_and_fluffy_cats_l580_580498

theorem spotted_and_fluffy_cats (total_cats : ℕ) (total_cats_eq : total_cats = 120) 
  (spotted_fraction : ℚ) (spotted_fraction_eq : spotted_fraction = 1/3)
  (fluffy_fraction : ℚ) (fluffy_fraction_eq : fluffy_fraction = 1/4) :
  let spotted_cats := (total_cats * spotted_fraction).natAbs in
  let fluffy_spotted_cats := (spotted_cats * fluffy_fraction).natAbs in
  fluffy_spotted_cats = 10 :=
by
  sorry

end spotted_and_fluffy_cats_l580_580498


namespace probability_of_ace_then_spade_l580_580421

theorem probability_of_ace_then_spade :
  let P := (1 / 52) * (12 / 51) + (3 / 52) * (13 / 51)
  P = (3 / 127) :=
by
  sorry

end probability_of_ace_then_spade_l580_580421


namespace a_general_term_theorem_T_k_theorem_R_lambda_theorem_l580_580247

variable (n : ℕ)
variable (a : ℕ → ℕ)
variable (b : ℕ → ℕ)
variable (c : ℕ → ℝ)
variable (S : ℕ → ℕ)
variable (T : ℕ → ℝ)
variable (R : ℕ → ℝ)

-- Given conditions for sequence a
axiom a_cond : ∀ n : ℕ, S n = 2 * a n - 2

-- Given conditions for sequence b
axiom b_cond1 : ∀ n : ℕ, b (n + 1) = 2 * b n - 2^(n + 1)
axiom b_cond2 : b 1 = 8

-- Given conditions for sequence c
axiom a_general_term : ∀ n : ℕ, a n = 2^n
axiom c_cond : ∀ n : ℕ, c n = (a (n + 1) : ℝ) / ((1 + a n) * (1 + a (n + 1) : ℝ))

-- Sum definitions
def T_def : ℕ → ℝ := λ n, (∑ i in range n, b i)
def R_def : ℕ → ℝ := λ n, (∑ i in range n, c i)

-- Proofs as Lean Theorems
theorem a_general_term_theorem : ∀ n : ℕ, a n = 2^n := by {
    sorry
}

theorem T_k_theorem : ∃ k : ℕ, (k = 4 ∨ k = 5) ∧ ∀ n : ℕ, T k ≥ T n := by {
    sorry
}

theorem R_lambda_theorem : ∀ n : ℕ, R n < (2 / 3 : ℝ) := by {
    sorry
}

end a_general_term_theorem_T_k_theorem_R_lambda_theorem_l580_580247


namespace monotonicity_and_inequality_l580_580625

noncomputable def f (x : ℝ) : ℝ := (Real.log (1 + x)) / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (1 - a * x) / (1 + x)

theorem monotonicity_and_inequality :
  (∀ x > 0, f x < (Real.log (1 + x)) / x ∧ ∀ x ∈ Ioo 0 (Real.to_nnreal (Real.infinity.to_nat () |>.nonpos_completed)), 
         f x < g x (-1/2)) :=
by
  sorry

end monotonicity_and_inequality_l580_580625


namespace no_such_function_exists_l580_580902

theorem no_such_function_exists (f : ℕ → ℕ) : ¬ ∀ x : ℕ, f(f(x)) = x + 1 :=
sorry

end no_such_function_exists_l580_580902


namespace count_factors_of_144_multiple_of_18_l580_580985

def is_factor (n k : ℕ) : Prop := k ∣ n

def is_multiple (x y : ℕ) : Prop := y ∣ x

theorem count_factors_of_144_multiple_of_18 : 
  (finset.card (finset.filter (λ x, is_multiple x 18) (finset.filter (λ k, is_factor 144 k) (finset.range 145)))) = 4 := 
sorry

end count_factors_of_144_multiple_of_18_l580_580985


namespace spotted_and_fluffy_cats_l580_580496

theorem spotted_and_fluffy_cats (total_cats : ℕ) (total_cats_equiv : total_cats = 120) (one_third_spotted : ℕ → ℕ) (one_fourth_fluffy_spotted : ℕ → ℕ) :
  (one_third_spotted total_cats * one_fourth_fluffy_spotted (one_third_spotted total_cats) = 10) :=
by
  sorry

end spotted_and_fluffy_cats_l580_580496


namespace karthik_read_fraction_l580_580698

theorem karthik_read_fraction (total_pages pages_unread: ℕ) (first_fraction : ℚ) 
  (h₀ : total_pages = 468) 
  (h₁ : first_fraction = 7 / 13) 
  (h₂ : pages_unread = 96) : 
  (remaining_fraction := 5 / 9) := 
by
  sorry

end karthik_read_fraction_l580_580698


namespace sally_has_18_nickels_and_total_value_98_cents_l580_580988

-- Define the initial conditions
def pennies_initial := 8
def nickels_initial := 7
def nickels_from_dad := 9
def nickels_from_mom := 2

-- Define calculations based on the initial conditions
def total_nickels := nickels_initial + nickels_from_dad + nickels_from_mom
def value_pennies := pennies_initial
def value_nickels := total_nickels * 5
def total_value := value_pennies + value_nickels

-- State the theorem to prove the correct answers
theorem sally_has_18_nickels_and_total_value_98_cents :
  total_nickels = 18 ∧ total_value = 98 := 
by {
  -- Proof goes here
  sorry
}

end sally_has_18_nickels_and_total_value_98_cents_l580_580988


namespace product_of_roots_proof_l580_580204

noncomputable def product_of_roots : ℚ :=
  let leading_coeff_poly1 := 3
  let leading_coeff_poly2 := 4
  let constant_term_poly1 := -15
  let constant_term_poly2 := 9
  let a := leading_coeff_poly1 * leading_coeff_poly2
  let b := constant_term_poly1 * constant_term_poly2
  (b : ℚ) / a

theorem product_of_roots_proof :
  product_of_roots = -45/4 :=
by
  sorry

end product_of_roots_proof_l580_580204


namespace triangle_perpendicular_l580_580663

variables {Point : Type} [EuclideanGeometry Point]

/-- Given an acute-angled triangle MKN, where KL is a bisector and point X on the side MK such that KX = KN. 
    O is the circumcenter of the triangle MKN. Prove that the lines KO and XL are perpendicular. -/
theorem triangle_perpendicular (M K N L X O F : Point)
  (h_triangle : Triangle M K N)
  (h_acute : AcuteTriangle M K N)
  (h_bisector : IsBisector K L M N)
  (h_pointX_eq : SegmentLength K X = SegmentLength K N)
  (h_circumcenter : IsCircumcenter O M K N)
  (h_intersect : IntersectsAt (LineThrough K O) (LineThrough X L) F) :
  IsPerpendicular (LineThrough K O) (LineThrough X L) :=
sorry

end triangle_perpendicular_l580_580663


namespace pen_cost_is_14_l580_580658

variable (students : ℕ)
variable (s : ℕ)
variable (c : ℕ)
variable (n : ℕ)
variable (total_cost : ℕ)

-- Conditions
def condition_1 := students = 50
def condition_2 := s > 25
def condition_3 := n > 1
def condition_4 := c > n
def condition_5 := total_cost = 2310
def condition_6 := s * c * n = total_cost

-- Goal: Determine the cost of one pen
noncomputable def pen_cost : ℕ :=
  if condition_1 ∧ condition_2 ∧ condition_3 ∧ condition_4 ∧ condition_5 ∧ condition_6 then c else 0

theorem pen_cost_is_14 : pen_cost students s c n total_cost = 14 :=
by
  unfold pen_cost
  rw [condition_1, condition_2, condition_3, condition_4, condition_5, condition_6]
  sorry

end pen_cost_is_14_l580_580658


namespace upper_bound_neg_expr_l580_580329

theorem upper_bound_neg_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  - (1 / (2 * a) + 2 / b) ≤ - (9 / 2) := 
sorry

end upper_bound_neg_expr_l580_580329


namespace solve_complex_eq_l580_580997

theorem solve_complex_eq (z : ℂ) (h : z = complex.I * (2 - z)) : z = 1 + complex.I := by
  sorry

end solve_complex_eq_l580_580997


namespace inequality_does_not_hold_l580_580276

theorem inequality_does_not_hold (x y : ℝ) (h : x > y) : ¬ (-3 * x > -3 * y) :=
by {
  sorry
}

end inequality_does_not_hold_l580_580276


namespace find_divisor_exists_four_numbers_in_range_l580_580415

theorem find_divisor_exists_four_numbers_in_range :
  ∃ n : ℕ, (n > 1) ∧ (∀ k : ℕ, 39 ≤ k ∧ k ≤ 79 → ∃ a : ℕ, k = n * a) ∧ (∃! (k₁ k₂ k₃ k₄ : ℕ), 39 ≤ k₁ ∧ k₁ ≤ 79 ∧ 39 ≤ k₂ ∧ k₂ ≤ 79 ∧ 39 ≤ k₃ ∧ k₃ ≤ 79 ∧ 39 ≤ k₄ ∧ k₄ ≤ 79 ∧ k₁ ≠ k₂ ∧ k₁ ≠ k₃ ∧ k₁ ≠ k₄ ∧ k₂ ≠ k₃ ∧ k₂ ≠ k₄ ∧ k₃ ≠ k₄ ∧ k₁ % n = 0 ∧ k₂ % n = 0 ∧ k₃ % n = 0 ∧ k₄ % n = 0) → n = 19 :=
by sorry

end find_divisor_exists_four_numbers_in_range_l580_580415


namespace find_angle_PQF_l580_580322

open Real EuclideanGeometry

section parabola_problem

def parabola (x : ℝ) : ℝ := - (1 / 4) * x ^ 2

def focus : ℝ × ℝ := (0, -1)

def tangent_at_P : AffineSpace.ConvexLine ℝ := { x | let y := 2 * x + 4 in y }

def P : ℝ × ℝ := (-4, -4)

def Q : ℝ × ℝ := (-2, 0)

theorem find_angle_PQF :
  ∠ (P - Q) (focus - Q) = π / 2 :=
sorry

end parabola_problem

end find_angle_PQF_l580_580322


namespace complex_eq_real_imag_l580_580217

noncomputable def complex_mul (x y : ℂ) : ℂ := x * y

theorem complex_eq_real_imag (a b : ℝ) (z : ℂ) (hz : (3 + 4 * complex.I) * (1 + a * complex.I) = b * complex.I) : a = 3 / 4 :=
by
  have h : (3 - 4 * a) + (3 * a + 4) * complex.I = b * complex.I := by sorry
  have h_real : 3 - 4 * a = 0 := by sorry
  exact eq_of_sub_eq_zero (by rw [← h_real, sub_self])

end complex_eq_real_imag_l580_580217


namespace smallest_positive_value_floor_l580_580336

def g (x : ℝ) : ℝ := 3 * Real.sin x - Real.cos x + 4 * Real.cot x

theorem smallest_positive_value_floor :
  (∃ x : ℝ, x > 0 ∧ g(x) = 0 ∧ x < π) ∧ (∀ x : ℝ, g(x) = 0 → x > 0 → x < π → x ≥ π / 2) →
  ⌊inf {x : ℝ | g x = 0 ∧ x > 0}⌋ = 2 :=
by
  sorry

end smallest_positive_value_floor_l580_580336


namespace one_liter_fills_five_cups_l580_580739

-- Define the problem conditions and question in Lean 4
def one_liter_milliliters : ℕ := 1000
def cup_volume_milliliters : ℕ := 200

theorem one_liter_fills_five_cups : one_liter_milliliters / cup_volume_milliliters = 5 := 
by 
  sorry -- proof skipped

end one_liter_fills_five_cups_l580_580739


namespace trigonometric_identity_l580_580238

variable (α : ℝ)

theorem trigonometric_identity (h : sin(α + π / 6) - cos α = 1 / 3) :
  2 * sin α * cos (α + π / 6) = 5 / 18 :=
sorry

end trigonometric_identity_l580_580238


namespace similar_triangles_height_ratio_l580_580072

theorem similar_triangles_height_ratio (area_ratio : ℝ) (h₁ : ℝ) (h₂ : ℝ) 
  (similar : Boolean) (h₁_value : h₁ = 5) (area_ratio_value : area_ratio = 9) :
  similar = true → area_ratio = (h₂ / h₁) ^ 2 → h₂ = 15 :=
by
  intro h_similar area_eq
  rw [h₁_value, area_ratio_value]
  sorry

end similar_triangles_height_ratio_l580_580072


namespace sample_size_l580_580854

theorem sample_size (f r n : ℕ) (freq_def : f = 36) (rate_def : r = 25 / 100) (relation : r = f / n) : n = 144 :=
sorry

end sample_size_l580_580854


namespace length_of_train_l580_580101

noncomputable def speed_kmph := 6 -- speed in km/hr
def time_sec := 2 -- time in seconds
noncomputable def speed_mps := 5 / 3 -- converted speed in m/s (6 km/hr = 5/3 m/s)
noncomputable def length_train := speed_mps * time_sec -- length of the train

theorem length_of_train : length_train = 3.34 :=
by
  sorry

end length_of_train_l580_580101


namespace impossible_sum_of_50_numbers_l580_580681

theorem impossible_sum_of_50_numbers :
  ¬ ∃ (a : Fin 50 → ℝ),
    ∀ k : Fin 34, (∑ i in (Fin.range (k.val + 17)).filter (λ i : Fin 50, i.val ≥ k.val), a i) > 0 :=
by
  sorry

end impossible_sum_of_50_numbers_l580_580681


namespace isosceles_triangle_points_on_geoboard_l580_580833

theorem isosceles_triangle_points_on_geoboard 
    (geoboard_size : ℕ)
    (segment_length : ℕ)
    (possible_points : ℕ) :
    geoboard_size = 7 → 
    segment_length = 3 → 
    possible_points = 6 → 
    ∃C, (number_of_points_making_triangle_ABC_isosceles geoboard_size segment_length possible_points) = 6 :=
by { sorry }

end isosceles_triangle_points_on_geoboard_l580_580833


namespace remainder_third_smallest_div_second_smallest_l580_580020

def five_numbers : List ℕ := [10, 11, 12, 13, 14]

theorem remainder_third_smallest_div_second_smallest :
  ∃ (a b : ℕ), a = five_numbers.nthLe 2 (by simp) ∧ b = five_numbers.nthLe 1 (by simp) ∧ a % b = 1 :=
by
  let a := (five_numbers.nthLe 2 (by simp))
  let b := (five_numbers.nthLe 1 (by simp))
  use a, b
  simp [a, b]
  sorry

end remainder_third_smallest_div_second_smallest_l580_580020


namespace range_log_sqrt_sin_l580_580091

theorem range_log_sqrt_sin (x : ℝ) (hx : 0 < x ∧ x < π) :
  ∃ y, y = log 3 (real.sqrt (real.sin x)) ∧ y ∈ set.Iic (0 : ℝ) :=
sorry

end range_log_sqrt_sin_l580_580091


namespace continuous_at_0_l580_580394

noncomputable def f (x : ℝ) : ℝ := (real.cbrt (1 + x) - 1) / (real.sqrt (4 + x) - 2)

theorem continuous_at_0 :
  ∃ L, tendsto f (nhds 0) (nhds L) ∧ f 0 = L ∧ L = 4 / 3 :=
begin
  sorry
end

end continuous_at_0_l580_580394


namespace polynomial_simplification_l580_580801

theorem polynomial_simplification (w : ℝ) : 
  3 * w + 4 - 6 * w - 5 + 7 * w + 8 - 9 * w - 10 + 2 * w ^ 2 = 2 * w ^ 2 - 5 * w - 3 :=
by
  sorry

end polynomial_simplification_l580_580801


namespace largest_average_is_10_l580_580096

open Nat

def average (s : Finset ℕ) : ℚ :=
  (s.sum id : ℚ) / s.card

def multiples (n d : ℕ) (b : ℕ) : Finset ℕ :=
  (Finset.range b).filter (λ x => x % d = 0)

def avg_7 : ℚ := average (multiples 7 7 (201 + 1))
def avg_8 : ℚ := average (multiples 8 8 (201 + 1))
def avg_9 : ℚ := average (multiples 9 9 (201 + 1))
def avg_10 : ℚ := average (multiples 10 10 (201 + 1))
def avg_11 : ℚ := average (multiples 11 11 (201 + 1))

theorem largest_average_is_10 :
  max (max (max avg_7 avg_8) avg_9) (max avg_10 avg_11) = avg_10 := sorry

end largest_average_is_10_l580_580096


namespace total_cost_of_stickers_l580_580345

-- Definitions based on given conditions
def initial_funds_per_person := 9
def cost_of_deck_of_cards := 10
def Dora_packs_of_stickers := 2

-- Calculate the total amount of money collectively after buying the deck of cards
def remaining_funds := 2 * initial_funds_per_person - cost_of_deck_of_cards

-- Calculate the total packs of stickers if split evenly
def total_packs_of_stickers := 2 * Dora_packs_of_stickers

-- Prove the total cost of the boxes of stickers
theorem total_cost_of_stickers : remaining_funds = 8 := by
  -- Given initial funds per person, cost of deck of cards, and packs of stickers for Dora, the theorem should hold.
  sorry

end total_cost_of_stickers_l580_580345


namespace yellow_paint_quarts_l580_580348

theorem yellow_paint_quarts (ratio_r : ℕ) (ratio_y : ℕ) (ratio_w : ℕ) (qw : ℕ) : 
  ratio_r = 5 → ratio_y = 3 → ratio_w = 7 → qw = 21 → (qw * ratio_y) / ratio_w = 9 :=
by
  -- No proof required, inserting sorry to indicate missing proof
  sorry

end yellow_paint_quarts_l580_580348


namespace spotted_and_fluffy_cats_l580_580501

theorem spotted_and_fluffy_cats (total_cats : ℕ) (h1 : total_cats = 120)
    (fraction_spotted : ℚ) (h2 : fraction_spotted = 1/3)
    (fraction_fluffy_of_spotted : ℚ) (h3 : fraction_fluffy_of_spotted = 1/4) :
    (total_cats * fraction_spotted * fraction_fluffy_of_spotted).toNat = 10 := by
  sorry

end spotted_and_fluffy_cats_l580_580501


namespace original_proposition_contrapositive_converse_inverse_converse_l580_580246

noncomputable def f : ℝ → ℝ := sorry

variables (a b : ℝ)

theorem original_proposition (h₁ : ∀ x y, x ≤ y → f(x) ≤ f(y)) (h₂ : a + b ≥ 0) 
  : f(a) + f(b) ≥ f(-a) + f(-b) := sorry

theorem contrapositive (h₁ : ∀ x y, x ≤ y → f(x) ≤ f(y)) (h₂ : f(a) + f(b) < f(-a) + f(-b))
  : a + b < 0 := sorry

theorem converse (h₁ : ∀ x y, x ≤ y → f(x) ≤ f(y)) (h₂ : a + b < 0)
  : f(a) + f(b) < f(-a) + f(-b) := sorry

theorem inverse_converse (h₁ : ∀ x y, x ≤ y → f(x) ≤ f(y)) (h₂ : f(a) + f(b) ≥ f(-a) + f(-b))
  : a + b ≥ 0 := sorry

end original_proposition_contrapositive_converse_inverse_converse_l580_580246


namespace maximize_volume_of_spherical_segment_l580_580884

noncomputable def surface_area_of_spherical_segment (R m : ℝ) : ℝ := 2 * real.pi * R * m
noncomputable def volume_of_spherical_segment (R m : ℝ) : ℝ := 
  let r₁_r₂_sum := 2 * R^2 - (1 / 2) * m^2 in
  (real.pi * m * R^2 - (1 / 12) * real.pi * m^3)

theorem maximize_volume_of_spherical_segment (R F : ℝ) (hF : F = 2 * real.pi * R * (F / (2 * real.pi * R))) :
  volume_of_spherical_segment R (2 * R) = (4 / 3) * real.pi * R^3 :=
by
  sorry

end maximize_volume_of_spherical_segment_l580_580884


namespace lowest_average_cost_at_x_minimum_subsidy_needed_l580_580389

noncomputable def monthly_processing_cost (x : ℝ) : ℝ :=
  (1 / 2) * x^2 - 200 * x + 80000

def average_processing_cost_per_ton (x : ℝ) : ℝ :=
  monthly_processing_cost x / x

theorem lowest_average_cost_at_x {x : ℝ} (h1 : 400 ≤ x) (h2 : x ≤ 600) : 
  ∃ x_min : ℝ, x_min = 400 ∧ average_processing_cost_per_ton x_min = 200 :=
sorry

noncomputable def monthly_profit (x : ℝ) : ℝ :=
  100 * x - monthly_processing_cost x

theorem minimum_subsidy_needed {x : ℝ} (h1 : 400 ≤ x) (h2 : x ≤ 600) :
  ∃ subsidy : ℝ, subsidy = 40000 ∧ ∀ x, (400 ≤ x ∧ x ≤ 600) → monthly_profit x + subsidy ≥ 0 :=
sorry

end lowest_average_cost_at_x_minimum_subsidy_needed_l580_580389


namespace rhombus_circle_radii_l580_580433

theorem rhombus_circle_radii (d1 d2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 30) :
  let a := sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) in
  let rhombus_area := (d1 * d2) / 2 in
  let r := (2 * a * _root_.sqrt 241) / rhombus_area in
  let R := a / _root_.sqrt 2 in
  r = 60 / _root_.sqrt 241 ∧ R = _root_.sqrt 482 / 2 :=
by
  sorry

end rhombus_circle_radii_l580_580433


namespace expansion_eq_l580_580557

variable (x y : ℝ) -- x and y are real numbers
def a := 5
def b := 3
def c := 15

theorem expansion_eq : (x + a) * (b * y + c) = 3 * x * y + 15 * x + 15 * y + 75 := by 
  sorry

end expansion_eq_l580_580557


namespace trigonometric_identity_l580_580749

theorem trigonometric_identity (x y : ℝ) :
  sin (x - y) * sin x + cos (x - y) * cos x = cos y := by
  sorry

end trigonometric_identity_l580_580749


namespace sum_first_9000_terms_l580_580780

noncomputable def geom_sum (a r : ℝ) (n : ℕ) : ℝ :=
a * ((1 - r^n) / (1 - r))

theorem sum_first_9000_terms (a r : ℝ) (h1 : geom_sum a r 3000 = 1000) 
                              (h2 : geom_sum a r 6000 = 1900) : 
                              geom_sum a r 9000 = 2710 := 
by sorry

end sum_first_9000_terms_l580_580780


namespace select_4_lineup_lineup_2_rows_lineup_girls_together_lineup_boys_not_adjacent_lineup_A_not_ends_lineup_A_not_left_B_not_right_lineup_ABC_order_l580_580014

def num_people := 5
def num_boys := 2
def num_girls := 3

-- 1. Select 4 people to line up
theorem select_4_lineup : 
  (∑ p in (finset.range 5).powerset, if p.card = 4 then 1 else 0) = 120 := sorry

-- 2. Line up in two rows, with 1 person in the front row and 4 people in the back row
theorem lineup_2_rows : 
  (num_people * (num_people - 1) * (num_people - 2) * (num_people - 3)) = 120 := sorry

-- 3. Line up all together, with girls standing together
theorem lineup_girls_together :
  ((num_girls + 1)! * num_girls!) = 36 := sorry

-- 4. Line up all together, with boys not adjacent to each other
theorem lineup_boys_not_adjacent :
  (num_girls! * (num_girls + 1)choose num_boys) * num_boys! = 72 := sorry

-- 5. Line up all together, with person A not standing at the far left or far right
def person_A := 1
def positions_except_ends := 3

theorem lineup_A_not_ends :
  (positions_except_ends * (num_people - 1)! ) = 72 := sorry

-- 6. Line up all together, with person A not standing at the far left and person B not standing at the far right
def person_B := 2

theorem lineup_A_not_left_B_not_right :
  ((num_people - 1)! - ((num_people - 1)! - (num_people - 2)! + (num_people - 2)!)) = 78 := sorry

-- 7. Line up all together, with person A in front of person B, and person B in front of person C
def person_C := 3

theorem lineup_ABC_order :
  (num_people * (num_people - 1) div 3!) = 20 := sorry

end select_4_lineup_lineup_2_rows_lineup_girls_together_lineup_boys_not_adjacent_lineup_A_not_ends_lineup_A_not_left_B_not_right_lineup_ABC_order_l580_580014


namespace profit_share_ratio_l580_580446

theorem profit_share_ratio (P Q : ℝ) (hP : P = 40000) (hQ : Q = 60000) : P / Q = 2 / 3 :=
by
  rw [hP, hQ]
  norm_num

end profit_share_ratio_l580_580446


namespace frog_lands_on_corner_once_in_four_hops_l580_580135

/-- The simplified grid for the frog hopping problem -/
inductive GridPos
| Center
| Edge
| Corner

/-- Define initial conditions and movement properties -/
def is_wraparound (p : GridPos) : Prop := sorry -- to define wraparound movement

/-- Define hopping probability based on the current position -/
def hop_prob (from to: GridPos) : ℚ :=
match from, to with
| GridPos.Center, GridPos.Edge   => 1
| GridPos.Edge, GridPos.Edge   => 1 / 2
| GridPos.Edge, GridPos.Corner => 1 / 4
| GridPos.Edge, GridPos.Center => 1 / 4
| _, _                           => 0 -- Invalid transitions within four hops
end

/-- Starting from the center with up to four hops, the frog lands exactly once 
  on a corner with probability 25/32 -/
theorem frog_lands_on_corner_once_in_four_hops :
  (Prob (λ paths, paths.start = GridPos.Center → ∃ hops ≤ 4, paths(hops) = GridPos.Corner)) = 25 / 32 :=
by
  sorry

end frog_lands_on_corner_once_in_four_hops_l580_580135


namespace minimal_coach_handshakes_l580_580155

theorem minimal_coach_handshakes (n k1 k2 : ℕ) (h1 : k1 < n) (h2 : k2 < n)
  (hn : (n * (n - 1)) / 2 + k1 + k2 = 300) : k1 + k2 = 0 := by
  sorry

end minimal_coach_handshakes_l580_580155


namespace problem1_problem2_l580_580879

theorem problem1 : -(-1)^4 - (1 - 0.5) * (1/3) * (2 - 3^2) = 1/6 := 
by 
  sorry

theorem problem2 : (5/13) * (-3 - 1/4) - 0.5 / (abs(-3-1)) = -11/8 := 
by 
  sorry

end problem1_problem2_l580_580879


namespace cakes_served_dinner_l580_580852

def total_cakes_today : Nat := 15
def cakes_served_lunch : Nat := 6

theorem cakes_served_dinner : total_cakes_today - cakes_served_lunch = 9 :=
by
  -- Define what we need to prove
  sorry -- to skip the proof

end cakes_served_dinner_l580_580852


namespace WallLengthBy40Men_l580_580992

-- Definitions based on the problem conditions
def men1 : ℕ := 20
def length1 : ℕ := 112
def days1 : ℕ := 6

def men2 : ℕ := 40
variable (y : ℕ)  -- given 'y' days

-- Establish the relationship based on the given conditions
theorem WallLengthBy40Men :
  ∃ x : ℕ, x = (men2 / men1) * length1 * (y / days1) :=
by
  sorry

end WallLengthBy40Men_l580_580992


namespace bisector_COA_angle_l580_580386

open EuclideanGeometry

-- Define the geometrical entities and conditions
variables {O A B C D : Point}
variable {k : Circle}

-- Circle with center O intersects line e at points A and B
axiom circle_intersect_line : ∀ (e : Line), intersects k e A ∧ intersects k e B 
-- Perpendicular bisector of OB intersects circle at C and D
axiom perp_bisector_intersect_circle : perp_bisector (Segment.mk O B) = Line.mk C D ∧ intersects k (Line.mk C D) C ∧ intersects k (Line.mk C D) D

-- Proof that the bisector of angle COA forms an angle of 60 degrees with line e
theorem bisector_COA_angle (e : Line) :
  (∃ (θ : ℝ), θ = 60 ∧ FormAngle (angle_bisector (∠ C O A) e) θ) :=
begin
  sorry -- Proof placeholder
end

end bisector_COA_angle_l580_580386


namespace greatest_of_given_numbers_l580_580803

-- Defining the given conditions
def a := 1000 + 0.01
def b := 1000 * 0.01
def c := 1000 / 0.01
def d := 0.01 / 1000
def e := 1000 - 0.01

-- Prove that c is the greatest
theorem greatest_of_given_numbers : c = max a (max b (max d e)) :=
by
  -- Placeholder for the proof
  sorry

end greatest_of_given_numbers_l580_580803


namespace relationship_between_f_l580_580976

-- Given definitions
def quadratic_parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def axis_of_symmetry (f : ℝ → ℝ) (α : ℝ) : Prop :=
  ∀ x y : ℝ, f x = f y ↔ x + y = 2 * α

-- The problem statement to prove in Lean 4
theorem relationship_between_f (a b c x : ℝ) (hpos : x > 0) (apos : a > 0) :
  axis_of_symmetry (quadratic_parabola a b c) 1 →
  quadratic_parabola a b c (3^x) > quadratic_parabola a b c (2^x) :=
by
  sorry

end relationship_between_f_l580_580976


namespace locus_of_D_l580_580868

theorem locus_of_D 
  (a b : ℝ)
  (hA : 0 ≤ a ∧ a ≤ (2 * Real.sqrt 3 / 3))
  (hB : 0 ≤ b ∧ b ≤ (2 * Real.sqrt 3 / 3))
  (AB_eq : Real.sqrt ((b - 2 * a)^2 + (Real.sqrt 3 * b)^2)  = 2) :
  3 * (b - a / 2)^2 + (Real.sqrt 3 / 2 * (a + b))^2 / 3 = 1 :=
sorry

end locus_of_D_l580_580868


namespace find_a_l580_580243

noncomputable def binomial_coeff (n k : ℕ) : ℚ := nat.choose n k

def poly_term_coeff (a : ℚ) (term : ℚ) : ℚ := 
  binomial_coeff 8 3 * (1/2)^(8-3) * (-a)^3

theorem find_a (a : ℚ) : 
  poly_term_coeff a 1 = -14 → a = 2 :=
by
  -- proof will be filled here 
  sorry

end find_a_l580_580243


namespace correct_statements_l580_580255

def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

/- given conditions -/
axiom f_at_origin (a b c : ℝ) : f 0 a b c = 0
axiom slope_at_plus_minus_one (a b c : ℝ) : deriv (f (1:ℝ) a b c) = -1 ∧ deriv (f (-1:ℝ) a b c) = -1

/- the statement to prove -/
theorem correct_statements (a b c : ℝ) :
  (∃ a b, f(x: xℝ) = x^3 -4x) ∧
  (¬(f(x: xℝ).extremum)) ∧ 
  (f(x.max x.min (x: xℝ)) = 0)
  sorry

end correct_statements_l580_580255


namespace trigonometric_expr_value_l580_580566

theorem trigonometric_expr_value :
  (sqrt 3 * tan (10 * Real.pi / 180) + 1) / ((4 * cos (10 * Real.pi / 180) ^ 2 - 2) * sin (10 * Real.pi / 180)) = 4 :=
sorry

end trigonometric_expr_value_l580_580566


namespace number_of_lattice_points_covered_by_AB_l580_580355

theorem number_of_lattice_points_covered_by_AB (A B : ℝ) (h1 : ∀ x, x ∈ ℤ → x ∈ set.Icc A B) (h2 : |B - A| = 2009) :
  (set.count (set.Icc A B ∩ ℤ)) = 2009 ∨ (set.count (set.Icc A B ∩ ℤ)) = 2010 := by
  sorry

end number_of_lattice_points_covered_by_AB_l580_580355


namespace solution1_solution2_l580_580911

-- Definitions of the conditions as inequalities
def condition1 (x : ℝ) : Prop := 2 * x^2 - 5 * x + 3 < 0
def condition2 (x : ℝ) : Prop := (x - 1) / (2 - x) ≤ 1

-- Proposition that these sets are the solution sets to respective inequalities
theorem solution1 :
  { x : ℝ | 1 < x ∧ x < 3 / 2 } = {x : ℝ | condition1 x } :=
by
  sorry

theorem solution2 :
  ({x : ℝ | x ≤ 3 / 2 } ∪ { x : ℝ | x > 2 }) = {x : ℝ | condition2 x } :=
by
  sorry

end solution1_solution2_l580_580911


namespace poly_has_root_l580_580577

theorem poly_has_root : ∃ x : ℝ, x^3 = Real.cbrt 2 + Real.cbrt 3 → 
  x^9 - 15 * x^6 - 87 * x^3 - 125 = 0 :=
begin
  sorry
end

end poly_has_root_l580_580577


namespace range_of_a_l580_580916

theorem range_of_a 
  (a : ℝ) (h : ∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) 
  : -2 < a ∧ a ≤ 2 :=
sorry

end range_of_a_l580_580916


namespace cos_AMB_l580_580114

def is_regular_tetrahedron (A B C D : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A ∧
  dist D A = dist A C ∧ dist A C = dist A D ∧ dist A D = dist B D

def point_on_line_segment (C D M : Point) (k l : ℕ) : Prop :=
  ∃ (t : ℚ), t = k/(k + l) ∧ M = t • C + (1 - t) • D

theorem cos_AMB (A B C D M : Point) 
  (h_reg_tetra : is_regular_tetrahedron A B C D) 
  (h_M_ratio : point_on_line_segment C D M 2 1) :
  \cos (angle A M B) = \frac{2}{3\sqrt{3}} := 
  sorry

end cos_AMB_l580_580114


namespace new_triangle_area_ratio_l580_580943

namespace TriangleArea

open Locale TopologicalSpace

-- Definitions
def triangle_area_ratio (T : ℝ) : ℝ :=
  let ratio := (2 / 3) in
  (ratio ^ 2)

-- Theorem statement
theorem new_triangle_area_ratio (T : ℝ) :
  let new_area_ratio := triangle_area_ratio T in
  new_area_ratio = 4 / 9 := sorry

end TriangleArea

end new_triangle_area_ratio_l580_580943


namespace polynomial_magnitude_sum_l580_580717

/-- 
  Let z1, z2, z3, z4 be the solutions to the equation 
  x^4 + 3x^3 + 3x^2 + 3x + 1 = 0.
  Then |z1| + |z2| + |z3| + |z4| can be written as 
  (a + b * sqrt(c)) / d, where c is a square-free positive
  integer, and a, b, d are positive integers with 
  gcd(a, b, d) = 1. Compute 1000a + 100b + 10c + d.
-/
theorem polynomial_magnitude_sum :
  let z1 z2 z3 z4 : ℂ := by sorry,
  let magSum := |z1| + |z2| + |z3| + |z4|
  in magSum = (7 + sqrt 5) / 2 :=
sorry

end polynomial_magnitude_sum_l580_580717


namespace better_modelA_l580_580429

/- Define the two models as functions -/
def modelA (x : ℝ) : ℝ := x^2 + 1
def modelB (x : ℝ) : ℝ := 3 * x - 1

/- Define the measured values -/
def measured_values : List (ℝ × ℝ) := [(1, 2), (2, 5), (3, 10.2)]

/- Define a predicate to check model fitting -/
def better_fitting_model (model : ℝ → ℝ) (values : List (ℝ × ℝ)) : Prop :=
  ∑ (x, y) in values, (y - model x)^2 = 
  min (∑ (x, y) in values, (y - modelA x)^2) (∑ (x, y) in values, (y - modelB x)^2)

/- The theorem to prove -/
theorem better_modelA : better_fitting_model modelA measured_values :=
sorry

end better_modelA_l580_580429


namespace caleb_total_spent_l580_580881

theorem caleb_total_spent :
  ∀ (total_hamburgers : ℕ) (cost_single : ℝ) (cost_double : ℝ) (num_double : ℕ),
  total_hamburgers = 50 →
  cost_single = 1.00 →
  cost_double = 1.50 →
  num_double = 33 →
  let num_single := total_hamburgers - num_double in
  let total_cost := (num_double * cost_double) + (num_single * cost_single) in
  total_cost = 66.50 :=
by
  intros total_hamburgers cost_single cost_double num_double
  intros H1 H2 H3 H4
  let num_single := 50 - 33
  let total_cost := (33 * 1.50) + (num_single * 1.00)
  calc total_cost = 66.50 : sorry

end caleb_total_spent_l580_580881


namespace summation_problem_l580_580567

-- Definitions for the conditions
def seq_an (n : ℕ) : ℝ := 3 * n + 1
def seq_bn (n : ℕ) : ℝ := 2^n * seq_an n
def sum_Sn (n : ℕ) : ℝ := finset.sum (finset.range n) seq_an
def sum_Tn (n : ℕ) : ℝ := finset.sum (finset.range n) seq_bn

-- Given conditions
axiom an_condition (n : ℕ) : seq_an n ^ 2 + 3 * seq_an n = 6 * sum_Sn n + 4
axiom bn_definition (n : ℕ) : seq_bn n = 2^n * seq_an n 

-- Proof problem (Statement only)
theorem summation_problem (n : ℕ) :
  let T_n := sum_Tn n in T_n = (3 * n - 2) * 2^(n + 1) + 4 :=
sorry

end summation_problem_l580_580567


namespace similar_triangles_height_l580_580073

theorem similar_triangles_height
  (a b : ℕ)
  (area_ratio: ℕ)
  (height_smaller : ℕ)
  (height_relation: height_smaller = 5)
  (area_relation: area_ratio = 9)
  (similarity: a / b = 1 / area_ratio):
  (∃ height_larger : ℕ, height_larger = 15) :=
by
  sorry

end similar_triangles_height_l580_580073


namespace total_cost_chairs_l580_580477

def living_room_chairs : Nat := 3
def kitchen_chairs : Nat := 6
def dining_room_chairs : Nat := 8
def outdoor_patio_chairs : Nat := 12

def living_room_price : Nat := 75
def kitchen_price : Nat := 50
def dining_room_price : Nat := 100
def outdoor_patio_price : Nat := 60

theorem total_cost_chairs : 
  living_room_chairs * living_room_price + 
  kitchen_chairs * kitchen_price + 
  dining_room_chairs * dining_room_price + 
  outdoor_patio_chairs * outdoor_patio_price = 2045 := by
  sorry

end total_cost_chairs_l580_580477


namespace badminton_tournament_matches_l580_580289

def num_pairs : ℕ := 6

def binom (n k : ℕ) : ℕ :=
  nat.choose n k

theorem badminton_tournament_matches : binom num_pairs 2 = 15 :=
by 
  simp [num_pairs, binom]
  sorry

end badminton_tournament_matches_l580_580289


namespace gas_volume_at_11C_l580_580922

theorem gas_volume_at_11C (initial_temp initial_volume : ℝ) (expansion_rate : ℝ)
    (initial_conditions : initial_temp = 25 ∧ initial_volume = 35 ∧ expansion_rate = 5) :
    ∃ V : ℝ, (11 = 25 - 4 * ((initial_temp - 11) / 4)) ∧ (V = initial_volume - expansion_rate * ((initial_temp - 11) / 4)) :=
begin
  rcases initial_conditions with ⟨h1, h2, h3⟩,
  use 17.5,
  refine ⟨_, _⟩,
  sorry, -- Step 1: Handle temperature equation, i.e., 11 = 25 - 4 * ((initial_temp - 11) / 4)
  sorry  -- Step 2: Handle volume equation to get V = 17.5
end

end gas_volume_at_11C_l580_580922


namespace sum_of_digits_of_largest_valid_n_l580_580711

open List

-- Defining the problem in Lean
def is_single_digit_prime (n: ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_valid_prime_triplet (d e: ℕ) : Prop :=
  is_single_digit_prime d ∧ is_single_digit_prime e ∧ Prime (10 * d + e)

def largest_valid_product : ℕ :=
  max (max (2 * 3 * 23) (3 * 7 * 37)) (max (5 * 3 * 53) (7 * 3 * 73))

def sum_of_digits (n: ℕ) : ℕ :=
  n.digits 10 |> foldl (·+·) 0

theorem sum_of_digits_of_largest_valid_n : sum_of_digits largest_valid_product = 12 := by
  sorry

end sum_of_digits_of_largest_valid_n_l580_580711


namespace commutative_not_associative_l580_580212

-- Define the binary operation * on positive reals
def star (x y : ℝ) : ℝ := (x^2 + y^2) / (x + y)

-- The main theorem statement
theorem commutative_not_associative (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  star x y = star y x ∧ (star (star x y) z ≠ star x (star y z)) :=
by
  sorry

end commutative_not_associative_l580_580212


namespace least_positive_x_multiple_of_53_l580_580807

theorem least_positive_x_multiple_of_53 :
  ∃ (x : ℕ), (x > 0) ∧ ((2 * x)^2 + 2 * 47 * (2 * x) + 47^2) % 53 = 0 ∧ x = 6 :=
by
  sorry

end least_positive_x_multiple_of_53_l580_580807


namespace textbook_thickness_false_l580_580848

-- Defining the condition in Lean, let’s define a predicate that checks the thickness of a textbook
def is_thick_enough (thickness_cm : ℕ) : Prop := thickness_cm = 100

-- The statement translation: If a typical math textbook is about 1 centimeter thick, then it is not approximately 1 meter thick
theorem textbook_thickness_false (thickness_cm : ℕ) (h : thickness_cm = 1) : ¬ is_thick_enough thickness_cm := 
by {
  -- We assume the textbook is 1 cm thick
  have h1 : thickness_cm = 1 := h,
  -- Since 1 cm is not equal to 1 meter, we have our conclusion
  sorry
}

end textbook_thickness_false_l580_580848


namespace min_value_a_l580_580606

theorem min_value_a (a : ℝ) : (∀ x ∈ set.Icc 1 5, x^2 - 6*x ≤ a + 2) ↔ a ≥ -7 :=
by {
  sorry
}

end min_value_a_l580_580606


namespace abs_ineq_solution_set_l580_580405

theorem abs_ineq_solution_set (x : ℝ) :
  |x - 5| + |x + 3| ≥ 10 ↔ x ≤ -4 ∨ x ≥ 6 :=
by
  sorry

end abs_ineq_solution_set_l580_580405


namespace min_positive_announcements_l580_580164

theorem min_positive_announcements (x y : ℕ) 
  (h1 : x * (x - 1) = 110) 
  (h2 : y * (y - 1) + (x - y) * (x - 1 - (y - 1)) = 50) : 
  y >= 5 := 
sorry

end min_positive_announcements_l580_580164


namespace distance_not_six_l580_580385

theorem distance_not_six (x : ℝ) : 
  (x = 6 → 10 + (x - 3) * 1.8 ≠ 17.2) ∧ 
  (10 + (x - 3) * 1.8 = 17.2 → x ≠ 6) :=
by {
  sorry
}

end distance_not_six_l580_580385


namespace geometric_series_sum_l580_580880

theorem geometric_series_sum :
  ∑ k in Finset.range 101, (1 / 2^k) = 2 - (1 / 2^100) :=
by sorry

end geometric_series_sum_l580_580880


namespace ratio_of_sum_l580_580819

theorem ratio_of_sum (x y : ℚ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 5) : 
  (x + y) / 3 = 11 / 9 := 
by 
  sorry

end ratio_of_sum_l580_580819


namespace range_of_m_non_perpendicular_tangent_l580_580970

noncomputable def f (m x : ℝ) : ℝ := Real.exp x - m * x

theorem range_of_m_non_perpendicular_tangent (m : ℝ) :
  (∀ x : ℝ, (deriv (f m) x ≠ -2)) → m ≤ 2 :=
by
  sorry

end range_of_m_non_perpendicular_tangent_l580_580970


namespace height_of_larger_triangle_l580_580063

-- Definitions from the conditions
variables (height_small height_large : ℝ)
variables (area_ratio : ℝ)
variables (k : ℝ)

-- Given conditions
def triangles_similar : Prop := area_ratio = 9
def height_small_defined : Prop := height_small = 5
def scale_factor : Prop := k = real.sqrt area_ratio

-- Proof problem statement
theorem height_of_larger_triangle
  (h_similar : triangles_similar)
  (h_height_small : height_small_defined)
  (h_scale_factor : scale_factor) :
  height_large = 15 := sorry

end height_of_larger_triangle_l580_580063


namespace incorrect_statement_D_l580_580587

def condition_1 (z1 z2 : ℂ) : Prop :=
  |z1 + z2| = |z1 - z2| → (vector_perpendicular (complex_to_vector z1) (complex_to_vector z2))

def condition_2 (z : ℂ) : Prop :=
  |conjugate (z)|^2 = |z|^2 ∧ |z|^2 = z * conjugate (z)

def condition_3 (z : ℂ) : Prop :=
  |z| = complex_modulus_distance_to_origin z

def condition_4 (z : ℂ) : Prop :=
  |z - complex.i| = sqrt 5 → complex_point_on_circle z (0,1) (sqrt 5)

theorem incorrect_statement_D : 
  ∀ (z : ℂ), |z - complex.i| = sqrt 5 → 
  ¬ (point_corresponding_to_complex z lies_on_circle (1, 0) (sqrt 5)) :=
by {
  intros,
  sorry
}

end incorrect_statement_D_l580_580587


namespace similar_triangles_height_l580_580045

theorem similar_triangles_height (h₁ h₂ : ℝ) (ratio_areas : ℝ) 
  (h₁_eq : h₁ = 5) (ratio_areas_eq : ratio_areas = 1 / 9)
  (similar : h₂^2 = (√ratio_areas)^2 * h₁^2) : h₂ = 15 :=
by {
  have ratio_areas_pos : ratio_areas > 0 := by (simp [ratio_areas_eq]),
  have k := √ratio_areas,
  have k_eq : k = 3 := by {
    rw [ratio_areas_eq, sqrt_div, sqrt_one, sqrt_nat_eq_iff_eq_sq, one_div_eq_inv] at *,
    norm_num },
  have h₂_def : h₂ = 3 * h₁ := by rw [h₁_eq, mul_comm, k_eq],
  rw [h₂_def],
  norm_num,
}

end similar_triangles_height_l580_580045


namespace lean4_statement_l580_580003

noncomputable def proof_problem (n : ℕ) (a : fin (n + 1) → ℝ) : Prop :=
  (∀ i, 0 ≤ i < n → -1 ≤ a (fin.mk i sorry) ∧ a (fin.mk i sorry) ≤ 1) ∧
  (a (fin.mk n sorry) = a (fin.mk 0 sorry)) →
  ∑ i in finset.range n, 1 / (1 + a (fin.mk i sorry) * a (fin.mk (i + 1) sorry)) ≥
  ∑ i in finset.range n, 1 / (1 + (a (fin.mk i sorry))^2)

theorem lean4_statement (n : ℕ) (a : fin (n + 1) → ℝ) (h : proof_problem n a) :
  ∑ i in finset.range n, 1 / (1 + a (fin.mk i sorry) * a (fin.mk (i + 1) sorry)) ≥
  ∑ i in finset.range n, 1 / (1 + (a (fin.mk i sorry))^2) := 
sorry

end lean4_statement_l580_580003


namespace breaks_to_pieces_l580_580841

-- Define the initial conditions of the chocolate bar.
def chocolate_bar : Type := { rows : ℕ // rows = 5 } × { cols : ℕ // cols = 8 }

-- Define the target number of pieces.
def target_pieces : ℕ := 40

-- Prove the number of breaks needed to reach the target number of pieces.
theorem breaks_to_pieces (n : ℕ) (bar : chocolate_bar) (h : bar.1.1 * bar.2.1 = target_pieces) : n + 1 = target_pieces → n = 39 :=
by
  intros h,
  linarith

end breaks_to_pieces_l580_580841


namespace height_of_larger_triangle_l580_580064

-- Definitions from the conditions
variables (height_small height_large : ℝ)
variables (area_ratio : ℝ)
variables (k : ℝ)

-- Given conditions
def triangles_similar : Prop := area_ratio = 9
def height_small_defined : Prop := height_small = 5
def scale_factor : Prop := k = real.sqrt area_ratio

-- Proof problem statement
theorem height_of_larger_triangle
  (h_similar : triangles_similar)
  (h_height_small : height_small_defined)
  (h_scale_factor : scale_factor) :
  height_large = 15 := sorry

end height_of_larger_triangle_l580_580064


namespace line_PC_intersects_AB_l580_580410

variable {A B C P K : Point}
variable [h_circle : Circle A B C]
variable [h_tangentA : Tangent h_circle A P]
variable [h_tangentB : Tangent h_circle B P]

theorem line_PC_intersects_AB :
  ∃ K : Point, IsInter AB PC K ∧ divides_ratio AK KB (AC^2) (BC^2) :=
sorry

end line_PC_intersects_AB_l580_580410


namespace proof_evaluate_expression_l580_580898

def evaluate_expression : Prop :=
  - (18 / 3 * 8 - 72 + 4 * 8) = 8

theorem proof_evaluate_expression : evaluate_expression :=
by 
  sorry

end proof_evaluate_expression_l580_580898


namespace find_initial_workers_l580_580124

-- Define the initial number of workers.
def initial_workers (W : ℕ) (A : ℕ) : Prop :=
  -- Condition 1: W workers can complete work A in 25 days.
  ( W * 25 = A )  ∧
  -- Condition 2: (W + 10) workers can complete work A in 15 days.
  ( (W + 10) * 15 = A )

-- The theorem states that given the conditions, the initial number of workers is 15.
theorem find_initial_workers {W A : ℕ} (h : initial_workers W A) : W = 15 :=
  sorry

end find_initial_workers_l580_580124


namespace similar_triangles_height_l580_580047

theorem similar_triangles_height (h₁ h₂ : ℝ) (ratio_areas : ℝ) 
  (h₁_eq : h₁ = 5) (ratio_areas_eq : ratio_areas = 1 / 9)
  (similar : h₂^2 = (√ratio_areas)^2 * h₁^2) : h₂ = 15 :=
by {
  have ratio_areas_pos : ratio_areas > 0 := by (simp [ratio_areas_eq]),
  have k := √ratio_areas,
  have k_eq : k = 3 := by {
    rw [ratio_areas_eq, sqrt_div, sqrt_one, sqrt_nat_eq_iff_eq_sq, one_div_eq_inv] at *,
    norm_num },
  have h₂_def : h₂ = 3 * h₁ := by rw [h₁_eq, mul_comm, k_eq],
  rw [h₂_def],
  norm_num,
}

end similar_triangles_height_l580_580047


namespace linda_max_servings_is_13_l580_580486

noncomputable def max_servings 
  (recipe_bananas : ℕ) (recipe_yogurt : ℕ) (recipe_honey : ℕ)
  (linda_bananas : ℕ) (linda_yogurt : ℕ) (linda_honey : ℕ)
  (servings_for_recipe : ℕ) : ℕ :=
  min 
    (linda_bananas * servings_for_recipe / recipe_bananas) 
    (min 
      (linda_yogurt * servings_for_recipe / recipe_yogurt)
      (linda_honey * servings_for_recipe / recipe_honey)
    )

theorem linda_max_servings_is_13 : 
  max_servings 3 2 1 10 9 4 4 = 13 :=
  sorry

end linda_max_servings_is_13_l580_580486


namespace nat_square_not_div_factorial_l580_580905

-- Define n as a natural number
def n : Nat := sorry  -- We assume n is given somewhere

-- Define a function to check if a number is prime
def is_prime (p : Nat) : Prop := sorry  -- Placeholder for prime checking function

-- The main theorem to prove
theorem nat_square_not_div_factorial (n : Nat) : (n = 4 ∨ is_prime n) → ¬ ((n * n) ∣ Nat.factorial n) := by
  sorry

end nat_square_not_div_factorial_l580_580905


namespace similar_triangle_of_second_intersection_points_l580_580793

theorem similar_triangle_of_second_intersection_points
  (ABC : Type*) [triangle : Triangle ABC]  
  (H : Point ABC) 
  (H1 H2 H3 : Point ABC) 
  (circle1 circle2 circle3 : Circle) :
  orthocenter triangle H →
  is_foot_of_altitude A H1 triangle →
  is_foot_of_altitude B H2 triangle →
  is_foot_of_altitude C H3 triangle →
  circle1.pass_through H →
  circle1.is_tangent_to_side H1 (side_of_triangle B C triangle) →
  circle2.pass_through H →
  circle2.is_tangent_to_side H2 (side_of_triangle C A triangle) →
  circle3.pass_through H →
  circle3.is_tangent_to_side H3 (side_of_triangle A B triangle) →
  (∃ A1 B1 C1 : Point ABC,
    second_intersection_point circle1 ≠ H →
    second_intersection_point circle2 ≠ H →
    second_intersection_point circle3 ≠ H →
    is_similar_triangle (Triangle.mk A1 B1 C1) triangle) :=
begin
  sorry -- proof goes here
end

end similar_triangle_of_second_intersection_points_l580_580793


namespace reading_enhusiasts_not_related_to_gender_l580_580867

noncomputable def contingency_table (boys_scores : List Nat) (girls_scores : List Nat) :
  (Nat × Nat × Nat × Nat × Nat × Nat) × (Nat × Nat × Nat × Nat × Nat × Nat) :=
  let boys_range := (2, 3, 5, 15, 18, 12)
  let girls_range := (0, 5, 10, 10, 7, 13)
  ((2, 3, 5, 15, 18, 12), (0, 5, 10, 10, 7, 13))

theorem reading_enhusiasts_not_related_to_gender (boys_scores : List Nat) (girls_scores : List Nat) :
  let table := contingency_table boys_scores girls_scores
  let (boys_range, girls_range) := table
  let a := 45 -- Boys who are reading enthusiasts
  let b := 10 -- Boys who are non-reading enthusiasts
  let c := 30 -- Girls who are reading enthusiasts
  let d := 15 -- Girls who are non-reading enthusiasts
  let n := a + b + c + d
  let k_squared := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))
  k_squared < 3.841 := 
sorry

end reading_enhusiasts_not_related_to_gender_l580_580867


namespace concyclic_quad_identity_l580_580843

variables (A B C D O : Type) [inner_product_space ℝ O] [metric_space O] [inhabited A] [inhabited B] [inhabited C] [inhabited D]

variables (a b c d : ℝ)
variables [convex_quadrilateral A B C D] [circumscribed_quadrilateral A B C D O]

theorem concyclic_quad_identity :
  dist O A * dist O C + dist O B * dist O D = real.sqrt (a * b * c * d) :=
sorry

end concyclic_quad_identity_l580_580843


namespace triangle_ratio_l580_580795

noncomputable def ratio_of_segments : ℝ :=
  let A := (0, 0) : ℝ × ℝ
  let B := (8, 0) : ℝ × ℝ
  let C := (4, 4 * Real.sqrt 3) : ℝ × ℝ -- Coordinates based on triangle properties
  let P := (x, 0) : ℝ × ℝ -- P is on AB, thus has coordinates (x, 0)
  let CP := Real.sqrt ((x - 4)^2 + (4 * Real.sqrt 3)^2) -- Distance formula for CP
  if 2 = CP then -- Given condition CP = 2
    0.6 + 0.4 * Real.sqrt 6
  else
    -1 -- invalid case, should never happen

theorem triangle_ratio (x : ℝ) : 
  let A := (0, 0) : ℝ × ℝ
  let B := (8, 0) : ℝ × ℝ
  let C := (4, 4 * Real.sqrt 3) : ℝ × ℝ -- Coordinates based on triangle properties
  let P := (x, 0) : ℝ × ℝ -- P is on AB, thus has coordinates (x, 0)
  let CP := Real.sqrt ((x - 4)^2 + (4 * Real.sqrt 3)^2) -- Distance formula for CP
  (CP = 2) → ratio_of_segments = 0.6 + 0.4 * Real.sqrt 6 := 
by 
  sorry

end triangle_ratio_l580_580795


namespace quad_roots_twice_l580_580006

noncomputable def quad_eq_roots_twice {α : Type} [LinearOrderedField α] 
  (x : α) (m n p : α) : Prop :=
  x^2 + m*x + n = 0 ∧ (∃ r1 r2, x^2 + p*x + m = 0 ∧ 
  (2 * r1, 2 * r2) are the roots of (x^2 + m*x + n = 0))

theorem quad_roots_twice (α : Type) [LinearOrderedField α] (m n p : α) 
  (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : p ≠ 0):
  (∀ x, quad_eq_roots_twice x m n p → n / p = 8) :=
by sorry

end quad_roots_twice_l580_580006


namespace tan_sin_cos_l580_580583

theorem tan_sin_cos (θ : ℝ) (h : Real.tan θ = 1 / 2) : 
  Real.sin (2 * θ) - 2 * Real.cos θ ^ 2 = - 4 / 5 := by 
  sorry

end tan_sin_cos_l580_580583


namespace max_value_sqrt_sum_l580_580591

theorem max_value_sqrt_sum {x y z : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  ∃ (M : ℝ), M = (Real.sqrt (abs (x - y)) + Real.sqrt (abs (y - z)) + Real.sqrt (abs (z - x))) ∧ M = Real.sqrt 2 + 1 :=
by sorry

end max_value_sqrt_sum_l580_580591


namespace great_white_shark_teeth_is_420_l580_580149

-- Define the number of teeth in a tiger shark
def tiger_shark_teeth : ℕ := 180

-- Define the number of teeth in a hammerhead shark based on the tiger shark's teeth
def hammerhead_shark_teeth : ℕ := tiger_shark_teeth / 6

-- Define the number of teeth in a great white shark based on the sum of tiger and hammerhead shark's teeth
def great_white_shark_teeth : ℕ := 2 * (tiger_shark_teeth + hammerhead_shark_teeth)

-- The theorem statement that we need to prove
theorem great_white_shark_teeth_is_420 : great_white_shark_teeth = 420 :=
by
  -- Provide space for the proof
  sorry

end great_white_shark_teeth_is_420_l580_580149


namespace spotted_and_fluffy_cats_l580_580494

theorem spotted_and_fluffy_cats (total_cats : ℕ) (total_cats_equiv : total_cats = 120) (one_third_spotted : ℕ → ℕ) (one_fourth_fluffy_spotted : ℕ → ℕ) :
  (one_third_spotted total_cats * one_fourth_fluffy_spotted (one_third_spotted total_cats) = 10) :=
by
  sorry

end spotted_and_fluffy_cats_l580_580494


namespace field_area_l580_580404

-- Given definitions based on the conditions
def ratio (a b : ℕ) := ∃ k, a = 3 * k ∧ b = 4 * k
def cost_per_meter := 0.25 -- 25 paise per meter
def total_cost := 94.5 -- Total cost given in the condition

-- Theorem to prove area of the field given the conditions
theorem field_area (a b : ℕ) (area : ℕ) 
  (h₁ : ratio a b) 
  (h₂ : total_cost = 94.5) 
  (h₃ : cost_per_meter = 0.25) 
  (perimeter := 2 * (a + b)) 
  (fencing_cost := perimeter * cost_per_meter) 
  (h₄ : fencing_cost = total_cost) :
  area = 8748 := 
by
  -- To be proved by substitution and solving equations from the conditions
  sorry

end field_area_l580_580404


namespace inequality1_inequality2_l580_580376

-- First proof problem
theorem inequality1 (x : ℝ) : (x ≠ 2 ∧ (2 * x + 1) / (x - 2) > 1) ↔ x ∈ Set.Icc (-∞ : ℝ) (-3) ∪ Set.Icc (2 : ℝ) (∞ : ℝ) :=
by
  sorry

-- Second proof problem
theorem inequality2 (x a : ℝ) : 
  (x^2 - 6 * a * x + 5 * a^2 ≤ 0) →
  ((0 < a → x ∈ Set.Icc a (5 * a)) ∧ 
   (a = 0 → x = 0) ∧ 
   (a < 0 → x ∈ Set.Icc (5 * a) a)) :=
by
  sorry

end inequality1_inequality2_l580_580376


namespace larger_number_is_34_l580_580784

theorem larger_number_is_34 (x y : ℕ) (h1 : x + y = 56) (h2 : y = x + 12) : y = 34 :=
by
  sorry

end larger_number_is_34_l580_580784


namespace total_cost_tom_pays_for_trip_l580_580033

/-- Tom needs to get 10 different vaccines and a doctor's visit to go to Barbados.
    Each vaccine costs $45.
    The doctor's visit costs $250.
    Insurance will cover 80% of these medical bills.
    The trip itself costs $1200.
    Prove that the total amount Tom has to pay for his trip to Barbados, including medical expenses, is $1340. -/
theorem total_cost_tom_pays_for_trip : 
  let cost_per_vaccine := 45
  let number_of_vaccines := 10
  let cost_doctor_visit := 250
  let insurance_coverage_rate := 0.8
  let trip_cost := 1200
  let total_medical_cost := (number_of_vaccines * cost_per_vaccine) + cost_doctor_visit
  let insurance_coverage := insurance_coverage_rate * total_medical_cost
  let net_medical_cost := total_medical_cost - insurance_coverage
  let total_cost := trip_cost + net_medical_cost
  total_cost = 1340 := 
by 
  sorry

end total_cost_tom_pays_for_trip_l580_580033


namespace grape_juice_percent_l580_580103

theorem grape_juice_percent (initial_volume additional_volume : ℝ) (initial_percentage added_volume : ℝ)
  (h1 : initial_volume = 30) 
  (h2 : initial_percentage = 0.10) 
  (h3 : additional_volume = 10) :
  (initial_percentage * initial_volume + added_volume) / (initial_volume + additional_volume) * 100 = 32.5 := by
  have h4 : initial_percentage * initial_volume = 3 := by sorry
  have h5 : initial_percentage * initial_volume + added_volume = 13 := by sorry
  have h6 : initial_volume + additional_volume = 40 := by sorry
  show (13 / 40) * 100 = 32.5 from by sorry

end grape_juice_percent_l580_580103


namespace symmetric_points_existence_l580_580966

-- Define the ellipse equation
def is_ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

-- Define the line equation parameterized by m
def line_eq (x y m : ℝ) : Prop :=
  y = 4 * x + m

-- Define the range for m such that symmetric points exist
def m_in_range (m : ℝ) : Prop :=
  - (2 * Real.sqrt 13) / 13 < m ∧ m < (2 * Real.sqrt 13) / 13

-- Prove the existence of symmetric points criteria for m
theorem symmetric_points_existence (m : ℝ) :
  (∀ (x y : ℝ), is_ellipse x y → line_eq x y m → 
    (∃ (x1 y1 x2 y2 : ℝ), is_ellipse x1 y1 ∧ is_ellipse x2 y2 ∧ line_eq x1 y1 m ∧ line_eq x2 y2 m ∧ 
      (x1 = x2) ∧ (y1 = -y2))) ↔ m_in_range m :=
sorry

end symmetric_points_existence_l580_580966


namespace divisors_must_be_four_l580_580378

theorem divisors_must_be_four (c d : ℤ) (h : 4 * d = 10 - 3 * c) : 
  ∃ S : Finset ℕ, S = {1, 2, 3, 5} ∧ ∀ x ∈ S, x ∣ (3 * d + 15) ∧ S.card = 4 :=
by
  sorry

end divisors_must_be_four_l580_580378


namespace power_of_product_l580_580876

variable (x y : ℝ)

theorem power_of_product (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 :=
  sorry

end power_of_product_l580_580876


namespace sum_of_digits_divisible_by_7_l580_580364

theorem sum_of_digits_divisible_by_7
  (a b : ℕ)
  (h_three_digit : 100 * a + 11 * b ≥ 100 ∧ 100 * a + 11 * b < 1000)
  (h_last_two_digits_equal : true)
  (h_divisible_by_7 : (100 * a + 11 * b) % 7 = 0) :
  (a + 2 * b) % 7 = 0 :=
sorry

end sum_of_digits_divisible_by_7_l580_580364


namespace triangles_in_dodecagon_l580_580539

theorem triangles_in_dodecagon : ∃ (n : ℕ), n = 12 → ∃ (k : ℕ), k = 3 → choose 12 3 = 220 :=
by
  existsi 12
  intro n12
  existsi 3
  intro k3
  rw [n12, k3]
  sorry

end triangles_in_dodecagon_l580_580539


namespace shortest_segment_dividing_triangle_equal_area_l580_580598

theorem shortest_segment_dividing_triangle_equal_area 
  (a b c : ℕ)
  (h : a = 6 ∧ b = 8 ∧ c = 10) 
  (h_triangle : a^2 + b^2 = c^2) :
  ∃ t : ℝ, t = 4 ∧ 
           (∃ (D : ℝ) (E : ℝ),
             (D = a ∧ E = b ∨ D = b ∧ E = a) ∧
             (∃ area : ℝ, 
                let S_ABC := 1/2 * a * b,
                let S_ADE := 1/2 * D * E,
                S_ADE = 1/2 * S_ABC ∧ S_ADE = 12) ∧
           t = sqrt(x^2 + y^2)) :=
sorry

end shortest_segment_dividing_triangle_equal_area_l580_580598


namespace hobbes_winning_strategy_l580_580882

theorem hobbes_winning_strategy : 
  ∀ (F : set (set (fin 2020))),
  (∀ (calvin_picks : fin 1010 → fin 2020) (hobbes_picks : fin 1010 → fin 2020),
     (∀ S ∈ F, ¬ ∀ x ∈ S, ∃ i, calvin_picks i = x)) →
  2^2020 - 3^1010 :=
begin
  sorry
end

end hobbes_winning_strategy_l580_580882


namespace product_not_ending_in_1_l580_580744

theorem product_not_ending_in_1 : ∃ a b : ℕ, 111111 = a * b ∧ (a % 10 ≠ 1) ∧ (b % 10 ≠ 1) := 
sorry

end product_not_ending_in_1_l580_580744


namespace log_monotonicity_necessary_but_not_sufficient_for_ln_l580_580451

noncomputable def necessary_but_not_sufficient (a b : ℝ) : Prop :=
a > b ∧ (ln a > ln b → a > b) ∧ (a > b → ¬ (ln a > ln b))

theorem log_monotonicity {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  (ln a > ln b) ↔ (a > b) :=
sorry

theorem necessary_but_not_sufficient_for_ln (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  necessary_but_not_sufficient a b :=
sorry

end log_monotonicity_necessary_but_not_sufficient_for_ln_l580_580451


namespace collinear_vectors_x_value_unique_k_value_l580_580637

-- Problem (I)
theorem collinear_vectors_x_value (x : ℝ) :
  let a := (1, 2) : ℝ × ℝ,
      b := (x, 1) : ℝ × ℝ
  in (a.1 + b.1) * (1 - b.2) = (a.2 - b.2) * (1 - a.1) →
     x = 1/2 := 
by {
    sorry
}

-- Problem (II)
theorem unique_k_value (k x : ℝ) :
  let a := (1, 2) : ℝ × ℝ,
      b := (x, 1) : ℝ × ℝ
  in (k^2 * (a.1 * a.1 + a.2 * a.2) - (b.1 * b.1 + b.2 * b.2) = k^2) →
     (k = sqrt 5 / 5 ∨ k = -sqrt 5 / 5) := 
by {
    sorry
}

end collinear_vectors_x_value_unique_k_value_l580_580637


namespace average_daily_sales_after_10_yuan_reduction_price_reduction_for_1200_yuan_profit_l580_580771

-- Conditions from the problem statement
def initial_daily_sales : ℕ := 20
def profit_per_box : ℕ := 40
def additional_sales_per_yuan_reduction : ℕ := 2

-- Part 1: New average daily sales after a 10 yuan reduction
theorem average_daily_sales_after_10_yuan_reduction :
  (initial_daily_sales + 10 * additional_sales_per_yuan_reduction) = 40 :=
  sorry

-- Part 2: Price reduction needed to achieve a daily sales profit of 1200 yuan
theorem price_reduction_for_1200_yuan_profit :
  ∃ (x : ℕ), 
  (profit_per_box - x) * (initial_daily_sales + x * additional_sales_per_yuan_reduction) = 1200 ∧ x = 20 :=
  sorry

end average_daily_sales_after_10_yuan_reduction_price_reduction_for_1200_yuan_profit_l580_580771


namespace total_population_l580_580292

variables (b g t : ℕ)

theorem total_population (h1 : b = 4 * g) (h2 : g = 5 * t) : b + g + t = 26 * t :=
sorry

end total_population_l580_580292


namespace max_five_digit_value_l580_580087

theorem max_five_digit_value (A B C D E F G H I : ℕ) 
  (hA : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E)
  (hB : B ≠ C ∧ B ≠ D ∧ B ≠ E)
  (hC : C ≠ D ∧ C ≠ E)
  (hD : D ≠ E)
  (hABCDE : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10)
  (hF : |A - B| = F)
  (hG : |B - C| = G)
  (hH : |C - D| = H)
  (hI : |D - E| = I)
  : A * 10000 + B * 1000 + C * 100 + D * 10 + E ≤ 98274 :=
sorry

end max_five_digit_value_l580_580087


namespace min_value_l580_580764

-- Definition of the functions f and g
def f (x : Real) : Real := 2 * sin x * cos x
def g (x : Real) : Real := sin (2 * x + π / 6) + 1

-- Given conditions
variable (x1 x2 : Real)
variable (h : f x1 * g x2 = 2)

-- Question: Prove the minimum value of |2 * x1 + x2| is π / 3
theorem min_value (h : f x1 * g x2 = 2) : |2 * x1 + x2| = π / 3 :=
  sorry

end min_value_l580_580764


namespace centroids_concyclic_of_cyclic_quadrilateral_l580_580113

noncomputable def centroid (A B C : Complex) : Complex := (A + B + C) / 3

theorem centroids_concyclic_of_cyclic_quadrilateral
  (A B C D : Complex)
  (h_circle : ∃ O R, ∀ z ∈ {A, B, C, D}, Complex.abs (z - O) = R) :
  ∃ O' R', ∀ G ∈ {
    centroid A B C,
    centroid C D A,
    centroid B C D,
    centroid D A B
  }, Complex.abs (G - O') = R' := sorry

end centroids_concyclic_of_cyclic_quadrilateral_l580_580113


namespace complex_expression_eq_zero_l580_580170

theorem complex_expression_eq_zero :
  ((1 + complex.i) / (1 - complex.i) + complex.i^3 = 0) := sorry

end complex_expression_eq_zero_l580_580170


namespace number_of_unordered_pairs_divisible_by_109_l580_580203

theorem number_of_unordered_pairs_divisible_by_109 : 
  let set_s := set.range (0, 108 + 1)
  in ∃ n, 
    n = 54 ∧ 
    ∀ a b : ℕ, a ∈ set_s ∧ b ∈ set_s → 
      (109 ∣ (a^3 + b^3 - a * b)) ↔ 
      ∃ pairs : set (ℕ × ℕ), 
      (pairs = { (a, b) | a ∈ set_s ∧ b ∈ set_s ∧ 109 ∣ (a^3 + b^3 - a * b)} ∧ 
      pairs.card = 54


sorry

end number_of_unordered_pairs_divisible_by_109_l580_580203


namespace power_of_product_l580_580877

variable (x y : ℝ)

theorem power_of_product (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 :=
  sorry

end power_of_product_l580_580877


namespace candidate_lost_by_l580_580122

noncomputable def votes_cast : ℝ := 10000.000000000002

def candidate_percentage : ℝ := 0.40

def rival_percentage : ℝ := 0.60

def candidate_votes := candidate_percentage * votes_cast
def rival_votes := rival_percentage * votes_cast

def votes_lost := rival_votes - candidate_votes

theorem candidate_lost_by :
  votes_lost = 2000 := by
  -- proof goes here
  sorry

end candidate_lost_by_l580_580122


namespace harmonic_set_odd_minimum_harmonic_set_size_l580_580575

def is_harmonic_set (A : Set ℕ) : Prop :=
  ∀ a ∈ A, ∃ B C : Set ℕ, B ∩ C = ∅ ∧ B ∪ C = A \ {a} ∧ B.sum = C.sum

example : ¬ is_harmonic_set {1, 2, 3, 4, 5} := by
  sorry

theorem harmonic_set_odd (A : Set ℕ) (hA : is_harmonic_set A) : A.size % 2 = 1 := by
  sorry

theorem minimum_harmonic_set_size : ∃ A : Set ℕ, is_harmonic_set A ∧ A.size = 7 := by
  sorry

end harmonic_set_odd_minimum_harmonic_set_size_l580_580575


namespace james_time_to_run_100_meters_l580_580690

theorem james_time_to_run_100_meters
  (john_time_to_run_100_meters : ℕ → Prop)
  (john_first_4_meters : nat := 4)
  (john_total_time : nat := 13)
  (james_first_10_meters_time : nat := 2)
  (james_top_speed_faster_by : nat := 2):
  john_time_to_run_100_meters john_total_time → 
  (john_first_4_meters = 4) →
  ∀ n, (john_total_time - 1) = n / 8 →
  ∀ m, (100 - 10) = m / 10 →
  ∀ p, james_first_10_meters_time * 1 + p * 10 = 100 →
  (james_first_10_meters_time + p) = 11 :=
  sorry

end james_time_to_run_100_meters_l580_580690


namespace probability_prime_or_square_sum_l580_580037

open Nat

theorem probability_prime_or_square_sum : 
  let primes := [2, 3, 5, 7, 11] in
  let perfect_squares := [4, 9] in
  (∃ (a b : ℕ), a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} 
    ∧ (a + b) ∈ (primes ∪ perfect_squares)) →
  (22 / 36 = 11 / 18) :=
begin
  sorry
end

end probability_prime_or_square_sum_l580_580037


namespace class_size_difference_l580_580489

def class_sizes : List ℕ := [40, 40, 30, 30, 20, 10]

def num_students : ℕ := 200
def num_teachers : ℕ := 6
def num_students_in_assembly : ℕ := 30
def num_students_in_class : ℕ := 170

def t : ℚ := (class_sizes.sum) / num_teachers
def s : ℚ := class_sizes.map (λ size => size * size).sum / num_students_in_class

theorem class_size_difference :
  t - s = 1.86 :=
by
  sorry

end class_size_difference_l580_580489


namespace sample_size_proof_l580_580121

theorem sample_size_proof (p : ℝ) (N : ℤ) (n : ℤ) (h1 : N = 200) (h2 : p = 0.25) : n = 50 :=
by
  sorry

end sample_size_proof_l580_580121


namespace power_function_properties_l580_580977

/-- Given a power function  f(x) = x^a (a ∈ ℝ) passing through the point (2, sqrt 2),
we prove that:
1. a = 1 / 2
2. The function f(x) is increasing on the interval [0, ∞) -/
theorem power_function_properties {a : ℝ} (h : (2 : ℝ) ^ a = real.sqrt 2) :
  a = 1 / 2 ∧ ∀ x, 0 ≤ x → 0 < x → deriv (λ x : ℝ, x ^ (1 / 2)) x > 0 :=
by
  sorry

end power_function_properties_l580_580977


namespace find_X_l580_580659

theorem find_X (X : ℕ) (h1 : 2 + 1 + 3 + X = 3 + 4 + 5) : X = 6 :=
by
  sorry

end find_X_l580_580659


namespace max_seat_capacity_l580_580160

noncomputable def max_students := 449

def seats_per_row (i : ℕ) : ℕ := 14 + i

def max_students_row (i : ℕ) : ℕ := (seats_per_row i + 1) / 2

def total_students : ℕ := ∑ i in finset.range 25, max_students_row (i + 1)

theorem max_seat_capacity : total_students = max_students := by
  sorry

end max_seat_capacity_l580_580160


namespace mikes_earnings_l580_580725

-- Definitions based on the conditions:
def blade_cost : ℕ := 47
def game_count : ℕ := 9
def game_cost : ℕ := 6

-- The total money Mike made:
def total_money (M : ℕ) : Prop :=
  M - (blade_cost + game_count * game_cost) = 0

theorem mikes_earnings (M : ℕ) : total_money M → M = 101 :=
by
  sorry

end mikes_earnings_l580_580725


namespace compute_expression_l580_580173

theorem compute_expression : 85 * 1500 + (1 / 2) * 1500 = 128250 :=
by
  sorry

end compute_expression_l580_580173


namespace CarltonUniqueOutfits_l580_580532

theorem CarltonUniqueOutfits:
  ∀ (buttonUpShirts sweaterVests : ℕ), 
    buttonUpShirts = 3 →
    sweaterVests = 2 * buttonUpShirts →
    (sweaterVests * buttonUpShirts) = 18 :=
by
  intros buttonUpShirts sweaterVests h1 h2
  rw [h1, h2]
  simp
  sorry

end CarltonUniqueOutfits_l580_580532


namespace max_M_range_a_l580_580258

noncomputable def f (a x : ℝ) := (x - a) ^ 2 * real.exp x
noncomputable def g (x : ℝ) := x ^ 3 - x ^ 2 - 3

theorem max_M (x1 x2 : ℝ) (hx1 : x1 ∈ set.Icc 0 2) (hx2 : x2 ∈ set.Icc 0 2) :
  g x1 - g x2 ≤ 112 / 27 :=
sorry

theorem range_a (a : ℝ) :
  (∀ s t ∈ set.Icc 0 2, f a s ≥ g t) ↔ (a ≤ -1 ∨ a ≥ 2 + 1 / real.exp 1) :=
sorry

end max_M_range_a_l580_580258


namespace common_ratio_is_integer_l580_580979

open Int

variables {a d b q : ℤ}
variables (k_n : ℕ → ℤ)

-- Given definitions:
-- Arithmetic Progression: a_n = a + (n-1)d
def a_n (n : ℕ) := a + (n - 1 : ℤ) * d

-- Geometric Progression: b_n = b * q^(n-1)
def b_n (n : ℕ) := b * q^(n - 1 : ℤ)

-- Condition: every term of GP appears in AP
variable h : ∀ n, ∃ k, b * q^(n - 1 : ℤ) = a + k * d

theorem common_ratio_is_integer : q ∈ Int :=
  sorry

end common_ratio_is_integer_l580_580979


namespace convert_yahs_to_bahs_l580_580991

theorem convert_yahs_to_bahs :
  (∀ (bahs rahs yahs : ℝ), (10 * bahs = 18 * rahs) 
    ∧ (6 * rahs = 10 * yahs) 
    → (1500 * yahs / (10 / 6) / (18 / 10) = 500 * bahs)) :=
by
  intros bahs rahs yahs h
  sorry

end convert_yahs_to_bahs_l580_580991


namespace base_4_digits_l580_580112

theorem base_4_digits (b : ℕ) (h1 : b^3 ≤ 216) (h2 : 216 < b^4) : b = 5 :=
sorry

end base_4_digits_l580_580112


namespace knight_knave_assignment_l580_580656

inductive Person : Type
| A : Person
| B : Person
| C : Person

inductive Role : Type
| Knight : Role
| Knave : Role

open Person Role

-- Definitions based on the conditions:
def tells_truth_or_lie (r : Role) (stmt : Prop) : Prop :=
  match r with
  | Knight => stmt
  | Knave => ¬stmt

def A_statement (rC rB : Role) : Prop :=
  tells_truth_or_lie rA (rC = Knight → rB = Knave)

def C_statement (rA rC : Role) : Prop :=
  tells_truth_or_lie rC (rA ≠ rC)

-- The main theorem to prove
theorem knight_knave_assignment (rA rB rC : Role) :
  A_statement rC rB →
  C_statement rA rC →
  (rA = Knave ∧ rB = Knight ∧ rC = Knight) :=
sorry

end knight_knave_assignment_l580_580656


namespace audrey_peaches_l580_580518

variable (A : ℕ)
variable (P : ℕ := 48)
variable (D : ℕ := 22)

theorem audrey_peaches : A - P = D → A = 70 :=
by
  intro h
  sorry

end audrey_peaches_l580_580518


namespace equilateral_triangle_area_perimeter_l580_580755

theorem equilateral_triangle_area_perimeter (altitude : ℝ) : 
  altitude = Real.sqrt 12 →
  (exists area perimeter : ℝ, area = 4 * Real.sqrt 3 ∧ perimeter = 12) :=
by
  intro h_alt
  sorry

end equilateral_triangle_area_perimeter_l580_580755


namespace geometric_seq_a7_l580_580303

theorem geometric_seq_a7 (a : ℕ → ℝ) (r : ℝ) (h1 : a 3 = 16) (h2 : a 5 = 4) (h_geom : ∀ n, a (n + 1) = a n * r) : a 7 = 1 := by
  sorry

end geometric_seq_a7_l580_580303


namespace product_of_20_consecutive_numbers_as_permutation_l580_580328

open Nat

theorem product_of_20_consecutive_numbers_as_permutation 
  (n : ℕ) (h1 : n > 0) (h2 : n > 19) : 
  (n * (n-1) * (n-2) * ... * (n-19)) = n.perm 20 := 
  sorry

end product_of_20_consecutive_numbers_as_permutation_l580_580328


namespace max_b_minus_a_is_pi_div_2_l580_580612

theorem max_b_minus_a_is_pi_div_2
    (a b : ℝ)
    (h : ∀ x ∈ set.Icc a b, sin x * cos x - (ℝ.sqrt 2 / 2) * (sin x + cos x) + (ℝ.sqrt 2 / 2) ^ 2 ≤ 0) :
    b - a ≤ π / 2 := 
sorry

end max_b_minus_a_is_pi_div_2_l580_580612


namespace nikolai_wins_bet_l580_580728

section Investments

def ruble_to_dollar (rub_amount : ℝ) (exchange_rate : ℝ) : ℝ :=
  rub_amount / exchange_rate

def dollar_to_ruble (usd_amount : ℝ) (exchange_rate : ℝ) : ℝ :=
  usd_amount * exchange_rate

def compound_interest (principal : ℝ) (rate : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  principal * (1 + rate / n) ^ (n * t)

def simple_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * (1 + rate)

def effective_amount_after_fee (amount : ℝ) (fee_rate : ℝ) : ℝ :=
  amount * (1 - fee_rate)

def final_amount_nikolai : ℝ :=
  compound_interest 150000 0.071 12 1

def final_amount_maxim : ℝ :=
  let usd_amount := ruble_to_dollar 80000 58.42 in
  let final_usd_amount := simple_interest usd_amount 0.036 in
  dollar_to_ruble final_usd_amount 58.61

def final_amount_oksana : ℝ :=
  let coin_price := 19801 in
  10 * coin_price

def final_amount_olga : ℝ :=
  let shares := 250000 / 5809 in
  let value := shares * 8074 in
  effective_amount_after_fee value 0.04

theorem nikolai_wins_bet :
  final_amount_nikolai > final_amount_maxim ∧
  final_amount_nikolai > final_amount_oksana ∧
  final_amount_nikolai > final_amount_olga :=
by
  sorry

end Investments

end nikolai_wins_bet_l580_580728


namespace area_of_dodecagon_l580_580487

theorem area_of_dodecagon (r : ℝ) : 
  ∃ A : ℝ, (∃ n : ℕ, n = 12) ∧ (A = 3 * r^2) := 
by
  sorry

end area_of_dodecagon_l580_580487


namespace smallest_sum_positive_value_l580_580552

/-
Each of the numbers a_1, a_2, ..., a_{100} is \pm 1. 
Find the smallest possible positive value of sum_{1 ≤ i < j ≤ 100} a_i a_j.
-/

noncomputable def smallest_possible_sum : ℕ :=
  let a : fin 100 → ℤ := sorry
  S := ∑ i in finset.range 100, ∑ j in finset.Ico i 100, a ⟨i, sorry⟩ * a ⟨j, sorry⟩
  22

theorem smallest_sum_positive_value
  (a : fin 100 → ℤ)
  (h1 : ∀ i, a i = 1 ∨ a i = -1) :
  ∑ i in finset.range 100, ∑ j in finset.Ico i 100, a i * a j = 22 :=
sorry

end smallest_sum_positive_value_l580_580552


namespace parallelepiped_surface_area_l580_580144

theorem parallelepiped_surface_area (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 12) 
  (h2 : a * b * c = 8) : 
  6 * (a^2) = 24 :=
by
  sorry

end parallelepiped_surface_area_l580_580144


namespace average_speed_of_second_girl_l580_580426

theorem average_speed_of_second_girl :
  ∃ v : ℝ, (∀ (d t : ℝ), d = (7 + v) * t → d = 120 ∧ t = 12) → v = 3 := 
by
  use 3
  intro h
  cases h with d h_td
  cases h_td with h_d h_t
  rw [h_t] at h_d
  norm_num at h_d
  sorry

end average_speed_of_second_girl_l580_580426


namespace monotonic_increasing_f_a_eq_1_range_of_a_l580_580251

-- Define the function f(x)
def f (a x : ℝ) : ℝ := 4 * a * x^3 + 3 * abs (a - 1) * x^2 + 2 * a * x - a

-- Prove Part (Ⅰ): f(x) is monotonically increasing when a = 1
theorem monotonic_increasing_f_a_eq_1 : (∀ x1 x2 : ℝ, x1 < x2 → f 1 x1 < f 1 x2) :=
by
  sorry

-- Prove Part (Ⅱ): The range of a such that for x in [0,1], |f(x)| ≤ f(1) is [-3/4, +∞)
theorem range_of_a : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → abs (f a x) ≤ f a 1) ↔ a ∈ Icc (-(3 / 4)) +∞ :=
by
  sorry

end monotonic_increasing_f_a_eq_1_range_of_a_l580_580251


namespace ellipse_equation_exists_fixed_point_l580_580448

-- Definitions based on conditions
def is_ellipse_eq (a b : Real) (e : Real) : Prop :=
  a > b ∧ b > 0 ∧ e = 1 / 2 ∧ (∀ (x y : Real), x^2 / a^2 + y^2 / b^2 = 1)

def is_major_axis (A : Real × Real) (a : Real) : Prop :=
  A.snd = (sqrt 3 / 2) * a

def midpoint_y_eq (y_mid : Real) (a A_y : Real) : Prop :=
  y_mid = 6 - 3 * sqrt 3 ∧ A_y = (sqrt 3 / 2) * a

def fixed_point_condition (F : Real × Real) (x_fixed y_fixed : Real) : Prop :=
  F = (1, 0) ∧ x_fixed = 4 ∧ y_fixed = 0

-- Lean statement for Part (1)
theorem ellipse_equation_exists (a b : Real) (e : Real) :
  is_ellipse_eq a b e → a = 2 ∧ b = sqrt 3 → ∀ (x y : Real), x^2 / 4 + y^2 / 3 = 1 := 
sorry

-- Lean statement for Part (2)
theorem fixed_point MN'_passes_through (F : Real × Real) (x_fixed y_fixed : Real) :
  fixed_point_condition F x_fixed y_fixed → (∃ (M N N' : Real × Real), true 
  → line_passing_through MN'_passes_through x_fixed y_fixed) :=
sorry

-- Additional helpers (e.g., line equation) if necessary can be defined here

end ellipse_equation_exists_fixed_point_l580_580448


namespace sum_of_squares_of_roots_eq_l580_580401

-- Definitions derived directly from conditions
def a := 5
def b := 2
def c := -15

-- Sum of roots
def sum_of_roots : ℚ := (-b : ℚ) / a

-- Product of roots
def product_of_roots : ℚ := (c : ℚ) / a

-- Sum of the squares of the roots
def sum_of_squares_of_roots : ℚ := sum_of_roots^2 - 2 * product_of_roots

-- The statement that needs to be proved
theorem sum_of_squares_of_roots_eq : sum_of_squares_of_roots = 154 / 25 :=
by
  sorry

end sum_of_squares_of_roots_eq_l580_580401


namespace no_lonely_points_eventually_l580_580934

structure Graph (α : Type) :=
(vertices : Finset α)
(edges : α → Finset α)

namespace Graph

def is_lonely {α : Type} (G : Graph α) (coloring : α → Bool) (v : α) : Prop :=
  let neighbors := G.edges v
  let different_color_neighbors := neighbors.filter (λ w => coloring w ≠ coloring v)
  2 * different_color_neighbors.card > neighbors.card

end Graph

theorem no_lonely_points_eventually
  {α : Type}
  (G : Graph α)
  (initial_coloring : α → Bool) :
  ∃ (steps : Nat),
  ∀ (coloring : α → Bool),
  (∃ (t : Nat), t ≤ steps ∧ 
    (∀ v, ¬ Graph.is_lonely G coloring v)) :=
sorry

end no_lonely_points_eventually_l580_580934


namespace Q_even_for_large_d_l580_580918

open Finset

def D (σ : Perm (Fin n)) : ℕ :=
  ∑ k in range n, abs (σ k - k)

def Q (n d : ℕ) : ℕ :=
  card { σ : Perm (Fin n) | D σ = d }

theorem Q_even_for_large_d (n d : ℕ) (h : d ≥ 2 * n) : even (Q n d) := by
  -- Proof goes here
  sorry

end Q_even_for_large_d_l580_580918


namespace not_fibonacci_sum_of_eight_consecutive_l580_580743

def fibonacci : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n+2)   => fibonacci (n+1) + fibonacci n

theorem not_fibonacci_sum_of_eight_consecutive :
  ∀ k : ℕ, let S := (fibonacci (k + 1) + fibonacci (k + 2) + fibonacci (k + 3) + fibonacci (k + 4)
                      + fibonacci (k + 5) + fibonacci (k + 6) + fibonacci (k + 7) + fibonacci (k + 8)) in
  ¬ (∃ n : ℕ, S = fibonacci n) :=
by
  intro k
  let S := (fibonacci (k + 1) + fibonacci (k + 2) + fibonacci (k + 3) + fibonacci (k + 4)
             + fibonacci (k + 5) + fibonacci (k + 6) + fibonacci (k + 7) + fibonacci (k + 8))
  sorry

end not_fibonacci_sum_of_eight_consecutive_l580_580743


namespace euler_sum_of_squares_euler_sum_of_quads_l580_580443

theorem euler_sum_of_squares :
  ∑' n : ℕ, 1 / (n.succ : ℚ)^2 = π^2 / 6 := sorry

theorem euler_sum_of_quads :
  ∑' n : ℕ, 1 / (n.succ : ℚ)^4 = π^4 / 90 := sorry

end euler_sum_of_squares_euler_sum_of_quads_l580_580443


namespace correct_propositions_l580_580706

-- Definitions for the planes and lines
variable (Plane : Type) [HasPerp Plane] [HasParallel Plane]
variable (Line : Type) [HasPerp Line Plane] [HasParallel Line Plane]

-- Given conditions
variable (α β : Plane)
variable (l : Line)
hypothesis h1 : α ≠ β

-- Propositions
def prop1 : Prop :=
  (∀ m : Line, ∃ m_β ∈ β, m ∈ α → Perp m m_β) → Perp α β

def prop2 : Prop :=
  (∀ m : Line, m ∈ α → Parallel m β) → Parallel α β

def prop3 : Prop :=
  (Perp α β ∧ l ∈ α) → Perp l β

def prop4 : Prop :=
  (Parallel α β ∧ l ∈ α) → Parallel l β

-- Proof goal
theorem correct_propositions :
  prop1 α β l → prop2 α β l ∧ ¬prop3 α β l ∧ prop4 α β l :=
by
  split
  -- Assume prop1 is true
  intro h1
  exact h1
  split
  -- Assume prop2 is true
  intro h2
  exact h2
  split
  -- Prove that prop3 is false
  intro h3
  have h3_not : ¬(prop3 α β l) := 
    sorry
  exact h3_not
  -- Assume prop4 is true
  intro h4
  exact h4

end correct_propositions_l580_580706


namespace inequality_for_positive_integer_n_l580_580363

theorem inequality_for_positive_integer_n (n : ℕ) (hn : n > 0) :
  (finset.prod (finset.range n) (λ k, (2 * k + 1) / (2 * (k + 1))))
  ≤ 1 / real.sqrt (3 * n + 1) := 
by sorry

end inequality_for_positive_integer_n_l580_580363


namespace proof_problem_l580_580232

noncomputable def a : ℝ := 5 / 3
noncomputable def b : ℝ := Real.sqrt 5 / 3

def Eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

def Ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def InFocusLine {a b t : ℝ} (x y : ℝ) : Prop :=
  x - y - 1 = 0 ∨ x + y - 1 = 0

def PointP (t : ℝ) : (ℝ × ℝ) :=
  ⟨2 / (2 * t^2 + 3), -4 * t / (2 * t^2 + 3)⟩

theorem proof_problem :
  let a := 5 / 3
  let b := Real.sqrt 5 / 3
  let e := Eccentricity a b in
  Ellipse a b (PointP (Real.sqrt 1 / 2)).fst (PointP (Real.sqrt 1 / 2)).snd ∧
  Ellipse a b (PointP (-Real.sqrt 1 / 2)).fst (PointP (-Real.sqrt 1 / 2)).snd ∧
  InFocusLine (PointP (Real.sqrt 1 / 2)).fst (PointP (Real.sqrt 1 / 2)).snd ∧
  InFocusLine (PointP (-Real.sqrt 1 / 2)).fst (PointP (-Real.sqrt 1 / 2)).snd :=
by
  sorry

end proof_problem_l580_580232


namespace max_identifiable_liars_l580_580731

-- Definitions
def knight (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 23
def liar (n : ℕ) : Prop := 24 ≤ n ∧ n ≤ 223

def is_knight (n : ℕ) : Prop :=
  ∀ (s : ℕ → Prop), (∀ k, knight k → s k) ↔ (∀ l, liar l → ¬ s l)

def is_liar (n : ℕ) : Prop :=
  liar n ∧ ¬ is_knight n

-- Main theorem
theorem max_identifiable_liars : ∃ (k : ℕ), k ≤ 200 ∧ ∀ (m : ℕ), m > k → ∃ (n : ℕ), liar n ∧ n ∈ (finset.range 223).erase 200 := sorry

end max_identifiable_liars_l580_580731


namespace remaining_last_year_budget_is_13_l580_580493

-- Variables representing the conditions of the problem
variable (cost1 cost2 given_budget remaining this_year_spent remaining_last_year : ℤ)

-- Define the conditions as hypotheses
def conditions : Prop :=
  cost1 = 13 ∧ cost2 = 24 ∧ 
  given_budget = 50 ∧ 
  remaining = 19 ∧ 
  (cost1 + cost2 = 37) ∧
  (this_year_spent = given_budget - remaining) ∧
  (remaining_last_year + (cost1 + cost2 - this_year_spent) = remaining)

-- The statement that needs to be proven
theorem remaining_last_year_budget_is_13 : conditions cost1 cost2 given_budget remaining this_year_spent remaining_last_year → remaining_last_year = 13 :=
by 
  intro h
  sorry

end remaining_last_year_budget_is_13_l580_580493


namespace height_of_larger_triangle_l580_580083

theorem height_of_larger_triangle 
  (area_ratio : ℝ)
  (height_small_triangle : ℝ)
  (similar_triangles : Prop)
  (height_large_triangle : ℝ) :
  area_ratio = 1 / 9 →
  height_small_triangle = 5 →
  similar_triangles →
  height_large_triangle = height_small_triangle * 3 :=
begin
  intros h_ratio h_height_small h_similar,
  rw h_ratio at *,
  rw h_height_small at *,
  exact eq.symm (mul_eq_mul_left_iff.1 (eq.trans (sqrt_eq (by norm_num) (by norm_num)) (by norm_num))),
sorry,
end

# The above code imports the necessary library, defines the theorem with the conditions and concludes with the height of the larger triangle.

end height_of_larger_triangle_l580_580083


namespace Keith_initial_picked_l580_580699

-- Definitions based on the given conditions
def Mike_picked := 12
def Keith_gave_away := 46
def remaining_pears := 13

-- Question: Prove that Keith initially picked 47 pears.
theorem Keith_initial_picked :
  ∃ K : ℕ, K = 47 ∧ (K - Keith_gave_away + Mike_picked = remaining_pears) :=
sorry

end Keith_initial_picked_l580_580699


namespace sequence_u5_value_l580_580379

theorem sequence_u5_value (u : ℕ → ℝ) 
  (h_rec : ∀ n, u (n + 2) = 2 * u (n + 1) + u n)
  (h_u3 : u 3 = 9) 
  (h_u6 : u 6 = 128) : 
  u 5 = 53 := 
sorry

end sequence_u5_value_l580_580379


namespace guards_will_catch_monkey_l580_580398

-- Define the equilateral triangle and medians
structure EquilateralTriangle :=
(A B C : Point)
(median_A : Line)
(median_B : Line)
(median_C : Line)
(is_equilateral : equilateral_triangle A B C)

-- The paths that can be taken (medians and sides)
inductive Path 
| side {A B C : Point} : Path
| median {median_A median_B median_C : Line} : Path

-- The participants and their visibility
structure Participant :=
(position : Point)
(speed : ℝ)

structure Scenario :=
(monkey : Participant)
(guard1 : Participant)
(guard2 : Participant)
(paths : Path)
(speeds_equal : monkey.speed = guard1.speed ∧ guard1.speed = guard2.speed)
(visibility : monkey.position ∈ Set (guard1.position) ∧ monkey.position ∈ Set (guard2.position))

-- The statement to prove that the guards will catch the monkey
theorem guards_will_catch_monkey (scenario : Scenario) : ∃ t, ∀ s ≥ t, scenario.monkey.position = scenario.guard1.position ∨ scenario.monkey.position = scenario.guard2.position :=
sorry

end guards_will_catch_monkey_l580_580398


namespace determinant_transform_l580_580249

theorem determinant_transform 
  (p q r s : ℝ) 
  (h : p * s - q * r = 7) : 
  (p + 2 * r) * s - (q + 2 * s) * r = 7 := 
by 
sory

end determinant_transform_l580_580249


namespace range_of_function_l580_580007

theorem range_of_function :
  ∃ (S : Set ℝ), (∀ x : ℝ, (1 / 2)^(x^2 - 2) ∈ S) ∧ S = Set.Ioc 0 4 := by
  sorry

end range_of_function_l580_580007


namespace part1_solution_set_part2_range_a_l580_580262

noncomputable def inequality1 (a x : ℝ) : Prop :=
|a * x - 2| + |a * x - a| ≥ 2

theorem part1_solution_set : 
  (∀ x : ℝ, inequality1 1 x ↔ x ≥ 2.5 ∨ x ≤ 0.5) := 
sorry

theorem part2_range_a :
  (∀ x : ℝ, inequality1 a x) ↔ a ≥ 4 :=
sorry

end part1_solution_set_part2_range_a_l580_580262


namespace price_second_oil_per_litre_is_correct_l580_580115

-- Definitions based on conditions
def price_first_oil_per_litre := 54
def volume_first_oil := 10
def volume_second_oil := 5
def mixture_rate_per_litre := 58
def total_volume := volume_first_oil + volume_second_oil
def total_cost_mixture := total_volume * mixture_rate_per_litre
def total_cost_first_oil := volume_first_oil * price_first_oil_per_litre

-- The statement to prove
theorem price_second_oil_per_litre_is_correct (x : ℕ) (h : total_cost_first_oil + (volume_second_oil * x) = total_cost_mixture) : x = 66 :=
by
  sorry

end price_second_oil_per_litre_is_correct_l580_580115


namespace special_fractions_sum_l580_580545

theorem special_fractions_sum :
  let special_fractions := { ab : ℚ | ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ a + b = 20 ∧ ab = (a : ℚ) / b }
  let sums := { x : ℤ | ∃ (f1 f2 : ℚ), f1 ∈ special_fractions ∧ f2 ∈ special_fractions ∧ x = (f1 + f2).to_int }
  sums.to_finset.card = 10 :=
by sorry

end special_fractions_sum_l580_580545


namespace lambda_range_l580_580958

noncomputable def sequence (λ : ℝ) (n : ℕ+) := n.val^2 + λ * n.val

theorem lambda_range (λ : ℝ) : (∀ n : ℕ+, sequence λ n < sequence λ (n + 1)) → λ > -3 := 
by 
  intros h
  sorry

end lambda_range_l580_580958


namespace max_marked_cells_100x100_board_l580_580806

theorem max_marked_cells_100x100_board : 
  ∃ n, (3 * n + 1 = 100) ∧ (2 * n + 1) * (n + 1) = 2278 :=
by
  sorry

end max_marked_cells_100x100_board_l580_580806


namespace similar_triangles_height_l580_580074

theorem similar_triangles_height
  (a b : ℕ)
  (area_ratio: ℕ)
  (height_smaller : ℕ)
  (height_relation: height_smaller = 5)
  (area_relation: area_ratio = 9)
  (similarity: a / b = 1 / area_ratio):
  (∃ height_larger : ℕ, height_larger = 15) :=
by
  sorry

end similar_triangles_height_l580_580074


namespace number_of_schools_l580_580789

def yellow_balloons := 3414
def additional_black_balloons := 1762
def balloons_per_school := 859

def black_balloons := yellow_balloons + additional_black_balloons
def total_balloons := yellow_balloons + black_balloons

theorem number_of_schools : total_balloons / balloons_per_school = 10 :=
by
  sorry

end number_of_schools_l580_580789


namespace length_of_PS_l580_580306

theorem length_of_PS {P Q R S T : Type*}
  [euclidean_geometry P Q R]
  (h1 : ∠ R = 90)
  (h2 : PR = 9)
  (h3 : QR = 12)
  (h4 : ∠ PTS = 90)
  (h5 : ST = 6) :
  PS = 10 :=
sorry

end length_of_PS_l580_580306


namespace find_x_l580_580859

noncomputable theory

def line_equation (m b x : ℝ) : ℝ := m * x + b

theorem find_x (m b y target_x : ℝ) (h_slope : m = -3.5) (h_intercept : b = 1.5) (h_y : y = 1025) (h_target_x : target_x = -292.42857142857144) :
  y = line_equation m b target_x :=
by
  sorry

end find_x_l580_580859


namespace find_number_l580_580840

-- Statement of the problem in Lean 4
theorem find_number (n : ℝ) (h : n / 3000 = 0.008416666666666666) : n = 25.25 :=
sorry

end find_number_l580_580840


namespace unique_quadruples_l580_580559

noncomputable def verify_quadruples (a b c d : ℝ) : Prop :=
  (a = b * c ∨ a = b * d ∨ a = c * d ∨ b = a * c ∨ b = a * d ∨ b = c * d ∨ 
   c = a * b ∨ c = a * d ∨ c = b * d ∨ d = a * b ∨ d = a * c ∨ d = b * c)

theorem unique_quadruples (a b c d : ℝ) :
  verify_quadruples a b c d →
  (⟨a, b, c, d⟩ = ⟨0, 0, 0, 0⟩ ∨ ⟨a, b, c, d⟩ = ⟨1, 1, 1, 1⟩ ∨ 
   ⟨a, b, c, d⟩ = ⟨1, 1, -1, -1⟩ ∨ ⟨a, b, c, d⟩ = ⟨1, -1, -1, -1⟩ ∨
   ∃ (p : Multiset ℝ), p = {0, 0, 0, 0}.join ∨ p = {1, 1, 1, 1}.join ∨
   p = {1, 1, -1, -1}.join ∨ p = {1, -1, -1, -1}.join) :=
sorry

end unique_quadruples_l580_580559


namespace second_number_exists_l580_580142

theorem second_number_exists (x : ℕ) (h : 150 / x = 15) : x = 10 :=
sorry

end second_number_exists_l580_580142


namespace math_problem_l580_580228

noncomputable def a : ℕ → ℕ
| 0     := 2
| 1     := 6
| (n+2) := 2 * a (n+1) - a n + 2

def seq_sum (m : ℕ) : ℝ := 
  ∑ i in finset.range m, m / (a (i + 1) : ℝ)

theorem math_problem (m : ℕ) : 
  ∀ m > 0, ⌊seq_sum m⌋ = m - 1 :=
by 
  sorry

end math_problem_l580_580228


namespace range_of_k_for_ellipse_l580_580647

theorem range_of_k_for_ellipse (k : ℝ) :
  (4 - k > 0) ∧ (k - 1 > 0) ∧ (4 - k ≠ k - 1) ↔ (1 < k ∧ k < 4 ∧ k ≠ 5 / 2) :=
by
  sorry

end range_of_k_for_ellipse_l580_580647


namespace train_speed_l580_580817

theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) (h_train_length : train_length = 100) (h_bridge_length : bridge_length = 300) (h_crossing_time : crossing_time = 12) : 
  (train_length + bridge_length) / crossing_time = 33.33 := 
by 
  -- sorry allows us to skip the proof
  sorry

end train_speed_l580_580817


namespace find_constant_b_l580_580202

theorem find_constant_b 
  (a b c : ℝ)
  (h1 : 3 * a = 9) 
  (h2 : (-2 * a + 3 * b) = -5) 
  : b = 1 / 3 :=
by 
  have h_a : a = 3 := by linarith
  
  have h_b : -2 * 3 + 3 * b = -5 := by linarith [h2]
  
  linarith

end find_constant_b_l580_580202


namespace exists_root_in_interval_2_3_l580_580158

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 3

theorem exists_root_in_interval_2_3 : ∃ x ∈ Ioo (2 : ℝ) (3 : ℝ), f x = 0 :=
by
  sorry

end exists_root_in_interval_2_3_l580_580158


namespace alpha_beta_square_l580_580273

noncomputable def roots_of_quadratic : set ℝ := 
  { x | x^2 = 2 * x + 1 }

theorem alpha_beta_square (α β : ℝ) (hα : α ∈ roots_of_quadratic) (hβ : β ∈ roots_of_quadratic) (hαβ : α ≠ β) :
  (α - β)^2 = 8 :=
sorry

end alpha_beta_square_l580_580273


namespace finite_good_numbers_not_divisible_by_k_l580_580919

def is_good_number (n : ℕ) : Prop :=
  ∀ m : ℕ, m < n → τ m < τ n

theorem finite_good_numbers_not_divisible_by_k (k : ℕ) (hpos : 0 < k) :
  ∃ N : ℕ, ∀ n : ℕ, is_good_number n → k ∣ n ∨ n ≤ N :=
sorry

end finite_good_numbers_not_divisible_by_k_l580_580919


namespace limit_n_a_n_l580_580544

noncomputable def L (x : ℝ) : ℝ := x - (x^3) / 3

def a_n (n : ℕ) (h : n > 0) : ℝ :=
  (L^[n]) (15 / n)

theorem limit_n_a_n :
  filter.tendsto (λ n : ℕ, n * a_n n (Nat.pos n)) filter.at_top (nhds 15) :=
sorry

end limit_n_a_n_l580_580544


namespace integral_evaluation_l580_580193

noncomputable def integral_expression : ℝ :=
  ∫ x in -1..1, (sqrt (1 - x^2) + x)

theorem integral_evaluation : integral_expression = Real.pi / 2 :=
by
  sorry

end integral_evaluation_l580_580193


namespace triangular_prism_volume_l580_580915

theorem triangular_prism_volume (a : ℝ) (H : ℝ)
  (S_base : ℝ := (a^2 * Real.sqrt 3) / 4)
  (S_lateral : ℝ := 3 * a * H)
  (sum_bases_eq_lateral : S_lateral = 2 * S_base) :
  let V := S_base * H in
  V = a^3 / 8 := by
  sorry

end triangular_prism_volume_l580_580915


namespace midpoint_proof_l580_580373

variables (BC: Type) (A I_A H O K_A: BC) (G: BC -> BC -> BC)

-- Definitions and hypotheses
def homothety (center : BC) (ratio : ℝ) (point : BC) : BC := sorry

-- Given conditions
axiom H_A_vector_equation : vector HA = -2 • vector (O - I_A)
axiom A_X_vector_equation : vector AX = 2 • vector (O - I_A)
axiom X_eq_H : X = H

-- Prove that I_A is the midpoint of H and K_A
theorem midpoint_proof :
  (vector I_A = (vector H + vector K_A) / 2) :=
sorry

end midpoint_proof_l580_580373


namespace height_of_larger_triangle_l580_580086

theorem height_of_larger_triangle 
  (area_ratio : ℝ)
  (height_small_triangle : ℝ)
  (similar_triangles : Prop)
  (height_large_triangle : ℝ) :
  area_ratio = 1 / 9 →
  height_small_triangle = 5 →
  similar_triangles →
  height_large_triangle = height_small_triangle * 3 :=
begin
  intros h_ratio h_height_small h_similar,
  rw h_ratio at *,
  rw h_height_small at *,
  exact eq.symm (mul_eq_mul_left_iff.1 (eq.trans (sqrt_eq (by norm_num) (by norm_num)) (by norm_num))),
sorry,
end

# The above code imports the necessary library, defines the theorem with the conditions and concludes with the height of the larger triangle.

end height_of_larger_triangle_l580_580086


namespace translate_vertex_l580_580387

/-- Given points A and B and their translations, verify the translated coordinates of B --/
theorem translate_vertex (A A' B B' : ℝ × ℝ)
  (hA : A = (0, 2))
  (hA' : A' = (-1, 0))
  (hB : B = (2, -1))
  (h_translation : A' = (A.1 - 1, A.2 - 2)) :
  B' = (B.1 - 1, B.2 - 2) :=
by
  sorry

end translate_vertex_l580_580387


namespace ratio_is_one_to_one_l580_580814

noncomputable def ratio_of_shaded_area_to_circle_area
  (x : ℝ)
  (h1 : 8 * x = AB)
  (h2 : 6 * x = AC)
  (h3 : 2 * x = CB)
  (h4 : CD = sqrt 3 * x)
  (h5 : CD ⊥ AB) : ℝ :=
let shaded_area := 8 * π * x^2 - (4.5 * π * x^2 + 0.5 * π * x^2) in
let circle_area := π * (sqrt 3 * x)^2 in
shaded_area / circle_area

theorem ratio_is_one_to_one (x : ℝ)
  (AB AC CB CD : ℝ)
  (h1 : 8 * x = AB)
  (h2 : 6 * x = AC)
  (h3 : 2 * x = CB)
  (h4 : CD = sqrt 3 * x)
  (h5 : CD ⊥ AB)
  : ratio_of_shaded_area_to_circle_area x h1 h2 h3 h4 h5 = 1 :=
by
  sorry

end ratio_is_one_to_one_l580_580814


namespace carlton_outfits_l580_580534

theorem carlton_outfits (button_up_shirts sweater_vests : ℕ) 
  (h1 : sweater_vests = 2 * button_up_shirts)
  (h2 : button_up_shirts = 3) :
  sweater_vests * button_up_shirts = 18 :=
by
  sorry

end carlton_outfits_l580_580534


namespace complex_quadrant_l580_580931

open Complex

theorem complex_quadrant :
  let z := (1 - I) * (3 + I)
  z.re > 0 ∧ z.im < 0 :=
by
  sorry

end complex_quadrant_l580_580931


namespace mode_and_median_l580_580777

theorem mode_and_median (data : List ℕ) (h_data : data = [6, 7, 6, 9, 8]) :
  let mode := 6
  let median := 7
  List.frequency data mode > List.frequency data (mode - 1) ∧ List.frequency data mode > List.frequency data (mode + 1) ∧ List.nthLe (List.sort (≤) data) 2 sorry = median :=
by
  sorry

end mode_and_median_l580_580777


namespace probability_first_ace_second_spade_l580_580420

theorem probability_first_ace_second_spade :
  let deck := List.range 52 in
  let first_is_ace (card : ℕ) := card % 13 = 0 in
  let second_is_spade (card : ℕ) := card / 13 = 3 in
  let events :=
    [ ((first_is_ace card, second_is_spade card') | card ∈ deck, card' ∈ List.erase deck card) ] in
  let favorable_events :=
    [(true, true)] in
  (List.count (λ event => event ∈ favorable_events) events).toRat /
  (List.length events).toRat = 1 / 52 :=
sorry

end probability_first_ace_second_spade_l580_580420


namespace tan_alpha_eq_two_and_expression_value_sin_tan_simplify_l580_580454

-- First problem: Given condition and expression to be proved equal to the correct answer.
theorem tan_alpha_eq_two_and_expression_value (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (2 * Real.pi - α) + Real.cos (Real.pi + α)) / 
  (Real.cos (α - Real.pi) - Real.cos (3 * Real.pi / 2 - α)) = -3 := sorry

-- Second problem: Given expression to be proved simplified to the correct answer.
theorem sin_tan_simplify :
  Real.sin (50 * Real.pi / 180) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi/180)) = 1 := sorry

end tan_alpha_eq_two_and_expression_value_sin_tan_simplify_l580_580454


namespace maximum_marks_l580_580740

theorem maximum_marks (M : ℝ) (P : ℝ) 
  (h1 : P = 0.45 * M) -- 45% of the maximum marks to pass
  (h2 : P = 210 + 40) -- Pradeep's marks plus failed marks

  : M = 556 := 
sorry

end maximum_marks_l580_580740


namespace simplification_of_exponential_expression_l580_580802

theorem simplification_of_exponential_expression :
  (3^3 * 3^(-4)) / (3^(-2) * 3^5) = 1 / 81 := by
  sorry

end simplification_of_exponential_expression_l580_580802


namespace point_B_possible_values_l580_580732

-- Define point A
def A : ℝ := 1

-- Define the condition that B is 3 units away from A
def units_away (a b : ℝ) : ℝ := abs (b - a)

theorem point_B_possible_values :
  ∃ B : ℝ, units_away A B = 3 ∧ (B = 4 ∨ B = -2) := by
  sorry

end point_B_possible_values_l580_580732


namespace Rachel_total_earnings_l580_580366

-- Define the constants for the conditions
def hourly_wage : ℝ := 12
def people_served : ℕ := 20
def tip_per_person : ℝ := 1.25

-- Define the problem
def total_money_made : ℝ := hourly_wage + (people_served * tip_per_person)

-- State the theorem to be proved
theorem Rachel_total_earnings : total_money_made = 37 := by
  sorry

end Rachel_total_earnings_l580_580366


namespace entrepreneur_should_reduce_production_by_20_percent_l580_580982

noncomputable def percentage_reduction : ℕ → ℕ → ℕ → ℕ → ℕ := 
  λ (total_items : ℕ) (price_per_item : ℕ) (profit : ℕ) (variable_cost_per_item : ℕ),
  let total_revenue := total_items * price_per_item in
  let constant_costs := total_revenue - profit in
  let new_production := constant_costs / (price_per_item - variable_cost_per_item) in
  let production_reduction := total_items - new_production in
  (production_reduction * 100) / total_items

theorem entrepreneur_should_reduce_production_by_20_percent 
  (total_items : ℕ) (price_per_item : ℕ) (profit : ℕ) (variable_cost_per_item : ℕ) 
  (h_total_items : total_items = 4000)
  (h_price_per_item : price_per_item = 6250)
  (h_profit : profit = 2000000) -- 2 million
  (h_variable_cost_per_item : variable_cost_per_item = 3750) :
  percentage_reduction total_items price_per_item profit variable_cost_per_item = 20 :=
sorry

end entrepreneur_should_reduce_production_by_20_percent_l580_580982


namespace infinite_solutions_x2y2z2_eq_x3y3z3_l580_580682

theorem infinite_solutions_x2y2z2_eq_x3y3z3 :
  ∃ infinitely_many (x y z : ℤ), x^2 + y^2 + z^2 = x^3 + y^3 + z^3 :=
sorry

end infinite_solutions_x2y2z2_eq_x3y3z3_l580_580682


namespace part1_store_a_cost_part1_store_b_cost_part2_cost_comparison_part3_cost_effective_plan_l580_580126

-- Defining the conditions
def racket_price : ℕ := 50
def ball_price : ℕ := 20
def num_rackets : ℕ := 10

-- Store A cost function
def store_A_cost (x : ℕ) : ℕ := 20 * x + 300

-- Store B cost function
def store_B_cost (x : ℕ) : ℕ := 16 * x + 400

-- Part (1): Express the costs in algebraic form
theorem part1_store_a_cost (x : ℕ) (hx : 10 < x) : store_A_cost x = 20 * x + 300 := by
  sorry

theorem part1_store_b_cost (x : ℕ) (hx : 10 < x) : store_B_cost x = 16 * x + 400 := by
  sorry

-- Part (2): Cost for x = 40
theorem part2_cost_comparison : store_A_cost 40 > store_B_cost 40 := by
  sorry

-- Part (3): Most cost-effective purchasing plan
def store_a_cost_rackets : ℕ := racket_price * num_rackets
def store_a_free_balls : ℕ := num_rackets
def remaining_balls (total_balls : ℕ) : ℕ := total_balls - store_a_free_balls
def store_b_cost_remaining_balls (remaining_balls : ℕ) : ℕ := remaining_balls * ball_price * 4 / 5

theorem part3_cost_effective_plan : store_a_cost_rackets + store_b_cost_remaining_balls (remaining_balls 40) = 980 := by
  sorry

end part1_store_a_cost_part1_store_b_cost_part2_cost_comparison_part3_cost_effective_plan_l580_580126


namespace exists_positive_naturals_with_S_conditions_l580_580333

noncomputable def S (x : ℕ) : ℕ :=
x.digits 10 |>.sum

theorem exists_positive_naturals_with_S_conditions :
  ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧
  S(a + b) < 5 ∧ S(b + c) < 5 ∧ S(c + a) < 5 ∧ S(a + b + c) > 50 :=
by
  sorry

end exists_positive_naturals_with_S_conditions_l580_580333


namespace glove_position_half_height_l580_580797

noncomputable def glove_position (v_e v_s H t : ℝ) : ℝ :=
  let combined_speed := v_s + v_e in
  H / (2 * v_e) * v_e

theorem glove_position_half_height (v_e v_s H : ℝ) (h1 : v_s = v_e) (h2 : H > 0) :
  glove_position v_e v_s H (H / (2 * v_e)) = H / 2 :=
by
  have combined_speed := v_s + v_e
  have time_to_top := H / combined_speed
  have h3 : combined_speed = 2 * v_e, from by rw [h1]; ring
  rw [h3] at time_to_top
  have h4 : time_to_top = H / (2 * v_e), from by congr; ring
  simp [glove_position, h4]
  sorry

end glove_position_half_height_l580_580797


namespace log_expression_evaluation_l580_580896

theorem log_expression_evaluation :
  (2^3 = 8) →
  (logBase 2 (2^3) = 3) →
  (logBase 2 2 = 1) →
  (logBase 2 (3 * logBase 2 8))^3 = 31.881 :=
by
  intro h1 h2 h3
  sorry

end log_expression_evaluation_l580_580896


namespace sum_first_9000_terms_l580_580781

noncomputable def geom_sum (a r : ℝ) (n : ℕ) : ℝ :=
a * ((1 - r^n) / (1 - r))

theorem sum_first_9000_terms (a r : ℝ) (h1 : geom_sum a r 3000 = 1000) 
                              (h2 : geom_sum a r 6000 = 1900) : 
                              geom_sum a r 9000 = 2710 := 
by sorry

end sum_first_9000_terms_l580_580781


namespace isosceles_triangle_geometric_mean_locus_l580_580946

theorem isosceles_triangle_geometric_mean_locus
  (A B C M : Point)
  (h_tri : isosceles_triangle A B C)
  (AB_eq_BC : distance A B = distance B C)
  (d1 d2 d3 : ℝ)
  (h_d1 : distance M (segment A B) = d1)
  (h_d2 : distance M (segment B C) = d2)
  (h_d3 : distance M (segment A C) = d3)
  (h_geom_mean : d3 = real.sqrt (d1 * d2)) :
  exists (O : Point) (r : ℝ), circle O r ∧ M ∈ arc O r A C :=
sorry

end isosceles_triangle_geometric_mean_locus_l580_580946


namespace CarltonUniqueOutfits_l580_580530

theorem CarltonUniqueOutfits:
  ∀ (buttonUpShirts sweaterVests : ℕ), 
    buttonUpShirts = 3 →
    sweaterVests = 2 * buttonUpShirts →
    (sweaterVests * buttonUpShirts) = 18 :=
by
  intros buttonUpShirts sweaterVests h1 h2
  rw [h1, h2]
  simp
  sorry

end CarltonUniqueOutfits_l580_580530


namespace triangle_angle_B_eq_pi_div_3_l580_580679

theorem triangle_angle_B_eq_pi_div_3 
  (a b c : ℝ) 
  (h : a^2 + c^2 = b^2 + a * c) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c)
  (triangle_ineq : a + b > c ∧ a + c > b ∧ b + c > a) :
  ∠B = π / 3 :=
sorry

end triangle_angle_B_eq_pi_div_3_l580_580679


namespace certain_number_mod_l580_580092

theorem certain_number_mod (n : ℤ) : (73 * n) % 8 = 7 → n % 8 = 7 := 
by sorry

end certain_number_mod_l580_580092


namespace inequality_range_l580_580097

theorem inequality_range (x : ℝ) (h : x > 0) : 
  (x^{0.5 * Real.log (x) / Real.log (0.5) - 3} >= 0.5^{3 - 2.5 * Real.log (x) / Real.log (0.5)}) ↔ (0.125 ≤ x ∧ x ≤ 4) :=
sorry

end inequality_range_l580_580097


namespace passes_through_midpoint_of_BC_iff_l580_580700

variable {α : Type*} [LinearOrderedField α]

def isAcuteTriangle (A B C D : EuclideanSpace α) : Prop :=
  ∠ B A C < (π/2 : Real.Angle) ∧ ∠ C B A < π/2 ∧ ∠ A C B < π/2 ∧ A ≠ C

def isFootOfAltitude (A B C D : EuclideanSpace α) : Prop :=
  angle_eq (∠ B A D) (π/2 : Real.Angle) ∧ angle_eq (∠ C A D) (π/2 : Real.Angle)

def isCircumcircle (A B C ω : EuclideanSpace α) : Prop :=
  ∀ x, (distance x A = distance x B ∧ distance x B = distance x C) ↔ x ∈ ω

def isTangentTo (O1 O2 : EuclideanSpace α) (ω AD BD CD : Set (EuclideanSpace α)) : Prop :=
  (∀ x, x ∈ O1 → distance x AD = distance x BD ∧ distance x AD = distance ω) ∧
  (∀ y, y ∈ O2 → distance y AD = distance y CD ∧ distance y AD = distance ω)

def isInteriorCommonTangent (ℓ AD : Set (EuclideanSpace α)) (ω1 ω2 : EuclideanSpace α) : Prop :=
  (∀ P1 P2, P1 ∈ ω1 → P2 ∈ ω2 → P1 ≠ P2 ∧ is_tangent ℓ ω1 ∧ is_tangent ℓ ω2) ∧ (ℓ ≠ AD)

theorem passes_through_midpoint_of_BC_iff :
  ∀ (A B C D : EuclideanSpace α) (ω ω1 ω2 : EuclideanSpace α) (AD AD' : Set (EuclideanSpace α)) (ℓ : Set (EuclideanSpace α)),
  isAcuteTriangle A B C D →
  isFootOfAltitude A B C D →
  isCircumcircle A B C ω →
  isTangentTo ω1 ω2 ω AD AD' →
  isInteriorCommonTangent ℓ AD' ω1 ω2 →
  (passes_through_midpoint ℓ B C ↔ 2 * distance B C = distance A B + distance A C) :=
sorry

end passes_through_midpoint_of_BC_iff_l580_580700


namespace max_value_f_l580_580547

def otimes (a b : ℝ) : ℝ :=
  if a ≤ b then a else b

def f (x : ℝ) : ℝ :=
  otimes (Real.sin x) (Real.cos x)

theorem max_value_f : ∃ x ∈ ℝ, f x = Real.sqrt 2 / 2 :=
  sorry

end max_value_f_l580_580547


namespace ratio_of_areas_l580_580676

-- Define the given elements of the triangle PQR with points S and T
variables (P Q R S T : Type) [HasDistance P] [HasDistance Q] [HasDistance R]
(PQ QR PR PS PT : ℝ)
(hPQ : PQ = 30)
(hQR : QR = 50)
(hPR : PR = 54)
(hPS : PS = 18)
(hPT : PT = 36)
(on_S_PQ : S ∈ line_segment P Q)
(on_T_PR : T ∈ line_segment P R)

-- Define the areas of triangles and quadrilateral
noncomputable def area_triangle (A B C : P) : ℝ := sorry -- Placeholder for the area of triangle A, B, C
noncomputable def area_quadrilateral (A B C D : P) : ℝ := sorry -- Placeholder for the area of quadrilateral A, B, C, D

-- Statement we need to prove
theorem ratio_of_areas (h_similar : ∃ U : P, U ∈ line_segment P R ∧ Triangle.similar (triangle P S T) (triangle P Q U)):
  (area_triangle P S T) / (area_quadrilateral Q R S T) = 9 / 16 :=
by sorry

end ratio_of_areas_l580_580676


namespace part1_part2_l580_580221

noncomputable def x : ℝ := 1 - Real.sqrt 2
noncomputable def y : ℝ := 1 + Real.sqrt 2

theorem part1 : x^2 + 3 * x * y + y^2 = 3 := by
  sorry

theorem part2 : (y / x) - (x / y) = -4 * Real.sqrt 2 := by
  sorry

end part1_part2_l580_580221


namespace right_angled_trapezoid_base_height_l580_580775

theorem right_angled_trapezoid_base_height {a b : ℝ} (h : a = b) :
  ∃ (base height : ℝ), base = a ∧ height = b := 
by
  sorry

end right_angled_trapezoid_base_height_l580_580775


namespace range_of_m_l580_580967

theorem range_of_m
  (m : ℝ)
  (h1 : (m - 1) * (3 - m) ≠ 0) 
  (h2 : 3 - m > 0) 
  (h3 : m - 1 > 0) 
  (h4 : 3 - m ≠ m - 1) :
  1 < m ∧ m < 3 ∧ m ≠ 2 :=
sorry

end range_of_m_l580_580967


namespace flow_rate_DE_flow_rate_BC_flow_rate_GF_l580_580130

open Classical

-- Define the nodes and channels
inductive Node
| A | B | C | D | E | F | G | H

open Node

-- Define the flow rate function
def flow_rate (channel : (Node × Node)) : ℝ := sorry

-- Conditions
variable (q0 : ℝ)

-- Conditions from the problem
axiom flow_in_A : (flow_rate (A, B)) = q0
axiom sum_of_flow_rates_constant : ∀ (n m : Node), sum_of_flow_rates n m (flow_rate) = q0

-- Symmetry in the system
axiom symmetry_bc_cd : flow_rate (B, C) = flow_rate (C, D)
axiom symmetry_bg_gd : flow_rate (B, G) = flow_rate (G, D)

-- Questions and Expected Answers
theorem flow_rate_DE (q0 : ℝ) : flow_rate (D, E) = (4 / 7) * q0 := sorry
theorem flow_rate_BC (q0 : ℝ) : flow_rate (B, C) = (2 / 7) * q0 := sorry
theorem flow_rate_GF (q0 : ℝ) : flow_rate (G, F) = (3 / 7) * q0 := sorry

end flow_rate_DE_flow_rate_BC_flow_rate_GF_l580_580130


namespace find_angle_C_l580_580662

variable {A B C a b c : ℝ}
variable (hAcute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
variable (hTriangle : A + B + C = π)
variable (hSides : a > 0 ∧ b > 0 ∧ c > 0)
variable (hCondition : Real.sqrt 3 * a = 2 * c * Real.sin A)

theorem find_angle_C (hA_pos : A ≠ 0) : C = π / 3 :=
  sorry

end find_angle_C_l580_580662


namespace boys_other_communities_l580_580660

/-- 
In a school of 850 boys, 44% are Muslims, 28% are Hindus, 
10% are Sikhs, and the remaining belong to other communities.
Prove that the number of boys belonging to other communities is 153.
-/
theorem boys_other_communities
  (total_boys : ℕ)
  (percentage_muslims percentage_hindus percentage_sikhs : ℚ)
  (h_total_boys : total_boys = 850)
  (h_percentage_muslims : percentage_muslims = 44)
  (h_percentage_hindus : percentage_hindus = 28)
  (h_percentage_sikhs : percentage_sikhs = 10) :
  let percentage_others := 100 - (percentage_muslims + percentage_hindus + percentage_sikhs)
  let number_others := (percentage_others / 100) * total_boys
  number_others = 153 := 
by
  sorry

end boys_other_communities_l580_580660


namespace program_K_final_value_l580_580390

theorem program_K_final_value :
  ∃ (K : ℕ), 
    (∃ (S : ℕ), S = 1 ∧ K = 1) →
    (∀ n, S = S^2 + 1 ∧ K = K + 1 ∧ S < 100) →
    K = 4 := 
by
  sorry

end program_K_final_value_l580_580390


namespace problem_a_problem_b_l580_580284

variable (w x y z t : ℝ)

theorem problem_a (h1 : w = 0.60 * x) (h2 : x = 0.60 * y) (h3 : z = 0.54 * y) (h4 : t = 0.48 * x) :
  (z - w) / w = 0.50 :=
by
  sorry

theorem problem_b (h1 : w = 0.60 * x) (h2 : x = 0.60 * y) (h3 : z = 0.54 * y) (h4 : t = 0.48 * x) :
  (w - t) / w = 0.20 :=
by
  sorry

end problem_a_problem_b_l580_580284


namespace probability_at_least_one_unqualified_can_is_three_fifths_l580_580123

noncomputable def probability_of_at_least_one_unqualified_can : ℚ :=
  let total_outcomes := Nat.choose 6 2 in
  let favorable_outcomes := Nat.choose 2 1 * Nat.choose 4 1 + Nat.choose 2 2 in
  favorable_outcomes / total_outcomes

theorem probability_at_least_one_unqualified_can_is_three_fifths :
  probability_of_at_least_one_unqualified_can = 3 / 5 :=
by
  sorry

end probability_at_least_one_unqualified_can_is_three_fifths_l580_580123


namespace sequence_pattern_l580_580195

def seq_sum (n : ℕ) : ℤ :=
(∑ i in range (n + 1), (-1)^(i+1) * (i : ℤ)^2)

def partial_sum (n : ℕ) : ℕ :=
(∑ i in range (n), (i + 1))

theorem sequence_pattern (n : ℕ) : seq_sum n = (-1)^(n+1) * (partial_sum n) := 
by 
  sorry

end sequence_pattern_l580_580195


namespace Carl_chops_more_onions_than_Brittney_l580_580520

theorem Carl_chops_more_onions_than_Brittney :
  let Brittney_rate := 15 / 5
  let Carl_rate := 20 / 5
  let Brittney_onions := Brittney_rate * 30
  let Carl_onions := Carl_rate * 30
  Carl_onions = Brittney_onions + 30 :=
by
  sorry

end Carl_chops_more_onions_than_Brittney_l580_580520


namespace ab_value_l580_580104

theorem ab_value (a b : ℤ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 50) : a * b = 7 := by
  sorry

end ab_value_l580_580104


namespace train_total_travel_time_l580_580152

noncomputable def totalTravelTime (d1 d2 s1 s2 : ℝ) : ℝ :=
  (d1 / s1) + (d2 / s2)

theorem train_total_travel_time : 
  totalTravelTime 150 200 50 80 = 5.5 :=
by
  sorry

end train_total_travel_time_l580_580152


namespace common_root_condition_l580_580214

theorem common_root_condition : ∃ x p : ℕ, (3 * x ^ 2 - 4 * x + p - 2 = 0) ∧ (x ^ 2 - 2 * p * x + 5 = 0) ∧ x = 1 ∧ p = 3 :=
by
  use 1
  use 3
  split
  { norm_num }
  split
  { norm_num }
  split
  { rfl }
  { rfl }

end common_root_condition_l580_580214


namespace area_inside_S_but_not_in_R_l580_580154

/-- A unit square is given. On each side of the square, one equilateral triangle with side length 1 is constructed outside the square and one inside the square. Let R be the region formed by the union of the square and these 8 triangles. Suppose S is the smallest convex polygon that contains R. Prove that the area inside S but not in R is 1. -/
theorem area_inside_S_but_not_in_R :
  let square_area := 1
  let triangle_area := (sqrt 3) / 4
  let R := square_area + 8 * triangle_area
  let S := 2 + 2 * sqrt 3
  S - R = 1 :=
begin
  let square_area := 1,
  let triangle_area := (sqrt 3) / 4,
  let R := square_area + 8 * triangle_area,
  let S := 2 + 2 * sqrt 3,
  show S - R = 1,
  sorry,
end

end area_inside_S_but_not_in_R_l580_580154


namespace Cinderella_finds_correct_labels_l580_580171

/-- Cinderella was given three mislabeled bags: “Poppy,” “Millet,” and “Mixture.” 
Each label is incorrect. By picking one seed from the bag labeled “Mixture,” 
she can determine the correct contents of all bags. -/
theorem Cinderella_finds_correct_labels:
  ∃ (B_mixture B_poppy B_millet : Type),
    (B_mixture ≠ B_poppy ∧ B_mixture ≠ B_millet ∧ B_poppy ≠ B_millet) ∧
    (∀ seed, seed ∈ B_mixture) ∧ -- picking one seed from "Mixture" labeled bag
    (if (some seed) ∈ B_mixture
    then 
      B_mixture = Poppy ∧ B_poppy = Millet ∧ B_millet = Mixture
    else 
      B_mixture = Millet ∧ B_poppy = Poppy ∧ B_millet = Mixture)
sorry

end Cinderella_finds_correct_labels_l580_580171


namespace complex_modulus_eq_sqrt_two_l580_580248

theorem complex_modulus_eq_sqrt_two (z : ℂ) (h : (3 - I) / (z - 3 * I) = 1 + I) : complex.abs z = real.sqrt 2 :=
sorry

end complex_modulus_eq_sqrt_two_l580_580248


namespace irrigation_system_flow_rates_l580_580132

variable (q0 qDE qBC qGF : ℝ)

-- Conditions
axiom total_flow_rate : q0 > 0
axiom identical_channels : True -- placeholder for the identically identical channels condition
axiom constant_flow_rates : ∀ path : List String, flow_rate path = q0 -- placeholder for constant flow rates condition

-- Prove the flow rates
theorem irrigation_system_flow_rates :
  qDE = (4 / 7) * q0 ∧ qBC = (2 / 7) * q0 ∧ qGF = (3 / 7) * q0 :=
by
  sorry

end irrigation_system_flow_rates_l580_580132


namespace perpendicular_vectors_angle_between_vectors_l580_580668

-- Problem 1:
theorem perpendicular_vectors (x : ℝ) (h1 : 0 < x ∧ x < π / 2) 
  (m_perp_n : (1:ℝ) * real.sin x + (-1 : ℝ) * real.cos x = 0) : 
  x = π / 4 :=
by
  sorry

-- Problem 2:
theorem angle_between_vectors (x : ℝ) (h1 : 0 < x ∧ x < π / 2) 
  (angle_condition : (1:ℝ) * real.sin x + (-1 : ℝ) * real.cos x = real.sqrt 2 / 2):
  x = 5 * π / 12 :=
by
  sorry

end perpendicular_vectors_angle_between_vectors_l580_580668


namespace train_length_l580_580100

theorem train_length (v : ℝ) (t : ℝ) (conversion_factor : ℝ) : v = 45 → t = 16 → conversion_factor = 1000 / 3600 → (v * (conversion_factor) * t) = 200 :=
  by
  intros hv ht hcf
  rw [hv, ht, hcf]
  -- Proof steps skipped
  sorry

end train_length_l580_580100


namespace similar_triangles_height_ratio_l580_580070

theorem similar_triangles_height_ratio (area_ratio : ℝ) (h₁ : ℝ) (h₂ : ℝ) 
  (similar : Boolean) (h₁_value : h₁ = 5) (area_ratio_value : area_ratio = 9) :
  similar = true → area_ratio = (h₂ / h₁) ^ 2 → h₂ = 15 :=
by
  intro h_similar area_eq
  rw [h₁_value, area_ratio_value]
  sorry

end similar_triangles_height_ratio_l580_580070


namespace angle_x_in_triangle_l580_580891

theorem angle_x_in_triangle :
  ∀ (x : ℝ), x + 2 * x + 50 = 180 → x = 130 / 3 :=
by
  intro x h
  sorry

end angle_x_in_triangle_l580_580891


namespace sunnydale_farm_arrangements_l580_580754

theorem sunnydale_farm_arrangements :
  let chickens := 5
  let dogs := 3
  let cats := 4
  let rabbits := 3
  let groups := 4
  (Fact (Nat.factorial groups)) * (Fact (Nat.factorial chickens)) * (Fact (Nat.factorial dogs)) * (Fact (Nat.factorial cats)) * (Fact (Nat.factorial rabbits)) = 2488320 :=
by
  -- Placeholders for the proof
  sorry

end sunnydale_farm_arrangements_l580_580754


namespace result_is_0_85_l580_580008

noncomputable def calc_expression := 1.85 - 1.85 / 1.85

theorem result_is_0_85 : calc_expression = 0.85 :=
by 
  sorry

end result_is_0_85_l580_580008


namespace rotate_line_l580_580368

noncomputable def l1_equation : ℝ → ℝ → Prop := fun x y => x - y - 3 = 0

theorem rotate_line
  (x y : ℝ)
  (H_fixed_point : x = 3 ∧ y = 0)
  (H_rotation : true)  -- For simplicity, represent the rotation condition with true.
  : ∃ l2 : ℝ → ℝ → Prop, l2 = fun x y => (sqrt 3) * x - y - 3 * (sqrt 3) = 0 :=
by
  sorry

end rotate_line_l580_580368


namespace infinite_series_sum_l580_580914

theorem infinite_series_sum :
  ∑' (n : ℕ), (1 / (1 + 3^n : ℝ) - 1 / (1 + 3^(n+1) : ℝ)) = 1/2 := 
sorry

end infinite_series_sum_l580_580914


namespace neg_p_sufficient_but_not_necessary_for_q_l580_580721

variables (x : ℝ)

def p : Prop := |x| > 1
def q : Prop := x^2 + x - 6 < 0

theorem neg_p_sufficient_but_not_necessary_for_q : (¬p → q) ∧ ¬(q → ¬p) :=
by
  sorry

end neg_p_sufficient_but_not_necessary_for_q_l580_580721


namespace high_jump_statistics_l580_580515

theorem high_jump_statistics :
  let heights : Multiset ℝ := {1.50, 1.50, 1.60, 1.60, 1.60, 1.65, 1.65, 1.65, 1.65, 1.65, 1.70, 1.70, 1.70, 1.70, 1.75}
  in (Multiset.mode heights = some 1.65) ∧ (median heights = 1.65) :=
by sorry

end high_jump_statistics_l580_580515


namespace similar_triangles_height_l580_580058

theorem similar_triangles_height (h₁ h₂ : ℝ) 
  (similar : ∀ (A₁ B₁ C₁ A₂ B₂ C₂ : Triangle), 
                (∃ k, k = 3 ∧ A₁ ≈ A₂ ∧ B₁ ≈ B₂ ∧ C₁ ≈ C₂) →
                (area A₁ / area A₂ = 1 / 9)) 
  (height_smaller : h₁ = 5)
  (area_ratio : area (Triangle.mk A₁ B₁ C₁) / area (Triangle.mk A₂ B₂ C₂) = 1 / 9) :
  h₂ = 15 := 
sorry

end similar_triangles_height_l580_580058


namespace sum_of_20_consecutive_integers_l580_580822

theorem sum_of_20_consecutive_integers : 
  let a := -9 in 
  let n := 20 in 
  let last_term := a + (n - 1) in 
  let sum := n / 2 * (a + last_term) in 
  sum = 10 :=
by
  let a := -9
  let n := 20
  let last_term := a + (n - 1)
  let sum := n / 2 * (a + last_term)
  sorry

end sum_of_20_consecutive_integers_l580_580822


namespace sum_of_norms_eq_144_l580_580718

variables {ℝ : Type*} [inner_product_space ℝ ℝ]

-- Define the vectors
variables (a b m : ℝ × ℝ)

-- Define the conditions
axiom cond_m_midpoint : m = (⟨4, 5⟩ : ℝ × ℝ)
axiom cond_dot_product : inner_product_space.has_inner_product_space.bilin_form ℝ a b = 10
axiom cond_midpoint : m = (a + b) / 2

theorem sum_of_norms_eq_144 : ∥a∥ ^ 2 + ∥b∥ ^ 2 = 144 :=
by sorry

end sum_of_norms_eq_144_l580_580718


namespace no_prime_divisor_of_form_8k_minus_1_l580_580320

theorem no_prime_divisor_of_form_8k_minus_1 (n : ℕ) (h : 0 < n) :
  ¬ ∃ p k : ℕ, Nat.Prime p ∧ p = 8 * k - 1 ∧ p ∣ (2^n + 1) :=
by
  sorry

end no_prime_divisor_of_form_8k_minus_1_l580_580320


namespace distance_to_left_focus_is_eight_l580_580259

noncomputable def hyperbola_distance_proof : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ 
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) ∧ 
  (b * c / real.sqrt (a^2 + b^2) = 4) ∧ 
  (∃ x y, x^2 / a^2 - y^2 / b^2 = 1 ∧ real.sqrt ((x - c)^2 + y^2) = 2) ∧ 
  (c + a = 8)

theorem distance_to_left_focus_is_eight : hyperbola_distance_proof :=
begin
  sorry
end

end distance_to_left_focus_is_eight_l580_580259


namespace armband_cost_eq_tickets_l580_580165

theorem armband_cost_eq_tickets (n : ℕ) (h: 0.75 * n = 15) : n = 20 :=
  sorry

end armband_cost_eq_tickets_l580_580165


namespace smallest_a_l580_580957

theorem smallest_a (a : ℕ) (b c x1 x2 : ℝ) (h1 : ∀ (x: ℝ), f(x) = a * x^2 + b * x + c)
  (h2 : ∃ (x1 x2 : ℝ), 0 < x1 ∧ x1 < 1 ∧ 0 < x2 ∧ x2 < 1 ∧ x1 ≠ x2 ∧ 
                       f(x1) = 0 ∧ f(x2) = 0) :
  ∃ a = 5 := sorry

end smallest_a_l580_580957


namespace knights_prob_sum_numerator_denominator_l580_580021

theorem knights_prob_sum_numerator_denominator :
  let n := 30
  let k := 4
  let Q := 1 - (Nat.choose (n - k) k / Nat.choose n k)
  Nat.gcd (numerator Q) (denominator Q) = 1 → 
  numerator (Q) + denominator (Q) = 16 := by
  sorry

end knights_prob_sum_numerator_denominator_l580_580021


namespace distance_between_Q_and_R_l580_580488

noncomputable def distance_QR : Real :=
  let YZ := 9
  let XZ := 12
  let XY := 15
  
  -- assume QY = QX and tangent to YZ at Y, and RX = RY and tangent to XZ at X
  let QY := 12.5
  let QX := 12.5
  let RY := 12.5
  let RX := 12.5

  -- calculate and return the distance QR based on these assumptions
  (QX^2 + RY^2 - 2 * QX * RX * Real.cos 90)^(1/2)

theorem distance_between_Q_and_R (YZ XZ XY : ℝ) (QY QX RY RX : ℝ) (h1 : YZ = 9) (h2 : XZ = 12) (h3 : XY = 15)
  (h4 : QY = 12.5) (h5 : QX = 12.5) (h6 : RY = 12.5) (h7 : RX = 12.5) :
  distance_QR = 15 :=
by
  sorry

end distance_between_Q_and_R_l580_580488


namespace tom_pays_1340_l580_580029

def vaccine_cost := 45
def number_of_vaccines := 10
def doctor_visit_cost := 250
def insurance_coverage := 0.8
def trip_cost := 1200

def total_vaccine_cost := vaccine_cost * number_of_vaccines
def total_medical_cost := total_vaccine_cost + doctor_visit_cost
def insurance_cover_amount := total_medical_cost * insurance_coverage
def amount_paid_after_insurance := total_medical_cost - insurance_cover_amount
def total_amount_tom_pays := amount_paid_after_insurance + trip_cost

theorem tom_pays_1340 :
  total_amount_tom_pays = 1340 :=
by
  sorry

end tom_pays_1340_l580_580029


namespace gamma_perp_beta_necessary_not_sufficient_l580_580601

-- Definitions and conditions
variable {α β γ : Type} [InnerProductSpace ℝ α] [InnerProductSpace ℝ β] [InnerProductSpace ℝ γ]

axiom perp (α β : Type) [InnerProductSpace ℝ α] [InnerProductSpace ℝ β] : Prop
axiom parallel (α γ : Type) [InnerProductSpace ℝ α] [InnerProductSpace ℝ γ] : Prop

axiom alpha_perp_beta : perp α β

-- The statement to be proved
theorem gamma_perp_beta_necessary_not_sufficient : 
  (parallel α γ) → (perp γ β) ∧ ¬ (perp γ β ↔ parallel α γ) :=
by
  sorry

end gamma_perp_beta_necessary_not_sufficient_l580_580601


namespace similar_triangles_height_l580_580048

theorem similar_triangles_height (h₁ h₂ : ℝ) (ratio_areas : ℝ) 
  (h₁_eq : h₁ = 5) (ratio_areas_eq : ratio_areas = 1 / 9)
  (similar : h₂^2 = (√ratio_areas)^2 * h₁^2) : h₂ = 15 :=
by {
  have ratio_areas_pos : ratio_areas > 0 := by (simp [ratio_areas_eq]),
  have k := √ratio_areas,
  have k_eq : k = 3 := by {
    rw [ratio_areas_eq, sqrt_div, sqrt_one, sqrt_nat_eq_iff_eq_sq, one_div_eq_inv] at *,
    norm_num },
  have h₂_def : h₂ = 3 * h₁ := by rw [h₁_eq, mul_comm, k_eq],
  rw [h₂_def],
  norm_num,
}

end similar_triangles_height_l580_580048


namespace arithmetic_sequence_contains_9_l580_580761

noncomputable def contains_digit_9 (n : ℕ) : Prop :=
  ∃ (d : ℕ), ∃ (b : ℕ), n = d * 10^b + 9 * (10^b / 10)

theorem arithmetic_sequence_contains_9 (a d : ℕ) :
  ∃ n : ℕ, contains_digit_9 (a + n * d) :=
begin
  sorry
end

end arithmetic_sequence_contains_9_l580_580761


namespace det_of_matrix_l580_580522

def determinant_2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem det_of_matrix :
  determinant_2x2 5 (-2) 3 1 = 11 := by
  sorry

end det_of_matrix_l580_580522


namespace count_phi_equals_one_third_l580_580986

theorem count_phi_equals_one_third (k : ℕ) : 
  (λ k, (φ k) = k / 3) → (count k < 1000) = 41 := 
sorry

end count_phi_equals_one_third_l580_580986


namespace find_a_9_l580_580952

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable (d : ℤ)

-- Assumptions and definitions from the problem
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n : ℕ, a (n + 1) = a n + d
def sum_of_arithmetic_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop := ∀ n : ℕ, S n = n * (a 1 + a n) / 2
def condition_one (a : ℕ → ℤ) : Prop := (a 1) + (a 2)^2 = -3
def condition_two (S : ℕ → ℤ) : Prop := S 5 = 10

-- Main theorem statement
theorem find_a_9 (h_arithmetic : arithmetic_sequence a d)
                 (h_sum : sum_of_arithmetic_sequence S a)
                 (h_cond1 : condition_one a)
                 (h_cond2 : condition_two S) : a 9 = 20 := 
sorry

end find_a_9_l580_580952


namespace part_I_part_II_l580_580620

-- Definition of M as the solution set of the quadratic inequality
def M (a : ℝ) : Set ℝ := { x | x^2 - (a + 1) * x + a < 0 }

-- Part I: if 2 is in the solution set M(a), then a > 2
theorem part_I (a : ℝ) : 2 ∈ M(a) → a > 2 := 
sorry

-- Part II: if M(a) is empty, find the solution set of 1 / (x - a) < 2
theorem part_II (a : ℝ) : M(a) = ∅ → {x : ℝ | 1 / (x - a) < 2} = (-∞, 1) ∪ (3 / 2, ∞) :=
sorry

end part_I_part_II_l580_580620


namespace sum_coords_S_l580_580773

theorem sum_coords_S (P Q R S : ℝ × ℝ)
  (hP : P = (-3, -2))
  (hQ : Q = (1, -5))
  (hR : R = (9, 1))
  (hParallelogram : let G := (P.1 + R.1) / 2, (P.2 + R.2) / 2 in 
                    let Q_mid := (Q.1 + S.1) / 2, (Q.2 + S.2) / 2 in 
                    G = Q_mid) :
  S.1 + S.2 = 9 :=
sorry

end sum_coords_S_l580_580773


namespace combined_area_of_football_shaped_regions_l580_580705

theorem combined_area_of_football_shaped_regions :
  let side := 3
  let diagonal := side * Real.sqrt 2
  let sector_area := (Real.pi * diagonal^2) / 4
  let triangle_area := (side^2) / 2
  let region_area := sector_area - triangle_area
  2 * region_area = 9 * Real.pi - 9 :=
by
  intros
  let side := 3
  let diagonal := side * Real.sqrt 2
  let sector_area := (Real.pi * diagonal^2) / 4
  let triangle_area := (side^2) / 2
  let region_area := sector_area - triangle_area
  calc
    2 * region_area = 2 * ((Real.pi * (3 * Real.sqrt 2)^2) / 4 - (3^2) / 2) : by 
    rw [sector_area, triangle_area, diagonal]
    ... = 2 * ((Real.pi * 18) / 4 - 4.5) : by 
    norm_num
    ... = 2 * (9 * Real.pi / 2 - 4.5) : by 
    norm_num
    ... = 9 * Real.pi - 9 : by
    norm_num

end combined_area_of_football_shaped_regions_l580_580705


namespace number_of_distinct_values_l580_580339

def w : ℂ := -1/2 + (complex.sqrt 3) / 2 * complex.I

def distinct_ints (m n l : ℕ) : Prop := 
  m ≠ n ∧ n ≠ l ∧ m ≠ l

theorem number_of_distinct_values (h_dist : ∀ (m n l : ℕ), distinct_ints m n l) :
  ∃ k, ∀ m n l : ℕ, distinct_ints m n l → k = 10 := 
sorry 

end number_of_distinct_values_l580_580339


namespace interest_received_l580_580695

-- Definitions for Bank A
def principal_A : ℝ := 13000
def rate_A_increased : ℝ := 4
def time : ℕ := 4

def simple_interest (P : ℝ) (R : ℝ) (T : ℕ) : ℝ :=
  P * T * R / 100

-- Definitions for Bank B
def principal_B : ℝ := 10200
def rate_B_increased : ℝ := 6

def compound_interest (P : ℝ) (R : ℝ) (T : ℕ) : ℝ :=
  P * (1 + R / 100)^T - P

-- Lean statement to prove the interest received in each bank
theorem interest_received :
  simple_interest principal_A rate_A_increased time = 2080
  ∧ abs (compound_interest principal_B rate_B_increased time - 2677.66) < 0.01 :=
by
  sorry

end interest_received_l580_580695


namespace sufficient_condition_for_parallel_planes_l580_580707

-- Define the conditions: two planes α, β and two lines l, m
variables (α β : Plane) (l m : Line)

-- Define the relationships between the objects as the conditions spell out
axioms 
  (non_overlapping_planes : ¬(α ∩ β ≠ ∅))
  (non_overlapping_lines : ¬(l ∩ m ≠ ∅))
  (l_perpendicular_to_α : l ⊥ α)
  (m_perpendicular_to_β : m ⊥ β)
  (l_parallel_to_m : l ∥ m)

-- Prove that α is parallel to β
theorem sufficient_condition_for_parallel_planes : α ∥ β :=
sorry -- Proof is omitted

end sufficient_condition_for_parallel_planes_l580_580707


namespace f_at_2015_l580_580611

noncomputable def f : ℝ → ℝ := sorry 

axiom odd_function : ∀ x : ℝ, f(-x) = -f(x)

axiom periodic_function : ∀ x : ℝ, f(x + 4) = f(x) + f(2)

axiom f_at_1 : f(1) = 2

theorem f_at_2015 : f(2015) = -2 :=
by
  sorry

end f_at_2015_l580_580611


namespace prism_base_shape_l580_580146

theorem prism_base_shape (n : ℕ) (hn : 3 * n = 12) : n = 4 := by
  sorry

end prism_base_shape_l580_580146


namespace translation_of_B_l580_580669

theorem translation_of_B (x_A y_A x_A' y_A' x_B y_B x_B' y_B' : ℤ)
  (h1 : x_A = -1)
  (h2 : y_A = 2)
  (h3 : x_A' = 3)
  (h4 : y_A' = -4)
  (h5 : x_B = 2)
  (h6 : y_B = 4)
  (hx : x_B' = x_B + (x_A' - x_A))
  (hy : y_B' = y_B + (y_A' - y_A)) :
  (x_B', y_B') = (6, -2) :=
by
  rw [h1, h2, h3, h4, h5, h6] at hx hy
  simp at hx hy
  rw [hx, hy]
  trivial

end translation_of_B_l580_580669


namespace fill_time_l580_580459

def inflow_rate : ℕ := 24 -- gallons per second
def outflow_rate : ℕ := 4 -- gallons per second
def basin_volume : ℕ := 260 -- gallons

theorem fill_time (inflow_rate outflow_rate basin_volume : ℕ) (h₁ : inflow_rate = 24) (h₂ : outflow_rate = 4) 
  (h₃ : basin_volume = 260) : basin_volume / (inflow_rate - outflow_rate) = 13 :=
by
  sorry

end fill_time_l580_580459


namespace total_cost_tom_pays_for_trip_l580_580032

/-- Tom needs to get 10 different vaccines and a doctor's visit to go to Barbados.
    Each vaccine costs $45.
    The doctor's visit costs $250.
    Insurance will cover 80% of these medical bills.
    The trip itself costs $1200.
    Prove that the total amount Tom has to pay for his trip to Barbados, including medical expenses, is $1340. -/
theorem total_cost_tom_pays_for_trip : 
  let cost_per_vaccine := 45
  let number_of_vaccines := 10
  let cost_doctor_visit := 250
  let insurance_coverage_rate := 0.8
  let trip_cost := 1200
  let total_medical_cost := (number_of_vaccines * cost_per_vaccine) + cost_doctor_visit
  let insurance_coverage := insurance_coverage_rate * total_medical_cost
  let net_medical_cost := total_medical_cost - insurance_coverage
  let total_cost := trip_cost + net_medical_cost
  total_cost = 1340 := 
by 
  sorry

end total_cost_tom_pays_for_trip_l580_580032


namespace part_a_part_b_l580_580873

-- Prove part a)
theorem part_a : 101^7 = 107213535210701 := by
  sorry

-- Prove part b)
theorem part_b : abs (0.9998^5 - 0.9990004) < 10^(-7) := by
  sorry

end part_a_part_b_l580_580873


namespace arithmetic_sequence_general_formula_maximum_integer_t_l580_580230

theorem arithmetic_sequence_general_formula (d : ℚ) (h₁ : d > 0) (h₂ : (1 + d) * (1 + 13 * d) = (1 + 4 * d) * (1 + 4 * d)) :
  (∀ n : ℕ, n > 0 → ∃ a : ℕ, a = 2 * n - 1) :=
by sorry

theorem maximum_integer_t (d : ℚ) (h₁ : d > 0) (h₂ : (1 + d) * (1 + 13 * d) = (1 + 4 * d) * (1 + 4 * d)) :
  (∀ n : ℕ, n > 0 → let a_n := 2 * n - 1,
      b_n := (a_n + 1)^2 / (a_n * (a_n + 2)),
      S_n := (finset.range n).sum (λ i, b_n) in S_n > 1) :=
by sorry

end arithmetic_sequence_general_formula_maximum_integer_t_l580_580230


namespace triangles_in_dodecagon_l580_580537

theorem triangles_in_dodecagon (n : ℕ) (h : n = 12) : (nat.choose n 3) = 220 :=
by
  rw h
  sorry

end triangles_in_dodecagon_l580_580537


namespace mary_initial_baseball_cards_l580_580724

theorem mary_initial_baseball_cards (X : ℕ) (torn : ℕ) (from_fred : ℕ) (bought : ℕ) (total : ℕ) :
  torn = 8 ∧ from_fred = 26 ∧ bought = 40 ∧ total = 84 ∧ X - torn + from_fred + bought = total → X = 26 :=
by
  intros h
  cases h with ht1 h1
  cases h1 with hf1 h2
  cases h2 with hb1 h3
  cases h3 with ht2 h4
  sorry

end mary_initial_baseball_cards_l580_580724


namespace circle_area_eq_13_pi_l580_580089

theorem circle_area_eq_13_pi (x y : ℝ) :
    (x^2 + y^2 - 4 = 6y - 4x + 4) → (π * 13 = 13 * π) :=
by
  intro h
  sorry

end circle_area_eq_13_pi_l580_580089


namespace no_real_solutions_l580_580987

theorem no_real_solutions : ¬ ∃ x : ℝ, sqrt ((x^2 - 2 * x + 1) + 1) = -x :=
by sorry

end no_real_solutions_l580_580987


namespace equal_work_women_l580_580117

-- Let W be the amount of work one woman can do in a day.
-- Let M be the amount of work one man can do in a day.
-- Let x be the number of women who do the same amount of work as 5 men.

def numWomenEqualWork (W : ℝ) (M : ℝ) (x : ℝ) : Prop :=
  5 * M = x * W

theorem equal_work_women (W M x : ℝ) 
  (h1 : numWomenEqualWork W M x)
  (h2 : (3 * M + 5 * W) * 10 = (7 * W) * 14) :
  x = 8 :=
sorry

end equal_work_women_l580_580117


namespace unique_sum_of_three_diff_positive_perfect_squares_l580_580295

theorem unique_sum_of_three_diff_positive_perfect_squares :
  (∃ (a b c : ℕ), (a^2 + b^2 + c^2 = 100) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c)) →
  (∀ (x y z : ℕ), (x^2 + y^2 + z^2 = 100) ∧ (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z) → {a, b, c} = {x, y, z}) :=
begin
  sorry
end

end unique_sum_of_three_diff_positive_perfect_squares_l580_580295


namespace solve_system_l580_580109

-- Define the variables 'x', 'y', 'z', and 'a' as real numbers
variables {x y z a : ℝ}

-- Define the conditions for the system of equations
def condition1 : Prop := x + y + z = a
def condition2 : Prop := x^2 + y^2 + z^2 = a^2
def condition3 : Prop := x^3 + y^3 + z^3 = a^3

-- The theorem we need to state
theorem solve_system (h1 : condition1) (h2 : condition2) (h3 : condition3) : {x, y, z} = {a, 0, 0} :=
sorry 

end solve_system_l580_580109


namespace secret_code_l580_580824

def Clue1 (code : List ℕ) : Prop :=
  (code[0] = 0 ∧ code[1] ≠ 7 ∧ code[2] ≠ 9) ∨
  (code[0] ≠ 0 ∧ code[1] = 7 ∧ code[2] ≠ 9) ∨
  (code[0] ≠ 0 ∧ code[1] ≠ 7 ∧ code[2] = 9)

def Clue2 (code : List ℕ) : Prop :=
  code[0] ≠ 0 ∧ code[1] ≠ 3 ∧ code[2] ≠ 2

def Clue3 (code : List ℕ) : Prop :=
  (code[0] ≠ 1 ∧ code[1] = 0 ∧ code[2] ≠ 8) ∨
  (code[0] ≠ 1 ∧ code[1] ≠ 0 ∧ code[2] = 8) ∨
  (code[0] = 1 ∧ code[1] ≠ 0 ∧ code[2] ≠ 8)

def Clue4 (code : List ℕ) : Prop :=
  (code[0] ≠ 9 ∧ code[1] = 2 ∧ code[2] ≠ 6) ∨
  (code[0] ≠ 9 ∧ code[1] ≠ 2 ∧ code[2] = 6) ∨
  (code[0] = 9 ∧ code[1] ≠ 2 ∧ code[2] ≠ 6)

def Clue5 (code : List ℕ) : Prop :=
  (code[0] ≠ 6 ∧ code[1] = 7 ∧ code[2] ≠ 8) ∨
  (code[0] ≠ 6 ∧ code[1] ≠ 7 ∧ code[2] = 8) ∨
  (code[0] = 6 ∧ code[1] ≠ 7 ∧ code[2] ≠ 8)

theorem secret_code (code : List ℕ) (h1 : Clue1 code) (h2 : Clue2 code) (h3 : Clue3 code) (h4 : Clue4 code) (h5 : Clue5 code) :
  code = [8, 1, 9] := 
sorry

end secret_code_l580_580824


namespace fraction_of_25_l580_580465

theorem fraction_of_25 (x : ℝ) (h1 : 0.65 * 40 = 26) (h2 : 26 = x * 25 + 6) : x = 4 / 5 :=
sorry

end fraction_of_25_l580_580465


namespace other_car_capacity_l580_580382

/--
We are given that:
    (1) The Rocket Coaster has 15 cars.
    (2) Some cars hold 4 people.
    (3) Total capacity of the Rocket Coaster is 72 people.
    (4) There are 9 four-passenger cars.
We are to prove that the number of people the other type of car can hold is 6.
-/
theorem other_car_capacity :
  ∀ (total_cars four_passenger_cars total_capacity passenger_per_four_car total_other_cars total_other_capacity people_per_other_car: ℕ),
  total_cars = 15 →
  four_passenger_cars = 9 →
  total_capacity = 72 →
  passenger_per_four_car = 4 →
  total_other_cars = total_cars - four_passenger_cars →
  total_other_capacity = total_capacity - (four_passenger_cars * passenger_per_four_car) →
  people_per_other_car = total_other_capacity / total_other_cars →
  people_per_other_car = 6 :=
begin
  assume _ _ _ _ _ _ _ h_total_cars h_four_passenger_cars h_total_capacity h_passenger_per_four_car h_total_other_cars h_total_other_capacity h_people_per_other_car,
  sorry
end

end other_car_capacity_l580_580382


namespace tom_total_cost_l580_580027

theorem tom_total_cost :
  let vaccines_cost := 10 * 45 in
  let total_medical_cost := vaccines_cost + 250 in
  let insurance_covered := 0.80 * total_medical_cost in
  let tom_pay_medical := total_medical_cost - insurance_covered in
  let trip_cost := 1200 in
  let total_cost := tom_pay_medical + trip_cost in
  total_cost = 1340 :=
by
  dsimp
  sorry

end tom_total_cost_l580_580027


namespace tangent_circles_m_eq_nine_l580_580995

theorem tangent_circles_m_eq_nine (m : ℝ) :
  ∀ (C1 : ∀ x y : ℝ, x^2 + y^2 = 1)
    (C2 : ∀ x y : ℝ, x^2 + y^2 - 6 * x - 8 * y + m = 0),
  (∃ x1 y1 x2 y2 : ℝ, (C1 x1 y1 ∧ C2 x2 y2 ∧ ((x2 - x1)^2 + (y2 - y1)^2 = (sqrt (25 - m) + 1)^2)
     ∧ ((x1, y1) = (0, 0)) ∧ ((x2, y2) = (3, 4)))) → m = 9 :=
by
  intros C1 C2 h
  sorry

end tangent_circles_m_eq_nine_l580_580995


namespace dk_bisects_bc_l580_580942

variables {A B C D K M : Type} [metric_space K] [euclidean_geometry K]

-- Given a triangle ABC
variables (A B C : K)
variable [Is_Triangle A B C]

-- The tangent at point C to the circumcircle of triangle ABC intersects line AB at point D
variables (D : K)
variable [On_Tangent_Circumcircle A B C D]

-- The tangents to the circumcircle of triangle ACD at points A and C intersect at point K
variables (K : K)
variable [Tangents_Intersect A C D K]

-- Point M is the midpoint of BC
variables (M : K)
variable [Midpoint M B C]

-- Prove that line DK bisects segment BC
theorem dk_bisects_bc : Midpoint M B C → Bisects D K M := by
  sorry

end dk_bisects_bc_l580_580942


namespace bananas_to_oranges_equivalence_l580_580380

namespace EquivalentProof

-- Assume the conditions in the problem
def bananas := ℕ
def oranges := ℕ

axiom equal_value_1 (b_to_o : bananas → oranges): 
  b_to_o (3 * 16 / 4) = 12

-- Define the target calculation
def bananas_equivalent_oranges (b_to_o : bananas → oranges) :=
  b_to_o (3 * 20 / 5) = 12

-- Generate the Lean statement for the proof problem
theorem bananas_to_oranges_equivalence :
  ∀ (b_to_o : bananas → oranges),
    equal_value_1 b_to_o → bananas_equivalent_oranges b_to_o :=
by
  intros b_to_o h
  sorry

end EquivalentProof

end bananas_to_oranges_equivalence_l580_580380


namespace sum_of_numbers_in_104th_bracket_l580_580370

/-- The sequence of numbers {2n+1} is divided into brackets where:
- The 1st bracket contains 1 number.
- The 2nd bracket contains 2 numbers.
- The 3rd bracket contains 3 numbers.
- The 4th bracket contains 4 numbers.
- The 5th bracket restarts with 1 number, and so on.
Given this sequence, prove that the sum of numbers in the 104th bracket equals 816. -/
theorem sum_of_numbers_in_104th_bracket : 
  let seq := λ n : ℕ, 2 * n + 1
  let bracket := λ m : ℕ, (1 + (m % 4), seq m)
  let sum_of_bracket := λ br : List ℕ, br.sum
  let sum_104th := sum_of_bracket [201, 203, 205, 207]
  in sum_104th = 816 :=
by
  sorry

end sum_of_numbers_in_104th_bracket_l580_580370


namespace probability_of_ace_then_spade_l580_580422

theorem probability_of_ace_then_spade :
  let P := (1 / 52) * (12 / 51) + (3 / 52) * (13 / 51)
  P = (3 / 127) :=
by
  sorry

end probability_of_ace_then_spade_l580_580422


namespace power_of_three_l580_580277

theorem power_of_three (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_mult : (3^a) * (3^b) = 81) : (3^a)^b = 81 :=
sorry

end power_of_three_l580_580277


namespace total_bulbs_produced_l580_580128

theorem total_bulbs_produced (X : ℕ) (H1 : 1 = 1) (H2 : 10 = 10) (H87 : 87 = 87) :
  (1 / 10 : ℚ) = (87 / X : ℚ) → X = 870 := 
by
  intro h
  field_simp at h
  linarith

end total_bulbs_produced_l580_580128


namespace ratio_area_rectangle_triangle_l580_580823

-- Define the lengths L and W as positive real numbers
variables {L W : ℝ} (hL : L > 0) (hW : W > 0)

-- Define the area of the rectangle
noncomputable def area_rectangle (L W : ℝ) : ℝ := L * W

-- Define the area of the triangle with base L and height W
noncomputable def area_triangle (L W : ℝ) : ℝ := (1 / 2) * L * W

-- Define the ratio between the area of the rectangle and the area of the triangle
noncomputable def area_ratio (L W : ℝ) : ℝ := area_rectangle L W / area_triangle L W

-- Prove that this ratio is equal to 2
theorem ratio_area_rectangle_triangle : area_ratio L W = 2 := by sorry

end ratio_area_rectangle_triangle_l580_580823


namespace intersection_point_independence_l580_580792

variable {A B C P Q R : Point}
variable (Gamma : Circle)
variable [OnLine : OnLine A B C]
variable [Tangents : Tangents A C P]
variable [Intersects : IntersectsGammaPB Gamma P B Q]
variable [Bisector : AngleBisector A Q C R]

theorem intersection_point_independence :
  (∀ Gamma : Circle, PassesThrough Gamma A C ∧ CenterNotOnLine Gamma A C ∧
   TangentsTo Gamma A C P ∧ IntersectsGammaPB  Gamma P B Q →
  Intersection (BisectorOf A Q C) (LineOf A C) = R)

-- Definitions required by above variables and theorem
def Point := ℝ × ℝ -- A point in 2D plane
def Circle := Point × ℝ -- A circle is defined by its center and radius
def Line := {a: Point // ∃ b: Point, b ≠ a} -- A line through 2 points

class OnLine (A B C : Point) : Prop := 
  (on_line : ∃ l : Line, ∀ p ∈ {A, B, C}, ∃ t : ℝ, p = l.1 + t * l.2)

class PassesThrough (Gamma : Circle) (A C : Point) : Prop :=
  (passes_through_a : (A.1 - Gamma.1.1)^2 + (A.2 - Gamma.1.2)^2 = Gamma.2^2)
  (passes_through_c : (C.1 - Gamma.1.1)^2 + (C.2 - Gamma.1.2)^2 = Gamma.2^2)

class CenterNotOnLine (Gamma : Circle) (A C : Point) : Prop :=
  (center_not_on_line : ∃ l : Line, ∃ t : ℝ, Gamma.1 ≠ l.1 + t * l.2)

class Tangents (A C P : Point) : Prop :=
  (tangent_a : ((A - P)•(C - P)) =  0)
  (tangent_c : ((C - P)•(A - P)) =  0)

class IntersectsGammaPB (Gamma : Circle) (P B : Point) (Q : Point) : Prop :=
  (intersection : P.1 = B.1 ∧ Q.1 = B.1 ∧ (Q.2 = (sqrt(Gamma.1.2^2 - (Q.1 - Gamma.1.1)^2))))

class AngleBisector (A Q C R : Point) : Prop :=
  (angle_bisector : (A - Q)•(R - Q) = (C - Q)•(R - Q))

def LineOf (A C : Point) := {p : Point // ∃ t : ℝ, p = A + t * (C - A)}

def BisectorOf (A Q C : Point) := {p : Point // ∃ t : ℝ, p = A + t * (C - A)}

def Intersection (L1 L2 : Line) : Point := sorry -- intersection logic of two lines

#check intersection_point_independence

end intersection_point_independence_l580_580792


namespace determine_counterfeit_weight_l580_580734

-- Definitions based on problem conditions:
def num_coins : ℕ := 239
def num_genuine : ℕ := 237
def num_counterfeit : ℕ := 2

-- Groups of coins:
def A : fin 60 := sorry
def B : fin 60 := sorry
def C : fin 60 := sorry
def D : fin 59 := sorry

-- The balance scale function, which compares two groups:
def balance_scale (group1 group2 : fin 60) : Ordering := sorry

-- Main theorem statement to determine if counterfeit coins are heavier or lighter:
theorem determine_counterfeit_weight :
    (balance_scale A B = Ordering.eq → 
    (balance_scale B C = Ordering.eq → 
    (balance_scale A D ≠ Ordering.eq))) ∨
    (balance_scale A B ≠ Ordering.eq ∨ 
    (balance_scale B C ≠ Ordering.eq ∧ 
    (balance_scale A D = balance_scale B C))) :=
sorry

end determine_counterfeit_weight_l580_580734


namespace investment_value_after_two_weeks_l580_580360

theorem investment_value_after_two_weeks (initial_investment : ℝ) (gain_first_week : ℝ) (gain_second_week : ℝ) : 
  initial_investment = 400 → 
  gain_first_week = 0.25 → 
  gain_second_week = 0.5 → 
  ((initial_investment * (1 + gain_first_week) * (1 + gain_second_week)) = 750) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  calc
    400 * (1 + 0.25) * (1 + 0.5)
    = 400 * 1.25 * 1.5 : by ring
    = 750 : by norm_num

end investment_value_after_two_weeks_l580_580360


namespace box_height_with_spheres_l580_580119

theorem box_height_with_spheres :
  ∃ h : ℝ, (0 < h) ∧ 
  (∃ large_sphere_radius small_sphere_radius : ℝ, 
     large_sphere_radius = 3 ∧ 
     small_sphere_radius = 1 ∧
     ∃ small_sphere_positions : list (ℝ × ℝ × ℝ), 
       (∀ pos ∈ small_sphere_positions, 
          let (x, y, z) := pos in 
          (0 ≤ x) ∧ (x ≤ 6) ∧
          (0 ≤ y) ∧ (y ≤ 6) ∧
          (0 ≤ z) ∧ (z ≤ h) ∧ 
          (x = 1 ∨ x = 5 ∨ y = 1 ∨ y = 5 ∨ z = 1 ∨ z = h-1)) ∧
       (∃ large_sphere_position : (ℝ × ℝ × ℝ), 
          let (lx, ly, lz) := large_sphere_position in
          (3 ≤ lx) ∧ (lx ≤ 3) ∧
          (3 ≤ ly) ∧ (ly ≤ 3) ∧
          (3 ≤ lz) ∧ (lz ≤ h-3) ∧
          (∀ pos ∈ small_sphere_positions, 
             let (x, y, z) := pos in
             ((lx - x) ^ 2 + (ly - y) ^ 2 + (lz - z) ^ 2 = (large_sphere_radius + small_sphere_radius) ^ 2)))) ∧ h = 8 :=
begin
  sorry
end

end box_height_with_spheres_l580_580119


namespace integer_roots_abs_sum_l580_580572

theorem integer_roots_abs_sum (p q r n : ℤ) :
  (∃ n : ℤ, (∀ x : ℤ, x^3 - 2023 * x + n = 0) ∧ p + q + r = 0 ∧ p * q + q * r + r * p = -2023) →
  |p| + |q| + |r| = 102 :=
by
  sorry

end integer_roots_abs_sum_l580_580572


namespace circle_and_line_equation_find_m_value_l580_580305

-- Definitions based on conditions
def parametric_circle_eq (t : ℝ) : ℝ × ℝ :=
  (1 + 3 * Real.cos t, -2 + 3 * Real.sin t)

def polar_line_eq (r θ : ℝ) (m : ℝ) : Prop :=
  r * Real.sin (θ - π/4) * √2 = m

-- Theorem to prove general form equation of circle C and rectangular coordinate equation of line l
theorem circle_and_line_equation (m : ℝ) (t : ℝ) :
  let (x, y) := parametric_circle_eq t in
  (x - 1)^2 + (y + 2)^2 = 9 ∧
  (polar_line_eq (Real.sqrt (x^2 + y^2)) (Real.atan2 y x) m) ∧
  (x - y + m = 0) :=
  sorry

-- Theorem to prove the value of m given the distance condition
theorem find_m_value (m : ℝ) :
  ∃ m : ℝ, (x - 1)^2 + (y + 2)^2 = 9 ∧ (x - y + m = 0) ∧
  (Real.abs (1 - (-2) + m) / √2 = 2) :=
  sorry

end circle_and_line_equation_find_m_value_l580_580305


namespace bushes_needed_for_48_zucchinis_l580_580551

-- Definition of conditions based on the problem
def containers_per_bush := 8
def zucchinis_per_five_containers := 2

-- Main proof problem: Prove that 15 bushes are needed for 48 zucchinis
theorem bushes_needed_for_48_zucchinis : 
  let num_zucchinis := 48 in
  let num_bushes_needed := 15 in
  (num_zucchinis * 5) / (2 * containers_per_bush) = num_bushes_needed := by
  sorry

end bushes_needed_for_48_zucchinis_l580_580551


namespace percentage_markup_correct_l580_580397

noncomputable def selling_price : ℝ := 8337
noncomputable def cost_price : ℝ := 6947.5
noncomputable def markup : ℝ := selling_price - cost_price
noncomputable def percentage_markup : ℝ := (markup / cost_price) * 100

theorem percentage_markup_correct : 
  percentage_markup ≈ 19.99 := by
  sorry

end percentage_markup_correct_l580_580397


namespace distinct_non_perfect_square_l580_580713

theorem distinct_non_perfect_square (d : ℕ) (h : 0 < d ∧ d ≠ 2 ∧ d ≠ 5 ∧ d ≠ 13) :
  ∃ a b ∈ ({2, 5, 13, d} : set ℕ), a ≠ b ∧ ¬ ∃ k : ℕ, (a * b) - 1 = k ^ 2 := 
sorry

end distinct_non_perfect_square_l580_580713


namespace exp_inequality_log_inequality_l580_580609

theorem exp_inequality_log_inequality (a b : ℝ) : 
  (2 ^ a > 2 ^ b) ↔ (a > b ∧ a > 0 ∧ b > 0) :=
by
  sorry

end exp_inequality_log_inequality_l580_580609


namespace wrong_mark_is_43_l580_580147

theorem wrong_mark_is_43
  (correct_mark : ℕ)
  (wrong_mark : ℕ)
  (num_students : ℕ)
  (avg_increase : ℕ)
  (h_correct : correct_mark = 63)
  (h_num_students : num_students = 40)
  (h_avg_increase : avg_increase = 40 / 2) 
  (h_wrong_avg : (num_students - 1) * (correct_mark + avg_increase) / num_students = (num_students - 1) * (wrong_mark + avg_increase + correct_mark) / num_students) :
  wrong_mark = 43 :=
sorry

end wrong_mark_is_43_l580_580147


namespace curve_not_parabola_l580_580921

theorem curve_not_parabola (k : ℝ) : ¬(∃ (a b c d e f : ℝ), k * x^2 + y^2 = a * x^2 + b * x * y + c * y^2 + d * x + e * y + f ∧ b^2 = 4*a*c ∧ (a = 0 ∨ c = 0)) := sorry

end curve_not_parabola_l580_580921


namespace min_handshakes_l580_580462

theorem min_handshakes (n m : ℕ) (h1 : n = 30) (h2 : m ≥ 3) :
  let total_handshakes := (n * m) / 2
  in total_handshakes = 45 :=
by
  simp [h1, h2]
  sorry

end min_handshakes_l580_580462


namespace integral_value_l580_580194

theorem integral_value :
  ∫ x in -1..0, (x - Real.exp x) = 1 / Real.exp 1 - 3 / 2 := by
  sorry

end integral_value_l580_580194


namespace james_time_to_run_100_meters_l580_580691

theorem james_time_to_run_100_meters
  (john_time_to_run_100_meters : ℕ → Prop)
  (john_first_4_meters : nat := 4)
  (john_total_time : nat := 13)
  (james_first_10_meters_time : nat := 2)
  (james_top_speed_faster_by : nat := 2):
  john_time_to_run_100_meters john_total_time → 
  (john_first_4_meters = 4) →
  ∀ n, (john_total_time - 1) = n / 8 →
  ∀ m, (100 - 10) = m / 10 →
  ∀ p, james_first_10_meters_time * 1 + p * 10 = 100 →
  (james_first_10_meters_time + p) = 11 :=
  sorry

end james_time_to_run_100_meters_l580_580691


namespace arrange_in_circle_l580_580213

open Nat

noncomputable def smallest_n := 70

theorem arrange_in_circle (n : ℕ) (h : n = 70) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ n →
    (∀ j : ℕ, 1 ≤ j ∧ j ≤ 40 → k > ((k + j) % n)) ∨
    (∀ p : ℕ, 1 ≤ p ∧ p ≤ 30 → k < ((k + p) % n))) :=
by
  sorry

end arrange_in_circle_l580_580213


namespace max_popsicles_l580_580346

theorem max_popsicles (total_money : ℝ) (cost_per_popsicle : ℝ) (h_money : total_money = 19.23) (h_cost : cost_per_popsicle = 1.60) : 
  ∃ (x : ℕ), x = ⌊total_money / cost_per_popsicle⌋ ∧ x = 12 :=
by
    sorry

end max_popsicles_l580_580346


namespace count_lines_intersecting_circle_l580_580210

theorem count_lines_intersecting_circle : 
  (∑ a in Finset.range 21, (a - 1)) = 190 := by
  sorry

end count_lines_intersecting_circle_l580_580210


namespace similar_triangles_height_l580_580044

open_locale classical

theorem similar_triangles_height
  (h_small : ℝ)
  (A_small A_large : ℝ)
  (area_ratio : ℝ)
  (h_large : ℝ) :
  A_small / A_large = 1 / 9 →
  h_small = 5 →
  area_ratio = A_large / A_small →
  area_ratio = 9 →
  h_large = h_small * sqrt area_ratio →
  h_large = 15 :=
by
  sorry

end similar_triangles_height_l580_580044


namespace triangles_in_dodecagon_l580_580538

theorem triangles_in_dodecagon : ∃ (n : ℕ), n = 12 → ∃ (k : ℕ), k = 3 → choose 12 3 = 220 :=
by
  existsi 12
  intro n12
  existsi 3
  intro k3
  rw [n12, k3]
  sorry

end triangles_in_dodecagon_l580_580538


namespace IH_perp_AD_l580_580512

open EuclideanGeometry

variables {A B C O M P Q H I N D : Point}
variables [Tangent PA PB QA QC : Circle O]
variables [Circumcircle ABC : Circle O]

-- Conditions
axiom ABC_circumcircle : Triangle A B C
axiom AB_less_AC : AB < AC
axiom angle_BAC_eq_120 : ∠ BAC = 120
axiom M_midpoint_arc_BAC : is_midpoint M (arc BAC)
axiom P_and_Q_tangents : Tangent PA PB ∧ Tangent QA QC
axiom H_orthocenter_POQ : is_orthocenter H (Triangle P O Q)
axiom I_incenter_POQ : is_incenter I (Triangle P O Q)
axiom N_midpoint_OI : midpoint N O I
axiom D_second_intersection_MN_O : second_intersection D (Line M N) (Circle O)

-- Proof Statement
theorem IH_perp_AD : Perpendicular IH AD :=
sorry

end IH_perp_AD_l580_580512


namespace spend_on_children_education_l580_580847

-- Conditions translated to Lean 4
def total_income : ℝ := 100 -- Assume total income is 100 units.

def food_expense (income : ℝ) : ℝ := 0.5 * income
def remaining_income_after_food (income : ℝ) : ℝ := income - food_expense(income)

def house_rent_expense (income : ℝ) : ℝ := 0.5 * remaining_income_after_food(income)
def remaining_income_after_house_rent (income : ℝ) : ℝ := remaining_income_after_food(income) - house_rent_expense(income)

def left_income_percentage : ℝ := 0.175 * total_income

-- The percentage spent on children's education
def percent_spent_on_children_education (income : ℝ) : ℝ :=
  1 - (food_expense(income) / income) - (left_income_percentage / income)

-- The theorem we want to prove
theorem spend_on_children_education : percent_spent_on_children_education total_income = 0.325 :=
by sorry

end spend_on_children_education_l580_580847


namespace smallest_integer_with_14_divisors_l580_580094

theorem smallest_integer_with_14_divisors : ∃ n : ℕ, (n > 0) ∧ number_of_divisors n = 14 ∧ ∀ m : ℕ, (m > 0 ∧ number_of_divisors m = 14) → n ≤ m := 
sorry

end smallest_integer_with_14_divisors_l580_580094


namespace part_I_part_II_l580_580968

noncomputable def f (x : ℝ) : ℝ := 5 + Real.log x
noncomputable def g (x k : ℝ) : ℝ := k * x / (x + 1)

theorem part_I (k : ℝ) : 
  (∃ x0, g x0 k = x0 + 4 ∧ (k / (x0 + 1)^2) = 1) ↔ (k = 1 ∨ k = 9) :=
by
  sorry

theorem part_II (k : ℕ) : (∀ x : ℝ, 1 < x → f x > g x k) → k ≤ 7 :=
by
  sorry

end part_I_part_II_l580_580968


namespace probability_first_ace_second_spade_l580_580418

theorem probability_first_ace_second_spade :
  let deck := List.range 52 in
  let first_is_ace (card : ℕ) := card % 13 = 0 in
  let second_is_spade (card : ℕ) := card / 13 = 3 in
  let events :=
    [ ((first_is_ace card, second_is_spade card') | card ∈ deck, card' ∈ List.erase deck card) ] in
  let favorable_events :=
    [(true, true)] in
  (List.count (λ event => event ∈ favorable_events) events).toRat /
  (List.length events).toRat = 1 / 52 :=
sorry

end probability_first_ace_second_spade_l580_580418


namespace range_of_x_l580_580589

theorem range_of_x (x : ℝ) (h1 : (x + 2) * (x - 3) ≤ 0) (h2 : |x + 1| ≥ 2) : 
  1 ≤ x ∧ x ≤ 3 :=
sorry

end range_of_x_l580_580589


namespace tan_identity_of_alpha_beta_l580_580340

theorem tan_identity_of_alpha_beta (α β : ℝ) (hα : 0 < α ∧ α < (π / 2)) (hβ : 0 < β ∧ β < (π / 2)) 
  (h: tan α - tan β = 1 / cos β) : 2 * α - β = π / 2 := 
sorry

end tan_identity_of_alpha_beta_l580_580340


namespace function_linear_and_decreasing_l580_580632

theorem function_linear_and_decreasing (k b : ℝ) :
  (∀ x : ℝ, y = (k - 1) * x^(k^2 - 3) + b + 1) → 
  (∀ x1 x2 : ℝ, x1 < x2 → y x1 > y x2) → 
  k = -2 ∧ ∃ (b : ℝ), True :=
by
  sorry

end function_linear_and_decreasing_l580_580632


namespace transformation_property_l580_580004

theorem transformation_property 
(Q : ℝ × ℝ) (c d : ℝ) (h1 : Q = (c, d)) 
(h2 : rotate (2,3) (π / 2) Q = Q') 
(h3 : reflect_line y_eq_x Q' = (-3, 8)) 
: d - c = 1 := 
sorry

end transformation_property_l580_580004


namespace maximum_value_negative_domain_l580_580617

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 - x else -(x^2 + x)

theorem maximum_value_negative_domain :
  (is_odd_function f) →
  (∀ x > 0, f x = x^2 - x) →
  ∃ x < 0, ∀ y < 0, f y ≤ f x ∧ f x = 1 / 4 :=
by
  intros oddf posf
  use (-1) / 2
  sorry

end maximum_value_negative_domain_l580_580617


namespace bryce_raisins_l580_580870

theorem bryce_raisins:
  ∃ x : ℕ, (x - 8 = x / 3) ∧ x = 12 :=
by 
  sorry

end bryce_raisins_l580_580870


namespace number_of_girls_l580_580855

variable (N n g : ℕ)
variable (h1 : N = 1600)
variable (h2 : n = 200)
variable (h3 : g = 95)

theorem number_of_girls (G : ℕ) (h : g * N = G * n) : G = 760 :=
by sorry

end number_of_girls_l580_580855


namespace arith_seq_ninth_term_value_l580_580954

variable {a : Nat -> ℤ}
variable {S : Nat -> ℤ}

def arith_seq (a : Nat -> ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + a 1^2

def arith_sum (S : Nat -> ℤ) (a : Nat -> ℤ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

theorem arith_seq_ninth_term_value
  (h_seq : arith_seq a)
  (h_sum : arith_sum S a)
  (h_cond1 : a 1 + a 2^2 = -3)
  (h_cond2 : S 5 = 10) :
  a 9 = 20 :=
by
  sorry

end arith_seq_ninth_term_value_l580_580954


namespace geese_more_than_ducks_l580_580166

theorem geese_more_than_ducks (initial_ducks: ℕ) (initial_geese: ℕ) (initial_swans: ℕ) (additional_ducks: ℕ)
  (additional_geese: ℕ) (leaving_swans: ℕ) (leaving_geese: ℕ) (returning_geese: ℕ) (returning_swans: ℕ)
  (final_leaving_ducks: ℕ) (final_leaving_swans: ℕ)
  (initial_ducks_eq: initial_ducks = 25)
  (initial_geese_eq: initial_geese = 2 * initial_ducks - 10)
  (initial_swans_eq: initial_swans = 3 * initial_ducks + 8)
  (additional_ducks_eq: additional_ducks = 4)
  (additional_geese_eq: additional_geese = 7)
  (leaving_swans_eq: leaving_swans = 9)
  (leaving_geese_eq: leaving_geese = 5)
  (returning_geese_eq: returning_geese = 15)
  (returning_swans_eq: returning_swans = 11)
  (final_leaving_ducks_eq: final_leaving_ducks = 2 * (initial_ducks + additional_ducks))
  (final_leaving_swans_eq: final_leaving_swans = (initial_swans + returning_swans) / 2):
  (initial_geese + additional_geese + returning_geese - leaving_geese - final_leaving_geese + returning_geese) -
  (initial_ducks + additional_ducks - final_leaving_ducks) = 57 :=
by
  sorry

end geese_more_than_ducks_l580_580166


namespace problem_l580_580597

noncomputable def a (b : ℝ) (a : ℝ) : Prop := b = 0 ∧ a = -1

theorem problem (a b : ℝ) (h1 : {a, b / a, 1} = {a^2, a + b, 0}) (h2 : a ≠ 0) :
  a^{2013} + b^{2013} = -1 :=
by
  have h3 : b = 0,
    sorry,
  have h4 : a = -1,
    sorry,
  calc
    a^{2013} + b^{2013} = (-1)^{2013} + 0^{2013} : by rw [h3, h4]
    ... = -1 : by norm_num

end problem_l580_580597


namespace sqrt_range_l580_580650

theorem sqrt_range (x : ℝ) : 3 - 2 * x ≥ 0 ↔ x ≤ 3 / 2 := 
    sorry

end sqrt_range_l580_580650


namespace simple_ordered_pairs_count_l580_580275

def is_simple_ordered_pair (m n : ℕ) : Prop :=
∀ (d : ℕ), (m % 10^d + n % 10^d) < 10 → m + n = 1942

theorem simple_ordered_pairs_count :
  {p : ℕ × ℕ // is_simple_ordered_pair p.1 p.2}.card = 300 :=
sorry

end simple_ordered_pairs_count_l580_580275


namespace complement_set_l580_580283

open Set

variable {U : Set ℝ}
variable {A : Set ℝ}

theorem complement_set :
  U = univ → A = { x | x ≤ 0 ∨ x ≥ 1 } → complement U A = (Ioo 0 1) :=
by
  intros hU hA
  rw [hU, univ, hA]
  sorry

end complement_set_l580_580283


namespace grain_equiv_system_l580_580299

-- Define x and y as the amount of sheng produced by high-quality and low-quality grain bundles respectively
variables (x y : ℝ)

-- Defining the conditions as per the problem statement:
def condition1 : Prop := 5 * x - 11 = 7 * y
def condition2 : Prop := 7 * x - 25 = 5 * y

-- The main theorem stating that given condition1 and condition2, the equations 5x-11=7y and 7x-25=5y hold.
theorem grain_equiv_system : condition1 ∧ condition2 :=
by
  split
  { exact sorry },
  { exact sorry }

end grain_equiv_system_l580_580299


namespace perimeter_shaded_region_l580_580302

-- Define the setup of the problem
def circle_center := ℝ
def radius := 6
def angle_subtended := (3 / 4) * (2 * Real.pi)  -- 270 degrees = 3/4 of 360 degrees

-- State the problem of finding the perimeter of the shaded region
theorem perimeter_shaded_region (O : circle_center) (OP OQ : ℝ) (arcPQ : ℝ) 
  (h1 : OP = radius) (h2 : OQ = radius) (h3 : arcPQ = angle_subtended * radius) : 
  OP + OQ + arcPQ = 12 + 9 * Real.pi :=
  sorry

end perimeter_shaded_region_l580_580302


namespace spacy_subsets_of_15_l580_580175

def is_spacy (s : set ℕ) : Prop :=
  ∀ (a b c d : ℕ), a ∈ s → b ∈ s → c ∈ s → d ∈ s → ¬ (b = a + 1 ∧ c = a + 2 ∧ d = a + 3)

def spacy_subsets_count (n : ℕ) : ℕ :=
  if h : n < 1 then 2
  else if h : n < 2 then 3
  else if h : n < 3 then 4
  else if h : n < 4 then 5
  else spacy_subsets_count (n - 1) + spacy_subsets_count (n - 4)

theorem spacy_subsets_of_15 : spacy_subsets_count 15 = 181 :=
  sorry

end spacy_subsets_of_15_l580_580175


namespace total_distance_proof_l580_580316

/-- Define the variables for the number of uphill and downhill jumps -/
variables (u d : ℕ)

/-- Define the total number of jumps -/
def total_jumps : ℕ := 2024

/-- First condition: total jumps equation -/
def condition1 : Prop := u + d = total_jumps

/-- Second condition: relation between uphill and downhill jumps -/
def condition2 : Prop := u = 3 * d

/-- The total distance Kenny jumps -/
def total_distance : ℕ := u * 1 + d * 3

/-- The final proof problem statement -/
theorem total_distance_proof (u d : ℕ) (h1 : condition1 u d) (h2 : condition2 u d) :
  total_distance u d = 3036 := 
sorry

end total_distance_proof_l580_580316


namespace similar_triangles_height_l580_580077

theorem similar_triangles_height
  (a b : ℕ)
  (area_ratio: ℕ)
  (height_smaller : ℕ)
  (height_relation: height_smaller = 5)
  (area_relation: area_ratio = 9)
  (similarity: a / b = 1 / area_ratio):
  (∃ height_larger : ℕ, height_larger = 15) :=
by
  sorry

end similar_triangles_height_l580_580077


namespace joan_paid_230_l580_580106

theorem joan_paid_230 (J K : ℝ) (h1 : J + K = 600) (h2 : 2 * J = K + 90) : J = 230 := 
by 
  sorry

end joan_paid_230_l580_580106


namespace intersection_points_and_final_value_l580_580174

def f (x : ℝ) : ℝ := x^2 - 4*x + 3
def g (x : ℝ) : ℝ := - f x
def h (x : ℝ) : ℝ := f (- x)

theorem intersection_points_and_final_value :
  let a := (2 : ℕ) in   -- points where f(x) and g(x) intersect
  let b := 0 in          -- points where f(x) and h(x) intersect
  7 * a + 2 * b = 14 :=   -- correct final value
by
  sorry

end intersection_points_and_final_value_l580_580174


namespace find_a_l580_580391

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x < 1 then 2 else x ^ 2 + a * x

theorem find_a (a : ℝ) (h : f a (f a 0) = 4 * a) : a = 2 := by
  sorry

end find_a_l580_580391


namespace exists_hamiltonian_cycle_l580_580856

variables (G : SimpleGraph (Fin (2 * n))) (n : ℕ)
  
noncomputable def degree_condition (G : SimpleGraph (Fin (2 * n))) (n : ℕ) : Prop :=
  ∀ k : ℕ, k ∈ Finset.range (n - 1) → G.degree_lt_k k < k
  
theorem exists_hamiltonian_cycle
  (h1 : ∀ v : Fin (2 * n), G.degree v = 2)
  (h2 : degree_condition G n)
  (h3 : 2 ≤ n):
  ¬ G.nonHamiltonian :=
sorry

end exists_hamiltonian_cycle_l580_580856


namespace prob_at_least_one_juice_l580_580962

/-- Given:
1. There are exactly 2 bottles of juice among 5 bottles of drinks.
2. 2 bottles are randomly selected from these 5 bottles.

Prove:
The probability that at least one of the selected bottles is a juice is 7/10.
-/
theorem prob_at_least_one_juice (H : 2 ≤ 5) : 
  let total_ways := nat.choose 5 2,
      non_juice_ways := nat.choose 3 2,
      prob := 1 - (non_juice_ways / total_ways : rat) in 
  prob = 7 / 10 :=
by
  let total_ways := nat.choose 5 2
  let non_juice_ways := nat.choose 3 2
  let prob := 1 - (non_juice_ways / total_ways : rat)
  have h1 : total_ways = 10 := by sorry
  have h2 : non_juice_ways = 3 := by sorry
  have h3 : prob = 1 - (3 / 10 : rat) := by sorry
  have h4 : prob = 7 / 10 := by sorry
  exact h4

end prob_at_least_one_juice_l580_580962


namespace high_jump_mode_median_l580_580514

def heights : list ℝ := [1.50, 1.50, 1.60, 1.60, 1.60, 1.65, 1.65, 1.65, 1.65, 1.65, 1.70, 1.70, 1.70, 1.70, 1.75]

def mode (l : list ℝ) : ℝ :=
  l.mode -- This assumes the mode function is defined elsewhere to return the most frequent element.

def median (l : list ℝ) : ℝ :=
  l.median -- This assumes the median function is defined elsewhere to return the middle value.

theorem high_jump_mode_median :
  mode heights = 1.65 ∧ median heights = 1.65 :=
by sorry

end high_jump_mode_median_l580_580514


namespace ex1_ex2_l580_580813

-- Definition of the "multiplication-subtraction" operation.
def mult_sub (a b : ℚ) : ℚ :=
  if a = 0 then abs b else if b = 0 then abs a else if abs a = abs b then 0 else
  if (a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0) then abs a - abs b else -(abs a - abs b)

theorem ex1 : mult_sub (mult_sub (3) (-2)) (mult_sub (-9) 0) = -8 :=
  sorry

theorem ex2 : ∃ (a b c : ℚ), (mult_sub (mult_sub a b) c) ≠ (mult_sub a (mult_sub b c)) :=
  ⟨3, -2, 4, by simp [mult_sub]; sorry⟩

end ex1_ex2_l580_580813


namespace angle_y_solution_l580_580301

theorem angle_y_solution
  (AB_parallel_DC : ∀ p q : ℝ, is_parallel AB DC)
  (ACF_straight : is_straight_line A C F)
  (angle_ADC_eq_120 : angle A D C = 120)
  (angle_ACF_eq_180 : angle A C F = 180)
  (angle_ACD_eq_95 : angle A C D = 95) :
  y = 55 := by
  sorry

end angle_y_solution_l580_580301


namespace ratio_of_times_l580_580472

-- Define the given conditions
def distance : ℝ := 144 -- in km
def orig_time : ℝ := 6 -- in hours
def new_speed : ℝ := 16 -- in kmph

-- Define the derived quantities based on the conditions
def orig_speed : ℝ := distance / orig_time -- Speed = Distance / Time
def new_time : ℝ := distance / new_speed -- Time = Distance / Speed

-- Define the proof problem as a theorem
theorem ratio_of_times (distance : ℝ) (orig_time : ℝ) (new_speed : ℝ) :
  distance = 144 → orig_time = 6 → new_speed = 16 →
  (distance / new_speed) / orig_time = 3 / 2 :=
by {
  intros h1 h2 h3,
  have orig_speed_def : orig_speed = 24 := by sorry,
  have new_time_def : new_time = 9 := by sorry,
  sorry
}

end ratio_of_times_l580_580472


namespace tan_theta_determined_l580_580889

theorem tan_theta_determined (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 4) (h_zero : Real.tan θ + Real.tan (4 * θ) = 0) :
  Real.tan θ = Real.sqrt (5 - 2 * Real.sqrt 5) :=
sorry

end tan_theta_determined_l580_580889


namespace min_value_product_l580_580110

noncomputable def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem min_value_product (a b c x1 x2 x3 : ℝ) (h1 : quadratic_function a b c (-1) = 0)
  (h2 : ∀ x : ℝ, quadratic_function a b c x ≥ x) 
  (h3 : ∀ x ∈ Set.Ioo 0 2, quadratic_function a b c x ≤ (x + 1)^2 / 4)
  (hx1 : x1 ∈ Set.Ioo 0 2) 
  (hx2 : x2 ∈ Set.Ioo 0 2) 
  (hx3 : x3 ∈ Set.Ioo 0 2)
  (hxsum : 1 / x1 + 1 / x2 + 1 / x3 = 3) : 
  f x1 * f x2 * f x3 = 1 :=
by
  sorry

end min_value_product_l580_580110


namespace closest_vector_is_3_over_7_l580_580206

def vector_u (t : ℚ) : ℚ × ℚ × ℚ :=
  (3 + 4 * t, -1 + 6 * t, 2 - 2 * t)

def vector_b : ℚ × ℚ × ℚ :=
  (5, 3, 6)

def direction_vector : ℚ × ℚ × ℚ :=
  (4, 6, -2)

def dot_product (v1 v2 : ℚ × ℚ × ℚ) : ℚ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem closest_vector_is_3_over_7 : 
  ∃ t : ℚ, dot_product ((vector_b.1 - (vector_u t).1, vector_b.2 - (vector_u t).2, vector_b.3 - (vector_u t).3)) direction_vector = 0 ∧ 
  t = 3 / 7 := 
by
  exists 3 / 7
  sorry

end closest_vector_is_3_over_7_l580_580206


namespace geometric_progressions_common_ratio_l580_580810

theorem geometric_progressions_common_ratio (a b p q : ℝ) :
  (∀ n : ℕ, (a * p^n + b * q^n) = (a * b) * ((p^n + q^n)/a)) →
  p = q := by
  sorry

end geometric_progressions_common_ratio_l580_580810


namespace sum_ceil_sqrt_5_to_39_l580_580897

def ceil (x : ℝ) : ℕ := ⌈x⌉

theorem sum_ceil_sqrt_5_to_39 : (∑ n in Finset.range (35) \ Finset.range (4), ceil (Real.sqrt (n + 5))) = 175 := by
  sorry

end sum_ceil_sqrt_5_to_39_l580_580897


namespace sum_of_possible_values_l580_580774

theorem sum_of_possible_values (M : ℝ) (h : M * (M + 4) = 12) : M + (if M = -6 then 2 else -6) = -4 :=
by
  sorry

end sum_of_possible_values_l580_580774


namespace remainder_of_product_mod_5_l580_580205

theorem remainder_of_product_mod_5 : 
  (∏ k in Finset.range 20, (7 + 10 * k)) % 5 = 1 := 
by
  sorry

end remainder_of_product_mod_5_l580_580205


namespace sum_of_squares_of_roots_l580_580541

theorem sum_of_squares_of_roots :
  let a := 5
  let b := -7
  let c := 2
  let x1 := (-b + (b^2 - 4*a*c)^(1/2)) / (2*a)
  let x2 := (-b - (b^2 - 4*a*c)^(1/2)) / (2*a)
  x1^2 + x2^2 = (b^2 - 2*a*c) / a^2 :=
by
  sorry

end sum_of_squares_of_roots_l580_580541


namespace powerFunctionAtPoint_l580_580595

def powerFunction (n : ℕ) (x : ℕ) : ℕ := x ^ n

theorem powerFunctionAtPoint (n : ℕ) (h : powerFunction n 2 = 8) : powerFunction n 3 = 27 :=
  by {
    sorry
}

end powerFunctionAtPoint_l580_580595


namespace gcd_abcd_plus_dcba_distinct_nonconsecutive_l580_580215

theorem gcd_abcd_plus_dcba_distinct_nonconsecutive : 
  ∀ a b c d : ℕ, 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    ¬abs (a - b) = 1 ∧ ¬abs (a - c) = 1 ∧ ¬abs (a - d) = 1 ∧ ¬abs (b - c) = 1 ∧ ¬abs (b - d) = 1 ∧ ¬abs (c - d) = 1 ∧ 
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10
  → gcd (1001 * (a + d) + 110 * (b + c)) 11 = 11 :=
by
  intros
  sorry

end gcd_abcd_plus_dcba_distinct_nonconsecutive_l580_580215


namespace telephone_number_A_is_8_l580_580861

theorem telephone_number_A_is_8 :
  ∃ A B C D E F G H I J : ℕ,
    A > B ∧ B > C ∧ D > E ∧ E > F ∧ G > H ∧ H > I ∧ I > J ∧
    -- D, E, F are consecutive odd digits:
    (D = E + 2) ∧ (E = F + 2) ∧ (D % 2 = 1) ∧ (E % 2 = 1) ∧ (F % 2 = 1) ∧
    -- G, H, I, J are consecutive even digits:
    (G = H + 2) ∧ (H = I + 2) ∧ (I = J + 2) ∧ (G % 2 = 0) ∧ (H % 2 = 0) ∧ (I % 2 = 0) ∧ (J % 2 = 0) ∧
    -- Other conditions:
    (A + B + C = 12) ∧
    ∀ x y z : ℕ, (x > y ∧ y > z → x ≠ D ∧ x ≠ E ∧ x ≠ F ∧ x ≠ G ∧ x ≠ H ∧ x ≠ I ∧ x ≠ J) →
    A = 8 :=
begin
  sorry -- Proof goes here
end

end telephone_number_A_is_8_l580_580861


namespace solve_system_equations_l580_580787

theorem solve_system_equations :
  ∃ x y : ℚ, (5 * x * (y + 6) = 0 ∧ 2 * x + 3 * y = 1) ∧
  (x = 0 ∧ y = 1 / 3 ∨ x = 19 / 2 ∧ y = -6) :=
by
  sorry

end solve_system_equations_l580_580787


namespace geometric_sequence_general_term_l580_580615

theorem geometric_sequence_general_term (a : ℝ) (a₀ a₁ a₂ : ℝ)
  (h₀ : a₀ = a - 1)
  (h₁ : a₁ = a + 1)
  (h₂ : a₂ = a + 4)
  (h_geom : ∃ q : ℝ, a₀ * q = a₁ ∧ a₁ * q = a₂) :
  a = 5 ∧ a₀ = 4 ∧ a₁ = 6 ∧ a₂ = 9 ∧ (∀ n : ℕ, ∃ q = (3/2), aₙ = 4 * q^(n-1)) :=
sorry

end geometric_sequence_general_term_l580_580615


namespace extreme_points_inequality_range_l580_580973

def f (x : ℝ) (a : ℝ) : ℝ := x - a * real.log x
def g (x : ℝ) (a : ℝ) : ℝ := - (a + 1) / x

theorem extreme_points (a : ℝ) : 
  (a ≤ 0 ∧ ∀ x > 0, f x a > f a a) ∨ 
  (a > 0 ∧ ∃! x > 0, f' x = 0) :=
sorry

theorem inequality_range (x : ℝ) (a : ℝ) :
  1 ≤ x ∧ x ≤ real.exp 1 ∧ -2 < a ∧ a < (real.exp 2 + 1) / (real.exp 1 - 1) → f x a > g x a :=
sorry

end extreme_points_inequality_range_l580_580973


namespace work_completion_days_l580_580098

theorem work_completion_days (h1 : (1:ℝ)/4 = (1:ℝ)/12 + (1:ℝ)/x) : 
  x = 6 :=
by sorry

end work_completion_days_l580_580098


namespace length_of_MN_l580_580677

theorem length_of_MN (XYZ : Type*) [plane_triangle XYZ] 
  (XY : segment XYZ) (Y : point_in_triangle XYZ) 
  (X_eq_YZ_length : length XY = 30) (angle_Y_45 : angle Y = 45)
  (M : point_on_segment XY) (midpoint_M : midpoint XY M)
  (N : point_on_segment (bisect YZ M)) : 
  length (segment M N) = 15 * real.sqrt(2) :=
by
sorry

end length_of_MN_l580_580677


namespace triangle_30_60_90_PQ_l580_580197

theorem triangle_30_60_90_PQ (PR : ℝ) (hPR : PR = 18 * Real.sqrt 3) : 
  ∃ PQ : ℝ, PQ = 54 :=
by
  sorry

end triangle_30_60_90_PQ_l580_580197


namespace power_of_power_l580_580875

theorem power_of_power (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 := 
  sorry

end power_of_power_l580_580875


namespace percentage_of_non_defective_products_l580_580445

-- Define the conditions
def totalProduction : ℕ := 100
def M1_production : ℕ := 25
def M2_production : ℕ := 35
def M3_production : ℕ := 40

def M1_defective_rate : ℝ := 0.02
def M2_defective_rate : ℝ := 0.04
def M3_defective_rate : ℝ := 0.05

-- Calculate the total defective units
noncomputable def total_defective_units : ℝ := 
  (M1_defective_rate * M1_production) + 
  (M2_defective_rate * M2_production) + 
  (M3_defective_rate * M3_production)

-- Calculate the percentage of defective products
noncomputable def defective_percentage : ℝ := (total_defective_units / totalProduction) * 100

-- Calculate the percentage of non-defective products
noncomputable def non_defective_percentage : ℝ := 100 - defective_percentage

-- The statement to prove
theorem percentage_of_non_defective_products :
  non_defective_percentage = 96.1 :=
by
  sorry

end percentage_of_non_defective_products_l580_580445


namespace proof_ff5_equals_5_l580_580624

def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log (1/3)
  else (1/3) ^ x

theorem proof_ff5_equals_5 : f (f 5) = 5 :=
by {
  sorry,
}

end proof_ff5_equals_5_l580_580624


namespace valid_votes_for_Candidate_A_l580_580294

-- We declare the variables and assume the given conditions.
variables (total_votes : ℕ) 
          (invalid_percentage : ℝ) 
          (valid_percentage_A : ℝ)

-- We define the conditions of the problem.
def conditions := 
  total_votes = 1200000 ∧ 
  invalid_percentage = 0.25 ∧ 
  valid_percentage_A = 0.45

-- We want to prove that the number of valid votes for Candidate A is 405,000.
theorem valid_votes_for_Candidate_A : conditions total_votes invalid_percentage valid_percentage_A → 
  (total_votes * (1 - invalid_percentage) * valid_percentage_A = 405000) := 
by 
  intros, 
  have valid_votes := total_votes * (1 - invalid_percentage),
  have valid_votes_A := valid_votes * valid_percentage_A,
  sorry

end valid_votes_for_Candidate_A_l580_580294


namespace intersection_M_N_l580_580265

-- Definitions of sets M and N
def M : Set ℕ := {1, 2, 5}
def N : Set ℕ := {x | x ≤ 2}

-- Lean statement to prove that the intersection of M and N is {1, 2}
theorem intersection_M_N : M ∩ N = {1, 2} :=
by
  sorry

end intersection_M_N_l580_580265


namespace distance_between_closest_points_of_two_circles_tangent_to_y_axis_l580_580036

noncomputable def distance_closest_points_of_tangent_circles
  (center1 center2 : ℝ × ℝ) (tangent1 tangent2 : Prop) : ℝ :=
let radius1 := center1.1 in
let radius2 := center2.1 in
let d := Real.sqrt ((center2.1 - center1.1) ^ 2 + (center2.2 - center1.2) ^ 2) in
d - (radius1 + radius2)

theorem distance_between_closest_points_of_two_circles_tangent_to_y_axis :
  distance_closest_points_of_tangent_circles (5,5) (25,15) (5 = 5) (25 = 15) = 10 * Real.sqrt 5 - 20 :=
by
  sorry

end distance_between_closest_points_of_two_circles_tangent_to_y_axis_l580_580036


namespace curve_distance_max_min_l580_580720

noncomputable def distance (x y : ℝ) : ℝ := real.sqrt (x^2 + y^2)

theorem curve_distance_max_min (x y : ℝ) (h : x^2 + y^2 = 1 + |x * y|) :
  (distance x y ≤ real.sqrt 2 ∧ distance x y ≥ 1) :=
  sorry

end curve_distance_max_min_l580_580720


namespace floor_shaded_area_l580_580834

-- Define the radius of each quarter circle
def quarter_circle_radius : ℝ := 3 / 4

-- Define the area of one quarter circle
def area_one_quarter_circle : ℝ := (Real.pi * quarter_circle_radius ^ 2) / 4

-- Define the area of the white sections in one tile
def total_white_area_per_tile : ℝ := 4 * area_one_quarter_circle

-- Define the area of one tile
def area_one_tile : ℝ := (1.5) ^ 2

-- Define the area of the shaded region in one tile
def area_shaded_per_tile : ℝ := area_one_tile - total_white_area_per_tile

-- Define the floor dimensions and number of tiles
def floor_dimensions : ℝ × ℝ := (9, 12)
def tile_dimension : ℝ := 1.5
def number_of_tiles : ℝ := (floor_dimensions.fst / tile_dimension) * (floor_dimensions.snd / tile_dimension)

-- Define the total shaded area on the floor
def total_shaded_area : ℝ := number_of_tiles * area_shaded_per_tile

-- The hypothesis statement
theorem floor_shaded_area : total_shaded_area = 108 - 27 * Real.pi / 4 :=
by
  sorry

end floor_shaded_area_l580_580834


namespace find_interest_rate_l580_580908

def compoundInterestRate (P t A : ℝ) (n : ℕ) : ℝ :=
  n * ((A / P)^(1 / (t * n)) - 1)

theorem find_interest_rate :
  let P := 5000
  let t := 1.5
  let n := 2
  let interest := 302.98
  let A := P + interest
  compoundInterestRate P t A n = 0.0396 :=
by
  sorry

end find_interest_rate_l580_580908


namespace perpendicular_condition_l580_580263

variable (a : ℝ)

def l1 : ℝ → Prop := λ (x y : ℝ), a * x + (a + 2) * y + 1 = 0
def l2 : ℝ → Prop := λ (x y : ℝ), x + a * y + 2 = 0

theorem perpendicular_condition (h : a = -3) : 
  (∃ (x y : ℝ), l1 a x y ∧ l2 a x y) → by sorry :=
sorry

end perpendicular_condition_l580_580263


namespace crayon_colors_correct_l580_580475

-- The Lean code will define the conditions and the proof statement as follows:
noncomputable def crayon_problem := 
  let crayons_per_box := (160 / (5 * 4)) -- Total crayons / Total boxes
  let colors := (crayons_per_box / 2) -- Crayons per box / Crayons per color
  colors = 4

-- This is the theorem that needs to be proven:
theorem crayon_colors_correct : crayon_problem := by
  sorry

end crayon_colors_correct_l580_580475


namespace evaluate_expression_l580_580555

theorem evaluate_expression :
  (2 / 10 + 3 / 100 + 5 / 1000 + 7 / 10000)^2 = 0.05555649 :=
by
  sorry

end evaluate_expression_l580_580555


namespace remainder_when_divided_l580_580105

theorem remainder_when_divided (x : ℤ) (k : ℤ) (h: x = 82 * k + 5) : 
  ((x + 17) % 41) = 22 := by
  sorry

end remainder_when_divided_l580_580105


namespace garage_sale_items_count_l580_580162

theorem garage_sale_items_count :
  (16 + 22) + 1 = 38 :=
by
  -- proof goes here
  sorry

end garage_sale_items_count_l580_580162


namespace solution_set_of_inequality_l580_580933

noncomputable def f : ℝ → ℝ :=
sorry  -- Definition of the function f isn't provided in the original problem.

theorem solution_set_of_inequality (h_deriv : ∀ x, deriv f x < f x)
  (h_even : ∀ x, f (x + 2) = f (2 - x))
  (h_value : f 4 = 1) :
  {x : ℝ | f x < real.exp x} = set.Ioi 0 :=
sorry

end solution_set_of_inequality_l580_580933


namespace radius_of_sphere_l580_580491

theorem radius_of_sphere {r x : ℝ} (h1 : 15^2 + x^2 = r^2) (h2 : r = x + 12) :
    r = 123 / 8 :=
  by
  sorry

end radius_of_sphere_l580_580491


namespace complex_addition_l580_580521

theorem complex_addition :
  (⟨6, -5⟩ : ℂ) + (⟨3, 2⟩ : ℂ) = ⟨9, -3⟩ := 
sorry

end complex_addition_l580_580521


namespace math_problem_l580_580709

noncomputable def g : ℝ → ℝ := sorry

theorem math_problem (
  H : ∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - x * y - x + 2
) : 
  let n := 1 in -- the number of possible values of g(1)
  let t := 3 in -- the sum of all possible values of g(1)
  n * t = 3 := 
sorry

end math_problem_l580_580709


namespace boat_travel_time_l580_580838

noncomputable def boat_speed : ℝ := 20.0
noncomputable def stream_speed1 : ℝ := 4.0
noncomputable def stream_speed2 : ℝ := 7.0
noncomputable def stream_speed3 : ℝ := 3.0
noncomputable def length_river1 : ℝ := 30.0
noncomputable def length_river2 : ℝ := 35.0
noncomputable def total_distance : ℝ := 94.0

theorem boat_travel_time :
  let effective_speed1 := boat_speed + stream_speed1 in
  let effective_speed2 := boat_speed + stream_speed2 in
  let effective_speed3 := boat_speed + stream_speed3 in
  let time1 := length_river1 / effective_speed1 in
  let time2 := length_river2 / effective_speed2 in
  let length_river3 := total_distance - (length_river1 + length_river2) in
  let time3 := length_river3 / effective_speed3 in
  let total_time := time1 + time2 + time3 in
  total_time ≈ 3.807 :=
  by sorry

end boat_travel_time_l580_580838


namespace unique_real_solution_l580_580564

-- Define the main equation
def main_equation (x : ℝ) : ℝ := (2^(4*x + 2)) * (4^(2*x + 3)) - (8^(3*x + 4))

-- State the theorem to prove there is exactly one real solution to the equation
theorem unique_real_solution : ∃! x : ℝ, main_equation x = 0 :=
by
  sorry

end unique_real_solution_l580_580564


namespace entrepreneur_should_reduce_production_by_20_percent_l580_580981

noncomputable def percentage_reduction : ℕ → ℕ → ℕ → ℕ → ℕ := 
  λ (total_items : ℕ) (price_per_item : ℕ) (profit : ℕ) (variable_cost_per_item : ℕ),
  let total_revenue := total_items * price_per_item in
  let constant_costs := total_revenue - profit in
  let new_production := constant_costs / (price_per_item - variable_cost_per_item) in
  let production_reduction := total_items - new_production in
  (production_reduction * 100) / total_items

theorem entrepreneur_should_reduce_production_by_20_percent 
  (total_items : ℕ) (price_per_item : ℕ) (profit : ℕ) (variable_cost_per_item : ℕ) 
  (h_total_items : total_items = 4000)
  (h_price_per_item : price_per_item = 6250)
  (h_profit : profit = 2000000) -- 2 million
  (h_variable_cost_per_item : variable_cost_per_item = 3750) :
  percentage_reduction total_items price_per_item profit variable_cost_per_item = 20 :=
sorry

end entrepreneur_should_reduce_production_by_20_percent_l580_580981


namespace problem_ratio_l580_580869

-- Define the conditions
variables 
  (R : ℕ) 
  (Bill_problems : ℕ := 20) 
  (Frank_problems_per_type : ℕ := 30)
  (types : ℕ := 4)

-- State the problem to prove
theorem problem_ratio (h1 : 3 * R = Frank_problems_per_type * types) :
  R / Bill_problems = 2 :=
by
  -- placeholder for proof
  sorry

end problem_ratio_l580_580869


namespace total_cost_tom_pays_for_trip_l580_580034

/-- Tom needs to get 10 different vaccines and a doctor's visit to go to Barbados.
    Each vaccine costs $45.
    The doctor's visit costs $250.
    Insurance will cover 80% of these medical bills.
    The trip itself costs $1200.
    Prove that the total amount Tom has to pay for his trip to Barbados, including medical expenses, is $1340. -/
theorem total_cost_tom_pays_for_trip : 
  let cost_per_vaccine := 45
  let number_of_vaccines := 10
  let cost_doctor_visit := 250
  let insurance_coverage_rate := 0.8
  let trip_cost := 1200
  let total_medical_cost := (number_of_vaccines * cost_per_vaccine) + cost_doctor_visit
  let insurance_coverage := insurance_coverage_rate * total_medical_cost
  let net_medical_cost := total_medical_cost - insurance_coverage
  let total_cost := trip_cost + net_medical_cost
  total_cost = 1340 := 
by 
  sorry

end total_cost_tom_pays_for_trip_l580_580034


namespace paintable_sum_l580_580638

def is_painted_once (h t u : ℕ) (n : ℕ) : Prop :=
  (n > 0) ∧
  ((n % h = 1) ∨ (n % t = 3) ∨ (n % u = 5)) ∧
  ∀ k ∈ [h, t, u], n % k ≠ 1 ∨ n % k ≠ 3 ∨ n % k ≠ 5

def is_paintable (h t u : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → is_painted_once h t u n

theorem paintable_sum :
  let S := {100 * h + 10 * t + u | h t u : ℕ, is_paintable h t u} in
  ∑ s in S, s = 670 := sorry

end paintable_sum_l580_580638


namespace number_of_ways_to_pick_three_cards_l580_580229

theorem number_of_ways_to_pick_three_cards :
  ∃ n : ℕ, n = 52 * 51 * 50 :=
by {
  use 52 * 51 * 50,
  sorry
}

end number_of_ways_to_pick_three_cards_l580_580229


namespace order_of_abc_l580_580928

noncomputable def a : ℝ := (1 / 3) ^ (2 / 5)
noncomputable def b : ℝ := 2 ^ (4 / 3)
noncomputable def c : ℝ := Real.log 1 / 3 / Real.log 2

theorem order_of_abc : c < a ∧ a < b :=
by {
  -- The proof would go here
  sorry
}

end order_of_abc_l580_580928


namespace sheila_picnic_probability_l580_580371

theorem sheila_picnic_probability :
  let P_rain := 0.5
  let P_go_given_rain := 0.3
  let P_go_given_sunny := 0.9
  let P_remember := 0.9  -- P(remember) = 1 - P(forget)
  let P_sunny := 1 - P_rain
  
  P_rain * P_go_given_rain * P_remember + P_sunny * P_go_given_sunny * P_remember = 0.54 :=
by
  sorry

end sheila_picnic_probability_l580_580371


namespace num_elementary_events_sum_four_l580_580646

def is_event_sum_four (a b : ℕ) : Prop :=
  a + b = 4

def uniform_dice : set ℕ :=
  {1, 2, 3, 4, 5, 6}

theorem num_elementary_events_sum_four : 
  (∃ (n : ℕ), n = 3) :=
sorry

end num_elementary_events_sum_four_l580_580646


namespace problem_equivalence_l580_580169

theorem problem_equivalence :
  (∑ n in Finset.range 2008, (n + 1) * (Nat.factorial (n + 1))) = Nat.factorial 2009 - 1 := 
by
  sorry

end problem_equivalence_l580_580169


namespace original_number_of_snails_l580_580417

theorem original_number_of_snails (a b : ℕ) (h₁ : a = 3482) (h₂ : b = 8278) : a + b = 11760 :=
by {
  rw [h₁, h₂],
  norm_num,
  sorry
}

end original_number_of_snails_l580_580417


namespace perpendicular_lines_l580_580208

def d1 (b : ℝ) : Mathlib.Vector3 := ⟨2 * b, 3, -1⟩
def d2 : Mathlib.Vector3 := ⟨2, -6, 0⟩

theorem perpendicular_lines (b : ℝ) : 
  Mathlib.Vector3.dot (d1 b) d2 = 0 ↔ b = 9 / 2 := 
by
  sorry

end perpendicular_lines_l580_580208


namespace probability_shaded_is_one_third_l580_580845

-- Define the total number of regions as a constant
def total_regions : ℕ := 12

-- Define the number of shaded regions as a constant
def shaded_regions : ℕ := 4

-- The probability that the tip of a spinner stopping in a shaded region
def probability_shaded : ℚ := shaded_regions / total_regions

-- Main theorem stating the probability calculation is correct
theorem probability_shaded_is_one_third : probability_shaded = 1 / 3 :=
by
  sorry

end probability_shaded_is_one_third_l580_580845


namespace paul_sold_books_is_94_l580_580736

-- Define the conditions as assumptions
axiom paul_books_original : ℕ := 2
axiom paul_books_bought : ℕ := 150
axiom paul_books_current : ℕ := 58

-- Define the number of books sold
def books_sold (S : ℕ) := 
  (paul_books_original - S + paul_books_bought = paul_books_current)

-- Prove that the number of books Paul sold is 94
theorem paul_sold_books_is_94 : books_sold 94 :=
  by
  -- Since we don't need to provide proof, insert sorry
  sorry

end paul_sold_books_is_94_l580_580736


namespace inequality_proof_l580_580741

namespace MyProof

theorem inequality_proof (n : ℕ) (a b : Fin n → ℝ) 
  (h_pos_a : ∀ i, 0 < a i) (h_pos_b : ∀ i, 0 < b i) :
  (∑ k : Fin n, a k * b k / (a k + b k)) ≤ 
  (∑ k : Fin n, a k) * (∑ k : Fin n, b k) / 
  (∑ k : Fin n, a k + ∑ k : Fin n, b k) := 
begin
  sorry
end

end MyProof

end inequality_proof_l580_580741


namespace investment_to_reach_target_l580_580024

-- Definitions of the given conditions
def annual_interest_rate : ℝ := 0.07
def compounding_frequency : ℕ := 2
def total_amount : ℝ := 80000
def time_in_years : ℕ := 10
def growth_factor : ℝ := (1 + annual_interest_rate / compounding_frequency)
def number_of_compoundings : ℕ := compounding_frequency * time_in_years

-- Theorem statement to be proved
theorem investment_to_reach_target (P : ℝ) 
  (hP : P ≈ total_amount / (growth_factor ^ number_of_compoundings)) : 
  P ≈ 38126 := 
sorry

end investment_to_reach_target_l580_580024


namespace degree_geq_p_minus_one_l580_580337

open Nat

-- Define the given conditions
variable {p : ℕ} (prime_p : Prime p)
variable {f : ℤ → ℤ}
variable {d : ℕ}

-- Define that f is a polynomial of degree d
variable (poly_f : ∃ coeff : List ℤ, ∀ x, f x = coeff.sum (λ (a, i), a * x^i) ∧ coeff.length - 1 = d)

-- Define the initial conditions
variable (f_0 : f 0 = 0)
variable (f_1 : f 1 = 1)

-- Define the divisibility condition
variable (div_cond : ∀ n : ℕ, IsNat (rn, (f n) % p) ∧ rn = 0 ∨ rn = 1)

theorem degree_geq_p_minus_one : d ≥ p - 1 :=
sorry

end degree_geq_p_minus_one_l580_580337


namespace locus_of_moving_circle_l580_580140

noncomputable theory
open Real

def circle_M (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := x^2 + y^2 - 8 * x + 12 = 0

theorem locus_of_moving_circle (x y : ℝ) :
  (∃ p q : ℝ, circle_M p q ∧ circle_N p q ∧ is_tangent (p, q) (x, y)) →
  ( (x + 2)^2 - y^2 / 169 = 1 ) :=
sorry

end locus_of_moving_circle_l580_580140


namespace max_colors_l580_580353

theorem max_colors {n : ℕ} (h₁ : n ≥ 3) 
  (distinct_lengths : ∀ (p q : ℕ), p ≠ q → (∃! l : ℕ, l = dist p q))
  (coloring : ∀ (P : ℕ → ℕ),
    (∀ Q R, Q ≠ R → P(Q) = P(R)) ∧ 
    ∀ L M, L ≠ M → P(L) = P(M)) :
    ∃ k : ℕ, k = (n + 1) / 4 :=
sorry

end max_colors_l580_580353


namespace supply_duration_approx_3_months_l580_580693

-- Definitions of the conditions from the problem
def pill_rate : ℚ := 3 / 4  -- three-quarters of a pill
def time_period : ℚ := 3  -- every three days
def total_pills : ℚ := 60  -- one supply contains 60 pills
def days_per_pill : ℚ := time_period / pill_rate  -- 3 / (3/4) = 4/3 days per pill
def avg_month_days : ℚ := 30  -- average month length

-- Theorem to be proven
theorem supply_duration_approx_3_months :
  (total_pills * days_per_pill) / avg_month_days ≈ 3 :=
by
  sorry

end supply_duration_approx_3_months_l580_580693


namespace distinct_triangles_count_l580_580269

def num_points : ℕ := 8
def num_rows : ℕ := 2
def num_cols : ℕ := 4

-- Define the number of ways to choose 3 points from the 8 available points.
def combinations (n k : ℕ) := Nat.choose n k
def total_combinations := combinations num_points 3

-- Define the number of degenerate cases of collinear points in columns.
def degenerate_cases_per_column := combinations num_cols 3
def total_degenerate_cases := num_cols * degenerate_cases_per_column

-- The number of distinct triangles is the total combinations minus the degenerate cases.
def distinct_triangles := total_combinations - total_degenerate_cases

theorem distinct_triangles_count : distinct_triangles = 40 := by
  -- the proof goes here
  sorry

end distinct_triangles_count_l580_580269


namespace sin_add_theta_phi_l580_580993

noncomputable def theta : ℂ := (1 / 5) + (2 * Real.sqrt 6 / 5) * Complex.I
noncomputable def phi : ℂ := - (5 / 13) - (12 / 13) * Complex.I
noncomputable def e_i_theta_phi := theta * phi
noncomputable def sin_theta_phi := - (12 - 10 * Real.sqrt 6) / 65

theorem sin_add_theta_phi :
  Complex.sin (Complex.argument e_i_theta_phi) = sin_theta_phi := 
sorry

end sin_add_theta_phi_l580_580993


namespace code_XYZ_to_base_10_l580_580844

def base_6_to_base_10 (x y z : ℕ) : ℕ :=
  x * 6^2 + y * 6^1 + z * 6^0

theorem code_XYZ_to_base_10 :
  ∀ (X Y Z : ℕ), 
    X = 5 ∧ Y = 0 ∧ Z = 4 →
    base_6_to_base_10 X Y Z = 184 :=
by
  intros X Y Z h
  cases' h with hX hYZ
  cases' hYZ with hY hZ
  rw [hX, hY, hZ]
  exact rfl

end code_XYZ_to_base_10_l580_580844


namespace october_1st_is_thursday_l580_580657

theorem october_1st_is_thursday (h1 : ∃ n : ℕ, n = 31) 
                                 (h2 : ∃ s : ℕ, s = 5 ∧ s = 4) 
                                 (days_in_week : ℕ := 7)
                                 (last_day : (oct_31 : ℕ := 31) → ℕ := (7 - 1))
                                 : 
                                 (oct_1 : ℕ := 1) → ℕ := (days_in_week * ((oct_31 - oct_1) % days_in_week) + 4) :=
sorry

end october_1st_is_thursday_l580_580657


namespace benny_placed_3_crayons_l580_580019

def initial_crayons : ℕ := 9
def current_crayons : ℕ := 12
def placed_crayons : ℕ := current_crayons - initial_crayons

theorem benny_placed_3_crayons (initial_crayons current_crayons placed_crayons : ℕ) (h1 : initial_crayons = 9) (h2 : current_crayons = 12) : placed_crayons = 3 := by
  rw [h1, h2]
  exact Nat.sub_eq_self.mpr rfl
  sorry

end benny_placed_3_crayons_l580_580019


namespace maximum_value_fraction_l580_580239

theorem maximum_value_fraction (a b : ℝ) (h : a + b = 4) : 
  (1 / (a ^ 2 + 1) + 1 / (b ^ 2 + 1)) ≤ (√5 + 2) / 4 :=
sorry

end maximum_value_fraction_l580_580239


namespace employees_without_increase_l580_580735

-- Define the constants and conditions
def total_employees : ℕ := 480
def salary_increase_percentage : ℕ := 10
def travel_allowance_increase_percentage : ℕ := 20

-- Define the calculations derived from conditions
def employees_with_salary_increase : ℕ := (salary_increase_percentage * total_employees) / 100
def employees_with_travel_allowance_increase : ℕ := (travel_allowance_increase_percentage * total_employees) / 100

-- Total employees who got increases assuming no overlap
def employees_with_increases : ℕ := employees_with_salary_increase + employees_with_travel_allowance_increase

-- The proof statement
theorem employees_without_increase :
  total_employees - employees_with_increases = 336 := by
  sorry

end employees_without_increase_l580_580735


namespace factorize_x4_plus_81_l580_580901

theorem factorize_x4_plus_81 (x : ℂ) : 
  (x^4 + 81) = (x^2 + 9 * complex.i) * (x^2 - 9 * complex.i) :=
  sorry

end factorize_x4_plus_81_l580_580901


namespace max_profit_l580_580474

-- Definition of the conditions
def production_requirements (tonAprodA tonAprodB tonBprodA tonBprodB: ℕ )
  := tonAprodA = 3 ∧ tonAprodB = 1 ∧ tonBprodA = 2 ∧ tonBprodB = 3

def profit_per_ton ( profitA profitB: ℕ )
  := profitA = 50000 ∧ profitB = 30000

def raw_material_limits ( rawA rawB: ℕ)
  := rawA = 13 ∧ rawB = 18

theorem max_profit 
  (production_requirements: production_requirements 3 1 2 3)
  (profit_per_ton: profit_per_ton 50000 30000)
  (raw_material_limits: raw_material_limits 13 18)
: ∃ (maxProfit: ℕ), maxProfit = 270000 := 
by 
  sorry

end max_profit_l580_580474


namespace find_C_monthly_income_l580_580396

theorem find_C_monthly_income (A_m B_m C_m : ℝ) (h1 : A_m / B_m = 5 / 2) (h2 : B_m = 1.12 * C_m) (h3 : 12 * A_m = 504000) : C_m = 15000 :=
sorry

end find_C_monthly_income_l580_580396


namespace find_XY_l580_580704

noncomputable def pointsOnCircle (A B C D P Q : Point) (circle : Circle) : Prop :=
  onCircle A circle ∧ onCircle B circle ∧ onCircle C circle ∧ onCircle D circle ∧ P ∈ segment AB ∧ Q ∈ segment CD

noncomputable def distances (A B C D P Q : Point) : Prop :=
  dist A B = 15 ∧ dist C D = 21 ∧ dist A P = 9 ∧ dist C Q = 9 ∧ dist P Q = 30

noncomputable def intersectLineCircle (P Q X Y : Point) (circle : Circle) : Prop :=
  lineThrough P Q ∧ intersect lineThrough(P, Q) circle = {X, Y}

noncomputable def required_distance (P Q X Y : Point) : Prop := XY = 34.8

theorem find_XY (A B C D P Q X Y : Point) (circle : Circle) (h1 : pointsOnCircle A B C D P Q circle) 
                (h2 : distances A B C D P Q) (h3 : intersectLineCircle P Q X Y circle) :
  required_distance X Y :=
sorry

end find_XY_l580_580704


namespace max_min_values_monotone_interval_l580_580974

def f (x a : ℝ) : ℝ := x^2 + 2*a*x + 2

theorem max_min_values (a : ℝ) (h : a = -1) :
  (∃ x_min x_max, -5 ≤ x_min ∧ x_min ≤ 5 ∧ -5 ≤ x_max ∧ x_max ≤ 5 ∧ 
  (∀ x ∈ Icc (-5:ℝ) 5, f x a ≥ f x_min a) ∧ 
  (∀ x ∈ Icc (-5:ℝ) 5, f x a ≤ f x_max a) ∧ 
  f x_min a = 1 ∧ f x_max a = 37) :=
by sorry

theorem monotone_interval (a : ℝ) :
  (∀ x y, x ∈ Icc (-5:ℝ) 5 ∧ y ∈ Icc (-5:ℝ) 5 ∧ x ≤ y → f x a ≤ f y a) ∨ 
  (∀ x y, x ∈ Icc (-5:ℝ) 5 ∧ y ∈ Icc (-5:ℝ) 5 ∧ x ≤ y → f y a ≤ f x a)
  ↔ a ∈ Set.Iic (-5) ∪ Set.Ici 5 :=
by sorry

end max_min_values_monotone_interval_l580_580974


namespace eccentricity_range_l580_580965

noncomputable def ellipse (m : ℝ) (x y : ℝ) : Prop :=
  (x^2 / (m + 2)) - (y^2 / (-1)) = 1

noncomputable def hyperbola (m : ℝ) (x y : ℝ) : Prop :=
  (x^2 / m) + (y^2 / (-1)) = 1

theorem eccentricity_range (m : ℝ) (n : ℝ) (h_ellipse : ∀ x y, ellipse m x y) (h_hyperbola: ∀ x y, hyperbola m x y) :
  -1 = n → 0 < m → (∀ e, ellipse m e e → e ∈ Ioo (sqrt 2 / 2) 1) := sorry

end eccentricity_range_l580_580965


namespace value_of_x_for_real_y_l580_580643

theorem value_of_x_for_real_y (x y : ℝ) (h : 4 * y^2 - 2 * x * y + 2 * x + 9 = 0) : x ≤ -3 ∨ x ≥ 12 :=
sorry

end value_of_x_for_real_y_l580_580643


namespace polynomial_bound_l580_580939

noncomputable def is_degree_at_most (p : ℝ[X]) (d : ℕ) : Prop :=
  p.degree ≤ d

theorem polynomial_bound 
  (p : ℝ[X]) (n : ℕ) 
  (hdeg : is_degree_at_most p (2 * n)) 
  (hbound : ∀ k : ℤ, k ≥ -n ∧ k ≤ n → |p.eval k| ≤ 1) : 
  ∀ x : ℝ, x ≥ -n ∧ x ≤ n → |p.eval x| ≤ 2^(2 * n) :=
begin
  sorry
end

end polynomial_bound_l580_580939


namespace mother_age_l580_580727

theorem mother_age (x : ℕ) (h1 : 3 * x + x = 40) : 3 * x = 30 :=
by
  -- Here we should provide the proof but for now we use sorry to skip it
  sorry

end mother_age_l580_580727


namespace spotted_and_fluffy_cats_l580_580497

theorem spotted_and_fluffy_cats (total_cats : ℕ) (total_cats_eq : total_cats = 120) 
  (spotted_fraction : ℚ) (spotted_fraction_eq : spotted_fraction = 1/3)
  (fluffy_fraction : ℚ) (fluffy_fraction_eq : fluffy_fraction = 1/4) :
  let spotted_cats := (total_cats * spotted_fraction).natAbs in
  let fluffy_spotted_cats := (spotted_cats * fluffy_fraction).natAbs in
  fluffy_spotted_cats = 10 :=
by
  sorry

end spotted_and_fluffy_cats_l580_580497


namespace geometric_sequence_sum_9000_l580_580782

noncomputable def sum_geometric_sequence (a r : ℝ) (n : ℕ) : ℝ := 
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_9000 (a r : ℝ) (h : r ≠ 1) 
  (h1 : sum_geometric_sequence a r 3000 = 1000)
  (h2 : sum_geometric_sequence a r 6000 = 1900) : 
  sum_geometric_sequence a r 9000 = 2710 :=
sorry

end geometric_sequence_sum_9000_l580_580782


namespace cheapest_third_company_l580_580791

theorem cheapest_third_company (x : ℕ) :
  (120 + 18 * x ≥ 150 + 15 * x) ∧ (220 + 13 * x ≥ 150 + 15 * x) → 36 ≤ x :=
by
  intro h
  cases h with
  | intro h1 h2 =>
    sorry

end cheapest_third_company_l580_580791


namespace tangent_point_coordinates_l580_580409

noncomputable def tangent_parallel_to_line (P : ℝ × ℝ) :=
  let f := λ x, x^3 + x^2 in
  let f' := λ x, 3 * x^2 + 2 * x in
  let yLine := λ x, 4 * x in
  (f(P.1) = P.2) ∧ (f'(P.1) = (yLine 1)) 

theorem tangent_point_coordinates :
  (tangent_parallel_to_line (1, 2)) ∨ (tangent_parallel_to_line (-1, -2)) :=
sorry

end tangent_point_coordinates_l580_580409


namespace sum_of_solutions_l580_580667

theorem sum_of_solutions (y : ℤ) (x1 x2 : ℤ) (h1 : y = 8) (h2 : x1^2 + y^2 = 145) (h3 : x2^2 + y^2 = 145) : x1 + x2 = 0 := by
  sorry

end sum_of_solutions_l580_580667


namespace find_intersection_point_l580_580011

theorem find_intersection_point {l_1 l_2 : Type} 
  (slope_l1: ℝ)
  (parallel_l1_l2: l_1 ∥ l_2)
  (point_passes_l2: ∃ (p : ℝ × ℝ), p = (-1, 1)) :
  ∃ (P : ℝ × ℝ), P = (0, 3) :=
by
  obtain ⟨p, h_p⟩ := point_passes_l2
  have : slope_l1 = 2 := sorry
  have : ∀ l, l_1 ∥ l → slope_l1 = slope_l2 := sorry
  have : ∀ point_on_l2, y_of_l2 point_on_l2 = 2 * x_of_l2 point_on_l2 + 3 := sorry
  existsi (0, 3)
  sorry

end find_intersection_point_l580_580011


namespace cannot_place_1965_points_in_square_l580_580311

theorem cannot_place_1965_points_in_square :
  ¬ (∃ (points : Finset (ℝ × ℝ)), 
    points.card = 1965 ∧
    ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ < x₂ → y₁ < y₂ → 
    (x₂ - x₁) * (y₂ - y₁) = (1 : ℝ) / 200 → 
    ∃ p ∈ points, 
    p.1 ∈ Icc x₁ x₂ ∧ p.2 ∈ Icc y₁ y₂) := 
sorry

end cannot_place_1965_points_in_square_l580_580311


namespace simplify_sqrt_eq_sin_sub_cos_l580_580435

theorem simplify_sqrt_eq_sin_sub_cos (α : ℝ) (h1 : (π / 4) < α) (h2 : α < (π / 2)) : 
  sqrt (1 - sin (2 * α)) = sin α - cos α := 
sorry

end simplify_sqrt_eq_sin_sub_cos_l580_580435


namespace f_of_2_eq_neg26_l580_580585

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x ^ 5 + b * x ^ 3 + Real.sin x - 8

-- Given conditions
variables (a b : ℝ)
hypothesis h : f a b (-2) = 10

-- Prove that f(2) = -26
theorem f_of_2_eq_neg26 : f a b 2 = -26 := by
  sorry

end f_of_2_eq_neg26_l580_580585


namespace problem_statement_l580_580862

variable (A B C O H : Point)
variables (R : ℝ)
variables (D E F : Point)

-- Assuming A, B, C are points of triangle with circumcenter O, orthocenter H, and circumradius R
def triangle_ABC_with_properties : Prop :=
  is_triangle A B C ∧
  circumcenter A B C = O ∧
  orthocenter A B C = H ∧
  circumradius A B C = R ∧
  reflection A (line_through B C) = D ∧
  reflection B (line_through C A) = E ∧
  reflection C (line_through A B) = F

-- Collinearity condition for points D, E, F
def collinear_def (D E F : Point) : Prop :=
  collinear D E F

-- Main proof problem statement
theorem problem_statement (A B C O H : Point) (R : ℝ) (D E F : Point)
  (h : triangle_ABC_with_properties A B C O H R D E F) :
  collinear_def D E F ↔ distance O H = 2 * R :=
sorry

end problem_statement_l580_580862


namespace op_value_l580_580887

noncomputable def op (a b c : ℝ) (k : ℤ) : ℝ :=
  b^2 - k * a^2 * c

theorem op_value : op 2 5 3 3 = -11 := by
  sorry

end op_value_l580_580887


namespace collinear_TXY_l580_580424

open EuclideanGeometry

variables (Γ₁ Γ₂ : Circle) (T A B C D X Y : Point)
variables (ℓ₁ ℓ₂ : Line)
variables (Ω : Circle)

-- Given conditions
def configuration (Γ₁ Γ₂ : Circle) (T A B C D X Y : Point) (ℓ₁ ℓ₂ : Line) (Ω : Circle) : Prop :=
  external_tangents Γ₁ Γ₂ ℓ₁ ℓ₂ ∧
  tangency_point ℓ₁ Γ₁ A ∧
  tangency_point ℓ₂ Γ₂ B ∧
  passes_through Ω A ∧
  passes_through Ω B ∧
  intersects Ω Γ₁ C ∧
  intersects Ω Γ₂ D ∧
  convex_quadrilateral A B C D ∧
  intersects_line AC BD X ∧
  intersects_line AD BC Y 

-- The proof statement
theorem collinear_TXY
  (h : configuration Γ₁ Γ₂ T A B C D X Y ℓ₁ ℓ₂ Ω) :
  collinear {T, X, Y} :=
sorry

end collinear_TXY_l580_580424


namespace ellipse_with_conditions_eq_l580_580599

noncomputable def standard_eq_of_ellipse (a b : ℝ) (h : a > b > 0) 
  (distance_foci : 2 * (a - b) = 1) (eccentricity : a = 2 * (a - 1)) : Bool := 
  (a = 2 ∧ b = real.sqrt 3)

theorem ellipse_with_conditions_eq :
  ∀ (x y : ℝ), (∃ a b : ℝ, a > b > 0 ∧ 
    (| 2 * (a-1)| = 2) ∧ (2 * (a-1) = a * 2) ∧ 
    ( (x^2 / 4 + y^2 / 3) = 1) ∧ 
    (
      ∀ P : ℝ × ℝ, 
      let P := (x, y);
      let F1 := (-1,0);
      let A := (-2, 0);
      let dot_product := ((-1 - x) * (-2 - x) + y^2);
      (0 ≤ dot_product ∧ dot_product ≤ 12) 
    )).

end ellipse_with_conditions_eq_l580_580599


namespace cookie_cost_is_correct_l580_580025

noncomputable def cookie_cost
    (cheeseburger_cost : ℚ)
    (milkshake_cost : ℚ)
    (coke_cost : ℚ)
    (fries_cost : ℚ)
    (tax : ℚ)
    (toby_initial_amount : ℚ)
    (toby_final_amount : ℚ)
    (number_of_cookies : ℕ)
    (total_spent : ℚ) : ℚ :=
  let total_cost_before_cookies := (2 * cheeseburger_cost) + milkshake_cost + coke_cost + fries_cost + tax in
  let total_expenditure := (toby_initial_amount - toby_final_amount) * 2 in
  (total_expenditure - total_cost_before_cookies) / number_of_cookies

theorem cookie_cost_is_correct
    (cheeseburger_cost := 3.65 : ℚ)
    (milkshake_cost := 2 : ℚ)
    (coke_cost := 1 : ℚ)
    (fries_cost := 4 : ℚ)
    (tax := 0.20 : ℚ)
    (toby_initial_amount := 15 : ℚ)
    (toby_final_amount := 7 : ℚ)
    (number_of_cookies := 3 : ℕ)
    (total_spent := 16 : ℚ) :
  cookie_cost cheeseburger_cost milkshake_cost coke_cost fries_cost tax toby_initial_amount toby_final_amount number_of_cookies total_spent = 0.50 := by
  sorry

end cookie_cost_is_correct_l580_580025


namespace find_m_for_perpendicular_vectors_l580_580235

theorem find_m_for_perpendicular_vectors :
  ∀ (m : ℝ),
  let O : ℝ × ℝ := (0, 0),
      A : ℝ × ℝ := (1, 2),
      B : ℝ × ℝ := (m, 6),
      OA : ℝ × ℝ := (1, 2),
      AB : ℝ × ℝ := (m - 1, 4) in
  (OA.fst * AB.fst + OA.snd * AB.snd = 0) → m = -7 :=
by
  intro m
  let O : ℝ × ℝ := (0, 0)
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (m, 6)
  let OA : ℝ × ℝ := (1, 2)
  let AB : ℝ × ℝ := (m - 1, 4)
  intro h
  -- Proof goes here
  sorry

end find_m_for_perpendicular_vectors_l580_580235


namespace elena_total_pens_l580_580894

theorem elena_total_pens (price_x price_y total_cost : ℝ) (num_x : ℕ) (hx1 : price_x = 4.0) (hx2 : price_y = 2.2) 
  (hx3 : total_cost = 42.0) (hx4 : num_x = 6) : 
  ∃ num_total : ℕ, num_total = 14 :=
by
  sorry

end elena_total_pens_l580_580894


namespace prove_original_sides_l580_580925

def original_parallelogram_sides (a b : ℕ) : Prop :=
  ∃ k : ℕ, (a, b) = (k * 1, k * 2) ∨ (a, b) = (1, 5) ∨ (a, b) = (4, 5) ∨ (a, b) = (3, 7) ∨ (a, b) = (4, 7) ∨ (a, b) = (3, 8) ∨ (a, b) = (5, 8) ∨ (a, b) = (5, 7) ∨ (a, b) = (2, 7)

theorem prove_original_sides (a b : ℕ) : original_parallelogram_sides a b → (1, 2) = (1, 2) :=
by
  intro h
  sorry

end prove_original_sides_l580_580925


namespace avg_speed_A_to_B_avg_speed_B_to_A_l580_580449

-- Define the basic conditions
variables (v1 v2 v3 v4 : ℝ)
variables (s : ℝ)
variables (t1 t2 t_total : ℝ)

-- Condition values
def v1 := 60
def v2 := 40
def v3 := 80
def v4 := 45

-- The values we are looking to prove
def avg_speed_AB := (v1 + v2) / 2
def avg_speed_BA := (2 * s) / (s / v3 + s / v4)

-- The average speeds we want to prove
theorem avg_speed_A_to_B : avg_speed_AB = 50 := by
  exact sorry

theorem avg_speed_B_to_A : avg_speed_BA = 57.6 := by
  exact sorry

end avg_speed_A_to_B_avg_speed_B_to_A_l580_580449


namespace sequence_terminates_final_value_l580_580723

-- Define the function Lisa uses to update the number
def f (x : ℕ) : ℕ :=
  let a := x / 10
  let b := x % 10
  a + 4 * b

-- Prove that for any initial value x0, the sequence eventually becomes periodic and ends.
theorem sequence_terminates (x0 : ℕ) : ∃ N : ℕ, ∃ j : ℕ, N ≠ j ∧ (Nat.iterate f N x0) = (Nat.iterate f j x0) :=
  by sorry

-- Given the starting value, show the sequence stabilizes at 39
theorem final_value (x0 : ℕ) (h : x0 = 53^2022 - 1) : ∃ N : ℕ, Nat.iterate f N x0 = 39 :=
  by sorry

end sequence_terminates_final_value_l580_580723


namespace parametric_to_regular_l580_580177

theorem parametric_to_regular (θ : ℝ) : 
  let x := 2 + (sin θ) ^ 2 in
  let y := (sin θ) ^ 2 in
  y = x - 2 ∧ 2 ≤ x ∧ x ≤ 3 :=
by
  let x := 2 + (sin θ) ^ 2
  let y := (sin θ) ^ 2
  have hyx : y = x - 2 := by sorry
  have hxrange : 2 ≤ x ∧ x ≤ 3 := by sorry
  exact ⟨hyx, hxrange⟩

end parametric_to_regular_l580_580177


namespace cyclic_quad_area_ratio_l580_580825

theorem cyclic_quad_area_ratio (ABC : Triangle) (A1 B1 C1 : Point) (hA1 : A1 ∈ side BC)
  (hB1 : B1 ∈ side CA) (hC1 : C1 ∈ side AB) (hcyclic : CyclicQuadrilateral ABC B1 A1 C1) :
  (area A1 B1 C1) / (area ABC) ≤ (distance B1 C1 / distance A A1)^2 := 
sorry

end cyclic_quad_area_ratio_l580_580825


namespace exists_sequence_iff_mod3_l580_580890

theorem exists_sequence_iff_mod3 (n : ℕ) (h : n ≥ 3) :
  (∃ (a : Fin (n+2) → ℝ), a ⟨n, h⟩ = a 0 ∧ a ⟨n+1, Nat.succ_le_of_lt (Nat.lt_of_lt_of_le h (Nat.le_of_eq (Nat.mod_add_div n 3).symm))⟩ = a 1 ∧ ∀ i : Fin n, a i * a ⟨i.val + 1, Nat.succ_le_succ i.is_lt⟩ + 1 = a ⟨i.val + 2, sorry⟩) ↔ n % 3 = 0 :=
by {
  sorry
}

end exists_sequence_iff_mod3_l580_580890


namespace largest_of_eight_consecutive_l580_580406

theorem largest_of_eight_consecutive (n : ℕ) (h : 8 * n + 28 = 2024) : n + 7 = 256 := by
  -- This means you need to solve for n first, then add 7 to get the largest number
  sorry

end largest_of_eight_consecutive_l580_580406


namespace cross_product_correct_l580_580872

open Matrix

def u : Fin 3 → ℝ
| 0 => 3
| 1 => 2
| 2 => 4

def v : Fin 3 → ℝ
| 0 => 4
| 1 => 3
| 2 => -1

def cross_product (a b : Fin 3 → ℝ) : Fin 3 → ℝ :=
λ i, match i with
| 0 => a 1 * b 2 - a 2 * b 1
| 1 => a 2 * b 0 - a 0 * b 2
| 2 => a 0 * b 1 - a 1 * b 0

theorem cross_product_correct :
  cross_product u v = 
  λ i, match i with
       | 0 => -14
       | 1 => 19
       | 2 => 1 :=
  by
    sorry

end cross_product_correct_l580_580872


namespace rectangular_garden_width_l580_580821

theorem rectangular_garden_width
  (w : ℝ)
  (h₁ : ∃ l, l = 3 * w)
  (h₂ : ∃ A, A = l * w ∧ A = 507) : 
  w = 13 :=
by
  sorry

end rectangular_garden_width_l580_580821


namespace prime_or_four_no_square_div_factorial_l580_580903

theorem prime_or_four_no_square_div_factorial (n : ℕ) :
  (n * n ∣ n!) = false ↔ Nat.Prime n ∨ n = 4 := by
  sorry

end prime_or_four_no_square_div_factorial_l580_580903


namespace linear_function_difference_l580_580710

variable (g : ℝ → ℝ)
variable (h_linear : ∀ x y, g (x + y) = g x + g y)
variable (h_value : g 8 - g 4 = 16)

theorem linear_function_difference : g 16 - g 4 = 48 := by
  sorry

end linear_function_difference_l580_580710


namespace angle_between_the_planes_l580_580826

noncomputable def angle_between_planes 
(plan1 plan2 : ℝ × ℝ × ℝ × ℝ) (n1 n2 : ℝ × ℝ × ℝ) : ℝ :=
  let cos_phi := (n1.1 * n2.1 + n1.2 * n2.2 + n1.3 * n2.3) / 
                 (Real.sqrt (n1.1^2 + n1.2^2 + n1.3^2) * Real.sqrt (n2.1^2 + n2.2^2 + n2.3^2)) in
  Real.arccos cos_phi

def plane1 : ℝ × ℝ × ℝ × ℝ := (3, 1, 1, -4)
def plane2 : ℝ × ℝ × ℝ × ℝ := (0, 1, 1, 5)

def normal_vector (plane : ℝ × ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (plane.1, plane.2, plane.3)

noncomputable def angle_between_specific_planes : ℝ := angle_between_planes plane1 plane2 (normal_vector plane1) (normal_vector plane2)

theorem angle_between_the_planes :
  angle_between_specific_planes = Real.arccos (Real.sqrt (2 / 11)) :=
sorry

end angle_between_the_planes_l580_580826


namespace no_six_odd_numbers_sum_to_one_l580_580511

theorem no_six_odd_numbers_sum_to_one (a b c d e f : ℕ)
  (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) (hd : d % 2 = 1) (he : e % 2 = 1) (hf : f % 2 = 1)
  (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f) :
  (1 / a : ℝ) + 1 / b + 1 / c + 1 / d + 1 / e + 1 / f ≠ 1 :=
by
  sorry

end no_six_odd_numbers_sum_to_one_l580_580511


namespace min_value_xy_l580_580234

theorem min_value_xy (x y : ℝ) (h : 1 / x + 2 / y = Real.sqrt (x * y)) : x * y ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_value_xy_l580_580234


namespace part1_inequality_part2_range_of_a_l580_580631

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 1)

-- Part (1)
theorem part1_inequality (x : ℝ) (h : f x 2 < 5) : -2 < x ∧ x < 3 := sorry

-- Part (2)
theorem part2_range_of_a (x a : ℝ) (h : ∀ x, f x a ≥ 4 - abs (a - 1)) : a ≤ -2 ∨ a ≥ 2 := sorry

end part1_inequality_part2_range_of_a_l580_580631


namespace t_shaped_tiling_impossible_l580_580361

theorem t_shaped_tiling_impossible : 
  ∀ (board : Fin 10 × Fin 10 → bool), 
  (∀ (piece : Fin 4 → (Fin 10 × Fin 10)), 
     (∀ i j, board (piece i) = board (piece j) ↔ i = j) ) → False := 
by 
  sorry

end t_shaped_tiling_impossible_l580_580361


namespace similar_triangles_height_ratio_l580_580069

theorem similar_triangles_height_ratio (area_ratio : ℝ) (h₁ : ℝ) (h₂ : ℝ) 
  (similar : Boolean) (h₁_value : h₁ = 5) (area_ratio_value : area_ratio = 9) :
  similar = true → area_ratio = (h₂ / h₁) ^ 2 → h₂ = 15 :=
by
  intro h_similar area_eq
  rw [h₁_value, area_ratio_value]
  sorry

end similar_triangles_height_ratio_l580_580069


namespace range_of_m_l580_580220

theorem range_of_m (m : ℝ) :
  (∀ x1 ∈ set.Icc (-1 : ℝ) 3, ∃ x2 ∈ set.Icc (0 : ℝ) 2, (x1^2 : ℝ) ≥ (1/2)^x2 - m) ↔ m ∈ set.Ici (1/4 : ℝ) :=
by
  sorry

end range_of_m_l580_580220


namespace words_per_page_l580_580751

-- Definitions based on conditions
def typing_speed : ℕ := 50 -- words per minute
def pages : ℕ := 5 -- number of pages
def water_per_hour : ℝ := 15 -- ounces of water per hour
def total_water : ℝ := 10 -- total ounces of water

-- Total words calculated from the conditions
def total_words := typing_speed * (total_water / water_per_hour * 60) -- words

-- Statement: each page contains 400 words given the conditions
theorem words_per_page : (total_words / ↑pages = 400) := by
  sorry

end words_per_page_l580_580751


namespace sum_of_angles_l580_580416

noncomputable def satisfies_conditions (z : ℂ) : Prop :=
  z^40 - z^10 - 1 = 0 ∧ abs z = 1

noncomputable def angle_conditions (θ : ℕ → ℕ) : Prop :=
  ∀ n, 0 ≤ θ n ∧ θ n < 360 ∧ ∀ i j, i < j → θ i < θ j

theorem sum_of_angles {z : ℂ} {θ : ℕ → ℕ} (h1 : satisfies_conditions z) (h2 : angle_conditions θ) :
  (∑ i in finset.range(20).filter (λ i, i % 2 = 1), θ i) = 2100 :=
sorry -- This is where the proof would go

end sum_of_angles_l580_580416


namespace milton_probability_l580_580726

/-- Milton starts at 0 on the real number line and tosses a fair coin 10 times. 
For every head, he moves 1 unit in the positive direction, 
and for every tail, he moves 1 unit in the negative direction. 
We need to prove that the probability that he reaches exactly 6 units 
in the positive direction at any time without ever hitting -2 units during this 
process is 0 (in simplest fraction form with a sum of numerator and denominator as 1). -/
theorem milton_probability : 
  let p := (0 : ℚ) / (1024 : ℚ) in p = 0 :=
by 
  sorry

end milton_probability_l580_580726


namespace height_of_equilateral_triangle_area_of_equilateral_triangle_l580_580563

-- Define an equilateral triangle with side length a
def equilateral_triangle (a : ℝ) : Prop :=
∀ (x y z : ℝ), x = y ∧ y = z ∧ x = a

-- Theorem for height of equilateral triangle
theorem height_of_equilateral_triangle (a : ℝ) (h : ℝ) :
  (equilateral_triangle a) → h = (a * sqrt 3) / 2 :=
sorry

-- Theorem for area of equilateral triangle
theorem area_of_equilateral_triangle (a : ℝ) (A : ℝ) :
  (equilateral_triangle a) → A = (sqrt 3 / 4) * a^2 :=
sorry

end height_of_equilateral_triangle_area_of_equilateral_triangle_l580_580563


namespace broadcasting_methods_count_l580_580835

-- Definitions based on the conditions.
def commercials : Set String := {"C1", "C2", "C3"}
def olympics : Set String := {"O1", "O2"}
def ends_with_olympic (s : List String) : Prop := s.last ∈ olympics
def no_consecutive_olympics (s : List String) : Prop :=
  ∀ i, i < s.length - 1 → (s.nth i ∈ olympics → s.nth (i + 1) ∉ olympics)

-- Main theorem statement based on the proof goal.
theorem broadcasting_methods_count :
  ∃ (seqs : Set (List String)), 
    ∀ s ∈ seqs, ends_with_olympic s ∧ no_consecutive_olympics s ∧ s.length = 5 ∧
    s.to_set.subset (commercials ∪ olympics) ∧
    seqs.size = 36 := sorry

end broadcasting_methods_count_l580_580835


namespace investment_value_after_two_weeks_l580_580359

theorem investment_value_after_two_weeks (initial_investment : ℝ) (gain_first_week : ℝ) (gain_second_week : ℝ) : 
  initial_investment = 400 → 
  gain_first_week = 0.25 → 
  gain_second_week = 0.5 → 
  ((initial_investment * (1 + gain_first_week) * (1 + gain_second_week)) = 750) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  calc
    400 * (1 + 0.25) * (1 + 0.5)
    = 400 * 1.25 * 1.5 : by ring
    = 750 : by norm_num

end investment_value_after_two_weeks_l580_580359


namespace find_coefficients_l580_580199

noncomputable def P (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^4 - 8 * a * x^3 + b * x^2 - 32 * c * x + 16 * c

theorem find_coefficients (a b c : ℝ) (h : a ≠ 0) :
  (∃ x1 x2 x3 x4 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ x4 > 0 ∧ P a b c x1 = 0 ∧ P a b c x2 = 0 ∧ P a b c x3 = 0 ∧ P a b c x4 = 0) →
  (b = 16 * a ∧ c = a) :=
by
  sorry

end find_coefficients_l580_580199


namespace min_value_a2_b2_l580_580279

theorem min_value_a2_b2 (a b : ℝ) (h : binom_exp_coeff (a x^2 + b / x)^6 3 = 20) : (a^2 + b^2) = 2 := by
  sorry

end min_value_a2_b2_l580_580279


namespace half_angle_quadrant_l580_580644

variables {α : ℝ} {k : ℤ} {n : ℤ}

theorem half_angle_quadrant (h : ∃ k : ℤ, k * 360 + 180 < α ∧ α < k * 360 + 270) :
  ∃ (n : ℤ), (n * 360 + 90 < α / 2 ∧ α / 2 < n * 360 + 135) ∨ 
      (n * 360 + 270 < α / 2 ∧ α / 2 < n * 360 + 315) :=
by sorry

end half_angle_quadrant_l580_580644


namespace odd_function_property_l580_580610

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x * Real.log (2 - x) else -x * Real.log (2 + x)

theorem odd_function_property (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_neg_interval : ∀ x : ℝ, x < 0 → f x = -x * Real.log (2 - x)) :
  ∀ x : ℝ, f x = (if x < 0 then -x * Real.log (2 - x) else -x * Real.log (2 + x)) :=
by
  sorry

end odd_function_property_l580_580610


namespace ben_apples_difference_l580_580519

theorem ben_apples_difference (B P T : ℕ) (h1 : P = 40) (h2 : T = 18) (h3 : (3 / 8) * B = T) :
  B - P = 8 :=
sorry

end ben_apples_difference_l580_580519


namespace regular_polygon_interior_angle_l580_580851

theorem regular_polygon_interior_angle (S : ℝ) (n : ℕ) (h1 : S = 720) (h2 : (n - 2) * 180 = S) : 
  (S / n) = 120 := 
by
  sorry

end regular_polygon_interior_angle_l580_580851


namespace similar_triangles_height_l580_580041

open_locale classical

theorem similar_triangles_height
  (h_small : ℝ)
  (A_small A_large : ℝ)
  (area_ratio : ℝ)
  (h_large : ℝ) :
  A_small / A_large = 1 / 9 →
  h_small = 5 →
  area_ratio = A_large / A_small →
  area_ratio = 9 →
  h_large = h_small * sqrt area_ratio →
  h_large = 15 :=
by
  sorry

end similar_triangles_height_l580_580041


namespace max_expr_value_l580_580324

noncomputable def max_expr (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) : ℝ :=
  real.sup (set.range (λ x, 2 * (a - x) * (x + c * real.sqrt (x^2 + b^2))))

theorem max_expr_value (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  max_expr a b c h_a h_b h_c = a^2 + c^2 * b^2 :=
sorry

end max_expr_value_l580_580324


namespace calc_expression_l580_580525

theorem calc_expression : 
  (abs (Real.sqrt 2 - Real.sqrt 3) + 2 * Real.cos (Real.pi / 4) - Real.sqrt 2 * Real.sqrt 6 = -Real.sqrt 3) :=
by
  -- Given that sqrt(3) > sqrt(2)
  have h1 : Real.sqrt 3 > Real.sqrt 2 := by sorry
  -- And cos(45°) = sqrt(2)/2
  have h2 : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  -- Now prove the expression equivalency
  sorry

end calc_expression_l580_580525


namespace sticker_ratio_l580_580542

variable (Dan Tom Bob : ℕ)

theorem sticker_ratio 
  (h1 : Dan = 2 * Tom) 
  (h2 : Tom = Bob) 
  (h3 : Bob = 12) 
  (h4 : Dan = 72) : 
  Tom = Bob :=
by
  sorry

end sticker_ratio_l580_580542


namespace evaluate_logarithmic_expression_l580_580191

theorem evaluate_logarithmic_expression : 5 ^ (Real.log 13 / Real.log 5) = 13 :=
by sorry

end evaluate_logarithmic_expression_l580_580191


namespace sent_away_correct_l580_580268

def stones_sent_away (original kept sent : ℕ) : Prop :=
  original - kept = sent

theorem sent_away_correct :
  ∀ (original kept sent : ℕ),
  original = 78 →
  kept = 15 →
  stones_sent_away original kept sent →
  sent = 63 :=
by
  intros original kept sent h_orig h_kept h_eq
  rw [h_orig, h_kept] at h_eq
  exact h_eq

end sent_away_correct_l580_580268


namespace keys_and_escape_l580_580776

theorem keys_and_escape (m n : ℕ) : (∃ path : list (ℕ × ℕ), 
  (∀ (x y : ℕ × ℕ), 
    (x ∈ path ∧ y ∈ path → 
    (x.fst = y.fst ∧ (x.snd = y.snd + 1 ∨ x.snd = y.snd - 1) ∨ 
    (x.snd = y.snd ∧ (x.fst = y.fst + 1 ∨ x.fst = y.fst - 1)))) ∧
    ∀ room : ℕ × ℕ, room ∈ path → 1 ≤ room.fst ∧ room.fst ≤ m ∧ 1 ≤ room.snd ∧ room.snd ≤ n ∧
    ∀ (x y : ℕ × ℕ), x ∈ path ∧ y ∈ path → x ≠ y) ↔ (m % 2 = 1 ∨ n % 2 = 1))
:= sorry

end keys_and_escape_l580_580776


namespace count_values_not_dividing_h_l580_580326

def is_proper_divisor (d n : ℕ) : Prop := d ∣ n ∧ d ≠ n ∧ d ≠ 0

def h (n : ℕ) : ℕ := ∑ d in Finset.filter (λ d, is_proper_divisor d n) (Finset.range (n + 1)), d

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_square_of_prime (n : ℕ) : Prop := ∃ p : ℕ, is_prime p ∧ p * p = n

def values_n_not_dividing_h (n : ℕ) : Prop := n ∈ Finset.range 61 ∧ 3 ≤ n ∧ n ≠ 1 ∧ ¬ n ∣ h n

theorem count_values_not_dividing_h : 
  Finset.card (Finset.filter values_n_not_dividing_h (Finset.range 61)) = 19 := by 
  sorry

end count_values_not_dividing_h_l580_580326


namespace range_real_part_of_z_range_norm_of_z_plus_3_conj_z_l580_580222

noncomputable theory

variables (z : ℂ)

def is_imaginary (z : ℂ) : Prop := Im z ≠ 0
def is_real (x : ℂ) : Prop := Im x = 0

theorem range_real_part_of_z (h_imag : is_imaginary z)
  (h_real : is_real (z / (1 + 1 * complex.I)))
  (h_bound : -1 < (z / (1 + 1 * complex.I)).re ∧ (z / (1 + 1 * complex.I)).re < 2) :
  -1 < z.re ∧ z.re < 2 :=
sorry

theorem range_norm_of_z_plus_3_conj_z (h_imag : is_imaginary z)
  (h_real : is_real (z / (1 + 1 * complex.I)))
  (h_bound : -1 < (z / (1 + 1 * complex.I)).re ∧ (z / (1 + 1 * complex.I)).re < 2) :
  0 < complex.norm (z + 3 * conj z) ∧ complex.norm (z + 3 * conj z) < 4 * real.sqrt 5 :=
sorry

end range_real_part_of_z_range_norm_of_z_plus_3_conj_z_l580_580222


namespace prob_both_boys_is_correct_prob_one_girl_is_correct_prob_at_least_one_girl_is_correct_l580_580578

section speech_contest

variables (boys girls : ℕ) (total_people : ℕ) (both_boys one_girl at_least_one_girl : ℝ)

-- Condition: There are 3 boys and 2 girls
def num_boys : ℕ := 3
def num_girls : ℕ := 2
def total_people := num_boys + num_girls

-- 1. Prove the probability that both selected people are boys is 3/10
def prob_both_boys : ℝ := 3 / 10

-- 2. Prove the probability that one of the selected people is a girl is 3/5
def prob_one_girl : ℝ := 3 / 5

-- 3. Prove the probability that at least one of the selected people is a girl is 7/10
def prob_at_least_one_girl : ℝ := 7 / 10

theorem prob_both_boys_is_correct : 
  (boys = 3) → (girls = 2) → total_people = boys + girls → 
  (both_boys = (3 / 10)) :=
by sorry

theorem prob_one_girl_is_correct : 
  (boys = 3) → (girls = 2) → total_people = boys + girls → 
  (one_girl = (3 / 5)) :=
by sorry

theorem prob_at_least_one_girl_is_correct : 
  (boys = 3) → (girls = 2) → total_people = boys + girls → 
  (at_least_one_girl = (7 / 10)) :=
by sorry

end speech_contest

end prob_both_boys_is_correct_prob_one_girl_is_correct_prob_at_least_one_girl_is_correct_l580_580578


namespace find_octahedron_side_length_l580_580476

noncomputable def cube_vertices : list (ℝ × ℝ × ℝ) :=
  [(-1, -1, -1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1),
   (1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, 1, 1)]

/-- Conditions on the vertices of the regular octahedron placed at midpoints of the edges of the cube. -/
def is_vertex_of_octahedron (v : ℝ × ℝ × ℝ) :=
  ∃ (t : ℝ), (v = (-1+2*t, -1, -1) ∨ v = (-1, -1+2*t, -1) ∨ v = (-1, -1, -1+2*t) ∨
            v = (1-2*t, 1, 1) ∨ v = (1, 1-2*t, 1) ∨ v = (1, 1, 1-2*t))

/-- The distance between two points in ℝ³ -/
def dist (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  (real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2))

theorem find_octahedron_side_length : 
  let vertices := list.filter is_vertex_of_octahedron cube_vertices in
  ∀ v1 v2 ∈ vertices, v1 ≠ v2 → dist v1 v2 = 2 * real.sqrt 2 :=
by sorry

end find_octahedron_side_length_l580_580476


namespace candidates_count_l580_580816

theorem candidates_count (n : ℕ) (h : n * (n - 1) = 72) : n = 9 := 
sorry

end candidates_count_l580_580816


namespace carlton_outfits_l580_580535

theorem carlton_outfits (button_up_shirts sweater_vests : ℕ) 
  (h1 : sweater_vests = 2 * button_up_shirts)
  (h2 : button_up_shirts = 3) :
  sweater_vests * button_up_shirts = 18 :=
by
  sorry

end carlton_outfits_l580_580535


namespace average_length_of_two_strings_l580_580733

theorem average_length_of_two_strings (a b : ℝ) (h1 : a = 3.2) (h2 : b = 4.8) :
  (a + b) / 2 = 4.0 :=
by
  sorry

end average_length_of_two_strings_l580_580733


namespace victor_total_stickers_l580_580431

/--
Victor has 12 flower stickers, 8 animal stickers, and 3 fewer insect stickers than animal stickers.
He also has 7 more space stickers than flower stickers. How many stickers does Victor have in total?
-/
theorem victor_total_stickers :
  let flower_stickers := 12
  let animal_stickers := 8
  let insect_stickers := animal_stickers - 3
  let space_stickers := flower_stickers + 7
  flower_stickers + animal_stickers + insect_stickers + space_stickers = 44 :=
by
  intros
  let flower_stickers := 12
  let animal_stickers := 8
  let insect_stickers := animal_stickers - 3
  let space_stickers := flower_stickers + 7
  have h_total := flower_stickers + animal_stickers + insect_stickers + space_stickers
  show h_total = 44
  sorry

end victor_total_stickers_l580_580431


namespace scientific_notation_exponent_0point0000502_l580_580640

theorem scientific_notation_exponent_0point0000502 :
  (∃ n : ℤ, 0.0000502 = 5.02 * 10^n) ∧ (∀ n : ℤ, 0.0000502 = 5.02 * 10^n → n = -4) :=
begin
  use -4,
  split,
  { sorry },
  { intros n h,
    sorry }
end

end scientific_notation_exponent_0point0000502_l580_580640


namespace conclusion_l580_580978

variables (y : ℝ → ℝ) (x : ℝ)

-- Definitions from conditions
def is_periodic (f : ℝ → ℝ) : Prop := ∃ T > 0, ∀ x, f (x + T) = f x
def is_trigonometric (f : ℝ → ℝ) : Prop := f = cos

-- Hypotheses from conditions
axiom h1 : is_trigonometric y
axiom h2 : ∀ f, is_trigonometric f → is_periodic f

-- Conclusion that needs to be proved
theorem conclusion : is_periodic y :=
by
  exact h2 y h1

end conclusion_l580_580978


namespace BP_eq_CP_l580_580330

variables {α : Type*} [linear_ordered_field α]

/-- Definitions and Constants for the Problem -/
structure Triangle :=
  (A B C L P : α)
  (circumcircle : set α)

def is_bisector (AL : α) (BAC : α) : Prop := 
  ∀ (A B C : α), AL = BAC / 2

def is_circumcircle (ω : set α) (A B C : α) : Prop :=
  ∀ (x : α), x ∈ ω → x ∈ {A, B, C}

/-- Theorem Statement -/
theorem BP_eq_CP 
  (T : Triangle) 
  (Hbisector : is_bisector T.L T.A) 
  (Hcircumcircle : is_circumcircle T.circumcircle T.A T.B T.C) 
  (Haltitude : ∃ H : α, (T.B - H) = 0)
  (Hextension : ∃ P : α, P ∈ T.circumcircle ∧ P ≠ T.B)
  (Hgiven_angle : T.L = 2 * T.A) :
  T.B = T.P ∧ T.C = T.P :=
begin
  sorry
end

end BP_eq_CP_l580_580330


namespace min_period_sin_cos_l580_580769

theorem min_period_sin_cos (x : ℝ) : 
  ∃ T > 0, (∀ x, ((sin x + cos x) ^ 2 + 1 = (sin (x + T) + cos (x + T)) ^ 2 + 1) ∧
  (∀ T' > 0, (∀ x, ((sin x + cos x) ^ 2 + 1 = (sin (x + T') + cos (x + T')) ^ 2 + 1) → T ≤ T')) :=
sorry

end min_period_sin_cos_l580_580769


namespace binom_12_9_plus_binom_12_3_l580_580554

theorem binom_12_9_plus_binom_12_3 : (Nat.choose 12 9) + (Nat.choose 12 3) = 440 := by
  sorry

end binom_12_9_plus_binom_12_3_l580_580554


namespace similar_triangles_height_l580_580051

theorem similar_triangles_height (h₁ h₂ : ℝ) (ratio_areas : ℝ) 
  (h₁_eq : h₁ = 5) (ratio_areas_eq : ratio_areas = 1 / 9)
  (similar : h₂^2 = (√ratio_areas)^2 * h₁^2) : h₂ = 15 :=
by {
  have ratio_areas_pos : ratio_areas > 0 := by (simp [ratio_areas_eq]),
  have k := √ratio_areas,
  have k_eq : k = 3 := by {
    rw [ratio_areas_eq, sqrt_div, sqrt_one, sqrt_nat_eq_iff_eq_sq, one_div_eq_inv] at *,
    norm_num },
  have h₂_def : h₂ = 3 * h₁ := by rw [h₁_eq, mul_comm, k_eq],
  rw [h₂_def],
  norm_num,
}

end similar_triangles_height_l580_580051


namespace problem_1_problem_2_l580_580635

-- Given triangle ABC with circumcircle O (radius R)
variables {A B C O I E : Point}
variable {R : ℝ}

-- Angle measures
axiom angle_B_60 : angle B = 60
axiom angle_A_lt_angle_C : angle A < angle C

-- Definitions and conditions about incenter I 
axiom incenter_I : is_incenter I A B C

-- External angle bisector of angle A intersects circumcircle at E
axiom external_angle_bisector_of_A_intersects_at_E :
  external_bisector A ∩ circumcircle O = {E}

-- Prove: IO = AE
theorem problem_1 (h1 : angle B = 60) (h2 : angle A < angle C)
                 (h3 : is_incenter I triangle_ABC) (h4 : external_bisector A ∩ circumcircle O = {E})
                 : dist I O = dist A E := sorry

-- Prove: 2R < IO + IA + IC < (1 + √3)R
theorem problem_2 (h1 : angle B = 60) (h2 : angle A < angle C)
                (h3 : is_incenter I triangle_ABC) (h4 : external_bisector A ∩ circumcircle O = {E})
                : 2 * R < dist I O + dist I A + dist I C ∧ dist I O + dist I A + dist I C < (1 + Real.sqrt 3) * R := sorry

end problem_1_problem_2_l580_580635


namespace spotted_and_fluffy_cats_l580_580502

theorem spotted_and_fluffy_cats (total_cats : ℕ) (h1 : total_cats = 120)
    (fraction_spotted : ℚ) (h2 : fraction_spotted = 1/3)
    (fraction_fluffy_of_spotted : ℚ) (h3 : fraction_fluffy_of_spotted = 1/4) :
    (total_cats * fraction_spotted * fraction_fluffy_of_spotted).toNat = 10 := by
  sorry

end spotted_and_fluffy_cats_l580_580502


namespace max_cos_sin_expr_l580_580181

theorem max_cos_sin_expr : ∃ θ, (0 ≤ θ ∧ θ ≤ Real.pi / 2 ∧ cos (θ / 2) * (1 + sin θ) = 1) :=
sorry

end max_cos_sin_expr_l580_580181


namespace nuts_in_tree_l580_580015

def num_squirrels := 4
def num_nuts := 2

theorem nuts_in_tree :
  ∀ (S N : ℕ), S = num_squirrels → S - N = 2 → N = num_nuts :=
by
  intros S N hS hDiff
  sorry

end nuts_in_tree_l580_580015


namespace width_of_bulletin_board_l580_580849

theorem width_of_bulletin_board (area length : ℝ) (h_area : area = 6400) (h_length : length = 160) :
  ∃ width : ℝ, width = 40 :=
by
  have h_width : width = area / length := sorry
  rw [h_area, h_length] at h_width
  exact ⟨40, h_width⟩

end width_of_bulletin_board_l580_580849


namespace three_digit_numbers_perfect_square_l580_580270

theorem three_digit_numbers_perfect_square : 
  {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m: ℕ, m^2 = n^3 - n^2}.to_finset.card = 22 :=
by
  sorry

end three_digit_numbers_perfect_square_l580_580270


namespace unique_function_satisfying_properties_l580_580365

noncomputable def S := { x : ℝ | x ≠ 0 }

theorem unique_function_satisfying_properties (f : S → ℝ) :
  (∀ x : S, f x = x * f (1 / x)) ∧ (∀ x y : S, x + y ≠ 0 → f (x + y) = f x + f y - 1) →
  (∀ x : S, f x = x + 1) :=
sorry

end unique_function_satisfying_properties_l580_580365


namespace max_min_difference_l580_580619

theorem max_min_difference : 
  ∀ (a b c : ℝ), a^2 + b^2 + c^2 = 1 → 
  let M := max (ab + bc + ca) and m := min (ab + bc + ca) in 
  M - m = 3 / 2 :=
begin
  intro a,
  intro b,
  intro c,
  intro h,
  sorry
end

end max_min_difference_l580_580619


namespace valid_factorizations_of_1870_l580_580639

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_valid_factor1 (n : ℕ) : Prop := 
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ n = p1 * p2

def is_valid_factor2 (n : ℕ) : Prop := 
  ∃ (p k : ℕ), is_prime p ∧ (k = 4 ∨ k = 6 ∨ k = 8 ∨ k = 9) ∧ n = p * k

theorem valid_factorizations_of_1870 : 
  ∃ a b : ℕ, a * b = 1870 ∧ 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ 
  ((is_valid_factor1 a ∧ is_valid_factor2 b) ∨ (is_valid_factor1 b ∧ is_valid_factor2 a)) ∧ 
  (a = 34 ∧ b = 55 ∨ a = 55 ∧ b = 34) ∧ 
  (¬∃ x y : ℕ, x * y = 1870 ∧ 10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 ∧ 
  ((is_valid_factor1 x ∧ is_valid_factor2 y) ∨ (is_valid_factor1 y ∧ is_valid_factor2 x)) ∧ 
  (x ≠ 34 ∨ y ≠ 55 ∨ x ≠ 55 ∨ y ≠ 34)) :=
sorry

end valid_factorizations_of_1870_l580_580639


namespace find_f_elog5_l580_580971

def f : ℝ → ℝ
| x := if x <= 0 then 2 ^ x else f(x - 3)

theorem find_f_elog5 : f (Real.exp (Real.log 5)) = 1 / 2 := by
  sorry

end find_f_elog5_l580_580971


namespace part1_part2_l580_580980

section 
variable {a b : ℚ}

-- Define the new operation as given in the condition
def odot (a b : ℚ) : ℚ := a * (a + b) - 1

-- Prove the given results
theorem part1 : odot 3 (-2) = 2 :=
by
  -- Proof omitted
  sorry

theorem part2 : odot (-2) (odot 3 5) = -43 :=
by
  -- Proof omitted
  sorry

end

end part1_part2_l580_580980


namespace part_a_part_b_l580_580568

def M (i : Nat) : Nat :=
  if Nat.popcount i % 2 = 0 then 0 else 1

theorem part_a : (Finset.filter (λ i, M i = M (i + 1)) (Finset.range 1000)).card ≥ 320 := sorry

theorem part_b : (Finset.filter (λ i, M i = M (i + 7)) (Finset.range 1000000)).card ≥ 450000 := sorry

end part_a_part_b_l580_580568


namespace angle_equality_l580_580176

variables {A B C D E F : Type*}
variables [is_parallelogram A B C D] [midpoint E B C] [point_on_segment F D E] [perpendicular AF DE]

theorem angle_equality :
  ∠ C D E = ∠ E F B :=
sorry

end angle_equality_l580_580176


namespace x_ge_y_l580_580590

variable (a : ℝ)

def x : ℝ := 2 * a * (a + 3)
def y : ℝ := (a - 3) * (a + 3)

theorem x_ge_y : x a ≥ y a := 
by 
  sorry

end x_ge_y_l580_580590


namespace dennis_pants_purchase_l580_580888

theorem dennis_pants_purchase
  (pants_cost : ℝ) 
  (pants_discount : ℝ) 
  (socks_cost : ℝ) 
  (socks_discount : ℝ) 
  (socks_quantity : ℕ)
  (total_spent : ℝ)
  (discounted_pants_cost : ℝ)
  (discounted_socks_cost : ℝ)
  (pants_quantity : ℕ) :
  pants_cost = 110.00 →
  pants_discount = 0.30 →
  socks_cost = 60.00 →
  socks_discount = 0.30 →
  socks_quantity = 2 →
  total_spent = 392.00 →
  discounted_pants_cost = pants_cost * (1 - pants_discount) →
  discounted_socks_cost = socks_cost * (1 - socks_discount) →
  total_spent = socks_quantity * discounted_socks_cost + pants_quantity * discounted_ppants_cost →
  pants_quantity = 4 :=
by
  intros
  sorry

end dennis_pants_purchase_l580_580888


namespace perpendicularity_proof_l580_580218

-- Definitions of geometric entities and properties
variable (Plane Line : Type)
variable (α β : Plane) -- α and β are planes
variable (m n : Line) -- m and n are lines

-- Geometric properties and relations
variable (subset : Line → Plane → Prop) -- Line is subset of plane
variable (perpendicular : Line → Plane → Prop) -- Line is perpendicular to plane
variable (line_perpendicular : Line → Line → Prop) -- Line is perpendicular to another line

-- Conditions
axiom planes_different : α ≠ β
axiom lines_different : m ≠ n
axiom m_in_beta : subset m β
axiom n_in_beta : subset n β

-- Proof problem statement
theorem perpendicularity_proof :
  (subset m α) → (perpendicular n α) → (line_perpendicular n m) :=
by
  sorry

end perpendicularity_proof_l580_580218


namespace rectangular_garden_width_l580_580767

theorem rectangular_garden_width (w : ℕ) (h1 : ∃ l : ℕ, l = 3 * w) (h2 : w * (3 * w) = 507) : w = 13 := 
by 
  sorry

end rectangular_garden_width_l580_580767


namespace problem_l580_580779

def a (n : ℕ) : ℚ := 1 / (n + 1) - 1 / (n + 2)

def S (n : ℕ) : ℚ := (Finset.range n).sum (λ k, a k)

theorem problem (n : ℕ) : S 6 = 6 / 7 := sorry

end problem_l580_580779


namespace geometric_series_sum_eq_l580_580192

open_locale big_operators

/-- Definition of an infinite geometric series -/
def infinite_geometric_series_sum (a r : ℚ) (h : abs r < 1) : ℚ :=
a / (1 - r)

/-- The first term of the series -/
def a : ℚ := 5 / 3

/-- The common ratio of the series -/
def r : ℚ := -1 / 4

/-- Verification that the sum of the infinite geometric series with the given first term and common ratio is 4/3 -/
theorem geometric_series_sum_eq :
  infinite_geometric_series_sum a r (by norm_num) = 4 / 3 :=
sorry

end geometric_series_sum_eq_l580_580192


namespace magnitude_of_z_l580_580592

noncomputable def z : ℂ := (2 - I) / (1 + I)^2

theorem magnitude_of_z : |z| = (Real.sqrt 5) / 2 :=
by
  have h1 : (1 + I)^2 = 2 * I := by norm_num
  have h2 : (2 - I) / (1 + I)^2 = -1/2 - I := by
    rw [h1]
    field_simp
    ring
  rw [h2]
  norm_num
  sorry

end magnitude_of_z_l580_580592


namespace probability_one_of_A_or_B_selected_l580_580473

-- Define the set of all groups
def groups : set string := {"A", "B", "C", "D"}

-- Define the function to count the number of ways to choose 2 groups from 4
def num_combinations (n k : ℕ) : ℕ := nat.choose n k

-- Define the set of combinations where exactly one of A and B is selected
def favorable_combinations : set (set string) :=
  {{"A", "C"}, {"A", "D"}, {"B", "C"}, {"B", "D"}}

-- Define the probability function
def probability := (favorable_combinations.to_finset.card : ℚ) / (num_combinations 4 2 : ℚ)

-- State the theorem
theorem probability_one_of_A_or_B_selected :
  probability = 2 / 3 :=
by
  sorry

end probability_one_of_A_or_B_selected_l580_580473


namespace height_of_larger_triangle_l580_580062

-- Definitions from the conditions
variables (height_small height_large : ℝ)
variables (area_ratio : ℝ)
variables (k : ℝ)

-- Given conditions
def triangles_similar : Prop := area_ratio = 9
def height_small_defined : Prop := height_small = 5
def scale_factor : Prop := k = real.sqrt area_ratio

-- Proof problem statement
theorem height_of_larger_triangle
  (h_similar : triangles_similar)
  (h_height_small : height_small_defined)
  (h_scale_factor : scale_factor) :
  height_large = 15 := sorry

end height_of_larger_triangle_l580_580062


namespace length_RT_in_trapezoid_l580_580674

open Trapezoid

theorem length_RT_in_trapezoid (PQ RS PR RT : ℝ) (T : Point)
  (h1 : PQ = 3 * RS)
  (h2 : PR = 18)
  (h3 : IntersectsAt PR RS PQ T) : RT = 6 := sorry

end length_RT_in_trapezoid_l580_580674


namespace carlton_outfit_count_l580_580528

-- Definitions of conditions
def sweater_vests (s : ℕ) : ℕ := 2 * s
def button_up_shirts : ℕ := 3
def outfits (v s : ℕ) : ℕ := v * s

-- Theorem statement
theorem carlton_outfit_count : outfits (sweater_vests button_up_shirts) button_up_shirts = 18 :=
by
  sorry

end carlton_outfit_count_l580_580528


namespace johns_average_speed_l580_580694

-- Definitions based on conditions
def cycling_distance_uphill := 3 -- in km
def cycling_time_uphill := 45 / 60 -- in hr (45 minutes)

def cycling_distance_downhill := 3 -- in km
def cycling_time_downhill := 15 / 60 -- in hr (15 minutes)

def walking_distance := 2 -- in km
def walking_time := 20 / 60 -- in hr (20 minutes)

-- Definition for total distance traveled
def total_distance := cycling_distance_uphill + cycling_distance_downhill + walking_distance

-- Definition for total time spent traveling
def total_time := cycling_time_uphill + cycling_time_downhill + walking_time

-- Definition for average speed
def average_speed := total_distance / total_time

-- Proof statement
theorem johns_average_speed : average_speed = 6 := by
  sorry

end johns_average_speed_l580_580694


namespace exists_problem_solution_l580_580917

noncomputable def exists_distinct_positive_integers (n : ℕ) (hn : 1 < n) : Prop :=
  ∃ (a b : Fin n → ℕ),
    (∀ i j : Fin n, i ≠ j → a i ≠ a j ∧ b i ≠ b j ∧ a i ≠ b j) ∧
    (∑ i : Fin n, a i = ∑ i : Fin n, b i) ∧
    (n - 1 : ℚ > ∑ i : Fin n, (a i - b i : ℚ) / (a i + b i) ∧ 
    ∑ i : Fin n, (a i - b i : ℚ) / (a i + b i) > n - 1 - 1 / 1998)

theorem exists_problem_solution : ∀ (n : ℕ) (hn : 1 < n), exists_distinct_positive_integers n hn := 
by
  simp [exists_distinct_positive_integers]
  -- The proof should be added here
  sorry

end exists_problem_solution_l580_580917


namespace tom_total_cost_l580_580026

theorem tom_total_cost :
  let vaccines_cost := 10 * 45 in
  let total_medical_cost := vaccines_cost + 250 in
  let insurance_covered := 0.80 * total_medical_cost in
  let tom_pay_medical := total_medical_cost - insurance_covered in
  let trip_cost := 1200 in
  let total_cost := tom_pay_medical + trip_cost in
  total_cost = 1340 :=
by
  dsimp
  sorry

end tom_total_cost_l580_580026


namespace part1_part2_l580_580250

variable (x k : ℝ)

-- Given the quadratic equation x² - 2kx + k² - k - 1 = 0
axiom quadratic_eq_has_two_distinct_real_roots :
  (x^2 - 2*k*x + k^2 - k - 1 = 0) ∧ 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 2*k*x1 + k^2 - k - 1 = 0) ∧ (x2^2 - 2*k*x2 + k^2 - k - 1 = 0)

-- Part (1) proving x1 x2^2 + x1^2 x2 = 190 when k = 5
theorem part1 (h : k = 5) :
  ∃ (x1 x2 : ℝ), quadratic_eq_has_two_distinct_real_roots k ∧ 
  (x1 * x2^2 + x1^2 * x2 = 190) := by 
  sorry

-- Part (2) proving the value of k is 3 given x1 - 3x2 = 2
theorem part2 (h : ∃ (x1 x2 : ℝ), (x1 - 3*x2 = 2) ∧ 
  quadratic_eq_has_two_distinct_real_roots k) :
  k = 3 := by
  sorry

end part1_part2_l580_580250


namespace similar_triangles_height_l580_580079

theorem similar_triangles_height
  (a b : ℕ)
  (area_ratio: ℕ)
  (height_smaller : ℕ)
  (height_relation: height_smaller = 5)
  (area_relation: area_ratio = 9)
  (similarity: a / b = 1 / area_ratio):
  (∃ height_larger : ℕ, height_larger = 15) :=
by
  sorry

end similar_triangles_height_l580_580079


namespace sum_of_areas_of_circles_l580_580153

theorem sum_of_areas_of_circles (u v w : ℝ) :
  u + v = 6 → u + w = 8 → v + w = 10 → 
  (u = 2 ∧ v = 4 ∧ w = 6) →
  π * u^2 + π * v^2 + π * w^2 = 56 * π :=
by 
  intros hu hv hw huvw,
  cases huvw with hu hrest,
  cases hrest with hv hw,
  rw [hu, hv, hw],
  ring,
  rfl, 
  sorry

end sum_of_areas_of_circles_l580_580153


namespace mrs_thomson_savings_l580_580352

def incentive : ℝ := 600
def food_percentage : ℝ := 0.30
def clothes_percentage : ℝ := 0.20
def household_percentage : ℝ := 0.15
def savings_percentage : ℝ := 0.60

theorem mrs_thomson_savings :
  let remaining_after_food := incentive * (1 - food_percentage) in
  let remaining_after_clothes := remaining_after_food * (1 - clothes_percentage) in
  let remaining_after_household := remaining_after_clothes * (1 - household_percentage) in
  let savings := remaining_after_household * savings_percentage in
  savings = 171.36 := by
  sorry

end mrs_thomson_savings_l580_580352


namespace find_t_l580_580237

-- Conditions:
-- \overrightarrow{m} and \overrightarrow{n} are unit vectors
-- There is an angle of 120 degrees between \overrightarrow{m} and \overrightarrow{n}
-- The vector \overrightarrow{a} = t\overrightarrow{m} + (1 - t)\overrightarrow{n}

variables {V : Type*} [inner_product_space ℝ V]
variables (m n : V)
variables (t : ℝ)

-- Conditions for \overrightarrow{m} and \overrightarrow{n} being unit vectors and their dot product due to angle 120°
def unit_vectors : Prop := ∥m∥ = 1 ∧ ∥n∥ = 1 ∧ ⟪m, n⟫ = -1/2

-- The vector \overrightarrow{a} = t\overrightarrow{m} + (1 - t)\overrightarrow{n}
def a := t • m + (1 - t) • n

-- Condition \overrightarrow{n} ⟂ \overrightarrow{a}
def perpendicular : Prop := ⟪n, a t m n⟫ = 0

-- Statement to prove:
theorem find_t (h : unit_vectors m n) : perpendicular t m n ↔ t = 2/3 :=
sorry

end find_t_l580_580237


namespace incircle_radius_of_triangle_l580_580941

theorem incircle_radius_of_triangle (A B C : Type) (AB AC BC : ℝ) (h_AB : AB = 29) (h_AC : AC = 28) (h_BC : BC = 27) :
  (∃ r : ℝ, r = sqrt 210) := 
begin
  sorry
end

end incircle_radius_of_triangle_l580_580941


namespace circumcircles_concurrence_l580_580304

noncomputable def single_point_circumcircles (A1 A2 B1 B2 C1 C2 : Point) : Prop :=
  ∃ P : Point,
    (is_circumcircle A1 B1 C1 P) ∧
    (is_circumcircle A1 B2 C2 P) ∧
    (is_circumcircle A2 B1 C2 P) ∧
    (is_circumcircle A2 B2 C1 P)

theorem circumcircles_concurrence (A1 A2 B1 B2 C1 C2 : Point) :
  single_point_circumcircles A1 A2 B1 B2 C1 C2 →
  ∃ Q : Point,
    (is_circumcircle A2 B2 C2 Q) ∧
    (is_circumcircle A2 B1 C1 Q) ∧
    (is_circumcircle A1 B2 C1 Q) ∧
    (is_circumcircle A1 B1 C2 Q) :=
begin
  sorry
end

end circumcircles_concurrence_l580_580304


namespace mn_value_log_sum_value_l580_580574

theorem mn_value (m n : ℝ) (h1 : log 10 m + log 10 n = 2) (h2 : log 10 m * log 10 n = 1 / 2) : 
  m * n = 100 := 
sorry

theorem log_sum_value (m n : ℝ) (h1 : log 10 m + log 10 n = 2) (h2 : log 10 m * log 10 n = 1 / 2) : 
  log n m + log m n = 6 := 
sorry

end mn_value_log_sum_value_l580_580574


namespace min_value_of_f_l580_580342

noncomputable def f (x : ℝ) (h : x > 0) : ℝ :=
  sorry

theorem min_value_of_f :
  ∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y > 0 → f y ≤ f x) ∧ f x = 1 := 
sorry

end min_value_of_f_l580_580342


namespace sum_of_altitudes_l580_580138

def line_eq (x y : ℝ) : Prop := 9 * x + 6 * y = 54

def altitude_from_y_axis : ℝ :=
  let d := 54 / Real.sqrt (9^2 + 6^2)
  d

def altitude_from_line (y : ℝ) : ℝ :=
  let d := abs (6 * y - 54) / Real.sqrt (9^2 + 6^2)
  d

theorem sum_of_altitudes : 
  altitude_from_y_axis + altitude_from_line 4 = 28 * Real.sqrt 13 / 13 := by
  sorry

end sum_of_altitudes_l580_580138


namespace centroid_condition_l580_580372

variable (A B C G : Point)

-- Conditions
def GA := G.vectorTo A
def GB := G.vectorTo B
def GC := G.vectorTo C

-- Statement
theorem centroid_condition : (GA + GB + GC = 0) ↔ (G = (1/3 : ℝ) • (A + B + C)) := sorry

end centroid_condition_l580_580372


namespace similar_triangles_height_ratio_l580_580071

theorem similar_triangles_height_ratio (area_ratio : ℝ) (h₁ : ℝ) (h₂ : ℝ) 
  (similar : Boolean) (h₁_value : h₁ = 5) (area_ratio_value : area_ratio = 9) :
  similar = true → area_ratio = (h₂ / h₁) ^ 2 → h₂ = 15 :=
by
  intro h_similar area_eq
  rw [h₁_value, area_ratio_value]
  sorry

end similar_triangles_height_ratio_l580_580071


namespace x_intercept_of_translated_line_l580_580000

theorem x_intercept_of_translated_line :
  let line_translation (y : ℝ) := y + 4
  let new_line_eq := fun (x : ℝ) => 2 * x - 2
  new_line_eq 1 = 0 :=
by
  sorry

end x_intercept_of_translated_line_l580_580000


namespace range_of_a_l580_580256

def f (x : ℝ) (a : ℝ) : ℝ := x * log x - (a / 2) * x^2

theorem range_of_a (x_0 : ℝ) (h1 : x_0 ∈ set.Icc real.exp (real.exp ^ 2)) (h2 : f x_0 a > 0) :
  a < 2 / real.exp :=
sorry

end range_of_a_l580_580256


namespace shaded_region_area_l580_580671

theorem shaded_region_area (r : ℝ) (π : ℝ) (h1 : r = 5) : 
  4 * ((1/2 * π * r * r) - (1/2 * r * r)) = 50 * π - 50 :=
by 
  sorry

end shaded_region_area_l580_580671


namespace spotted_and_fluffy_cats_l580_580500

theorem spotted_and_fluffy_cats (total_cats : ℕ) (h1 : total_cats = 120)
    (fraction_spotted : ℚ) (h2 : fraction_spotted = 1/3)
    (fraction_fluffy_of_spotted : ℚ) (h3 : fraction_fluffy_of_spotted = 1/4) :
    (total_cats * fraction_spotted * fraction_fluffy_of_spotted).toNat = 10 := by
  sorry

end spotted_and_fluffy_cats_l580_580500


namespace math_proof_problem_l580_580231

variable (a_n : ℕ → ℤ) (S_n : ℕ → ℤ)
variable (a_1 d : ℤ)
variable (n : ℕ)

def arith_seq : Prop := ∀ n, a_n n = a_1 + (n - 1) * d

def sum_arith_seq : Prop := ∀ n, S_n n = n * (a_1 + (n - 1) * d / 2)

def condition1 : Prop := a_n 5 + a_n 9 = -2

def condition2 : Prop := S_n 3 = 57

noncomputable def general_formula : Prop := ∀ n, a_n n = 27 - 4 * n

noncomputable def max_S_n : Prop := ∀ n, S_n n ≤ 78 ∧ ∃ n, S_n n = 78

theorem math_proof_problem : 
  arith_seq a_n a_1 d ∧ sum_arith_seq S_n a_1 d ∧ condition1 a_n ∧ condition2 S_n 
  → general_formula a_n ∧ max_S_n S_n := 
sorry

end math_proof_problem_l580_580231


namespace six_digit_number_count_l580_580576

theorem six_digit_number_count:
  ∃ n : ℕ, n = 1440 ∧ 
  ∀ (s : fin 6 → ℕ), 
    multiset.card (multiset.of_fn s) = 6 →
    (∀ x, x ∈ multiset.of_fn s → x ∈ {1, 2, 3, 4, 5, 6}) →
    (∀ i, s i ≠ s (i + 1)) →
    (∀ i, (s i = 3 ∧ s (i + 1) ≠ 4) ∧ (s i = 4 ∧ s (i + 1) ≠ 3)) →
    (∃ i, (s i = 1 ∧ s (i + 1) = 2) ∨ (s i = 2 ∧ s (i + 1) = 1)) :=
  sorry


end six_digit_number_count_l580_580576


namespace james_100m_time_l580_580685

def john_time : ℝ := 13
def john_initial_distance : ℝ := 4
def james_speed_advantage : ℝ := 2
def james_initial_distance : ℝ := 10
def james_initial_time : ℝ := 2

theorem james_100m_time : true := by
  -- John's specifications
  let john_total_distance := 100
  let john_top_speed_distance := john_total_distance - john_initial_distance
  let john_top_speed_time := john_time - 1
  let john_top_speed := john_top_speed_distance / john_top_speed_time
  
  -- James's specifications
  let james_top_speed := john_top_speed + james_speed_advantage
  let james_remaining_distance := john_total_distance - james_initial_distance
  let james_remaining_time := james_remaining_distance / james_top_speed
  let james_total_time := james_initial_time + james_remaining_time
  
  -- The condition to prove
  have : james_total_time = 11 := sorry
  
  exact trivial

end james_100m_time_l580_580685


namespace height_of_larger_triangle_l580_580061

-- Definitions from the conditions
variables (height_small height_large : ℝ)
variables (area_ratio : ℝ)
variables (k : ℝ)

-- Given conditions
def triangles_similar : Prop := area_ratio = 9
def height_small_defined : Prop := height_small = 5
def scale_factor : Prop := k = real.sqrt area_ratio

-- Proof problem statement
theorem height_of_larger_triangle
  (h_similar : triangles_similar)
  (h_height_small : height_small_defined)
  (h_scale_factor : scale_factor) :
  height_large = 15 := sorry

end height_of_larger_triangle_l580_580061


namespace inclination_angle_is_correct_l580_580765

noncomputable def inclination_angle_of_line (t : ℝ) : ℝ :=
let x := 3 + t * Real.sin (25 * Real.pi / 180)
let y := - t * Real.cos (25 * Real.pi / 180)
let m := -Real.tan (65 * Real.pi / 180)
in Real.atan2 m 1

theorem inclination_angle_is_correct : 
  inclination_angle_of_line t = 115 * Real.pi / 180 :=
sorry

end inclination_angle_is_correct_l580_580765


namespace eccentricity_range_correct_l580_580260

variable {a b c : ℝ}
variable (h1 : a > 0) (h2 : b > 0) (h3 : b^2 + a^2 = c^2)
variable (P : ℝ × ℝ) (x y : ℝ)

def hyperbola : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def parabola : Prop := y^2 = 8 * a * x
def vertex_A : Prop := P = (a, 0)
def point_F : Prop := P = (2 * a, 0)
def on_asymptote : Prop := y = (b / a) * x ∨ y = -(b / a) * x
def perpendicular : Prop := (a - x, -y) ⬝ (2 * a - x, -y) = 0
def eccentricity_range (e : ℝ) : Prop := 1 < e ∧ e ≤ (3 * sqrt 2) / 4

theorem eccentricity_range_correct : 
  (hyperbola ∧ parabola ∧ vertex_A ∧ point_F ∧ on_asymptote ∧ perpendicular) → 
  (∃ e, eccentricity_range e) := 
sorry

end eccentricity_range_correct_l580_580260


namespace trig_problem_l580_580580

theorem trig_problem 
  (α : ℝ) 
  (h1 : Real.cos α = -1/2) 
  (h2 : 180 * (Real.pi / 180) < α ∧ α < 270 * (Real.pi / 180)) : 
  α = 240 * (Real.pi / 180) :=
sorry

end trig_problem_l580_580580


namespace number_of_correct_propositions_l580_580866

theorem number_of_correct_propositions :
  let y1 (x : ℝ) := 1 + x,
      y2 (x : ℝ) := Real.sqrt ((1 + x) ^ 2),
      f (x : ℝ) := x^2 + 2 * (a - 1) * x + 2,
      g (x : ℝ) := x^2 - 2 * Real.abs x - 3 in
  (∀ x : ℝ, y1 x = y2 x ↔ x = 0 ∧ y1 x ≠ y2 x) ∧
  (∀ x : ℝ, x > 0 → f x > f (x - 1) ∧ x < 0 → f x < f (x + 1)) ∧
  (∀ a : ℝ, (∀ x ∈ Set.Iic (4 : ℝ), f x < f (x + 1)) → a ≤ -3) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 0, y x < y (x + 1)) → (1 : ℝ) :=
by
  sorry

end number_of_correct_propositions_l580_580866


namespace inequality_positive_reals_l580_580742

theorem inequality_positive_reals (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) :
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧ 
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ (3 * Real.sqrt 2 / 2) :=
sorry

end inequality_positive_reals_l580_580742


namespace rate_increase_factor_l580_580863

-- Define the initial rate equation
def reaction_rate (k : ℝ) (C_CO C_O2 : ℝ) : ℝ :=
  k * C_CO^2 * C_O2

-- Initial concentrations and rate
variables (k C_CO C_O2 : ℝ)
def initial_rate : ℝ := reaction_rate k C_CO C_O2

-- New concentrations when volume is reduced by a factor of 3
variables (factor : ℝ) (k' : k' = k) (factor_eq : factor = 3)

-- New rate with increased concentrations
def new_rate : ℝ := reaction_rate k (factor * C_CO) (factor * C_O2)

-- Proof goal: new_rate is 27 times the initial_rate
theorem rate_increase_factor :
  new_rate / initial_rate = 27 :=
by
  sorry

end rate_increase_factor_l580_580863


namespace move_symmetric_point_left_l580_580655

-- Define the original point and the operations
def original_point : ℝ × ℝ := (-2, 3)

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

def move_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

-- Prove the resulting point after the operations
theorem move_symmetric_point_left : move_left (symmetric_point original_point) 2 = (0, -3) :=
by
  sorry

end move_symmetric_point_left_l580_580655


namespace find_number_l580_580207

theorem find_number (n : ℕ) (h : 582964 * n = 58293485180) : n = 100000 :=
by
  sorry

end find_number_l580_580207


namespace number_of_ways_to_choose_starters_l580_580837

-- Define the quadruplets
def quadruplets : Finset String := { "Ben", "Bob", "Bill", "Brent" }

-- Define the total set of players
def total_players : Finset String := { "Ben", "Bob", "Bill", "Brent", "Player5", "Player6", "Player7", "Player8", "Player9", "Player10", "Player11", "Player12", "Player13", "Player14", "Player15", "Player16" }

-- Define the calculation for the number of ways to choose exactly one from quadruplets and 6 from the remaining players
def num_ways_to_choose_starters : ℕ := 
  (quadruplets.card * nat.choose 12 6)

-- The theorem statement that asserts the total number of ways to choose the 7 starters
theorem number_of_ways_to_choose_starters : num_ways_to_choose_starters = 3696 :=
by
  -- Proof omitted
  sorry

end number_of_ways_to_choose_starters_l580_580837


namespace mapping_count_l580_580605

open Function

def M := {a, b, c}
def N := {-1, 0, 1}
def valid_mapping (f : M → N) := f a + f b + f c = 0

theorem mapping_count : ∃ (S : Finset (M → N)), S.card = 7 ∧ ∀ f ∈ S, valid_mapping f ∧ ∀ g, valid_mapping g → g ∈ S :=
by
  sorry

end mapping_count_l580_580605


namespace find_C0_to_B_distance_sum_l580_580172

noncomputable def distance_center_to_point_B : ℚ :=
let r := (11 : ℚ) / 60 in
let d := 49 / 61 in d

-- Lean statement to express the proof problem
theorem find_C0_to_B_distance_sum (r : ℚ) (m n : ℕ) (h_r : r = 11 / 60) (h_rel_prime : Nat.coprime m n)
  (d : ℚ) (h_d : d = 49 / 61) : 
  m + n = 110 :=
begin
  have : d = 49 / 61 := h_d,
  have : r = 11 / 60 := h_r,
  have : Nat.coprime 49 61 := by { -- by calculation or known fact
    sorry 
  },
  sorry 
end

end find_C0_to_B_distance_sum_l580_580172


namespace black_balls_count_l580_580836

theorem black_balls_count 
  (B : ℕ)
  (h : 6 / (B + 5) ^ 2 = 0.14814814814814814) :
  B = 1 :=
sorry

end black_balls_count_l580_580836


namespace isosceles_triangle_triangle_is_isosceles_l580_580653

theorem isosceles_triangle (a b : ℝ) (A B C : ℝ) (h : a * Real.cos B = b * Real.cos A):
  A + B + C = Real.pi → ∃ k > 0, (a = b ∧ (C = Real.pi - 2 * A ∨ C = Real.pi - 2 * B)) := 
by 
  -- Ensuring triangle inequality holds in ℝ
  intro h1
  use b
  split
  {
    sorry -- Proof of existence of k, namely b
  },
  {
    split
    {
      sorry -- Proof that b = a
    },
    {
      left,
      sorry -- Proof that either C = Real.pi - 2 * A 
    }
  }

>>> An isosceles triangle has two equal angles, which we can state equivalently using the sides.

theorem triangle_is_isosceles (a b: ℝ) (A B C: ℝ) (h: a * cos B = b * cos A):
  (sin A ≠ 0 ∧ sin B ≠ 0) → 
  A + B + C = π → 
  A = B := 
by 
  -- This statement directly implies that the triangle is isosceles.
  sorry

end isosceles_triangle_triangle_is_isosceles_l580_580653


namespace inclination_angle_parametric_eq_l580_580766

-- Define the parametric equations as conditions
def parametric_eq_x (t : ℝ) : ℝ := 1 + t
def parametric_eq_y (t : ℝ) : ℝ := 1 - t

-- Define the line from the parametric equations
def line_eq (x y : ℝ) : Prop := x + y = 2

-- Define the inclination angle
def inclination_angle (m : ℝ) : ℝ := 
if m = 1 then Real.pi / 4 else 
if m = -1 then 3 * Real.pi / 4 else 
if m = 0 then 0 else sorry

-- State the theorem
theorem inclination_angle_parametric_eq :
  inclination_angle (-1) = 3 * Real.pi / 4 :=
sorry

end inclination_angle_parametric_eq_l580_580766


namespace identify_truth_teller_l580_580550

-- Define each villager's statement
def villager1 := "All four of us are liars"
def villager2 := "Among us four, only one is a liar"
def villager3 := "Among us, there are two liars"
def villager4 := "I am a truth-teller"

-- Define the proposition that the fourth villager is the truth-teller
def fourth_is_truth_teller : Prop :=
  villager1 = "All four of us are liars" → 
  villager2 = "Among us four, only one is a liar" → 
  villager3 = "Among us, there are two liars" → 
  villager4 = "I am a truth-teller" → 
  (some condition that ensures only the fourth villager is telling the truth)

theorem identify_truth_teller :
  fourth_is_truth_teller :=
sorry

end identify_truth_teller_l580_580550


namespace period_of_tan_2x_pi_over_6_l580_580399

noncomputable def function_period (x : ℝ) : ℝ := tan (2 * x + real.pi / 6)

theorem period_of_tan_2x_pi_over_6 : 
  ∃ T, (∀ x, function_period (x + T) = function_period x) ∧ T = real.pi / 2 :=
by
  sorry

end period_of_tan_2x_pi_over_6_l580_580399


namespace power_of_power_l580_580874

theorem power_of_power (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 := 
  sorry

end power_of_power_l580_580874


namespace tickets_sold_at_full_price_correct_l580_580893

-- Given conditions as definitions
def R : ℕ := 5400
def T : ℕ := 25200
def T_full_price : ℕ := 5 * R

-- The mathematical proof problem
theorem tickets_sold_at_full_price_correct :
  T = R + T_full_price → T_full_price = 21000 :=
by
  intro h
  have hR : R = 4200 := by
    calc
      R = 25200 / 6 : by sorry
  have hT_full_price : T_full_price = 5 * 4200 := by sorry
  calc
    T_full_price = 5 * 4200 : hT_full_price
                ... = 21000 : by sorry

end tickets_sold_at_full_price_correct_l580_580893


namespace find_f2_l580_580586

namespace ProofProblem

-- Define the polynomial function f
def f (x a b : ℤ) : ℤ := x^5 + a * x^3 + b * x - 8

-- Conditions given in the problem
axiom f_neg2 : ∃ a b : ℤ, f (-2) a b = 10

-- Define the theorem statement
theorem find_f2 : ∃ a b : ℤ, f 2 a b = -26 :=
by
  sorry

end ProofProblem

end find_f2_l580_580586


namespace total_cows_l580_580478

theorem total_cows (cows : ℕ) (h1 : cows / 3 + cows / 5 + cows / 6 + 12 = cows) : cows = 40 :=
sorry

end total_cows_l580_580478


namespace unit_vectors_have_equal_magnitude_l580_580582

variables (e1 e2 : ℝ^3) -- Specify the vectors are in 3D space for concreteness (could be generalized)

-- Definition of a unit vector
def is_unit_vector (v : ℝ^3) : Prop := ‖v‖ = 1

theorem unit_vectors_have_equal_magnitude (h1 : is_unit_vector e1) (h2 : is_unit_vector e2) : 
  ‖e1‖ = ‖e2‖ := 
sorry

end unit_vectors_have_equal_magnitude_l580_580582


namespace interest_rate_l580_580759

-- Define the sum of money
def P : ℝ := 1800

-- Define the time period in years
def T : ℝ := 2

-- Define the difference in interests
def interest_difference : ℝ := 18

-- Define the relationship between simple interest, compound interest, and the interest rate
theorem interest_rate (R : ℝ) 
  (h1 : SI = P * R * T / 100)
  (h2 : CI = P * (1 + R/100)^2 - P)
  (h3 : CI - SI = interest_difference) :
  R = 10 :=
by
  sorry

end interest_rate_l580_580759


namespace permutation_value_l580_580828

theorem permutation_value : ∀ (n r : ℕ), n = 5 → r = 3 → (n.choose r) * r.factorial = 60 := 
by
  intros n r hn hr 
  rw [hn, hr]
  -- We use the permutation formula A_{n}^{r} = n! / (n-r)!
  -- A_{5}^{3} = 5! / 2!
  -- Simplifies to 5 * 4 * 3 = 60.
  sorry

end permutation_value_l580_580828


namespace green_square_area_percentage_l580_580857

variable (s a : ℝ)
variable (h : a^2 + 4 * a * (s - 2 * a) = 0.49 * s^2)

theorem green_square_area_percentage :
  (a^2 / s^2) = 0.1225 :=
sorry

end green_square_area_percentage_l580_580857


namespace person6_number_l580_580035

theorem person6_number (a : ℕ → ℕ) (x : ℕ → ℕ) 
  (mod12 : ∀ i, a (i % 12) = a i)
  (h5 : x 5 = 5)
  (h6 : x 6 = 8)
  (h7 : x 7 = 11) 
  (h_avg : ∀ i, x i = (a (i-1) + a (i+1)) / 2) : 
  a 6 = 6 := sorry

end person6_number_l580_580035


namespace trig_identity_l580_580581

theorem trig_identity (α : ℝ) (h1 : Real.cos α = -4/5) (h2 : π/2 < α ∧ α < π) : 
  - (Real.sin (2 * α) / Real.cos α) = -6/5 :=
by
  sorry

end trig_identity_l580_580581


namespace roots_of_quadratic_eq_l580_580012

theorem roots_of_quadratic_eq (a b : ℝ) (h1 : a * (-2)^2 + b * (-2) = 6) (h2 : a * 3^2 + b * 3 = 6) :
    ∃ (x1 x2 : ℝ), x1 = -2 ∧ x2 = 3 ∧ ∀ x, a * x^2 + b * x = 6 ↔ (x = x1 ∨ x = x2) :=
by
  use -2, 3
  sorry

end roots_of_quadratic_eq_l580_580012


namespace dot_product_is_correct_l580_580384

open Real

variables (A B C D O : Point)
variables (AB CD AD BC AO BO : Vector)
variables (base1 : Real)
variables (base2 : Real)
variables (AB_length CD_length AO_length BO_length : Real)
variables (perpendicular : Prop)

def common_point := O ∈ Line AC ∧ O ∈ Line BD

-- Length conditions
def length_conditions : Prop :=
  |AB_length = 55 ∧ CD_length = 31 ∧ |AB_length| > |CD_length|

-- Perpendicularity condition
def perpendicular_diagonals : Prop :=
  perpendicular (Line AC) (Line BD)

-- Given both conditions, and lengths define the lengths correctly
def lengths_are_set_correctly : Prop :=
  |AD_length^2 + BO_length^2 = AB_length^2 ∧ |AD_length| = AO_length ∧ |BO_length| = 55 ∧ 31*|BO_length^2| = 55 ∧ CA*AO = 31

-- Target theorem
theorem dot_product_is_correct :
  length_conditions → perpendicular_diagonals → 
  ∃ A B,
    dot_product(AD, BC) = 1705 := sorry

end dot_product_is_correct_l580_580384


namespace part1_part2_l580_580972

noncomputable def f (a : ℝ) (x : ℝ) := Real.log (2 - x) + a * x

theorem part1 (a : ℝ) : (∀ x ∈ Ioo (0 : ℝ) 1, deriv (f a) x > 0) ↔ a ≥ 1 := by
  sorry

theorem part2 : (∀ x ∈ Ioo (1 : ℝ) 2, deriv (f 1) x < 0) := by
  sorry

end part1_part2_l580_580972


namespace equation_of_trajectory_l580_580594

open Real

variable (P : ℝ → ℝ → Prop)
variable (C : ℝ → ℝ → Prop)
variable (L : ℝ → ℝ → Prop)

-- Definition of the fixed circle C
def fixed_circle (x y : ℝ) : Prop :=
  (x + 2) ^ 2 + y ^ 2 = 1

-- Definition of the fixed line L
def fixed_line (x y : ℝ) : Prop := 
  x = 1

noncomputable def moving_circle (P : ℝ → ℝ → Prop) (r : ℝ) : Prop :=
  ∃ x y : ℝ, P x y ∧ r > 0 ∧
  (∀ a b : ℝ, fixed_circle a b → ((x - a) ^ 2 + (y - b) ^ 2) = (r + 1) ^ 2) ∧
  (∀ a b : ℝ, fixed_line a b → (abs (x - a)) = (r + 1))

theorem equation_of_trajectory
  (P : ℝ → ℝ → Prop)
  (r : ℝ)
  (h : moving_circle P r) :
  ∀ x y : ℝ, P x y → y ^ 2 = -8 * x :=
by
  sorry

end equation_of_trajectory_l580_580594


namespace domain_of_f_2x_minus_1_l580_580244

theorem domain_of_f_2x_minus_1 (f : ℝ → ℝ) (dom : ∀ x, f x ≠ 0 → (0 < x ∧ x < 1)) :
  ∀ x, f (2*x - 1) ≠ 0 → (1/2 < x ∧ x < 1) :=
by
  sorry

end domain_of_f_2x_minus_1_l580_580244


namespace sqrt_of_4_l580_580809

theorem sqrt_of_4 :
  set_is_eq (sqrt 4) 2,
    -2 : sqrt 4 := sorry

end sqrt_of_4_l580_580809


namespace prove_eccentricity_range_l580_580482

noncomputable def eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) (k : ℝ) 
    (h_k : (1 : ℝ)/3 < k ∧ k < 1/2) : Prop :=
    let e := (a^2 - b^2)^0.5 / a in
    1/2 < e ∧ e < 2/3

theorem prove_eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) (k : ℝ) 
    (h_k : (1 : ℝ)/3 < k ∧ k < 1/2) : eccentricity_range a b h k h_k := 
sorry

end prove_eccentricity_range_l580_580482


namespace perimeter_of_E_l580_580136

theorem perimeter_of_E
  (rectangle1 rectangle2 rectangle3 : ℝ)
  (h1 : rectangle1 = 3)
  (h2 : rectangle2 = 6)
  (h3 : rectangle3 = 3)
  (h4 : rectangle1 = 6)
  (h5 : rectangle2 = 3)
  (h6 : rectangle3 = 6)
  (h7 : rectangle1 = 3)
  (h8 : rectangle2 = 6)
  (h9 : rectangle3 = 3) :
  15 + (12 + 12) = 39 :=
by {
  sorry,
}

end perimeter_of_E_l580_580136


namespace minimum_grade_Ahmed_l580_580156

theorem minimum_grade_Ahmed (assignments : ℕ) (Ahmed_grade : ℕ) (Emily_grade : ℕ) (final_assignment_grade_Emily : ℕ) 
  (sum_grades_Emily : ℕ) (sum_grades_Ahmed : ℕ) (total_points_Ahmed : ℕ) (total_points_Emily : ℕ) :
  assignments = 9 →
  Ahmed_grade = 91 →
  Emily_grade = 92 →
  final_assignment_grade_Emily = 90 →
  sum_grades_Emily = 828 →
  sum_grades_Ahmed = 819 →
  total_points_Ahmed = sum_grades_Ahmed + 100 →
  total_points_Emily = sum_grades_Emily + final_assignment_grade_Emily →
  total_points_Ahmed > total_points_Emily :=
by
  sorry

end minimum_grade_Ahmed_l580_580156


namespace spotted_and_fluffy_cats_l580_580499

theorem spotted_and_fluffy_cats (total_cats : ℕ) (total_cats_eq : total_cats = 120) 
  (spotted_fraction : ℚ) (spotted_fraction_eq : spotted_fraction = 1/3)
  (fluffy_fraction : ℚ) (fluffy_fraction_eq : fluffy_fraction = 1/4) :
  let spotted_cats := (total_cats * spotted_fraction).natAbs in
  let fluffy_spotted_cats := (spotted_cats * fluffy_fraction).natAbs in
  fluffy_spotted_cats = 10 :=
by
  sorry

end spotted_and_fluffy_cats_l580_580499


namespace average_price_over_3_months_l580_580107

theorem average_price_over_3_months (dMay : ℕ) 
  (pApril pMay pJune : ℝ) 
  (h1 : pApril = 1.20) 
  (h2 : pMay = 1.20) 
  (h3 : pJune = 3.00) 
  (h4 : dApril = 2 / 3 * dMay) 
  (h5 : dJune = 2 * dApril) :
  ((dApril * pApril + dMay * pMay + dJune * pJune) / (dApril + dMay + dJune) = 2) := 
by sorry

end average_price_over_3_months_l580_580107


namespace tom_pays_1340_l580_580031

def vaccine_cost := 45
def number_of_vaccines := 10
def doctor_visit_cost := 250
def insurance_coverage := 0.8
def trip_cost := 1200

def total_vaccine_cost := vaccine_cost * number_of_vaccines
def total_medical_cost := total_vaccine_cost + doctor_visit_cost
def insurance_cover_amount := total_medical_cost * insurance_coverage
def amount_paid_after_insurance := total_medical_cost - insurance_cover_amount
def total_amount_tom_pays := amount_paid_after_insurance + trip_cost

theorem tom_pays_1340 :
  total_amount_tom_pays = 1340 :=
by
  sorry

end tom_pays_1340_l580_580031


namespace find_X_l580_580593

variables (a : ℝ)

structure Point := (x : ℝ) (y : ℝ) (z : ℝ)

def A : Point := ⟨0, 0, 0⟩
def B : Point := ⟨a, 0, 0⟩
def C : Point := ⟨a, a, 0⟩
def D : Point := ⟨0, a, 0⟩
def A₁ : Point := ⟨0, 0, a⟩
def B₁ : Point := ⟨a, 0, a⟩
def C₁ : Point := ⟨a, a, a⟩
def D₁ : Point := ⟨0, a, a⟩

def K : Point := ⟨a, a / 2, a⟩
def M : Point := ⟨a, a, 0.75 * a⟩
def P : Point := ⟨0.75 * a, a, 0⟩

def line (P Q : Point) (t : ℝ) : Point := ⟨P.x + t * (Q.x - P.x), P.y + t * (Q.y - P.y), P.z + t * (Q.z - P.z)⟩

def X (t : ℝ) : Point := ⟨ a - a * t, a * t, a * t ⟩

theorem find_X : ∃ t : ℝ, ( ∃ X : Point, X = line B D₁ t ∧ (X.x, X.y, X.z) = (a / 2, a / 2, a / 2) ) :=
begin
  use 0.5,
  use X 0.5,
  sorry

end find_X_l580_580593


namespace function_properties_l580_580812

theorem function_properties :
  ∃ f : ℝ → ℝ, (∀ x : ℝ, x < 0 → 0 < f x) ∧ (∀ x y : ℝ, x < y → f(x) > f(y)) ∧ (f = λ x, -x + 1) :=
by
  -- Function definition
  let f := λ x : ℝ, -x + 1
  -- Conditions that f must satisfy
  have h1 : ∀ x : ℝ, x < 0 → 0 < f x := by
    sorry
  have h2 : ∀ x y : ℝ, x < y → f(x) > f(y) := by
    sorry
  -- Final proof combining everything
  exact ⟨f, h1, h2, rfl⟩

end function_properties_l580_580812


namespace max_value_of_ratio_l580_580240

theorem max_value_of_ratio (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 1) : 
  ∃ z, z = (x / y) ∧ z ≤ 1 := sorry

end max_value_of_ratio_l580_580240


namespace ferry_max_weight_capacity_l580_580133

def automobile_max_weight : ℝ := 3200
def automobile_count : ℝ := 62.5
def pounds_to_tons : ℝ := 2000

theorem ferry_max_weight_capacity : 
  (automobile_max_weight * automobile_count) / pounds_to_tons = 100 := 
by 
  sorry

end ferry_max_weight_capacity_l580_580133


namespace sqrt_sqrt_eq_pm_4_iff_x_eq_256_l580_580651

theorem sqrt_sqrt_eq_pm_4_iff_x_eq_256 (x : ℝ) : (sqrt (sqrt x) = 4 ∨ sqrt (sqrt x) = -4) → x = 256 :=
by
  -- Proof omitted
  sorry

end sqrt_sqrt_eq_pm_4_iff_x_eq_256_l580_580651


namespace probability_of_sum_cis_eq_zero_l580_580318

noncomputable def probability_sum_cis_eq_zero : ℚ :=
  let a := Fin 24 in
  let cic_theta := λ θ : ℚ, Complex.ofReal (Real.cos θ) + Complex.I * Complex.ofReal (Real.sin θ) in
  let f := finset.finrange 24 in
  let points := (f^5 : finset (Fin 24 × Fin 24 × Fin 24 × Fin 24 × Fin 24)) in
  let valid_sum_zero := points.filter (λ p, (arraySum 5 (λ i, cic_theta ((p.fst.fst.val : ℚ) * π / 12))).im = 0 ∧
                                          (arraySum 5 (λ i, cic_theta ((p.fst.fst.val : ℚ) * π / 12))).re = 0) in
  (valid_sum_zero.card : ℚ) / (points.card : ℚ)

theorem probability_of_sum_cis_eq_zero :
  probability_sum_cis_eq_zero = 35 / (2 * 24 ^ 3) := sorry

end probability_of_sum_cis_eq_zero_l580_580318


namespace interest_ratio_l580_580778

theorem interest_ratio (P_SI : ℝ) (R_SI : ℝ) (T_SI : ℝ) (P_CI : ℝ) (R_CI : ℝ) (T_CI : ℝ) :
  P_SI = 3225 →
  R_SI = 8 →
  T_SI = 5 →
  P_CI = 8000 →
  R_CI = 15 →
  T_CI = 2 →
  let SI := (P_SI * R_SI * T_SI) / 100 in
  let CI := P_CI * ((1 + R_CI / 100)^T_CI - 1) in
  (SI / CI) = 1 / 10 :=
by
  intros
  simp only [SI, CI]
  sorry

end interest_ratio_l580_580778


namespace farmer_crops_arrangement_l580_580479

-- Definition of the adjacency restrictions and grid constraints
def is_valid_arrangement (arr : matrix (fin 3) (fin 2) string) : Prop :=
  ∀ i j, ((arr i j = "corn" ∧ (i < 2 ∧ (arr (i+1) j = "wheat" ∨ arr (i+1) j = "potatoes")))
            ∨ (arr i j = "wheat" ∧ i < 2 ∧ arr (i+1) j = "corn")
            ∨ (arr i j = "soybeans" ∧ i < 2 ∧ arr (i+1) j = "potatoes")
            ∨ (arr i j = "potatoes" ∧ (i < 2 ∧ (arr (i+1) j = "corn" ∨ arr (i+1) j = "soybeans"))))
           → false

-- The count of valid arrangements
def num_valid_arrangements : ℕ :=
  by sorry

-- The theorem statement proving the number of valid arrangements
theorem farmer_crops_arrangement : num_valid_arrangements = 283 :=
  by sorry

end farmer_crops_arrangement_l580_580479


namespace extremum_condition_l580_580392

noncomputable def quadratic_polynomial (a x : ℝ) : ℝ := a * x^2 + x + 1

theorem extremum_condition (a : ℝ) :
  (∃ x : ℝ, ∃ f' : ℝ → ℝ, 
     (f' = (fun x => 2 * a * x + 1)) ∧ 
     (f' x = 0) ∧ 
     (∃ (f'' : ℝ → ℝ), (f'' = (fun x => 2 * a)) ∧ (f'' x ≠ 0))) ↔ a < 0 := 
sorry

end extremum_condition_l580_580392


namespace alpha_value_l580_580264

theorem alpha_value (m : ℝ) (α : ℝ) (h : m * 8 ^ α = 1 / 4) : α = -2 / 3 :=
by
  sorry

end alpha_value_l580_580264


namespace sum_ceiling_sqrt_l580_580190

theorem sum_ceiling_sqrt : ∑ n in Finset.Icc 10 58, Nat.ceil (Real.sqrt n) = 302 :=
by {
  have h1 : ∀ n ∈ Finset.Icc 10 16, Nat.ceil (Real.sqrt n) = 4,
  { intros n hn, sorry }, -- Proof omitted
  have h2 : ∀ n ∈ Finset.Icc 17 25, Nat.ceil (Real.sqrt n) = 5,
  { intros n hn, sorry }, -- Proof omitted
  have h3 : ∀ n ∈ Finset.Icc 26 36, Nat.ceil (Real.sqrt n) = 6,
  { intros n hn, sorry }, -- Proof omitted
  have h4 : ∀ n ∈ Finset.Icc 37 49, Nat.ceil (Real.sqrt n) = 7,
  { intros n hn, sorry }, -- Proof omitted
  have h5 : ∀ n ∈ Finset.Icc 50 58, Nat.ceil (Real.sqrt n) = 8,
  { intros n hn, sorry }, -- Proof omitted
  -- Sum using the blocks defined by h1 through h5
  calc ∑ n in Finset.Icc 10 58, Nat.ceil (Real.sqrt n) 
        = ∑ n in Finset.Icc 10 16, 4
        + ∑ n in Finset.Icc 17 25, 5
        + ∑ n in Finset.Icc 26 36, 6
        + ∑ n in Finset.Icc 37 49, 7
        + ∑ n in Finset.Icc 50 58, 8 : by sorry -- Proof omitted
    ... = 7 * 4 + 9 * 5 + 11 * 6 + 13 * 7 + 9 * 8 : by sorry -- Proof omitted
    ... = 302 : by norm_num,
}

end sum_ceiling_sqrt_l580_580190


namespace sum_abs_of_roots_l580_580571

variables {p q r : ℤ}

theorem sum_abs_of_roots:
  p + q + r = 0 →
  p * q + q * r + r * p = -2023 →
  |p| + |q| + |r| = 94 := by
  intro h1 h2
  sorry

end sum_abs_of_roots_l580_580571


namespace flow_rate_DE_flow_rate_BC_flow_rate_GF_l580_580129

open Classical

-- Define the nodes and channels
inductive Node
| A | B | C | D | E | F | G | H

open Node

-- Define the flow rate function
def flow_rate (channel : (Node × Node)) : ℝ := sorry

-- Conditions
variable (q0 : ℝ)

-- Conditions from the problem
axiom flow_in_A : (flow_rate (A, B)) = q0
axiom sum_of_flow_rates_constant : ∀ (n m : Node), sum_of_flow_rates n m (flow_rate) = q0

-- Symmetry in the system
axiom symmetry_bc_cd : flow_rate (B, C) = flow_rate (C, D)
axiom symmetry_bg_gd : flow_rate (B, G) = flow_rate (G, D)

-- Questions and Expected Answers
theorem flow_rate_DE (q0 : ℝ) : flow_rate (D, E) = (4 / 7) * q0 := sorry
theorem flow_rate_BC (q0 : ℝ) : flow_rate (B, C) = (2 / 7) * q0 := sorry
theorem flow_rate_GF (q0 : ℝ) : flow_rate (G, F) = (3 / 7) * q0 := sorry

end flow_rate_DE_flow_rate_BC_flow_rate_GF_l580_580129


namespace find_p_q_l580_580716

theorem find_p_q :
  ∀ (p q : ℝ),
    (let f := λ x : ℝ, x^2 - p * x + q in f (p + q) = 0 ∧ f (p - q) = 0) →
    (∃ (m : ℝ), (p = m ∧ q = 0) ∨ (p = 0 ∧ q = -1)) :=
by
  sorry

end find_p_q_l580_580716


namespace tan_sin_equality_l580_580524

theorem tan_sin_equality :
  (Real.tan (30 * Real.pi / 180))^2 + (Real.sin (45 * Real.pi / 180))^2 = 5 / 6 :=
by sorry

end tan_sin_equality_l580_580524


namespace scientific_notation_of_150_million_l580_580760

theorem scientific_notation_of_150_million :
  ∃ a n, (a * 10 ^ n = 150000000) ∧ (1 ≤ |a|) ∧ (|a| < 10) ∧ (a = 1.5) ∧ (n = 8) :=
by
  -- We construct values satisfying the conditions directly
  use 1.5, 8
  split
  -- Prove 1.5 * 10 ^ 8 = 150000000
  . norm_num
  -- Prove 1 ≤ |1.5|
  . norm_num
  -- Prove |1.5| < 10
  . norm_num
  -- Prove a = 1.5
  . refl
  -- Prove n = 8
  . refl

end scientific_notation_of_150_million_l580_580760


namespace geometric_sequence_sum_9000_l580_580783

noncomputable def sum_geometric_sequence (a r : ℝ) (n : ℕ) : ℝ := 
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_9000 (a r : ℝ) (h : r ≠ 1) 
  (h1 : sum_geometric_sequence a r 3000 = 1000)
  (h2 : sum_geometric_sequence a r 6000 = 1900) : 
  sum_geometric_sequence a r 9000 = 2710 :=
sorry

end geometric_sequence_sum_9000_l580_580783


namespace part_i_part_ii_l580_580338

def s_n (n : ℕ) : ℚ :=
  ∑ i in Finset.range (n+1), (1 / (i + 1 : ℚ))

theorem part_i (n : ℕ) (h1 : 1 < n) :
  n * (n+1 : ℚ) ^ (1 / n : ℚ) < n + s_n n :=
sorry

theorem part_ii (n : ℕ) (h2 : 2 < n) :
  (n-1) * (n : ℚ) ^ (-1 / (n-1 : ℚ)) < n - s_n n :=
sorry

end part_i_part_ii_l580_580338


namespace largest_integer_less_than_log_sum_theorem_l580_580805

noncomputable def largest_integer_less_than_log_sum : ℕ := 
  let logs : ℕ → ℝ := λ n, Real.log 3 (n + 1) / Real.log 3 n
  let sum_logs : ℝ := Finset.sum (Finset.range 2022) logs
  ⌊sum_logs⌋

theorem largest_integer_less_than_log_sum_theorem : largest_integer_less_than_log_sum = 6 := by
  sorry

end largest_integer_less_than_log_sum_theorem_l580_580805


namespace translation_value_of_m_l580_580763

def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem translation_value_of_m (m : ℝ) (h1 : m ∈ Set.Ioo 0 Real.pi) 
    (h2 : ∀ x : ℝ, f (x - m) = f x) : 
    m = Real.pi / 2 :=
  by
    sorry

end translation_value_of_m_l580_580763


namespace percent_of_div_l580_580830

theorem percent_of_div (P: ℝ) (Q: ℝ) (R: ℝ) : ( ( P / 100 ) * Q ) / R = 354.2 :=
by
  -- Given P = 168, Q = 1265, R = 6
  let P := 168
  let Q := 1265
  let R := 6
  -- sorry to skip the actual proof.
  sorry

end percent_of_div_l580_580830


namespace f_monotonically_increasing_intervals_max_area_of_triangle_l580_580930

open Real

namespace MathProof

def f (x : ℝ) : ℝ := sin x * cos x - cos (x + π / 4) ^ 2

theorem f_monotonically_increasing_intervals :
  ∀ k : ℤ, ∀ x : ℝ, -π / 4 + k * π ≤ x ∧ x ≤ π / 4 + k * π → 0 ≤ (f x - f x) :=
sorry

theorem max_area_of_triangle (A : ℝ) (a b c : ℝ) :
  ft (A / 2) = 0 ∧ a = 1 →
  ∃ S : ℝ, S ≤ (2 + sqrt 3) / 4 ∧ is_area_of_triangle A b c S :=
sorry

end MathProof

end f_monotonically_increasing_intervals_max_area_of_triangle_l580_580930


namespace sum_of_distances_l580_580878

theorem sum_of_distances (A B C D M P : ℝ × ℝ) 
    (hA : A = (0, 0))
    (hB : B = (4, 0))
    (hC : C = (4, 4))
    (hD : D = (0, 4))
    (hM : M = (2, 0))
    (hP : P = (0, 2)) :
    dist A M + dist A P = 4 :=
by
  sorry

end sum_of_distances_l580_580878


namespace system_solution_l580_580540

theorem system_solution (x b y : ℝ) (h1 : 4 * x + 2 * y = b) (h2 : 3 * x + 4 * y = 3 * b) (h3 : x = 3) :
  b = -1 :=
by
  -- proof to be filled in
  sorry

end system_solution_l580_580540


namespace illegally_parked_percentage_l580_580444

theorem illegally_parked_percentage (total_cars : ℕ) (towed_cars : ℕ)
  (ht : towed_cars = 2 * total_cars / 100) (not_towed_percentage : ℕ)
  (hp : not_towed_percentage = 80) : 
  (100 * (5 * towed_cars) / total_cars) = 10 :=
by
  sorry

end illegally_parked_percentage_l580_580444


namespace starting_number_l580_580411

theorem starting_number (n : ℤ) : 
  (∃ n, (200 - n) / 3 = 33 ∧ (200 % 3 ≠ 0) ∧ (n % 3 = 0 ∧ n ≤ 200)) → n = 102 :=
by
  sorry

end starting_number_l580_580411


namespace height_of_larger_triangle_l580_580085

theorem height_of_larger_triangle 
  (area_ratio : ℝ)
  (height_small_triangle : ℝ)
  (similar_triangles : Prop)
  (height_large_triangle : ℝ) :
  area_ratio = 1 / 9 →
  height_small_triangle = 5 →
  similar_triangles →
  height_large_triangle = height_small_triangle * 3 :=
begin
  intros h_ratio h_height_small h_similar,
  rw h_ratio at *,
  rw h_height_small at *,
  exact eq.symm (mul_eq_mul_left_iff.1 (eq.trans (sqrt_eq (by norm_num) (by norm_num)) (by norm_num))),
sorry,
end

# The above code imports the necessary library, defines the theorem with the conditions and concludes with the height of the larger triangle.

end height_of_larger_triangle_l580_580085


namespace tan_beta_expression_tan_beta_maximum_l580_580224

noncomputable def tan_beta : ℝ → ℝ := λ α, (Real.tan α) / (1 + 2 * (Real.tan α) ^ 2)

theorem tan_beta_expression (α β : ℝ) (hα : 0 < α ∧ α < Real.pi / 2) (hβ : 0 < β ∧ β < Real.pi / 2)
(h_sum : α + β ≠ Real.pi / 2) (h_sin : Real.sin β = Real.sin α * Real.cos (α + β)) : 
  Real.tan β = (Real.tan α) / (1 + 2 * (Real.tan α) ^ 2) :=
sorry

theorem tan_beta_maximum (α β : ℝ) (hα : 0 < α ∧ α < Real.pi / 2) (hβ : 0 < β ∧ β < Real.pi / 2)
(h_sum : α + β ≠ Real.pi / 2) (h_sin : Real.sin β = Real.sin α * Real.cos (α + β)) : 
  Real.tan β ≤ Real.sqrt 2 / 4 :=
sorry

end tan_beta_expression_tan_beta_maximum_l580_580224


namespace optimal_pricing_l580_580683

-- Define the conditions given in the problem
def cost_price : ℕ := 40
def selling_price : ℕ := 60
def weekly_sales : ℕ := 300

def sales_volume (price : ℕ) : ℕ := weekly_sales - 10 * (price - selling_price)
def profit (price : ℕ) : ℕ := (price - cost_price) * sales_volume price

-- Statement to prove
theorem optimal_pricing : ∃ (price : ℕ), price = 65 ∧ profit price = 6250 :=
by {
  sorry
}

end optimal_pricing_l580_580683


namespace graph_transformation_l580_580794

-- Definitions based on the conditions from Step a)
def func1 (x : ℝ) : ℝ := 4 * sin (x + π / 5)
def func2 (x : ℝ) : ℝ := 4 * sin (2 * x + π / 5)

-- The statement to be proved
theorem graph_transformation :
  ∀ x : ℝ, func2 x = func1 (x / 2) :=
by sorry

end graph_transformation_l580_580794


namespace solve_complex_equation_l580_580999

-- Define the condition: z = i * (2 - z) for complex number z
def satisfies_condition (z : ℂ) : Prop := z = complex.I * (2 - z)

-- The theorem to prove: if z satisfies the condition, then z = 1 + i
theorem solve_complex_equation (z : ℂ) (h : satisfies_condition z) : z = 1 + complex.I :=
by 
  sorry

end solve_complex_equation_l580_580999


namespace frank_more_miles_than_jim_in_an_hour_l580_580313

theorem frank_more_miles_than_jim_in_an_hour
    (jim_distance : ℕ) (jim_time : ℕ)
    (frank_distance : ℕ) (frank_time : ℕ)
    (h_jim : jim_distance = 16)
    (h_jim_time : jim_time = 2)
    (h_frank : frank_distance = 20)
    (h_frank_time : frank_time = 2) :
    (frank_distance / frank_time) - (jim_distance / jim_time) = 2 := 
by
  -- Placeholder for the proof, no proof steps included as instructed.
  sorry

end frank_more_miles_than_jim_in_an_hour_l580_580313


namespace min_f_on_interval_l580_580969

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 / (x - 1)

-- Define the interval
def interval := set.Icc 3 6

-- The theorem to prove the minimum value of f(x) on the interval [3, 6]
theorem min_f_on_interval : ∀ x ∈ interval, f 6 ≤ f x :=
by {
  sorry  -- Proof not provided as per instructions
}

end min_f_on_interval_l580_580969


namespace fraction_solution_l580_580989

theorem fraction_solution (x : ℝ) (h : 4 - 9 / x + 4 / x^2 = 0) : 3 / x = 12 ∨ 3 / x = 3 / 4 :=
by
  -- Proof to be written here
  sorry

end fraction_solution_l580_580989


namespace tau2_is_topology_tau4_is_topology_l580_580272

-- Define the given set X
def X : Set := {a, b, c}

-- Define the candidate topologies
def tau1 : Set := {∅, {a}, {c}, {a, b, c}}
def tau2 : Set := {∅, {b}, {c}, {b, c}, {a, b, c}}
def tau3 : Set := {∅, {a}, {a, b}, {a, c}}
def tau4 : Set := {∅, {a, c}, {b, c}, {c}, {a, b, c}}

-- Define the concept of a topology on the set X
def is_topology (τ : Set) : Prop := 
  (X ∈ τ ∧ ∅ ∈ τ) ∧
  (∀ s ⊆ τ, ⋃₀ s ∈ τ) ∧
  (∀ s ⊆ τ, ⋂₀ s ∈ τ)

-- Statements to prove
theorem tau2_is_topology : is_topology tau2 := sorry
theorem tau4_is_topology : is_topology tau4 := sorry

end tau2_is_topology_tau4_is_topology_l580_580272


namespace dominic_speed_proof_l580_580187

def dominic_distance : ℝ := 184
def dominic_time : ℝ := 8
def dominic_average_speed : ℝ := dominic_distance / dominic_time

theorem dominic_speed_proof : dominic_average_speed = 23 :=
by
  sorry

end dominic_speed_proof_l580_580187


namespace coloring_triangles_l580_580858

theorem coloring_triangles (n : ℕ) (k : ℕ) (h_n : n = 18) (h_k : k = 6) :
  (Nat.choose n k) = 18564 :=
by
  rw [h_n, h_k]
  sorry

end coloring_triangles_l580_580858


namespace y_work_completion_time_l580_580108

noncomputable def work_completion_days : ℕ :=
  let d := 5 * 3 in d

theorem y_work_completion_time
  (x_work_days : ℕ)
  (y_initial_work_days : ℕ)
  (x_remaining_work_days : ℕ) :
  x_work_days = 18 →
  y_initial_work_days = 5 →
  x_remaining_work_days = 12 →
  work_completion_days = 15 :=
by
  intros h1 h2 h3
  simp [work_completion_days, h1, h2, h3]
  sorry

end y_work_completion_time_l580_580108


namespace james_100m_time_l580_580684

def john_time : ℝ := 13
def john_initial_distance : ℝ := 4
def james_speed_advantage : ℝ := 2
def james_initial_distance : ℝ := 10
def james_initial_time : ℝ := 2

theorem james_100m_time : true := by
  -- John's specifications
  let john_total_distance := 100
  let john_top_speed_distance := john_total_distance - john_initial_distance
  let john_top_speed_time := john_time - 1
  let john_top_speed := john_top_speed_distance / john_top_speed_time
  
  -- James's specifications
  let james_top_speed := john_top_speed + james_speed_advantage
  let james_remaining_distance := john_total_distance - james_initial_distance
  let james_remaining_time := james_remaining_distance / james_top_speed
  let james_total_time := james_initial_time + james_remaining_time
  
  -- The condition to prove
  have : james_total_time = 11 := sorry
  
  exact trivial

end james_100m_time_l580_580684


namespace james_time_to_run_100_meters_l580_580692

theorem james_time_to_run_100_meters
  (john_time_to_run_100_meters : ℕ → Prop)
  (john_first_4_meters : nat := 4)
  (john_total_time : nat := 13)
  (james_first_10_meters_time : nat := 2)
  (james_top_speed_faster_by : nat := 2):
  john_time_to_run_100_meters john_total_time → 
  (john_first_4_meters = 4) →
  ∀ n, (john_total_time - 1) = n / 8 →
  ∀ m, (100 - 10) = m / 10 →
  ∀ p, james_first_10_meters_time * 1 + p * 10 = 100 →
  (james_first_10_meters_time + p) = 11 :=
  sorry

end james_time_to_run_100_meters_l580_580692


namespace successive_monomial_removal_l580_580343

-- Define the polynomial P and its properties
variables {R : Type*} [linear_ordered_field R]
def has_real_root (P : R[X]) : Prop := ∃ x : R, P.eval x = 0

-- Main theorem statement
theorem successive_monomial_removal {P : R[X]} (h_root : has_real_root P)
  (a0_ne_zero : P.nat_degree ≠ 0 → P.coeff 0 ≠ 0) :
  ∃ (order_deletion : list (ℕ × R)), ∀ (Q : R[X]), 
      (Q ∈ intermediate_polynomials P order_deletion) → has_real_root Q :=
sorry

end successive_monomial_removal_l580_580343


namespace similar_triangles_height_l580_580053

theorem similar_triangles_height (h₁ h₂ : ℝ) 
  (similar : ∀ (A₁ B₁ C₁ A₂ B₂ C₂ : Triangle), 
                (∃ k, k = 3 ∧ A₁ ≈ A₂ ∧ B₁ ≈ B₂ ∧ C₁ ≈ C₂) →
                (area A₁ / area A₂ = 1 / 9)) 
  (height_smaller : h₁ = 5)
  (area_ratio : area (Triangle.mk A₁ B₁ C₁) / area (Triangle.mk A₂ B₂ C₂) = 1 / 9) :
  h₂ = 15 := 
sorry

end similar_triangles_height_l580_580053


namespace height_of_larger_triangle_l580_580059

-- Definitions from the conditions
variables (height_small height_large : ℝ)
variables (area_ratio : ℝ)
variables (k : ℝ)

-- Given conditions
def triangles_similar : Prop := area_ratio = 9
def height_small_defined : Prop := height_small = 5
def scale_factor : Prop := k = real.sqrt area_ratio

-- Proof problem statement
theorem height_of_larger_triangle
  (h_similar : triangles_similar)
  (h_height_small : height_small_defined)
  (h_scale_factor : scale_factor) :
  height_large = 15 := sorry

end height_of_larger_triangle_l580_580059


namespace find_AD_in_triangle_ABC_l580_580309

noncomputable def triangle_abc (A B C D : Type*) (AB CD AD : ℝ) :=
  ∃ (angle_BAD angle_ABC angle_BCD : Type*),
  angle_BAD = 60 ∧ angle_ABC = 30 ∧ angle_BCD = 30 ∧
  AB = 15 ∧ CD = 8 ∧ AD = 3.5

theorem find_AD_in_triangle_ABC
  (A B C D : Type*) 
  (angle_BAD angle_ABC angle_BCD : Type*)
  (AB CD AD : ℝ) 
  (h : triangle_abc A B C D AB CD AD) :
  AD = 3.5 := by 
  sorry

end find_AD_in_triangle_ABC_l580_580309


namespace cone_volume_l580_580023

theorem cone_volume (p q : ℕ) (a α : ℝ) :
  V = (2 * π * a^3) / (3 * (Real.sin (2 * α)) * (Real.cos (180 * q / (p + q)))^2 * (Real.cos α)) :=
sorry

end cone_volume_l580_580023


namespace right_triangle_sin_Z_l580_580296

open Real

theorem right_triangle_sin_Z (XYZ : Triangle) (angleX angleY angleZ : ℝ) (sinX cosY : ℝ) :
  XYZ.right_triangle ∧ sinX = 3 / 5 ∧ cosY = 0 →
  sin angleZ = 3 / 5 :=
by
  sorry

end right_triangle_sin_Z_l580_580296


namespace find_first_offset_l580_580561

theorem find_first_offset (d b : ℝ) (Area : ℝ) :
  d = 22 → b = 6 → Area = 165 → (first_offset : ℝ) → 22 * (first_offset + 6) / 2 = 165 → first_offset = 9 :=
by
  intros hd hb hArea first_offset heq
  sorry

end find_first_offset_l580_580561


namespace period_of_f_is_4_and_f_2pow_n_zero_l580_580935

noncomputable def f : ℝ → ℝ := sorry

variables (hf_diff : differentiable ℝ f)
          (hf_nonzero : ∃ x, f x ≠ 0)
          (hf_odd_2 : ∀ x, f (x + 2) = -f (-x - 2))
          (hf_even_2x1 : ∀ x, f (2 * x + 1) = f (-(2 * x + 1)))

theorem period_of_f_is_4_and_f_2pow_n_zero (n : ℕ) (hn : 0 < n) :
  (∀ x, f (x + 4) = f x) ∧ f (2^n) = 0 :=
sorry

end period_of_f_is_4_and_f_2pow_n_zero_l580_580935


namespace intersection_A_B_l580_580266

-- Definitions of the sets A and B.
def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | x > 0}

-- The theorem we want to prove.
theorem intersection_A_B : A ∩ B = {1, 2} :=
by 
  sorry

end intersection_A_B_l580_580266


namespace misha_is_lying_l580_580134

theorem misha_is_lying
  (truth_tellers_scores : Fin 9 → ℕ)
  (h_all_odd : ∀ i, truth_tellers_scores i % 2 = 1)
  (total_scores_truth_tellers : (Fin 9 → ℕ) → ℕ)
  (h_sum_scores : total_scores_truth_tellers truth_tellers_scores = 18) :
  ∀ (misha_score : ℕ), misha_score = 2 → misha_score % 2 = 1 → False :=
by
  intros misha_score hms hmo
  sorry

end misha_is_lying_l580_580134


namespace max_value_of_AP_l580_580600

-- Define the problem
theorem max_value_of_AP
  (A B C P : ℝ × ℝ)
  (hAB : dist A B = 2 * √3)
  (hBC : dist B C = 2 * √3)
  (hCA : dist C A = 2 * √3)
  (h : |((P.1 - A.1) - (B.1 - A.1) - (C.1 - A.1), (P.2 - A.2) - (B.2 - A.2) - (C.2 - A.2))| = 1) :
  ∃ P, |(P.1 - A.1, P.2 - A.2)| ≤ 7 := sorry

end max_value_of_AP_l580_580600


namespace alloy_mixing_l580_580118

theorem alloy_mixing (x : ℕ) :
  (2 / 5) * 60 + (1 / 5) * x = 44 → x = 100 :=
by
  intros h1
  sorry

end alloy_mixing_l580_580118


namespace superVisibleFactorNumbers_count_l580_580485

def isSuperVisibleFactorNumber (n : ℕ) : Prop :=
  ∀ d : ℕ, d ≠ 0 → d ∈ (Int.digits 10 n) → n % d = 0 ∧ n % (List.sum (Int.digits 10 n)) = 0

theorem superVisibleFactorNumbers_count :
  {n : ℕ | 200 ≤ n ∧ n ≤ 250 ∧ isSuperVisibleFactorNumber n}.toFinset.card = 6 :=
sorry

end superVisibleFactorNumbers_count_l580_580485


namespace find_m_for_integer_solution_l580_580558

theorem find_m_for_integer_solution :
  ∀ (m x : ℤ), (x^3 - m*x^2 + m*x - (m^2 + 1) = 0) → (m = -3 ∨ m = 0) :=
by
  sorry

end find_m_for_integer_solution_l580_580558


namespace diana_can_paint_statues_l580_580185

theorem diana_can_paint_statues (total_paint : ℚ) (paint_per_statue : ℚ) 
  (h1 : total_paint = 3 / 6) (h2 : paint_per_statue = 1 / 6) : 
  total_paint / paint_per_statue = 3 :=
by
  sorry

end diana_can_paint_statues_l580_580185


namespace triangles_in_dodecagon_l580_580536

theorem triangles_in_dodecagon (n : ℕ) (h : n = 12) : (nat.choose n 3) = 220 :=
by
  rw h
  sorry

end triangles_in_dodecagon_l580_580536


namespace not_basic_logical_structure_l580_580157

def basic_structures : Set String := {"Sequential structure", "Conditional structure", "Loop structure"}

theorem not_basic_logical_structure : "Operational structure" ∉ basic_structures := by
  sorry

end not_basic_logical_structure_l580_580157


namespace value_of_A_l580_580216

theorem value_of_A (A C : ℤ) (h₁ : 2 * A - C + 4 = 26) (h₂ : C = 6) : A = 14 :=
by sorry

end value_of_A_l580_580216


namespace benny_placed_3_crayons_l580_580018

def initial_crayons : ℕ := 9
def current_crayons : ℕ := 12
def placed_crayons : ℕ := current_crayons - initial_crayons

theorem benny_placed_3_crayons (initial_crayons current_crayons placed_crayons : ℕ) (h1 : initial_crayons = 9) (h2 : current_crayons = 12) : placed_crayons = 3 := by
  rw [h1, h2]
  exact Nat.sub_eq_self.mpr rfl
  sorry

end benny_placed_3_crayons_l580_580018


namespace perimeter_ABFCDE_l580_580885

noncomputable def isosceles_right_triangle_perimeter (perimeter_ABCD : ℝ) (BF FC : ℝ) (angle_BFC : ℝ) : ℝ :=
  let side_length := perimeter_ABCD / 4
  let hypotenuse := real.sqrt (BF^2 + FC^2)
  2 * side_length + BF + FC + hypotenuse

theorem perimeter_ABFCDE (angle_BFC : ℝ) :
  isosceles_right_triangle_perimeter 48 12 12 (π / 2) = 60 + 12 * real.sqrt 2 :=
by
  sorry

end perimeter_ABFCDE_l580_580885


namespace train_speed_approx_l580_580150

open Real

def train_length : ℝ := 284
def crossing_time : ℝ := 18
def speed (d t : ℝ) := d / t

theorem train_speed_approx :
  (Real.round (speed train_length crossing_time * 100) * 0.01 = 15.78) :=
by
  sorry

end train_speed_approx_l580_580150


namespace polynomial_no_ab_term_l580_580280

theorem polynomial_no_ab_term (a b m : ℝ) :
  let p := 2 * (a^2 + a * b - 5 * b^2) - (a^2 - m * a * b + 2 * b^2)
  ∃ (m : ℝ), (p = a^2 - 12 * b^2) → (m = -2) :=
by
  let p := 2 * (a^2 + a * b - 5 * b^2) - (a^2 - m * a * b + 2 * b^2)
  intro h
  use -2
  sorry

end polynomial_no_ab_term_l580_580280


namespace number_of_divisors_S_l580_580701

noncomputable def omega := Complex.exp (2 * Real.pi * Complex.I / 91)

theorem number_of_divisors_S :
  let k := (Nat.totient 91)
  let a : Fin k → ℕ := fun i => 
    if h : (i : ℕ) < 91 then 
      ((Finset.filter (Nat.coprime 91) (Finset.range (91 + 1))).val.nthLe i (by simp [Nat.lt_succ_iff, h]))
    else 
      0
  let S := ∏ (i j : Fin k) (hij : i < j), (omega ^ (a j) - omega ^ (a i))
  Nat.factors_count S = 1054 := by sorry

end number_of_divisors_S_l580_580701


namespace sequence_formula_and_sum_l580_580940

def positive_sequence (a : ℕ → ℝ) := ∀ n, a n > 0
def arithmetic_sum (S : ℕ → ℝ) (a : ℕ → ℝ) := ∀ n, S n = ∑ i in finset.range n, a (i + 1)
def satisfies_condition (a : ℕ → ℝ) (S : ℕ → ℝ) := (a 1 = 2) ∧ (∀ n, n ≥ 2 → a n ^ 2 = 4 * S (n - 1) + 4 * n)

theorem sequence_formula_and_sum :
  ∃ (a : ℕ → ℝ), positive_sequence a ∧
  ∃ (S : ℕ → ℝ),
    arithmetic_sum S a ∧ 
    satisfies_condition a S ∧ 
    (∀ n, n ≥ 1 → a n = 2 * n) ∧ 
    (∑ k in finset.range 30, a (3 * k + 2) = 2730) :=
by
  sorry

end sequence_formula_and_sum_l580_580940


namespace find_x_l580_580001

theorem find_x (x : ℤ) (h_neg : x < 0)
  (h_median_mean : median ({20, 55, 68, x, 25} : set ℤ) = mean ({20, 55, 68, x, 25} : set ℤ) - 8) :
  x = -3 := by
  sorry

end find_x_l580_580001


namespace similar_triangles_height_l580_580050

theorem similar_triangles_height (h₁ h₂ : ℝ) (ratio_areas : ℝ) 
  (h₁_eq : h₁ = 5) (ratio_areas_eq : ratio_areas = 1 / 9)
  (similar : h₂^2 = (√ratio_areas)^2 * h₁^2) : h₂ = 15 :=
by {
  have ratio_areas_pos : ratio_areas > 0 := by (simp [ratio_areas_eq]),
  have k := √ratio_areas,
  have k_eq : k = 3 := by {
    rw [ratio_areas_eq, sqrt_div, sqrt_one, sqrt_nat_eq_iff_eq_sq, one_div_eq_inv] at *,
    norm_num },
  have h₂_def : h₂ = 3 * h₁ := by rw [h₁_eq, mul_comm, k_eq],
  rw [h₂_def],
  norm_num,
}

end similar_triangles_height_l580_580050


namespace find_odd_nat_with_divisors_digit_sum_l580_580200

theorem find_odd_nat_with_divisors_digit_sum :
  ∃ x : ℕ, (500 < x) ∧ (x < 1000) ∧ (x % 2 = 1) ∧ 
  (∑ (d : ℕ) in (Finset.filter (λ d, d ∣ x) (Finset.range (x + 1))), (d % 10)) = 33 ∧ 
  x = 729 :=
by sorry

end find_odd_nat_with_divisors_digit_sum_l580_580200


namespace max_m_value_l580_580649

variables {x y m : ℝ}

theorem max_m_value (h1 : 4 * x + 3 * y = 4 * m + 5)
                     (h2 : 3 * x - y = m - 1)
                     (h3 : x + 4 * y ≤ 3) :
                     m ≤ -1 :=
sorry

end max_m_value_l580_580649


namespace integer_roots_abs_sum_l580_580573

theorem integer_roots_abs_sum (p q r n : ℤ) :
  (∃ n : ℤ, (∀ x : ℤ, x^3 - 2023 * x + n = 0) ∧ p + q + r = 0 ∧ p * q + q * r + r * p = -2023) →
  |p| + |q| + |r| = 102 :=
by
  sorry

end integer_roots_abs_sum_l580_580573


namespace circumscribed_circle_radius_l580_580307

-- Given: triangle ABC with given conditions
variables (A B C : Type) [Triangle A B C] 
variables [angleA_eq_60 : A.angle = 60]
variables (I : Type) [Incenter I A B C]
variables [distBI : dist B I = 3]
variables [distCI : dist C I = 4]

-- Theorem: radius of the circumscribed circle
theorem circumscribed_circle_radius :
  ∃ R : ℝ, R = sqrt (37 / 3) :=
sorry

end circumscribed_circle_radius_l580_580307


namespace parabola_focus_directrix_distance_l580_580242

-- Define statements given in the problem 
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 4 * p * x
def distance (x1 y1 x2 y2 : ℝ) : ℝ := sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Main theorem statement
theorem parabola_focus_directrix_distance 
  (a p : ℝ) 
  (h_parabola : parabola p 8 a) 
  (h_focus_distance : distance 8 a (p / 2) 0 = 10) : 
  p = 4 := 
sorry

end parabola_focus_directrix_distance_l580_580242


namespace jessies_current_weight_l580_580506

theorem jessies_current_weight (initial_weight lost_weight : ℝ) (h1 : initial_weight = 69) (h2 : lost_weight = 35) :
  initial_weight - lost_weight = 34 :=
by sorry

end jessies_current_weight_l580_580506


namespace height_of_larger_triangle_l580_580065

-- Definitions from the conditions
variables (height_small height_large : ℝ)
variables (area_ratio : ℝ)
variables (k : ℝ)

-- Given conditions
def triangles_similar : Prop := area_ratio = 9
def height_small_defined : Prop := height_small = 5
def scale_factor : Prop := k = real.sqrt area_ratio

-- Proof problem statement
theorem height_of_larger_triangle
  (h_similar : triangles_similar)
  (h_height_small : height_small_defined)
  (h_scale_factor : scale_factor) :
  height_large = 15 := sorry

end height_of_larger_triangle_l580_580065


namespace sum_of_coordinates_image_of_midpoint_l580_580738

def point (x y : ℝ) := (x, y)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def reflect_y_axis(p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def sum_of_coordinates(p : ℝ × ℝ) : ℝ :=
  p.1 + p.2

theorem sum_of_coordinates_image_of_midpoint :
  let A : ℝ × ℝ := point 3 2
  let B : ℝ × ℝ := point 13 16
  let N := midpoint A B
  let A' := reflect_y_axis A
  let B' := reflect_y_axis B
  let N' := midpoint A' B'
  sum_of_coordinates N' = 1 :=
by
  let A : ℝ × ℝ := point 3 2
  let B : ℝ × ℝ := point 13 16
  let N := midpoint A B
  let A' := reflect_y_axis A
  let B' := reflect_y_axis B
  let N' := midpoint A' B'
  show sum_of_coordinates N' = 1
  sorry

end sum_of_coordinates_image_of_midpoint_l580_580738


namespace probability_X_eq_Y_is_one_twentieth_l580_580864

open Real

noncomputable def probability_X_eq_Y : ℝ :=
  let interval : Set (ℝ × ℝ) := {p : ℝ × ℝ | (-10 * π ≤ p.1 ∧ p.1 ≤ 10 * π) ∧ (-10 * π ≤ p.2 ∧ p.2 ≤ 10 * π)}
  let valid_pairs : Set (ℝ × ℝ) := {p : ℝ × ℝ | sin (sin p.1) = sin (sin p.2)}
  let valid_pairs_in_interval := interval ∩ valid_pairs
  let total_pairs := (20 * π) * (20 * π)
  let valid_count := valid_pairs_in_interval.to_finset.card
  valid_count / total_pairs

-- The theorem to state the final result
theorem probability_X_eq_Y_is_one_twentieth :
  probability_X_eq_Y = 1 / 20 := by
  sorry

end probability_X_eq_Y_is_one_twentieth_l580_580864


namespace inequality_result_l580_580584

theorem inequality_result
  (a b : ℝ) 
  (x y : ℝ)
  (h1 : 1 < a)
  (h2 : a < b)
  (h3 : a^x + b^y ≤ a^(-x) + b^(-y)) :
  x + y ≤ 0 :=
sorry

end inequality_result_l580_580584


namespace base9_to_base10_l580_580697

theorem base9_to_base10 : Nat.ofDigits 9 [3, 5, 6, 2] = 2648 := by
  sorry

end base9_to_base10_l580_580697


namespace remainder_of_polynomials_l580_580910

noncomputable def remainder : ℂ → ℂ :=
  λ x, x^60 + x^45 + x^30 + x^15 + 1

theorem remainder_of_polynomials :
  let β := Complex.exp (2 * Real.pi * Complex.I / 4) in
  (β^4 = 1 ∧ β^3 + β^2 + β + 1 = 0) → remainder β = 5 := by
{sorry}

end remainder_of_polynomials_l580_580910


namespace avg_rate_of_change_l580_580628

def f (x : ℝ) : ℝ := 2 * x + 1

theorem avg_rate_of_change : 
  (f(2) - f(0)) / (2 - 0) = 2 := by
  sorry

end avg_rate_of_change_l580_580628


namespace smallest_root_abs_eq_six_l580_580434

theorem smallest_root_abs_eq_six : 
  (∃ x : ℝ, (abs (x - 1)) / (x^2) = 6 ∧ ∀ y : ℝ, (abs (y - 1)) / (y^2) = 6 → y ≥ x) → x = -1 / 2 := by
  sorry

end smallest_root_abs_eq_six_l580_580434


namespace find_f_6_l580_580936

noncomputable def f : ℝ → ℝ :=
  sorry

theorem find_f_6 :
  (∀ x < 0, f x = x^3 - 1) →
  (∀ x, -1 ≤ x ∧ x ≤ 1 → f (-x) = -f x) →
  (∀ x > (1/2 : ℝ), f(x + (1/2 : ℝ)) = f(x - (1/2 : ℝ))) →
  f 6 = 2 := 
  by
  intros h1 h2 h3
  sorry

end find_f_6_l580_580936


namespace element_in_exactly_two_pairs_l580_580440

noncomputable def pairs (n : ℕ) : Type := finset (fin n × fin n)

def has_common_element {n : ℕ} (P : pairs n) (i j : fin n) : Prop :=
  ∃ x : fin n, x ∈ P ∧ (i, x) ∈ P ∧ (j, x) ∈ P

theorem element_in_exactly_two_pairs (X : finset ℕ) (n : ℕ) (P : fin n → fin n × fin n) 
  (h1 : ∀ i j : fin n, i ≠ j → P i ≠ P j) 
  (h2 : ∀ i j : fin n, i ≠ j → (∃ k : fin n, k ∈ {i, j}) ↔ ∃ e, e ∈ X ∧ e ∈ {P i, P j}) :
  ∀ x ∈ X, (∃! i j : fin n, x ∈ {P i, P j}) :=
begin
  sorry
end

end element_in_exactly_two_pairs_l580_580440


namespace unique_real_y_l580_580179

def star (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y

theorem unique_real_y (y : ℝ) : (∃! y : ℝ, star 4 y = 10) :=
  by {
    sorry
  }

end unique_real_y_l580_580179


namespace height_of_larger_triangle_l580_580082

theorem height_of_larger_triangle 
  (area_ratio : ℝ)
  (height_small_triangle : ℝ)
  (similar_triangles : Prop)
  (height_large_triangle : ℝ) :
  area_ratio = 1 / 9 →
  height_small_triangle = 5 →
  similar_triangles →
  height_large_triangle = height_small_triangle * 3 :=
begin
  intros h_ratio h_height_small h_similar,
  rw h_ratio at *,
  rw h_height_small at *,
  exact eq.symm (mul_eq_mul_left_iff.1 (eq.trans (sqrt_eq (by norm_num) (by norm_num)) (by norm_num))),
sorry,
end

# The above code imports the necessary library, defines the theorem with the conditions and concludes with the height of the larger triangle.

end height_of_larger_triangle_l580_580082


namespace hyperbola_eccentricity_l580_580261

-- Define the conditions
variable (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
variable (h3 : (3 : ℝ) / a = (Real.sqrt 3) / b)

-- Define the goal
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : (3 : ℝ) / a = (Real.sqrt 3) / b) :
  let c := Real.sqrt (a^2 + b^2)
  in Real.sqrt (c^2 / a^2) = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end hyperbola_eccentricity_l580_580261


namespace geometric_sequence_properties_l580_580226

noncomputable def a_n (n : ℕ) : ℚ :=
if n = 0 then 1 / 4 else (1 / 4) * 2^(n-1)

def S_n (n : ℕ) : ℚ :=
(1/4) * (1 - 2^n) / (1 - 2)

theorem geometric_sequence_properties :
  (a_n 2 = 1 / 2) ∧ (∀ n : ℕ, 1 ≤ n → a_n n = 2^(n-3)) ∧ S_n 5 = 31 / 16 :=
by {
  sorry
}

end geometric_sequence_properties_l580_580226


namespace min_number_of_4_dollar_frisbees_l580_580815

theorem min_number_of_4_dollar_frisbees 
  (x y : ℕ) 
  (h1 : x + y = 60)
  (h2 : 3 * x + 4 * y = 200) 
  : y = 20 :=
sorry

end min_number_of_4_dollar_frisbees_l580_580815


namespace symmetric_points_sum_l580_580613

-- Define the points A and A' and the requirement for symmetry with respect to the y-axis.
structure Point where
  x : ℝ
  y : ℝ

def isSymmetricY (A A' : Point) : Prop :=
  A.y = A'.y ∧ A.x = -A'.x

-- Given conditions
def A : Point := {x := a, y := 1}
def A' : Point := {x := 5, y := b}

-- The theorem to prove
theorem symmetric_points_sum (a b : ℝ) 
  (hA : A = {x := a, y := 1}) 
  (hA' : A' = {x := 5, y := b}) 
  (hsym : isSymmetricY A A') :
  a + b = -4 :=
by
  sorry

end symmetric_points_sum_l580_580613


namespace frank_more_miles_than_jim_in_an_hour_l580_580314

theorem frank_more_miles_than_jim_in_an_hour
    (jim_distance : ℕ) (jim_time : ℕ)
    (frank_distance : ℕ) (frank_time : ℕ)
    (h_jim : jim_distance = 16)
    (h_jim_time : jim_time = 2)
    (h_frank : frank_distance = 20)
    (h_frank_time : frank_time = 2) :
    (frank_distance / frank_time) - (jim_distance / jim_time) = 2 := 
by
  -- Placeholder for the proof, no proof steps included as instructed.
  sorry

end frank_more_miles_than_jim_in_an_hour_l580_580314


namespace monthly_income_of_P_l580_580757

variable (P Q R : ℝ)

theorem monthly_income_of_P (h1 : (P + Q) / 2 = 5050) 
                           (h2 : (Q + R) / 2 = 6250) 
                           (h3 : (P + R) / 2 = 5200) : 
    P = 4000 := 
sorry

end monthly_income_of_P_l580_580757


namespace order_abcd_l580_580603

noncomputable def a : ℝ := 1.7 ^ 0.3
noncomputable def b : ℝ := 0.9 ^ 0.1
noncomputable def c : ℝ := Real.log 5 / Real.log 2
noncomputable def d : ℝ := Real.log 1.8 / Real.log 0.3

theorem order_abcd : c > a ∧ a > b ∧ b > d :=
by
  have ha : a = 1.7 ^ 0.3 := by rfl
  have hb : b = 0.9 ^ 0.1 := by rfl
  have hc : c = Real.log 5 / Real.log 2 := by rfl
  have hd : d = Real.log 1.8 / Real.log 0.3 := by rfl
  -- Prove the inequalities one by one
  have h1 : c > a := sorry
  have h2 : a > b := sorry
  have h3 : b > d := sorry
  exact ⟨h1, h2, h3⟩

end order_abcd_l580_580603


namespace find_cosine_of_smallest_angle_l580_580503

noncomputable def cos_smallest_angle (n : ℕ) (h1 : n ≥ 2) : ℝ :=
  let a := (n - 1 : ℝ)
  let b := (n : ℝ)
  let c := (n + 1 : ℝ)
  let y := 2 * (real.acos ((a*a + b*b - c*c) / (2 * a * b))) + 2 * real.pi / 3
  real.cos (real.acos ((a*a + b*b - c*c) / (2 * a * b)) : ℝ)

theorem find_cosine_of_smallest_angle (n : ℕ) (h1 : n ≥ 2) (h2 : ∃ x y : ℝ, y = 2 * x + (2 * real.pi / 3) ∧ 
  real.cos y = ((n-1)^2 + n^2 - (n+1)^2) / (2 * (n-1) * n)) : 
  cos_smallest_angle n h1 = 5 / 16 :=
by
  sorry

end find_cosine_of_smallest_angle_l580_580503


namespace zero_of_F_when_a_is_zero_range_of_a_if_P_and_Q_l580_580627

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x
noncomputable def g (a x : ℝ) : ℝ := Real.log (x^2 - 2*x + a)
noncomputable def F (a x : ℝ) : ℝ := f a x + g a x

theorem zero_of_F_when_a_is_zero (x : ℝ) : a = 0 → F a x = 0 → x = 3 := by
  sorry

theorem range_of_a_if_P_and_Q (a : ℝ) :
  (∀ x ∈ Set.Icc (1/4 : ℝ) (1/2 : ℝ), a - 1/x ≤ 0) ∧
  (∀ x : ℝ, (x^2 - 2*x + a) > 0) →
  1 < a ∧ a ≤ 2 := by
  sorry

end zero_of_F_when_a_is_zero_range_of_a_if_P_and_Q_l580_580627


namespace percentage_cut_l580_580349

-- Define conditions as constants
def latte_price := 4.00
def iced_coffee_price := 2.00
def lattes_per_week := 5
def iced_coffees_per_week := 3
def weeks_per_year := 52
def savings := 338.00

-- Proof goal
theorem percentage_cut : 
  (savings / (lattes_per_week * latte_price * weeks_per_year + iced_coffees_per_week * iced_coffee_price * weeks_per_year)) * 100 = 25 :=
by
  -- This is where you would typically build the proof
  sorry

end percentage_cut_l580_580349


namespace floor_10L_eq_105_l580_580331

-- Define a 3-digit number formed by digits a, b, c
def three_digit_number (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

-- Define the sum of its digits
def digit_sum (a b c : ℕ) : ℕ :=
  a + b + c

-- Define the quotient of number by the sum of its digits
def F (a b c : ℕ) : ℝ :=
  (three_digit_number a b c : ℝ) / (digit_sum a b c : ℝ)

-- Define L as the minimum quotient value with distinct digits a, b, and c
def L : ℝ :=
  sorry  -- Minimize the quotient over all valid (a, b, c)

-- The final statement to prove
theorem floor_10L_eq_105 : ⌊10 * L⌋ = 105 :=
sorry

end floor_10L_eq_105_l580_580331


namespace order_of_magnitude_l580_580950

noncomputable def a : ℝ := 2 ^ 1.1
noncomputable def b : ℝ := 3 ^ 0.6
noncomputable def c : ℝ := Real.logb (1 / 2) 3

theorem order_of_magnitude : a > b ∧ b > c := by
  sorry

end order_of_magnitude_l580_580950


namespace cos_B_right_triangle_l580_580549

/-- Statement of the problem:
Given a right triangle with sides AC=7 units and hypotenuse BC=25 units,
prove that cos(B) = 24/25.
-/
theorem cos_B_right_triangle (AC BC : ℕ) (h : AC = 7) (h1 : BC = 25) (right_triangle: right_triangle A B C) : 
  (cos B = 24 / 25) := by
  sorry

end cos_B_right_triangle_l580_580549


namespace trajectory_of_midpoint_slope_of_line_l580_580960
-- Import necessary library

-- Define midpoint of a line segment
def midpoint (A B : (ℝ × ℝ)) : (ℝ × ℝ) := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Problem 1: Define the trajectory of the midpoint M
theorem trajectory_of_midpoint (A B M : (ℝ × ℝ)) 
  (hB : B = (0, 3)) 
  (hA_on_circle : ∀ A, (A.1 + 1)^2 + A.2^2 = 4) 
  (hM : M = midpoint A B) 
  : M.1^2 + (M.2 - 1.5)^2 = 1 := 
sorry

-- Problem 2: Define the slope of the line l
theorem slope_of_line (A B : (ℝ × ℝ)) 
  (hB : B = (0, 3)) 
  (hA_on_circle : ∀ A, (A.1 + 1)^2 + A.2^2 = 4) 
  (length_of_chord : ∥A - B∥ = (2 * sqrt 19) / 5) 
  : ∃ k, (k = 3 + sqrt 22 / 2) ∨ (k = 3 - sqrt 22 / 2) := 
sorry

end trajectory_of_midpoint_slope_of_line_l580_580960


namespace new_train_distance_l580_580484

-- Given conditions
def distance_older_train : ℝ := 200
def percent_more : ℝ := 0.20

-- Conclusion to prove
theorem new_train_distance : (distance_older_train * (1 + percent_more)) = 240 := by
  -- Placeholder to indicate that we are skipping the actual proof steps
  sorry

end new_train_distance_l580_580484


namespace basketball_team_games_left_l580_580469

theorem basketball_team_games_left (played_games : ℕ) (won_percentage : ℝ) (additional_losses : ℕ) (target_win_percentage : ℝ) (total_games : ℕ) :
  played_games = 40 →
  won_percentage = 0.70 →
  additional_losses = 8 →
  target_win_percentage = 0.60 →
  let won_games := won_percentage * played_games in
  let lost_games := played_games - (⌊won_games⌋ : ℕ) in
  let total_costs := played_games + additional_losses in
  (⌊won_games⌋ : ℕ) = target_win_percentage * total_games →
  total_games - played_games = 7 :=
by
  intros h1 h2 h3 h4 won_games lost_games total_costs h5
  sorry

end basketball_team_games_left_l580_580469


namespace sum_of_b_n_l580_580236

-- Given conditions
variable S6 : ℕ
variable a5 : ℕ
variable S_n b_n : ℕ → ℕ

-- Defining the arithmetic sequence and sum operations
noncomputable def a_n (n : ℕ) := 3 * n - 2
noncomputable def b_n_def (n : ℕ) := 2 ^ (a_n n)

theorem sum_of_b_n (n : ℕ) (hS6: S6 = 51) (ha5: a5 = 13)
  (hSbn: S_n n = b_n n) : b_n n = 2 / 7 * (8 ^ n - 1) :=
by
  -- Skipping the proof
  sorry

end sum_of_b_n_l580_580236


namespace line_in_plane_l580_580111

-- Define a type for points
axiom Point : Type

-- Define a line as a set of points
def Line : Type := Set Point

-- Define a plane as a set of points
def Plane : Type := Set Point

-- Introducing variables l and α
variable (l : Line) (α : Plane)

-- The statement that needs to be proved
theorem line_in_plane : (∀ p : Point, p ∈ l → p ∈ α) ↔ l ⊆ α := 
by
  -- This theorem is what we need to prove, skipping the proof with sorry
  sorry

end line_in_plane_l580_580111


namespace polynomial_not_1_3_5_7_9_l580_580332

theorem polynomial_not_1_3_5_7_9 {a : ℕ → ℤ} {x1 x2 x3 x4 : ℤ} :
   (∀ i, 0 ≤ i → i ≤ k → x_i ∈ ℤ) →
   x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 →
   (P x1 = 2 ∧ P x2 = 2 ∧ P x3 = 2 ∧ P x4 = 2) →
   ∀ x : ℤ, P x ≠ 1 ∧ P x ≠ 3 ∧ P x ≠ 5 ∧ P x ≠ 7 ∧ P x ≠ 9
 := sorry

end polynomial_not_1_3_5_7_9_l580_580332


namespace james_100m_time_l580_580687

-- Definitions based on conditions
def john_total_time_to_run_100m : ℝ := 13 -- seconds
def john_first_second_distance : ℝ := 4 -- meters
def james_first_10m_time : ℝ := 2 -- seconds
def james_speed_increment : ℝ := 2 -- meters per second

-- Derived definitions based on conditions
def john_remaining_distance : ℝ := 100 - john_first_second_distance -- meters
def john_remaining_time : ℝ := john_total_time_to_run_100m - 1 -- seconds
def john_speed : ℝ := john_remaining_distance / john_remaining_time -- meters per second
def james_speed : ℝ := john_speed + james_speed_increment -- meters per second
def james_remaining_distance : ℝ := 100 - 10 -- meters
def james_time_for_remaining_distance : ℝ := james_remaining_distance / james_speed -- seconds
def james_total_time : ℝ := james_first_10m_time + james_time_for_remaining_distance -- seconds

-- Theorem statement
theorem james_100m_time : james_total_time = 11 := 
by 
  -- Place proof here
  sorry

end james_100m_time_l580_580687


namespace g_three_eights_l580_580546

noncomputable def g : ℝ → ℝ := sorry

axiom g_0 : g 0 = 0
axiom g_monotone : ∀ (x y : ℝ), 0 ≤ x → x < y → y ≤ 1 → g(x) ≤ g(y)
axiom g_midpoint : ∀ (x : ℝ), 0 ≤ x → x ≤ 1 → g(1 - x) = 1 - g(x)
axiom g_quarter : ∀ (x : ℝ), 0 ≤ x → x ≤ 1 → g(x / 4) = g(x) / 2

theorem g_three_eights : g (3 / 8) = 1 / 4 :=
sorry

end g_three_eights_l580_580546


namespace alcohol_solution_l580_580447

theorem alcohol_solution (x y : ℕ) (v_x v_y : ℝ) (v_new : ℝ) (a_x a_y : ℝ) (a_new : ℝ) : 
  v_x = 300 → 
  a_x = 0.10 → 
  a_y = 0.30 → 
  a_new = 0.15 → 
  v_new = 300 + y → 
  a_new * v_new = a_x * v_x + a_y * y → 
  y = 100 :=
begin
  sorry
end

end alcohol_solution_l580_580447


namespace correct_proposition_l580_580297

/-- In space, for a plane α and two coplanar lines m and n, 
the following proposition is true: 
If m ⊂ α and n is parallel to α, then m is parallel to n. -/
theorem correct_proposition (α : Plane) (m n : Line) 
  (h_m_in_alpha : m ⊂ α) (h_n_parallel_alpha : n ∥ α) (h_coplanar : coplanar m n) : 
  m ∥ n := 
sorry

end correct_proposition_l580_580297


namespace coefficient_of_determination_l580_580278

-- Define the observations and conditions for the problem
def observations (n : ℕ) := 
  {x : ℕ → ℝ // ∃ b a : ℝ, ∀ i : ℕ, i < n → ∃ y_i : ℝ, y_i = b * x i + a}

/-- 
  Given a set of observations (x_1, y_1), (x_2, y_2), ..., (x_n, y_n) 
  that satisfies the equation y_i = bx_i + a for i = 1, 2, ..., n, 
  prove that the coefficient of determination R² is 1.
-/
theorem coefficient_of_determination (n : ℕ) (obs : observations n) : 
  ∃ R_squared : ℝ, R_squared = 1 :=
sorry

end coefficient_of_determination_l580_580278


namespace log_50000_estimate_l580_580189

theorem log_50000_estimate :
  ∃ (a b : ℕ), (4 < log 10 50000) ∧ (log 10 50000 < 5) ∧ (a = 4) ∧ (b = 5) ∧ (a + b = 9) :=
by
  -- Here we specify the values a and b
  let a := 4
  let b := 5
  use a, b
  -- Specify the conditions as stated in the problem
  have h1 : 4 < log 10 50000 := sorry
  have h2 : log 10 50000 < 5 := sorry
  -- The required statement:
  exact ⟨h1, h2, rfl, rfl, by simp⟩

end log_50000_estimate_l580_580189


namespace computer_price_geometric_sequence_l580_580388

theorem computer_price_geometric_sequence (initial_price : ℕ) (years : ℕ) (price_decrease_ratio : ℚ) : 
  initial_price = 8100 → years = 15 → price_decrease_ratio = 1 / 3 → 
  let common_ratio := 2 / 3 in
  let n := (years / 5) + 1 in
  (initial_price * common_ratio^(n - 1) = 2400) :=
by
  intros h1 h2 h3
  simp only [h1, h2, h3]
  let common_ratio := 2 / 3
  let n := (15 / 5) + 1
  calc 
    8100 * common_ratio^(n - 1)
      = 8100 * common_ratio^3 : by sorry
      = 8100 * (2 / 3)^3 : by sorry
      = 8100 * 8 / 27 : by sorry
      = 2400 : by sorry

end computer_price_geometric_sequence_l580_580388


namespace solve_eqn_l580_580441

theorem solve_eqn :
  3 * log 5 2 + 2 - x = log 5 (3 ^ x - 5 ^ (2 - x))
  ∧  25 ^ (log 2 (sqrt (x + 3)) - 0.5 * log 2 (x ^ 2 - 9)) = sqrt (2 * (7 - x))
  → x = 2 := by 
  sorry

end solve_eqn_l580_580441


namespace candy_cost_l580_580312

theorem candy_cost :
  let jake_allowance := 4.0
  let fraction_given := 1 / 4
  let money_given := jake_allowance * fraction_given
  let candies_purchased := 5
  let cost_per_candy := money_given / candies_purchased
  cost_per_candy = 0.20 :=
by
  let jake_allowance := 4.0
  let fraction_given := 1 / 4
  let money_given := jake_allowance * fraction_given
  let candies_purchased := 5
  let cost_per_candy := money_given / candies_purchased
  show cost_per_candy = 0.20
  sorry

end candy_cost_l580_580312


namespace prime_or_four_no_square_div_factorial_l580_580904

theorem prime_or_four_no_square_div_factorial (n : ℕ) :
  (n * n ∣ n!) = false ↔ Nat.Prime n ∨ n = 4 := by
  sorry

end prime_or_four_no_square_div_factorial_l580_580904


namespace union_of_A_and_B_complement_of_A_intersect_B_intersection_of_A_and_C_l580_580604

open Set

def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | x^2 - 12*x + 20 < 0 }
def C (a : ℝ) : Set ℝ := { x | x < a }

theorem union_of_A_and_B :
  A ∪ B = { x : ℝ | 2 < x ∧ x < 10 } :=
sorry

theorem complement_of_A_intersect_B :
  ((univ \ A) ∩ B) = { x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) } :=
sorry

theorem intersection_of_A_and_C (a : ℝ) (h : (A ∩ C a).Nonempty) :
  a > 3 :=
sorry

end union_of_A_and_B_complement_of_A_intersect_B_intersection_of_A_and_C_l580_580604


namespace cube_surface_area_is_24_l580_580408

def edge_length : ℝ := 2

def surface_area_of_cube (a : ℝ) : ℝ := 6 * a * a

theorem cube_surface_area_is_24 : surface_area_of_cube edge_length = 24 := 
by 
  -- Compute the surface area of the cube with given edge length
  -- surface_area_of_cube 2 = 6 * 2 * 2 = 24
  sorry

end cube_surface_area_is_24_l580_580408


namespace zero_in_interval_l580_580180

noncomputable def f (x : ℝ) : ℝ := x - 3 + log x / log 3

theorem zero_in_interval : ∃ x : ℝ, 1 < x ∧ x < 3 ∧ f x = 0 :=
by
  -- Proof will be provided here.
  sorry

end zero_in_interval_l580_580180


namespace slope_angle_range_l580_580402

variables (a : ℝ) (α : ℝ)

def lineEquation := (x y : ℝ) → x + (a^2 + 1) * y + 1 = 0

theorem slope_angle_range 
  (h α_range : α ∈ Ico 0 π) 
  (h_tangent : Real.tan α = -1 / (a^2 + 1)) :
  α ∈ Ico (3 * Real.pi / 4) Real.pi :=
sorry

end slope_angle_range_l580_580402


namespace original_time_between_maintenance_checks_l580_580125

theorem original_time_between_maintenance_checks (x : ℝ) 
  (h1 : 2 * x = 60) : x = 30 := sorry

end original_time_between_maintenance_checks_l580_580125


namespace python_length_correct_l580_580507

def garden_snake_length : ℝ := 10.0
def boa_constrictor_length : ℝ := garden_snake_length / 7.0
def combined_length : ℝ := garden_snake_length + boa_constrictor_length
def python_length : ℝ := combined_length + 0.5 * combined_length

theorem python_length_correct : python_length = 17.1 := 
by
  sorry

end python_length_correct_l580_580507


namespace fixed_point_exists_l580_580729

theorem fixed_point_exists :
  ∃ x y, (∀ λ : ℝ, (λ + 2) * x - (λ - 1) * y + 6 * λ + 3 = 0) ↔ (x = -3 ∧ y = 3) :=
by
  sorry

end fixed_point_exists_l580_580729


namespace natural_numbers_solution_l580_580088

noncomputable def nat_triple_sum_gcd_prime_distinct_primes (a b c : ℕ) (p p1 p2 p3 : ℕ) :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  a + b + c = 406 ∧ 
  Nat.gcd a (Nat.gcd b c) = p ∧ Nat.prime p ∧ p > 2 ∧ 
  a = p * p1 ∧ b = p * p2 ∧ c = p * p3 ∧ 
  Nat.prime p1 ∧ Nat.prime p2 ∧ Nat.prime p3 ∧ 
  p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3

theorem natural_numbers_solution (a b c : ℕ) (p p1 p2 p3 : ℕ) 
  (h : nat_triple_sum_gcd_prime_distinct_primes a b c p p1 p2 p3) : 
  (a = 14 ∧ b = 21 ∧ c = 371) ∨ (a = 14 ∧ b = 91 ∧ c = 301) ∨ 
  (a = 14 ∧ b = 133 ∧ c = 259) ∨ (a = 58 ∧ b = 145 ∧ c = 203) :=
sorry

end natural_numbers_solution_l580_580088


namespace solve_equation1_solve_equation2_l580_580375

-- Let x be a real number
variable {x : ℝ}

-- The first equation and its solutions
def equation1 (x : ℝ) : Prop := (x - 1) ^ 2 - 25 = 0

-- Asserting that the solutions to the first equation are x = 6 or x = -4
theorem solve_equation1 (x : ℝ) : equation1 x ↔ x = 6 ∨ x = -4 :=
by
  sorry

-- The second equation and its solution
def equation2 (x : ℝ) : Prop := (1 / 4) * (2 * x + 3) ^ 3 = 16

-- Asserting that the solution to the second equation is x = 1/2
theorem solve_equation2 (x : ℝ) : equation2 x ↔ x = 1 / 2 :=
by
  sorry

end solve_equation1_solve_equation2_l580_580375


namespace modulus_of_root_unit_circle_l580_580945

theorem modulus_of_root_unit_circle 
  {n : ℤ} (hn : n ≥ 2) 
  {a : ℝ} (ha : 0 < a ∧ a < (n + 1) / (n - 1)) 
  {z : ℂ} (hz : z ^ (n + 1) - a * z ^ n + a * z - 1 = 0) : 
  abs z = 1 := 
sorry

end modulus_of_root_unit_circle_l580_580945


namespace sum_of_three_numbers_l580_580839

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h : a + (b * c) = (a + b) * (a + c)) : a + b + c = 1 :=
by
  sorry

end sum_of_three_numbers_l580_580839


namespace students_not_enrolled_in_any_subject_l580_580288

theorem students_not_enrolled_in_any_subject
  (total_students : ℕ)
  (math_students : ℕ)
  (chem_students : ℕ)
  (both_students : ℕ)
  (h_total : total_students = 150)
  (h_math : math_students = 90)
  (h_chem : chem_students = 70)
  (h_both : both_students = 20) : 
  (total_students - (math_students - both_students + chem_students - both_students + both_students) = 10) :=
by
  rw [h_total, h_math, h_chem, h_both]
  sorry

end students_not_enrolled_in_any_subject_l580_580288


namespace coefficient_x3_l580_580672

open Polynomial

noncomputable def polynomial : Polynomial ℤ := (2 * X + 1) * (X - 2) * (X + 3) * (X - 4)

theorem coefficient_x3 : coeff polynomial 3 = -5 :=
sorry

end coefficient_x3_l580_580672


namespace infinite_pals_exists_l580_580427

theorem infinite_pals_exists :
  ∃ (ABC XYZ : triangle) (M : midpoint ABC.BC) (W : midpoint XYZ.YZ),
  ABC.area = XYZ.area ∧
  (∃ A B C X Y Z : ℕ, 
    {ABC.AB, ABC.AM, ABC.AC} = {XYZ.XY, XYZ.XW, XYZ.XZ} ∧
    pairwise_rel_prime {ABC.AB, ABC.AM, ABC.AC} ∧
    pairwise_rel_prime {XYZ.XY, XYZ.XW, XYZ.XZ}) ∧
  infinite_pals ABC XYZ :=
begin
  sorry
end

end infinite_pals_exists_l580_580427


namespace maxRegions_four_planes_maxRegions_n_planes_l580_580022

noncomputable def maxRegions (n : ℕ) : ℕ :=
  1 + (n * (n + 1)) / 2

theorem maxRegions_four_planes : maxRegions 4 = 11 := by
  sorry

theorem maxRegions_n_planes (n : ℕ) : maxRegions n = 1 + (n * (n + 1)) / 2 := by
  sorry

end maxRegions_four_planes_maxRegions_n_planes_l580_580022


namespace three_pow_zero_eq_one_l580_580523

theorem three_pow_zero_eq_one : 3^0 = 1 :=
by {
  -- Proof would go here
  sorry
}

end three_pow_zero_eq_one_l580_580523


namespace derivative_y_l580_580453

variables (x : ℝ) (y : ℝ)

-- Define the condition y = 2 * x
def y_def : y = 2 * x := by
  sorry

-- The Lean statement for proving that y' = 2 given y = 2 * x
theorem derivative_y {x y : ℝ} (h : y = 2 * x) : has_deriv_at (λ x, y) 2 x := by
  rw [h]
  have : has_deriv_at (λ x, 2 * x) 2 x := by
    exact has_deriv_at.const_mul 2 (has_deriv_at_id x)
  exact this

end derivative_y_l580_580453


namespace range_of_a_l580_580325

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    (if x1 ≤ 1 then (-x1^2 + a*x1)
     else (a*x1 - 1)) = 
    (if x2 ≤ 1 then (-x2^2 + a*x2)
     else (a*x2 - 1))) → a < 2 :=
sorry

end range_of_a_l580_580325


namespace infinite_representable_and_nonrepresentable_terms_l580_580548

def a (n : ℕ) : ℕ :=
  2^n + 2^(n / 2)

def is_representable (k : ℕ) : Prop :=   
  -- A nonnegative integer is defined to be representable if it can
  -- be expressed as a sum of distinct terms from the sequence a(n).
  sorry  -- Definition will depend on the specific notion of representability

theorem infinite_representable_and_nonrepresentable_terms :
  (∃ᶠ n in at_top, is_representable (a n)) ∧ (∃ᶠ n in at_top, ¬is_representable (a n)) :=
sorry  -- This is the main theorem claiming infinitely many representable and non-representable terms.

end infinite_representable_and_nonrepresentable_terms_l580_580548


namespace max_value_l580_580618

def f (x a b : ℝ) : ℝ := - (x + a) / (b * x + 1)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f(x)

theorem max_value (a b : ℝ) (h : is_odd_function (λ x, f x a b)) :
  a = 0 → b = 0 → ∀ x, -1 ≤ x ∧ x ≤ 1 → x ≤ 1 :=
begin
  sorry
end

end max_value_l580_580618


namespace length_of_AB_l580_580616

-- Define the parabola and the line passing through the focus F
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def line (x y : ℝ) : Prop := y = x - 1

theorem length_of_AB : 
  (∃ F : ℝ × ℝ, F = (1, 0) ∧ line F.1 F.2) →
  (∃ A B : ℝ × ℝ, parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
    line A.1 A.2 ∧ line B.1 B.2 ∧
    A ≠ B ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 64)) :=
by
  sorry

end length_of_AB_l580_580616


namespace system1_l580_580750

theorem system1 {x y : ℝ} 
  (h1 : x + y = 3) 
  (h2 : x - y = 1) : 
  x = 2 ∧ y = 1 :=
by
  sorry

end system1_l580_580750


namespace students_not_picked_l580_580790

theorem students_not_picked (total_students groups group_size : ℕ) (h1 : total_students = 64)
(h2 : groups = 4) (h3 : group_size = 7) :
total_students - groups * group_size = 36 :=
by
  sorry

end students_not_picked_l580_580790


namespace number_of_special_five_letter_words_l580_580984

theorem number_of_special_five_letter_words : 
  ∃ n, n = 17576 ∧ 
       ∀ (letters : Fin 26 → Char), 
       ∀ (word : Fin 5 → Char), 
       (word 0 = word 4 ∧ word 1 = word 3) → 
       ((n = 26 * 26 * 26) ∧ n = 17576) :=
begin
  existsi 17576,
  split,
  { refl },
  { intros letters word h,
    sorry  -- proof of the condition using 26^3 would go here
  }
end

end number_of_special_five_letter_words_l580_580984


namespace starting_number_l580_580413

theorem starting_number (n : ℕ) (h1 : 200 ≥ n) (h2 : 33 = ((200 / 3) - (n / 3))) : n = 102 :=
by
  sorry

end starting_number_l580_580413


namespace range_of_k_l580_580623

theorem range_of_k (n : ℕ) (k : ℝ) (k_pos : 0 < k) :
  (∀ x : ℝ, x ∈ Ioc (2 * n - 1 : ℝ) (2 * n + 1 : ℝ) → |x - 2 * n| = k * sqrt x) →
  k ≤ 1 / sqrt (2 * n + 1) :=
by
  intros h
  sorry

end range_of_k_l580_580623


namespace tom_pays_1340_l580_580030

def vaccine_cost := 45
def number_of_vaccines := 10
def doctor_visit_cost := 250
def insurance_coverage := 0.8
def trip_cost := 1200

def total_vaccine_cost := vaccine_cost * number_of_vaccines
def total_medical_cost := total_vaccine_cost + doctor_visit_cost
def insurance_cover_amount := total_medical_cost * insurance_coverage
def amount_paid_after_insurance := total_medical_cost - insurance_cover_amount
def total_amount_tom_pays := amount_paid_after_insurance + trip_cost

theorem tom_pays_1340 :
  total_amount_tom_pays = 1340 :=
by
  sorry

end tom_pays_1340_l580_580030


namespace binom_product_is_zero_l580_580209

def binom (a : ℝ) (k : ℕ) : ℝ :=
  if k = 0 then 1 else (List.prod (List.map (λ i => a - i) (List.range k)) / Nat.factorial k)

theorem binom_product_is_zero (k : ℕ) (a1 a2 : ℝ) (h1 : a1 = 1) (h2 : a2 = 2) (hk : k = 3) :
  binom a1 k * binom a2 k = 0 := by
  sorry

end binom_product_is_zero_l580_580209


namespace similar_triangles_height_l580_580078

theorem similar_triangles_height
  (a b : ℕ)
  (area_ratio: ℕ)
  (height_smaller : ℕ)
  (height_relation: height_smaller = 5)
  (area_relation: area_ratio = 9)
  (similarity: a / b = 1 / area_ratio):
  (∃ height_larger : ℕ, height_larger = 15) :=
by
  sorry

end similar_triangles_height_l580_580078


namespace parabola_properties_l580_580937

noncomputable def parabola_focus_distance (p : ℝ) : ℝ :=
  sqrt ((p / 2)^2 + (4 + sqrt (2 * p * 4))^2)

theorem parabola_properties :
  (∀ p > 0, parabola_focus_distance p = 5 → p = 2 ∧ ∀ x y, y^2 = 2 * p * x → y^2 = 4 * x)
  ∧ (∀ m t, ∃ (x y : ℝ), (x = m * (y + 4) + t) ∧ (y^2 = 4 * x) ∧
    (t = 4 * m + 8) ∨ (t = -4 * m + 4) → 
    (x, y) = (8, -4)) :=
by sorry

end parabola_properties_l580_580937


namespace shaded_area_correct_l580_580468

noncomputable def total_shaded_area (floor_length : ℝ) (floor_width : ℝ) (tile_size : ℝ) (circle_radius : ℝ) : ℝ :=
  let tile_area := tile_size ^ 2
  let circle_area := Real.pi * circle_radius ^ 2
  let shaded_area_per_tile := tile_area - circle_area
  let floor_area := floor_length * floor_width
  let number_of_tiles := floor_area / tile_area
  number_of_tiles * shaded_area_per_tile 

theorem shaded_area_correct : total_shaded_area 12 15 2 1 = 180 - 45 * Real.pi := sorry

end shaded_area_correct_l580_580468


namespace line_in_y_equals_mx_plus_b_form_l580_580481

theorem line_in_y_equals_mx_plus_b_form :
  ∃ m b, (∀ x y, (⟨2, -1⟩ : ℝ × ℝ) • (⟨x, y⟩ - ⟨3, -4⟩) = 6 → y = m * x + b) ∧ (m = 2 ∧ b = -16) :=
begin
  use 2,
  use -16,
  split,
  {
    intros x y h,
    -- Steps to show y = 2 * x - 16 based on the initial condition will go here,
    -- but we'll leave it as "sorry" to focus on setting up the statement correctly.
    sorry,
  },
  split,
  { refl },
  { refl },
end

end line_in_y_equals_mx_plus_b_form_l580_580481


namespace sin_cos_identity_max_area_l580_580285

-- Problem 1: Prove that for triangle ABC with cos A = 1/3, 
-- the value of sin^2((B + C)/2) + cos 2A = -1/9
theorem sin_cos_identity
  (A B C a b c : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : A = π - B - C)
  (h5 : cos A = 1 / 3) :
  sin^2 ((B + C) / 2) + cos (2 * A) = -1 / 9 :=
sorry

-- Problem 2: Given a = √3 and cos A = 1/3, prove that the maximum area of triangle ABC is 3√2 / 4
theorem max_area
  (A B C a b c : ℝ)
  (h1 : a = sqrt 3)
  (h2 : cos A = 1 / 3)
  (h3 : b = 3 / 2)
  (h4 : c = 3 / 2) :
  (1 / 2) * b * c * sin A = (3 sqrt 2) / 4 :=
sorry

end sin_cos_identity_max_area_l580_580285


namespace solve_problem_l580_580116

-- Define the polynomial p(x)
noncomputable def p (x : ℂ) : ℂ := x^2 - x + 1

-- Define the root condition
def is_root (α : ℂ) : Prop := p (p (p (p α))) = 0

-- Define the expression to evaluate
noncomputable def expression (α : ℂ) : ℂ := (p α - 1) * p α * p (p α) * p (p (p α))

-- State the theorem asserting the required equality
theorem solve_problem (α : ℂ) (hα : is_root α) : expression α = -1 :=
sorry

end solve_problem_l580_580116


namespace starting_number_l580_580414

theorem starting_number (n : ℕ) (h1 : 200 ≥ n) (h2 : 33 = ((200 / 3) - (n / 3))) : n = 102 :=
by
  sorry

end starting_number_l580_580414


namespace units_digit_of_10_digit_even_number_with_sum_89_is_8_l580_580510

-- Definitions based on the given conditions
def is_10_digit_number (n : ℕ) : Prop := n ≥ 10^9 ∧ n < 10^10
def is_even (n : ℕ) : Prop := n % 2 = 0
def sum_of_digits (n : ℕ) : ℕ := (nat.digits 10 n).sum

-- Main theorem statement
theorem units_digit_of_10_digit_even_number_with_sum_89_is_8 (n : ℕ) 
  (h1 : is_10_digit_number n) 
  (h2 : is_even n) 
  (h3 : sum_of_digits n = 89) : 
  n % 10 = 8 :=
sorry  -- proof to be filled in


end units_digit_of_10_digit_even_number_with_sum_89_is_8_l580_580510


namespace similar_triangles_height_l580_580042

open_locale classical

theorem similar_triangles_height
  (h_small : ℝ)
  (A_small A_large : ℝ)
  (area_ratio : ℝ)
  (h_large : ℝ) :
  A_small / A_large = 1 / 9 →
  h_small = 5 →
  area_ratio = A_large / A_small →
  area_ratio = 9 →
  h_large = h_small * sqrt area_ratio →
  h_large = 15 :=
by
  sorry

end similar_triangles_height_l580_580042


namespace photos_taken_by_Octavia_l580_580291

variable (O J : ℕ)

theorem photos_taken_by_Octavia
  (h1 : J = 24 + 12)
  (h2 : O + J - 24 = 48) :
  O = 36 :=
by
  rw [h1] at h2
  linarith

end photos_taken_by_Octavia_l580_580291


namespace messenger_speed_l580_580860

theorem messenger_speed (x : ℕ) :
  (∀ (team_len team_speed : ℕ) (time_taken : ℚ),
     team_len = 6 ∧ team_speed = 5 ∧ time_taken = 1 / 2 →
     (6 / (x + 5 : ℚ) + 6 / (x - 5 : ℚ) = 1 / 2)) →
  x = 25 :=
by
  intro h
  specialize h 6 5 (1 / 2)
  cases h ⟨rfl, rfl, rfl⟩
  sorry

end messenger_speed_l580_580860


namespace find_j_l580_580095

theorem find_j (n j : ℕ) (h1 : n % j = 28) (h2 : (n : ℝ) / j = 142.07) : j = 400 :=
by
  sorry

end find_j_l580_580095


namespace max_take_home_income_l580_580287

theorem max_take_home_income : ∃ y, even y ∧ y = 50 ∧ (∀ z, even z → (1000*z - 10*z^2 ≤ 1000*50 - 10*50^2)) :=
by
  sorry

end max_take_home_income_l580_580287


namespace find_second_apartment_rent_l580_580483

theorem find_second_apartment_rent :
  let rent_a := 800
  let utilities_a := 260
  let dist_a := 31
  let utilities_b := 200
  let dist_b := 21
  let cost_per_mile := 0.58
  let workdays_per_month := 20
  let cost_difference := 76
  let total_cost_a := rent_a + utilities_a + dist_a * workdays_per_month * cost_per_mile
  in ∃ R : ℝ, 
    total_cost_a - (R + utilities_b + dist_b * workdays_per_month * cost_per_mile) = cost_difference
    ∧ R = 900 :=
by
  let rent_a := 800
  let utilities_a := 260
  let dist_a := 31
  let utilities_b := 200
  let dist_b := 21
  let cost_per_mile := 0.58
  let workdays_per_month := 20
  let cost_difference := 76
  let total_cost_a := rent_a + utilities_a + dist_a * workdays_per_month * cost_per_mile
  let total_cost_a := (800 : ℝ) + 260 + 31 * 20 * 0.58
  have h_total_cost_a : total_cost_a = 1419.60, by norm_num
  let total_cost_b (R : ℝ) := (R : ℝ) + 200 + 21 * 20 * 0.58
  have h_total_cost_b_eq : ∀ (R : ℝ), total_cost_b R = (R : ℝ) + 443.60, from
  λR, by norm_num
  use 900
  unfold total_cost_a
  split
  norm_num
  norm_num
  unfold total_cost_a
  split
  norm_num
  norm_num
  sorry

end find_second_apartment_rent_l580_580483


namespace min_likes_both_l580_580356

theorem min_likes_both (total people mozart beethoven : ℕ) (h₁ : total = 120) (h₂ : mozart = 95) (h₃ : beethoven = 80) :
  ∃ x, x = mozart + beethoven - total ∧ x = 40 :=
by {
  use mozart + beethoven - total,
  split;
  repeat { sorry },
}

end min_likes_both_l580_580356


namespace transaction_gain_per_year_l580_580099

theorem transaction_gain_per_year
  (P : ℕ) (T : ℕ) (R1 R2 : ℕ) :
  P = 8000 →
  T = 2 →
  R1 = 4 →
  R2 = 6 →
  let gain_per_year := (P * R2 * T / 100 - P * R1 * T / 100) / T
  in gain_per_year = 160 :=
by
  intros
  sorry

end transaction_gain_per_year_l580_580099


namespace triangle_angles_geometric_progression_l580_580652

-- Theorem: If the sides of a triangle whose angles form an arithmetic progression are in geometric progression, then all three angles are 60°.
theorem triangle_angles_geometric_progression (A B C : ℝ) (a b c : ℝ)
  (h_arith_progression : 2 * B = A + C)
  (h_sum_angles : A + B + C = 180)
  (h_geo_progression : (a / b) = (b / c))
  (h_b_angle : B = 60) :
  A = 60 ∧ B = 60 ∧ C = 60 :=
by
  sorry

end triangle_angles_geometric_progression_l580_580652


namespace cannot_solve_for_X_l580_580467

theorem cannot_solve_for_X (X : ℕ) (x : ℕ) (h₁ : 90 = (x * 0.5 * X)) (h₂ : 50 = 50 * X) : false :=
by 
  sorry

end cannot_solve_for_X_l580_580467


namespace height_of_larger_triangle_l580_580084

theorem height_of_larger_triangle 
  (area_ratio : ℝ)
  (height_small_triangle : ℝ)
  (similar_triangles : Prop)
  (height_large_triangle : ℝ) :
  area_ratio = 1 / 9 →
  height_small_triangle = 5 →
  similar_triangles →
  height_large_triangle = height_small_triangle * 3 :=
begin
  intros h_ratio h_height_small h_similar,
  rw h_ratio at *,
  rw h_height_small at *,
  exact eq.symm (mul_eq_mul_left_iff.1 (eq.trans (sqrt_eq (by norm_num) (by norm_num)) (by norm_num))),
sorry,
end

# The above code imports the necessary library, defines the theorem with the conditions and concludes with the height of the larger triangle.

end height_of_larger_triangle_l580_580084


namespace part_a_part_b_l580_580818

-- Part a: Prove for specific numbers 2015 and 2017
theorem part_a : ∃ (x y : ℕ), (2015^2 + 2017^2) / 2 = x^2 + y^2 := sorry

-- Part b: Prove for any two different odd natural numbers
theorem part_b (a b : ℕ) (h1 : a ≠ b) (h2 : a % 2 = 1) (h3 : b % 2 = 1) :
  ∃ (x y : ℕ), (a^2 + b^2) / 2 = x^2 + y^2 := sorry

end part_a_part_b_l580_580818


namespace similar_triangles_height_l580_580039

open_locale classical

theorem similar_triangles_height
  (h_small : ℝ)
  (A_small A_large : ℝ)
  (area_ratio : ℝ)
  (h_large : ℝ) :
  A_small / A_large = 1 / 9 →
  h_small = 5 →
  area_ratio = A_large / A_small →
  area_ratio = 9 →
  h_large = h_small * sqrt area_ratio →
  h_large = 15 :=
by
  sorry

end similar_triangles_height_l580_580039


namespace height_of_larger_triangle_l580_580060

-- Definitions from the conditions
variables (height_small height_large : ℝ)
variables (area_ratio : ℝ)
variables (k : ℝ)

-- Given conditions
def triangles_similar : Prop := area_ratio = 9
def height_small_defined : Prop := height_small = 5
def scale_factor : Prop := k = real.sqrt area_ratio

-- Proof problem statement
theorem height_of_larger_triangle
  (h_similar : triangles_similar)
  (h_height_small : height_small_defined)
  (h_scale_factor : scale_factor) :
  height_large = 15 := sorry

end height_of_larger_triangle_l580_580060


namespace fibonacci_divisibility_l580_580753

noncomputable def fibonacci (n : ℕ) : ℕ :=
if n = 1 ∨ n = 2 then 1 else fibonacci (n - 1) + fibonacci (n - 2)

theorem fibonacci_divisibility (m : ℕ) (h : 0 < m) : 
  ∃ k : ℕ, (fibonacci k)^4 - fibonacci k - 2 ≡ 0 [MOD m] := by
  sorry

end fibonacci_divisibility_l580_580753


namespace similar_triangles_height_l580_580075

theorem similar_triangles_height
  (a b : ℕ)
  (area_ratio: ℕ)
  (height_smaller : ℕ)
  (height_relation: height_smaller = 5)
  (area_relation: area_ratio = 9)
  (similarity: a / b = 1 / area_ratio):
  (∃ height_larger : ℕ, height_larger = 15) :=
by
  sorry

end similar_triangles_height_l580_580075


namespace am_gm_inequality_example_l580_580636

theorem am_gm_inequality_example (x y : ℝ) (hx : x = 16) (hy : y = 64) : 
  (x + y) / 2 ≥ Real.sqrt (x * y) :=
by
  rw [hx, hy]
  sorry

end am_gm_inequality_example_l580_580636


namespace least_possible_sum_l580_580719

theorem least_possible_sum
  (a b x y z : ℕ)
  (hpos_a : 0 < a) (hpos_b : 0 < b)
  (hpos_x : 0 < x) (hpos_y : 0 < y)
  (hpos_z : 0 < z)
  (h : 3 * a = 7 * b ∧ 7 * b = 5 * x ∧ 5 * x = 4 * y ∧ 4 * y = 6 * z) :
  a + b + x + y + z = 459 :=
by
  sorry

end least_possible_sum_l580_580719


namespace minimum_handshakes_l580_580463

theorem minimum_handshakes (n : ℕ) (h : n = 30) (cond : ∀ i : ℕ, i < n → ∃ k : ℕ, k ≥ 3 ∧ i < n ∧ shake_hands i k) : ∃ (h_num : ℕ), h_num = 45 :=
by
  -- conditions setup
  sorry

end minimum_handshakes_l580_580463


namespace sum_of_fib_factorials_last_two_digits_l580_580184

-- Define the condition that factorials greater than 10 end in 00
def end_in_00_if_gt_10 {n : ℕ} (hn : n > 10) : (n ! % 100) = 0 := sorry

-- Define the factorials of the specific Fibonacci numbers
def fib_factorials := [1!, 1!, 2!, 3!, 5!, 8!, 13!, 21!, 34!, 55!, 89!, 144!]

-- Define a function to get the last two digits of a number
def last_two_digits (n : ℕ) : ℕ := n % 100

-- Calculate the sum of the last two digits of the factorials of the Fibonacci numbers
def sum_of_last_two_digits : ℕ := 
  fib_factorials.take 6.map last_two_digits.sum + 0 + 0 + 0 + 0 + 0 + 0

-- The statement to prove
theorem sum_of_fib_factorials_last_two_digits : sum_of_last_two_digits = 5 := by
  sorry

end sum_of_fib_factorials_last_two_digits_l580_580184


namespace min_handshakes_l580_580461

theorem min_handshakes (n m : ℕ) (h1 : n = 30) (h2 : m ≥ 3) :
  let total_handshakes := (n * m) / 2
  in total_handshakes = 45 :=
by
  simp [h1, h2]
  sorry

end min_handshakes_l580_580461


namespace modulus_of_given_complex_l580_580622

noncomputable def givenComplex : ℂ := 2 / (1 + Complex.i)

theorem modulus_of_given_complex :
  |givenComplex| = Real.sqrt 2 :=
sorry

end modulus_of_given_complex_l580_580622


namespace min_value_fraction_l580_580596

theorem min_value_fraction
  (a c : ℝ)
  (h1 : a > c)
  (h2 : ∃ x, f(x) = a * x^2 + 4 * x + c)
  (h3 : ∃ x, f(x) = 0) :
  ∀ a > c, ac = 4 -> (min (4 * a^2 + c^2) / (2 * a - c)) = 8 :=
by
  sorry

end min_value_fraction_l580_580596


namespace prove_B_is_guilty_l580_580310

variables (A B C : Prop)

def guilty_conditions (A B C : Prop) : Prop :=
  (A → ¬ B → C) ∧
  (C → B ∨ A) ∧
  (A → ¬ (A ∧ C)) ∧
  (A ∨ B ∨ C) ∧ 
  ¬ (¬ A ∧ ¬ B ∧ ¬ C)

theorem prove_B_is_guilty : guilty_conditions A B C → B :=
by
  intros h
  sorry

end prove_B_is_guilty_l580_580310


namespace problem1_problem2_problem3_problem4_l580_580526

-- Problem 1
theorem problem1 : (-8) + 10 - 2 + (-1) = -1 := by
  sorry

-- Problem 2
theorem problem2 : 12 - 7 * (-4) + 8 / (-2) = 36 := by
  sorry

-- Problem 3
theorem problem3 : ((1/2) + (1/3) - (1/6)) / (- (1/18)) = -12 := by
  sorry

-- Problem 4
theorem problem4 : -(1 :ℤ)^ 4 - (1 + 0.5) * (1 / 3) / (-(4 :ℤ)) ^ 2 = -33 / 32 := by
  sorry

end problem1_problem2_problem3_problem4_l580_580526


namespace nat_square_not_div_factorial_l580_580906

-- Define n as a natural number
def n : Nat := sorry  -- We assume n is given somewhere

-- Define a function to check if a number is prime
def is_prime (p : Nat) : Prop := sorry  -- Placeholder for prime checking function

-- The main theorem to prove
theorem nat_square_not_div_factorial (n : Nat) : (n = 4 ∨ is_prime n) → ¬ ((n * n) ∣ Nat.factorial n) := by
  sorry

end nat_square_not_div_factorial_l580_580906


namespace ticket_difference_l580_580517

theorem ticket_difference (won left : ℕ) (h1 : won = 48) (h2 : left = 32) :
  won - left = 16 :=
by
  rw [h1, h2]
  rfl

end ticket_difference_l580_580517


namespace interest_difference_example_l580_580492

def interest_difference (R dR : ℝ) : Prop :=
  let principal := 750
  let time := 2
  let original_interest := principal * R * time / 100
  let higher_interest := principal * (R + dR) * time / 100
  (higher_interest - original_interest = 60) → (dR = 4)

theorem interest_difference_example : interest_difference R dR := by
  sorry

end interest_difference_example_l580_580492


namespace area_between_parabola_and_circle_l580_580562

noncomputable def parabola := {p : ℝ × ℝ | p.2 ^ 2 = 2 * p.1}
noncomputable def circle := {p : ℝ × ℝ | p.2 ^ 2 = 4 * p.1 - p.1 ^ 2}

theorem area_between_parabola_and_circle :
    let S1 := 2 * (∫ x in (0 : ℝ)..(2 : ℝ), sqrt (4 * x - x^2) - sqrt (2 * x))
    let S2 := (4 * π) - S1
    S1 = 2 * (π - 4 * sqrt (2) / 3) ∧ S2 = 2 * (π + 8 / 3) :=
sorry

end area_between_parabola_and_circle_l580_580562


namespace find_k_l580_580708

variable {ℝ : Type} [Field ℝ] [OrderedRing ℝ] [AddCommGroup ℝ] [Module ℝ ℝ]

-- Variables and conditions identified from the problem
variables (a b c : ℝ^3)
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (hc : ∥c∥ = 1)
variables (hab : a ⬝ b = 0) (hac : a ⬝ c = 0)
variables (θ : ℝ) (hθ : θ = π / 4)

theorem find_k (hangle : angle b c = θ) :
  ∃ k : ℝ, a = k • (b × c) ∧ (k = sqrt 2 ∨ k = -sqrt 2) := 
  sorry

end find_k_l580_580708


namespace similar_triangles_height_l580_580054

theorem similar_triangles_height (h₁ h₂ : ℝ) 
  (similar : ∀ (A₁ B₁ C₁ A₂ B₂ C₂ : Triangle), 
                (∃ k, k = 3 ∧ A₁ ≈ A₂ ∧ B₁ ≈ B₂ ∧ C₁ ≈ C₂) →
                (area A₁ / area A₂ = 1 / 9)) 
  (height_smaller : h₁ = 5)
  (area_ratio : area (Triangle.mk A₁ B₁ C₁) / area (Triangle.mk A₂ B₂ C₂) = 1 / 9) :
  h₂ = 15 := 
sorry

end similar_triangles_height_l580_580054


namespace similar_triangles_height_ratio_l580_580068

theorem similar_triangles_height_ratio (area_ratio : ℝ) (h₁ : ℝ) (h₂ : ℝ) 
  (similar : Boolean) (h₁_value : h₁ = 5) (area_ratio_value : area_ratio = 9) :
  similar = true → area_ratio = (h₂ / h₁) ^ 2 → h₂ = 15 :=
by
  intro h_similar area_eq
  rw [h₁_value, area_ratio_value]
  sorry

end similar_triangles_height_ratio_l580_580068


namespace similar_triangles_height_l580_580043

open_locale classical

theorem similar_triangles_height
  (h_small : ℝ)
  (A_small A_large : ℝ)
  (area_ratio : ℝ)
  (h_large : ℝ) :
  A_small / A_large = 1 / 9 →
  h_small = 5 →
  area_ratio = A_large / A_small →
  area_ratio = 9 →
  h_large = h_small * sqrt area_ratio →
  h_large = 15 :=
by
  sorry

end similar_triangles_height_l580_580043


namespace polynomial_degree_ge_prime_minus_one_l580_580317

theorem polynomial_degree_ge_prime_minus_one
  (p : ℕ) (hp : Nat.Prime p)
  (f : Polynomial ℤ)
  (hf0 : f.eval 0 = 0)
  (hf1 : f.eval 1 = 1)
  (hmod : ∀ n : ℤ, f.eval n % p = 0 ∨ f.eval n % p = 1) :
  f.degree ≥ (p - 1) := 
sorry

end polynomial_degree_ge_prime_minus_one_l580_580317


namespace right_triangle_inscribed_circle_inequality_l580_580362

theorem right_triangle_inscribed_circle_inequality 
  {a b c r : ℝ} (h : a^2 + b^2 = c^2) (hr : r = (a + b - c) / 2) : 
  r ≤ (c / 2) * (Real.sqrt 2 - 1) :=
sorry

end right_triangle_inscribed_circle_inequality_l580_580362


namespace find_y_for_given_slope_l580_580948

open Real

theorem find_y_for_given_slope (y : ℝ) : 
  let P := (-3: ℝ, 5: ℝ)
      Q := (6: ℝ, y)
  in (y - 5) / (6 + 3) = (-5) / 3 -> y = -10 :=
by 
  intro h
  let P := (-3: ℝ, 5: ℝ)
  let Q := (6: ℝ, y)
  sorry

end find_y_for_given_slope_l580_580948


namespace sufficient_condition_for_g_increasing_l580_580455

-- Define the conditions
variable (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)

-- Define the functions f and g
def f (x : ℝ) : ℝ := a ^ x
def g (x : ℝ) : ℝ := (2 - a) * x ^ 3

-- The theorem statement
theorem sufficient_condition_for_g_increasing (h_decreasing_f : ∀ x y : ℝ, x < y → f x > f y) :
  (∀ x : ℝ, 0 < (2 - a) * x ^ 3) ∧ ¬ (∀ x : ℝ, g x is increasing) 
  → (∃ x : ℝ, 0 < (2 - a) * x ^ 3) :=
sorry

end sufficient_condition_for_g_increasing_l580_580455


namespace midpoint_trajectory_l580_580951

open Real

-- Definition of a circle equation as a predicate
def circle (h k r : ℝ) (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x - h)^2 + (y - k)^2 = r

-- Given condition: point P moves on the initial circle
def initial_circle (P : ℝ × ℝ) : Prop :=
  let (x₀, y₀) := P
  x₀^2 + y₀^2 - 8 * x₀ - 10 * y₀ + 21 = 0

-- Midpoint M is the midpoint of O (origin) and P
def is_midpoint (M P : ℝ × ℝ) : Prop :=
  let (x₀, y₀) := P
  let (x, y) := M
  x₀ = 2 * x ∧ y₀ = 2 * y

-- Proposition: Prove that the trajectory of midpoint M is the desired circle
theorem midpoint_trajectory :
  ∀ (P M : ℝ × ℝ),
    initial_circle P →
    is_midpoint M P →
    circle 2 (5 / 2) 5 M := by
  sorry

end midpoint_trajectory_l580_580951


namespace constant_term_in_binomial_expansion_l580_580282

theorem constant_term_in_binomial_expansion (n : ℕ) (h : 2^n = 64) :
  (∃ k, 0 < k ∧ k < n ∧ binom n k * (-1)^k = -20) :=
sorry

end constant_term_in_binomial_expansion_l580_580282


namespace find_a_9_l580_580953

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable (d : ℤ)

-- Assumptions and definitions from the problem
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n : ℕ, a (n + 1) = a n + d
def sum_of_arithmetic_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop := ∀ n : ℕ, S n = n * (a 1 + a n) / 2
def condition_one (a : ℕ → ℤ) : Prop := (a 1) + (a 2)^2 = -3
def condition_two (S : ℕ → ℤ) : Prop := S 5 = 10

-- Main theorem statement
theorem find_a_9 (h_arithmetic : arithmetic_sequence a d)
                 (h_sum : sum_of_arithmetic_sequence S a)
                 (h_cond1 : condition_one a)
                 (h_cond2 : condition_two S) : a 9 = 20 := 
sorry

end find_a_9_l580_580953


namespace daniel_total_spent_l580_580886

def price_magazine : ℝ := 0.85
def price_pencil : ℝ := 0.50
def price_notebook : ℝ := 1.25
def price_pens : ℝ := 3.75
def discount_rate : ℝ := 0.10
def store_credit : ℝ := 1.00
def tax_rate : ℝ := 0.07

theorem daniel_total_spent :
  let total_cost_before_discounts := price_magazine + price_pencil + price_notebook + price_pens in
  let discount_amount := discount_rate * total_cost_before_discounts in
  let total_cost_after_discount := total_cost_before_discounts - discount_amount in
  let total_cost_after_credit := total_cost_after_discount - store_credit in
  let sales_tax := tax_rate * total_cost_after_credit in
  let total_spent := total_cost_after_credit + sales_tax in
  total_spent ≈ 5.04 :=
begin
  sorry
end

end daniel_total_spent_l580_580886


namespace pairs_of_parallel_edges_l580_580148

-- Define a rectangular prism with specific dimensional relationships
structure RectangularPrism where
  w l h : ℝ
  length_twice_width : l = 2 * w
  height_thrice_width : h = 3 * w

-- Define the theorem to prove the number of pairs of parallel edges
theorem pairs_of_parallel_edges (P : RectangularPrism) : 
  ∃ n, n = 8 :=
by
  -- We need to add a proof here, but we just provide the statement as per instructions.
  sorry

end pairs_of_parallel_edges_l580_580148


namespace minimum_distance_from_curve_C_to_line_l_l580_580298

def curve_C (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in x^2 + (y - 2)^2 = 4

def line_l (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in y = x - 4

theorem minimum_distance_from_curve_C_to_line_l :
  ∀ P : ℝ × ℝ, curve_C P → ∃ Q : ℝ × ℝ, line_l Q ∧ dist P Q = 3 * sqrt 2 - 2 :=
sorry

end minimum_distance_from_curve_C_to_line_l_l580_580298


namespace similar_triangles_height_l580_580057

theorem similar_triangles_height (h₁ h₂ : ℝ) 
  (similar : ∀ (A₁ B₁ C₁ A₂ B₂ C₂ : Triangle), 
                (∃ k, k = 3 ∧ A₁ ≈ A₂ ∧ B₁ ≈ B₂ ∧ C₁ ≈ C₂) →
                (area A₁ / area A₂ = 1 / 9)) 
  (height_smaller : h₁ = 5)
  (area_ratio : area (Triangle.mk A₁ B₁ C₁) / area (Triangle.mk A₂ B₂ C₂) = 1 / 9) :
  h₂ = 15 := 
sorry

end similar_triangles_height_l580_580057


namespace find_cos_A_l580_580665

variable (A B C D : Type) [OrderedRing A]
variable (α : A) (a b c d : A)
variable [ConvexQuadrilateral A B C D]
variable (cos : A → A)

theorem find_cos_A 
  (h1 : Convex A B C D)
  (h2 : ∠A = ∠B = ∠C = α)
  (h3 : a = 150)
  (h4 : c = 150)
  (h5 : a ≠ c)
  (h6 : perimeter A B C D = 580) :
  cos α = 14 / 15 := 
sorry

end find_cos_A_l580_580665


namespace range_of_a_l580_580648

theorem range_of_a (h : ¬ ∃ x : ℝ, x^2 + (a-1) * x + 1 ≤ 0) : -1 < a ∧ a < 3 :=
sorry

end range_of_a_l580_580648


namespace hose_removal_rate_l580_580381

theorem hose_removal_rate (w l d : ℝ) (capacity_fraction : ℝ) (drain_time : ℝ) 
  (h_w : w = 60) 
  (h_l : l = 150) 
  (h_d : d = 10) 
  (h_capacity_fraction : capacity_fraction = 0.80) 
  (h_drain_time : drain_time = 1200) : 
  ((w * l * d * capacity_fraction) / drain_time) = 60 :=
by
  -- the proof is omitted here
  sorry

end hose_removal_rate_l580_580381


namespace domain_of_rational_function_l580_580909

open Set

def rational_function_domain (h : ℝ → ℝ) : Set ℝ := {x | ∃ y, h x = y}

theorem domain_of_rational_function :
  ∀ x : ℝ, x ∈ rational_function_domain (λ x, (x^3 - 3*x^2 + 6*x - 8) / (x^2 - 5*x + 6))
  ↔ x < 2 ∨ (2 < x ∧ x < 3) ∨ x > 3 :=
by
  intro x
  unfold rational_function_domain
  split
  { intro h
    cases h with y hy
    sorry },
  { intro h
    sorry }

end domain_of_rational_function_l580_580909


namespace correct_propositions_count_l580_580865

theorem correct_propositions_count :
  let p1 := (∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) ∧
            (∀ x : ℝ, x > 1 ∨ x < -1 → x^2 > 1)
  let p2 := (∀ x : ℝ, sin x ≤ 1) ∧
            (∀ a b : ℝ, a < b → a^2 < b^2)
  let p3 := ¬ ( ∃ x : ℝ, x^2 - x > 0 ) ↔ ( ∀ x : ℝ, x^2 - x ≤ 0 )
  let p4 := ∀ a b : ℝ, (a = 0 ↔ a + b.i = b.i) ∨ (a = 0 ∧ b ≠ 0)
  (p1 → false) ∧ (p2 → false) ∧ p3 ∧ (p4 → false) :=
by {
  sorry
}

end correct_propositions_count_l580_580865


namespace Bryan_skittles_50_l580_580167

variables (skittles_Ben : ℕ) (candies_Bryan : ℕ)

def Ben_MMs : skittles_Ben = 20 :=
by sorry

def Bryan_more_candies : candies_Bryan = skittles_Ben + 30 :=
by sorry

theorem Bryan_skittles_50 : candies_Bryan = 50 :=
by
  rw [Ben_MMs, Bryan_more_candies]
  sorry

end Bryan_skittles_50_l580_580167


namespace limit_f_zero_zero_l580_580178

-- Define the function f(x, y)
noncomputable def f (x y : ℝ) : ℝ :=
  if x ≠ 0 then (x * y) / (x^2 + y^2 * (Real.log (x^2))^2) else 0

-- State the theorem for the limit
theorem limit_f_zero_zero :
  Tendsto (fun p : ℝ × ℝ => f p.1 p.2) (nhds (0,0)) (nhds 0) :=
sorry

end limit_f_zero_zero_l580_580178


namespace greatest_value_of_x_for_7x_factorial_100_l580_580804

open Nat

theorem greatest_value_of_x_for_7x_factorial_100 : 
  ∃ x : ℕ, (∀ y : ℕ, 7^y ∣ factorial 100 → y ≤ x) ∧ x = 16 :=
by
  sorry

end greatest_value_of_x_for_7x_factorial_100_l580_580804


namespace range_q_l580_580892

def q (x : ℝ) : ℝ := x^4 - 4 * x^2 + 4

theorem range_q : set.range q = set.Icc 0 4 :=
sorry

end range_q_l580_580892


namespace value_of_a_l580_580344

universe u

-- Definitions and Conditions
def U : Set ℕ := {1, 3, 5, 7}
def M (a : ℕ) : Set ℕ := {1, |a - 5|}

def M_subset_U (a : ℕ) : Prop := M a ⊆ U
def C_U (S : Set ℕ) : Set ℕ := U \ S

-- The mathematical proof problem given the conditions
theorem value_of_a (a : ℕ) (h1: M_subset_U a) (h2 : C_U (M a) = {5, 7}) : a = 2 ∨ a = 8 :=
by
  sorry

end value_of_a_l580_580344


namespace sin_abs_not_periodic_l580_580831

def f (x : ℝ) : ℝ := Real.sin ( | x | )

theorem sin_abs_not_periodic : ¬ ∃ T ≠ 0, ∀ x : ℝ, f (x + T) = f x := 
sorry

end sin_abs_not_periodic_l580_580831


namespace kona_additional_miles_l580_580579

theorem kona_additional_miles 
  (d_apartment_to_bakery : ℕ := 9) 
  (d_bakery_to_grandmother : ℕ := 24) 
  (d_grandmother_to_apartment : ℕ := 27) : 
  (d_apartment_to_bakery + d_bakery_to_grandmother + d_grandmother_to_apartment) - (2 * d_grandmother_to_apartment) = 6 := 
by 
  sorry

end kona_additional_miles_l580_580579


namespace atlantic_call_charge_l580_580800

theorem atlantic_call_charge :
  let united_base := 6.00
  let united_per_min := 0.25
  let atlantic_base := 12.00
  let same_bill_minutes := 120
  let atlantic_total (charge_per_minute : ℝ) := atlantic_base + charge_per_minute * same_bill_minutes
  let united_total := united_base + united_per_min * same_bill_minutes
  united_total = atlantic_total 0.20 :=
by
  sorry

end atlantic_call_charge_l580_580800


namespace range_of_a_l580_580633

def point (α : Type*) := ℝ × ℝ

def line_intersects_segment (a : ℝ) (p q : point ℝ) : Prop :=
  let l := λ (x y : ℝ), a * x + y + 2 in
  (l p.1 p.2) * (l q.1 q.2) ≤ 0

theorem range_of_a (a : ℝ) :
  (line_intersects_segment a (-2, 1) (3, 2)) → (a ≤ -4/3 ∨ a ≥ 3/2) :=
sorry

end range_of_a_l580_580633


namespace cost_per_meal_is_4_l580_580983

-- Define the initial conditions
def number_of_days_per_week : ℕ := 5
def number_of_people : ℕ := 4
def number_of_weeks : ℕ := 16
def total_cost : ℕ := 1280

-- Define the main proposition we want to prove
theorem cost_per_meal_is_4 :
  let meals_per_week := number_of_people * number_of_days_per_week in
  let total_meals := meals_per_week * number_of_weeks in
  total_cost / total_meals = 4 := by
  sorry

end cost_per_meal_is_4_l580_580983


namespace largest_of_consecutive_odd_sum_200_l580_580407

theorem largest_of_consecutive_odd_sum_200 :
  ∃ x : ℤ, (x - 6) + (x - 4) + (x - 2) + x = 200 ∧ x = 53 :=
by
  use 53
  split
  { norm_num }
  { reflexivity }

end largest_of_consecutive_odd_sum_200_l580_580407


namespace first_year_equals_15_l580_580762

-- Define the conditions
def first_year_human_years (X : ℕ) : Prop :=
  X + 9 + 8 * 5 = 64

-- State the proof problem
theorem first_year_equals_15 : ∃ (X : ℕ), first_year_human_years X ∧ X = 15 :=
by
  use 15
  split
  case left => 
    unfold first_year_human_years
    norm_num
  case right => 
    rfl

end first_year_equals_15_l580_580762


namespace length_of_train_l580_580151

variable (d_train d_bridge v t : ℝ)

theorem length_of_train
  (h1 : v = 12.5) 
  (h2 : t = 30) 
  (h3 : d_bridge = 255) 
  (h4 : v * t = d_train + d_bridge) : 
  d_train = 120 := 
by {
  -- We should infer from here that d_train = 120
  sorry
}

end length_of_train_l580_580151


namespace probability_of_longest_segment_correct_l580_580853

def probabilityOfLongestSegment (l a : ℝ) : ℝ :=
  if 0 ≤ a ∧ a ≤ l/3 then 0
  else if l/3 < a ∧ a ≤ l/2 then (3 * (a / l) - 1) ^ 2
  else if l/2 < a ∧ a ≤ l then 1 - 3 * (1 - a / l) ^ 2
  else 0

theorem probability_of_longest_segment_correct (l a : ℝ) :
    0 ≤ a ∧ a ≤ l → probabilityOfLongestSegment l a = 
      if a ≤ l/3 then 0
      else if l/3 < a ∧ a ≤ l/2 then (3 * (a / l) - 1) ^ 2
      else 1 - 3 * (1 - a / l) ^ 2 := by
  intros h
  unfold probabilityOfLongestSegment
  split_ifs with h1 h2
  · rfl
  · rfl
  · rfl
  · have : ¬ (0 ≤ a ∧ a ≤ l) := by
      push_neg
      cases h with h3 h4
      split
      · intro; linarith
      · intro; linarith
    contradiction

end probability_of_longest_segment_correct_l580_580853


namespace range_of_a_minus_b_l580_580183

-- We define the conditions
def has_root_in_interval (a b : ℝ) (I : set ℝ) : Prop :=
  ∃ x ∈ I, x^2 + a * x + (b - 2) = 0

-- The actual theorem we want to prove
theorem range_of_a_minus_b (a b : ℝ) :
  (has_root_in_interval a b {x | x < -1}) ∧ (has_root_in_interval a b {x | -1 < x ∧ x < 0}) → a - b > -1 :=
by
  sorry

end range_of_a_minus_b_l580_580183


namespace problem_part1_problem_part2_l580_580629

noncomputable def f (x : ℝ) : ℝ := exp x - x + (1/2) * x^2

noncomputable def g (x : ℝ) (a b : ℝ) : ℝ := (1/2) * x^2 + a * x + b

theorem problem_part1 :
  ∀ x : ℝ,
  (∀ f0 : ℝ, f0 = 1 → 
  ∀ f1' : ℝ, f1' = exp 1 → 
  f(x) = f1' * exp (x - 1) - f0 * x + (1/2) * x^2 → 
  f(x) = exp x - x + (1/2) * x^2) :=
by intros; sorry

theorem problem_part2 (a b : ℝ) :
  ∀ x : ℝ,
  (let f' (x : ℝ) := exp x - 1 in
  let g' (x : ℝ) := x + a in
  f' x - g' x > 0 ↔ (a + 1 ≤ 0 ∨ (a + 1 > 0 ∧ x > log(a + 1)))) :=
by intros; sorry

end problem_part1_problem_part2_l580_580629


namespace grace_mulch_charge_l580_580267

/-- Given that Grace charges $6 per hour for mowing, $11 per hour for weeding, 
         worked 63 hours mowing, 9 hours weeding, 10 hours mulching, and earned a total of $567.
    Grace's hourly charge for mulching is $9. -/
theorem grace_mulch_charge
  (mowing_rate : ℕ := 6)
  (weeding_rate : ℕ := 11)
  (hours_mowing : ℕ := 63)
  (hours_weeding : ℕ := 9)
  (hours_mulching : ℕ := 10)
  (total_earned : ℕ := 567) : 
  let earnings_mowing := hours_mowing * mowing_rate,
      earnings_weeding := hours_weeding * weeding_rate,
      earnings_from_mowing_and_weeding := earnings_mowing + earnings_weeding,
      earnings_mulching := total_earned - earnings_from_mowing_and_weeding
  in
  earnings_mulching / hours_mulching = 9 := 
sorry

end grace_mulch_charge_l580_580267


namespace det_scaled_matrix_l580_580927

variable {R : Type*} [CommRing R]

def det2x2 (a b c d : R) : R := a * d - b * c

theorem det_scaled_matrix 
  (x y z w : R) 
  (h : det2x2 x y z w = 3) : 
  det2x2 (3 * x) (3 * y) (6 * z) (6 * w) = 54 := by
  sorry

end det_scaled_matrix_l580_580927


namespace exists_rational_in_0_1_l580_580748

-- Define the sets A_n
def A (n : ℕ) : set ℚ :=
  {q | ∃ (k : fin n → ℕ+), q = ∑ i, (1 : ℚ) / (k i)}

-- Prove the main statement
theorem exists_rational_in_0_1 (h2021 : (0 : ℚ) < 1 ) :
  ∃ (q : ℚ), 0 < q ∧ q < 1 ∧ q ∈ A 2021 ∧ q ∉ A 2020 :=
sorry

end exists_rational_in_0_1_l580_580748


namespace faster_train_speed_l580_580798

theorem faster_train_speed
  (slower_train_speed : ℝ := 60) -- speed of the slower train in km/h
  (length_train1 : ℝ := 1.10) -- length of the slower train in km
  (length_train2 : ℝ := 0.9) -- length of the faster train in km
  (cross_time_sec : ℝ := 47.99999999999999) -- crossing time in seconds
  (cross_time : ℝ := cross_time_sec / 3600) -- crossing time in hours
  (total_distance : ℝ := length_train1 + length_train2) -- total distance covered
  (relative_speed : ℝ := total_distance / cross_time) -- relative speed
  (faster_train_speed : ℝ := relative_speed - slower_train_speed) -- speed of the faster train
  : faster_train_speed = 90 :=
by
  sorry

end faster_train_speed_l580_580798


namespace similar_triangles_height_ratio_l580_580066

theorem similar_triangles_height_ratio (area_ratio : ℝ) (h₁ : ℝ) (h₂ : ℝ) 
  (similar : Boolean) (h₁_value : h₁ = 5) (area_ratio_value : area_ratio = 9) :
  similar = true → area_ratio = (h₂ / h₁) ^ 2 → h₂ = 15 :=
by
  intro h_similar area_eq
  rw [h₁_value, area_ratio_value]
  sorry

end similar_triangles_height_ratio_l580_580066


namespace cyclic_product_congruence_l580_580319

theorem cyclic_product_congruence (n c : ℕ) (h1 : 0 < c) (h2 : c < n) (a : ℕ → ℕ) : 
  ∃ (a : Fin n → ℕ), (∏ i : Fin n, (a i - a (i + 1)) % n) = 0 ∨ (∏ i : Fin n, (a i - a (i + 1)) % n) = c := 
sorry

end cyclic_product_congruence_l580_580319


namespace arithmetic_sequence_sum_l580_580159

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (n : ℕ) 
  (h1 : ∑ i in finset.filter (λ i, i % 2 = 1) (finset.range (2 * n + 1)), a i = 4)
  (h2 : ∑ i in finset.filter (λ i, i % 2 = 0) (finset.range (2 * n + 1)), a i = 3) : 
  n = 3 := 
sorry

end arithmetic_sequence_sum_l580_580159


namespace correct_statements_count_l580_580182

theorem correct_statements_count :
  let angles_with_same_terminal_side := ∀ k : ℤ, ∃ α, α = ℝ.pi / 5 + 2 * k * ℝ.pi
  let area_of_sector := ∀ (radius : ℝ) (angle_deg : ℝ), radius = 6 ∧ angle_deg = 15 ∧ angle_deg = 15 * ℝ.pi / 180 →
    1 / 2 * (angle_deg * ℝ.pi / 180) * radius ^ 2 = 3 * ℝ.pi / 2
  let positive_correlation := "lower-left" → "upper-right"
  let cos_260 := "third-quadrant" → ℝ.cos (260 * ℝ.pi / 180) < 0
  (¬ angles_with_same_terminal_side ∧ area_of_sector 6 15 ∧ ¬ positive_correlation ∧ ¬ cos_260) ∨
  (angles_with_same_terminal_side ∧ ¬ area_of_sector 6 15 ∧ ¬ positive_correlation ∧ ¬ cos_260) ∨
  (¬ angles_with_same_terminal_side ∧ ¬ area_of_sector 6 15 ∧ positive_correlation ∧ ¬ cos_260) ∨
  (¬ angles_with_same_terminal_side ∧ ¬ area_of_sector 6 15 ∧ ¬ positive_correlation ∧ cos_260) :=
sorry

end correct_statements_count_l580_580182


namespace max_combinatorial_shapes_l580_580452

noncomputable def max_lines_planes_tetrahedrons 
  (α β : Type) [plane α] [plane β] 
  (points_α : finset α) (points_β : finset β) : ℕ × ℕ × ℕ :=
  let points := points_α ∪ points_β in
  let max_lines := (points.card.choose 2) in
  let max_planes := (points_α.card.choose 2 * points_β.card) + 
                    (points_α.card * points_β.card.choose 2) + 2 in
  let max_tetrahedrons := (points_α.card.choose 3 * points_β.card) + 
                          (points_α.card.choose 2 * points_β.card.choose 2) +
                          (points_α.card * points_β.card.choose 3) in
  (max_lines, max_planes, max_tetrahedrons)

theorem max_combinatorial_shapes {α β : Type} [plane α] [plane β]
  (points_α : finset α) (points_β : finset β)
  (hα : points_α.card = 4) (hβ : points_β.card = 5)
  (h_disjoint : points_α ∩ points_β = ∅)
  (h_not_coplanar : ∀ p ∈ points_α, ∀ q ∈ points_β, p ≠ q) :
  max_lines_planes_tetrahedrons α β points_α points_β = (36, 72, 120) :=
by sorry

end max_combinatorial_shapes_l580_580452


namespace compound_interest_correct_l580_580281

variables (SI : ℚ) (R : ℚ) (T : ℕ) (P : ℚ)

def calculate_principal (SI R T : ℚ) : ℚ := SI * 100 / (R * T)

def calculate_compound_interest (P R : ℚ) (T : ℕ) : ℚ :=
  P * ((1 + R / 100)^T - 1)

theorem compound_interest_correct (h1: SI = 52) (h2: R = 5) (h3: T = 2) :
  calculate_compound_interest (calculate_principal SI R T) R T = 53.30 :=
by
  sorry

end compound_interest_correct_l580_580281


namespace polygon_properties_l580_580842

def sum_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

def triangle_area (a b C : ℝ) : ℝ := 0.5 * a * b * Real.sin C

theorem polygon_properties (n : ℕ) (theta : ℝ) (a : ℝ) (C : ℝ) (A : ℝ) :
    sum_interior_angles n - theta = 3240 ∧
    triangle_area a a C = A ∧
    a = 15 ∧
    C = Real.pi / 3 →  --C = 60 degrees
    n = 20 ∧ theta = 0 :=
by {
  sorry
}

end polygon_properties_l580_580842


namespace fill_time_l580_580460

def inflow_rate : ℕ := 24 -- gallons per second
def outflow_rate : ℕ := 4 -- gallons per second
def basin_volume : ℕ := 260 -- gallons

theorem fill_time (inflow_rate outflow_rate basin_volume : ℕ) (h₁ : inflow_rate = 24) (h₂ : outflow_rate = 4) 
  (h₃ : basin_volume = 260) : basin_volume / (inflow_rate - outflow_rate) = 13 :=
by
  sorry

end fill_time_l580_580460


namespace origin_moves_distance_under_transformation_l580_580799

-- Defining the centers and radii of the original and transformed circles
def original_center : (ℝ × ℝ) := (1, 3)
def original_radius : ℝ := 4

def transformed_center : (ℝ × ℝ) := (7, 10)
def transformed_radius : ℝ := 6

-- Defining the distance calculation function
def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Calculate the distance the origin moves under the transformation
theorem origin_moves_distance_under_transformation :
  distance (0, 0) (7, 10) = real.sqrt 149 :=
by
  -- Define the transformation properties (dilation and translation)
  let dilation_factor : ℝ := 1.5
  let translation_vector : (ℝ × ℝ) := (18, 19)

  -- Calculate the theoretical new position of the origin
  let new_origin_position : (ℝ × ℝ) := (7, 10)

  -- Use Lean's ‘sorry’ as a placeholder for the proof.
  exact sorry

end origin_moves_distance_under_transformation_l580_580799


namespace constant_term_of_binomial_expansion_l580_580758

-- Definitions translated from problem conditions
def general_term (r : ℕ) : ℚ :=
  binomial 6 r * (-1)^r * (6 - (3/2 : ℚ) * r)

-- The theorem stating the proof problem
theorem constant_term_of_binomial_expansion :
  general_term 4 = 15 :=
sorry

end constant_term_of_binomial_expansion_l580_580758


namespace benny_crayons_l580_580016

theorem benny_crayons : ∀ (c_initial c_total : ℕ), c_initial = 9 → c_total = 12 → c_total - c_initial = 3 :=
by {
  intros c_initial c_total h_initial h_total,
  rw [h_initial, h_total],
  norm_num,
  sorry
}

end benny_crayons_l580_580016


namespace perpendicular_lines_condition_l580_580932

theorem perpendicular_lines_condition (a : ℝ) :
  (¬ a = 1/2 ∨ ¬ a = -1/2) ∧ a * (-4 * a) = -1 ↔ a = 1/2 :=
by
  sorry

end perpendicular_lines_condition_l580_580932


namespace CarltonUniqueOutfits_l580_580531

theorem CarltonUniqueOutfits:
  ∀ (buttonUpShirts sweaterVests : ℕ), 
    buttonUpShirts = 3 →
    sweaterVests = 2 * buttonUpShirts →
    (sweaterVests * buttonUpShirts) = 18 :=
by
  intros buttonUpShirts sweaterVests h1 h2
  rw [h1, h2]
  simp
  sorry

end CarltonUniqueOutfits_l580_580531


namespace problem_statement_l580_580254

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (1/2) * x^2 - 2 * a * x

theorem problem_statement (a : ℝ) (x1 x2 : ℝ) (h_a : a > 1) (h1 : x1 < x2) (h_extreme : f a x1 = 0 ∧ f a x2 = 0) : 
  f a x2 < -3/2 :=
sorry

end problem_statement_l580_580254


namespace maximum_value_of_k_l580_580956

-- Define the variables and conditions
variables {a b c k : ℝ}
axiom h₀ : a > b
axiom h₁ : b > c
axiom h₂ : 4 / (a - b) + 1 / (b - c) + k / (c - a) ≥ 0

-- State the theorem
theorem maximum_value_of_k : k ≤ 9 := sorry

end maximum_value_of_k_l580_580956


namespace cost_per_can_of_tuna_l580_580369

theorem cost_per_can_of_tuna
  (num_cans : ℕ) -- condition 1
  (num_coupons : ℕ) -- condition 2
  (coupon_discount_cents : ℕ) -- condition 2 detail
  (amount_paid_dollars : ℚ) -- condition 3
  (change_received_dollars : ℚ) -- condition 3 detail
  (cost_per_can_cents: ℚ) : -- the quantity we want to prove
  num_cans = 9 →
  num_coupons = 5 →
  coupon_discount_cents = 25 →
  amount_paid_dollars = 20 →
  change_received_dollars = 5.5 →
  cost_per_can_cents = 175 :=
by
  intros hn hc hcd hap hcr
  sorry

end cost_per_can_of_tuna_l580_580369


namespace sum_imaginary_parts_l580_580947

-- Given definitions
variables (p r s u v x y q : ℝ)
variables (i : ℂ := complex.I)
variables (c1 : ℂ := p + q * i)
variables (c2 : ℂ := r + s * i)
variables (c3 : ℂ := u + v * i)
variables (c4 : ℂ := x + y * i)

-- Conditions
def condition1 : q = 4 := sorry
def condition2 : u = -p - r - x := sorry
def condition3 : c1 + c2 + c3 + c4 = 7 * i := sorry

-- Problem statement
theorem sum_imaginary_parts : s + v + y = 3 :=
by
  have h1 : q = 4 := condition1
  have h2 : u = -p - r - x := condition2
  have h3 : c1 + c2 + c3 + c4 = 7 * i := condition3
  sorry

end sum_imaginary_parts_l580_580947


namespace value_of_ab_cd_over_ad_bc_l580_580949

-- Definitions of the conditions
variables {a b c d : ℝ}
variables (habcd : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
variables (h1 : a^2 + d^2 - a * d = b^2 + c^2 + b * c)
variables (h2 : a^2 + b^2 = c^2 + d^2)

-- The goal statement proving the value of the given expression
theorem value_of_ab_cd_over_ad_bc :
  (habcd) → (h1) → (h2) → (ab + cd) / (ad + bc) = sqrt 3 / 2 :=
  by
  intro habcd h1 h2
  sorry

end value_of_ab_cd_over_ad_bc_l580_580949


namespace greatest_oranges_thrown_out_l580_580120

-- Definitions based on the problem conditions
def total_oranges := ℕ
def number_of_kids := 8
def oranges_each_kid_receives (n : total_oranges) : ℕ := n / number_of_kids
def oranges_thrown_out (n : total_oranges) : ℕ := n % number_of_kids

-- Theorem statement based on the problem question and correct answer
theorem greatest_oranges_thrown_out : ∀ (n : total_oranges), oranges_thrown_out n ≤ 7 :=
by
  sorry

end greatest_oranges_thrown_out_l580_580120


namespace complex_fraction_eval_l580_580457

theorem complex_fraction_eval (i : ℂ) (hi : i^2 = -1) : (3 + i) / (1 + i) = 2 - i := 
by 
  sorry

end complex_fraction_eval_l580_580457


namespace compare_abc_l580_580929

noncomputable def a : ℝ := Real.pi^(1 / 3)
noncomputable def b : ℝ := Real.logBase Real.pi 3
noncomputable def c : ℝ := Real.log (Real.sqrt 3 - 1)

theorem compare_abc : c < b ∧ b < a := 
  sorry

end compare_abc_l580_580929


namespace similar_triangles_height_ratio_l580_580067

theorem similar_triangles_height_ratio (area_ratio : ℝ) (h₁ : ℝ) (h₂ : ℝ) 
  (similar : Boolean) (h₁_value : h₁ = 5) (area_ratio_value : area_ratio = 9) :
  similar = true → area_ratio = (h₂ / h₁) ^ 2 → h₂ = 15 :=
by
  intro h_similar area_eq
  rw [h₁_value, area_ratio_value]
  sorry

end similar_triangles_height_ratio_l580_580067


namespace count_k_values_l580_580923

theorem count_k_values :
  let six_to_seven := 2^7 * 3^7
  let eight_to_nine := 2^27
  let twelve_to_twelve := 2^24 * 3^12
  ∃ (a b c : ℕ), c = 0 ∧ a ≤ 27 ∧ b ≤ 12 ∧ (λ k, 
    lcm (lcm (2^7 * 3^7) (2^27)) (2^a * 3^b * 5^c) = 2^24 * 3^12) ∧
  (finset.Icc 0 27).card * (finset.Icc 0 12).card = 364 := 
sorry

end count_k_values_l580_580923


namespace election_ballot_l580_580553

theorem election_ballot (n : ℕ) 
  (ballots : Fin (n + 1) → Set (Set (Fin n))) 
  (h1 : ∀ i, (ballots i).Nonempty)
  (h2 : ∀ (s : Fin (n + 1) → Set (Fin n)) (hs : ∀ i, s i ∈ ballots i), 
    ∃ c : Fin n, ∀ i, c ∈ s i) : 
  ∃ i j : Fin (n + 1), ∀ b1 b2 ∈ ballots i, ∀ c ∈ b1, ∃ c' ∈ b2, c = c' := sorry

end election_ballot_l580_580553


namespace height_of_larger_triangle_l580_580081

theorem height_of_larger_triangle 
  (area_ratio : ℝ)
  (height_small_triangle : ℝ)
  (similar_triangles : Prop)
  (height_large_triangle : ℝ) :
  area_ratio = 1 / 9 →
  height_small_triangle = 5 →
  similar_triangles →
  height_large_triangle = height_small_triangle * 3 :=
begin
  intros h_ratio h_height_small h_similar,
  rw h_ratio at *,
  rw h_height_small at *,
  exact eq.symm (mul_eq_mul_left_iff.1 (eq.trans (sqrt_eq (by norm_num) (by norm_num)) (by norm_num))),
sorry,
end

# The above code imports the necessary library, defines the theorem with the conditions and concludes with the height of the larger triangle.

end height_of_larger_triangle_l580_580081


namespace segment_length_l580_580634

open Real

noncomputable def eccentricity := sqrt 2 / 2
noncomputable def focal_distance := 2
noncomputable def a := sqrt 2
noncomputable def b := sqrt (a^2 - 1) -- since b = sqrt(a^2 - c^2) and c = 1

def line (x : ℝ) : ℝ := -x + 1
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

theorem segment_length :
  let A := (0, line 0)
  let B := (4 / 3, line (4 / 3))
  dist (fst A) (fst B) + dist (snd A) (snd B) = 4*sqrt 2 / 3 :=
by
  sorry

end segment_length_l580_580634


namespace triangle_similarity_l580_580926

open EuclideanGeometry

theorem triangle_similarity
  (O : Circle)
  (P : Point)
  (A B C D M : Point)
  (h_tangent_PA : Tangent P A O)
  (h_tangent_PB : Tangent P B O)
  (h_secant_PCD : Secant P C D O)
  (h_midpoint_M : Midpoint M A B) :
  Similar (Triangle D A C) (Triangle D M B) ∧
  Similar (Triangle D M B) (Triangle B M C) :=
by {
  -- proof is omitted
  sorry
}

end triangle_similarity_l580_580926


namespace initial_salt_percentage_l580_580832

theorem initial_salt_percentage (P : ℕ) : 
  let initial_solution := 100 
  let added_salt := 20 
  let final_solution := initial_solution + added_salt 
  (P / 100) * initial_solution + added_salt = (25 / 100) * final_solution → 
  P = 10 := 
by
  sorry

end initial_salt_percentage_l580_580832


namespace significant_figures_and_accuracy_l580_580756

def scientific_notation := "6.30 × 10^5"

def significant_figures (notation: String) : Nat :=
  -- number of significant figures in the given scientific notation
  if notation = "6.30 × 10^5" then 3 else 0

def accuracy_place (notation: String) : String :=
  -- the place to which the number is accurate in the given scientific notation
  if notation = "6.30 × 10^5" then "ten thousand's place" else "unknown"

theorem significant_figures_and_accuracy :
  significant_figures scientific_notation = 3 ∧
  accuracy_place scientific_notation = "ten thousand's place" :=
by
  sorry

end significant_figures_and_accuracy_l580_580756


namespace find_area_bounded_by_curves_l580_580425

-- Define the linear equation passing through points (0,3) and (7,0)
def line (x : ℝ) : ℝ := -3 / 7 * x + 3

-- Define the parabolic equation with vertex at (3,3) passing through (5,0)
def parabola (x : ℝ) : ℝ := -3 / 4 * (x - 3) ^ 2 + 3

-- Define the integral for the area between the curves from x = 0 to x = 5
def bounded_area : ℝ :=
  ∫ (x : ℝ) in 0..5, parabola x - line x

-- The target proof statement that the bounded area is 103/14
theorem find_area_bounded_by_curves :
  bounded_area = 103 / 14 :=
by
  sorry

end find_area_bounded_by_curves_l580_580425


namespace hurricane_damage_in_GBP_l580_580480

def damage_in_AUD : ℤ := 45000000
def conversion_rate : ℚ := 1 / 2 -- 1 AUD = 1/2 GBP

theorem hurricane_damage_in_GBP : 
  (damage_in_AUD : ℚ) * conversion_rate = 22500000 := 
by
  sorry

end hurricane_damage_in_GBP_l580_580480


namespace sin_half_alpha_value_l580_580607

theorem sin_half_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 4 / 3) :
  Real.sin (α / 2) = sqrt 5 / 5 ∨ Real.sin (α / 2) = - sqrt 5 / 5 := by
  sorry

end sin_half_alpha_value_l580_580607


namespace exists_monochromatic_triangle_l580_580432

theorem exists_monochromatic_triangle (points : Fin 6 → Point) (color : (Point × Point) → Color) :
  ∃ (a b c : Point), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (color (a, b) = color (b, c) ∧ color (b, c) = color (c, a)) :=
by
  sorry

end exists_monochromatic_triangle_l580_580432


namespace ratio_of_numbers_l580_580785

variable (a b : ℝ) (h : 0 < b) (h₁ : a + b = 7 * (a - b))

theorem ratio_of_numbers (h₂ : a > b) : a / b = 4 / 3 := 
begin
  sorry
end

end ratio_of_numbers_l580_580785


namespace output_increase_percentage_l580_580772

theorem output_increase_percentage (O : ℝ) (P : ℝ) (h : (O * (1 + P / 100) * 1.60) * 0.5682 = O) : P = 10.09 :=
by 
  sorry

end output_increase_percentage_l580_580772


namespace trader_gain_percentage_is_25_l580_580871

noncomputable def trader_gain_percentage (C : ℝ) : ℝ :=
  ((22 * C) / (88 * C)) * 100

theorem trader_gain_percentage_is_25 (C : ℝ) (h : C ≠ 0) : trader_gain_percentage C = 25 := by
  unfold trader_gain_percentage
  field_simp [h]
  norm_num
  sorry

end trader_gain_percentage_is_25_l580_580871


namespace remainder_S_mod_500_l580_580323

def R := {n : ℕ // ∃ k : ℕ, n = 3^k % 500}
def S : ℕ := ∑ r in R.to_finset, r

theorem remainder_S_mod_500 : S % 500 = 0 :=
sorry

end remainder_S_mod_500_l580_580323


namespace books_per_week_l580_580745

theorem books_per_week (total_books weeks: ℕ) (h1 : total_books = 30) (h2 : weeks = 5) :
  total_books / weeks = 6 :=
by
  rw [h1, h2]
  norm_num

end books_per_week_l580_580745


namespace isosceles_triangle_count_l580_580293

theorem isosceles_triangle_count (A B C : Type) [triangle_acute : triangle ABC] (hAC_lt_AB : AC < AB) (hAB_lt_BC : AB < BC) :
  ∃ (P : Type), count_points P = 15 ∧ (isosceles PAB) ∧ (isosceles PBC) :=
sorry

end isosceles_triangle_count_l580_580293


namespace lengths_equal_l580_580403

-- a rhombus AFCE inscribed in a rectangle ABCD
variables {A B C D E F : Type}
variables {width length perimeter side_BF side_DE : ℝ}
variables {AF CE FC AF_side FC_side : ℝ}
variables {h1 : width = 20} {h2 : length = 25} {h3 : perimeter = 82}
variables {h4 : side_BF = (82 / 4 - 20)} {h5 : side_DE = (82 / 4 - 20)} 

-- prove that the lengths of BF and DE are equal
theorem lengths_equal :
  side_BF = side_DE :=
by
  sorry

end lengths_equal_l580_580403


namespace more_men_than_women_l580_580145

variables (M W A C : ℕ)

-- Given conditions
def total_persons := M + W + C = 240
def number_of_men := M = 90
def more_adults_than_children := A = C + 40
def adults_composition := A = M + W

-- To prove
theorem more_men_than_women (h1 : total_persons) (h2 : number_of_men) (h3 : more_adults_than_children) (h4 : adults_composition) : M - W = 40 :=
sorry

end more_men_than_women_l580_580145


namespace imaginary_part_of_complex_l580_580395

theorem imaginary_part_of_complex (z : ℂ) (h : z = (1 + complex.I) / (1 - complex.I)) :
  (z^2023).im = -1 :=
by
  sorry

end imaginary_part_of_complex_l580_580395


namespace minimum_area_triangle_ABC_l580_580829

-- Define the vertices of the triangle
def A : ℤ × ℤ := (0,0)
def B : ℤ × ℤ := (30,18)

-- Define a function to calculate the area of the triangle using the Shoelace formula
def area_of_triangle (A B C : ℤ × ℤ) : ℤ := 15 * (C.2).natAbs

-- State the theorem
theorem minimum_area_triangle_ABC : 
  ∀ C : ℤ × ℤ, C ≠ (0,0) → area_of_triangle A B C ≥ 15 :=
by
  sorry -- Skip the proof

end minimum_area_triangle_ABC_l580_580829


namespace wheel_radius_l580_580505

theorem wheel_radius (d_100_revs : ℝ) (h_d100 : d_100_revs = 91.77) : 
  ∃ r_cm : ℝ, r_cm = 14.61 :=
by
  have circumference : ℝ := d_100_revs / 100
  have radius_m : ℝ := circumference / (2 * Real.pi)
  let radius_cm := radius_m * 100
  use radius_cm
  have h_radius_cm : radius_cm = 0.9177 / (2 * Real.pi) * 100 := by
    calc
      radius_cm = (0.9177 / (2 * Real.pi)) * 100 : by sorry
  have h_radius_cm_eval : radius_cm ≈ 14.61 := by
    sorry
  exact h_radius_cm_eval

end wheel_radius_l580_580505


namespace find_m_value_l580_580621

theorem find_m_value (m : ℝ) (θ : ℝ) (P : ℝ × ℝ)
  (h_vertex_at_origin : P = (-real.sqrt 3, m))
  (h_sin_theta : real.sin θ = real.sqrt 13 / 13) :
  m = 1 / 2 :=
by
  // sorry to skip the complete proof
  sorry

end find_m_value_l580_580621


namespace hyperbola_standard_eq_l580_580227

-- Define the given conditions
def asymptotic_equations (x y : ℝ) : Prop := (2 * x + y = 0) ∨ (2 * x - y = 0)
def passes_through (C : ℝ → ℝ → Prop) (x y : ℝ) : Prop := C x y

-- Define the equation of the hyperbola
def hyperbola_eq (x y : ℝ) (λ : ℝ) : Prop := y^2 - 4 * x^2 = λ

-- State the theorem
theorem hyperbola_standard_eq :
  (∃ (C : ℝ → ℝ → Prop),
    (asymptotic_equations = λ x y, (2 * x + y = 0) ∨ (2 * x - y = 0)) ∧
    (passes_through C (sqrt 3) 4)) →
  (∃ (λ : ℝ), hyperbola_eq x y λ ∧ (hyperbola_eq (sqrt 3) 4 λ)) →
  hyperbola_eq x y 4 →
  ∀ (x y : ℝ), (hyperbola_eq x y 4 → (y^2) / 4 - (x^2) = 1) :=
sorry

end hyperbola_standard_eq_l580_580227


namespace ellipse_sum_l580_580509

theorem ellipse_sum (h k a b : ℝ) (h_val : h = 3) (k_val : k = -5) (a_val : a = 6) (b_val : b = 2) : h + k + a + b = 6 :=
by
  rw [h_val, k_val, a_val, b_val]
  norm_num

end ellipse_sum_l580_580509


namespace cost_of_corn_per_acre_l580_580383

def TotalLand : ℕ := 4500
def CostWheat : ℕ := 35
def Capital : ℕ := 165200
def LandWheat : ℕ := 3400
def LandCorn := TotalLand - LandWheat

theorem cost_of_corn_per_acre :
  ∃ C : ℕ, (Capital = (C * LandCorn) + (CostWheat * LandWheat)) ∧ C = 42 :=
by
  sorry

end cost_of_corn_per_acre_l580_580383


namespace determine_max_weight_l580_580439

theorem determine_max_weight {a b : ℕ} (n : ℕ) (x : ℕ) (ha : a > 0) (hb : b > 0) (hx : 1 ≤ x ∧ x ≤ n) :
  n = 9 :=
sorry

end determine_max_weight_l580_580439


namespace sin_6θ_l580_580642

noncomputable def eulerIdentityCondition (θ : ℝ) : Prop :=
  complex.exp (complex.I * θ) = (3 + complex.I * real.sqrt 8) / 4

theorem sin_6θ (θ : ℝ) (h : eulerIdentityCondition θ) : 
  real.sin (6 * θ) = - (855 * real.sqrt 2) / 1024 :=
by {
  sorry
}

end sin_6θ_l580_580642


namespace good_eggs_collected_l580_580430

/-- 
Uncle Ben has 550 chickens on his farm, consisting of 49 roosters and the rest being hens. 
Out of these hens, there are three types:
1. Type A: 25 hens do not lay eggs at all.
2. Type B: 155 hens lay 2 eggs per day.
3. Type C: The remaining hens lay 4 eggs every three days.

Moreover, Uncle Ben found that 3% of the eggs laid by Type B and Type C hens go bad before being collected. 
Prove that the total number of good eggs collected by Uncle Ben after one day is 716.
-/
theorem good_eggs_collected 
    (total_chickens : ℕ) (roosters : ℕ) (typeA_hens : ℕ) (typeB_hens : ℕ) 
    (typeB_eggs_per_day : ℕ) (typeC_eggs_per_3days : ℕ) (percent_bad_eggs : ℚ) :
  total_chickens = 550 →
  roosters = 49 →
  typeA_hens = 25 →
  typeB_hens = 155 →
  typeB_eggs_per_day = 2 →
  typeC_eggs_per_3days = 4 →
  percent_bad_eggs = 0.03 →
  (total_chickens - roosters - typeA_hens - typeB_hens) * (typeC_eggs_per_3days / 3) + (typeB_hens * typeB_eggs_per_day) - 
  round (percent_bad_eggs * ((total_chickens - roosters - typeA_hens - typeB_hens) * (typeC_eggs_per_3days / 3) + (typeB_hens * typeB_eggs_per_day))) = 716 :=
by
  intros
  sorry

end good_eggs_collected_l580_580430


namespace ramu_profit_percent_is_21_64_l580_580367

-- Define the costs and selling price as constants
def cost_of_car : ℕ := 42000
def cost_of_repairs : ℕ := 13000
def selling_price : ℕ := 66900

-- Define the total cost and profit
def total_cost : ℕ := cost_of_car + cost_of_repairs
def profit : ℕ := selling_price - total_cost

-- Define the profit percent formula
def profit_percent : ℚ := ((profit : ℚ) / (total_cost : ℚ)) * 100

-- State the theorem we want to prove
theorem ramu_profit_percent_is_21_64 : profit_percent = 21.64 := by
  sorry

end ramu_profit_percent_is_21_64_l580_580367


namespace circle_condition_l580_580002

   theorem circle_condition (m : ℝ) : (0 < m ∧ m < 1) ∨ (m < 1 / 4) ∨ (m > 1) ↔ (x y : ℝ) : x^2 + y^2 + (4 : ℝ) * m * x - (2 : ℝ) * y + (5 : ℝ) * m = (0 : ℝ) -> ∃(h k r : ℝ), x^2 - 2 * h * x + y^2 - 2 * k * y + h^2 + k^2 - r^2 = 0 :=
   sorry
   
end circle_condition_l580_580002


namespace complex_number_in_second_quadrant_l580_580300

theorem complex_number_in_second_quadrant :
  let z := (2 + 4 * Complex.I) / (1 + Complex.I) 
  ∃ (im : ℂ), z = im ∧ im.re < 0 ∧ 0 < im.im := by
  sorry

end complex_number_in_second_quadrant_l580_580300


namespace missing_digit_l580_580196

theorem missing_digit (B : ℕ) (h : B < 10) : 
  (15 ∣ (200 + 10 * B)) ↔ B = 1 ∨ B = 4 :=
by sorry

end missing_digit_l580_580196


namespace find_TS_l580_580161

-- Definitions of the conditions as given:
def PQ : ℝ := 25
def PS : ℝ := 25
def QR : ℝ := 15
def RS : ℝ := 15
def PT : ℝ := 15
def ST_parallel_QR : Prop := true  -- ST is parallel to QR (used as a given fact)

-- Main statement in Lean:
theorem find_TS (h1 : PQ = 25) (h2 : PS = 25) (h3 : QR = 15) (h4 : RS = 15) (h5 : PT = 15)
               (h6 : ST_parallel_QR) : TS = 24 :=
by
  sorry

end find_TS_l580_580161


namespace evaluate_g_at_8_l580_580994

def g (n : ℕ) : ℕ := n^2 - 3 * n + 29

theorem evaluate_g_at_8 : g 8 = 69 := by
  unfold g
  calc
    8^2 - 3 * 8 + 29 = 64 - 24 + 29 := by simp
                      _ = 69 := by norm_num

end evaluate_g_at_8_l580_580994


namespace correct_solutions_l580_580198

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn : ∀ (x y : ℝ), f (x * y) = f x * f y - 2 * x * y

theorem correct_solutions :
  (∀ x : ℝ, f x = 2 * x) ∨ (∀ x : ℝ, f x = -x) := sorry

end correct_solutions_l580_580198


namespace part_a_part_b_l580_580450

noncomputable def G : ℝ → ℝ := sorry

axiom G_properties : ∃ (x1 x2 x3 x4 x5 : ℝ)
  (h1 : x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ x4 < x5)
  (h2 : G x1 = 2022) 
  (h3 : G x2 = 2022)
  (h4 : G x3 = 2022)
  (h5 : G x4 = 2022)
  (h6 : G x5 = 2022)
  (h7 : ∀ x, G (-16 - x) = G x)  -- Symmetry about x = -8

theorem part_a : ∃ (x1 x3 x5 : ℝ), 
  (x1 + x3 + x5 = -24) :=
sorry

theorem part_b : ∃ (n : ℕ), 
  (n ≥ 6 ∧ polynomial.degree G = n) :=
sorry

end part_a_part_b_l580_580450


namespace a_3_and_a_4_sum_l580_580703

theorem a_3_and_a_4_sum (x a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℚ) :
  (1 - (1 / (2 * x))) ^ 6 = a_0 + a_1 * (1 / x) + a_2 * (1 / x) ^ 2 + a_3 * (1 / x) ^ 3 + 
  a_4 * (1 / x) ^ 4 + a_5 * (1 / x) ^ 5 + a_6 * (1 / x) ^ 6 →
  a_3 + a_4 = -25 / 16 :=
sorry

end a_3_and_a_4_sum_l580_580703


namespace carlton_outfit_count_l580_580527

-- Definitions of conditions
def sweater_vests (s : ℕ) : ℕ := 2 * s
def button_up_shirts : ℕ := 3
def outfits (v s : ℕ) : ℕ := v * s

-- Theorem statement
theorem carlton_outfit_count : outfits (sweater_vests button_up_shirts) button_up_shirts = 18 :=
by
  sorry

end carlton_outfit_count_l580_580527


namespace no_equal_partition_of_173_ones_and_neg_ones_l580_580225

theorem no_equal_partition_of_173_ones_and_neg_ones
  (L : List ℤ) (h1 : L.length = 173) (h2 : ∀ x ∈ L, x = 1 ∨ x = -1) :
  ¬ (∃ (L1 L2 : List ℤ), L = L1 ++ L2 ∧ L1.sum = L2.sum) :=
by
  sorry

end no_equal_partition_of_173_ones_and_neg_ones_l580_580225


namespace divisor_of_10n_l580_580211

theorem divisor_of_10n {n : ℕ} : 
  n > 0 →  (∑ k in Finset.range (n + 1), k) ∣ (10 * n) → 
  (∃ n_vals : Finset ℕ, n_vals.card = 5 ∧ ∀ x ∈ n_vals, x ∈ {1, 3, 4, 9, 19}) :=
begin
  sorry
end

end divisor_of_10n_l580_580211


namespace documentaries_count_l580_580899

def number_of_documents
  (novels comics albums crates capacity : ℕ)
  (total_items := crates * capacity)
  (known_items := novels + comics + albums)
  (documentaries := total_items - known_items) : ℕ :=
  documentaries

theorem documentaries_count
  : number_of_documents 145 271 209 116 9 = 419 :=
by
  sorry

end documentaries_count_l580_580899


namespace percent_germinated_approx_l580_580664

noncomputable def total_seeds : ℝ := 500 + 200 + 150 + 350 + 100
noncomputable def germinated_seeds_first : ℝ := 0.30 * 500
noncomputable def germinated_seeds_second : ℝ := 0.50 * 200
noncomputable def germinated_seeds_third : ℝ := 0.40 * 150
noncomputable def germinated_seeds_fourth : ℝ := 0.35 * 350
noncomputable def germinated_seeds_fifth : ℝ := 0.25 * 100

noncomputable def total_germinated_seeds : ℝ :=
  germinated_seeds_first + germinated_seeds_second +
  germinated_seeds_third + germinated_seeds_fourth + germinated_seeds_fifth

noncomputable def percent_germinated : ℝ :=
  (total_germinated_seeds / total_seeds) * 100

theorem percent_germinated_approx : percent_germinated ≈ 35.19 := by
  sorry

end percent_germinated_approx_l580_580664


namespace smallest_n_l580_580127

theorem smallest_n (Q : ℕ → ℝ) :
  (∀ n, n ≥ 50 → Q n = (2 / (n + 2))) →
  ∀ n, n = 1011 ↔ Q n < (1 / 2023) :=
by
  intro hQ
  split
  case mp =>
    intro h
    rw [h, hQ 1011 (le_of_eq (rfl : 1011 ≥ 50))]
    norm_num
  case mpr =>
    intro hQn
    sorry

end smallest_n_l580_580127


namespace speed_rowing_upstream_l580_580139

theorem speed_rowing_upstream (V_m V_down : ℝ) (V_s V_up : ℝ)
  (h1 : V_m = 28) (h2 : V_down = 30) (h3 : V_down = V_m + V_s) (h4 : V_up = V_m - V_s) : 
  V_up = 26 :=
by
  sorry

end speed_rowing_upstream_l580_580139


namespace sum_of_ages_l580_580137

variable (A1 : ℝ) (A2 : ℝ) (A3 : ℝ) (A4 : ℝ) (A5 : ℝ) (A6 : ℝ) (A7 : ℝ)

noncomputable def age_first_scroll := 4080
noncomputable def age_difference := 2040

theorem sum_of_ages :
  let r := (age_difference:ℝ) / (age_first_scroll:ℝ)
  let A2 := (age_first_scroll:ℝ) + age_difference
  let A3 := A2 + (A2 - age_first_scroll) * r
  let A4 := A3 + (A3 - A2) * r
  let A5 := A4 + (A4 - A3) * r
  let A6 := A5 + (A5 - A4) * r
  let A7 := A6 + (A6 - A5) * r
  (age_first_scroll:ℝ) + A2 + A3 + A4 + A5 + A6 + A7 = 41023.75 := 
  by sorry

end sum_of_ages_l580_580137


namespace least_n_for_A0An_ge_200_l580_580321

theorem least_n_for_A0An_ge_200 :
  ∃ n : ℕ, (A_0 = (0, 0) ∧ (∀ i, A_i ∈ set_of (λ p, p.2 = 0)) ∧ (∀ j, B_j ∈ set_of (λ q, q.2 = q.1^2)) ∧ 
  (∀ k > 0, equilateral_triangle A_(k-1) B_k A_k) ∧ dist (A_0) (A_n) ≥ 200) ∧ n = 11 :=
by
  sorry

end least_n_for_A0An_ge_200_l580_580321


namespace paths_from_A_to_B_l580_580471

-- Conditions
variable (hexagonal_lattice : Type)
variable (A B : hexagonal_lattice)
variable (segments : hexagonal_lattice → hexagonal_lattice → Prop)
variable (directional_arrow : Π {x y: hexagonal_lattice}, segments x y → Prop)
variable (no_revisit_segment : Π {x y: hexagonal_lattice}, segments x y → Prop)
variable (updated_lattice : Prop) -- includes purple arrows and changed directions

-- Define the number of different paths
def different_paths (A B : hexagonal_lattice) : ℕ :=
2100

theorem paths_from_A_to_B {hexagonal_lattice : Type} (A B : hexagonal_lattice)
  (segments : hexagonal_lattice → hexagonal_lattice → Prop)
  (directional_arrow : Π {x y: hexagonal_lattice}, segments x y → Prop)
  (no_revisit_segment : Π {x y: hexagonal_lattice}, segments x y → Prop)
  (updated_lattice : Prop) :
  different_paths A B = 2100 := sorry

end paths_from_A_to_B_l580_580471


namespace similar_triangles_height_l580_580055

theorem similar_triangles_height (h₁ h₂ : ℝ) 
  (similar : ∀ (A₁ B₁ C₁ A₂ B₂ C₂ : Triangle), 
                (∃ k, k = 3 ∧ A₁ ≈ A₂ ∧ B₁ ≈ B₂ ∧ C₁ ≈ C₂) →
                (area A₁ / area A₂ = 1 / 9)) 
  (height_smaller : h₁ = 5)
  (area_ratio : area (Triangle.mk A₁ B₁ C₁) / area (Triangle.mk A₂ B₂ C₂) = 1 / 9) :
  h₂ = 15 := 
sorry

end similar_triangles_height_l580_580055


namespace water_saving_percentage_l580_580696

/-- 
Given:
1. The old toilet uses 5 gallons of water per flush.
2. The household flushes 15 times per day.
3. John saved 1800 gallons of water in June.

Prove that the percentage of water saved per flush by the new toilet compared 
to the old one is 80%.
-/
theorem water_saving_percentage 
  (old_toilet_usage_per_flush : ℕ)
  (flushes_per_day : ℕ)
  (savings_in_june : ℕ)
  (days_in_june : ℕ) :
  old_toilet_usage_per_flush = 5 →
  flushes_per_day = 15 →
  savings_in_june = 1800 →
  days_in_june = 30 →
  (old_toilet_usage_per_flush * flushes_per_day * days_in_june - savings_in_june)
  * 100 / (old_toilet_usage_per_flush * flushes_per_day * days_in_june) = 80 :=
by 
  sorry

end water_saving_percentage_l580_580696


namespace square_tiles_count_l580_580470

theorem square_tiles_count (a b : ℕ) (h1 : a + b = 25) (h2 : 3 * a + 4 * b = 84) : b = 9 := by
  sorry

end square_tiles_count_l580_580470


namespace james_100m_time_l580_580689

-- Definitions based on conditions
def john_total_time_to_run_100m : ℝ := 13 -- seconds
def john_first_second_distance : ℝ := 4 -- meters
def james_first_10m_time : ℝ := 2 -- seconds
def james_speed_increment : ℝ := 2 -- meters per second

-- Derived definitions based on conditions
def john_remaining_distance : ℝ := 100 - john_first_second_distance -- meters
def john_remaining_time : ℝ := john_total_time_to_run_100m - 1 -- seconds
def john_speed : ℝ := john_remaining_distance / john_remaining_time -- meters per second
def james_speed : ℝ := john_speed + james_speed_increment -- meters per second
def james_remaining_distance : ℝ := 100 - 10 -- meters
def james_time_for_remaining_distance : ℝ := james_remaining_distance / james_speed -- seconds
def james_total_time : ℝ := james_first_10m_time + james_time_for_remaining_distance -- seconds

-- Theorem statement
theorem james_100m_time : james_total_time = 11 := 
by 
  -- Place proof here
  sorry

end james_100m_time_l580_580689


namespace unit_digit_is_nine_l580_580504

theorem unit_digit_is_nine (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : a ≠ 0) (h4 : a + b + a * b = 10 * a + b) : b = 9 := 
by 
  sorry

end unit_digit_is_nine_l580_580504


namespace factorial_equation_solution_l580_580907

theorem factorial_equation_solution (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  a.factorial * b.factorial = a.factorial + b.factorial + c.factorial → (a, b, c) = (3, 3, 4) :=
by
  sorry

end factorial_equation_solution_l580_580907


namespace benny_crayons_l580_580017

theorem benny_crayons : ∀ (c_initial c_total : ℕ), c_initial = 9 → c_total = 12 → c_total - c_initial = 3 :=
by {
  intros c_initial c_total h_initial h_total,
  rw [h_initial, h_total],
  norm_num,
  sorry
}

end benny_crayons_l580_580017


namespace walter_age_in_2001_l580_580654

/-- In 1996, Walter was one-third as old as his grandmother, 
and the sum of the years in which they were born is 3864.
Prove that Walter will be 37 years old at the end of 2001. -/
theorem walter_age_in_2001 (y : ℕ) (H1 : ∃ g, g = 3 * y)
  (H2 : 1996 - y + (1996 - (3 * y)) = 3864) : y + 5 = 37 :=
by sorry

end walter_age_in_2001_l580_580654


namespace how_many_bones_in_adult_woman_l580_580290

-- Define the conditions
def numSkeletons : ℕ := 20
def halfSkeletons : ℕ := 10
def numAdultWomen : ℕ := 10
def numMenAndChildren : ℕ := 10
def numAdultMen : ℕ := 5
def numChildren : ℕ := 5
def totalBones : ℕ := 375

-- Define the proof statement
theorem how_many_bones_in_adult_woman (W : ℕ) (H : 10 * W + 5 * (W + 5) + 5 * (W / 2) = 375) : W = 20 :=
sorry

end how_many_bones_in_adult_woman_l580_580290


namespace triangle_area_l580_580286

theorem triangle_area (a b c : ℝ) (C : ℝ) (h1 : c^2 = (a - b)^2 + 6) (h2 : C = π / 3) :
    abs ((1 / 2) * a * b * Real.sin C) = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end triangle_area_l580_580286


namespace modulus_of_z_l580_580341

noncomputable def z : ℂ := sorry
def condition (z : ℂ) : Prop := z * (1 - Complex.I) = 2 * Complex.I

theorem modulus_of_z (hz : condition z) : Complex.abs z = Real.sqrt 2 := sorry

end modulus_of_z_l580_580341


namespace pima_investment_value_l580_580357

noncomputable def pima_investment_worth (initial_investment : ℕ) (first_week_gain_percentage : ℕ) (second_week_gain_percentage : ℕ) : ℕ :=
  let first_week_value := initial_investment + (initial_investment * first_week_gain_percentage / 100)
  let second_week_value := first_week_value + (first_week_value * second_week_gain_percentage / 100)
  second_week_value

-- Conditions
def initial_investment := 400
def first_week_gain_percentage := 25
def second_week_gain_percentage := 50

theorem pima_investment_value :
  pima_investment_worth initial_investment first_week_gain_percentage second_week_gain_percentage = 750 := by
  sorry

end pima_investment_value_l580_580357


namespace max_distance_D_l580_580400

-- Definitions for the problem
def A (z : ℂ) : ℂ := z
def B (z : ℂ) : ℂ := (2 + complex.I) * z
def C (z : ℂ) : ℂ := 3 * complex.conj z

/--
Given that |z| = 1, and points A, B, C are represented by complex numbers z, (2 + i)z, and 3 * \overline{z} respectively.
Prove that the maximum distance between the fourth vertex D of the parallelogram ABCD and the origin is \sqrt{22}.
-/
theorem max_distance_D (z : ℂ) (h1 : complex.norm z = 1) (h2 : A z ≠ B z ∧ B z ≠ C z ∧ A z ≠ C z ∧ ∃ D, A z + B z + C z + D = 0) :
  ∃ D, complex.norm D = real.sqrt 22 :=
sorry

end max_distance_D_l580_580400


namespace lcm_of_two_numbers_l580_580428

theorem lcm_of_two_numbers (x y : ℕ) (h1 : Nat.gcd x y = 12) (h2 : x * y = 2460) : Nat.lcm x y = 205 :=
by
  -- Proof omitted
  sorry

end lcm_of_two_numbers_l580_580428


namespace luke_pages_lemma_l580_580827

def number_of_new_cards : ℕ := 3
def number_of_old_cards : ℕ := 9
def cards_per_page : ℕ := 3
def total_number_of_cards := number_of_new_cards + number_of_old_cards
def total_number_of_pages := total_number_of_cards / cards_per_page

theorem luke_pages_lemma : total_number_of_pages = 4 := by
  sorry

end luke_pages_lemma_l580_580827


namespace arith_seq_ninth_term_value_l580_580955

variable {a : Nat -> ℤ}
variable {S : Nat -> ℤ}

def arith_seq (a : Nat -> ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + a 1^2

def arith_sum (S : Nat -> ℤ) (a : Nat -> ℤ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

theorem arith_seq_ninth_term_value
  (h_seq : arith_seq a)
  (h_sum : arith_sum S a)
  (h_cond1 : a 1 + a 2^2 = -3)
  (h_cond2 : S 5 = 10) :
  a 9 = 20 :=
by
  sorry

end arith_seq_ninth_term_value_l580_580955


namespace original_price_l580_580005

theorem original_price (p : ℝ) (h : 0 ≤ p) : 
    ∃ (x : ℝ), x * (1 - (p / 100) ^ 2 / 10000) = 1 → x = 10000 / (10000 - p ^ 2) :=
by
  intro x hx,
  use (10000 / (10000 - p ^ 2)),
  sorry

end original_price_l580_580005


namespace curve_E_equation_hyperbola_C_equation_coordinates_Q_l580_580233

noncomputable def circle_M := {p : ℝ × ℝ | (p.1 + 1) ^ 2 + p.2 ^ 2 = 1 / 4}
noncomputable def circle_N := {p : ℝ × ℝ | (p.1 - 1) ^ 2 + p.2 ^ 2 = 49 / 4}
noncomputable def circle_D (p : ℝ × ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧
    (∀ q ∈ circle_M, (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 = (r + 1 / 2) ^ 2) ∧
    (∀ q ∈ circle_N, (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 = (7 / 2 - r) ^ 2)

theorem curve_E_equation :
  ∀ p : ℝ × ℝ, circle_D p →
    (p.1 ^ 2) / 4 + (p.2 ^ 2) / 3 = 1 :=
sorry

theorem hyperbola_C_equation :
  ∀ p : ℝ × ℝ, ((p.1 ^ 2) / 4 + (p.2 ^ 2) / 3 = 1) →
    (p.1 ^ 2) - (p.2 ^ 2) / 3 = 1 :=
sorry

theorem coordinates_Q :
  ∃ (Q : ℝ × ℝ), (∀ l : ℝ × ℝ → Prop, ∃ k ∈ ℝ, l = (λ p, p.snd = k * p.fst + 4) ∧
    (∀ p ∈ l ∩ ({p | (p.1 ^ 2) - (p.2 ^ 2) / 3 = 1}), true)) →
    (Q = (±2, 0)) :=
sorry

end curve_E_equation_hyperbola_C_equation_coordinates_Q_l580_580233


namespace tabby_swimming_speed_l580_580752

theorem tabby_swimming_speed :
  ∃ S : ℝ, (∀ t1 t2 : ℝ, t1 = 6 ∧ t2 = 3.5 →
                 3.5 = (2 / (1 / S + 1 / t1))) ∧
              (S ≈ 2.47) :=
by
  existsi (2 / (2 / 3.5 - 1 / 6))
  intro t1 t2 H
  cases H with H1 H2
  rw [H1]
  have : 2 / t2 = 0.2,
  { sorry }, -- Here you would provide the detailed steps which are omitted
  finish 
  -- Prove S ≈ 2.47 based on the mathematical conditions
  sorry

end tabby_swimming_speed_l580_580752


namespace third_row_sequence_l580_580377

theorem third_row_sequence (sequence : ℕ → ℕ) (row1 row2 row3 : ℕ → ℕ) (third_row_num : ℕ) :
  (∀ n, sequence n = (n % 3) + 1) →
  (∀ i, (row1 i = sequence i) ∧ (row2 i = sequence (i + 3)) ∧ (row3 i = sequence (i + 6))) →
  (∀ j, ∃ i, row1 i = j ∧ row2 i = j ∧ row3 i = j) →
  (∀ k, third_row_num = 10000 * row3 0 + 1000 * row3 1 + 100 * row3 2 + 10 * row3 3 + row3 4) →
  third_row_num = 10302 :=
by
  intros sequence row1 row2 row3 third_row_num hseq hrows hcols hthird_row
  sorry

end third_row_sequence_l580_580377


namespace problem1_problem2_problem3_problem4_l580_580912

-- Define predicate conditions and solutions in Lean 4 for each problem

theorem problem1 (x : ℝ) :
  -2 * x^2 + 3 * x + 9 > 0 ↔ (-3 / 2 < x ∧ x < 3) := by
  sorry

theorem problem2 (x : ℝ) :
  (8 - x) / (5 + x) > 1 ↔ (-5 < x ∧ x ≤ 3 / 2) := by
  sorry

theorem problem3 (x : ℝ) :
  ¬ (-x^2 + 2 * x - 3 > 0) ↔ True := by
  sorry

theorem problem4 (x : ℝ) :
  x^2 - 14 * x + 50 > 0 ↔ True := by
  sorry

end problem1_problem2_problem3_problem4_l580_580912


namespace problem_solution_l580_580959

-- Definitions for the conditions
variables (A B C D : Type) [inst : EuclideanGeometry]
open EuclideanGeometry (Segment_length Angle)

-- Condition 1: Angle BAC = 2π/3
def angle_BAC_eq_two_pi_over_three :=
  Angle A B C = 2 * π / 3

-- Condition 2: AB = 2
def segment_AB_eq_two :=
  Segment_length A B = 2

-- Condition 3: AC = 1
def segment_AC_eq_one :=
  Segment_length A C = 1

-- Condition 4: DC = 2BD
def point_D_conditions :=
  let BD := Segment_length B D in
  let DC := Segment_length D C in
  DC = 2 * BD

-- The theorem statement
theorem problem_solution (A B C D : Point) [inst : EuclideanGeometry] :
  angle_BAC_eq_two_pi_over_three ∧
  segment_AB_eq_two ∧
  segment_AC_eq_one ∧
  point_D_conditions →
  (dot (A → D) (B → C) = -8/3) :=
by sorry

end problem_solution_l580_580959


namespace alpha_beta_square_l580_580274

noncomputable def roots_of_quadratic : set ℝ := 
  { x | x^2 = 2 * x + 1 }

theorem alpha_beta_square (α β : ℝ) (hα : α ∈ roots_of_quadratic) (hβ : β ∈ roots_of_quadratic) (hαβ : α ≠ β) :
  (α - β)^2 = 8 :=
sorry

end alpha_beta_square_l580_580274


namespace max_area_of_triangle_AOB_range_perimeter_GQF2_l580_580975

noncomputable def hyperbola_C := {p : ℝ × ℝ | p.1^2 / 3 - p.2^2 = 1}

theorem max_area_of_triangle_AOB : 
  ∀ (A B P : ℝ × ℝ) (λ : ℝ), 
  (A.snd = (√3 / 3) * A.fst) ∧ (B.snd = -(√3 / 3) * B.fst) ∧ (λ ∈ [1/3, 2]) ∧ 
  (P ∈ hyperbola_C) ∧ (P.1 - A.1, P.2 - A.2) = λ * (B.1 - P.1, B.2 - P.2) →
  ∃ (max_area : ℝ), max_area = 4 * (√3 / 3) := sorry

theorem range_perimeter_GQF2 : 
  ∀ (F1 F2 G Q : ℝ × ℝ) (k : ℝ), 
  F1 = (-2, 0) ∧ G.1.2 = (x, -√3 / 3 * x) ∧ k^2 > 1/3 →
  ∃ (perimeter : set ℝ), 
  perimeter = set.Icc (16 * √3 / 3) ∞ := sorry

end max_area_of_triangle_AOB_range_perimeter_GQF2_l580_580975


namespace height_of_larger_triangle_l580_580080

theorem height_of_larger_triangle 
  (area_ratio : ℝ)
  (height_small_triangle : ℝ)
  (similar_triangles : Prop)
  (height_large_triangle : ℝ) :
  area_ratio = 1 / 9 →
  height_small_triangle = 5 →
  similar_triangles →
  height_large_triangle = height_small_triangle * 3 :=
begin
  intros h_ratio h_height_small h_similar,
  rw h_ratio at *,
  rw h_height_small at *,
  exact eq.symm (mul_eq_mul_left_iff.1 (eq.trans (sqrt_eq (by norm_num) (by norm_num)) (by norm_num))),
sorry,
end

# The above code imports the necessary library, defines the theorem with the conditions and concludes with the height of the larger triangle.

end height_of_larger_triangle_l580_580080


namespace original_square_area_l580_580186

theorem original_square_area :
  ∀ (a b : ℕ), 
  (a * a = 24 * 1 * 1 + b * b ∧ 
  ((∃ m n : ℕ, (a + b = m ∧ a - b = n ∧ m * n = 24) ∨ 
  (a + b = n ∧ a - b = m ∧ m * n = 24)))) →
  a * a = 25 :=
by
  sorry

end original_square_area_l580_580186


namespace cos_square_sum_eq_one_l580_580560

theorem cos_square_sum_eq_one (x : ℝ) : 
  (cos x)^2 + (cos (2 * x))^2 + (cos (3 * x))^2 = 1 ↔ 
  ∃ k : ℤ, 
    x = (π / 2) + k * π ∨ 
    x = (π / 4) + 2 * k * π ∨ 
    x = (3 * π / 4) + 2 * k * π ∨ 
    x = (π / 6) + 2 * k * π ∨ 
    x = (5 * π / 6) + 2 * k * π := 
by
  sorry

end cos_square_sum_eq_one_l580_580560


namespace constant_is_arithmetic_l580_580438

def is_constant_sequence (a : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ n : ℕ, a n = c

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem constant_is_arithmetic (a : ℕ → ℝ) (h : is_constant_sequence a) : is_arithmetic_sequence a := by
  sorry

end constant_is_arithmetic_l580_580438


namespace correct_statements_l580_580508

-- Defining the data sets
def data_set1 : List ℕ := [4, 6, 7, 7, 9, 4]
def data_set2 : List ℕ := [3, 5, 7, 9]
def data_set3 : List ℕ := [6, 10, 14, 18]

-- Definitions for various statistical measures
def mode (l : List ℕ) : List ℕ :=
l.foldr (fun x acc => if l.count x = acc.head?.casesOn 0 l.count ((· = acc.head)) then x :: acc else acc) []

def standard_deviation (l : List ℕ) : ℝ :=
Real.sqrt ((l.foldl (λ acc x => acc + (x - l.mean) ^ 2) 0 : ℕ) / l.length)

def median (l : List ℕ) : ℝ :=
let sorted := l.toArray.qsort (· ≤ ·)
if sorted.size % 2 == 0 then
  (sorted.get! (sorted.size / 2 - 1 : Fin _)+ sorted.get! (sorted.size / 2 : Fin _)) / 2
else
  sorted.get! (sorted.size / 2 : Fin _)

-- Introducing proof problem
theorem correct_statements : (mode data_set1 = [4]) → 
                             (standard_deviation data_set2 = standard_deviation data_set3 / 2) →
                             (median data_set1 = 6.5) →
                             True := by
  intro h_mode h_std h_median
  sorry


end correct_statements_l580_580508


namespace omega_sum_condition_omega_sum_l580_580393

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem omega_sum_condition : 
  ∀ (ω : ℝ), 
    (ω > 0) ∧ 
    ∃ (φ : ℝ), |φ| ≤ Real.pi / 2 ∧ 
              (∀ x ∈ Set.Icc (π:ℝ) (3 * π / 2), ∀ y ∈ Set.Icc (π:ℝ) (3 * π / 2), (x < y → f ω φ x ≥ f ω φ y)) ∧
              (∃ (x : ℝ), f ω φ x = 0 ∧ (x = -π/4) ∧ (Real.cos (ω * x + φ) = 0)) ∧
              (∃ (x : ℝ), f ω φ x = f ω φ (π - x)) ∧ 
              (∀ (x1 x2 : ℝ), (x1 + π / 4 = x2 → f ω φ x1 = -f ω φ x2)) 
              → (ω = 2/5 ∨ ω = 2) :=
sorry

theorem omega_sum : 
  ∑ (ω : ℝ) in {ω | (ω > 0 ∧
                      ∃ (φ : ℝ), |φ| ≤ Real.pi / 2 ∧
                                (∀ x ∈ Set.Icc (π:ℝ) (3 * π / 2), ∀ y ∈ Set.Icc (π:ℝ) (3 * π / 2), (x < y → f ω φ x ≥ f ω φ y)) ∧
                                (∃ (x : ℝ), f ω φ x = 0 ∧ (x = -π/4) ∧ (Real.cos (ω * x + φ) = 0)) ∧
                                (∃ (x : ℝ), f ω φ x = f ω φ (π - x)) ∧ 
                                (∀ (x1 x2 : ℝ), (x1 + π / 4 = x2 → f ω φ x1 = -f ω φ x2)))}, 
           ω = 12/5 :=
sorry

end omega_sum_condition_omega_sum_l580_580393


namespace train_pass_time_l580_580666

theorem train_pass_time (speed_km_hr length_meter : ℕ) (km_to_meter_hr_to_sec : ℕ → ℕ → ℕ := fun km hr => (km * 1000) / hr):
  speed_km_hr = 54 → length_meter = 150 → 1.km = 1000.meter → 1.hr = 3600.sec → 
  (length_meter / (km_to_meter_hr_to_sec speed_km_hr 3600)) = 10 := 
  by
  intros h_speed h_length h_km h_hr
  rw [h_speed, h_length, h_km, h_hr]
  sorry

end train_pass_time_l580_580666


namespace triangle_medians_condition_l580_580675

theorem triangle_medians_condition (A B C : Point) (h : are_perpendicular (median A B C) (median C A B)) :
  AC^2 > area_of_triangle A B C := 
sorry

end triangle_medians_condition_l580_580675


namespace max_value_function_l580_580811

-- Define the function f(t)
def f (t : ℝ) : ℝ := (1 + Real.sin t) * (1 + Real.cos t)

-- State and prove the theorem
theorem max_value_function :
  ∃ t ∈ Icc 0 (2 * Real.pi), f t = (3 + 2 * Real.sqrt 2) / 2 :=
begin
  -- Proof would go here
  sorry
end

end max_value_function_l580_580811


namespace solution_l580_580253

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then 2^(x + 1) else 1 - (Real.logb (by norm_num : ℝ) 2 x)

theorem solution : f (f 3) = 4 / 3 := by
  sorry

end solution_l580_580253


namespace pima_investment_value_l580_580358

noncomputable def pima_investment_worth (initial_investment : ℕ) (first_week_gain_percentage : ℕ) (second_week_gain_percentage : ℕ) : ℕ :=
  let first_week_value := initial_investment + (initial_investment * first_week_gain_percentage / 100)
  let second_week_value := first_week_value + (first_week_value * second_week_gain_percentage / 100)
  second_week_value

-- Conditions
def initial_investment := 400
def first_week_gain_percentage := 25
def second_week_gain_percentage := 50

theorem pima_investment_value :
  pima_investment_worth initial_investment first_week_gain_percentage second_week_gain_percentage = 750 := by
  sorry

end pima_investment_value_l580_580358


namespace christel_gave_andrena_l580_580543

theorem christel_gave_andrena (d m c a: ℕ) (h1: d = 20 - 2) (h2: c = 24) 
  (h3: a = c + 2) (h4: a = d + 3) : (24 - c = 5) :=
by { sorry }

end christel_gave_andrena_l580_580543


namespace village_population_l580_580466

theorem village_population (P : ℝ) (h : 0.8 * P = 64000) : P = 80000 := by
  sorry

end village_population_l580_580466


namespace units_digit_base9_addition_l580_580565

theorem units_digit_base9_addition : 
  (∃ (d₁ d₂ : ℕ), d₁ < 9 ∧ d₂ < 9 ∧ (85 % 9 = d₁) ∧ (37 % 9 = d₂)) → ((d₁ + d₂) % 9 = 3) :=
by
  sorry

end units_digit_base9_addition_l580_580565


namespace probability_of_event_B_l580_580796

section
open Classical

def is_event_B (x y : ℕ) : Prop := x + y = 9
def total_possible_outcomes : ℕ := 36
def favorable_outcomes : ℕ := 4

theorem probability_of_event_B :
  let P_B := (favorable_outcomes : ℚ) / (total_possible_outcomes : ℚ) in
  P_B = 1 / 9 :=
by
  let P_B := (favorable_outcomes : ℚ) / (total_possible_outcomes : ℚ)
  have eq1 := rat.cast_coe_nat favorable_outcomes
  have eq2 := rat.cast_coe_nat total_possible_outcomes
  rw [eq1, eq2, ←rat.div_def, rat.coe_nat_div (show 36 ≠ 0, by norm_num), 
      show 4 / 36 = (1 : ℚ) / 9, by norm_num]
  simp
  sorry
end

end probability_of_event_B_l580_580796


namespace smallest_positive_period_f_intervals_of_increase_g_l580_580626

def f (x : ℝ) : ℝ := sqrt 3 * (cos (x / 2) - sin (x / 2)) * (cos (x / 2) + sin (x / 2)) + 2 * sin (x / 2) * cos (x / 2)

noncomputable def g (x : ℝ) : ℝ := 2 * sin (x + π / 6)

theorem smallest_positive_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) :=
sorry

theorem intervals_of_increase_g : ∀ k : ℤ, interval_integral (g' (le_interval_of_increase_g k)) 
=
   sorry

end smallest_positive_period_f_intervals_of_increase_g_l580_580626


namespace carlton_outfit_count_l580_580529

-- Definitions of conditions
def sweater_vests (s : ℕ) : ℕ := 2 * s
def button_up_shirts : ℕ := 3
def outfits (v s : ℕ) : ℕ := v * s

-- Theorem statement
theorem carlton_outfit_count : outfits (sweater_vests button_up_shirts) button_up_shirts = 18 :=
by
  sorry

end carlton_outfit_count_l580_580529


namespace problem_problem_contrapositive_l580_580335

def is_rational (r : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ r = p / q

def can_be_expressed_as_quotient (f : ℝ → ℝ) : Prop :=
  ∃ (p q : polynomial ℤ), q ≠ 0 ∧ ∀ x : ℝ, f x = (polynomial.eval x p) / (polynomial.eval x q)

def a (n : ℕ) : Nat := 0 -- This would normally come from the conditions, to ensure a_𝑛 ∈ {0, 1}

noncomputable def f (x : ℝ) : ℝ := ∑' n, (a n) * x^n

theorem problem (h : is_rational (f (1/2))) :
  can_be_expressed_as_quotient f :=
sorry

theorem problem_contrapositive (h : ¬ is_rational (f (1/2))) :
  ¬ can_be_expressed_as_quotient f :=
sorry

end problem_problem_contrapositive_l580_580335


namespace simson_lines_intersection_on_nine_point_circle_l580_580334

-- Given:
-- - Points P and P' are diametrically opposite on the circumcircle Γ of triangle ABC.
def circumcircle (ABC : Triangle) : Circle := sorry

axiom diametrically_opposite_on_circumcircle (ABC : Triangle) (P P' : Point) :
  P ∈ circumcircle ABC ∧ P' ∈ circumcircle ABC ∧ P ≠ P' ∧ 
  ∃ O, (O = circumcenter ABC ∧ dist P O = dist P' O ∧ Line.contains (Line.mk O P) P')

-- Prove:
-- - The Simson lines of P and P' intersect at a point on the nine-point circle of triangle ABC.
theorem simson_lines_intersection_on_nine_point_circle (ABC : Triangle) (P P' : Point)
  (h : diametrically_opposite_on_circumcircle ABC P P') :
  ∃ R, nine_point_circle ABC R ∧ Simson_line ABC P R ∧ Simson_line ABC P' R :=
sorry

end simson_lines_intersection_on_nine_point_circle_l580_580334


namespace university_students_l580_580163

theorem university_students (total_students students_both math_students physics_students : ℕ) 
  (h1 : total_students = 75) 
  (h2 : total_students = (math_students - students_both) + (physics_students - students_both) + students_both)
  (h3 : math_students = 2 * physics_students) 
  (h4 : students_both = 10) : 
  math_students = 56 := by
  sorry

end university_students_l580_580163


namespace f_decreasing_on_0_1_l580_580630

noncomputable def f (x : ℝ) : ℝ := x + 1 / x

theorem f_decreasing_on_0_1 : ∀ (x1 x2 : ℝ), (x1 ∈ Set.Ioo 0 1) → (x2 ∈ Set.Ioo 0 1) → (x1 < x2) → (f x1 < f x2) := by
  sorry

end f_decreasing_on_0_1_l580_580630


namespace odd_tiling_numbers_l580_580920

def f (n k : ℕ) : ℕ := sorry -- Assume f(n, 2k) is defined appropriately.

theorem odd_tiling_numbers (n : ℕ) : (∀ k : ℕ, f n (2*k) % 2 = 1) ↔ ∃ i : ℕ, n = 2^i - 1 := sorry

end odd_tiling_numbers_l580_580920


namespace find_area_triangle_ABC_l580_580350

variables {A B C F G E : Type*}
variables [NormedAddCommGroup A] [NormedAddCommGroup B] [NormedAddCommGroup C]
variables [NormedAddCommGroup F] [NormedAddCommGroup G] [NormedAddCommGroup E]

-- Assuming medians AF, BE intersect at G, m∠FGB = 60°, AF = 10, and BE = 15
variables (AF BE : ℝ) (mAG mBG : ℝ)
def triangle_conditions : Prop :=
  AF = 10 ∧ BE = 15 ∧ mAG = AF * 2 / 3 ∧ mBG = BE * 2 / 3 ∧
  ∃ (G : Type*), ∃ (angle_FGB : ℝ), angle_FGB = 60

def area_triangle_ABC_given_conditions : Prop :=
  let area_ABC := (100 * Real.sqrt 3) / 3 in
  area_ABC = sorry -- proof placeholder to assert the correct area

-- Main theorem statement
theorem find_area_triangle_ABC (AF BE : ℝ) (mAG mBG : ℝ) (angle_FGB : ℝ)
  (h_conditions : triangle_conditions AF BE mAG mBG) : 
  area_triangle_ABC_given_conditions AF BE mAG mBG :=
sorry

end find_area_triangle_ABC_l580_580350


namespace eddie_rate_l580_580746

variables (hours_sam hours_eddie rate_sam total_crates rate_eddie : ℕ)

def sam_conditions :=
  hours_sam = 6 ∧ rate_sam = 60

def eddie_conditions :=
  hours_eddie = 4 ∧ total_crates = hours_sam * rate_sam

theorem eddie_rate (hs : sam_conditions hours_sam rate_sam)
                   (he : eddie_conditions hours_sam hours_eddie rate_sam total_crates) :
  rate_eddie = 90 :=
by sorry

end eddie_rate_l580_580746


namespace find_constant_a_l580_580456

noncomputable def polar_line_converts_to_cartesian (a : ℝ) :=
  ∀ (theta : ℝ), let p := 2 * (Mathlib.trigonometric.cos theta) in
  p * (Mathlib.trigonometric.sin theta - Mathlib.trigonometric.cos theta) = a

-- Stating the conditions
def condition_line (p : ℝ) (theta : ℝ) (a : ℝ) : Prop :=
  p * (Mathlib.trigonometric.sin theta - Mathlib.trigonometric.cos theta) = a 

def condition_curve (p : ℝ) (theta : ℝ) : Prop :=
  p = 2 * Mathlib.trigonometric.cos theta

-- Define that the line divides the curve into equal parts
def divides_into_equal_parts (line_eq : ℝ → ℝ → Prop) (curve_eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, curve_eq x y → (line_eq x y → (x = 1 ∧ y = 0)) 

-- Prove that a = -1
theorem find_constant_a (a : ℝ) : 
  (∀(theta : ℝ), ∃ p, condition_line p theta a ∧ condition_curve p theta ∧ divides_into_equal_parts (λ x y, y - x = a) (λ x y, x^2 + y^2 = 2*x)) → a = -1 :=
sorry

end find_constant_a_l580_580456


namespace max_power_of_2_in_factorial_l580_580090

open Nat

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem max_power_of_2_in_factorial : ∑ k in range (log2 50 + 1), 50 / (2^k) ⌋ = 47 :=
by sorry

end max_power_of_2_in_factorial_l580_580090


namespace range_of_expression_l580_580722

theorem range_of_expression
  (a b c : ℝ)
  (h1 : a - b + c = 0)
  (h2 : c > 0)
  (h3 : 3 * a - 2 * b + c > 0) :
  set.Ioo (4 / 3) (7 / 2) =
    { y : ℝ | ∃ k : ℝ, 1 < k ∧ k < 2 ∧ y = (-6 + 10 * k) / (2 + k) } :=
by
  sorry

end range_of_expression_l580_580722


namespace spotted_and_fluffy_cats_l580_580495

theorem spotted_and_fluffy_cats (total_cats : ℕ) (total_cats_equiv : total_cats = 120) (one_third_spotted : ℕ → ℕ) (one_fourth_fluffy_spotted : ℕ → ℕ) :
  (one_third_spotted total_cats * one_fourth_fluffy_spotted (one_third_spotted total_cats) = 10) :=
by
  sorry

end spotted_and_fluffy_cats_l580_580495


namespace impossible_same_line_projections_of_skew_lines_l580_580436

-- Definition of skew lines
def skew_lines (L1 L2 : ℝ^3 → ℝ^3) : Prop :=
  ∃ (p1 p2 : ℝ^3), 
  L1 p1 ≠ L2 p1 ∧ L1 p2 ≠ L2 p2 ∧
  ∀ (x y : ℝ^3), (L1 x - L2 y) ≠ 0

-- Definition of projections onto a plane
def projection (L : ℝ^3 → ℝ^3) (plane : ℝ^3 → Bool) : ℝ^3 → ℝ^3 :=
  λ p, if plane p then L p else 0

-- Theorem statement
theorem impossible_same_line_projections_of_skew_lines (L1 L2 : ℝ^3 → ℝ^3) (plane : ℝ^3 → Bool) :
  skew_lines L1 L2 →
  ¬ ∃ (proj : ℝ^3 → ℝ^3), projection L1 plane = projection L2 plane := 
sorry

end impossible_same_line_projections_of_skew_lines_l580_580436


namespace find_y_intercept_l580_580201

theorem find_y_intercept (x1 y1 x2 y2 : ℝ) (h₁ : (x1, y1) = (2, -2)) (h₂ : (x2, y2) = (6, 6)) : 
  ∃ b : ℝ, (∀ x : ℝ, y = 2 * x + b) ∧ b = -6 :=
by
  sorry

end find_y_intercept_l580_580201


namespace arrow_sequence_1997_to_2000_l580_580093

def arrow_sequence : List Char := ['→', '↓', '↓', '↑', '↑', '→']

def position (n : Nat) : Nat := n % 6

theorem arrow_sequence_1997_to_2000 :
  position 1997 = 5 ∧ position 2000 = 2 → 
  List.take 3 [arrow_sequence[position 1997], arrow_sequence[position 1997 + 1], arrow_sequence[position 1997 + 2]] = ['↑', '→', '↓'] :=
by
  sorry

end arrow_sequence_1997_to_2000_l580_580093


namespace find_hyperbola_eccentricity_l580_580245

-- Define the given conditions and question
def ellipse_eccentricity (a b : ℝ) (h : a > b) (e : ℝ) : Prop :=
  e = sqrt (1 - (b / a)^2)

def hyperbola_eccentricity (a b : ℝ) (h : a > b) (e : ℝ) : Prop :=
  e = sqrt (1 + (b / a)^2)

theorem find_hyperbola_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : ellipse_eccentricity a b h1 (sqrt 3 / 2)) :
  hyperbola_eccentricity a b h1 (sqrt 5 / 2) :=
  sorry

end find_hyperbola_eccentricity_l580_580245


namespace leftover_pipes_l580_580788

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem leftover_pipes (total_pipes : ℕ) : total_pipes = 200 → 
  ∃ n : ℕ, triangular_number n ≤ total_pipes ∧ total_pipes - triangular_number n = 10 :=
by
  assume h : total_pipes = 200
  use 19
  have h₁ : triangular_number 19 = 190 := by
    unfold triangular_number
    norm_num
  have h₂ : 190 ≤ total_pipes := by
    rw h
    exact le_refl 200
  use_and_simplify (h₂, sorry) -- This line simplifies the conjunction
  rw [h, h₁]
  norm_num

end leftover_pipes_l580_580788


namespace complex_imaginary_part_l580_580963

noncomputable def z : ℂ := (3 + Complex.i) / (1 + Complex.i)

theorem complex_imaginary_part (z : ℂ) : z = (3 + Complex.i) / (1 + Complex.i) → z.im = -1 :=
by
  intro hz
  simp [Complex.div_eq_mul_inv, Complex.of_real_one, Complex.inv_def, Complex.mul_re, Complex.add_im, Complex.sub_im]
  simp [hz]
  sorry

end complex_imaginary_part_l580_580963


namespace shape_of_triangle_l580_580996

theorem shape_of_triangle (A B C : ℝ) (h : (sin (A + B)) * (sin (A - B)) = (sin C)^2) : 
  ∃ (C : ℝ), C = π / 2 ∨ C = -π / 2 := 
sorry

end shape_of_triangle_l580_580996


namespace similar_triangles_height_l580_580038

open_locale classical

theorem similar_triangles_height
  (h_small : ℝ)
  (A_small A_large : ℝ)
  (area_ratio : ℝ)
  (h_large : ℝ) :
  A_small / A_large = 1 / 9 →
  h_small = 5 →
  area_ratio = A_large / A_small →
  area_ratio = 9 →
  h_large = h_small * sqrt area_ratio →
  h_large = 15 :=
by
  sorry

end similar_triangles_height_l580_580038


namespace df_length_in_range_l580_580673

variables {A B C A₁ B₁ C₁ G E D F : Type*}
variables [inner_product_space ℝ (A B C A₁ B₁ C₁ G E D F)]
variables [finite_dimensional ℝ (A B C A₁ B₁ C₁ G E D F)]

-- Definitions based on geometric conditions
def is_right_triang_prism (A B C A1 : Type*) : Prop :=
  ∃ (A : Type*) (B : Type*) (C : Type*) (A1 : Type*),
  ∠ B A C = π / 2 ∧ dist A B = 1 ∧ dist A C = 1 ∧ dist A A₁ = 1

def is_midpoint (P Q M : Type*) :=
  ∃ (k : ℝ), dist P M = k * dist P Q ∧ dist Q M = k * dist Q P ∧ k = 1 / 2

def perpendicular (u v : Type*) :=
  ∃ (u v : Type*), dot_product u v = 0

def DF (D F : Type*) :=
  dist D F

-- Problem statement
theorem df_length_in_range (h_prism : is_right_triang_prism A B C A₁)
  (h_G : is_midpoint A₁ B₁ G)
  (h_E : is_midpoint C C₁ E)
  (h_perp : perpendicular (G D) (E F)) :
  1 / real.sqrt 5 ≤ DF D F ∧ DF D F < 1 :=
sorry

end df_length_in_range_l580_580673


namespace max_intersections_circle_quadrilateral_l580_580808

theorem max_intersections_circle_quadrilateral :
  ∀ (quad : ℝ² → Prop) (circle : ℝ² → Prop),
    (∀ (side : set ℝ²), (side ⊆ quad) → ∃ (intersections : set ℝ²),
    (intersections ⊆ side ∩ circle) ∧ (card intersections ≤ 2)) →
    (card (quad ∩ circle) ≤ 8) :=
by
  sorry

end max_intersections_circle_quadrilateral_l580_580808


namespace necessary_condition_for_q_implies_m_in_range_neg_p_or_neg_q_false_implies_x_in_range_l580_580588

-- Proof Problem 1
theorem necessary_condition_for_q_implies_m_in_range (m : ℝ) (h1 : 0 < m) :
  (∀ x : ℝ, 2 - m ≤ x ∧ x ≤ 2 + m → -2 ≤ x ∧ x ≤ 6) →
  0 < m ∧ m ≤ 4 :=
by
  sorry

-- Proof Problem 2
theorem neg_p_or_neg_q_false_implies_x_in_range (m : ℝ) (x : ℝ)
  (h2 : m = 2)
  (h3 : (x + 2) * (x - 6) ≤ 0)
  (h4 : 2 - m ≤ x ∧ x ≤ 2 + m)
  (h5 : ¬ ((x + 2) * (x - 6) > 0 ∨ x < 2 - m ∨ x > 2 + m)) :
  0 ≤ x ∧ x ≤ 4 :=
by
  sorry

end necessary_condition_for_q_implies_m_in_range_neg_p_or_neg_q_false_implies_x_in_range_l580_580588


namespace sequence_sqrt_limit_sequence_exp_over_factorial_limit_l580_580458

-- Problem 1: Prove the limit of the sequence defined by \(u_{1}=\sqrt{2}, u_{n+1}=\sqrt{2+u_{n}}\) is 2.
theorem sequence_sqrt_limit (u : ℕ → ℝ) 
  (h₁ : u 1 = real.sqrt 2)
  (h₂ : ∀ n, u (n + 1) = real.sqrt (2 + u n)) 
  (h₃ : ∃ l, filter.tendsto u filter.at_top (nhds l)) :
  ∃ c, c = 2 ∧ filter.tendsto u filter.at_top (nhds c) :=
by sorry

-- Problem 2: Prove the limit of the sequence defined by \(\nu_{n} = \frac{2^n}{n!}\) is 0.
theorem sequence_exp_over_factorial_limit (ν : ℕ → ℝ)
  (h₁ : ∀ n, ν n = (2:ℝ)^n / nat.factorial n) :
  filter.tendsto ν filter.at_top (nhds 0) :=
by sorry

end sequence_sqrt_limit_sequence_exp_over_factorial_limit_l580_580458


namespace vector_magnitude_and_angle_l580_580223

variables (n b : ℝ^3)
variables (hnorm : ∥n∥ = 6) (bnorm : ∥b∥ = 6)
variables (angle_nb : real.angle (n, b) = real.pi / 3)

theorem vector_magnitude_and_angle :
  ∥n + b∥ = 6 * real.sqrt 3 ∧
  ∥n - b∥ = 6 ∧
  real.angle (n + b) (n - b) = real.pi / 2 :=
by
  sorry

end vector_magnitude_and_angle_l580_580223


namespace similar_triangles_height_l580_580052

theorem similar_triangles_height (h₁ h₂ : ℝ) 
  (similar : ∀ (A₁ B₁ C₁ A₂ B₂ C₂ : Triangle), 
                (∃ k, k = 3 ∧ A₁ ≈ A₂ ∧ B₁ ≈ B₂ ∧ C₁ ≈ C₂) →
                (area A₁ / area A₂ = 1 / 9)) 
  (height_smaller : h₁ = 5)
  (area_ratio : area (Triangle.mk A₁ B₁ C₁) / area (Triangle.mk A₂ B₂ C₂) = 1 / 9) :
  h₂ = 15 := 
sorry

end similar_triangles_height_l580_580052


namespace monotonic_interval_f_l580_580252

theorem monotonic_interval_f (ω : ℝ) (φ : ℝ) (k : ℤ) :
  (ω > 0) →
  (|φ| < π) →
  (2 * Real.sin(ω * (π / 3) + φ) - 1 = 0) →
  (Real.sin(-ω * (π / 6) + φ) = 1 ∨ Real.sin(-ω * (π / 6) + φ) = -1) →
  ω = (2 / 3) →
  ∃ (k : ℤ), ∀ (x : ℝ),
    (3 * k * π - (5 * π / 3) ≤ x ∧ x ≤ 3 * k * π - π / 6) ↔
    (1 ≤ 2 * Real.sin(ω * x + φ) ∧ 2 * Real.sin(ω * x + φ) ≤ 1) :=
by sorry

end monotonic_interval_f_l580_580252


namespace identify_counterfeit_in_three_weighings_l580_580013

def CoinType := {x // x = "gold" ∨ x = "silver"}

structure Coins where
  golds: Fin 13
  silvers: Fin 14
  is_counterfeit: CoinType
  counterfeit_weight: Int

def is_lighter (c1 c2: Coins): Prop := sorry
def is_heavier (c1 c2: Coins): Prop := sorry
def balance (c1 c2: Coins): Prop := sorry

def find_counterfeit_coin (coins: Coins): Option Coins := sorry

theorem identify_counterfeit_in_three_weighings (coins: Coins) :
  ∃ (identify: Coins → Option Coins),
  ∀ coins, ( identify coins ≠ none ) :=
sorry

end identify_counterfeit_in_three_weighings_l580_580013


namespace friends_left_after_process_l580_580347

def process_initial : Nat := 100
def add_before_first_cycle : Nat := 20
def add_before_second_cycle : Nat := 10
def add_before_third_cycle : Nat := 5
def keep_first_cycle : Float := 0.40
def respond_first_cycle : Float := 0.50
def keep_second_cycle : Float := 0.60
def respond_second_cycle : Float := 0.70
def keep_third_cycle : Float := 0.80
def respond_third_cycle : Float := 0.40

theorem friends_left_after_process :
  let initial := process_initial
  let after_add_first := initial + add_before_first_cycle
  let keep_first := Float.toNat (after_add_first * keep_first_cycle)
  let contact_first := Float.toNat (after_add_first * (1 - keep_first_cycle))
  let respond_first := Float.toNat (contact_first * respond_first_cycle)
  let after_first_cycle := keep_first + respond_first
  let after_add_second := after_first_cycle + add_before_second_cycle
  let keep_second := Float.toNat (after_add_second * keep_second_cycle)
  let contact_second := Float.toNat (after_add_second * (1 - keep_second_cycle))
  let respond_second := Float.toNat (contact_second * respond_second_cycle)
  let after_second_cycle := keep_second + respond_second
  let after_add_third := after_second_cycle + add_before_third_cycle
  let keep_third := Float.toNat (after_add_third * keep_third_cycle)
  let contact_third := Float.toNat (after_add_third * (1 - keep_third_cycle))
  let respond_third := Float.toNat (contact_third * respond_third_cycle)
  let final_friend_count := keep_third + respond_third
  final_friend_count = 74 :=
by
  sorry

end friends_left_after_process_l580_580347


namespace three_digit_number_solution_l580_580786

theorem three_digit_number_solution :
  ∃ (x y : ℕ), x ∈ {0, 1, 2, … , 9} ∧ y ∈ {0, 1, 2, … , 9} ∧
  y = 20 - 10 * x ∧ (100 * x + 10 * y + 2 = 202) :=
begin
  use [2, 0],
  split,
  { exact Decidable.is_true (by norm_num) },
  split,
  { exact Decidable.is_true (by norm_num) },
  split,
  { exact rfl },
  { exact rfl }
end

end three_digit_number_solution_l580_580786


namespace similar_triangles_height_l580_580046

theorem similar_triangles_height (h₁ h₂ : ℝ) (ratio_areas : ℝ) 
  (h₁_eq : h₁ = 5) (ratio_areas_eq : ratio_areas = 1 / 9)
  (similar : h₂^2 = (√ratio_areas)^2 * h₁^2) : h₂ = 15 :=
by {
  have ratio_areas_pos : ratio_areas > 0 := by (simp [ratio_areas_eq]),
  have k := √ratio_areas,
  have k_eq : k = 3 := by {
    rw [ratio_areas_eq, sqrt_div, sqrt_one, sqrt_nat_eq_iff_eq_sq, one_div_eq_inv] at *,
    norm_num },
  have h₂_def : h₂ = 3 * h₁ := by rw [h₁_eq, mul_comm, k_eq],
  rw [h₂_def],
  norm_num,
}

end similar_triangles_height_l580_580046


namespace sequence_explicit_formula_l580_580009

theorem sequence_explicit_formula (a : ℕ → ℕ) (h₁ : a 1 = 0) (h₂ : ∀ n ≥ 1, a (n + 1) = ∑ k in finset.range(n), (a (k + 1) + 1)) :
∃ f : ℕ → ℕ, (∀ n ≥ 1, a n = (2^(n-1)) - 1) :=
by
  sorry

end sequence_explicit_formula_l580_580009


namespace line_intersects_circle_l580_580938

-- Definitions
structure Point where
  a : ℝ
  b : ℝ

-- Conditions
def outside_circle (M : Point) : Prop :=
  (M.a ^ 2 + M.b ^ 2) > 1

def on_line (p : Point) : Prop :=
  p.a * x + p.b * y = 1

def on_circle (p : Point) : Prop :=
  x ^ 2 + y ^ 2 = 1

-- Theorem
theorem line_intersects_circle (M : Point) (h: outside_circle M) :
  ∃ (x y : ℝ), on_line ⟨x, y⟩ ∧ on_circle ⟨x, y⟩ :=
sorry

end line_intersects_circle_l580_580938


namespace neg_one_quadratic_residue_iff_l580_580712

theorem neg_one_quadratic_residue_iff (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 2 = 1) : 
  (∃ x : ℤ, x^2 ≡ -1 [ZMOD p]) ↔ p % 4 = 1 :=
sorry

end neg_one_quadratic_residue_iff_l580_580712


namespace combination_identity_arrangement_identity_l580_580374

theorem combination_identity (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 9) 
  (h3 : nat.choose 9 x = nat.choose 9 (2 * x - 3)) : x = 3 ∨ x = 4 := 
sorry

theorem arrangement_identity (x : ℕ) (h1 : 0 < x) (h2 : x ≤ 8)
  (h3 : nat.factorial 8 / nat.factorial (8 - x) = 6 * (nat.factorial 8 / nat.factorial (10 - x))) : x = 7 := 
sorry

end combination_identity_arrangement_identity_l580_580374


namespace parabola_equation_l580_580143

-- Define the constants and the conditions
def parabola_focus : ℝ × ℝ := (3, 3)
def directrix : ℝ × ℝ × ℝ := (3, 7, -21)

theorem parabola_equation :
  ∃ a b c d e f : ℤ,
  a > 0 ∧
  Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd a b) c) d) e) f = 1 ∧
  (a : ℝ) * x^2 + (b : ℝ) * x * y + (c : ℝ) * y^2 + (d : ℝ) * x + (e : ℝ) * y + (f : ℝ) = 
  49 * x^2 - 42 * x * y + 9 * y^2 - 222 * x - 54 * y + 603 := sorry

end parabola_equation_l580_580143


namespace ratio_AH_HC_l580_580680

theorem ratio_AH_HC (A B C N M H : Point) (h_ABC : Triangle A B C)
  (h_CNM : Triangle C N M) (h_perpendicular : Perpendicular B C M)
  (angle_C : angle A B C = 30)
  (angle_Acute : angle B A C < 90)
  (h_area_ratio : area h_CNM = 3 / 16 * area h_ABC)
  (h_MN_half_BH : distance M N = 1 / 2 * height B H h_ABC)
  (h_Altitude: Altitude BH h_ABC) :
  ratio (distance A H) (distance H C) = 1 / 3 :=
by {
  sorry
}

end ratio_AH_HC_l580_580680


namespace michael_work_time_l580_580351

theorem michael_work_time (M A L : ℚ) 
  (h1 : M + A + L = 1/15) 
  (h2 : A + L = 1/24) :
  1 / M = 40 := 
by
  sorry

end michael_work_time_l580_580351


namespace find_b_l580_580645

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * b) : b = 49 :=
by
  sorry

end find_b_l580_580645


namespace range_of_f_x_minus_1_plus_f_x_squared_minus_x_gt_0_l580_580614

def f(x: ℝ) : ℝ := sorry
def f''(x: ℝ) : ℝ := x^2 + 2 * Real.cos x

theorem range_of_f_x_minus_1_plus_f_x_squared_minus_x_gt_0 :
  (∀ x, f'(x) = (1/3) * x^3 + 2 * Real.sin x + C ∧ f(0) = 0) →
  (∀ x ∈ Ioo 1 2, f(x - 1) + f(x^2 - x) > 0) :=
sorry

end range_of_f_x_minus_1_plus_f_x_squared_minus_x_gt_0_l580_580614


namespace min_value_of_function_l580_580770

theorem min_value_of_function (p : ℝ) : 
  ∃ x : ℝ, (x^2 - 2 * p * x + 2 * p^2 + 2 * p - 1) = -2 := sorry

end min_value_of_function_l580_580770


namespace product_of_area_and_perimeter_of_rectangle_l580_580354

open Real EuclideanGeometry

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def area_rectangle (A B C D : ℝ × ℝ) : ℝ :=
  distance A B * distance A D

def perimeter_rectangle (A B C D : ℝ × ℝ) : ℝ :=
  2 * (distance A B + distance A D)

noncomputable def product_area_perimeter (A B C D : ℝ × ℝ) : ℝ :=
  area_rectangle A B C D * perimeter_rectangle A B C D

theorem product_of_area_and_perimeter_of_rectangle :
  product_area_perimeter (3, 4) (4, 1) (2, 0) (1, 3) = 20 * sqrt 5 + 10 * sqrt 10 :=
by
  sorry

end product_of_area_and_perimeter_of_rectangle_l580_580354


namespace true_proposition_l580_580602

variable (p q : Prop)
variable (hp : p = true)
variable (hq : q = false)

theorem true_proposition : (¬p ∨ ¬q) = true := by
  sorry

end true_proposition_l580_580602


namespace quadratic_has_two_distinct_real_roots_l580_580010

theorem quadratic_has_two_distinct_real_roots : 
  ∃ α β : ℝ, (α ≠ β) ∧ (2 * α^2 - 3 * α + 1 = 0) ∧ (2 * β^2 - 3 * β + 1 = 0) :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l580_580010


namespace prime_eq_sum_of_two_squares_l580_580241

theorem prime_eq_sum_of_two_squares (p : ℕ) (hp_prime : Nat.Prime p) (hp_mod : p % 4 = 1) : 
  ∃ a b : ℤ, p = a^2 + b^2 := 
sorry

end prime_eq_sum_of_two_squares_l580_580241


namespace find_solution_set_and_validate_l580_580913

theorem find_solution_set_and_validate :=
  let condition1 (x : ℝ) := 5 / 2 * x - 1 > 3 * x
  let solution_set (x : ℝ) := x < -2
  show ∀ x, condition1 x ↔ solution_set x ∧ ¬ solution_set (-real.sqrt 2), from
  sorry

end find_solution_set_and_validate_l580_580913


namespace pages_with_same_units_digit_l580_580141

theorem pages_with_same_units_digit :
  (∃ (x : Finset ℕ), 
     (∀ (p ∈ x), 1 ≤ p ∧ p ≤ 60 ∧ (p % 10 = (61 - p) % 10)) ∧
     x.card = 6) :=
sorry

end pages_with_same_units_digit_l580_580141


namespace similar_triangles_height_l580_580049

theorem similar_triangles_height (h₁ h₂ : ℝ) (ratio_areas : ℝ) 
  (h₁_eq : h₁ = 5) (ratio_areas_eq : ratio_areas = 1 / 9)
  (similar : h₂^2 = (√ratio_areas)^2 * h₁^2) : h₂ = 15 :=
by {
  have ratio_areas_pos : ratio_areas > 0 := by (simp [ratio_areas_eq]),
  have k := √ratio_areas,
  have k_eq : k = 3 := by {
    rw [ratio_areas_eq, sqrt_div, sqrt_one, sqrt_nat_eq_iff_eq_sq, one_div_eq_inv] at *,
    norm_num },
  have h₂_def : h₂ = 3 * h₁ := by rw [h₁_eq, mul_comm, k_eq],
  rw [h₂_def],
  norm_num,
}

end similar_triangles_height_l580_580049


namespace part_I_part_II_l580_580257

section
-- Definitions for Part (I)
def f (x : ℝ) (a : ℝ) : ℝ := |x - a|

theorem part_I (x : ℝ) (a : ℝ) (h : a = 3) : 
  f x a ≥ 4 - |x + 1| := 
by 
  rw [h, f]
  sorry

-- Definitions for Part (II)
def solution_set (f : ℝ → ℝ) : set ℝ := {x : ℝ | f x ≤ 1}

theorem part_II (a m n : ℝ) (m_pos : m > 0) (n_pos : n > 0)
  (h1 : solution_set (λ x, |x - a|) = set.Icc 1 3)
  (h2 : a = 2)
  (h3 : 1 / m + 1 / (2 * n) = a) :
  ∀ m n > 0, m + 2 * n ≥ 4 * real.sqrt 2 := 
by
  sorry
end

end part_I_part_II_l580_580257


namespace find_f_of_half_l580_580219

theorem find_f_of_half (f : ℝ → ℝ) (α : ℝ) (h : f (sin α + cos α) = sin α * cos α) : 
  f (sin (π / 6)) = -3 / 8 :=
by
  have h2 : sin (π / 6) = 1 / 2 := by sorry
  rw [h2]
  sorry

end find_f_of_half_l580_580219


namespace probability_first_ace_second_spade_l580_580419

theorem probability_first_ace_second_spade :
  let deck := List.range 52 in
  let first_is_ace (card : ℕ) := card % 13 = 0 in
  let second_is_spade (card : ℕ) := card / 13 = 3 in
  let events :=
    [ ((first_is_ace card, second_is_spade card') | card ∈ deck, card' ∈ List.erase deck card) ] in
  let favorable_events :=
    [(true, true)] in
  (List.count (λ event => event ∈ favorable_events) events).toRat /
  (List.length events).toRat = 1 / 52 :=
sorry

end probability_first_ace_second_spade_l580_580419


namespace probability_x_gt_3y_l580_580737

theorem probability_x_gt_3y :
  let A := 3030 * 1010 / 2 : ℝ
  let R := 3030 * 3031 : ℝ
  A / R = (505 / 3031 : ℝ) := sorry

end probability_x_gt_3y_l580_580737


namespace probability_of_ace_then_spade_l580_580423

theorem probability_of_ace_then_spade :
  let P := (1 / 52) * (12 / 51) + (3 / 52) * (13 / 51)
  P = (3 / 127) :=
by
  sorry

end probability_of_ace_then_spade_l580_580423


namespace largest_number_l580_580437

-- Definitions based on the conditions
def numA := 0.893
def numB := 0.8929
def numC := 0.8931
def numD := 0.839
def numE := 0.8391

-- The statement to be proved 
theorem largest_number : numB = max numA (max numB (max numC (max numD numE))) := by
  sorry

end largest_number_l580_580437


namespace smallest_product_is_128_l580_580883

def smallest_result_from_set := ∀ (A B C D : ℕ), 
  A ∈ {3, 5, 7, 9, 11, 13} ∧ 
  B ∈ {3, 5, 7, 9, 11, 13} ∧ 
  C ∈ {3, 5, 7, 9, 11, 13} ∧ 
  D ∈ {3, 5, 7, 9, 11, 13} ∧ 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ 
  B ≠ C ∧ B ≠ D ∧ 
  C ≠ D → 
  let sum1 := A + B in
  let sum2 := C + D in
  (sum1 * sum2) ≥ 128

theorem smallest_product_is_128 : smallest_result_from_set :=
by
  -- Insert the proof here
  sorry

end smallest_product_is_128_l580_580883


namespace cost_price_of_watch_l580_580102

theorem cost_price_of_watch (CP : ℝ) (h1 : SP1 = CP * 0.64) (h2 : SP2 = CP * 1.04) (h3 : SP2 = SP1 + 140) : CP = 350 :=
by
  sorry

end cost_price_of_watch_l580_580102


namespace james_100m_time_l580_580688

-- Definitions based on conditions
def john_total_time_to_run_100m : ℝ := 13 -- seconds
def john_first_second_distance : ℝ := 4 -- meters
def james_first_10m_time : ℝ := 2 -- seconds
def james_speed_increment : ℝ := 2 -- meters per second

-- Derived definitions based on conditions
def john_remaining_distance : ℝ := 100 - john_first_second_distance -- meters
def john_remaining_time : ℝ := john_total_time_to_run_100m - 1 -- seconds
def john_speed : ℝ := john_remaining_distance / john_remaining_time -- meters per second
def james_speed : ℝ := john_speed + james_speed_increment -- meters per second
def james_remaining_distance : ℝ := 100 - 10 -- meters
def james_time_for_remaining_distance : ℝ := james_remaining_distance / james_speed -- seconds
def james_total_time : ℝ := james_first_10m_time + james_time_for_remaining_distance -- seconds

-- Theorem statement
theorem james_100m_time : james_total_time = 11 := 
by 
  -- Place proof here
  sorry

end james_100m_time_l580_580688


namespace evaluate_expression_l580_580556

theorem evaluate_expression : (532 * 532) - (531 * 533) = 1 := by
  sorry

end evaluate_expression_l580_580556


namespace pure_imaginary_solution_l580_580327

theorem pure_imaginary_solution (a : ℝ) :
  let i := complex.I in 
  let z := a^2 + 2 * a - 2 + (2 * i) / (1 - i) in 
  im z ≠ 0 ∧ re z = 0 ↔ a = -3 ∨ a = 1 := sorry

end pure_imaginary_solution_l580_580327


namespace least_n_divisible_subset_l580_580702

noncomputable def A (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) : Finset ℕ :=
  { n | ∃ a b c : ℕ, 0 ≤ a ∧ a ≤ 5 ∧ 0 ≤ b ∧ b ≤ 5 ∧ 0 ≤ c ∧ c ≤ 5 ∧ n = p^a * q^b * r^c }.toFinset

theorem least_n_divisible_subset (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  ∃ n : ℕ, ∀ B : Finset ℕ, B ⊆ A p q r hp hq hr → B.card = n →
  ∃ x y ∈ B, x ≠ y ∧ x ∣ y :=
begin
  sorry
end

end least_n_divisible_subset_l580_580702


namespace A_wins_match_prob_correct_l580_580661

def probA_wins_game : ℝ := 0.6
def probB_wins_game : ℝ := 0.4

def probA_wins_match : ℝ :=
  let probA_wins_first_two := probA_wins_game * probA_wins_game
  let probA_wins_first_and_third := probA_wins_game * probB_wins_game * probA_wins_game
  let probA_wins_last_two := probB_wins_game * probA_wins_game * probA_wins_game
  probA_wins_first_two + probA_wins_first_and_third + probA_wins_last_two

theorem A_wins_match_prob_correct : probA_wins_match = 0.648 := by
  sorry

end A_wins_match_prob_correct_l580_580661


namespace high_jump_statistics_l580_580516

theorem high_jump_statistics :
  let heights : Multiset ℝ := {1.50, 1.50, 1.60, 1.60, 1.60, 1.65, 1.65, 1.65, 1.65, 1.65, 1.70, 1.70, 1.70, 1.70, 1.75}
  in (Multiset.mode heights = some 1.65) ∧ (median heights = 1.65) :=
by sorry

end high_jump_statistics_l580_580516


namespace similar_triangles_height_l580_580056

theorem similar_triangles_height (h₁ h₂ : ℝ) 
  (similar : ∀ (A₁ B₁ C₁ A₂ B₂ C₂ : Triangle), 
                (∃ k, k = 3 ∧ A₁ ≈ A₂ ∧ B₁ ≈ B₂ ∧ C₁ ≈ C₂) →
                (area A₁ / area A₂ = 1 / 9)) 
  (height_smaller : h₁ = 5)
  (area_ratio : area (Triangle.mk A₁ B₁ C₁) / area (Triangle.mk A₂ B₂ C₂) = 1 / 9) :
  h₂ = 15 := 
sorry

end similar_triangles_height_l580_580056


namespace average_price_l580_580820

theorem average_price (books1 books2 : ℕ) (price1 price2 : ℝ)
  (h1 : books1 = 65) (h2 : price1 = 1380)
  (h3 : books2 = 55) (h4 : price2 = 900) :
  (price1 + price2) / (books1 + books2) = 19 :=
by
  sorry

end average_price_l580_580820


namespace minimum_handshakes_l580_580464

theorem minimum_handshakes (n : ℕ) (h : n = 30) (cond : ∀ i : ℕ, i < n → ∃ k : ℕ, k ≥ 3 ∧ i < n ∧ shake_hands i k) : ∃ (h_num : ℕ), h_num = 45 :=
by
  -- conditions setup
  sorry

end minimum_handshakes_l580_580464


namespace arithmetic_geometric_sequence_l580_580670

theorem arithmetic_geometric_sequence
    (a : ℕ → ℕ)
    (b : ℕ → ℕ)
    (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0) -- Definition of arithmetic sequence
    (h_geom_seq : ∀ n, b (n + 1) / b n = b 1 / b 0) -- Definition of geometric sequence
    (h_a3_a11 : a 3 + a 11 = 8) -- Condition a_3 + a_11 = 8
    (h_b7_a7 : b 7 = a 7) -- Condition b_7 = a_7
    : b 6 * b 8 = 16 := -- Prove that b_6 * b_8 = 16
sorry

end arithmetic_geometric_sequence_l580_580670


namespace pq_parallel_ab_l580_580308

-- Given problem setup
variables (A B C M K L P Q : Type*) [MetricSpace A]

-- Conditions
variables (h_M_on_AB : M ∈ line[A, B])
variables (h_K_on_BC : K ∈ line[B, C])
variables (h_L_on_AC : L ∈ line[A, C])
variables (h_MK_parallel_AC : Parallel (line[M, K]) (line[A, C]))
variables (h_ML_parallel_BC : Parallel (line[M, L]) (line[B, C]))
variables (h_BL_inter_MK : ∃ P, P ∈ line[B, L] ∧ P ∈ line[M, K])
variables (h_AK_inter_ML : ∃ Q, Q ∈ line[A, K] ∧ Q ∈ line[M, L])

-- Question
theorem pq_parallel_ab : Parallel (line[P, Q]) (line[A, B]) :=
sorry

end pq_parallel_ab_l580_580308


namespace infinite_rectangular_prism_containment_l580_580944

open Classical

theorem infinite_rectangular_prism_containment :
  ∀ (S : Set (ℕ × ℕ × ℕ)), Set.Infinite S →
  ∃ p q ∈ S, p ≠ q ∧ (p.1 ≤ q.1 ∧ p.2 ≤ q.2 ∧ p.3 ≤ q.3 ∨ q.1 ≤ p.1 ∧ q.2 ≤ p.2 ∧ q.3 ≤ p.3) :=
begin
  sorry
end

end infinite_rectangular_prism_containment_l580_580944


namespace max_value_of_f_l580_580715

noncomputable def f (x : ℝ) : ℝ := min (4 * x + 1) (min (x + 2) (-2 * x + 4))

theorem max_value_of_f : ∃ x : ℝ, f(x) = (8 / 3) ∧ ∀ y : ℝ, f(y) ≤ (8 / 3) := by
  sorry

end max_value_of_f_l580_580715


namespace tom_total_cost_l580_580028

theorem tom_total_cost :
  let vaccines_cost := 10 * 45 in
  let total_medical_cost := vaccines_cost + 250 in
  let insurance_covered := 0.80 * total_medical_cost in
  let tom_pay_medical := total_medical_cost - insurance_covered in
  let trip_cost := 1200 in
  let total_cost := tom_pay_medical + trip_cost in
  total_cost = 1340 :=
by
  dsimp
  sorry

end tom_total_cost_l580_580028


namespace james_100m_time_l580_580686

def john_time : ℝ := 13
def john_initial_distance : ℝ := 4
def james_speed_advantage : ℝ := 2
def james_initial_distance : ℝ := 10
def james_initial_time : ℝ := 2

theorem james_100m_time : true := by
  -- John's specifications
  let john_total_distance := 100
  let john_top_speed_distance := john_total_distance - john_initial_distance
  let john_top_speed_time := john_time - 1
  let john_top_speed := john_top_speed_distance / john_top_speed_time
  
  -- James's specifications
  let james_top_speed := john_top_speed + james_speed_advantage
  let james_remaining_distance := john_total_distance - james_initial_distance
  let james_remaining_time := james_remaining_distance / james_top_speed
  let james_total_time := james_initial_time + james_remaining_time
  
  -- The condition to prove
  have : james_total_time = 11 := sorry
  
  exact trivial

end james_100m_time_l580_580686


namespace irrigation_system_flow_rates_l580_580131

variable (q0 qDE qBC qGF : ℝ)

-- Conditions
axiom total_flow_rate : q0 > 0
axiom identical_channels : True -- placeholder for the identically identical channels condition
axiom constant_flow_rates : ∀ path : List String, flow_rate path = q0 -- placeholder for constant flow rates condition

-- Prove the flow rates
theorem irrigation_system_flow_rates :
  qDE = (4 / 7) * q0 ∧ qBC = (2 / 7) * q0 ∧ qGF = (3 / 7) * q0 :=
by
  sorry

end irrigation_system_flow_rates_l580_580131


namespace clerical_percentage_correct_l580_580730

def initial_employees : ℕ := 5400
def clerical_fraction : ℚ := 2/5
def reduction_fraction : ℚ := 1/4

def initial_clerical_employees : ℕ := (clerical_fraction * initial_employees).toNat
def reduced_clerical_employees : ℕ := (reduction_fraction * initial_clerical_employees).toNat
def new_clerical_employees : ℕ := initial_clerical_employees - reduced_clerical_employees
def remaining_employees : ℕ := initial_employees - reduced_clerical_employees
def clerical_percentage : ℚ := (new_clerical_employees : ℚ) / remaining_employees * 100

theorem clerical_percentage_correct : clerical_percentage ≈ 33.33 := 
by
  have h1 : initial_clerical_employees = 2160 := rfl
  have h2 : reduced_clerical_employees = 540 := rfl
  have h3 : new_clerical_employees = 1620 := rfl
  have h4 : remaining_employees = 4860 := rfl
  have h5 : clerical_percentage = (1620 : ℚ) / 4860 * 100 := rfl
  suffices : (1620 / 4860 * 100 : ℚ) ≈ 33.33, from this
  sorry

end clerical_percentage_correct_l580_580730


namespace max_a_condition_l580_580961

theorem max_a_condition (x a : ℝ) (h : ∀ x : ℝ, (x + a)^2 - 16 > 0 ↔ x ≤ -3 ∨ x ≥ 2) : a ≤ 2 :=
by
  have h1 : ∀ x : ℝ, (x + a)^2 > 16 ↔ x ≤ -3 ∨ x ≥ 2 := by 
    intro x
    exact (h x).mpr
  have h2 : -1 ≤ a ∧ a ≤ 2 := sorry
  exact h2.2

end max_a_condition_l580_580961


namespace similar_triangles_height_l580_580040

open_locale classical

theorem similar_triangles_height
  (h_small : ℝ)
  (A_small A_large : ℝ)
  (area_ratio : ℝ)
  (h_large : ℝ) :
  A_small / A_large = 1 / 9 →
  h_small = 5 →
  area_ratio = A_large / A_small →
  area_ratio = 9 →
  h_large = h_small * sqrt area_ratio →
  h_large = 15 :=
by
  sorry

end similar_triangles_height_l580_580040


namespace log_base_change_l580_580641

theorem log_base_change (a b : ℝ) (h1 : log 10 2 = a) (h2 : log 10 3 = b) :
  log 6 18 = (a + 2 * b) / (a + b) :=
sorry

end log_base_change_l580_580641


namespace starting_number_l580_580412

theorem starting_number (n : ℤ) : 
  (∃ n, (200 - n) / 3 = 33 ∧ (200 % 3 ≠ 0) ∧ (n % 3 = 0 ∧ n ≤ 200)) → n = 102 :=
by
  sorry

end starting_number_l580_580412


namespace solve_complex_eq_l580_580998

theorem solve_complex_eq (z : ℂ) (h : z = complex.I * (2 - z)) : z = 1 + complex.I := by
  sorry

end solve_complex_eq_l580_580998


namespace searchlight_revolutions_l580_580490

theorem searchlight_revolutions
  (prob_dark : ℝ)
  (D : ℝ)
  (seconds_per_minute : ℝ)
  (prob_dark_eq_half : prob_dark = 0.5)
  (dark_period_eq_10 : D = 10)
  (seconds_per_minute_eq_60 : seconds_per_minute = 60) :
  ∃ r : ℝ, r = 3 :=
by
  -- Assumptions and given conditions
  have L_eq_D : L = D, from sorry -- because probability of dark and light are both 0.5
  have L_plus_D_eq_20 : L + D = 20, from sorry
  
  -- Calculation for number of revolutions per minute
  have time_per_revolution : 60 / r = 20, from calc
    60 / r = 60 / r : by rfl
    ... = 20        : by sorry
    
  existsi 3
  -- Confirm r = 3 from our equation by solving
  have r_eq_3 : r = 3, from calc
    r = 60 / 20 : by sorry
    ... = 3     : by sorry
  
  exact r_eq_3

end searchlight_revolutions_l580_580490


namespace geometric_sequence_property_l580_580608

theorem geometric_sequence_property (a_n : ℕ → ℚ) (q : ℚ) (h1 : a_n 1 = 1) (h3 : a_n 3 = 2) (h_geometric : ∀ n : ℕ, a_n (n+1) = q * a_n n ) :
  (a_n 5 + a_n 10) / (a_n 1 + a_n 6) = 4 := 
begin
  sorry
end

end geometric_sequence_property_l580_580608


namespace range_of_a_l580_580924

variable (a x y : ℝ)

theorem range_of_a (h1 : 2 * x + y = 1 + 4 * a) (h2 : x + 2 * y = 2 - a) (h3 : x + y > 0) : a > -1 :=
sorry

end range_of_a_l580_580924


namespace number_of_six_digit_numbers_not_adjacent_l580_580168

theorem number_of_six_digit_numbers_not_adjacent :
  let digits := [1, 2, 3, 4, 5, 6]
  in (∀ seq : List ℕ, set.to_finset (list.map set.to_finset seq) = set.to_finset (list.map set.to_finset digits) 
  ∧ (∀ i j, (1 = seq.nth i ∧ 3 = seq.nth j) → abs (i - j) > 1)) →
  (number_of_six_digit_numbers_not_adjacent digits = 192) :=
sorry

end number_of_six_digit_numbers_not_adjacent_l580_580168


namespace minimum_triples_to_evaluate_l580_580714

def f : ℤ × ℤ × ℤ → ℝ := sorry

lemma recurrence_relation1 (a b c : ℤ) : 
  f (a, b, c) = (f (a+1, b, c) + f (a-1, b, c)) / 2 := sorry

lemma recurrence_relation2 (a b c : ℤ) : 
  f (a, b, c) = (f (a, b+1, c) + f (a, b-1, c)) / 2 := sorry

lemma recurrence_relation3 (a b c : ℤ) : 
  f (a, b, c) = (f (a, b, c+1) + f (a, b, c-1)) / 2 := sorry

theorem minimum_triples_to_evaluate (f : ℤ × ℤ × ℤ → ℝ)
  (h1 : ∀ a b c, f (a, b, c) = (f (a+1, b, c) + f (a-1, b, c)) / 2)
  (h2 : ∀ a b c, f (a, b, c) = (f (a, b+1, c) + f (a, b-1, c)) / 2)
  (h3 : ∀ a b c, f (a, b, c) = (f (a, b, c+1) + f (a, b, c-1)) / 2) : 
  ∃ (s : set (ℤ × ℤ × ℤ)), s.card = 8 ∧ ∀ a b c, ∃ (x ∈ s), f (a, b, c) = f x := sorry

end minimum_triples_to_evaluate_l580_580714


namespace _l580_580747

open Real

noncomputable def f (x : ℝ) : ℝ := x - sin x

lemma f_non_decreasing_in_0_to_pi_div_2 {a b : ℝ} (ha : a ∈ Icc (0 : ℝ) (π / 2)) (hb : b ∈ Icc (0 : ℝ) (π / 2)) (h : a < b) :
  f a < f b := by
  sorry

lemma f_increasing_in_pi_to_3pi_div_2 {a b : ℝ} (ha : a ∈ Icc (π : ℝ) (3 * π / 2)) (hb : b ∈ Icc (π : ℝ) (3 * π / 2)) (h : a < b) :
  f a < f b := by
  sorry

lemma main_theorem {a b : ℝ} (h1 : a < b)
  (h2 : (a ∈ Icc (0 : ℝ) (π / 2) ∧ b ∈ Icc (0 : ℝ) (π / 2)) ∨
        (a ∈ Icc (π : ℝ) (3 * π / 2) ∧ b ∈ Icc (π : ℝ) (3 * π / 2))) :
  f a < f b := by
  cases h2 with h2_1 h2_2
  case inl =>
    exact f_non_decreasing_in_0_to_pi_div_2 h2_1.left h2_1.right h1
  case inr =>
    exact f_increasing_in_pi_to_3pi_div_2 h2_2.left h2_2.right h1

-- main_theorem is the core statement derived from the problem

end _l580_580747


namespace value_of_x_plus_y_l580_580990

-- Definitions of the conditions
def is_real_domain (x : ℝ) : Prop :=
  3 - 2 * x ≥ 0 ∧ 2 * x - 3 ≥ 0

-- Problem statement
theorem value_of_x_plus_y (x y : ℝ) (h1 : y = sqrt (3 - 2 * x) + sqrt (2 * x - 3)) (h2 : is_real_domain x) :
  x + y = 3 / 2 :=
sorry

end value_of_x_plus_y_l580_580990


namespace count_squares_in_6x5_grid_l580_580188

-- We formalize the given grid dimensions and the nature of squares in a grid
def grid : Type := nat × nat

def count_squares (m n : ℕ) : ℕ :=
let num_1x1 := m * n in
let num_2x2 := (m - 1) * (n - 1) in
let num_3x3 := (m - 2) * (n - 2) in
let num_4x4 := (m - 3) * (n - 3) in
num_1x1 + num_2x2 + num_3x3 + num_4x4

-- The claim to prove
theorem count_squares_in_6x5_grid :
  count_squares 6 5 = 68 :=
by
  -- We would provide the proof here, but it is omitted as per instructions
  sorry

end count_squares_in_6x5_grid_l580_580188


namespace find_certain_number_l580_580271

theorem find_certain_number (x : ℕ) (h1 : 172 = 4 * 43) (h2 : 43 - 172 / x = 28) (h3 : 172 % x = 7) : x = 11 := by
  sorry

end find_certain_number_l580_580271


namespace lowest_test_score_dropped_l580_580315

theorem lowest_test_score_dropped (A B C D : ℝ) 
  (h1: A + B + C + D = 280)
  (h2: A + B + C = 225) : D = 55 := 
by 
  sorry

end lowest_test_score_dropped_l580_580315


namespace path_area_l580_580850

theorem path_area 
  (field_length : ℝ) (field_width : ℝ) (path_width : ℝ)
  (path_cost_per_sqm : ℝ) (total_construction_cost : ℝ)
  (field_length = 75) (field_width = 55)
  (path_width = 2.5) (path_cost_per_sqm = 2) (total_construction_cost = 1350) :
  let total_length := field_length + 2 * path_width in
  let total_width := field_width + 2 * path_width in
  let total_area := total_length * total_width in
  let field_area := field_length * field_width in
  let path_area := total_area - field_area in
  path_area = 675 :=
by 
  sorry

end path_area_l580_580850


namespace carlton_outfits_l580_580533

theorem carlton_outfits (button_up_shirts sweater_vests : ℕ) 
  (h1 : sweater_vests = 2 * button_up_shirts)
  (h2 : button_up_shirts = 3) :
  sweater_vests * button_up_shirts = 18 :=
by
  sorry

end carlton_outfits_l580_580533


namespace count_integers_x_l580_580569

theorem count_integers_x (count_x : ℕ) :
  count_x = {
    | x : ℤ | x^4 - 51 * x^2 + 50 < 0
  }.toFinset.card = 12 :=
sorry

end count_integers_x_l580_580569


namespace train_stoppage_time_l580_580900

-- Definitions of the conditions
def speed_excluding_stoppages : ℝ := 48 -- in kmph
def speed_including_stoppages : ℝ := 32 -- in kmph
def time_per_hour : ℝ := 60 -- 60 minutes in an hour

-- The problem statement
theorem train_stoppage_time :
  (speed_excluding_stoppages - speed_including_stoppages) * time_per_hour / speed_excluding_stoppages = 20 :=
by
  -- Initial statement
  sorry

end train_stoppage_time_l580_580900


namespace high_jump_mode_median_l580_580513

def heights : list ℝ := [1.50, 1.50, 1.60, 1.60, 1.60, 1.65, 1.65, 1.65, 1.65, 1.65, 1.70, 1.70, 1.70, 1.70, 1.75]

def mode (l : list ℝ) : ℝ :=
  l.mode -- This assumes the mode function is defined elsewhere to return the most frequent element.

def median (l : list ℝ) : ℝ :=
  l.median -- This assumes the median function is defined elsewhere to return the middle value.

theorem high_jump_mode_median :
  mode heights = 1.65 ∧ median heights = 1.65 :=
by sorry

end high_jump_mode_median_l580_580513


namespace one_inch_model_represents_15_feet_l580_580768

def modelHeight : ℝ := 6 / 12  -- converting inches to feet
def statueHeight : ℝ := 90

theorem one_inch_model_represents_15_feet :
  1 * (statueHeight / modelHeight) = 15 :=
by
  rw [modelHeight]
  norm_num
  
sorry

end one_inch_model_represents_15_feet_l580_580768


namespace share_of_B_l580_580442

theorem share_of_B (x : ℝ) (profit : ℝ) (h_profit : profit = 4400)
  (h_B_invest : x > 0 ∧ (2/3 : ℝ) * x > 0) :
  let A_invest := 2 * x,
      B_invest := (2/3 : ℝ) * x,
      C_invest := x,
      total_invest := A_invest + B_invest + C_invest,
      B_share := (B_invest / total_invest) * profit
  in B_share = 1760 := 
by
  sorry

end share_of_B_l580_580442


namespace profit_distribution_l580_580846

/--
A and B are partners in a business. A contributes Rs. 5000 and B contributes Rs. 1000 to the business.
A receives 10% of the profit for managing the business, and the remaining profit is divided in proportion
to their capitals. The total profit is Rs. 9600. Prove that the total money received by A is Rs. 8160.
-/
theorem profit_distribution (profit : ℝ) (A_contribution B_contribution : ℝ) (A_management_percentage : ℝ) :
  profit = 9600 → A_contribution = 5000 → B_contribution = 1000 → A_management_percentage = 0.10 →
  let A_management_share := A_management_percentage * profit in
  let remaining_profit := profit - A_management_share in
  let total_capital := A_contribution + B_contribution in
  let A_ratio := A_contribution / total_capital in
  let B_ratio := B_contribution / total_capital in
  let remaining_profit_A_share := A_ratio * remaining_profit in
  A_management_share + remaining_profit_A_share = 8160 := by
  sorry

end profit_distribution_l580_580846


namespace eric_has_9306_erasers_l580_580895

-- Define the conditions as constants
def number_of_friends := 99
def erasers_per_friend := 94

-- Define the total number of erasers based on the conditions
def total_erasers := number_of_friends * erasers_per_friend

-- Theorem stating the total number of erasers Eric has
theorem eric_has_9306_erasers : total_erasers = 9306 := by
  -- Proof to be filled in
  sorry

end eric_has_9306_erasers_l580_580895


namespace sum_abs_of_roots_l580_580570

variables {p q r : ℤ}

theorem sum_abs_of_roots:
  p + q + r = 0 →
  p * q + q * r + r * p = -2023 →
  |p| + |q| + |r| = 94 := by
  intro h1 h2
  sorry

end sum_abs_of_roots_l580_580570


namespace similar_triangles_height_l580_580076

theorem similar_triangles_height
  (a b : ℕ)
  (area_ratio: ℕ)
  (height_smaller : ℕ)
  (height_relation: height_smaller = 5)
  (area_relation: area_ratio = 9)
  (similarity: a / b = 1 / area_ratio):
  (∃ height_larger : ℕ, height_larger = 15) :=
by
  sorry

end similar_triangles_height_l580_580076


namespace incorrect_conjugate_statement_l580_580964

-- Define the complex number z using the given condition.
def z : ℂ := 16 * Complex.I / (Real.sqrt 7 + 3 * Complex.I)

-- State the mathematically equivalent proof problem: z's conjugate is not -3 + sqrt(7)i
theorem incorrect_conjugate_statement : Complex.conj z ≠ -3 + Real.sqrt 7 * Complex.I :=
by
  sorry

end incorrect_conjugate_statement_l580_580964


namespace angle_C_length_CD_area_range_l580_580678

-- 1. Prove C = π / 3 given (2a - b)cos C = c cos B
theorem angle_C (a b c : ℝ) (A B C : ℝ) (h : (2 * a - b) * Real.cos C = c * Real.cos B) : 
  C = Real.pi / 3 := sorry

-- 2. Prove the length of CD is 6√3 / 5 given a = 2, b = 3, and CD is the angle bisector of angle C
theorem length_CD (a b x : ℝ) (C D : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : x = (6 * Real.sqrt 3) / 5) : 
  x = (6 * Real.sqrt 3) / 5 := sorry

-- 3. Prove the range of values for the area of acute triangle ABC is (8√3 / 3, 4√3] given a cos B + b cos A = 4
theorem area_range (a b : ℝ) (A B C : ℝ) (S : Set ℝ) (h1 : a * Real.cos B + b * Real.cos A = 4) 
  (h2 : S = Set.Ioc (8 * Real.sqrt 3 / 3) (4 * Real.sqrt 3)) : 
  S = Set.Ioc (8 * Real.sqrt 3 / 3) (4 * Real.sqrt 3) := sorry

end angle_C_length_CD_area_range_l580_580678
