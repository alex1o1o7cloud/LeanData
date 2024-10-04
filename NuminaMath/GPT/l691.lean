import Mathlib
import Mathlib.Algebra.ArithSeq
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.EuclideanDomain
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Combinations
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Perm.Basic
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Trigonometric
import Mathlib.Data.Set
import Mathlib.Data.Set.Basic
import Mathlib.Data.Zmod.Basic
import Mathlib.Geometry.Euclidean.ThreeDim
import Mathlib.Init.Algebra
import Mathlib.LinearAlgebra.AffineSpace
import Mathlib.LinearAlgebra.Basic
import Mathlib.MeasureTheory.Measure.Lebesgue
import Mathlib.NumberTheory.LegendreSymbol.QuadraticReciprocity
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic
import Mathlib.Topology.Sequences.Basic
import Set

namespace cost_price_of_article_l691_691640

theorem cost_price_of_article :
  ∃ (CP : ℝ), (616 = 1.10 * (1.17 * CP)) → CP = 478.77 :=
by
  sorry

end cost_price_of_article_l691_691640


namespace half_abs_diff_squares_l691_691278

theorem half_abs_diff_squares (a b : ℤ) (h₁ : a = 21) (h₂ : b = 17) :
  (|a^2 - b^2| / 2) = 76 :=
by 
  sorry

end half_abs_diff_squares_l691_691278


namespace nth_inequality_l691_691774

theorem nth_inequality (n : ℕ) (H1 : 1 + 1/2 + 1/3 > 1)
                      (H2 : 1 + 1/2 + 1/3 + 1/4 + 1/5 + 1/6 + 1/7 > 3/2)
                      (H3 : 1 + 1/2 + 1/3 + 1/4 + 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 > 2) :
  1 + (∑ k in finset.range (2^(n+1)-1), 1/(k+1)) > (n + 1) / 2 := 
begin
  sorry
end

end nth_inequality_l691_691774


namespace james_fraction_of_pizza_slices_l691_691108

theorem james_fraction_of_pizza_slices :
  (2 * 6 = 12) ∧ (8 / 12 = 2 / 3) :=
by
  sorry

end james_fraction_of_pizza_slices_l691_691108


namespace PQRS_result_l691_691021

variable (sqrt2010 sqrt2011 : ℝ)
variable (P Q R S : ℝ)

def P := 3 * sqrt2010 + 2 * sqrt2011
def Q := -3 * sqrt2010 - 2 * sqrt2011
def R := 3 * sqrt2010 - 2 * sqrt2011
def S := 2 * sqrt2011 - 3 * sqrt2010

theorem PQRS_result : P * Q * R * S = -40434584 := by
  sorry

end PQRS_result_l691_691021


namespace find_original_number_l691_691325

theorem find_original_number (x : ℕ) (h : 3 * (2 * x + 9) = 57) : x = 5 := 
by sorry

end find_original_number_l691_691325


namespace min_contribution_l691_691814

theorem min_contribution (x : ℝ) (h1 : 0 < x) (h2 : 10 * x = 20) (h3 : ∀ p, p ≠ 1 → p ≠ 2 → p ≠ 3 → p ≠ 4 → p ≠ 5 → p ≠ 6 → p ≠ 7 → p ≠ 8 → p ≠ 9 → p ≠ 10 → p ≤ 11) : 
  x = 2 := sorry

end min_contribution_l691_691814


namespace probability_same_problem_l691_691505

theorem probability_same_problem (p : Fin 3) : 
  let outcomes := List.product [1, 2, 3] [1, 2, 3] 
  in let favorable_outcomes := outcomes.filter (λ pair, pair.1 = pair.2)
  in (favorable_outcomes.length : ℚ) / (outcomes.length : ℚ) = 1 / 3 :=
by
  sorry

end probability_same_problem_l691_691505


namespace fubini_failure_without_sigma_finite_l691_691899

open MeasureTheory

noncomputable def leb : Measure ℝ := volume.restrict (Icc 0 1)
noncomputable def count : Measure ℝ := counting.restrict (Icc 0 1)
noncomputable def diag (x y : ℝ) := x = y

theorem fubini_failure_without_sigma_finite :
  (∫ y in 0..1, ∫ x in 0..1, indicator (λ p, diag p.1 p.2) (x, y) ∂leb ∂count) = 0 ∧
  (∫ x in 0..1, ∫ y in 0..1, indicator (λ p, diag p.1 p.2) (x, y) ∂count ∂leb) = 1 :=
by 
  sorry

end fubini_failure_without_sigma_finite_l691_691899


namespace mad_hatter_must_secure_at_least_70_percent_l691_691835

theorem mad_hatter_must_secure_at_least_70_percent :
  ∀ (N : ℕ) (uM uH uD : ℝ) (α : ℝ),
    uM = 0.2 ∧ uH = 0.25 ∧ uD = 0.3 → 
    uM + α * 0.25 ≥ 0.25 + (1 - α) * 0.25 ∧
    uM + α * 0.25 ≥ 0.3 + (1 - α) * 0.25 →
    α ≥ 0.7 :=
by
  intros N uM uH uD α h hx
  sorry 

end mad_hatter_must_secure_at_least_70_percent_l691_691835


namespace triangle_MOI_area_zero_l691_691083

noncomputable def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (6, 0)
def C : point := (0, 8)
def O : point := (3, 4)
def I : point := (2, 2)
def M : point := (1, 1)

def area_ΔMOI (M O I : point) : ℝ :=
1 / 2 * |(M.fst * (O.snd - I.snd) + O.fst * (I.snd - M.snd) + I.fst * (M.snd - O.snd))|

theorem triangle_MOI_area_zero
  (A B C O I M : point)
  (h1 : A = (0, 0))
  (h2 : B = (6, 0))
  (h3 : C = (0, 8))
  (h4 : O = (3, 4))
  (h5 : I = (2, 2))
  (h6 : M = (1, 1)) :
  area_ΔMOI M O I = 0 :=
sorry

end triangle_MOI_area_zero_l691_691083


namespace sum_integers_30_to_50_l691_691990

theorem sum_integers_30_to_50 : 
  (∑ i in Finset.range 51, if 30 ≤ i ∧ i ≤ 50 then i else 0) = 840 :=
by sorry

end sum_integers_30_to_50_l691_691990


namespace tan_pi_over_4_plus_alpha_eq_two_l691_691446

theorem tan_pi_over_4_plus_alpha_eq_two
  (α : ℂ) 
  (h : Complex.tan ((π / 4) + α) = 2) : 
  (1 / (2 * Complex.sin α * Complex.cos α + (Complex.cos α)^2)) = (2 / 3) :=
by
  sorry

end tan_pi_over_4_plus_alpha_eq_two_l691_691446


namespace solve_inequality_l691_691174

theorem solve_inequality (a : ℝ) :
  (a < 1 / 2 ∧ ∀ x : ℝ, x^2 - x + a - a^2 < 0 ↔ a < x ∧ x < 1 - a) ∨
  (a > 1 / 2 ∧ ∀ x : ℝ, x^2 - x + a - a^2 < 0 ↔ 1 - a < x ∧ x < a) ∨
  (a = 1 / 2 ∧ ∀ x : ℝ, x^2 - x + a - a^2 < 0 ↔ false) :=
sorry

end solve_inequality_l691_691174


namespace medians_of_right_triangle_l691_691091

-- Define the input problem
variables {α : Real} {c : Real}

-- Define the conditions of the problem
def is_right_triangle (A B C : Point) : Prop :=
  angle B = α ∧ dist A B = c

-- Define the correct answers as the definitions of the medians
def median_CK (A B C : Point) [is_right_triangle A B C] : Real :=
  c / 2

def median_AM (A B C : Point) [is_right_triangle A B C] : Real :=
  c * sqrt (cos α ^ 2 + (1 / 4) * sin α ^ 2)

-- Theorem statement for the proof problem (Lean 4 statement)
theorem medians_of_right_triangle (A B C : Point) 
  [is_right_triangle A B C] :
  (median_CK A B C = c / 2) ∧ 
  (median_AM A B C = c * sqrt (cos α ^ 2 + (1 / 4) * sin α ^ 2)) := 
sorry

end medians_of_right_triangle_l691_691091


namespace rectangle_perimeter_l691_691184

theorem rectangle_perimeter :
  ∃ (a1 a2 a3 a4 a5 a6 a7 a8 a9 l w: ℕ),
    l * w = (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9) ∧
    Nat.Coprime l w ∧
    a1 + a2 = a3 ∧
    a1 + a3 = a4 ∧
    a3 + a4 = a5 ∧
    a4 + a5 = a6 ∧
    a2 + a3 + a5 = a7 ∧
    a2 + a7 = a8 ∧
    a1 + a4 + a6 = a9 ∧
    a6 + a9 = a7 + a8 ∧
    l = 61 ∧ w = 69 ∧
    2 * l + 2 * w = 260 :=
by
  let a1 := 2
  let a2 := 5
  let a3 := 7
  let a4 := 9
  let a5 := 16
  let a6 := 25
  let a7 := 23
  let a8 := 33
  let a9 := 36
  let l := 61
  let w := 69
  have coprime : Nat.Coprime l w := sorry -- Justification needed for coprimeness
  exists a1, a2, a3, a4, a5, a6, a7, a8, a9, l, w
  simp
  have rect_area : l * w = (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9) := sorry -- This condition simplifying here
  exact ⟨
    rect_area, coprime,
    rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl,
    rfl, rfl
  ⟩  

end rectangle_perimeter_l691_691184


namespace find_t_l691_691062

theorem find_t (t : ℝ) (h : sqrt (3 * sqrt (t - 3)) = (10 - t)^(1/4)) : t = 3.7 :=
sorry

end find_t_l691_691062


namespace tan_theta_neg_sqrt_3_l691_691192

theorem tan_theta_neg_sqrt_3 (θ : ℝ) :
  (∀ x : ℝ, (sqrt 3 * cos (3 * x - θ) - sin (3 * x - θ)) = -(sqrt 3 * cos (-3 * x - θ) + sin (-3 * x - θ))) →
  tan θ = -sqrt 3 :=
sorry

end tan_theta_neg_sqrt_3_l691_691192


namespace find_angle_B_l691_691450

noncomputable def angle_B (a b c A : ℝ) : ℝ :=
  if a = sqrt 3 ∧ b = 1 ∧ A = 60 then 30 else 0

theorem find_angle_B : angle_B (sqrt 3) 1 c 60 = 30 := by
  unfold angle_B
  rw [if_pos]
  apply And.intro
  case left => rfl
  case right =>
    apply And.intro
    case left => rfl
    case right => rfl
  sorry

end find_angle_B_l691_691450


namespace minimum_value_of_y_l691_691487

theorem minimum_value_of_y (x : ℝ) (h : x > 2) : 
  ∃ y, y = x + 4 / (x - 2) ∧ y = 6 :=
by
  sorry

end minimum_value_of_y_l691_691487


namespace solution_set_l691_691124

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry

theorem solution_set (h_diff : ∀ x ∈ Ioi 0, differentiable_at ℝ f x)
(h_ineq : ∀ x > 0, 2 * x * f x + x^2 * f' x > 0) :
  {x : ℝ | (x - 2014)^2 * f (x - 2014) - 4 * f 2 > 0} = Ioi 2016 :=
by
  sorry

end solution_set_l691_691124


namespace find_angle_BDC_l691_691841

noncomputable def angleSum (A B C : ℝ) : Prop := A + B + C = 180
noncomputable def incenter (A B C D : Point) : Prop := 
  tangent_to_circle Ctr D (Triangle A B C)

theorem find_angle_BDC
  (A B C D : Point)
  (BAC ABC BCA: ℝ)
  (is_incenter : incenter A B C D)
  (angle_BAC : BAC = 70)
  (angle_ABC : ABC = 52)
  (angle_sum : angleSum BAC ABC BCA) :
  ∃ BDC : ℝ, BDC = 29 := 
sorry

end find_angle_BDC_l691_691841


namespace compute_c_l691_691859

noncomputable def f : ℕ → (ℝ → ℝ)
| 1 => λ x, real.sqrt (2 - x)
| (n+1) => λ x, f n (real.sqrt ((n + 2)^2 - x))

def domain_nonempty (f : ℝ → ℝ) (n : ℕ) : Prop :=
  ∃ x : ℝ, f x = f (n+1)

theorem compute_c :
  let N := 4
  ∃ c : ℝ, ∀ x : ℝ, (f N x = f N c) :=
sorry

end compute_c_l691_691859


namespace remainder_sum_first_150_div_10000_l691_691951
open Nat

theorem remainder_sum_first_150_div_10000 :
  ((∑ i in Finset.range 151, i) % 10000) = 1325 :=
sorry

end remainder_sum_first_150_div_10000_l691_691951


namespace half_abs_diff_of_squares_l691_691252

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 20) (h2 : b = 15) :
  (1/2 : ℚ) * |(a:ℚ)^2 - (b:ℚ)^2| = 87.5 := by
  sorry

end half_abs_diff_of_squares_l691_691252


namespace find_S_2016_l691_691439

-- (Assume definitions and necessary properties of geometric sequences are available in the imported libraries)

-- Given: Sum of the first n terms of a geometric sequence, S_n
variables {α : Type*} [LinearOrderedField α]
variables {a r : α} (q : α)

-- Specific conditions given in the problem
def S_n : ℕ → α := λ n, a * (1 - q^n) / (1 - q)

axiom S_8 : S_n 8 = 2
axiom S_24 : S_n 24 = 14

-- The theorem to prove
theorem find_S_2016 : S_n 2016 = 2^253 - 2 :=
by sorry

end find_S_2016_l691_691439


namespace regular_tiles_cover_146_67_sq_ft_l691_691995

noncomputable def regular_tiles_area (total_area : ℝ) (tiles_ratio_jumbo : ℝ) : ℝ :=
  (2 / 3) * total_area

theorem regular_tiles_cover_146_67_sq_ft (total_area : ℝ) (tiles_ratio_jumbo : ℝ) 
  (length_ratio_jumbo : ℝ) (A_reg : ℝ) :
  total_area = 220 →
  tiles_ratio_jumbo = 1 / 3 →
  length_ratio_jumbo = 3 →
  A_reg = regular_tiles_area total_area tiles_ratio_jumbo →
  A_reg = 146.67 :=
  by
  intros ht_area ht_ratio_j ht_length_ratio hA_reg
  rw [ht_area, ht_ratio_j]
  exact hA_reg

end regular_tiles_cover_146_67_sq_ft_l691_691995


namespace wire_cut_l691_691683

theorem wire_cut (x : ℝ) :
  (let first_piece := x + 2 in
   let third_piece := 2 * x - 3 in
   x + first_piece + third_piece = 50 → first_piece = 14.75 ∧ x = 12.75 ∧ third_piece = 22.5) :=
by
  intros h
  -- sorry

end wire_cut_l691_691683


namespace area_ratio_triangle_segments_l691_691103

theorem area_ratio_triangle_segments 
  (A B C D E F G H M N P Q: Point)
  (AD DF FM MP PB : ℝ)
  (h1 : AD = DF)
  (h2 : DF = FM)
  (h3 : FM = MP)
  (h4 : MP = PB)
  (h5 : Parallel DE FG)
  (h6 : Parallel FG MN)
  (h7 : Parallel MN PQ)
  (h8 : Parallel PQ BC)
  (h9 : Triangle ABC)
  (h10 : ∀ X, X ∈ {D, F, M, P} ⊆ Segment A B) :
  (area_ratio (triangle A D E) : area_ratio (Quadrilateral D E G F) : area_ratio (Quadrilateral F G N M) : area_ratio (Quadrilateral M N Q P) : area_ratio (Quadrilateral P Q C B) = 1 : 3 : 5 : 7 : 9) :=
sorry

end area_ratio_triangle_segments_l691_691103


namespace sum_of_first_10_terms_l691_691436

def max (a b : ℝ) : ℝ := if a < b then b else a

def a_n (n : ℕ) : ℝ :=
  max (n^2) (2^n)

theorem sum_of_first_10_terms :
  (finset.range 10).sum a_n = 2046 :=
by
  sorry

end sum_of_first_10_terms_l691_691436


namespace xy_product_l691_691943

theorem xy_product (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 :=
sorry

end xy_product_l691_691943


namespace upper_bound_y_l691_691489

/-- 
  Theorem:
  For any real numbers x and y such that 3 < x < 6 and 6 < y, 
  if the greatest possible positive integer difference between x and y is 6,
  then the upper bound for y is 11.
 -/
theorem upper_bound_y (x y : ℝ) (h₁ : 3 < x) (h₂ : x < 6) (h₃ : 6 < y) (h₄ : y < some_number) (h₅ : y - x = 6) : y = 11 := 
by
  sorry

end upper_bound_y_l691_691489


namespace proof_problem_l691_691098

-- Define the curves in Cartesian coordinates
def C2 (x y : ℝ) : Prop := x^2 + y^2 = 2 * y
def C3 (x y : ℝ) : Prop := x^2 + y^2 = 2 * sqrt 3 * x

-- Define the equation for the curve C1
def C1 (x y α : ℝ) (t : ℝ) : Prop :=
  x = t * cos α ∧ y = t * sin α

-- Define the maximum value of |AB|
noncomputable def max_AB (α : ℝ) : ℝ :=
  4 * abs (sin (α - π / 3))

-- The theorem statement
theorem proof_problem (α : ℝ) (t : ℝ) (x y : ℝ) (hα : 0 ≤ α ∧ α < π) :
  ((C2 x y ∧ C3 x y) → ((x = 0 ∧ y = 0) ∨ (x = sqrt 3 / 2 ∧ y = 3 / 2))) ∧
  (∀ A B : ℝ × ℝ, (C2 A.1 A.2 ∧ C1 A.1 A.2 α t) ∧ (C3 B.1 B.2 ∧ C1 B.1 B.2 α t) →
    max_AB α = 4) :=
by sorry

end proof_problem_l691_691098


namespace area_expression_ABC_is_correct_l691_691515

-- Define the properties of the quadrilateral ABCD
def quadrilateral_ABCD (A B C D : ℝ × ℝ) (AB CD DA : ℝ) (angle_CDA : ℝ) :=
  dist A B = AB ∧
  dist B C = BC ∧
  dist C D = CD ∧
  dist D A = DA ∧
  angle C D A = angle_CDA

-- Define the problem context
theorem area_expression_ABC_is_correct :
  ∀ (A B C D : ℝ × ℝ),
  quadrilateral_ABCD A B C D 8 10 10 (real.pi / 3) →
  let area := sqrt 231 + 25 * sqrt 3 in
  let (a, b, c) := (231, 25, 3) in
  a + b + c = 259 :=
by
  intros A B C D h
  -- Place area calculation and result verification here if proving
  sorry

end area_expression_ABC_is_correct_l691_691515


namespace dave_guitar_strings_l691_691704

theorem dave_guitar_strings (strings_per_night : ℕ) (shows_per_week : ℕ) (weeks : ℕ)
  (h1 : strings_per_night = 4)
  (h2 : shows_per_week = 6)
  (h3 : weeks = 24) : 
  strings_per_night * shows_per_week * weeks = 576 :=
by
  sorry

end dave_guitar_strings_l691_691704


namespace mary_spent_on_jacket_l691_691144

def shirt_cost : ℝ := 13.04
def total_cost : ℝ := 25.31
def jacket_cost : ℝ := total_cost - shirt_cost

theorem mary_spent_on_jacket :
  jacket_cost = 12.27 := by
  sorry

end mary_spent_on_jacket_l691_691144


namespace find_range_of_m_l691_691413

variable {x m : ℝ}

theorem find_range_of_m (p : -2 ≤ 1 - (x - 1) / 3 ∧ 1 - (x - 1) / 3 ≤ 2)
  (q : (x + m - 1) * (x - m - 1) ≤ 0) (hm : 0 < m)
  (necessary_but_not_sufficient : ∀ x, q → p) :
  m ≥ 9 :=
sorry

end find_range_of_m_l691_691413


namespace line_through_two_points_l691_691193

-- Define the points (2,5) and (0,3)
structure Point where
  x : ℝ
  y : ℝ

def P1 : Point := {x := 2, y := 5}
def P2 : Point := {x := 0, y := 3}

-- General form of a line equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the target line equation as x - y + 3 = 0
def targetLine : Line := {a := 1, b := -1, c := 3}

-- The proof statement to show that the general equation of the line passing through the points (2, 5) and (0, 3) is x - y + 3 = 0
theorem line_through_two_points : ∃ a b c, ∀ x y : ℝ, 
    (a * x + b * y + c = 0) ↔ 
    ((∀ {P : Point}, P = P1 → targetLine.a * P.x + targetLine.b * P.y + targetLine.c = 0) ∧ 
     (∀ {P : Point}, P = P2 → targetLine.a * P.x + targetLine.b * P.y + targetLine.c = 0)) :=
sorry

end line_through_two_points_l691_691193


namespace train_crossing_time_l691_691297

-- Definitions from conditions
def length_of_train : ℕ := 120
def length_of_bridge : ℕ := 150
def speed_kmph : ℕ := 36
def speed_mps : ℕ := speed_kmph * 1000 / 3600 -- Convert km/h to m/s
def total_distance : ℕ := length_of_train + length_of_bridge

-- Theorem statement
theorem train_crossing_time : total_distance / speed_mps = 27 := by
  sorry

end train_crossing_time_l691_691297


namespace maria_sold_in_first_hour_l691_691142

variable (x : ℕ)

-- Conditions
def sold_in_first_hour := x
def sold_in_second_hour := 2
def average_sold_in_two_hours := 6

-- Proof Goal
theorem maria_sold_in_first_hour :
  (sold_in_first_hour + sold_in_second_hour) / 2 = average_sold_in_two_hours → sold_in_first_hour = 10 :=
by
  sorry

end maria_sold_in_first_hour_l691_691142


namespace tan_alpha_eq_neg_one_l691_691455

-- Define the point P and the angle α
def P : ℝ × ℝ := (-1, 1)
def α : ℝ := sorry  -- α is the angle whose terminal side passes through P

-- Statement to be proved
theorem tan_alpha_eq_neg_one (h : (P.1, P.2) = (-1, 1)) : Real.tan α = -1 :=
by
  sorry

end tan_alpha_eq_neg_one_l691_691455


namespace half_absolute_difference_of_squares_l691_691264

-- Defining the variables a and b involved in the problem
def a : ℤ := 20
def b : ℤ := 15

-- Statement to prove the solution
theorem half_absolute_difference_of_squares : 
    (1 / 2 : ℚ) * (abs ((a ^ 2) - (b ^ 2))) = 87.5 :=
by
    -- Proof omitted
    sorry

end half_absolute_difference_of_squares_l691_691264


namespace smallest_n_satisfying_condition_l691_691013

-- Definitions
variables {a : ℕ → ℝ} {S : ℕ → ℝ} (n : ℕ)

-- Given conditions
def cond1 : Prop := S 15 > 0
def cond2 : Prop := a 8 + a 9 < 0

-- Statement to prove
theorem smallest_n_satisfying_condition (h1 : cond1) (h2 : cond2) : 11 = 
  Argmin (λ n, a n + S n / n < 0) sorry


end smallest_n_satisfying_condition_l691_691013


namespace length_BC_over_AD_l691_691155

theorem length_BC_over_AD {A B C D : Point} 
  (h1 : collinear A B D) (h2 : collinear A C D) 
  (h3 : dist A B = 3 * dist B D) 
  (h4 : dist A C = 7 * dist C D) 
  : (dist B C) / (dist A D) = 1 / 8 := by
sorry

end length_BC_over_AD_l691_691155


namespace width_of_plot_is_60_l691_691333

-- Defining the conditions
def length_of_plot := 90
def distance_between_poles := 5
def number_of_poles := 60

-- The theorem statement
theorem width_of_plot_is_60 :
  ∃ width : ℕ, 2 * (length_of_plot + width) = number_of_poles * distance_between_poles ∧ width = 60 :=
sorry

end width_of_plot_is_60_l691_691333


namespace minimum_value_l691_691014

theorem minimum_value (a b : ℝ) (h1 : a > b) (h2 : a * b = 2) : 
  ∃ (m : ℝ), m = 2 * real.sqrt 5 ∧ ∀ (x y : ℝ), x > y → x * y = 2 → (x^2 + y^2 + 1) / (x - y) ≥ m :=
by
  sorry

end minimum_value_l691_691014


namespace systematic_sampling_first_segment_l691_691227

theorem systematic_sampling_first_segment:
  ∀ (total_students sample_size segment_size 
     drawn_16th drawn_first : ℕ),
  total_students = 160 →
  sample_size = 20 →
  segment_size = 8 →
  drawn_16th = 125 →
  drawn_16th = drawn_first + segment_size * (16 - 1) →
  drawn_first = 5 :=
by
  intros total_students sample_size segment_size drawn_16th drawn_first
         htots hsamp hseg hdrw16 heq
  sorry

end systematic_sampling_first_segment_l691_691227


namespace hyperbola_asymptote_m_l691_691381

def isAsymptote (x y : ℝ) (m : ℝ) : Prop :=
  y = m * x ∨ y = -m * x

theorem hyperbola_asymptote_m (m : ℝ) : 
  (∀ x y, (x^2 / 25 - y^2 / 16 = 1 → isAsymptote x y m)) ↔ m = 4 / 5 := 
by
  sorry

end hyperbola_asymptote_m_l691_691381


namespace find_cost_price_l691_691965

-- Definitions based on the conditions
def original_cost_price (C : ℝ) : Prop :=
  let S := 1.05 * C in
  let C_new := 0.95 * C in
  let S_new := 1.05 * C - 4 in
  let S_new_10_profit := 1.045 * C in
  S_new = S_new_10_profit

-- The theorem we want to prove
theorem find_cost_price : ∃ C : ℝ, original_cost_price C ∧ C = 800 :=
by
  sorry

end find_cost_price_l691_691965


namespace total_number_of_workers_l691_691909

theorem total_number_of_workers 
  (W : ℕ) 
  (avg_all : ℕ) 
  (n_technicians : ℕ) 
  (avg_technicians : ℕ) 
  (avg_non_technicians : ℕ) :
  avg_all * W = avg_technicians * n_technicians + avg_non_technicians * (W - n_technicians) →
  avg_all = 8000 →
  n_technicians = 7 →
  avg_technicians = 12000 →
  avg_non_technicians = 6000 →
  W = 21 :=
by 
  intro h1 h2 h3 h4 h5
  sorry

end total_number_of_workers_l691_691909


namespace half_abs_diff_squares_l691_691230

theorem half_abs_diff_squares (a b : ℤ) (ha : a = 20) (hb : b = 15) : 
  (|a^2 - b^2| / 2) = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691230


namespace cost_price_correct_l691_691337

noncomputable def cost_price_per_metre := 95

theorem cost_price_correct:
  ∃ cp_per_met,
    (∀ selling_price total_loss, 
      (total_loss = 5 * 200) →
      (selling_price = 18000) →
      (cp_per_met = (selling_price / 200) + (total_loss / 200)) →
      cp_per_met = cost_price_per_metre
    ) :=
begin
  use 95,
  intros,
  sorry
end

end cost_price_correct_l691_691337


namespace sequence_correct_initial_condition_final_form_l691_691008

noncomputable def sequence : ℕ → ℝ
| 0 := 1
| (n + 1) := Real.sqrt (2 * sequence n ^ 4 + 6 * sequence n ^ 2 + 3)

theorem sequence_correct : ∀ n, sequence (n + 1) = Real.sqrt (2 * sequence n ^ 4 + 6 * sequence n ^ 2 + 3) :=
sorry

theorem initial_condition : sequence 0 = 1 :=
sorry

theorem final_form (n : ℕ) : 
  sequence n = Real.sqrt (1/2 * (5 ^ (2^(n-1)) - 3)) :=
sorry

end sequence_correct_initial_condition_final_form_l691_691008


namespace determine_m_range_l691_691745

theorem determine_m_range (m : ℝ) (h : (∃ (x y : ℝ), x^2 + y^2 + 2 * m * x + 2 = 0) ∧ 
                                    (∃ (r : ℝ) (h_r : r^2 = m^2 - 2), π * r^2 ≥ 4 * π)) :
  (m ≤ -Real.sqrt 6 ∨ m ≥ Real.sqrt 6) :=
by
  sorry

end determine_m_range_l691_691745


namespace product_of_a_distinct_integer_roots_l691_691393

theorem product_of_a_distinct_integer_roots :
  (∀ a : ℚ, (∃ (x1 x2 : ℤ), x1 ≠ x2 ∧ (x1^2 + 2 * a * x1 = 8 * a) ∧ (x2^2 + 2 * a * x2 = 8 * a) 
   ∧ x1 + x2 = (-2 * a) ∧ x1 * x2 = (-8 * a)) → 
   (∏ a in {a | ∃ (x1 x2 : ℤ), x1 ≠ x2 ∧ (x1^2 + 2 * a * x1 = 8 * a) ∧ (x2^2 + 2 * a * x2 = 8 * a)}, a)) = 506.25 := 
sorry

end product_of_a_distinct_integer_roots_l691_691393


namespace chocolate_bars_in_large_box_l691_691986

theorem chocolate_bars_in_large_box : 
  let small_boxes := 19 
  let bars_per_small_box := 25 
  let total_bars := small_boxes * bars_per_small_box 
  total_bars = 475 := by 
  -- declarations and assumptions
  let small_boxes : ℕ := 19 
  let bars_per_small_box : ℕ := 25 
  let total_bars : ℕ := small_boxes * bars_per_small_box 
  sorry

end chocolate_bars_in_large_box_l691_691986


namespace ellipse_major_axis_value_l691_691767

theorem ellipse_major_axis_value (m : ℝ) (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ)
  (h1 : ∀ {x y : ℝ}, (x, y) = P → (x^2 / m) + (y^2 / 16) = 1)
  (h2 : dist P F1 = 3)
  (h3 : dist P F2 = 7)
  : m = 25 :=
sorry

end ellipse_major_axis_value_l691_691767


namespace tree_growth_fraction_l691_691626

theorem tree_growth_fraction :
    ∀ (H₀ : ℕ) (increase_yearly : ℕ), H₀ = 4 → increase_yearly = 1 →
    (let H4 := H₀ + 4 * increase_yearly in
     let H6 := H₀ + 6 * increase_yearly in
     (H6 - H4) / H4 = 1 / 4) :=
by
    intros H₀ increase_yearly h₀ h_increase_yearly
    let H4 := H₀ + 4 * increase_yearly
    let H6 := H₀ + 6 * increase_yearly
    have fraction_increase := (H6 - H4) / H4
    have target_fraction := 1 / 4
    have proof := sorry
    exact proof

end tree_growth_fraction_l691_691626


namespace parabola_vertex_parabola_point_condition_l691_691837

-- Define the parabola function 
def parabola (x m : ℝ) : ℝ := x^2 - 2*m*x + m^2 - 1

-- 1. Prove the vertex of the parabola
theorem parabola_vertex (m : ℝ) : ∃ x y, (∀ x m, parabola x m = (x - m)^2 - 1) ∧ (x = m ∧ y = -1) :=
by
  sorry

-- 2. Prove the range of values for m given the conditions on points A and B
theorem parabola_point_condition (m : ℝ) (y1 y2 : ℝ) :
  (y1 > y2) ∧ 
  (parabola (1 - 2*m) m = y1) ∧ 
  (parabola (m + 1) m = y2) → m < 0 ∨ m > 2/3 :=
by
  sorry

end parabola_vertex_parabola_point_condition_l691_691837


namespace sin_S9_l691_691756

-- Define an arithmetic sequence and the sum of its terms
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + m + 1) = a (n + 1) + m * (a 1 + a 1 - a 0)

-- Define the sum of the first n terms of the sequence
def sumOfFirstNTerms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in range (n + 1), a i

-- The main statement
theorem sin_S9 (a : ℕ → ℝ) (S : ℕ → ℝ) (h_arith : isArithmeticSequence a)
  (h_sum : ∀ n, S n = sumOfFirstNTerms a n)
  (h_cond : a 1 + a 4 + a 7 = π / 4) :
  sin (S 8) = sqrt 2 / 2 :=
by sorry

end sin_S9_l691_691756


namespace profit_percentage_is_20_l691_691998

variable (C : ℝ) -- Assuming the cost price C is a real number.

theorem profit_percentage_is_20 
  (h1 : 10 * 1 = 12 * (C / 1)) :  -- Shopkeeper sold 10 articles at the cost price of 12 articles.
  ((12 * C - 10 * C) / (10 * C)) * 100 = 20 := 
by
  sorry

end profit_percentage_is_20_l691_691998


namespace min_value_sqrt_sum_eq_sqrt_20_l691_691399

def f (x : ℝ) : ℝ := Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2)

theorem min_value_sqrt_sum_eq_sqrt_20 :
  ∃ x : ℝ, ∀ y : ℝ, f y ≥ f x ∧ f x = Real.sqrt 20 := by
  sorry

end min_value_sqrt_sum_eq_sqrt_20_l691_691399


namespace final_statue_weight_l691_691340

-- Define the initial weight of the statue
def initial_weight : ℝ := 250

-- Define the percentage of weight remaining after each week
def remaining_after_week1 (w : ℝ) : ℝ := 0.70 * w
def remaining_after_week2 (w : ℝ) : ℝ := 0.80 * w
def remaining_after_week3 (w : ℝ) : ℝ := 0.75 * w

-- Define the final weight of the statue after three weeks
def final_weight : ℝ := 
  remaining_after_week3 (remaining_after_week2 (remaining_after_week1 initial_weight))

-- Prove the weight of the final statue is 105 kg
theorem final_statue_weight : final_weight = 105 := 
  by
    sorry

end final_statue_weight_l691_691340


namespace present_age_of_son_l691_691635

variable (S M : ℕ)

theorem present_age_of_son :
  (M = S + 30) ∧ (M + 2 = 2 * (S + 2)) → S = 28 :=
by
  sorry

end present_age_of_son_l691_691635


namespace hexagonal_H5_find_a_find_t_find_m_l691_691703

section problem1

-- Define the hexagonal number formula
def hexagonal_number (n : ℕ) : ℕ :=
  2 * n^2 - n

-- Define that H_5 should equal 45
theorem hexagonal_H5 : hexagonal_number 5 = 45 := sorry

end problem1

section problem2

variables (a b c : ℕ)

-- Given hexagonal number equations
def H1 := a + b + c
def H2 := 4 * a + 2 * b + c
def H3 := 9 * a + 3 * b + c

-- Conditions given in problem
axiom H1_def : H1 = 1
axiom H2_def : H2 = 7
axiom H3_def : H3 = 19

-- Prove that a = 3
theorem find_a : a = 3 := sorry

end problem2

section problem3

variables (p q r t : ℕ)

-- Given ratios in problem
axiom ratio1 : p * 3 = 2 * q
axiom ratio2 : q * 5 = 4 * r

-- Prove that t = 12
theorem find_t : t = 12 := sorry

end problem3

section problem4

variables (x y m : ℕ)

-- Given proportional conditions
axiom ratio3 : x * 3 = y * 4
axiom ratio4 : (x + y) * 3 = x * m

-- Prove that m = 7
theorem find_m : m = 7 := sorry

end problem4

end hexagonal_H5_find_a_find_t_find_m_l691_691703


namespace incorrect_statement_C_l691_691036

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin x - Real.sin (2 * x) + Real.sqrt 3

theorem incorrect_statement_C :
  ¬ ∃ x, f x = -Real.sqrt 3 :=
by
  sorry

end incorrect_statement_C_l691_691036


namespace percentage_of_students_getting_second_division_l691_691095

theorem percentage_of_students_getting_second_division
  (total_students : ℕ)
  (first_division_percentage : ℚ)
  (just_passed_students : ℕ)
  (no_student_failed : bool)
  (h1 : total_students = 300)
  (h2 : first_division_percentage = 30 / 100)
  (h3 : just_passed_students = 48)
  (h4 : no_student_failed = true) :
  let first_division_students := (first_division_percentage * total_students)
  let second_division_students := total_students - (first_division_students + just_passed_students)
  let second_division_percentage := (second_division_students / total_students) * 100 in
  second_division_percentage = 54 :=
by 
  sorry

end percentage_of_students_getting_second_division_l691_691095


namespace limit_ln_ex_power_l691_691375

noncomputable def limit_expr (x : ℝ) : ℝ :=
  (Real.log(ex) ^ 2) ^ (1 / (x ^ 2 + 1))

theorem limit_ln_ex_power (x : ℝ) : 
  filter.tendsto (λ x, limit_expr x) (nhds 1) (nhds 1) := sorry

end limit_ln_ex_power_l691_691375


namespace diophantine_eq_solutions_count_l691_691379

theorem diophantine_eq_solutions_count :
  let eq := (∃ x y : ℤ, 2 * x + 3 * y = 780 ∧ x > 0 ∧ y > 0)
  in (∃ count : ℕ, count = 130) :=
begin
  -- Problem statement and definition based on given conditions
  let eq := (∃ x y : ℤ, 2 * x + 3 * y = 780 ∧ x > 0 ∧ y > 0),

  -- Providing the conclusion that there exist 130 solutions
  have count := 130,
  
  -- Existential quantifier asserting the number of positive integer solutions 
  existsi count,
  
  -- Statement of equality confirming the number
  exact rfl,
end

end diophantine_eq_solutions_count_l691_691379


namespace find_starting_number_l691_691602

theorem find_starting_number (n : ℕ) (hn : n = 11) (end_num : ℕ) (hend_num : end_num = 43)
    (not_divisible_by_3 : ¬(end_num % 3 = 0)) : 
    ∃ start_num, (∀ k, 0 ≤ k ∧ k < n → (start_num + k * 3) ≤ end_num) ∧ 
                 (∀ k, 0 ≤ k ∧ k < n → (start_num + k * 3) % 3 = 0) ∧
                 (∃ end_multiple, end_multiple = (end_num - 1) div 3 * 3 + 3 ∧ (start_num = end_multiple - (n - 1) * 3)) :=
begin
  sorry
end

end find_starting_number_l691_691602


namespace find_R_jumps_l691_691895

def Ronald_jumps (R : ℕ) : Prop :=
  ∃ R' : ℕ, 
    (R' = R + 86) ∧ 
    (R + R' = 400) ∧ 
    (R = 157)

theorem find_R_jumps : ∃ R : ℕ, Ronald_jumps R :=
  by {
    use 157,
    unfold Ronald_jumps,
    sorry,
  }

end find_R_jumps_l691_691895


namespace three_digit_number_sum_of_factorials_l691_691287

open Nat

-- Define a function to calculate the factorial
def factorial : Nat → Nat
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * factorial (n + 1)

-- Define a theorem to state the problem
theorem three_digit_number_sum_of_factorials (n : ℕ) (h1 : 100 ≤ n ∧ n < 1000) (h2 : ∃ d1 d2 d3, n = d1 * 100 + d2 * 10 + d3 ∧ (d1 = 3 ∨ d2 = 3 ∨ d3 = 3)) :
  ∃ d1 d2 d3, n = d1 * 100 + d2 * 10 + d3 ∧ n = factorial d1 + factorial d2 + factorial d3 :=
by
  existsi (1, 4, 5)
  simp [factorial]
  sorry

end three_digit_number_sum_of_factorials_l691_691287


namespace distance_between_5th_and_29th_red_light_in_feet_l691_691575

-- Define the repeating pattern length and individual light distance
def pattern_length := 7
def red_light_positions := {k | k % pattern_length < 3}
def distance_between_lights := 8 / 12  -- converting inches to feet

-- Positions of the 5th and 29th red lights in terms of pattern repetition
def position_of_nth_red_light (n : ℕ) : ℕ :=
  ((n-1) / 3) * pattern_length + (n-1) % 3 + 1

def position_5th_red_light := position_of_nth_red_light 5
def position_29th_red_light := position_of_nth_red_light 29

theorem distance_between_5th_and_29th_red_light_in_feet :
  (position_29th_red_light - position_5th_red_light - 1) * distance_between_lights = 37 := by
  sorry

end distance_between_5th_and_29th_red_light_in_feet_l691_691575


namespace sequence_limit_l691_691697

open Filter Real

noncomputable def sequence_term (n : ℕ) : ℝ :=
  sqrt (n * (n + 1) * (n + 2)) * (sqrt (n^3 - 3) - sqrt (n^3 - 2))

theorem sequence_limit :
  tendsto (λ n : ℕ, sequence_term n) at_top (𝓝 (-1/2)) :=
sorry

end sequence_limit_l691_691697


namespace evaluate_powers_of_i_l691_691720

noncomputable def imag_unit := Complex.I

theorem evaluate_powers_of_i :
  (imag_unit^11 + imag_unit^16 + imag_unit^21 + imag_unit^26 + imag_unit^31) = -imag_unit :=
by
  sorry

end evaluate_powers_of_i_l691_691720


namespace intersect_empty_range_of_a_union_subsets_range_of_a_l691_691788

variable {x a : ℝ}

def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | (x - 6) * (x + 2) > 0}

theorem intersect_empty_range_of_a (h : A a ∩ B = ∅) : -2 ≤ a ∧ a ≤ 3 :=
by
  sorry

theorem union_subsets_range_of_a (h : A a ∪ B = B) : a < -5 ∨ a > 6 :=
by
  sorry

end intersect_empty_range_of_a_union_subsets_range_of_a_l691_691788


namespace three_digit_number_satisfies_conditions_l691_691724

-- Definitions for the digits of the number
def x := 9
def y := 6
def z := 4

-- Define the three-digit number
def number := 100 * x + 10 * y + z

-- Define the conditions
def geometric_progression := y * y = x * z

def reverse_order_condition := (number - 495) = 100 * z + 10 * y + x

def arithmetic_progression := (z - 1) + (x - 2) = 2 * (y - 1)

-- The theorem to prove
theorem three_digit_number_satisfies_conditions :
  geometric_progression ∧ reverse_order_condition ∧ arithmetic_progression :=
by {
  sorry
}

end three_digit_number_satisfies_conditions_l691_691724


namespace probability_below_x_axis_l691_691559

structure Point where
  x : ℝ
  y : ℝ

structure Parallelogram where
  E : Point
  F : Point
  G : Point
  H : Point

def E := {x := -1, y := 2 : Point}
def F := {x := 5, y := 2 : Point}
def G := {x := 1, y := -2 : Point}
def H := {x := -5, y := -2 : Point}

def parallelogram_EFGH : Parallelogram := {
  E := E,
  F := F,
  G := G,
  H := H
}

theorem probability_below_x_axis (P : Parallelogram) :
  P = parallelogram_EFGH → (∃ p : Point, p ≠ P.E ∧ p ≠ P.F ∧ p ≠ P.G ∧ p ≠ P.H ∧ p.y < 0) → ((p : Point) ∈ P → y < 0) → 
  ∃ p : Point, p ∈ P → p.y < 0 :=
sorry

end probability_below_x_axis_l691_691559


namespace no_factors_l691_691385

open Polynomial

noncomputable def polynomial := (X ^ 6 + 3 * X ^ 3 + 18)
noncomputable def option1 := (X ^ 3 + 6)
noncomputable def option2 := (X - 2)
noncomputable def option3 := (X ^ 3 - 6)
noncomputable def option4 := (X ^ 3 - 3 * X - 9)

theorem no_factors (p : Polynomial ℚ) :
  ¬ p = polynomial.divByMonic p X * X + polynomial.modByMonic p X → 
  ¬ p = polynomial.divByMonic p option1 * option1 + polynomial.modByMonic p option1 → 
  ¬ p = polynomial.divByMonic p option2 * option2 + polynomial.modByMonic p option2 → 
  ¬ p = polynomial.divByMonic p option3 * option3 + polynomial.modByMonic p option3 → 
  ¬ p = polynomial.divByMonic p option4 * option4 + polynomial.modByMonic p option4 → 
  p != polynomial :=
by 
  sorry

end no_factors_l691_691385


namespace thirty_minute_programs_commercials_l691_691152

theorem thirty_minute_programs_commercials (n : ℕ) 
    (h1 : ∀ (x : ℕ), commercials_per_program = 7.5) 
    (h2 : total_commercials_time = 45) :
    (7.5 * n = 45) → n = 6 := 
by 
  intro h
  have h3 := (h2 + h1)
  sorry 

end thirty_minute_programs_commercials_l691_691152


namespace different_tens_digit_probability_l691_691175

open BigOperators

theorem different_tens_digit_probability :
  let total_count := finset.card (finset.Icc 20 69),
      combinations := nat.choose total_count 5,
      favorable_ways := 10 ^ 5 in
  (favorable_ways : ℚ) / combinations = 2500 / 52969 :=
by
  let total_count := finset.card (finset.Icc 20 69),
      combinations := nat.choose total_count 5,
      favorable_ways := 10 ^ 5
  have h1 : total_count = 50 := by sorry
  have h2 : favorable_ways = 100000 := by sorry
  have h3 : combinations = 2118760 := by sorry
  have ratio : (100000 : ℚ) / 2118760 = 2500 / 52969 := by sorry
  exact ratio

end different_tens_digit_probability_l691_691175


namespace teacups_after_all_boxes_taken_l691_691896

def total_boxes := 26
def boxes_with_pans := 6
def rows_per_box := 5
def cups_per_row := 4
def broken_cups_per_box := 2

def remaining_boxes := total_boxes - boxes_with_pans
def boxes_with_decorations := remaining_boxes / 2
def boxes_with_teacups := remaining_boxes - boxes_with_decorations
def cups_per_box := rows_per_box * cups_per_row

def total_teacups_before_breakage := boxes_with_teacups * cups_per_box
def total_broken_teacups := boxes_with_teacups * broken_cups_per_box
def final_teacups := total_teacups_before_breakage - total_broken_teacups

theorem teacups_after_all_boxes_taken (total_boxes boxes_with_pans rows_per_box cups_per_row broken_cups_per_box : ℕ) :
  final_teacups = 180 :=
by
  unfold total_boxes boxes_with_pans rows_per_box cups_per_row broken_cups_per_box remaining_boxes boxes_with_decorations boxes_with_teacups cups_per_box total_teacups_before_breakage total_broken_teacups final_teacups
  -- Formal proof steps would go here
  sorry

end teacups_after_all_boxes_taken_l691_691896


namespace triangle_perimeter_range_l691_691440

noncomputable def perimeter_range (A B C : ℝ) (a b c : ℝ) : Prop :=
  triangle_acute A B C ∧
  c = 2 ∧
  (a * cos B + b * cos A = (sqrt 3 * c) / (2 * sin C)) →
  4 < (a + b + c) ∧ (a + b + c) ≤ 6

theorem triangle_perimeter_range (A B C : ℝ) (a b c : ℝ) :
  perimeter_range A B C a b c :=
sorry

end triangle_perimeter_range_l691_691440


namespace a_200_correct_l691_691928

def a_seq : ℕ → ℕ
| 0     := 1
| (n+1) := a_seq n + 2 * a_seq n / n

theorem a_200_correct : a_seq 200 = 20100 := 
by sorry

end a_200_correct_l691_691928


namespace solution_set_f_x_plus_1_l691_691127

noncomputable def f : ℝ → ℝ := sorry

axiom decreasing_f : ∀ x y : ℝ, x < y → f y < f x
axiom f_at_0 : f 0 = 1
axiom f_at_3 : f 3 = -1

theorem solution_set_f_x_plus_1 :
  {x : ℝ | |f(x + 1)| < 1} = set.Ioo (-1 : ℝ) (2 : ℝ) :=
by
  sorry

end solution_set_f_x_plus_1_l691_691127


namespace lab_preparation_is_correct_l691_691919

def correct_operation (m_CuSO4 : ℝ) (m_CuSO4_5H2O : ℝ) (V_solution : ℝ) : Prop :=
  let molar_mass_CuSO4 := 160 -- g/mol
  let molar_mass_CuSO4_5H2O := 250 -- g/mol
  let desired_concentration := 0.1 -- mol/L
  let desired_volume := 0.480 -- L
  let prepared_volume := 0.500 -- L
  (m_CuSO4 = 8.0 ∧ V_solution = 0.500 ∧ m_CuSO4_5H2O = 12.5 ∧ desired_concentration * prepared_volume * molar_mass_CuSO4_5H2O = 12.5)

-- Example proof statement to show the problem with "sorry"
theorem lab_preparation_is_correct : correct_operation 8.0 12.5 0.500 :=
by
  sorry

end lab_preparation_is_correct_l691_691919


namespace original_number_of_men_l691_691295

theorem original_number_of_men (M : ℤ) (h1 : 8 * M = 5 * (M + 10)) : M = 17 := by
  -- Proof goes here
  sorry

end original_number_of_men_l691_691295


namespace minimum_pipes_l691_691672

theorem minimum_pipes (h : ℝ) : 
  (let volume_8_inch := π * (4^2) * h in
   let volume_3_inch := π * (1.5^2) * h in
   let number_of_pipes := (volume_8_inch / volume_3_inch).ceil in
   number_of_pipes = 8) :=
sorry

end minimum_pipes_l691_691672


namespace range_of_m_l691_691778

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (1 / 3) * x^3 - m * x^2 - 3 * m^2 * x + 1

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x ≤ f y

theorem range_of_m : 
  {m : ℝ | is_increasing_on (λ x, (1 / 3) * x^3 - m * x^2 - 3 * m^2 * x + 1) 1 2} = set.Icc (-1:ℝ) (1/3:ℝ) :=
  sorry

end range_of_m_l691_691778


namespace proportion_of_fathers_with_full_time_jobs_l691_691093

theorem proportion_of_fathers_with_full_time_jobs
  (P : ℕ) -- Total number of parents surveyed
  (mothers_proportion : ℝ := 0.4) -- Proportion of mothers in the survey
  (mothers_ftj_proportion : ℝ := 0.9) -- Proportion of mothers with full-time jobs
  (parents_no_ftj_proportion : ℝ := 0.19) -- Proportion of parents without full-time jobs
  (hfathers : ℝ := 0.6) -- Proportion of fathers in the survey
  (hfathers_ftj_proportion : ℝ) -- Proportion of fathers with full-time jobs
  : hfathers_ftj_proportion = 0.75 := 
by 
  sorry

end proportion_of_fathers_with_full_time_jobs_l691_691093


namespace largest_prime_m_satisfying_quadratic_inequality_l691_691731

theorem largest_prime_m_satisfying_quadratic_inequality :
  ∃ (m : ℕ), m = 5 ∧ m^2 - 11 * m + 28 < 0 ∧ Prime m :=
by sorry

end largest_prime_m_satisfying_quadratic_inequality_l691_691731


namespace b_value_rational_polynomial_l691_691015

theorem b_value_rational_polynomial (a b : ℚ) :
  (Polynomial.aeval (2 + Real.sqrt 3) (Polynomial.C (-15) + Polynomial.C b * X + Polynomial.C a * X^2 + X^3 : Polynomial ℚ) = 0) →
  b = -44 :=
by
  sorry

end b_value_rational_polynomial_l691_691015


namespace distinct_arrangements_BOOKKEEPER_l691_691387

theorem distinct_arrangements_BOOKKEEPER :
  let n := 9
  let nO := 2
  let nK := 2
  let nE := 3
  ∃ arrangements : ℕ,
  arrangements = Nat.factorial n / (Nat.factorial nO * Nat.factorial nK * Nat.factorial nE) ∧
  arrangements = 15120 :=
by { sorry }

end distinct_arrangements_BOOKKEEPER_l691_691387


namespace find_m_n_and_angle_AOC_l691_691538

noncomputable def vector_OA := (-2 : ℝ, m : ℝ)
noncomputable def vector_OB := (n : ℝ, 1 : ℝ)
noncomputable def vector_OC := (5 : ℝ, -1 : ℝ)

def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (fst B - fst A) = k * (fst C - fst A) ∧ (snd B - snd A) = k * (snd C - snd A)

def orthogonal (v w : ℝ × ℝ) : Prop := 
  fst v * fst w + snd v * snd w = 0

theorem find_m_n_and_angle_AOC
  (m n : ℝ) 
  (h_collinear : collinear vector_OA vector_OB vector_OC)
  (h_orthogonal : orthogonal vector_OA vector_OB) :
  ((m = 6 ∧ n = 3) ∨ (m = 3 ∧ n = 3 / 2)) ∧ 
  (orthocenter (0, 0) vector_OA vector_OC = (0, 0) → vector_OB = (3 / 2, 0) ∧ acos ((fst vector_OA * fst vector_OC + snd vector_OA * snd vector_OC) / (sqrt (fst vector_OA ^ 2 + snd vector_OA ^ 2) * sqrt (fst vector_OC ^ 2 + snd vector_OC ^ 2))) = 3 * pi / 4) :=
sorry


end find_m_n_and_angle_AOC_l691_691538


namespace cos_double_angle_of_tan_l691_691811

theorem cos_double_angle_of_tan (θ : ℝ) (h : Real.tan θ = -1 / 3) : Real.cos (2 * θ) = 4 / 5 :=
sorry

end cos_double_angle_of_tan_l691_691811


namespace mutual_exclusive_B_C_l691_691411

def items := ℕ

def A (s : set items) := (∀ x ∈ s, x = 0) -- all three products are non-defective
def B (s : set items) := (∀ x ∈ s, x = 1) -- all three products are defective
def C (s : set items) := (¬ B s) -- not all three products are defective

theorem mutual_exclusive_B_C : ∀ (s : set items), B s → ¬ C s :=
by
  intros s hB
  unfold C
  exact hB

end mutual_exclusive_B_C_l691_691411


namespace farmer_field_configuration_l691_691664

noncomputable def number_of_ways_to_plant_crops (R C : Type) [Fintype R] [Fintype C] (grid : R × C → Type) :=
  ∃ (corn wheat soybeans potatoes : Type),
  ∀ (plant : R × C → Type),
    (∀ (r r' : R) (c c' : C), ((r, c).adjacent (r', c') → plant (r, c) ≠ plant (r', c'))) ∧
    (∀ (r r' : R) (c c' : C), (plant (r, c) = corn ∧ plant (r', c) = wheat → ¬ ((r, c).adjacent (r', c')))) ∧
    (∀ (r r' : R) (c c' : C), (plant (r, c) = soybeans ∧ plant (r', c) = potatoes → ¬((r, c).adjacent (r', c')))) ∧
    card (fintype.of (λ plant : R × C → Type, ∀ x y, x ≠ y → plant x ≠ plant y)) = 1024

theorem farmer_field_configuration :
  number_of_ways_to_plant_crops (fin 3) (fin 3) (λ _, fin 4) := by sorry

end farmer_field_configuration_l691_691664


namespace cost_price_of_article_l691_691071

theorem cost_price_of_article (C SP1 SP2 G1 G2 : ℝ) 
  (h_SP1 : SP1 = 160) 
  (h_SP2 : SP2 = 220) 
  (h_gain_relation : G2 = 1.05 * G1) 
  (h_G1 : G1 = SP1 - C) 
  (h_G2 : G2 = SP2 - C) : C = 1040 :=
by
  sorry

end cost_price_of_article_l691_691071


namespace compare_magnitudes_l691_691123

noncomputable def log_base_3_of_2 : ℝ := Real.log 2 / Real.log 3   -- def a
noncomputable def ln_2 : ℝ := Real.log 2                          -- def b
noncomputable def five_minus_pi : ℝ := 5 - Real.pi                -- def c

theorem compare_magnitudes :
  let a := log_base_3_of_2
  let b := ln_2
  let c := five_minus_pi
  c < a ∧ a < b :=
by
  sorry

end compare_magnitudes_l691_691123


namespace part1_geometric_part2_range_of_t_l691_691011

-- Define the sequence a_n and its initial condition
def a : ℕ → ℕ
| 0     := 4
| (n+1) := 2 * a n - 2 * n + 1

-- Define the sequence b_n
def b (t: ℝ) (n: ℕ) : ℝ :=
  t * n + 2

-- Part 1: Prove {a_n - 2n - 1} is a geometric sequence with a common ratio of 2
theorem part1_geometric : ∃ (r : ℕ), ∀ n : ℕ, (a n) - 2 * n - 1 = r * (2 ^ n) := 
sorry

-- Part 2: Prove the range for t
theorem part2_range_of_t (t : ℝ) : (∀ n : ℕ, n > 0 → b t n < 2 * a n) ↔ (t < 6) := 
sorry

end part1_geometric_part2_range_of_t_l691_691011


namespace find_m_l691_691474

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (-1, m)
def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_m (m : ℝ) (h : is_perpendicular vector_a (vector_b m)) : m = 1 / 2 :=
by 
  sorry

end find_m_l691_691474


namespace common_difference_l691_691441

theorem common_difference (a : ℕ → ℤ) (d : ℤ) 
    (h1 : ∀ n : ℕ, a (n + 1) = a 1 + n * d)
    (h2 : a 1 + a 3 + a 5 = 15)
    (h3 : a 4 = 3) : 
    d = -2 := 
sorry

end common_difference_l691_691441


namespace circle_inscribed_radius_l691_691224

theorem circle_inscribed_radius (R α : ℝ) (hα : α < Real.pi) : 
  ∃ x : ℝ, x = R * (Real.sin (α / 4))^2 :=
sorry

end circle_inscribed_radius_l691_691224


namespace hall_length_l691_691639

theorem hall_length (b : ℕ) (h1 : b + 5 > 0) (h2 : (b + 5) * b = 750) : b + 5 = 30 :=
by {
  -- Proof goes here
  sorry
}

end hall_length_l691_691639


namespace subset_A_imp_range_a_disjoint_A_imp_range_a_l691_691039

-- Definition of sets A and B
def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a)*(x - 3*a) < 0}

-- Proof problem for Question 1
theorem subset_A_imp_range_a (a : ℝ) (h : A ⊆ B a) : 
  (4 / 3) ≤ a ∧ a ≤ 2 ∧ a ≠ 0 :=
sorry

-- Proof problem for Question 2
theorem disjoint_A_imp_range_a (a : ℝ) (h : A ∩ B a = ∅) : 
  a ≤ (2 / 3) ∨ a ≥ 4 :=
sorry

end subset_A_imp_range_a_disjoint_A_imp_range_a_l691_691039


namespace find_innings_l691_691578

noncomputable def calculate_innings (A : ℕ) (n : ℕ) : Prop :=
  (n * A + 140 = (n + 1) * (A + 8)) ∧ (A + 8 = 28)

theorem find_innings (n : ℕ) (A : ℕ) :
  calculate_innings A n → n = 14 :=
by
  intros h
  -- Here you would prove that h implies n = 14, but we use sorry to skip the proof steps.
  sorry

end find_innings_l691_691578


namespace not_possible_arrangement_l691_691686

theorem not_possible_arrangement : 
  ¬∃ (f : Fin 9 → Fin 9), (Bijective f) ∧ 
  (∀ i, f i + 1 = (if i.val = 8 then 0 else i.val + 1)) ∧ 
  (∀ i, ((f (if i.val = 8 then 0 else i.val + 1) - f i) % f i = 0)) := 
sorry

end not_possible_arrangement_l691_691686


namespace sum_of_three_squares_l691_691680

theorem sum_of_three_squares (a b : ℝ)
  (h1 : 3 * a + 2 * b = 18)
  (h2 : 2 * a + 3 * b = 22) :
  3 * b = 18 :=
sorry

end sum_of_three_squares_l691_691680


namespace correct_growth_rate_equation_l691_691670

-- Define the conditions
def packages_first_day := 200
def packages_third_day := 242

-- Define the average daily growth rate
variable (x : ℝ)

-- State the theorem to prove
theorem correct_growth_rate_equation :
  packages_first_day * (1 + x)^2 = packages_third_day :=
by
  sorry

end correct_growth_rate_equation_l691_691670


namespace arithmeticSeqModulus_l691_691695

-- Define the arithmetic sequence
def arithmeticSeqSum (a d l : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

-- The main theorem to prove
theorem arithmeticSeqModulus : arithmeticSeqSum 2 5 102 % 20 = 12 := by
  sorry

end arithmeticSeqModulus_l691_691695


namespace instantaneous_velocity_at_3_l691_691198

def s (t : ℝ) : ℝ := 1 - t + t^2

def velocity_at_t (t : ℝ) : ℝ := (deriv s) t

theorem instantaneous_velocity_at_3 : velocity_at_t 3 = 5 := by
  sorry

end instantaneous_velocity_at_3_l691_691198


namespace units_digit_T_l691_691851

def factorial_units_digit_sum(n : ℕ) : ℕ :=
  (List.range (n + 1)).map (λ k, ((Nat.factorial k) % 10)).sum % 10

theorem units_digit_T : factorial_units_digit_sum 100 = 3 := by
  sorry

end units_digit_T_l691_691851


namespace YZ_over_BD_l691_691973

noncomputable section

variables {A B C D X Y Z : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variable (a b c d : ℝ) -- lengths of sides AB, BC, CD, AD respectively

-- Given conditions
def cyclic_quadrilateral (A B C D : Type) : Prop :=
  true -- placeholder for cyclic quadrilateral property

def diagonals_intersect (X : Type) (A B C D : Type) : Prop :=
  true -- placeholder for intersection of diagonals AC and BD at X

def circle_tangent_at_X (A X : Type) (BD : Type) : Prop :=
  true -- placeholder for circle ω passing through A and tangent to BD at X

def circle_intersects_segment (circle : Type) (segment : Type) (point1 point2 : Type) : Prop :=
  true -- placeholder for circle ω intersecting a segment at two points

variables (AB BC CD AD : Real) (k : ℝ)
  (h1 : AB = 4) (h2 : BC = 3) (h3 : CD = 2) (h4 : AD = 5)
  (h5 : cyclic_quadrilateral A B C D) 
  (h6 : diagonals_intersect X A B C D)
  (h7 : circle_tangent_at_X A X (BD : Real))
  (h8 : circle_intersects_segment (ω : Type) (AB : Type) (Y : Type) (D : Type))
  (h9 : circle_intersects_segment (ω2 : Type) (AD : Type) (Y : Type) (Z : Type))

-- Final statement to prove
theorem YZ_over_BD : 
  YZ / BD = 115 / 143 :=
sorry

end YZ_over_BD_l691_691973


namespace triangle_perimeter_l691_691922

theorem triangle_perimeter (r A : ℝ) (h1 : r = 2.5) (h2 : A = 50) : ∃ p : ℝ, p = 40 ∧ A = r * (p / 2) :=
by
  use 40
  split
  { exact rfl }
  { rw [h1, h2, ← mul_assoc, ← div_div, div_eq_mul_inv, mul_comm] }
  sorry

end triangle_perimeter_l691_691922


namespace parallel_vectors_equal_l691_691047

theorem parallel_vectors_equal (x : ℝ) (h : ∃ k : ℝ, (2, 3) = k • (x, 6)) : x = 4 :=
sorry

end parallel_vectors_equal_l691_691047


namespace probability_of_odd_sum_l691_691959

-- Defining the problem conditions
def odd_probability_is_four_times_even : Prop :=
  ∃ (p_even p_odd : ℝ), p_even + p_odd = 1 ∧ p_odd = 4 * p_even

-- Define the probability calculation based on the given conditions
def probability_sum_is_odd : ℝ :=
  let p_even := 1/5 in
  let p_odd := 4/5 in
  (p_even * p_odd) + (p_odd * p_even)

-- Problem statement
theorem probability_of_odd_sum :
  odd_probability_is_four_times_even →
  probability_sum_is_odd = 8/25 :=
by
  intros _,
  unfold probability_sum_is_odd,
  exact sorry

end probability_of_odd_sum_l691_691959


namespace hands_per_hoopit_l691_691151

-- Defining conditions
def num_hoopits := 7
def num_neglarts := 8
def total_toes := 164
def toes_per_hand_hoopit := 3
def toes_per_hand_neglart := 2
def hands_per_neglart := 5

-- The statement to prove
theorem hands_per_hoopit : 
  ∃ (H : ℕ), (H * toes_per_hand_hoopit * num_hoopits + hands_per_neglart * toes_per_hand_neglart * num_neglarts = total_toes) → H = 4 :=
sorry

end hands_per_hoopit_l691_691151


namespace incorrect_statement_l691_691931

-- Given scores list
def scores : List ℝ := [7, 8, 9, 7, 4, 8, 9, 9, 7, 2]

-- Definition of mean
def mean (l : List ℝ) : ℝ := (l.sum) / (l.length)

-- Definition of variance
def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / (l.length)

-- Compute median
def median (l : List ℝ) : ℝ :=
  let sorted := l.qsort (· ≤ ·)
  if sorted.length % 2 = 0 then
    (sorted.get (sorted.length / 2 - 1) + sorted.get (sorted.length / 2)) / 2
  else
    sorted.get (sorted.length / 2)

-- Prove that the incorrect statement is C
theorem incorrect_statement :
  let modes := [7, 9]
  let avg := 7
  let med := 7.5
  let var := 4.8
  (modes = [7, 9]) ∧ (mean scores = avg) ∧ (median scores = med) ∧ (variance scores = var) →
  "The incorrect statement is: C"
:= by
  intros
  sorry

end incorrect_statement_l691_691931


namespace add_three_times_more_l691_691282

theorem add_three_times_more (n : ℝ) (k : ℝ) : 
  12.8 + (3 * n + n) = k → (k = 2444.8) := by
  intro h
  calc
    12.8 + 4 * 608 = 2444.8 : by sorry

end add_three_times_more_l691_691282


namespace sum_is_18_less_than_abs_sum_l691_691210

theorem sum_is_18_less_than_abs_sum : 
  (-5 + -4) = (|-5| + |-4| - 18) :=
by
  sorry

end sum_is_18_less_than_abs_sum_l691_691210


namespace complex_number_value_l691_691548

-- Declare the imaginary unit 'i'
noncomputable def i : ℂ := Complex.I

-- Define the problem statement
theorem complex_number_value : (i / ((1 - i) ^ 2)) = -1/2 := 
by
  sorry

end complex_number_value_l691_691548


namespace solve_for_c_l691_691457

variables (m c b a : ℚ) -- Declaring variables as rationals for added precision

theorem solve_for_c (h : m = (c * b * a) / (a - c)) : 
  c = (m * a) / (m + b * a) := 
by 
  sorry -- Proof not required as per the instructions

end solve_for_c_l691_691457


namespace part1_part2_l691_691456

-- Given conditions for part 1
def ellipse_eq (a b : ℝ) (h1 : a > b) (h2 : b > 0) (e : ℝ) (h3 : e = sqrt 6 / 3) : Prop :=
  ∃ C : ℝ, C = sqrt 3 ∧ C = a ∧ b = 1 ∧ (x : ℝ) (y : ℝ), (x^2 / 3) + y^2 = 1

-- Statement for part 1
theorem part1 (h1 : 0 < 1) (h2 : 1 < sqrt 3) (h3 : sqrt 6 / (sqrt 3 * 2) = sqrt 3) :
  (∃ a b : ℝ, ellipse_eq a b h1 h2 (sqrt 6 / 3)) →
  ∃ C : ℝ, C = sqrt 3 ∧ ∀ x y : ℝ, (x^2 / 3) + y^2 = 1 := sorry

-- Given conditions for part 2
def slope_range (k : ℝ) : Prop :=
  ∃ P : ℝ × ℝ, P = (0,2) ∧ 
  (1 < k^2 ∧ k^2 < (13 / 3))

-- Statement for part 2
theorem part2 (k : ℝ) (P : ℝ × ℝ)
  (hk : P = (0, 2))
  (h1 : slope_range k) :
  k ∈ (Set.Ioo (1 : ℝ) (sqrt 13 / 3 : ℝ)) ∪ Set.Ioo (-(sqrt 39 / 3 : ℝ)) (-1 : ℝ) ∪ Set.Ioo (1 : ℝ) (sqrt 39 / 3 : ℝ) := sorry

end part1_part2_l691_691456


namespace fish_buckets_last_l691_691688

theorem fish_buckets_last (buckets_sharks : ℕ) (buckets_total : ℕ) 
  (h1 : buckets_sharks = 4)
  (h2 : ∀ (buckets_dolphins : ℕ), buckets_dolphins = buckets_sharks / 2)
  (h3 : ∀ (buckets_other : ℕ), buckets_other = 5 * buckets_sharks)
  (h4 : buckets_total = 546)
  : 546 / ((buckets_sharks + (buckets_sharks / 2) + (5 * buckets_sharks)) * 7) = 3 :=
by
  -- Calculation steps skipped for brevity
  sorry

end fish_buckets_last_l691_691688


namespace added_number_is_nine_l691_691327

theorem added_number_is_nine (y : ℤ) : 
  3 * (2 * 4 + y) = 51 → y = 9 :=
by
  sorry

end added_number_is_nine_l691_691327


namespace symmetric_point_wrt_y_axis_l691_691432

-- Define the point A
def A : ℝ × ℝ × ℝ := (-3, 1, -4)

-- Define the transformation function for y-axis symmetry
def symmetric_wrt_y_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.1, p.2, -p.3)

-- Aim to prove that the symmetric point of A with respect to the y-axis is (3, 1, 4)
theorem symmetric_point_wrt_y_axis :
  symmetric_wrt_y_axis A = (3, 1, 4) :=
by
  sorry

end symmetric_point_wrt_y_axis_l691_691432


namespace hyperbola_eccentricity_is_sqrt3_l691_691009

def hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :=
  ∀ (x0 y0 : ℝ), (x0^2 / a^2 - y0^2 / b^2 = 1) →
  (x0 ≠ a) →
  (x0 ≠ -a) →
  (y0^2 / (x0^2 - a^2) = 2 * b^2 / a^2) →
  let c := sqrt(a^2 + b^2) in
  let e := c / a in
  e = sqrt(3)

theorem hyperbola_eccentricity_is_sqrt3 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  hyperbola_eccentricity a b ha hb :=
by
  sorry

end hyperbola_eccentricity_is_sqrt3_l691_691009


namespace max_rooks_queens_on_chessboard_l691_691624

theorem max_rooks_queens_on_chessboard : 
  ∃ (n : ℕ), n = 10 ∧ 
  (∀ (R : ℕ → ℕ → Prop), (∀ i j, R i j → (i < 8) ∧ (j < 8)) ∧ (∀ i, ∃ j1 j2, j1 ≠ j2 ∧ R i j1 ∧ R i j2) ∧ (∃ fn : fin 8 → fin 8, ∀ i, R i (fn i) → n = 10)) ∧
  (∀ (Q : ℕ → ℕ → Prop), (∀ i j, Q i j → (i < 8) ∧ (j < 8)) ∧ (∀ i, ∃ j1 j2, j1 ≠ j2 ∧ Q i j1 ∧ Q i j2) ∧ (∃ fn : fin 8 → fin 8, ∀ i, Q i (fn i) → n = 10))
:= 
  sorry

end max_rooks_queens_on_chessboard_l691_691624


namespace total_ranking_sequences_at_end_l691_691090

-- Define the teams
inductive Team
| E
| F
| G
| H

open Team

-- Conditions of the problem
def split_groups : (Team × Team) × (Team × Team) :=
  ((E, F), (G, H))

def saturday_matches : (Team × Team) × (Team × Team) :=
  ((E, F), (G, H))

-- Function to count total ranking sequences
noncomputable def total_ranking_sequences : ℕ := 4

-- Define the main theorem
theorem total_ranking_sequences_at_end : total_ranking_sequences = 4 :=
by
  sorry

end total_ranking_sequences_at_end_l691_691090


namespace prime_gt_three_times_n_l691_691555

def nth_prime (n : ℕ) : ℕ :=
  -- Define the nth prime function, can use mathlib functionality
  sorry

theorem prime_gt_three_times_n (n : ℕ) (h : 12 ≤ n) : nth_prime n > 3 * n :=
  sorry

end prime_gt_three_times_n_l691_691555


namespace half_absolute_difference_of_squares_l691_691262

-- Defining the variables a and b involved in the problem
def a : ℤ := 20
def b : ℤ := 15

-- Statement to prove the solution
theorem half_absolute_difference_of_squares : 
    (1 / 2 : ℚ) * (abs ((a ^ 2) - (b ^ 2))) = 87.5 :=
by
    -- Proof omitted
    sorry

end half_absolute_difference_of_squares_l691_691262


namespace function_monotonicity_l691_691589

theorem function_monotonicity (a b : ℝ) : 
  (∀ x, -1 < x ∧ x < 1 → (3 * x^2 + a) < 0) ∧ 
  (∀ x, 1 < x → (3 * x^2 + a) > 0) → 
  (a = -3 ∧ ∃ b : ℝ, true) :=
by {
  sorry
}

end function_monotonicity_l691_691589


namespace measure_minor_arc_KT_l691_691526

open Locale.RealInnerProductSpace

def angle_measure (θ : ℝ) : ℝ := θ  -- defining angle measure function 

def inscribed_angle_intercepts_arc_twice (θ : ℝ) : Prop :=
  ∃ arc_measure : ℝ, arc_measure = 2 * θ

theorem measure_minor_arc_KT (angle_KAT : ℝ) (h : angle_KAT = 60) :
  ∃ arc_KT : ℝ, arc_KT = 2 * angle_KAT ∧ arc_KT = 120 := 
by 
  have h1 : angle_KAT = 60 := h
  use 120
  split
  · rw h1
    norm_num
  · norm_num

end measure_minor_arc_KT_l691_691526


namespace unique_bit_position_l691_691643

variable {n : ℕ}

-- Definition of a sequence of zeros and ones of length n
def sequence := Fin n → Bool

-- Assume we have 2^(n-1) different sequences
variable {seqs : Fin (2 ^ (n-1)) → sequence}

-- Condition: For any three sequences, there exists a position with 1 in all three sequences
def condition (seqs : Fin (2 ^ (n-1)) → sequence) : Prop :=
  ∀ i j k : Fin (2 ^ (n-1)), ∃ p : Fin n, seqs i p = true ∧ seqs j p = true ∧ seqs k p = true

theorem unique_bit_position (seqs : Fin (2 ^ (n-1)) → sequence) (h : condition seqs) :
  ∃! p : Fin n, ∀ i : Fin (2 ^ (n-1)), seqs i p = true :=
sorry

end unique_bit_position_l691_691643


namespace multiples_of_six_units_digit_six_l691_691797

theorem multiples_of_six_units_digit_six (n : ℕ) : 
  (number_of_six_less_150 n) = 25 
where 
  number_of_six_less_150 (n : ℕ) := 
    ∃ M : ℕ, M * 6 = n ∧ n < 150 
:= 
  sorry

end multiples_of_six_units_digit_six_l691_691797


namespace rachel_homework_difference_l691_691162

def total_difference (r m h s : ℕ) : ℕ :=
  (r - m) + (s - h)

theorem rachel_homework_difference :
    ∀ (r m h s : ℕ), r = 7 → m = 5 → h = 3 → s = 6 → total_difference r m h s = 5 :=
by
  intros r m h s hr hm hh hs
  rw [hr, hm, hh, hs]
  rfl

end rachel_homework_difference_l691_691162


namespace geom_seq_sixth_term_l691_691588

theorem geom_seq_sixth_term (a : ℝ) (r : ℝ) (h1: a * r^3 = 512) (h2: a * r^8 = 8) : 
  a * r^5 = 128 := 
by 
  sorry

end geom_seq_sixth_term_l691_691588


namespace conjugate_of_z_l691_691134

open Complex

theorem conjugate_of_z {z : ℂ} (h : z / (z - 2 * I) = I) : conj z = 1 - I := 
sorry

end conjugate_of_z_l691_691134


namespace circle_area_increase_l691_691079

theorem circle_area_increase (r : ℝ) :
  let initial_area := real.pi * r^2
  let new_radius := 2 * r
  let new_area := real.pi * (new_radius)^2
  let area_increase := new_area - initial_area
  let percent_increase := (area_increase / initial_area) * 100
  percent_increase = 300 :=
by {
  sorry
}

end circle_area_increase_l691_691079


namespace difference_largest_smallest_l691_691609

def num1 : ℕ := 10
def num2 : ℕ := 11
def num3 : ℕ := 12

theorem difference_largest_smallest :
  (max num1 (max num2 num3)) - (min num1 (min num2 num3)) = 2 :=
by
  -- Proof can be filled here
  sorry

end difference_largest_smallest_l691_691609


namespace vasya_triangles_l691_691616

theorem vasya_triangles (A B C D : Type) (e1 e2 e3 e4 e5 e6 : ℝ)
  (h1 : e1 < e4 + e5)
  (h2 : e1 < e3 + e6) :
  ∃ T1 T2 : set (ℝ), T1 ∪ T2 = {e1, e2, e3, e4, e5, e6} ∧
  (∀ e ∈ {e1, e2, e3, e4, e5, e6}, e ∈ T1 ⊕ e ∈ T2) ∧
  (T1.card = 3 ∧ T2.card = 3) := 
begin
  sorry
end

end vasya_triangles_l691_691616


namespace line_intersects_at_least_four_disks_l691_691844

theorem line_intersects_at_least_four_disks
  (square_side : ℝ)
  (square_side_eq_one : square_side = 1)
  (disks : list (disk ℝ))
  (circumference_sum_eq_ten : ∑ d in disks, disk.circumference d = 10) :
  ∃ l : line ℝ, ∃ subset_disks : finset (disk ℝ), subset_disks.card ≥ 4 ∧
    ∀ d ∈ subset_disks, line.intersects l d :=
begin
  sorry
end

end line_intersects_at_least_four_disks_l691_691844


namespace tan_half_angle_sum_identity_l691_691485

theorem tan_half_angle_sum_identity
  (α β γ : ℝ)
  (h : Real.sin α + Real.sin γ = 2 * Real.sin β) :
  Real.tan ((α + β) / 2) + Real.tan ((β + γ) / 2) = 2 * Real.tan ((γ + α) / 2) :=
sorry

end tan_half_angle_sum_identity_l691_691485


namespace number_of_workers_l691_691906

open Real

theorem number_of_workers (W : ℝ) 
    (average_salary_workers average_salary_technicians average_salary_non_technicians : ℝ)
    (h1 : average_salary_workers = 8000)
    (h2 : average_salary_technicians = 12000)
    (h3 : average_salary_non_technicians = 6000)
    (h4 : average_salary_workers * W = average_salary_technicians * 7 + average_salary_non_technicians * (W - 7)) :
    W = 21 :=
by
  rw [h1, h2, h3] at h4
  simp at h4
  linarith

end number_of_workers_l691_691906


namespace find_a200_l691_691930

def seq (a : ℕ → ℕ) : Prop :=
a 1 = 1 ∧ ∀ n ≥ 1, a (n + 1) = a n + 2 * a n / n

theorem find_a200 (a : ℕ → ℕ) (h : seq a) : a 200 = 20100 :=
sorry

end find_a200_l691_691930


namespace minimum_value_expression_l691_691815

theorem minimum_value_expression (a b c : ℤ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
    3 * a^2 + 2 * b^2 + 4 * c^2 - a * b - 3 * b * c - 5 * c * a ≥ 6 :=
sorry

end minimum_value_expression_l691_691815


namespace gcd_sequence_l691_691207

noncomputable def P : ℕ → ℕ := sorry   -- Placeholder for the polynomial with positive integer coefficients

def sequence (a0 : ℕ) (P : ℕ → ℕ) : ℕ → ℕ
| 0       := a0
| (n + 1) := P (sequence n)

theorem gcd_sequence {m k d : ℕ} (hmk : d = Nat.gcd m k) :
  ∀ (a : ℕ) (P : ℕ → ℕ) (hP : ∀ x, 0 < x → 0 < P x),
    (sequence 0 P m) ∣ (sequence 0 P k) :=
begin
  intros a P hP,
  sorry
end

end gcd_sequence_l691_691207


namespace find_rate_of_interest_l691_691163

theorem find_rate_of_interest (P SI : ℝ) (r : ℝ) (hP : P = 1200) (hSI : SI = 108) (ht : r = r) :
  SI = P * r * r / 100 → r = 3 := by
  intros
  sorry

end find_rate_of_interest_l691_691163


namespace excircle_bounds_l691_691941

-- Define the triangle and basic constraints
variables {a b c p : ℝ}
variables {r_a r_b r_c S : ℝ}

-- Conditions for the problem
def triangle (a b c p S : ℝ) : Prop :=
  p = (a + b + c) / 2 ∧
  S = p ∧
  r_a = p / (p - a) ∧
  r_b = p / (p - b) ∧
  r_c = p / (p - c) ∧
  a ≤ b ∧ b ≤ c

theorem excircle_bounds (a b c p S r_a r_b r_c : ℝ) 
  (h : triangle a b c p S) : 
  (1 < r_a ∧ r_a ≤ 3) ∧
  (2 < r_b ∧ r_b < ∞) ∧
  (3 ≤ r_c ∧ r_c < ∞) :=
by
  cases h with p_eq semi_eq p_def
  sorry

end excircle_bounds_l691_691941


namespace silver_volume_is_approx_66_048_l691_691901

noncomputable def wire_volume
  (d_mm : Float)
  (l_m : Float)
  : Float :=
  let d_cm := d_mm / 10.0
  let r_cm := d_cm / 2.0
  let l_cm := l_m * 100.0
  Float.pi * (r_cm ^ 2) * l_cm

theorem silver_volume_is_approx_66_048 :
  wire_volume 1.0 84.03380995252074 ≈ 66.048 :=
by
  -- This part is to perform the check that the wire volume is approximately 66.048 cm^3
  sorry

end silver_volume_is_approx_66_048_l691_691901


namespace geometric_seq_102nd_term_l691_691507

theorem geometric_seq_102nd_term :
  let a : ℕ := 12 in
  let r : ℕ := -2 in
  12 * (-2) ^ 101 = -2 ^ 101 * 12 :=
by sorry

end geometric_seq_102nd_term_l691_691507


namespace nat_representations_l691_691157

theorem nat_representations (n : ℕ) : 
  (∑ (a : ℕ) in {b : ℕ | b < n}.powerset, finset.card a) = 2^(n-1) - 1 :=
by
  sorry

end nat_representations_l691_691157


namespace find_area_sum_l691_691612

noncomputable def area_of_disks (a b c : ℕ) : ℝ := 12 * π * (a - b * Real.sqrt c)

theorem find_area_sum
    (a b c : ℕ)
    (r_C : ℝ) (h_rC : r_C = 1)
    (r_disk : ℝ) (h_rd : r_disk = 2 - Real.sqrt 3)
    (sum_form : ℝ) (h_sf : sum_form = 12 * π * (7 - 4 * Real.sqrt 3))
    (h_eq : π * (a - b * Real.sqrt c) = sum_form)
    (h_c : ¬ (∃ p : ℕ, prime p ∧ p^2 ∣ c)) :
    a + b + c = 135 :=
sorry

end find_area_sum_l691_691612


namespace whale_population_ratio_l691_691599

theorem whale_population_ratio 
  (W_last : ℕ)
  (W_this : ℕ)
  (W_next : ℕ)
  (h1 : W_last = 4000)
  (h2 : W_next = W_this + 800)
  (h3 : W_next = 8800) :
  (W_this / W_last) = 2 := by
  sorry

end whale_population_ratio_l691_691599


namespace min_quotient_value_l691_691386

-- Definitions for the digits and the constraints.
variable (a b d : ℕ)
variable (h_distinct : a ≠ b ∧ b ≠ d ∧ a ≠ d)
variable (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ d ≠ 0)
variable (h_hundred_not_one : a ≠ 1)
variable (h_units_digit : d = 1)

-- Definition of the three-digit number and the sum of its digits.
def three_digit_number : ℕ := 100 * a + 10 * b + d
def digit_sum : ℕ := a + b + d

-- Definition of the quotient to minimize.
def quotient : ℚ := (three_digit_number a b d) / (digit_sum a b d)

-- Proof statement for finding the minimum value of the quotient.
theorem min_quotient_value (a b d : ℕ) (h_distinct : a ≠ b ∧ b ≠ d ∧ a ≠ d)
    (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ d ≠ 0) (h_hundred_not_one : a ≠ 1) (h_units_digit : d = 1) :
    ∃ min_quot : ℚ, min_quot = 24.25 ∧
    ∀ (a b : ℕ) (h_distinct : a ≠ b ∧ b ≠ d ∧ a ≠ d) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ d ≠ 0)
    (h_hundred_not_one : a ≠ 1) (h_units_digit : d = 1),
    quotient a b d ≥ min_quot :=
begin
  -- placeholder for the proof
  sorry
end

end min_quotient_value_l691_691386


namespace half_abs_diff_squares_l691_691277

theorem half_abs_diff_squares : 
  let a := 20 in
  let b := 15 in
  let square (x : ℕ) := x * x in
  let abs_diff := abs (square a - square b) in
  abs_diff / 2 = 87.5 := 
by
  let a := 20
  let b := 15
  let square (x : ℕ) := x * x
  let abs_diff := abs (square a - square b)
  have h1 : square a = 400 := by native_decide
  have h2 : square b = 225 := by native_decide
  have h3 : abs_diff = abs (400 - 225) := by simp [square, h1, h2]
  have h4 : abs_diff = 175 := by simp [h3]
  have h5 : abs_diff / 2 = 87.5 := by norm_num [h4]
  exact h5

end half_abs_diff_squares_l691_691277


namespace half_abs_diff_squares_l691_691240

theorem half_abs_diff_squares : (1 / 2) * |20^2 - 15^2| = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691240


namespace find_special_sequences_l691_691391

def is_special_sequence (seq : List ℕ) : Prop :=
  ∀ m, seq.count m = seq.getD m 0

theorem find_special_sequences :
  is_special_sequence [1, 2, 1, 0] ∧
  is_special_sequence [2, 0, 2, 0] ∧
  is_special_sequence [2, 1, 2, 0, 0] ∧
  ∀ N > 2, is_special_sequence (List.concat [N, 2, 1] (List.replicate (N + 1) 0)) :=
by
  sorry

end find_special_sequences_l691_691391


namespace intersection_of_A_and_B_l691_691054

def A : Set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ 4 * x + y = 6}
def B : Set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ 3 * x + 2 * y = 7}

theorem intersection_of_A_and_B : (1, 2) ∈ A ∩ B := 
begin
  sorry
end

end intersection_of_A_and_B_l691_691054


namespace initial_jelly_beans_l691_691610

theorem initial_jelly_beans (total_children : ℕ) (percentage : ℕ) (jelly_per_child : ℕ) (remaining_jelly : ℕ) :
  (percentage = 80) → (total_children = 40) → (jelly_per_child = 2) → (remaining_jelly = 36) →
  (total_children * percentage / 100 * jelly_per_child + remaining_jelly = 100) :=
by
  intros h1 h2 h3 h4
  sorry

end initial_jelly_beans_l691_691610


namespace product_num_digits_l691_691795

-- Define the parameters involved in the problem
def three := 3
def six := 6
def eight := 8
def four := 4

-- Define the functions or lemmas to calculate the number of digits
def num_digits (n : ℕ) : ℕ := (Real.log10 n).floor + 1

-- Prove the main theorem
theorem product_num_digits : num_digits (three^eight * six^four) = 7 := by
  sorry

end product_num_digits_l691_691795


namespace correct_statements_l691_691426

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ [0, 2] then log (x + 1) / log 2 else 0  -- providing a default value for other ranges

axiom f_property1 : ∀ x, (f (x - 4) = -f x)
axiom f_property2 : is_odd_function f
axiom f_segment : ∀ x, x ∈ [0, 2] → f x = log (x + 1) / log 2

theorem correct_statements :
  [((f 3 = 1) = true),
   ( (∀ x, x ∈ [-6, -2] → (monotone f x)) = false),
   ( (∀ x, (f (-x + 4) = f (x + 4)) = false)),
   ( (∀ m, m ∈ (0, 1) → (∑ x in (roots (f x - m = 0)) (-8, 8), x) = -8) = true) ] := 
sorry

end correct_statements_l691_691426


namespace tangent_line_at_0_l691_691586

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem tangent_line_at_0 : ∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b ↔ x - y + 1 = 0) ∧ 
    m = (deriv f 0) ∧ b = f 0 :=
by
  have h_deriv : deriv f 0 = 1 := by
    show deriv (λ x, Real.sin x + Real.cos x) 0 = 1
    rw [deriv_add, deriv_sin, deriv_cos]
    norm_num
  have h_f0 : f 0 = 1 := by
    show Real.sin 0 + Real.cos 0 = 1
    rw [Real.sin_zero, Real.cos_zero]
    norm_num
  use 1
  use 1
  split
  { intros x y
    split
    { intro h
      rw [h]
      linarith }
    { intro h
      linarith } }
  { rw [h_deriv, h_f0] }

end tangent_line_at_0_l691_691586


namespace Marty_paint_combination_count_l691_691143

/--
Marty can choose to use either blue, green, yellow, black, or red paint.
He can style the paint by painting with a brush, a roller, or a sponge.
If he uses a sponge, he can only do so with green or yellow paints.
We want to prove that the total number of different combinations of color and painting method Marty has 
is 12.
-/
theorem Marty_paint_combination_count :
  let colors := {blue, green, yellow, black, red}
  let methods := {brush, roller, sponge}
  let restricted_methods := {method : methods | method = sponge}
  let unrestricted_colors := colors \ {green, yellow}
  let total_combinations := (unrestricted_colors.card * ({brush, roller}.card)) + ({green, yellow}.card * (restricted_methods.card))

  total_combinations = 12 :=
by
  -- Define the variables for colors and methods
  let colors := {blue, green, yellow, black, red}
  let methods := {brush, roller, sponge}
  let restricted_methods := {method : methods | method = sponge}
  let unrestricted_colors := colors \ {green, yellow}
  let unrestricted_methods := methods \ restricted_methods

  -- Calculate the unrestricted part
  have h1: unrestricted_colors.card * unrestricted_methods.card = 5 * 2 := by
    unfold unrestricted_colors unrestricted_methods
    -- Show the step for the calculation
    sorry

  -- Calculate the restricted part
  have h2: {green, yellow}.card * restricted_methods.card = 2 * 1 := by
    unfold restricted_methods
    -- Show the step for calculation
    sorry

  -- Sum up both parts
  have h_comb: total_combinations = h1 + h2 := by
    unfold total_combinations
    -- Show the step for calculation
    sorry

  -- Conclude with the final result
  exact h_comb

end Marty_paint_combination_count_l691_691143


namespace vincent_books_cost_l691_691620

theorem vincent_books_cost :
  let num_animals := 10
  let num_outer_space := 1
  let num_trains := 3
  let total_books := num_animals + num_outer_space + num_trains
  let total_spent := 224
  let cost_per_book := total_spent / total_books
  cost_per_book = 16 :=
by
  let num_animals := 10
  let num_outer_space := 1
  let num_trains := 3
  let total_books := num_animals + num_outer_space + num_trains
  let total_spent := 224
  let cost_per_book := total_spent / total_books
  show cost_per_book = 16
  sorry

end vincent_books_cost_l691_691620


namespace average_of_first_six_is_98_l691_691181

def is_average_of_first_six (A : ℝ) : Prop :=
  let sum_first_six := 6 * A in
  let sum_last_six := 6 * 65 in
  let sum_total := 11 * 60 in
  let sixth_number := 318 in
  sum_first_six + sum_last_six - sixth_number = sum_total

theorem average_of_first_six_is_98 :
  (∃ A : ℝ, is_average_of_first_six A ∧ A = 98) :=
  sorry

end average_of_first_six_is_98_l691_691181


namespace equilateral_triangle_circumcircle_area_is_36pi_l691_691718

def is_equilateral (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def tangency_point (r : ℝ) (line : Line) (point : Point) : Prop :=
  -- Assume the tangency condition is defined appropriately
  sorry

noncomputable def circumradius (triangle : Triangle) : ℝ :=
  -- Assume the formula for the circumradius is defined appropriately
  sorry

noncomputable def circumcircle_area (R : ℝ) : ℝ :=
  Real.pi * R^2

theorem equilateral_triangle_circumcircle_area_is_36pi (A B C O : Point) :
  is_equilateral A B C →
  (tangency_point 3 (line_through A B) B ∧ tangency_point 3 (line_through A C) C) →
  circumcircle_area (circumradius ⟨A, B, C⟩) = 36 * Real.pi := by
  sorry

end equilateral_triangle_circumcircle_area_is_36pi_l691_691718


namespace sequence_opposite_signs_l691_691763

theorem sequence_opposite_signs (a : ℕ → ℝ) (m n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ m → a i ≠ 0)
  (h2 : m ≥ 2) (h3 : n < m - 1) (h4 : ∀ k, 0 ≤ k ∧ k ≤ n → ∑ i in range m, (i : ℕ)^k * (a (i + 1)) = 0) :
  ∃ pairs : ℕ, pairs ≥ n + 1 ∧ pairs ≤ m - 1 ∧ ∀ j, 1 ≤ j ∧ j < pairs + 1 → (a j) * (a (j + 1)) < 0 := 
sorry

end sequence_opposite_signs_l691_691763


namespace round_to_nearest_thousandth_l691_691165

theorem round_to_nearest_thousandth : 
  (Real.round (32.67892 * 1000) / 1000 = 32.679) := by
    sorry

end round_to_nearest_thousandth_l691_691165


namespace _l691_691825

noncomputable theorem sugar_flour_ratio (F B : ℝ) (sugar : ℝ) (h1 : F = 10 * B) 
(h2 : F = 8 * (B + 60)) (h3 : sugar = 6000) : sugar / F = 5 / 2 :=
by
  let B1 := 240
  have hB : B = B1, from
    calc
      B = (B - 240 : ℝ) + 240 : by ring
      ... = 0 + 240 : by sorry
  have hF : F = 2400, from
    calc
      F = 10 * 240 : by { rw hB, sorry }
  calc
    sugar / F = 6000 / 2400 : by { rw [h3, hF], sorry }
            ... = 5 / 2 : by norm_num

end _l691_691825


namespace rate_on_simple_interest_l691_691356

theorem rate_on_simple_interest (P A : ℝ) (T : ℕ) (R : ℝ) (hP : P = 25000) (hA : A = 45000) (hT : T = 12) :
  P + (P * R / 100) * T = A → R ≈ 6.67 :=
by
  sorry

end rate_on_simple_interest_l691_691356


namespace correct_calculation_l691_691289

theorem correct_calculation :
  (\(\forall (a b : ℝ), a ≥ 0 ∧ b ≥ 0 → \sqrt {a} + \sqrt {b} ≠ \sqrt {a + b}\)) →  
  (\(∀ (x : ℝ), \sqrt {15} / 3 ≠ \sqrt {5}\)) →  
  (\(√ (2 : ℝ) * \sqrt {3} = \sqrt {6}\)) →  
  (\(forall (y : ℝ), \sqrt {(-5 : ℝ) ^ 2} = 5 ≠ -5\)) →

    true :=
by
  intros h1 h2 h3 h4
  exact true.intro

end correct_calculation_l691_691289


namespace undamaged_triangles_needed_l691_691987

-- Given the conditions summarized from the problem
def side_length_large : ℝ := 20
def side_length_small : ℝ := 2
def damaged_triangles : ℕ := 10

-- We aim to prove that the number of undamaged triangles needed is 90
theorem undamaged_triangles_needed : 
  let area_large := (Real.sqrt 3 / 4) * side_length_large^2
  let area_small := (Real.sqrt 3 / 4) * side_length_small^2
  let total_small_triangles := area_large / area_small
  let undamaged_triangles := total_small_triangles - damaged_triangles
  undamaged_triangles = 90 := by
    sorry

end undamaged_triangles_needed_l691_691987


namespace half_abs_diff_of_squares_l691_691248

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 20) (h2 : b = 15) :
  (1/2 : ℚ) * |(a:ℚ)^2 - (b:ℚ)^2| = 87.5 := by
  sorry

end half_abs_diff_of_squares_l691_691248


namespace exp_pos_for_all_x_l691_691200

theorem exp_pos_for_all_x (h : ¬ ∃ x_0 : ℝ, Real.exp x_0 ≤ 0) : ∀ x : ℝ, Real.exp x > 0 :=
by
  sorry

end exp_pos_for_all_x_l691_691200


namespace smallest_positive_period_f_intervals_monotonically_increasing_f_l691_691779

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) * (Real.sin x + Real.cos x)

-- 1. Proving the smallest positive period is π
theorem smallest_positive_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = Real.pi := 
sorry

-- 2. Proving the intervals where the function is monotonically increasing
theorem intervals_monotonically_increasing_f : 
  ∀ k : ℤ, ∀ x : ℝ, x ∈ Set.Icc (k * Real.pi - (3 * Real.pi / 8)) (k * Real.pi + (Real.pi / 8)) → 
    0 < deriv f x :=
sorry

end smallest_positive_period_f_intervals_monotonically_increasing_f_l691_691779


namespace multiples_of_4_divisible_384_l691_691195

theorem multiples_of_4_divisible_384 :
  ∀ (n : ℕ), ∃ k : ℕ, (∀ m : ℕ, (m = 4 * n) ∨ (m = 4 * (n + 1)) ∨ ... ∨ (m = 4 * (n + k - 1))) →
    4^k * n * (n + 1) * ... * (n + k - 1) % 384 = 0 ∧ k = 7 :=
begin
  sorry
end

end multiples_of_4_divisible_384_l691_691195


namespace half_absolute_difference_of_squares_l691_691269

-- Defining the variables a and b involved in the problem
def a : ℤ := 20
def b : ℤ := 15

-- Statement to prove the solution
theorem half_absolute_difference_of_squares : 
    (1 / 2 : ℚ) * (abs ((a ^ 2) - (b ^ 2))) = 87.5 :=
by
    -- Proof omitted
    sorry

end half_absolute_difference_of_squares_l691_691269


namespace tangent_lines_create_regions_l691_691217

theorem tangent_lines_create_regions (n : ℕ) (h : n = 26) : ∃ k, k = 68 :=
by
  have h1 : ∃ k, k = 68 := ⟨68, rfl⟩
  exact h1

end tangent_lines_create_regions_l691_691217


namespace sqrt_sum_eq_nine_l691_691067

theorem sqrt_sum_eq_nine (x : ℝ) (h : Real.sqrt (7 + x) + Real.sqrt (28 - x) = 9) :
  (7 + x) * (28 - x) = 529 :=
sorry

end sqrt_sum_eq_nine_l691_691067


namespace autograph_possibilities_two_inhabitants_complement_l691_691905

theorem autograph_possibilities :
  let n := 1111
      m := 11
      combinations := 2^m
  in combinations = 2048 :=
begin
  -- proof to be filled in
  sorry
end

theorem two_inhabitants_complement :
  let n := 1111
      m := 11
      combinations := 2^m
      inhabitants : Finset (Fin (2^m)) := Finset.range n
  in ∃ (a b : inhabitants), a ≠ b ∧ ∀ i, (inhabitants a i xor inhabitants b i) = 1 :=
begin
  -- proof to be filled in
  sorry
end

end autograph_possibilities_two_inhabitants_complement_l691_691905


namespace k_values_l691_691780

noncomputable def k_range := {k : ℝ // ∀ x₁ x₂ : ℝ, 
  (x₁ ≠ x₂) → ((exp x₁ - exp x₂) / (x₁ - x₂) < |k| * (exp x₁ + exp x₂))}

theorem k_values :
  k_range = {k : ℝ // k ≤ -1/2 ∨ k ≥ 1/2} :=
sorry

end k_values_l691_691780


namespace lines_parallel_to_same_line_are_parallel_l691_691204

open_locale affine

-- Defining a 3D affine space
variables (V : Type) [AddCommGroup V] [Module ℝ V]
variables (P : Type) [AffineSpace V P] 

-- Definitions of lines being parallel in 3D space
def line_parallel_to (l1 l2 : AffineSpan V P) : Prop :=
∀ p1 p2 : P, p1 ∈ l1 → p2 ∈ l2 → ∃ (v1 v2 : V), 
  (p2 -ᵥ p1 = v2 -ᵥ v1) ∧ p1 +ᵥ v2 ∈ l1 ∧ p1 +ᵥ v1 ∈ l2

-- The theorem to prove the positional relationship
theorem lines_parallel_to_same_line_are_parallel 
  (l1 l2 l : AffineSpan V P)
  (hl1 : ∀ p ∈ l1, ∃ v : V, p +ᵥ v ∈ l)
  (hl2 : ∀ p ∈ l2, ∃ v : V, p +ᵥ v ∈ l) :
  line_parallel_to l1 l2 :=
sorry

end lines_parallel_to_same_line_are_parallel_l691_691204


namespace half_abs_diff_squares_l691_691259

theorem half_abs_diff_squares (a b : ℤ) (h₁ : a = 20) (h₂ : b = 15) : 
  (|a^2 - b^2| / 2 : ℚ) = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691259


namespace half_absolute_difference_of_squares_l691_691268

-- Defining the variables a and b involved in the problem
def a : ℤ := 20
def b : ℤ := 15

-- Statement to prove the solution
theorem half_absolute_difference_of_squares : 
    (1 / 2 : ℚ) * (abs ((a ^ 2) - (b ^ 2))) = 87.5 :=
by
    -- Proof omitted
    sorry

end half_absolute_difference_of_squares_l691_691268


namespace coeff_x2_term_l691_691396

theorem coeff_x2_term : 
  let p1 : ℕ → ℕ := λ n, match n with
    | 0 => 5
    | 1 => 4
    | 2 => 3
    | _ => 0
  let p2 : ℕ → ℕ := λ n, match n with
    | 0 => 8
    | 1 => 7
    | 2 => 6
    | _ => 0
  in (p1 0 * p2 2) + (p1 1 * p2 1) + (p1 2 * p2 0) = 82 := by
  sorry

end coeff_x2_term_l691_691396


namespace calculate_xy_product_l691_691365

theorem calculate_xy_product (a b : ℝ) : 
    let x := a + b 
    let y := a - b in 
    (x - y) * (x + y) = 4 * a * b := by 
  sorry

end calculate_xy_product_l691_691365


namespace positive_difference_two_solutions_abs_eq_15_l691_691281

theorem positive_difference_two_solutions_abs_eq_15 :
  ∀ (x1 x2 : ℝ), (|x1 - 3| = 15) ∧ (|x2 - 3| = 15) ∧ (x1 > x2) → (x1 - x2 = 30) :=
by
  intros x1 x2 h
  sorry

end positive_difference_two_solutions_abs_eq_15_l691_691281


namespace num_multiples_of_6_with_units_digit_6_lt_150_l691_691802

theorem num_multiples_of_6_with_units_digit_6_lt_150 : 
  ∃ (n : ℕ), (n = 5) ∧ (∀ m : ℕ, (m < 150) → (m % 6 = 0) → (m % 10 = 6) → (∃ k : ℕ, m = 6 * (5 * k + 1) ∧ 6 * (5 * k + 1) < 150)) :=
by 
  have multiples : list ℕ := [6, 36, 66, 96, 126] 
  apply Exists.intro 5
  split
  {
    exact rfl,
  }
  {
    intros m h1 h2 h3 
    sorry
  }

end num_multiples_of_6_with_units_digit_6_lt_150_l691_691802


namespace exists_x_y_l691_691158

theorem exists_x_y (n : ℕ) (hn : 0 < n) :
  ∃ x y : ℕ, n < x ∧ ¬ x ∣ y ∧ x^x ∣ y^y :=
by sorry

end exists_x_y_l691_691158


namespace carlos_class_number_l691_691202

theorem carlos_class_number (b : ℕ) :
  (100 < b ∧ b < 200) ∧
  (b + 2) % 4 = 0 ∧
  (b + 3) % 5 = 0 ∧
  (b + 4) % 6 = 0 →
  b = 122 ∨ b = 182 :=
by
  -- The proof implementation goes here
  sorry

end carlos_class_number_l691_691202


namespace sum_x_coords_l691_691665

def segment1 : (ℝ × ℝ) × (ℝ × ℝ) := ((-4, -5), (-2, -1))
def segment2 : (ℝ × ℝ) × (ℝ × ℝ) := ((-2, -1), (-1, -2))
def segment3 : (ℝ × ℝ) × (ℝ × ℝ) := ((-1, -2), (1, 2))
def segment4 : (ℝ × ℝ) × (ℝ × ℝ) := ((1, 2), (2, 1))
def segment5 : (ℝ × ℝ) × (ℝ × ℝ) := ((2, 1), (4, 5))

noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Icc (-4) (-2) then 2 * x + 3
  else if x ∈ Icc (-2) (-1) then -x - 3
  else if x ∈ Icc (-1) 1 then 2 * x
  else if x ∈ Icc 1 2 then -x + 3
  else if x ∈ Icc 2 4 then 2 * x - 3
  else 0

theorem sum_x_coords : 
  let x_intersections := {x : ℝ | f x = x + 2},
      sum_coords := ∑ x in x_intersections, x
  in sum_coords = 0.5 := 
by
  sorry

end sum_x_coords_l691_691665


namespace length_BD_l691_691523

open Real

-- Define the given parameters
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (45, 0)
def C : ℝ × ℝ := (0, 120)
def D : ℝ × ℝ := (0, 0)  -- D is yet unknown, but assume D's y-coordinate is 0 since AD ⊥ BC

-- Define the condition of the right-angled triangle at A
def triangle_right_angled (A B C : ℝ × ℝ) : Prop :=
  let AB := dist A B in
  let AC := dist A C in
  let BC := dist B C in
  AB^2 + AC^2 = BC^2

-- Main theorem to prove
theorem length_BD :
  (AB = 45) →
  (AC = 120) →
  (AD ⊥ BC) →
  (triangle_right_angled A B C) →
  dist B D = 80 * sqrt 2 :=
by
  sorry

end length_BD_l691_691523


namespace range_of_f_range_of_a_l691_691428

noncomputable def f (x : ℝ) : ℝ := 3 * |x - 1| + |3 * x + 1|
noncomputable def g (x a : ℝ) : ℝ := |x + 2| + |x - a|
def A : set ℝ := {y | ∃ x, y = f x}
def B (a : ℝ) : set ℝ := {y | ∃ x, y = g x a}

theorem range_of_f : A = {y | 4 ≤ y} := by
  sorry

theorem range_of_a (a : ℝ) : A ∪ B a = B a → -6 ≤ a ∧ a ≤ 2 := by
  have range_of_f : A = {y | 4 ≤ y} := sorry
  sorry

end range_of_f_range_of_a_l691_691428


namespace operation_1_2010_l691_691715

def operation (m n : ℕ) : ℕ := sorry

axiom operation_initial : operation 1 1 = 2
axiom operation_step (m n : ℕ) : operation m (n + 1) = operation m n + 3

theorem operation_1_2010 : operation 1 2010 = 6029 := sorry

end operation_1_2010_l691_691715


namespace choose_sphere_tetrahedron_plane_to_intersect_in_equal_areas_l691_691535

noncomputable def equal_intersection_areas
  (AB CD : Line)
  (I : Line)
  (S : ℝ)
  (r : ℝ)
  (hTetrahedron : Tetrahedron)
  (hSphere : Sphere) : Prop :=
  let M_AB := midpoint AB
  let M_CD := midpoint CD
  let hI := line_segment I M_AB M_CD
  let hRect := rectangle_area (cross_section hTetrahedron M_AB M_CD)
  let r := real.sqrt (S / real.pi)
  let hLengthI := length I = 2 * r
  in
  ∀ (plane : Plane), is_parallel plane (reference_plane M_AB M_CD) → 
    (area (plane ∩ hTetrahedron) = area (plane ∩ hSphere))

theorem choose_sphere_tetrahedron_plane_to_intersect_in_equal_areas 
  (AB CD : Line)
  (I : Line)
  (S : ℝ)
  (r : ℝ)
  (hTetrahedron : Tetrahedron)
  (hSphere : Sphere) 
  (hAB_CD_perpendicular : is_perpendicular AB CD)
  (hI_midpoints : I = line_segment (midpoint AB) (midpoint CD))
  (hRect_area : rectangle_area (cross_section hTetrahedron (midpoint AB) (midpoint CD)) = S)
  (hradius : r = real.sqrt (S / real.pi))
  (hI_length : length I = 2 * r)
  : equal_intersection_areas AB CD I S r hTetrahedron hSphere := 
  sorry

end choose_sphere_tetrahedron_plane_to_intersect_in_equal_areas_l691_691535


namespace beth_sells_half_of_coins_l691_691358

theorem beth_sells_half_of_coins (x y : ℕ) (h₁ : x = 125) (h₂ : y = 35) : (x + y) / 2 = 80 :=
by
  sorry

end beth_sells_half_of_coins_l691_691358


namespace half_abs_diff_squares_l691_691244

theorem half_abs_diff_squares : (1 / 2) * |20^2 - 15^2| = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691244


namespace countDivisors_180_l691_691705

def countDivisors (n : ℕ) : ℕ :=
  let factors := n.factorization
  factors.fold (λ acc p e => acc * (e + 1)) 1

theorem countDivisors_180 : countDivisors 180 = 18 := by
  sorry

end countDivisors_180_l691_691705


namespace triangle_properties_l691_691022

/-- Given a triangle ABC with sides a, b, c corresponding to angles A, B, C respectively.
    Given the conditions a = 1, b = 2, and cos C = 1/4, we aim to prove:
    1. The perimeter of the triangle is 5.
    2. The value of cos A is 7/8. -/
theorem triangle_properties (a b c : ℝ) (cosC cosA : ℝ)
  (ha : a = 1)
  (hb : b = 2)
  (hcosC : cosC = 1/4) :
  (c^2 = a^2 + b^2 - 2 * a * b * cosC ∧
   c = real.sqrt 4 ∧
   a + b + c = 5 ∧
   cosA = (b^2 + c^2 - a^2) / (2 * b * c) ∧
   cosA = 7/8) := sorry

end triangle_properties_l691_691022


namespace carpet_needed_correct_l691_691334

def length_room : ℕ := 15
def width_room : ℕ := 9
def length_closet : ℕ := 3
def width_closet : ℕ := 2

def area_room : ℕ := length_room * width_room
def area_closet : ℕ := length_closet * width_closet
def area_to_carpet : ℕ := area_room - area_closet
def sq_ft_to_sq_yd (sqft: ℕ) : ℕ := (sqft + 8) / 9  -- Adding 8 to ensure proper rounding up

def carpet_needed : ℕ := sq_ft_to_sq_yd area_to_carpet

theorem carpet_needed_correct :
  carpet_needed = 15 := by
  sorry

end carpet_needed_correct_l691_691334


namespace purple_balls_count_l691_691652

theorem purple_balls_count (k : ℕ) (h_pos : k > 0) : 
    let expected_value := ((5 : ℚ) / (5 + k) * 2) + (k / (5 + k) * -2) in 
    expected_value = 1 / 2 → k = 3 :=
by
  intros expected_value_def h_expected_value_eq
  sorry

end purple_balls_count_l691_691652


namespace optimal_pricing_for_max_profit_l691_691659

noncomputable def sales_profit (x : ℝ) : ℝ :=
  -5 * x^3 + 45 * x^2 - 75 * x + 675

theorem optimal_pricing_for_max_profit :
  ∃ x : ℝ, 0 ≤ x ∧ x < 9 ∧ ∀ y : ℝ, 0 ≤ y ∧ y < 9 → sales_profit y ≤ sales_profit 5 ∧ (14 - 5 = 9) :=
by
  sorry

end optimal_pricing_for_max_profit_l691_691659


namespace p_sufficient_not_necessary_q_l691_691861

open Function

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) : ℝ := x^2 - m * x

-- Define condition p where f(x) is monotonically increasing in (0, ∞)
def p (m : ℝ) : Prop := ∀ x y : ℝ, 0 < x → x < y → 0 < y → f(x, m) ≤ f(y, m)

-- Define condition q where m ≤ 1/2
def q (m : ℝ) : Prop := m ≤ 1 / 2

-- Define the theorem which states that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_q (m : ℝ) : (p m) → (q m) ∧ ((¬ q m) → ¬ (p m)) := by
  sorry

end p_sufficient_not_necessary_q_l691_691861


namespace pi_floor_is_3_l691_691810

-- Define the floor function representing the greatest integer not exceeding x
def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem to prove
theorem pi_floor_is_3 : floor Real.pi = 3 :=
sorry

end pi_floor_is_3_l691_691810


namespace intersection_A_B_l691_691472

def A : set ℝ := {x | x ≤ 3}

def B : set ℕ := {x | x ≥ 1}

theorem intersection_A_B :
  (A ∩ (B : set ℝ)) = ({1, 2, 3} : set ℝ) :=
by
  sorry

end intersection_A_B_l691_691472


namespace value_of_expression_l691_691081

theorem value_of_expression (x y : ℚ) (hx : x = -5/4) (hy : y = -3/2) : -2 * x - y ^ 2 = 1 / 4 :=
by {
  -- Substitute values of x and y
  rw [hx, hy],
  -- Simplify the expression
  calc 
    -2 * (-5 / 4) - (-3 / 2) ^ 2 = (2 * 5 / 4) - ((3 / 2) ^ 2)  : by ring
    ...                           = 5 / 2 - (9 / 4)              : by ring
    ...                           = 10 / 4 - 9 / 4               : by norm_num
    ...                           = 1 / 4                        : by norm_num
}

end value_of_expression_l691_691081


namespace smallest_prime_with_conditions_l691_691402

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n
def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def reverse_digits (n : ℕ) : ℕ := 
  let tens := n / 10 
  let units := n % 10 
  units * 10 + tens

theorem smallest_prime_with_conditions : 
  ∃ (p : ℕ), is_prime p ∧ 20 ≤ p ∧ p < 30 ∧ (reverse_digits p) < 100 ∧ is_composite (reverse_digits p) ∧ p = 23 :=
by
  sorry

end smallest_prime_with_conditions_l691_691402


namespace cone_lateral_surface_area_ratio_l691_691080

theorem cone_lateral_surface_area_ratio (r l S_lateral S_base : ℝ) (h1 : l = 3 * r)
  (h2 : S_lateral = π * r * l) (h3 : S_base = π * r^2) :
  S_lateral / S_base = 3 :=
by
  sorry

end cone_lateral_surface_area_ratio_l691_691080


namespace problem_q_value_l691_691044

theorem problem_q_value (p q : ℝ) (hpq : 1 < p ∧ p < q) 
  (h1 : 1 / p + 1 / q = 1) 
  (h2 : p * q = 8) : 
  q = 4 + 2 * real.sqrt 2 :=
by sorry

end problem_q_value_l691_691044


namespace find_f_of_f_l691_691449

def f (x : ℝ) : ℝ :=
  if |x| ≤ 1 then |x - 1| - 2 else 1 / (1 + x^2)

theorem find_f_of_f :
  f (f (1/2)) = 4 / 13 :=
by
  -- Proof goes here
  sorry

end find_f_of_f_l691_691449


namespace problem_statement_l691_691452

variable {α : Type*} [LinearOrder α] [AddCommGroup α] [Nontrivial α]

def is_monotone_increasing (f : α → α) (s : Set α) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x ≤ f y

theorem problem_statement (f : ℝ → ℝ) (x1 x2 : ℝ)
  (h1 : ∀ x, f (-x) = -f (x + 4))
  (h2 : is_monotone_increasing f {x | x > 2})
  (hx1 : x1 < 2) (hx2 : 2 < x2) (h_sum : x1 + x2 < 4) :
  f (x1) + f (x2) < 0 :=
sorry

end problem_statement_l691_691452


namespace count_multiples_of_6_with_units_digit_6_l691_691801

theorem count_multiples_of_6_with_units_digit_6 : 
  ∃ (n : ℕ), n = 5 ∧ ∀ (m : ℕ), (m > 0 ∧ m < 150 ∧ m % 6 = 0 ∧ m % 10 = 6) ↔ m ∈ {6, 36, 66, 96, 126} :=
by
  sorry

end count_multiples_of_6_with_units_digit_6_l691_691801


namespace inequality_for_pos_reals_l691_691741

open Real Nat

theorem inequality_for_pos_reals
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (h : 1/a + 1/b = 1)
  (n : ℕ) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n + 1) :=
by 
  sorry

end inequality_for_pos_reals_l691_691741


namespace sequence_inequality_l691_691946

section
variable {n : ℕ} (x y : Fin n → ℝ)

def is_sequence_of_length_n (a : Fin n → ℝ) : Prop :=
∀ i : Fin n, ∑ j in Fin.range (i+1), a j / (i + 1).1 ≥ ∑ j in Fin.range (i+2), a j / (i + 2).1

theorem sequence_inequality (hx : ∀ i, 0 ≤ x i)
  (hy : ∀ i, 0 ≤ y i)
  (hxl : is_sequence_of_length_n x)
  (hyl : is_sequence_of_length_n y) :
  ∑ i, x i * y i ≥ (1 : ℝ) / n * (∑ i, x i) * (∑ i, y i) := by
  sorry
end

end sequence_inequality_l691_691946


namespace problem1_problem2_l691_691645

-- First problem: Prove cos(α + β) = -1
theorem problem1 (α β : ℝ)
  (h1 : sin α = 3 / 5)
  (h2 : cos β = 4 / 5)
  (h3 : α ∈ Ioc (π / 2) π)
  (h4 : β ∈ Ioc 0 (π / 2)) :
  cos (α + β) = -1 :=
sorry

-- Second problem: Prove β = π / 3
theorem problem2 (α β : ℝ)
  (h1 : cos α = 1 / 7)
  (h2 : cos (α - β) = 13 / 14)
  (h3 : 0 < β ∧ β < α ∧ α < π / 2) :
  β = π / 3 :=
sorry

end problem1_problem2_l691_691645


namespace solution_l691_691785

theorem solution (x : ℝ) : (x = -2/5) → (x < x^3 ∧ x^3 < x^2) :=
by
  intro h
  rw [h]
  -- sorry to skip the proof
  sorry

end solution_l691_691785


namespace combined_height_l691_691876

/-- Given that Mr. Martinez is two feet taller than Chiquita and Chiquita is 5 feet tall, prove that their combined height is 12 feet. -/
theorem combined_height (h_chiquita : ℕ) (h_martinez : ℕ) 
  (h1 : h_chiquita = 5) (h2 : h_martinez = h_chiquita + 2) : 
  h_chiquita + h_martinez = 12 :=
by sorry

end combined_height_l691_691876


namespace length_breadth_difference_l691_691179

theorem length_breadth_difference (b l : ℕ) (h1 : b = 5) (h2 : l * b = 15 * b) : l - b = 10 :=
by
  sorry

end length_breadth_difference_l691_691179


namespace measure_F_correct_l691_691533

noncomputable def measure_F (D E F : ℝ) (h : D + E + F = 180) (hD : D = 74) (hE : E = 4 * F + 18) : ℝ :=
  F

theorem measure_F_correct : ∃ F : ℝ, measure_F 74 (4 * F + 18) F (by linarith) (by rfl) (by linarith) = 17.6 :=
sorry

end measure_F_correct_l691_691533


namespace tangent_line_eq_l691_691910

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 2

def point := (Real.pi / 4, 1 / 2 : ℝ × ℝ)

theorem tangent_line_eq :
  let x := Real.pi / 4
  let y := 1 / 2
  let slope := -2 * Real.sin x * Real.cos x
  ∃ m b : ℝ, tangent_line := (m = slope) ∧ (f x = y) ∧ (b = y - m * x) ∧ (∀ x1 : ℝ, ∀ y1 : ℝ, (y1 = f x1) → y1 = slope * (x1 - x) + y) :=
  ∃ m b : ℝ, m = -1 ∧ (b = slope - m * x) ∧ (tangent_line = "x + y - (1 / 2) - (Real.pi / 4) = 0") 
:=
begin
  let slope := -2 * sin (pi / 4) * cos (pi / 4),
  have slope_eval : slope = -1, 
    -- add proof steps here, omitted for brevity.
    sorry,
  exact ⟨-1, 1/2 + pi / 4, slope_eval, _⟩,
  sorry
end

end tangent_line_eq_l691_691910


namespace days_b_worked_l691_691296

-- Definitions of the conditions
def a_days (a : ℕ) : Prop := a = 6
def c_days (c : ℕ) : Prop := c = 4
def daily_wage_ratio (x : ℕ) (wage_a wage_b wage_c : ℕ) : Prop :=
  wage_a = 3 * x ∧ wage_b = 4 * x ∧ wage_c = 5 * x
def wage_c (wage_c : ℕ) : Prop := wage_c = 125
def total_earnings (earnings_a earnings_b earnings_c total : ℕ) : Prop :=
  earnings_a + earnings_b + earnings_c = total ∧ total = 1850

-- Theorem to be proved
theorem days_b_worked (d a c x wage_a wage_b wage_c earnings_a earnings_c : ℕ)
  (ha : a_days a) (hc : c_days c) (hr : daily_wage_ratio x wage_a wage_b wage_c)
  (wage_of_c : wage_c wage_c) (he : total_earnings (6 * wage_a) (d * wage_b) (4 * wage_c) 1850) :
  d = 9 := sorry

end days_b_worked_l691_691296


namespace chord_of_ellipse_bisected_by_point_l691_691771

theorem chord_of_ellipse_bisected_by_point :
  ∀ (x y : ℝ),
  (∃ (x₁ x₂ y₁ y₂ : ℝ), 
    ( (x₁ + x₂) / 2 = 4 ∧ (y₁ + y₂) / 2 = 2) ∧ 
    (x₁^2 / 36 + y₁^2 / 9 = 1) ∧ 
    (x₂^2 / 36 + y₂^2 / 9 = 1)) →
  (x + 2 * y = 8) :=
by
  sorry

end chord_of_ellipse_bisected_by_point_l691_691771


namespace barbara_wins_2023_coins_l691_691512

/-- 
  In a game where Barbara and Jenna remove coins from a table,
  Barbara can remove either 3 or 5 coins on her turn, and Jenna
  can remove either 2 or 4 coins on her turn. The game starts 
  with Barbara's turn unless a random coin flip decides otherwise.
  Whoever takes the last coin wins. Given both players use their 
  best strategic moves, prove that Barbara will win if the game 
  starts with 2023 coins.
-/
theorem barbara_wins_2023_coins :
  ∀ (n : ℕ), n = 2023 → barbara_wins n :=
begin
  sorry
end

end barbara_wins_2023_coins_l691_691512


namespace tiling_2xn_strip_l691_691516

def phi : ℝ := (1 + Real.sqrt 5) / 2
def psi : ℝ := (1 - Real.sqrt 5) / 2

def fib_sol (n : ℕ) : ℝ :=
  (1 / Real.sqrt 5) * (phi ^ (n + 1) - psi ^ (n + 1))

def F : ℕ → ℝ
| 0     := 1
| 1     := 1
| (n+2) := F (n+1) + F n

theorem tiling_2xn_strip (n : ℕ) : F n = fib_sol n :=
sorry

end tiling_2xn_strip_l691_691516


namespace sin_theta_values_l691_691421

theorem sin_theta_values (θ : ℝ) :
    let P := λ x : ℝ, sin θ * x^2 + (cos θ + tan θ) * x + 1 in
    (∃ x : ℝ, P x = 0) ∧ (∃! x : ℝ, P x = 0) :=
    sin θ = 0 ∨ sin θ = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end sin_theta_values_l691_691421


namespace tan_pi_over_4_plus_alpha_eq_two_l691_691447

theorem tan_pi_over_4_plus_alpha_eq_two
  (α : ℂ) 
  (h : Complex.tan ((π / 4) + α) = 2) : 
  (1 / (2 * Complex.sin α * Complex.cos α + (Complex.cos α)^2)) = (2 / 3) :=
by
  sorry

end tan_pi_over_4_plus_alpha_eq_two_l691_691447


namespace half_abs_diff_squares_l691_691261

theorem half_abs_diff_squares (a b : ℤ) (h₁ : a = 20) (h₂ : b = 15) : 
  (|a^2 - b^2| / 2 : ℚ) = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691261


namespace factorization_of_expression_l691_691721

-- Define variables
variables {a x y : ℝ}

-- State the problem
theorem factorization_of_expression : a * x^2 - 4 * a * y^2 = a * (x + 2 * y) * (x - 2 * y) :=
  sorry

end factorization_of_expression_l691_691721


namespace mean_is_2_l691_691012

variables (x1 x2 x3 x4 : ℝ)
variables (h_pos1 : 0 < x1) (h_pos2 : 0 < x2) (h_pos3 : 0 < x3) (h_pos4 : 0 < x4)
variables (s2 : ℝ)
variables (h_variance : s2 = (1/4) * (x1^2 + x2^2 + x3^2 + x4^2 - 16))

def mean_x {x1 x2 x3 x4 : ℝ} :=
  (x1 + x2 + x3 + x4) / 4

theorem mean_is_2 (x1 x2 x3 x4 : ℝ) (h_pos1 : 0 < x1) (h_pos2 : 0 < x2) (h_pos3 : 0 < x3) (h_pos4 : 0 < x4)
(s2 : ℝ) (h_variance : s2 = (1/4) * (x1^2 + x2^2 + x3^2 + x4^2 - 16)) :
  mean_x x1 x2 x3 x4 = 2 :=
sorry

end mean_is_2_l691_691012


namespace card_arrangement_l691_691604

theorem card_arrangement : 
  ∃ (f : fin 20 → nat), 
  (∀ i, 1 ≤ f i ∧ f i ≤ 20) ∧
  function.injective f :=
begin
  sorry
end

end card_arrangement_l691_691604


namespace half_abs_diff_squares_l691_691273

theorem half_abs_diff_squares : 
  let a := 20 in
  let b := 15 in
  let square (x : ℕ) := x * x in
  let abs_diff := abs (square a - square b) in
  abs_diff / 2 = 87.5 := 
by
  let a := 20
  let b := 15
  let square (x : ℕ) := x * x
  let abs_diff := abs (square a - square b)
  have h1 : square a = 400 := by native_decide
  have h2 : square b = 225 := by native_decide
  have h3 : abs_diff = abs (400 - 225) := by simp [square, h1, h2]
  have h4 : abs_diff = 175 := by simp [h3]
  have h5 : abs_diff / 2 = 87.5 := by norm_num [h4]
  exact h5

end half_abs_diff_squares_l691_691273


namespace find_original_number_l691_691326

theorem find_original_number (x : ℕ) (h : 3 * (2 * x + 9) = 57) : x = 5 := 
by sorry

end find_original_number_l691_691326


namespace water_volume_correct_l691_691167

def total_initial_solution : ℚ := 0.08 + 0.04 + 0.02
def fraction_water_in_initial : ℚ := 0.04 / total_initial_solution
def desired_total_volume : ℚ := 0.84
def required_water_volume : ℚ := desired_total_volume * fraction_water_in_initial

theorem water_volume_correct : 
  required_water_volume = 0.24 :=
by
  -- The proof is omitted
  sorry

end water_volume_correct_l691_691167


namespace simplest_square_root_among_choices_l691_691630

variable {x : ℝ}

def is_simplest_square_root (n : ℝ) : Prop :=
  ∀ m, (m^2 = n) → (m = n)

theorem simplest_square_root_among_choices :
  is_simplest_square_root 7 ∧ ∀ n, n = 24 ∨ n = 1/3 ∨ n = 0.2 → ¬ is_simplest_square_root n :=
by
  sorry

end simplest_square_root_among_choices_l691_691630


namespace reflections_concyclic_l691_691543

theorem reflections_concyclic 
  (A B C D E : Type)
  [ConvexQuadrilateral A B C D]
  (h1 : IntersectsAtRightAngles A C B D E) : 
  Concyclic (Reflection E A B) (Reflection E B C) (Reflection E C D) (Reflection E D A) := 
sorry

end reflections_concyclic_l691_691543


namespace half_abs_diff_squares_l691_691276

theorem half_abs_diff_squares : 
  let a := 20 in
  let b := 15 in
  let square (x : ℕ) := x * x in
  let abs_diff := abs (square a - square b) in
  abs_diff / 2 = 87.5 := 
by
  let a := 20
  let b := 15
  let square (x : ℕ) := x * x
  let abs_diff := abs (square a - square b)
  have h1 : square a = 400 := by native_decide
  have h2 : square b = 225 := by native_decide
  have h3 : abs_diff = abs (400 - 225) := by simp [square, h1, h2]
  have h4 : abs_diff = 175 := by simp [h3]
  have h5 : abs_diff / 2 = 87.5 := by norm_num [h4]
  exact h5

end half_abs_diff_squares_l691_691276


namespace num_multiples_of_6_with_units_digit_6_lt_150_l691_691804

theorem num_multiples_of_6_with_units_digit_6_lt_150 : 
  ∃ (n : ℕ), (n = 5) ∧ (∀ m : ℕ, (m < 150) → (m % 6 = 0) → (m % 10 = 6) → (∃ k : ℕ, m = 6 * (5 * k + 1) ∧ 6 * (5 * k + 1) < 150)) :=
by 
  have multiples : list ℕ := [6, 36, 66, 96, 126] 
  apply Exists.intro 5
  split
  {
    exact rfl,
  }
  {
    intros m h1 h2 h3 
    sorry
  }

end num_multiples_of_6_with_units_digit_6_lt_150_l691_691804


namespace mike_pays_200_l691_691556

def xray_cost : ℕ := 250
def mri_cost := 3 * xray_cost
def total_cost := xray_cost + mri_cost
def insurance_coverage := (80 / 100 : ℝ) * total_cost
def mike_payment := total_cost - insurance_coverage

theorem mike_pays_200 : mike_payment = 200 := by
  sorry

end mike_pays_200_l691_691556


namespace area_less_than_circumference_l691_691991

-- Conditions
def sum_of_dice (d1 d2 : ℕ) : ℕ := d1 + d2

def probability (event : set (ℕ × ℕ)) : ℚ :=
  (event.card : ℚ) / 36

-- The formalized problem statement
theorem area_less_than_circumference (d1 d2 : ℕ) (h1 : 1 ≤ d1) (h2 : d1 ≤ 6) (h3 : 1 ≤ d2) (h4 : d2 ≤ 6) :
  probability {pair | let d := sum_of_dice pair.1 pair.2 in d < 4} = 1 / 12 :=
sorry

end area_less_than_circumference_l691_691991


namespace value_of_f_g_of_5_l691_691073

def g (x : ℤ) : ℤ := 4 * x + 6
def f (x : ℤ) : ℤ := x^2 - 4 * x - 5

theorem value_of_f_g_of_5 : f (g 5) = 567 := by
  sorry

end value_of_f_g_of_5_l691_691073


namespace sum_result_l691_691862

noncomputable def sumInf (x : ℝ) : ℝ :=
  ∑' n, 1 / (x^(2^n) + x^(-2^n))

theorem sum_result (x : ℝ) (h : x > 1) : sumInf x = 1 / (x + 1) := 
by
  sorry

end sum_result_l691_691862


namespace half_abs_diff_of_squares_l691_691247

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 20) (h2 : b = 15) :
  (1/2 : ℚ) * |(a:ℚ)^2 - (b:ℚ)^2| = 87.5 := by
  sorry

end half_abs_diff_of_squares_l691_691247


namespace total_price_of_books_l691_691229

theorem total_price_of_books
  (total_books : ℕ)
  (math_books_cost : ℕ)
  (history_books_cost : ℕ)
  (math_books_bought : ℕ)
  (total_books_eq : total_books = 80)
  (math_books_cost_eq : math_books_cost = 4)
  (history_books_cost_eq : history_books_cost = 5)
  (math_books_bought_eq : math_books_bought = 10) :
  (math_books_bought * math_books_cost + (total_books - math_books_bought) * history_books_cost = 390) := 
by
  sorry

end total_price_of_books_l691_691229


namespace Sally_total_spending_l691_691166

theorem Sally_total_spending (peaches_after_coupon cherries total: ℝ) (coupon: ℝ) 
  (h1: peaches_after_coupon = 12.32) 
  (h2: cherries = 11.54)
  (h3: coupon = 3) 
  (h_total: total = peaches_after_coupon + cherries) : 
  total = 23.86 :=
by 
  rw [h1, h2, h3]
  simp [h_total]
  sorry

end Sally_total_spending_l691_691166


namespace number_of_paths_l691_691945

theorem number_of_paths (m n : ℕ) : 
  (finset.card (finset.range (m + n)).choose m) = nat.choose (m + n) m := by
sorry

end number_of_paths_l691_691945


namespace complex_fraction_simplification_l691_691366

theorem complex_fraction_simplification (a b c d : ℂ) (h₁ : a = 3 + i) (h₂ : b = 1 + i) (h₃ : c = 1 - i) (h₄ : d = 2 - i) : (a / b) = d := by
  sorry

end complex_fraction_simplification_l691_691366


namespace savings_left_after_purchases_l691_691569

noncomputable def initial_savings : ℝ := 200
noncomputable def earrings_price : ℝ := 20
noncomputable def necklace_price : ℝ := 42
noncomputable def bracelet_price : ℝ := 30
noncomputable def jewelry_set_price : ℝ := 70
noncomputable def discount_rate : ℝ := 0.25
noncomputable def sales_tax_rate : ℝ := 0.05
noncomputable def exchange_rate : ℝ := 0.85

theorem savings_left_after_purchases :
  let total_cost_before_discount := earrings_price + necklace_price + bracelet_price + jewelry_set_price in
  let discount := discount_rate * jewelry_set_price in
  let total_cost_after_discount := total_cost_before_discount - discount in
  let total_cost_with_tax := total_cost_after_discount * (1 + sales_tax_rate) in
  let total_cost_in_dollars := total_cost_with_tax / exchange_rate in
  initial_savings - total_cost_in_dollars = 21.45 :=
by
  /- proof steps omitted, sorry used to skip -/
  sorry

end savings_left_after_purchases_l691_691569


namespace compare_y_coordinates_l691_691024

theorem compare_y_coordinates (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁: (x₁ = -3) ∧ (y₁ = 2 * x₁ - 1)) 
  (h₂: (x₂ = -5) ∧ (y₂ = 2 * x₂ - 1)) : 
  y₁ > y₂ := 
by 
  sorry

end compare_y_coordinates_l691_691024


namespace complex_sum_value_l691_691380

-- Definition for the cyclic nature of i powers
def cyclic_powers (n : ℕ) : ℂ :=
  match n % 4 with
  | 0 => 1
  | 1 => complex.I
  | 2 => -1
  | 3 => -complex.I
  | _ => 0  -- this case should never be needed

-- Definition for specific sine function values
def sine_values (θ : ℝ) : ℂ :=
  if θ % 360 = 30 then 1 / 2
  else if θ % 360 = 150 then 1 / 2
  else if θ % 360 = 270 then 0
  else if θ % 360 = 390 then 1 / 2
  else 0  -- default case, should not occur with given angles

-- Problem statement: Sum of sin and cyclic powers of i equals 5/2
theorem complex_sum_value :
  ∑ n in finset.range 21, cyclic_powers n * sine_values (30 + 120 * n) = 5 / 2 :=
by sorry

end complex_sum_value_l691_691380


namespace connected_circles_have_eulerian_path_l691_691838

noncomputable def connected_circles_follow_eulerian_path : Prop :=
  ∀ (G : SimpleGraph ℝ), connected G ∧ (∀ v : G.V, even (G.degree v)) →
  (∃ p : G.path, Eulerian_path p)

theorem connected_circles_have_eulerian_path 
  (G : SimpleGraph ℝ)
  (h_connected : connected G)
  (h_degree_even : ∀ v : G.V, even (G.degree v)) :
  ∃ p : G.path, Eulerian_path p :=
sorry

end connected_circles_have_eulerian_path_l691_691838


namespace area_triangle_ABC_extension_l691_691970

open_locale big_operators

variables {A B C A1 B1 C1 : Type} [add_group A] [module ℝ A] [add_group B] [module ℝ B]
          [add_group C] [module ℝ C] [add_group A1] [module ℝ A1] [add_group B1] [module ℝ B1]
          [add_group C1] [module ℝ C1]

noncomputable def vector_relationships (A B C A1 B1 C1 : Type) [add_group A] [module ℝ A]
  [add_group B] [module ℝ B] [add_group C] [module ℝ C]
  [add_group A1] [module ℝ A1] [add_group B1] [module ℝ B1]
  [add_group C1] [module ℝ C1] :=
  ∃ (O : A →+ B →+ C →+ A1), 
    (O (A.B1) = 2 • O (A.B)) ∧ (O (B.C1) = 2 • O (B.C)) ∧
    (O (C.A1) = 2 • O (C.A))

variables {S : ℝ}

-- Prove that the area of triangle A1B1C1 is 7S, given the specified conditions.
theorem area_triangle_ABC_extension
  (h1 : ∃ (O : A →+ B →+ C →+ A1), (O (A.B1) = 2 • O (A.B)) ∧
   (O (B.C1) = 2 • O (B.C)) ∧ (O (C.A1) = 2 • O (C.A)))
  (h2 : ℝ) (hS : area_triangle ABC = S) :
  area_triangle A1B1C1 = 7 * S :=
sorry

end area_triangle_ABC_extension_l691_691970


namespace amgm_inequality_proof_l691_691561

noncomputable def amgm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : Prop :=
  1 < (a / (Real.sqrt (a^2 + b^2))) + (b / (Real.sqrt (b^2 + c^2))) + (c / (Real.sqrt (c^2 + a^2))) 
  ∧ (a / (Real.sqrt (a^2 + b^2))) + (b / (Real.sqrt (b^2 + c^2))) + (c / (Real.sqrt (c^2 + a^2))) 
  ≤ (3 * Real.sqrt 2) / 2

theorem amgm_inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  amgm_inequality a b c ha hb hc := 
sorry

end amgm_inequality_proof_l691_691561


namespace a_200_correct_l691_691927

def a_seq : ℕ → ℕ
| 0     := 1
| (n+1) := a_seq n + 2 * a_seq n / n

theorem a_200_correct : a_seq 200 = 20100 := 
by sorry

end a_200_correct_l691_691927


namespace binom_26_7_l691_691017

theorem binom_26_7 
  (h₁ : nat.choose 24 5 = 42504) 
  (h₂ : nat.choose 24 6 = 134596) 
  (h₃ : nat.choose 24 7 = 346104) : 
  nat.choose 26 7 = 657800 := 
by 
  sorry

end binom_26_7_l691_691017


namespace correct_propositions_l691_691349

-- Define the propositions
def prop1 (P : Type) [Prism P] (cutPlane : Plane) : Prop :=
  ¬ (∀ part1 part2 : P, cut(P, cutPlane) = (part1, part2) → Prism part1 ∧ Prism part2)

def prop2 (G : Type) [GeometricBody G] : Prop :=
  (parallel_faces G = 2 ∧ all_other_faces G Trapezoids) → frustum G

def prop3 (C : Type) [Cone C] (cutPlane : Plane) : Prop :=
  (parallel_to_base C cutPlane) ∧ (GeometricBody_formed C cutPlane base = TruncatedCone)

def prop4 (G : Type) [GeometricBody G] : Prop :=
  (parallel_faces G = 2 ∧ all_other_faces G Parallelograms) → Prism G

-- Define the main goal theorem
theorem correct_propositions {P : Type} [Prism P] {G : Type} [GeometricBody G] {C : Type} [Cone C] 
{cutPlane : Plane} :
  prop1 P cutPlane ∧ ¬ prop2 G ∧ ¬ prop3 C cutPlane ∧ ¬ prop4 G :=
by
  sorry

end correct_propositions_l691_691349


namespace cost_of_old_car_l691_691689

theorem cost_of_old_car (C_old C_new : ℝ): 
  C_new = 2 * C_old → 
  1800 + 2000 = C_new → 
  C_old = 1900 :=
by
  intros H1 H2
  sorry

end cost_of_old_car_l691_691689


namespace positive_difference_median_mode_l691_691949

def stem_leaf_plot : list (ℕ × list ℕ) := [
  (4, [7, 8, 8, 9, 9]),
  (5, [0, 1, 1, 1, 2]),
  (6, [5, 6, 7, 8, 9]),
  (7, [1, 2, 3, 3, 4]),
  (8, [0, 2, 5, 7, 9])
]

noncomputable def stem_leaf_data : list ℕ := 
  stem_leaf_plot.bind (λ (x : ℕ × list ℕ), x.2.map (λ (y : ℕ), x.1 * 10 + y))

noncomputable def median : ℕ := (stem_leaf_data.sort.nth 12).getD 0
noncomputable def mode : ℕ :=
  let freq_map := stem_leaf_data.foldl (λ (m : NatMap ℕ), λ (x : ℕ), m.insert x ((m.find x).getD 0 + 1)) ∅ in
  let max_freq := freq_map.foldl (λ (acc : ℕ × ℕ) (k v : ℕ), if v > acc.2 then (k, v) else acc) (0, 0) in
  max_freq.1

theorem positive_difference_median_mode :
  (if median > mode then median - mode else mode - median) = 16 :=
by
  sorry

end positive_difference_median_mode_l691_691949


namespace notebook_cost_l691_691321

noncomputable theory

variables (n p : ℝ)

theorem notebook_cost :
  n + p = 3.20 ∧ n = 2.50 + p → n = 2.85 :=
by
  intros h
  sorry

end notebook_cost_l691_691321


namespace half_absolute_difference_of_squares_l691_691263

-- Defining the variables a and b involved in the problem
def a : ℤ := 20
def b : ℤ := 15

-- Statement to prove the solution
theorem half_absolute_difference_of_squares : 
    (1 / 2 : ℚ) * (abs ((a ^ 2) - (b ^ 2))) = 87.5 :=
by
    -- Proof omitted
    sorry

end half_absolute_difference_of_squares_l691_691263


namespace find_valid_n_l691_691407

noncomputable def tau (n : ℕ) : ℕ := 
  n.divisors.card

noncomputable def sigma (n : ℕ) : ℕ := 
  n.divisors.sum id

theorem find_valid_n : ∀ n : ℕ, n > 0 → n ∈ {1, 2, 4, 6, 12} ↔ n * (Real.sqrt (tau n)) ≤ sigma n :=
by
  sorry

end find_valid_n_l691_691407


namespace periodic_f_determine_f_l691_691866

noncomputable def f (n : ℤ) : ℤ :=
  if n % 4 = 0 then 1
  else if n % 4 = 1 then 0
  else if n % 4 = 2 then -1
  else 0

theorem periodic_f (n : ℤ) : f (n + 4) = f n := by
  sorry

theorem determine_f :
  f (1^2 + 2^2 + ⋯ + 2015^2) = 1 ∧ (∑ k : ℤ in finset.range (2015 + 1), f (k^2)) = 1007 ∧
  ∑ k : ℤ in finset.range (2015 + 1), f (k^2) = 1007 → (f (1^2 + 2^2 + ⋯ + 2015^2)) / (∑ k : ℤ in finset.range (2015 + 1), f (k^2)) = 1/1007 := by
  sorry

end periodic_f_determine_f_l691_691866


namespace half_abs_diff_squares_l691_691237

theorem half_abs_diff_squares (a b : ℤ) (ha : a = 20) (hb : b = 15) : 
  (|a^2 - b^2| / 2) = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691237


namespace first_player_forces_win_l691_691191

-- Definitions for vertices and edge set of the pentagon
inductive Vertex : Type
| P1 | P2 | P3 | P4 | P5

open Vertex

-- Definitions for edges and diagonals of the pentagon
def edges : list (Vertex × Vertex) :=
  [(P1, P2), (P2, P3), (P3, P4), (P4, P5), (P5, P1), -- Edges
   (P1, P3), (P2, P4), (P3, P5), (P4, P1), (P5, P2)] -- Diagonals

-- Definitions for players
inductive Player : Type
| Player1
| Player2

open Player

-- Definitions for colors
inductive Color : Type
| Blue
| Red

open Color

-- An assignment of colors to edges
def Coloring : Type :=
  edges → option Color

-- Defining a condition for winning: forming a monochromatic triangle
def is_monochromatic_triangle (col : Coloring) (c : Color) : Prop :=
  ∃ (v1 v2 v3 : Vertex), 
    col (v1, v2) = some c ∧ 
    col (v2, v3) = some c ∧ 
    col (v3, v1) = some c

-- Formal statement of theorem
theorem first_player_forces_win :
  ∃ col : Coloring, is_monochromatic_triangle col Blue :=
sorry

end first_player_forces_win_l691_691191


namespace squares_arrangement_l691_691131

noncomputable def arrangement_possible (n : ℕ) (cond : n ≥ 5) : Prop :=
  ∃ (position : ℕ → ℕ × ℕ),
    (∀ i, 1 ≤ i ∧ i ≤ n → 
        ∃ j k, j ≠ k ∧ 
             dist (position i) (position j) = 1 ∧
             dist (position i) (position k) = 1)

theorem squares_arrangement (n : ℕ) (hn : n ≥ 5) :
  arrangement_possible n hn :=
  sorry

end squares_arrangement_l691_691131


namespace compare_expressions_l691_691194

-- Define the conditions of the problem
variables (a b c d : ℝ)
hypothesis h1 : a + b = 0
hypothesis h2 : c + d = 0

-- Define the final statement to prove
theorem compare_expressions : a^3 + c^2 = d^2 - b^3 :=
by 
  sorry

end compare_expressions_l691_691194


namespace percent_not_filler_l691_691654

theorem percent_not_filler (total_weight filler_weight : ℕ) (h1 : total_weight = 180) (h2 : filler_weight = 45) : 
  ((total_weight - filler_weight) * 100 / total_weight = 75) :=
by 
  sorry

end percent_not_filler_l691_691654


namespace fraction_power_zero_l691_691947

variable (a b : ℤ)
variable (h_a : a ≠ 0) (h_b : b ≠ 0)

theorem fraction_power_zero : (a / b)^0 = 1 := by
  sorry

end fraction_power_zero_l691_691947


namespace systematic_sampling_largest_number_l691_691828

theorem systematic_sampling_largest_number (total_students : ℕ) (interval start smallest second_smallest : ℕ) (n : ℕ)
  (h_total : total_students = 160)
  (h_start : start = 6)
  (h_smallest : smallest = 6)
  (h_second_smallest : second_smallest = 22)
  (h_interval : interval = second_smallest - smallest)
  (sample_size : n = 10) :
  let a_n : ℕ := start + interval * (n - 1) in
  a_n = 150 :=
by
  sorry

end systematic_sampling_largest_number_l691_691828


namespace tangent_at_point_l691_691498

open Real

noncomputable def curve (a b : ℝ) (x : ℝ) : ℝ := -1/a * exp x + b

theorem tangent_at_point (a b : ℝ) :
  let f := curve a b
  let F' := fun x => (-1/a) * exp x
  let tangent_line := fun x => -x + 1
  (F' 0 = -1 ∧ f 0 = 1) → (a = 1 ∧ b = 2) :=
by
  intros h
  let f := curve a b
  have h1 : F' 0 = -1 := h.1
  have h2 : f 0 = 1 := h.2
  sorry

end tangent_at_point_l691_691498


namespace num_candidates_appeared_each_state_l691_691087

-- Definitions
def candidates_appear : ℕ := 8000
def sel_pct_A : ℚ := 0.06
def sel_pct_B : ℚ := 0.07
def additional_selections_B : ℕ := 80

-- Proof Problem Statement
theorem num_candidates_appeared_each_state (x : ℕ) 
  (h1 : x = candidates_appear) 
  (h2 : sel_pct_A * ↑x = 0.06 * ↑x) 
  (h3 : sel_pct_B * ↑x = 0.07 * ↑x) 
  (h4 : sel_pct_B * ↑x = sel_pct_A * ↑x + additional_selections_B) : 
  x = candidates_appear := sorry

end num_candidates_appeared_each_state_l691_691087


namespace percentage_hospitalized_smokers_l691_691827

theorem percentage_hospitalized_smokers :
  ∀ (total_students non_hospitalized_smokers : ℕ) (smoking_percentage : ℚ),
  total_students = 300 →
  smoking_percentage = 0.40 →
  non_hospitalized_smokers = 36 →
  let total_smokers := (smoking_percentage * total_students).to_nat in
  let hospitalized_smokers := total_smokers - non_hospitalized_smokers in
  (hospitalized_smokers * 100 / total_smokers) = 70 :=
by
  intros total_students non_hospitalized_smokers smoking_percentage
  intro h1 h2 h3
  let total_smokers := (smoking_percentage * total_students).to_nat
  let hospitalized_smokers := total_smokers - non_hospitalized_smokers
  sorry

end percentage_hospitalized_smokers_l691_691827


namespace rice_grains_difference_l691_691666

theorem rice_grains_difference :
  let grains_on_square (k : ℕ) := 2^k in
  grains_on_square 12 - ∑ k in finset.range 11, grains_on_square (k + 1) = 2050 :=
by
  sorry

end rice_grains_difference_l691_691666


namespace reduced_residue_system_mod_mn_l691_691448

open Nat

-- Hypothetical inputs for the problem
variables {m n : ℕ}
variables (s t : ℕ)
variables (a : Fin t → ℕ) (b : Fin s → ℕ)

-- Assumptions
variables (h_coprime_mn : gcd m n = 1)
variables (h_coprime_ajm : ∀ j, gcd (a j) m = 1)
variables (h_coprime_bin : ∀ i, gcd (b i) n = 1)

theorem reduced_residue_system_mod_mn :
  ∃ S : Fin (s * t) → ℕ, S = λ idx, let ⟨i, j⟩ := idx.div_mod s in m * (b i) + n * (a j) ∧
  (∀ idx, gcd (S idx) (m * n) = 1) ∧
  (∀ (idx1 idx2 : Fin (s * t)), S idx1 ≠ S idx2 → (S idx1 + m * n) % (m * n) ≠ (S idx2 + m * n) % (m * n)) := by
  sorry

end reduced_residue_system_mod_mn_l691_691448


namespace apples_left_by_the_time_they_got_home_l691_691663

theorem apples_left_by_the_time_they_got_home
  (initial_children : ℕ)
  (apples_per_child : ℕ)
  (children_ate : ℕ)
  (apples_eaten_per_child : ℕ)
  (children_sold : ℕ)
  (apples_sold_per_child : ℕ):
  initial_children = 5 →
  apples_per_child = 15 →
  children_ate = 2 →
  apples_eaten_per_child = 4 →
  children_sold = 1 →
  apples_sold_per_child = 7 →
  initial_children * apples_per_child - (children_ate * apples_eaten_per_child + children_sold * apples_sold_per_child) = 60 :=
by
  intros h_initial_children h_apples_per_child h_children_ate h_apples_eaten_per_child h_children_sold h_apples_sold_per_child
  rw [h_initial_children, h_apples_per_child, h_children_ate, h_apples_eaten_per_child, h_children_sold, h_apples_sold_per_child]
  apply congrArg
  apply congrArg
  apply congrArg
  norm_num
  sorry

end apples_left_by_the_time_they_got_home_l691_691663


namespace half_abs_diff_squares_l691_691257

theorem half_abs_diff_squares (a b : ℤ) (h₁ : a = 20) (h₂ : b = 15) : 
  (|a^2 - b^2| / 2 : ℚ) = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691257


namespace regular_ngon_sum_of_squares_l691_691964

noncomputable def sum_of_squares_of_distances (n : ℕ) (h : n > 2) : ℝ := 
  n * real.sqrt (2 - 2 * real.cos (2 * real.pi / n))

theorem regular_ngon_sum_of_squares (n : ℕ) (h : n > 2) :
  ∃ P : ℝ, P ∈ set.unit_circle → 
  (∑ 1 ≤ i < j ≤ n, (dist (exp(2 * real.pi * complex.i * (i - 1) / n)) (exp(2 * real.pi * complex.i * (j - 1) / n)))^2)
  = sum_of_squares_of_distances n h := 
sorry

end regular_ngon_sum_of_squares_l691_691964


namespace trapezoid_area_is_180_l691_691684

noncomputable def area_of_trapezoid (x y : ℝ) (h1 : IsoscelesTrapezoidCircumscribedCircle x y) (h2 : baseIs20 y) (h3 : baseAngleIsArctanThreeOverTwo x) : ℝ :=
  (1 / 2) * (y + 20) * (1.5 * x)

theorem trapezoid_area_is_180 : ∀ (x y : ℝ), 
    IsoscelesTrapezoidCircumscribedCircle x y →
    baseIs20 y →
    baseAngleIsArctanThreeOverTwo x →
    area_of_trapezoid x y _ _ _ = 180 := 
by
  intros x y h1 h2 h3
  sorry


end trapezoid_area_is_180_l691_691684


namespace parallel_vectors_l691_691531

noncomputable def a : Vector ℝ := sorry
noncomputable def b : Vector ℝ := sorry
noncomputable def c : Vector ℝ := sorry

noncomputable def M : Vector ℝ := (2/3) • a
noncomputable def N : Vector ℝ := (1/2) • (b + c)
noncomputable def MN : Vector ℝ := N - M

theorem parallel_vectors : 
(MN = - (2/3) • a + (1/2) • b + (1/2) • c) ∧
((-4 : ℝ) • a + (3 : ℝ) • b + (3 : ℝ) • c) = (-6) • MN ∧ 
((4 / 3 : ℝ) • a - (1 : ℝ) • b - (1 : ℝ) • c) = (-1) • MN :=
by sorry

end parallel_vectors_l691_691531


namespace smallest_two_digit_prime_with_conditions_l691_691403

open Nat

def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def is_composite (n : ℕ) : Prop := n ≥ 2 ∧ ∃ m, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

def reverse_digits (n : ℕ) : ℕ :=
  let digits := to_digits 10 n
  of_digits 10 digits.reverse

noncomputable def smallest_prime_with_conditions : ℕ :=
  23 -- The smallest prime satisfying the conditions, verified manually

theorem smallest_two_digit_prime_with_conditions :
  ∃ p, is_prime p ∧ (10 ≤ p ∧ p < 100) ∧ (p % 100 / 10 = 2) ∧ is_composite (reverse_digits p) ∧ 
  (∀ q, is_prime q ∧ (10 ≤ q ∧ q < 100) ∧ (q % 100 / 10 = 2) ∧ is_composite (reverse_digits q) → p ≤ q) := 
begin
  use 23,
  split, {
    dsimp [is_prime],
    split,
    { exact dec_trivial }, -- 23 ≥ 2
    { intros m hm,
      interval_cases m; simp [hm],
    },
  },
  split, {
    split,
    { norm_num }, -- 10 ≤ 23
    { norm_num }, -- 23 < 100
  },
  split, {
    norm_num, -- tens digit is 2
  },
  split, {
    dsimp [is_composite, reverse_digits],
    use 2,
    split,
    { exact dec_trivial, }, -- 10 ≤ 32
    split,
    { exact dec_trivial, }, -- 2 ≠ 1
    { exact dec_trivial },  -- 2 ≠ 32
  },
  intros q hq,
  rcases hq with ⟨prm_q, ⟨hq_low, hq_high⟩, t_d, q_comp⟩,
  rw t_d at *,
  linarith,
end

end smallest_two_digit_prime_with_conditions_l691_691403


namespace difference_in_zits_l691_691916

variable (avgZitsSwanson : ℕ := 5)
variable (avgZitsJones : ℕ := 6)
variable (numKidsSwanson : ℕ := 25)
variable (numKidsJones : ℕ := 32)
variable (totalZitsSwanson : ℕ := avgZitsSwanson * numKidsSwanson)
variable (totalZitsJones : ℕ := avgZitsJones * numKidsJones)

theorem difference_in_zits :
  totalZitsJones - totalZitsSwanson = 67 := by
  sorry

end difference_in_zits_l691_691916


namespace largest_y_l691_691280

theorem largest_y : ∃ (y : ℤ), (y ≤ 3) ∧ (∀ (z : ℤ), (z > y) → ¬ (z / 4 + 6 / 7 < 7 / 4)) :=
by
  -- There exists an integer y such that y <= 3 and for all integers z greater than y, the inequality does not hold
  sorry

end largest_y_l691_691280


namespace zeros_in_sequence_l691_691481

theorem zeros_in_sequence (n : ℕ) (h : n = 2007) : 
  count_zeros (list.range' 1 h.succ) = 506 :=
sorry

noncomputable def count_zeros (l : list ℕ) : ℕ :=
l.sum (λ x, (x.digits 10).count 0)

end zeros_in_sequence_l691_691481


namespace length_of_second_train_l691_691226

theorem length_of_second_train
  (length_first_train : ℝ)
  (speed_first_train : ℝ)
  (speed_second_train : ℝ)
  (crossing_time : ℝ)
  (total_distance : ℝ)
  (relative_speed_mps : ℝ)
  (length_second_train : ℝ) :
  length_first_train = 130 ∧ 
  speed_first_train = 60 ∧
  speed_second_train = 40 ∧
  crossing_time = 10.439164866810657 ∧
  relative_speed_mps = (speed_first_train + speed_second_train) * (5/18) ∧
  total_distance = relative_speed_mps * crossing_time ∧
  length_first_train + length_second_train = total_distance →
  length_second_train = 160 :=
by
  sorry

end length_of_second_train_l691_691226


namespace lambda_5_geq_2_sin_54_l691_691417

-- Define the problem
theorem lambda_5_geq_2_sin_54 (points : Fin 5 → ℝ × ℝ) :
  let distances := {dist | ∃ i j, i ≠ j ∧ dist = (dist.points points i j)} in
  let λ5 := (distances.max' sorry) / (distances.min' sorry) in
  λ5 ≥ 2 * Real.sin (54 * Real.pi / 180) ∧
  (λ5 = 2 * Real.sin (54 * Real.pi / 180) ↔
   ∃ pentagon, (∀ i, ∃ j, j ∈ pentagon ∧ j ≠ i ∧ dist.points pentagon i j) ∧
   RegularPentagon pentagon) :=
sorry

-- Define helper functions
noncomputable def dist.points (points : Fin 5 → ℝ × ℝ) (i j : Fin 5) : ℝ :=
  Real.sqrt (points i - points j).1^2 + (points i - points j).2^2

-- Regular Pentagon
structure RegularPentagon (points : Fin 5 → ℝ × ℝ) : Prop :=
(equilateral : ∀ i j, dist.points points i j = dist.points points (Fin.iterate Fin.succ 2 i) (Fin.iterate Fin.succ 2 j))

end lambda_5_geq_2_sin_54_l691_691417


namespace total_ticket_cost_l691_691847

-- Conditions
def website_price := 50
def website_tickets := 4
def scalper_price_increase := 2.75
def scalper_tickets := 6
def scalper_discount := 20
def job_discounts := [0.60, 0.75, 0.50]
def loyalty_discounts := [0.20, 0.10]
def service_fee_rate := 0.12

-- Calculations for website tickets
def website_total_cost := 
  let base_cost := website_tickets * website_price
  let fee := base_cost * service_fee_rate
  base_cost + fee

-- Calculations for scalper tickets
def scalper_total_cost :=
  let base_cost := scalper_tickets * (scalper_price_increase * website_price)
  let discounted_cost := base_cost - scalper_discount
  let fee := discounted_cost * service_fee_rate
  discounted_cost + fee

-- Calculations for job discounted tickets
def job_total_cost := 
  let base_cost := job_discounts.map(fun d => d * website_price) |>.sum
  let fee := base_cost * service_fee_rate
  base_cost + fee

-- Calculations for loyalty points tickets
def loyalty_total_cost := 
  let base_cost := loyalty_discounts.map(fun d => d * website_price) |>.sum
  let fee := base_cost * service_fee_rate
  base_cost + fee

-- Total cost
def total_cost := website_total_cost + scalper_total_cost + job_total_cost + loyalty_total_cost

-- Theorem
theorem total_ticket_cost : total_cost = 1246 := 
by
  sorry

end total_ticket_cost_l691_691847


namespace head_start_fraction_l691_691966

variables (v_A v_B L H : ℝ)
hypothesis h1 : v_A = 32 / 27 * v_B
hypothesis h2 : L / v_A = (L - H) / v_B

theorem head_start_fraction : H = 5 / 32 * L :=
by sorry

end head_start_fraction_l691_691966


namespace min_value_fraction_l691_691842

theorem min_value_fraction (AB AC CB CA : ℝ) (A B C P : Type)
  (S : ℝ) 
  (h1 : AB * AC = 9) 
  (h2 : sin B = cos A * sin C) 
  (h3 : S = 6) 
  (h4 : P ∈ segment A B)
  (h5 : ∀ (x y : ℝ), CP = x * (CA / |CA|) + y * (CB / |CB|))
  (h6 : ∀ (x y : ℝ), (x / 3) + (y / 4) = 1) : 
  ∃ (min_val : ℝ), (∀ (x y : ℝ), (2 / x) + (1 / y) ≥ min_val) ∧ min_val = (11 / 12) + (sqrt 6 / 3) :=
sorry

end min_value_fraction_l691_691842


namespace son_l691_691879

noncomputable def my_age_in_years : ℕ := 84
noncomputable def total_age_in_years : ℕ := 140
noncomputable def months_in_a_year : ℕ := 12
noncomputable def weeks_in_a_year : ℕ := 52

theorem son's_age_in_weeks (G_d S_m G_m S_y : ℕ) (G_y : ℚ) :
  G_d = S_m →
  G_m = my_age_in_years * months_in_a_year →
  G_y = (G_m : ℚ) / months_in_a_year →
  G_y + S_y + my_age_in_years = total_age_in_years →
  S_y * weeks_in_a_year = 2548 :=
by
  intros h1 h2 h3 h4
  sorry

end son_l691_691879


namespace half_abs_diff_squares_l691_691235

theorem half_abs_diff_squares (a b : ℤ) (ha : a = 20) (hb : b = 15) : 
  (|a^2 - b^2| / 2) = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691235


namespace tan_of_exponential_point_l691_691818

theorem tan_of_exponential_point (a : ℝ) (h : 9 = 3^a) : Real.tan (a * Real.pi / 6) = Real.sqrt 3 := by
  sorry

end tan_of_exponential_point_l691_691818


namespace sqrt_diff_inequality_l691_691492

theorem sqrt_diff_inequality (x : ℝ) (hx : x ≥ 1) : 
  sqrt x - sqrt (x - 1) > sqrt (x + 1) - sqrt x :=
sorry

end sqrt_diff_inequality_l691_691492


namespace find_original_price_l691_691345

noncomputable def original_price (new_price : ℝ) (decrease_percentage : ℝ) : ℝ :=
  new_price / (1 - decrease_percentage / 100)

theorem find_original_price :
  original_price 820 24 ≈ 1078.95 :=
by
  sorry

end find_original_price_l691_691345


namespace value_of_abc_l691_691812

theorem value_of_abc : ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (ab + c + 10 = 51) ∧ (bc + a + 10 = 51) ∧ (ac + b + 10 = 51) ∧ (a + b + c = 41) :=
by
  sorry

end value_of_abc_l691_691812


namespace number_of_valid_ordered_pairs_l691_691738

noncomputable def valid_ordered_pairs : Set (ℕ × ℕ) :=
  {(b, c) | b > 0 ∧ c > 0 ∧ b^2 - 4 * c ≤ 1 ∧ c^2 - 4 * b ≤ 1}

theorem number_of_valid_ordered_pairs :
  (valid_ordered_pairs.to_finset.card = 3) :=
by
  sorry

end number_of_valid_ordered_pairs_l691_691738


namespace Madison_minimum_score_l691_691938

theorem Madison_minimum_score (q1 q2 q3 q4 q5 : ℕ) (h1 : q1 = 84) (h2 : q2 = 81) (h3 : q3 = 87) (h4 : q4 = 83) (h5 : 85 * 5 ≤ q1 + q2 + q3 + q4 + q5) : 
  90 ≤ q5 := 
by
  sorry

end Madison_minimum_score_l691_691938


namespace coloring_ways_without_two_corner_cells_l691_691936

/-
There are exactly 120 ways to color five cells in a 5 × 5 grid such that each row and each column contains exactly one colored cell.
There are exactly 96 ways to color five cells in a 5 × 5 grid without the corner cell, such that each row and each column contains exactly one colored cell.
Prove that there are 78 ways to color five cells in a 5 × 5 grid without two corner cells, such that each row and each column contains exactly one colored cell.
-/
theorem coloring_ways_without_two_corner_cells :
  let total_ways := 120
  let ways_without_one_corner := 96
  let ways_without_corner_A_B := total_ways - 2 * (total_ways - ways_without_one_corner) + ((5 - 2)!) in
  ways_without_corner_A_B = 78 := sorry

end coloring_ways_without_two_corner_cells_l691_691936


namespace angle_between_line_and_plane_l691_691525

variables (α β : ℝ) -- angles in radians
-- Definitions to capture the provided conditions
def dihedral_angle (α : ℝ) : Prop := true -- The angle between the planes γ₁ and γ₂
def angle_with_edge (β : ℝ) : Prop := true -- The angle between line AB and edge l

-- The angle between line AB and the plane γ₂
theorem angle_between_line_and_plane (α β : ℝ) (h1 : dihedral_angle α) (h2 : angle_with_edge β) : 
  ∃ θ : ℝ, θ = Real.arcsin (Real.sin α * Real.sin β) :=
by
  sorry

end angle_between_line_and_plane_l691_691525


namespace sum_of_possible_n_for_continuous_f_l691_691554

noncomputable def f (x n k : ℝ) : ℝ :=
if x < n then x^2 + 3 else 3*x + k

theorem sum_of_possible_n_for_continuous_f (k : ℝ) (h_k : k = 8) :
  (∃ n : ℝ, continuous (f x n k) ∧ ∑ roots n in {n | f x n 8 ≤ 0} = 3) :=
begin
  sorry
end

end sum_of_possible_n_for_continuous_f_l691_691554


namespace paula_remaining_money_l691_691154

theorem paula_remaining_money 
  (M : Int) (C_s : Int) (N_s : Int) (C_p : Int) (N_p : Int)
  (h1 : M = 250) 
  (h2 : C_s = 15) 
  (h3 : N_s = 5) 
  (h4 : C_p = 25) 
  (h5 : N_p = 3) : 
  M - (C_s * N_s + C_p * N_p) = 100 := 
by
  sorry

end paula_remaining_money_l691_691154


namespace limit_f_at_1_l691_691376

-- Define the function f(x)
def f (x : ℝ) : ℝ := (ln (e * x))^2^(1/(x^2 + 1))

-- Conditions
axiom limit_of_x : filter.tendsto (λ (x : ℝ), x) filter.cofinite 1
axiom ln_ex_eq : ∀ (x : ℝ), ln (e * x) = 1 + ln x

-- The theorem with the precise statement
theorem limit_f_at_1 : filter.tendsto (λ (x : ℝ), f x) filter.cofinite 1 :=
by
  sorry -- proof goes here

end limit_f_at_1_l691_691376


namespace inequality_proof_l691_691762

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  (b + c) * (c + a) * (a + b) ≥ 4 * ((a + b + c) * ((a + b + c) / 3)^(1 / 8) - 1) :=
by
  sorry

end inequality_proof_l691_691762


namespace sum_of_digits_divisible_by_six_l691_691493

theorem sum_of_digits_divisible_by_six (A B : ℕ) (h1 : 10 * A + B % 6 = 0) (h2 : A + B = 12) : A + B = 12 :=
by
  sorry

end sum_of_digits_divisible_by_six_l691_691493


namespace find_c_l691_691285

theorem find_c (c : ℝ) : 
  (∀ x : ℝ, x * (3 * x + 1) < c ↔ x ∈ Set.Ioo (-(7 / 3) : ℝ) (2 : ℝ)) → c = 14 :=
by
  intro h
  sorry

end find_c_l691_691285


namespace intersection_P_Q_l691_691545

def P : Set ℝ := {x | x^2 - 9 < 0}
def Q : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}

theorem intersection_P_Q :
  {x : ℤ | (x : ℝ) ∈ P} ∩ Q = {-1, 0, 1, 2} := 
by
  sorry

end intersection_P_Q_l691_691545


namespace problem_statement_l691_691136

noncomputable def f (x : ℝ) : ℝ := sorry -- Assume f is defined elsewhere

axiom domain_f : ∀ x, 0 < x → ∃ y, f y = f x
axiom increasing_f : ∀ x y, 0 < x → 0 < y → x < y → f(x) < f(y)
axiom functional_f : ∀ x y, 0 < x → 0 < y → f(x / y) = f(x) - f(y)
axiom f_of_2 : f 2 = 1

-- Prove the properties
theorem problem_statement :
  (f 1 = 0) ∧
  (∀ x y, 0 < x → 0 < y → f(x * y) = f(x) + f(y)) ∧
  (∀ x, 3 < x → x ≤ 4 → f(x) - f(1 / (x-3)) ≤ 2) :=
by sorry

end problem_statement_l691_691136


namespace verify_exactly_two_talents_l691_691092

variable (total_students : ℕ) (total_with_three_talents : ℕ)
variable (cannot_sing : ℕ) (cannot_dance : ℕ) (cannot_act : ℕ)
variable (exactly_two_talents : ℕ)

-- Define the conditions based on the given problem
def num_students_can_sing (total_students cannot_sing : ℕ) : ℕ := total_students - cannot_sing
def num_students_can_dance (total_students cannot_dance : ℕ) : ℕ := total_students - cannot_dance
def num_students_can_act (total_students cannot_act : ℕ) : ℕ := total_students - cannot_act

def total_talents_with_all (sing dance act three_talents : ℕ) : ℕ :=
  sing + dance + act - 3 * three_talents

def inclusion_exclusion_result (sing dance act two_talents three_talents total_students : ℕ) : ℕ :=
  sing + dance + act - two_talents - 2 * three_talents

-- Lean statement to prove
theorem verify_exactly_two_talents :
  exactly_two_talents = 80 :=
by
  -- Define the relevant numbers based on provided variables
  let sing : ℕ := num_students_can_sing total_students cannot_sing
  let dance : ℕ := num_students_can_dance total_students cannot_dance
  let act : ℕ := num_students_can_act total_students cannot_act
  let total_with_all := total_talents_with_all sing dance act total_with_three_talents
  let result := inclusion_exclusion_result sing dance act exactly_two_talents total_with_three_talents total_students

  -- Prove the required equality
  have h : result = total_students := sorry
  exact h

end verify_exactly_two_talents_l691_691092


namespace zits_difference_l691_691918

variable (avg_zits_swanson : ℕ)
variable (num_students_swanson : ℕ)
variable (avg_zits_jones : ℕ)
variable (num_students_jones : ℕ)

-- Conditions
def total_zits_swanson := avg_zits_swanson * num_students_swanson
def total_zits_jones := avg_zits_jones * num_students_jones

-- Theorem to prove the difference in total zits
theorem zits_difference : 
  avg_zits_swanson = 5 → 
  num_students_swanson = 25 → 
  avg_zits_jones = 6 → 
  num_students_jones = 32 → 
  total_zits_jones avg_zits_jones num_students_jones - total_zits_swanson avg_zits_swanson num_students_swanson = 67 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  show 6 * 32 - 5 * 25 = 67
  norm_num
  sorry

end zits_difference_l691_691918


namespace center_ball_distance_l691_691309

-- Define the conditions
def diameter_ball := 6
def radius_ball := diameter_ball / 2
def R₁ := 120
def R₂ := 50
def R₃ := 90
def R₄ := 75

-- Define the adjusted radii for the arcs
def adjusted_R₁ := R₁ - radius_ball
def adjusted_R₂ := R₂ + radius_ball
def adjusted_R₃ := R₃ - radius_ball
def adjusted_R₄ := R₄ + radius_ball

-- The adjusted distances in terms of π
def distance_1 := adjusted_R₁ * π
def distance_2 := adjusted_R₂ * π
def distance_3 := adjusted_R₃ * π
def distance_4 := adjusted_R₄ * π

-- The total distance
def total_distance := distance_1 + distance_2 + distance_3 + distance_4

-- The statement we need to prove
theorem center_ball_distance : total_distance = 335 * π :=
by
  sorry

end center_ball_distance_l691_691309


namespace max_rational_points_l691_691864

def is_rational (x : ℝ) : Prop :=
  ∃ (q : ℚ), (x : ℝ) = q

def is_rational_point (p : ℝ × ℝ) : Prop :=
  is_rational p.1 ∧ is_rational p.2

def is_on_circle (p : ℝ × ℝ) (center : ℝ × ℝ) (r : ℝ) : Prop :=
  (p.1 - center.1) ^ 2 + (p.2 - center.2) ^ 2 = r ^ 2

theorem max_rational_points (r : ℝ) (h_r_pos : 0 < r) :
  ∀ (p1 p2 : ℝ × ℝ),
  is_on_circle p1 (sqrt 2, sqrt 3) r →
  is_on_circle p2 (sqrt 2, sqrt 3) r →
  is_rational_point p1 →
  is_rational_point p2 →
  p1 = p2 :=
begin
  sorry
end

end max_rational_points_l691_691864


namespace ratio_eq_2_sqrt_2_div_pi_l691_691086

noncomputable def radius (r : ℝ) :=
  ∃ A B C : ℝ × ℝ,
    (dist A B = dist A C) ∧ (dist A B > r) ∧ 
    (2 * π * r / 4 = π * r / 2)

theorem ratio_eq_2_sqrt_2_div_pi (r : ℝ) (h₁ : radius r) :
    ∀ A B C : ℝ × ℝ, 
      (dist A B = dist A C) → (dist A B > r) → 
      (2 * π * r / 4 = π * r / 2) → 
      (dist A B) / (π * r / 2) = 2 * Real.sqrt 2 / π :=
  sorry

end ratio_eq_2_sqrt_2_div_pi_l691_691086


namespace positive_solution_of_x_l691_691041

theorem positive_solution_of_x :
  ∃ x y z : ℝ, (x * y = 6 - 2 * x - 3 * y) ∧ (y * z = 6 - 4 * y - 2 * z) ∧ (x * z = 30 - 4 * x - 3 * z) ∧ x > 0 ∧ x = 3 :=
by
  sorry

end positive_solution_of_x_l691_691041


namespace geometric_sequence_a5_eq_8_l691_691527

variable (a : ℕ → ℝ)
variable (n : ℕ)

-- Conditions
axiom pos (n : ℕ) : a n > 0
axiom prod_eq (a3 a7 : ℝ) : a 3 * a 7 = 64

-- Statement to prove
theorem geometric_sequence_a5_eq_8
  (pos : ∀ n, a n > 0)
  (prod_eq : a 3 * a 7 = 64) :
  a 5 = 8 :=
sorry

end geometric_sequence_a5_eq_8_l691_691527


namespace janice_water_balloons_l691_691382

-- Define the conditions given in the problem
variables (J : ℕ) -- Number of water balloons Janice has
variable  (R : ℕ) -- Number of water balloons Randy has
variable  (C : ℕ) -- Number of water balloons Cynthia has

-- Define the relationships from the conditions
def randy_has_half_of_janice (J : ℕ) : Prop := R = J / 2
def cynthia_has_four_times_randy (R : ℕ) : Prop := C = 4 * R
def cynthia_left_with_12_after_janice_throws (J C : ℕ) : Prop := 2 * J + J / 2 = J + 12

-- The theorem to prove
theorem janice_water_balloons : randy_has_half_of_janice J → cynthia_has_four_times_randy R → cynthia_left_with_12_after_janice_throws J C → J = 8 :=
by
  intros h1 h2 h3
  sorry

end janice_water_balloons_l691_691382


namespace equal_area_segments_l691_691739

variable (r : ℝ) (α : ℝ)
variable (O A B C D : Point) (radius : ℝ)
variable (M N : Circle)
variable (cut_length : ℝ)

-- Definitions and conditions
-- Assuming O is the center of the quarter-circle, M and N are points on the circumference
-- A and B are points after lengths are cut off from M and N.
-- C and D are projections on one radius OM.
-- The area of the sector sliced by OA and OB is equal to the area enclosed by perpendiculars to OM.

-- Given conditions
def is_quarter_circle (c : Circle) (center : Point) (r : ℝ) : Prop :=
  ∃ (p1 p2 : Point), dist center p1 = r ∧ dist center p2 = r ∧ ∠ center p1 p2 = π / 2

def cut_points (c : Circle) (center : Point) (cut_length : ℝ) : Prop :=
  ∃ (a b : Point), dist center a = cut_length ∧ dist center b = cut_length

def is_perpendicular_projection (p1 p2 : Point) (r : Radius) : Prop :=
  ∃ (c d : Point), dist r.center c = dist r.center p1 ∧ dist r.center d = dist r.center p2 ∧
      ∠ r.center p1 c = π / 2 ∧ ∠ r.center p2 d = π / 2

-- Proof Statement
theorem equal_area_segments (quarter_circle : is_quarter_circle M O r) 
                            (cut : cut_points N O cut_length)
                            (proj : is_perpendicular_projection A B (radius O M)) :
  area (sector O A B) = area (segment_projection O C D (radius O M)) :=
sorry

end equal_area_segments_l691_691739


namespace sum_of_prime_factors_172944_l691_691955

theorem sum_of_prime_factors_172944 : 
  (∃ (a b c : ℕ), 2^a * 3^b * 1201^c = 172944 ∧ a = 4 ∧ b = 2 ∧ c = 1) → 2 + 3 + 1201 = 1206 := 
by 
  intros h 
  exact sorry

end sum_of_prime_factors_172944_l691_691955


namespace half_abs_diff_of_squares_l691_691250

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 20) (h2 : b = 15) :
  (1/2 : ℚ) * |(a:ℚ)^2 - (b:ℚ)^2| = 87.5 := by
  sorry

end half_abs_diff_of_squares_l691_691250


namespace roots_relationship_l691_691000

theorem roots_relationship (a b c : ℝ) (α β : ℝ) 
  (h_eq : a * α^2 + b * α + c = 0)
  (h_triple : β = 3 * α)
  (h_vieta1 : α + β = -b / a)
  (h_vieta2 : α * β = c / a) : 
  3 * b^2 = 16 * a * c :=
sorry

end roots_relationship_l691_691000


namespace largest_fraction_consecutive_primes_l691_691438

theorem largest_fraction_consecutive_primes (p q r s : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s)
  (h0 : 0 < p) (h1 : p < q) (h2 : q < r) (h3 : r < s)
  (hconsec : p + 2 = q ∧ q + 2 = r ∧ r + 2 = s) :
  (r + s) / (p + q) > max ((p + q) / (r + s)) (max ((p + s) / (q + r)) (max ((q + r) / (p + s)) ((q + s) / (p + r)))) :=
sorry

end largest_fraction_consecutive_primes_l691_691438


namespace max_area_of_triangle_l691_691824
-- Import necessary components from the Lean 4 library

-- Define the conditions of the problem
variable (A B C : Type)
variable (AC : ℝ)
variable (k : ℝ)
variable (sinA sinC : ℝ)
variable (area : ℝ)

-- Define the properties of the problem
axiom AC_eq_3 : AC = 3
axiom sinC_eq_k_sinA : sinC = k * sinA
axiom k_geq_2 : k ≥ 2

-- Define the area of triangle ABC
noncomputable def max_area_triangle (AC k sinA sinC area : ℝ) : Prop :=
  AC = 3 ∧ sinC = k * sinA ∧ k ≥ 2 → area = 3

-- The statement of the problem in Lean 4
theorem max_area_of_triangle : max_area_triangle AC k sinA sinC 3 :=
by {
  -- We will assume the conditions are true and conclude the area equals 3
  have h1 : AC = 3 := AC_eq_3,
  have h2 : sinC = k * sinA := sinC_eq_k_sinA,
  have h3 : k ≥ 2 := k_geq_2,
  sorry
}

end max_area_of_triangle_l691_691824


namespace cos_angle_is_correct_l691_691140

variables {E : Type*} [inner_product_space ℝ E]

-- Definition of two unit vectors and the angle between them
variables (e1 e2 : E) (h1 : ∥e1∥ = 1) (h2 : ∥e2∥ = 1) (h_angle : real.angle e1 e2 = real.angle.pi_div_three)

-- Definition of the cosine of the angle between the given vectors
noncomputable def cos_angle : ℝ :=
  (3 • e1 + 4 • e2) ⬝ e1 / (∥3 • e1 + 4 • e2∥ * ∥e1∥)

theorem cos_angle_is_correct : cos_angle e1 e2 = 5 / real.sqrt 37 :=
sorry

end cos_angle_is_correct_l691_691140


namespace incorrect_comparison_is_d_l691_691290

theorem incorrect_comparison_is_d :
  (-0.02 < 1) ∧
  (4 / 5 > 3 / 4) ∧
  (-( - (3 / 4)) > -(abs (-0.75))) ∧
  (-(22 / 7) > -3.14) → false := by
  intro h,
  cases h,
  cases h_right,
  cases h_right_right,
  have h_contradiction : ¬( - (22 / 7) > -3.14),
    -- Detail of proof skipped for contradiction
    sorry,
  exact h_contradiction h_right_right_right

end incorrect_comparison_is_d_l691_691290


namespace possible_values_of_n_l691_691777

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 2 then (1 / 2) - |x - (3 / 2)| else real.exp (x - 2) * (-x^2 + 8 * x - 12)

theorem possible_values_of_n :
  ∃ (x : ℕ → ℝ), ∃ (n ≥ 2), (∀ i j, i ≠ j → x i ∈ (1, ∞) ∧ x j ∈ (1, ∞) 
  ∧ 0 < f (x i) ∧ f (x i) / (x i) = f (x j) / (x j)) → n ∈ {2, 3, 4} :=
sorry

end possible_values_of_n_l691_691777


namespace pentagon_BM_length_l691_691993

noncomputable def circle_radius : ℝ := 10
noncomputable def length_PB_PE : ℝ := 24
noncomputable def angle_BPD : ℝ := 30
def BM : ℝ := 13

theorem pentagon_BM_length 
    (ABCDE_inscribed : ∃ (K : Type) (circ : K → K → ℝ), ∀ (A B C D E : K), circ A B = circle_radius ∧ circ B C = circle_radius ∧ circ C D = circle_radius ∧ circ D E = circle_radius ∧ circ E A = circle_radius)
    (BC_parallel_AD : ∃ (K : Type) (parallel : K → K → K), ∀ (B C A D : K), parallel B C = parallel A D)
    (AD_intersects_CE_at_M : ∃ (K : Type) (intersect : K → K → K), ∀ (A D C E : K) (M : K), intersect A D C E = M)
    (tangents_intersect_DA_at_P : ∃ (K : Type) (tangent : K → K → K), ∀ (B E A D P : K), tangent B A = tangent E D ∧ tangent B P = length_PB_PE ∧ tangent E P = length_PB_PE)
    (angle_BPD_given : ∠ B P D = angle_BPD) :
  BM = 13 := 
sorry

end pentagon_BM_length_l691_691993


namespace pseudoprime_pow_minus_one_l691_691160

theorem pseudoprime_pow_minus_one (n : ℕ) (hpseudo : 2^n ≡ 2 [MOD n]) : 
  ∃ m : ℕ, 2^(2^n - 1) ≡ 1 [MOD (2^n - 1)] :=
by
  sorry

end pseudoprime_pow_minus_one_l691_691160


namespace proof_y_solves_diff_eqn_l691_691573

noncomputable def y (x : ℝ) : ℝ := Real.exp (2 * x)

theorem proof_y_solves_diff_eqn : ∀ x : ℝ, (deriv^[3] y x) - 8 * y x = 0 := by
  sorry

end proof_y_solves_diff_eqn_l691_691573


namespace problem_statement_l691_691530

def parametric_curve (θ : ℝ) : ℝ × ℝ :=
  (1 + 2 * cos θ, 2 * sin θ)

def polar_line (ρ θ : ℝ) (m : ℝ) : Prop :=
  ρ * (cos θ + sin θ) = m

def polar_curve (ρ θ : ℝ) : Prop :=
  ρ^2 - 2 * ρ * cos θ - 3 = 0

theorem problem_statement (m > 0)
  (intersects_line₁ : ∃ ρₐ, polar_line ρₐ (π/4) m)
  (intersects_curve : ∃ θₘ θₙ, (θₘ ≠ θₙ) ∧ (∃ ρₘ ρₙ, polar_curve ρₘ θₘ ∧ polar_curve ρₙ θₙ ∧ 
                                  ρₘ * (cos θₘ + sin θₘ) = m ∧ 
                                  ρₙ * (cos θₙ + sin θₙ) = m))
  (magnitude_condition : ∀ ρₐ ρₘ ρₙ, polar_line ρₐ (π/4) m → 
                        polar_curve ρₘ (π/4) → polar_curve ρₙ (π/4) → 
                        ρₐ * ρₘ * ρₙ = 6) : m = 2 * sqrt(2) :=
by {
  sorry
}

end problem_statement_l691_691530


namespace mapping_f_correct_l691_691458

theorem mapping_f_correct (a1 a2 a3 a4 b1 b2 b3 b4 : ℤ) :
  (∀ (x : ℤ), x^4 + a1 * x^3 + a2 * x^2 + a3 * x + a4 = (x + 1)^4 + b1 * (x + 1)^3 + b2 * (x + 1)^2 + b3 * (x + 1) + b4) →
  a1 = 4 → a2 = 3 → a3 = 2 → a4 = 1 →
  b1 = 0 → b1 + b2 + b3 + b4 = 0 →
  (b1, b2, b3, b4) = (0, -3, 4, -1) :=
by
  intros
  sorry

end mapping_f_correct_l691_691458


namespace small_cubes_count_l691_691662

/--
Given a cube with an edge length of 4 cm is cut into N smaller cubes, 
such that not all smaller cubes are of the same size and each edge of 
the smaller cubes is a whole number of centimeters, prove that N = 57.
-/
theorem small_cubes_count (N : ℕ) 
  (edge_lengths : ∀ (i : fin N), ℕ) 
  (total_volume : ∑ (i : fin N), (edge_lengths i)^3 = 4^3) 
  (not_all_same : ¬∀ i j, edge_lengths i = edge_lengths j) 
  (edges_are_integers : ∀ i, edge_lengths i ∈ {1, 2, 3, 4}) :
  N = 57 := 
  sorry

end small_cubes_count_l691_691662


namespace find_function_value_at_2_l691_691072

variables {f : ℕ → ℕ}

theorem find_function_value_at_2 (H : ∀ x : ℕ, Nat.succ (Nat.succ x * Nat.succ x + f x) = 12) : f 2 = 4 :=
by
  sorry

end find_function_value_at_2_l691_691072


namespace remaining_volume_correct_l691_691338

noncomputable def diameter_sphere : ℝ := 24
noncomputable def radius_sphere : ℝ := diameter_sphere / 2
noncomputable def height_hole1 : ℝ := 10
noncomputable def diameter_hole1 : ℝ := 3
noncomputable def radius_hole1 : ℝ := diameter_hole1 / 2
noncomputable def height_hole2 : ℝ := 10
noncomputable def diameter_hole2 : ℝ := 3
noncomputable def radius_hole2 : ℝ := diameter_hole2 / 2
noncomputable def height_hole3 : ℝ := 5
noncomputable def diameter_hole3 : ℝ := 4
noncomputable def radius_hole3 : ℝ := diameter_hole3 / 2

noncomputable def volume_sphere : ℝ := (4 / 3) * Real.pi * (radius_sphere ^ 3)
noncomputable def volume_hole1 : ℝ := Real.pi * (radius_hole1 ^ 2) * height_hole1
noncomputable def volume_hole2 : ℝ := Real.pi * (radius_hole2 ^ 2) * height_hole2
noncomputable def volume_hole3 : ℝ := Real.pi * (radius_hole3 ^ 2) * height_hole3

noncomputable def remaining_volume : ℝ := 
  volume_sphere - (2 * volume_hole1 + volume_hole3)

theorem remaining_volume_correct : remaining_volume = 2239 * Real.pi := by
  sorry

end remaining_volume_correct_l691_691338


namespace pyramid_surface_area_l691_691856

noncomputable def surface_area_pyramid (a b c : ℝ) (abcd : ℝ) : ℝ := sorry

theorem pyramid_surface_area
    (XYZ_plane : Type*)
    (W : XYZ_plane → Prop)
    (W_outside_plane : ¬ ∃ (x y z : XYZ_plane), (W x y z ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x))
    (lengths : set ℝ)
    (face1 face2 face3 face4 : set ℝ)
    (faces_are_triangles : ∀ (f : set ℝ), f = face1 ∨ f = face2 ∨ f = face3 ∨ f = face4 → ∃ x y z : XYZ_plane, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f = {ab x, ab y, ab z})
    (no_face_is_equilateral : ∀ f ∈ {face1, face2, face3, face4}, ¬ (∃ x y z, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ ab x = ab y ∧ ab y = ab z))
    (edges_conditions : ∀ (e : ℝ), e ∈ lengths → e = 24 ∨ e = 49)
    (face_sizes : ∀ f ∈ {face1, face2, face3, face4}, f = {24, 49, 49})
    : surface_area_pyramid 24 49 24 49 = 48 * Real.sqrt 2257 := sorry

end pyramid_surface_area_l691_691856


namespace equivalence_of_conditions_l691_691468

open Set Classical Real

-- Definitions from the conditions
def a := 1
def b := Real.sqrt 3
def hyperbola : Set (ℝ × ℝ) := {p | (p.1^2) - (p.2^2) / 3 = 1}
def moving_line (m n : ℝ) : Set (ℝ × ℝ) := {p | p.1 = m * p.2 + n}
def vertices : Set (ℝ × ℝ) := {v | v = (1,0) ∨ v = (-1,0)}
def right_branch : Set (ℝ × ℝ) := {p | p.1 > 1 ∧ p ∈ hyperbola}

-- Main proof statement: show the equivalence
theorem equivalence_of_conditions (m n x0 y0 : ℝ) :
  (∀ p1 p2 ∈ right_branch, 
    Point ∈ moving_line m n → 
    let A1 := (-1, 0) 
    , A2 := (1, 0) 
    , line1 := line_through A1 p1
    , line2 := line_through A2 p2
    in
    point_of_intersection line1 line2 = (x0, y0)) ↔ 
    (n = 2 ↔ x0 = 1 / 2) :=
by
  sorry

end equivalence_of_conditions_l691_691468


namespace last_donation_on_saturday_l691_691679

def total_amount : ℕ := 2010
def daily_donation : ℕ := 10
def first_day_donation : ℕ := 0 -- where 0 represents Monday, 6 represents Sunday

def total_days : ℕ := total_amount / daily_donation

def last_donation_day_of_week : ℕ := (total_days % 7 + first_day_donation) % 7

theorem last_donation_on_saturday : last_donation_day_of_week = 5 := by
  -- Prove it by calculation
  sorry

end last_donation_on_saturday_l691_691679


namespace half_abs_diff_squares_l691_691272

theorem half_abs_diff_squares : 
  let a := 20 in
  let b := 15 in
  let square (x : ℕ) := x * x in
  let abs_diff := abs (square a - square b) in
  abs_diff / 2 = 87.5 := 
by
  let a := 20
  let b := 15
  let square (x : ℕ) := x * x
  let abs_diff := abs (square a - square b)
  have h1 : square a = 400 := by native_decide
  have h2 : square b = 225 := by native_decide
  have h3 : abs_diff = abs (400 - 225) := by simp [square, h1, h2]
  have h4 : abs_diff = 175 := by simp [h3]
  have h5 : abs_diff / 2 = 87.5 := by norm_num [h4]
  exact h5

end half_abs_diff_squares_l691_691272


namespace helium_balloon_height_l691_691791

theorem helium_balloon_height :
    let total_budget := 200
    let cost_sheet := 42
    let cost_rope := 18
    let cost_propane := 14
    let cost_per_ounce_helium := 1.5
    let height_per_ounce := 113
    let amount_spent := cost_sheet + cost_rope + cost_propane
    let money_left_for_helium := total_budget - amount_spent
    let ounces_helium := money_left_for_helium / cost_per_ounce_helium
    let total_height := height_per_ounce * ounces_helium
    total_height = 9492 := sorry

end helium_balloon_height_l691_691791


namespace correct_propositions_l691_691608

-- Definitions of the propositions
def proposition1 : Prop := (∀ b : ℕ, b^2 = 9 → b = 3)
def proposition2 : Prop := ∀ (T1 T2 : Triangle), ¬ congruent T1 T2 → ¬ (area T1 = area T2)
def proposition3 : Prop := ∀ (c : ℝ), c ≤ 1 → (∃ x : ℝ, x^2 + 2 * x + c = 0)
def proposition4 : Prop := (∀ (A B : Set α), (A ∪ B = A) → (B ⊆ A))

-- We now state the theorem to be proved
theorem correct_propositions : proposition3 ∧ proposition4 ∧ ¬proposition1 ∧ ¬proposition2 :=
by
  sorry

end correct_propositions_l691_691608


namespace complex_div_result_l691_691020

-- Define the imaginary unit i
def i : ℂ := complex.I

-- Define the main problem theorem
theorem complex_div_result (a b : ℝ) (h : (1 + i) / (1 - i) = a + b * i) : a + b = 1 :=
sorry

end complex_div_result_l691_691020


namespace half_abs_diff_squares_l691_691238

theorem half_abs_diff_squares : (1 / 2) * |20^2 - 15^2| = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691238


namespace intersection_A_B_l691_691501

open Set

def A : Set ℕ := { x | x * (x - 3) ≤ 0 }
def B : Set ℤ := {-1, 0, 1}

theorem intersection_A_B :
  (A ∩ B) = {0, 1} := by
  sorry

end intersection_A_B_l691_691501


namespace pipe_fill_time_without_leak_l691_691994

theorem pipe_fill_time_without_leak (T : ℕ) :
  let pipe_with_leak_time := 10
  let leak_empty_time := 10
  ((1 / T : ℚ) - (1 / leak_empty_time) = (1 / pipe_with_leak_time)) →
  T = 5 := 
sorry

end pipe_fill_time_without_leak_l691_691994


namespace matrix_expression_l691_691454

variable {F : Type} [Field F] {n : Type} [Fintype n] [DecidableEq n]
variable (B : Matrix n n F)

-- Suppose B is invertible
variable [Invertible B]

-- Condition given in the problem
theorem matrix_expression (h : (B - 3 • (1 : Matrix n n F)) * (B - 5 • (1 : Matrix n n F)) = 0) :
  B + 10 • (B⁻¹) = 10 • (B⁻¹) + (32 / 3 : F) • (1 : Matrix n n F) :=
sorry

end matrix_expression_l691_691454


namespace restore_missing_digits_l691_691352

theorem restore_missing_digits :
  ∃ d1 d2 d3 : ℕ,
  let P := 1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11 in
  d1 ≠ 0 ∧ d2 ≠ 0 ∧ d3 ≠ 0 ∧ 
  P = 3990000 + d1 * 10000 + 6800 + d2 * 10 + d3∧ 
  d1 * 10000 + d2 * 10 + d3 = 16800 := 
  ∃ d1 d2 d3, d1 = 1 ∧ d2 = 6 ∧ d3 = 8.
  
sorry

end restore_missing_digits_l691_691352


namespace exists_positive_integers_x_y_such_that_sums_of_powers_are_squares_l691_691711

theorem exists_positive_integers_x_y_such_that_sums_of_powers_are_squares :
  ∃ (x y : ℕ), (0 < x) ∧ (0 < y) ∧ (∃ k₁ k₂ k₃ : ℕ, (x + y) = k₁^2 ∧ (x^2 + y^2) = k₂^2 ∧ (x^3 + y^3) = k₃^2) :=
by
  use 184, 305
  split; norm_num
  use 23, 391, 299
  split; norm_num
  split; norm_num
  sorry  -- Skipping detailed proof steps

end exists_positive_integers_x_y_such_that_sums_of_powers_are_squares_l691_691711


namespace polynomial_two_distinct_negative_real_roots_l691_691727

theorem polynomial_two_distinct_negative_real_roots :
  ∀ (p : ℝ), 
  (∃ (x1 x2 : ℝ), x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ 
    (x1^4 + p*x1^3 + 3*x1^2 + p*x1 + 4 = 0) ∧ 
    (x2^4 + p*x2^3 + 3*x2^2 + p*x2 + 4 = 0)) ↔ 
  (p ≤ -2 ∨ p ≥ 2) :=
by
  sorry

end polynomial_two_distinct_negative_real_roots_l691_691727


namespace calculate_10_odot_5_l691_691921

def odot (a b : ℚ) : ℚ := a + (4 * a) / (3 * b)

theorem calculate_10_odot_5 : odot 10 5 = 38 / 3 := by
  sorry

end calculate_10_odot_5_l691_691921


namespace half_abs_diff_squares_l691_691241

theorem half_abs_diff_squares : (1 / 2) * |20^2 - 15^2| = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691241


namespace complex_product_symmetric_imaginary_axis_l691_691699

-- Define the complex numbers and conditions
def z1 : ℂ := 2 + I
def z2 : ℂ := -2 + I

-- Prove the given problem
theorem complex_product_symmetric_imaginary_axis :
  z1 * z2 = -5 := 
by sorry

end complex_product_symmetric_imaginary_axis_l691_691699


namespace find_y_l691_691903

theorem find_y (y : ℝ) (h : sqrt (9 + sqrt (4 * y - 5)) = sqrt 10) : y = 1.5 := 
by
  sorry

end find_y_l691_691903


namespace closest_to_one_l691_691629

theorem closest_to_one:
  let a1 := 11/10
  let a2 := 111/100
  let a3 := 1.101
  let a4 := 1111/1000
  let a5 := 1.011
  in |a5 - 1| < |a1 - 1| ∧ |a5 - 1| < |a2 - 1| ∧ |a5 - 1| < |a3 - 1| ∧ |a5 - 1| < |a4 - 1| :=
  by
  sorry

end closest_to_one_l691_691629


namespace washed_shirts_l691_691147

-- Definitions based on the conditions
def short_sleeve_shirts : ℕ := 39
def long_sleeve_shirts : ℕ := 47
def unwashed_shirts : ℕ := 66

-- The total number of shirts is the sum of short and long sleeve shirts
def total_shirts : ℕ := short_sleeve_shirts + long_sleeve_shirts

-- The problem to prove that Oliver washed 20 shirts
theorem washed_shirts :
  total_shirts - unwashed_shirts = 20 := 
sorry

end washed_shirts_l691_691147


namespace parabola_c_value_l691_691671

theorem parabola_c_value (b c : ℝ) 
  (h1 : 2 * b + c = 6) 
  (h2 : -2 * b + c = 2)
  (vertex_cond : ∃ x y : ℝ, y = x^2 + b * x + c ∧ y = -x + 4) : 
  c = 4 :=
sorry

end parabola_c_value_l691_691671


namespace arithmetic_sum_property_l691_691520

variable {a : ℕ → ℤ} -- declare the sequence as a sequence of integers

-- Define the condition of the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 2 - a 1

-- Given condition: sum of specific terms in the sequence equals 400
def sum_condition (a : ℕ → ℤ) : Prop :=
  a 3 + a 4 + a 5 + a 6 + a 7 = 400

-- The goal: if the sum_condition holds, then a_2 + a_8 = 160
theorem arithmetic_sum_property
  (h_sum : sum_condition a)
  (h_arith : arithmetic_sequence a) :
  a 2 + a 8 = 160 := by
  sorry

end arithmetic_sum_property_l691_691520


namespace half_abs_diff_squares_l691_691242

theorem half_abs_diff_squares : (1 / 2) * |20^2 - 15^2| = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691242


namespace q_value_l691_691045

theorem q_value (p q : ℝ) (hpq1 : 1 < p) (hpql : p < q) (hq_condition : (1 / p) + (1 / q) = 1) (hpq2 : p * q = 8) : q = 4 + 2 * Real.sqrt 2 :=
  sorry

end q_value_l691_691045


namespace cost_per_book_l691_691617

theorem cost_per_book (num_animal_books : ℕ) (num_space_books : ℕ) (num_train_books : ℕ) (total_cost : ℕ) 
                      (h1 : num_animal_books = 10) (h2 : num_space_books = 1) (h3 : num_train_books = 3) (h4 : total_cost = 224) :
  (total_cost / (num_animal_books + num_space_books + num_train_books) = 16) :=
by sorry

end cost_per_book_l691_691617


namespace log_inequality_solution_l691_691642

theorem log_inequality_solution (a x : ℝ) (h : 0 < a ∧ a < 1/4) :
  1 + log 5 (x^2 + 1) ≥ log 5 (a * x^2 + 4 * x + a) :=
sorry

end log_inequality_solution_l691_691642


namespace domain_of_f_l691_691185

noncomputable def domain (k : ℤ) : set ℝ :=
  {x | 2 * k * π - π / 6 ≤ x ∧ x ≤ 2 * k * π + π / 6}

theorem domain_of_f :
  f : ℝ → ℝ := λ x, real.sqrt (real.cos x - (real.sqrt 3) / 2)
  ∃ k : ℤ, f x = real.sqrt (real.cos x - (real.sqrt 3) / 2) → 2 * k * π - π / 6 ≤ x ∧ x ≤ 2 * k * π + π / 6 := 
sorry

end domain_of_f_l691_691185


namespace angle_measure_l691_691755

noncomputable def isosceles_triangle (B1 B2 B3 : Type*) [MetricSpace B1] :=
  dist B1 B2 = dist B2 B3

noncomputable def midpoint (B_n B_n1 B_n3 : Type*) [MetricSpace B_n] :=
  B_n3 = (B_n + B_n1) / 2

theorem angle_measure (B : Type*) [MetricSpace B]
  (B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11 : B)
  (iso : isosceles_triangle B1 B2 B3)
  (A1 : ∠B1 B2 B3 = 100) -- angles are understood to be in degrees
  (M1 : midpoint B1 B2 B4)
  (M2 : midpoint B2 B3 B5)
  (M3 : midpoint B3 B1 B6)
  (M4 : midpoint B4 B5 B7)
  (M5 : midpoint B5 B6 B8)
  (M6 : midpoint B6 B4 B9)
  (M7 : midpoint B7 B8 B10)
  (M8 : midpoint B8 B9 B11) :
  ∠B10 B11 B9 = 100 :=
sorry

end angle_measure_l691_691755


namespace single_digit_in_12_steps_l691_691669

def operation (n : ℕ) : ℕ :=
  -- operation is a placeholder function that represents splitting and summing operation described
  sorry

noncomputable def reduce_to_single_digit : ℕ → ℕ
| n := 
  if n < 10 then n
  else reduce_to_single_digit (operation n)

theorem single_digit_in_12_steps (N : ℕ) : ∃ (k : ℕ), k ≤ 12 ∧ reduce_to_single_digit N < 10 :=
sorry

end single_digit_in_12_steps_l691_691669


namespace time_difference_l691_691212

/-- The time on a digital clock is 5:55. We need to calculate the number
of minutes that will pass before the clock next shows a time with all digits identical,
which is 11:11. -/
theorem time_difference : 
  let t1 := 5 * 60 + 55  -- Time 5:55 in minutes past midnight
  let t2 := 11 * 60 + 11 -- Time 11:11 in minutes past midnight
  in t2 - t1 = 316 := 
by 
  let t1 := 5 * 60 + 55 
  let t2 := 11 * 60 + 11 
  have h : t2 - t1 = 316 := by sorry
  exact h

end time_difference_l691_691212


namespace geom_mean_inequality_l691_691553

-- Lean Proposition
theorem geom_mean_inequality (A B C : Angle) (GC : ∃ D : Point, (D ∈ AB) ∧ (CD^2 = AD * BD)) :
  sqrt (sin A * sin B) ≤ sin (C / 2) :=
sorry

end geom_mean_inequality_l691_691553


namespace min_value_abc2_l691_691567

variables (a b c d : ℝ)

def condition_1 : Prop := a + b = 9 / (c - d)
def condition_2 : Prop := c + d = 25 / (a - b)

theorem min_value_abc2 :
  condition_1 a b c d → condition_2 a b c d → (a^2 + b^2 + c^2 + d^2) = 34 :=
by
  intros h1 h2
  sorry

end min_value_abc2_l691_691567


namespace true_propositions_l691_691772

def p1 (a b : ℝ) : Prop := a < b → a^2 < b^2
def p2 (x : ℝ) : Prop := x > 0 → sin x < x
def p3 (f : ℝ → ℝ) : Prop := (∀ x, f x / f (-x) = -1) ↔ ∀ x, f (-x) = -f x
def p4 (a : ℕ → ℝ) : Prop := (∀ n, a n > a (n + 1)) ↔ (∀ n, (a (n + 1)) / (a n) < 1)

def proposition_correctness : Prop := (¬(∀ a b, p1 a b)) ∧ (∀ x, p2 x) ∧ (¬(∀ f, p3 f)) ∧ (∀ a, p4 a)

theorem true_propositions : proposition_correctness :=
sorry

end true_propositions_l691_691772


namespace find_b_for_perpendicular_lines_l691_691406

-- Assume the conditions as functions or definitions
def line1_slope := 3
def line2_slope (b : ℝ) := -b / 4

-- The theorem to be proved
theorem find_b_for_perpendicular_lines (b : ℝ) : (line1_slope * (line2_slope b) = -1) → b = 4 / 3 :=
by
  sorry

end find_b_for_perpendicular_lines_l691_691406


namespace balloon_height_l691_691793

theorem balloon_height :
  let initial_money : ℝ := 200
  let cost_sheet : ℝ := 42
  let cost_rope : ℝ := 18
  let cost_tank_and_burner : ℝ := 14
  let helium_price_per_ounce : ℝ := 1.5
  let lift_per_ounce : ℝ := 113
  let remaining_money := initial_money - cost_sheet - cost_rope - cost_tank_and_burner
  let ounces_of_helium := remaining_money / helium_price_per_ounce
  let height := ounces_of_helium * lift_per_ounce
  height = 9492 :=
by
  sorry

end balloon_height_l691_691793


namespace determine_polynomial_l691_691301

open Polynomial

noncomputable def polynomial_example (n : ℕ) (P : Polynomial ℝ × Polynomial ℝ → Polynomial ℝ) :=
  (∀ t x y, P (t * x, t * y) = t^n * P (x, y)) ∧
  (∀ a b c, P (a + b, c) + P (b + c, a) + P (c + a, b) = 0) ∧
  (P (1, 0) = 1)

theorem determine_polynomial (n : ℕ) (P : Polynomial ℝ × Polynomial ℝ → Polynomial ℝ) :
  polynomial_example n P →
  P = λ xy, ((xy.1 + xy.2)^(n-1) * (xy.1 - 2 * xy.2)) :=
by { sorry }

end determine_polynomial_l691_691301


namespace inverse_square_value_l691_691813

def g (x : ℝ) : ℝ := 25 / (4 + 2 * x)

theorem inverse_square_value : (g⁻¹ 5)^2 = 1 / 4 :=
sorry

end inverse_square_value_l691_691813


namespace min_value_abs_complex_expr_correct_l691_691006

noncomputable def min_value_abs_complex_expr : Prop :=
  ∃ Z : ℂ, abs Z = 1 ∧ abs (Z^2 - 2 * Z + 1) = 0

theorem min_value_abs_complex_expr_correct : min_value_abs_complex_expr :=
by
  sorry

end min_value_abs_complex_expr_correct_l691_691006


namespace sqrt_x_minus_2_domain_l691_691836

theorem sqrt_x_minus_2_domain {x : ℝ} : (∃y : ℝ, y = Real.sqrt (x - 2)) ↔ x ≥ 2 :=
by sorry

end sqrt_x_minus_2_domain_l691_691836


namespace tan_pi_over_4_plus_alpha_l691_691445

theorem tan_pi_over_4_plus_alpha (α : ℝ) 
  (h : Real.tan (Real.pi / 4 + α) = 2) : 
  1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 5 / 7 := 
by {
  sorry
}

end tan_pi_over_4_plus_alpha_l691_691445


namespace number_of_yellow_parrots_l691_691614

theorem number_of_yellow_parrots (total_parrots red_fraction green_fraction : ℚ) 
    (h1 : red_fraction = 2 / 3) 
    (h2 : green_fraction = 1 / 6) 
    (h3 : total_parrots = 180) 
    (h4 : red_fraction + green_fraction < 1) :
    let yellow_fraction := 1 - (red_fraction + green_fraction) in 
    let yellow_parrots := yellow_fraction * total_parrots in 
    yellow_parrots = 30 :=
by
  -- Definitions to use the provided conditions
  let yellow_fraction := 1 - (red_fraction + green_fraction)
  let yellow_parrots := yellow_fraction * total_parrots
  -- Sorry is used to skip the actual proof
  sorry

end number_of_yellow_parrots_l691_691614


namespace car_speed_on_local_roads_l691_691310

theorem car_speed_on_local_roads
    (v : ℝ) -- Speed of the car on local roads
    (h1 : v > 0) -- The speed is positive
    (h2 : 40 / v + 3 = 5) -- Given equation based on travel times and distances
    : v = 20 := 
sorry

end car_speed_on_local_roads_l691_691310


namespace line_through_point_determines_a_l691_691591

theorem line_through_point_determines_a (a : ℝ) :
  (∃ (x y : ℝ), x = 1 ∧ y = 3 ∧ a * x - y - 1 = 0) → a = 4 :=
by
  intro h
  cases h with x hx
  cases hx with y hy
  cases hy with hx1 hy1
  cases hy1 with hy2 hxy
  rw [hx1, hy2] at hxy
  -- Remaining proof omitted
  sorry

end line_through_point_determines_a_l691_691591


namespace find_angle_A_find_triangle_area_l691_691503

open Real

noncomputable def triangle_angle_a (a b c A B C : ℝ) : Prop :=
  c * sin (π / 2 - A) = (a * cos B + b * cos A) / 2

noncomputable def triangle_area (a b c A : ℝ) : ℝ :=
  (1 / 2) * b * c * sin A

theorem find_angle_A (a b c A B C : ℝ) 
  (h1a : triangle_angle_a a b c A B C)
  (h1b : 2 * a = b + c)
  (h1c : a = 2 * sin A) 
  (h1d : sin C ≠ 0)
  (h1e : ∀ x, 0 < x → sin x < 1) :
  A = π / 3 :=
by
  sorry

theorem find_triangle_area (a b c A B C : ℝ)
  (h1a : triangle_angle_a a b c A B C)
  (h2a : 2 * a = b + c)
  (h2b : a = 2)
  (h2c : a^2 = b^2 + c^2 - 2 * b * c * cos (π / 3))
  (h2d : b * c = 3) :
  triangle_area a b c A = (3 * √3) / 4 :=
by
  sorry

end find_angle_A_find_triangle_area_l691_691503


namespace half_abs_diff_squares_l691_691239

theorem half_abs_diff_squares : (1 / 2) * |20^2 - 15^2| = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691239


namespace storage_ratio_correct_l691_691694

--	definition of the problem
def radius (diameter : ℝ) : ℝ := diameter / 2
def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h
def effective_storage (V : ℝ) (fill_percentage : ℝ) : ℝ := fill_percentage * V

noncomputable def bryans_silo_volume := volume_cylinder (radius 8) 20
noncomputable def saras_silo_volume := volume_cylinder (radius 10) 16

noncomputable def bryans_effective_storage := effective_storage bryans_silo_volume 0.9
noncomputable def saras_effective_storage := effective_storage saras_silo_volume 0.85

noncomputable def storage_ratio := bryans_effective_storage / saras_effective_storage

theorem storage_ratio_correct : storage_ratio = 18 / 17 := by
  unfold storage_ratio bryans_effective_storage saras_effective_storage effective_storage
  unfold bryans_silo_volume saras_silo_volume volume_cylinder radius
  norm_num
  sorry

end storage_ratio_correct_l691_691694


namespace find_m_l691_691416

variable {α : Type}
variable f : α → ℝ
variable m : α

theorem find_m
  (h1 : ∀ x, f (x / 2 - 1) = 2 * x + 3)
  (h2 : f m = 6) :
  m = -1 / 4 :=
sorry

end find_m_l691_691416


namespace grace_mowing_hours_l691_691048

-- Definitions for conditions
def earnings_mowing (x : ℕ) : ℕ := 6 * x
def earnings_weeds : ℕ := 11 * 9
def earnings_mulch : ℕ := 9 * 10
def total_september_earnings (x : ℕ) : ℕ := earnings_mowing x + earnings_weeds + earnings_mulch

-- Proof statement (with the total earnings of 567 specified)
theorem grace_mowing_hours (x : ℕ) (h : total_september_earnings x = 567) : x = 63 := by
  sorry

end grace_mowing_hours_l691_691048


namespace all_points_lie_on_two_circles_l691_691830

open set

theorem all_points_lie_on_two_circles (P : set (ℝ × ℝ)) (n : ℕ)
  (h₁ : n ≥ 9)
  (h₂ : ∀ S ⊆ P, S.card = 9 → ∃ c₁ c₂ : set (ℝ × ℝ), is_circle c₁ ∧ is_circle c₂ ∧ ∀ p ∈ S, p ∈ c₁ ∪ c₂) :
  ∃ c₁ c₂ : set (ℝ × ℝ), is_circle c₁ ∧ is_circle c₂ ∧ ∀ p ∈ P, p ∈ c₁ ∪ c₂ :=
sorry

end all_points_lie_on_two_circles_l691_691830


namespace intersection_is_correct_union_is_correct_l691_691040
noncomputable theory

-- Define the sets A and B
def A : set ℝ := { y | ∃ x, y = x^2 - 4x + 3 }
def B : set ℝ := { y | ∃ x, y = -x^2 - 2x }

-- Define the expected results for intersection and union
def intersection : set ℝ := { y | -1 <= y ∧ y <= 1 }
def union : set ℝ := set.univ

-- Define the theorem to be proved
theorem intersection_is_correct : (A ∩ B) = intersection := by sorry

theorem union_is_correct : (A ∪ B) = union := by sorry

end intersection_is_correct_union_is_correct_l691_691040


namespace initial_population_l691_691923

theorem initial_population (P : ℝ)
  (H1 : 1.60 * P = P + 0.60 * P)
  (H2 : ∑ i from 1 to 10, 2000 = 20000)
  (H3 : ∑ i from 1 to 10, 2500 = 25000)
  (H4 : 1.60 * P + (25000 - 20000) = 165000) :
  P = 100000 := 
by
  sorry

end initial_population_l691_691923


namespace Lloyd_worked_hours_l691_691872

variable (regular_hours : ℝ) (pay_rate : ℝ) (overtime_rate : ℝ) (total_earnings : ℝ)

theorem Lloyd_worked_hours : 
  let regular_hours := 7.5,
      pay_rate := 4,
      overtime_rate := 1.5 * pay_rate,
      total_earnings := 48 in
  let regular_pay := regular_hours * pay_rate in
  let excess_earnings := total_earnings - regular_pay in
  let overtime_hours := excess_earnings / overtime_rate in
  let total_hours_worked := regular_hours + overtime_hours in
  total_hours_worked = 10.5 := 
by
  sorry

end Lloyd_worked_hours_l691_691872


namespace john_sarah_money_total_l691_691540

theorem john_sarah_money_total (j_money s_money : ℚ) (H1 : j_money = 5/8) (H2 : s_money = 7/16) :
  (j_money + s_money : ℚ) = 1.0625 := 
by
  sorry

end john_sarah_money_total_l691_691540


namespace cost_per_piece_is_two_l691_691112

-- Define the conditions
def pieces_per_section : ℕ := 30
def number_of_sections : ℕ := 8
def total_cost : ℝ := 480
def total_pieces : ℕ := pieces_per_section * number_of_sections
def cost_per_piece : ℝ := total_cost / total_pieces

-- State the theorem that we want to prove
theorem cost_per_piece_is_two :
  cost_per_piece = 2 :=
by
  -- Proof goes here
  sorry

end cost_per_piece_is_two_l691_691112


namespace limit_f_at_1_l691_691377

-- Define the function f(x)
def f (x : ℝ) : ℝ := (ln (e * x))^2^(1/(x^2 + 1))

-- Conditions
axiom limit_of_x : filter.tendsto (λ (x : ℝ), x) filter.cofinite 1
axiom ln_ex_eq : ∀ (x : ℝ), ln (e * x) = 1 + ln x

-- The theorem with the precise statement
theorem limit_f_at_1 : filter.tendsto (λ (x : ℝ), f x) filter.cofinite 1 :=
by
  sorry -- proof goes here

end limit_f_at_1_l691_691377


namespace zits_difference_l691_691917

variable (avg_zits_swanson : ℕ)
variable (num_students_swanson : ℕ)
variable (avg_zits_jones : ℕ)
variable (num_students_jones : ℕ)

-- Conditions
def total_zits_swanson := avg_zits_swanson * num_students_swanson
def total_zits_jones := avg_zits_jones * num_students_jones

-- Theorem to prove the difference in total zits
theorem zits_difference : 
  avg_zits_swanson = 5 → 
  num_students_swanson = 25 → 
  avg_zits_jones = 6 → 
  num_students_jones = 32 → 
  total_zits_jones avg_zits_jones num_students_jones - total_zits_swanson avg_zits_swanson num_students_swanson = 67 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  show 6 * 32 - 5 * 25 = 67
  norm_num
  sorry

end zits_difference_l691_691917


namespace relationship_among_sets_l691_691863

-- Definitions based on the conditions
def RegularQuadrilateralPrism (x : Type) : Prop := -- prisms with a square base and perpendicular lateral edges
  sorry

def RectangularPrism (x : Type) : Prop := -- prisms with a rectangular base and perpendicular lateral edges
  sorry

def RightQuadrilateralPrism (x : Type) : Prop := -- prisms whose lateral edges are perpendicular to the base, and the base can be any quadrilateral
  sorry

def RightParallelepiped (x : Type) : Prop := -- prisms with lateral edges perpendicular to the base
  sorry

-- Sets
def M : Set Type := { x | RegularQuadrilateralPrism x }
def P : Set Type := { x | RectangularPrism x }
def N : Set Type := { x | RightQuadrilateralPrism x }
def Q : Set Type := { x | RightParallelepiped x }

-- Proof problem statement
theorem relationship_among_sets : M ⊂ P ∧ P ⊂ Q ∧ Q ⊂ N := 
  by
    sorry

end relationship_among_sets_l691_691863


namespace revision_cost_is_3_l691_691566

def cost_first_time (pages : ℕ) : ℝ := 5 * pages

def cost_for_revisions (rev1 rev2 : ℕ) (rev_cost : ℝ) : ℝ := (rev1 * rev_cost) + (rev2 * 2 * rev_cost)

def total_cost (pages rev1 rev2 : ℕ) (rev_cost : ℝ) : ℝ := 
  cost_first_time pages + cost_for_revisions rev1 rev2 rev_cost

theorem revision_cost_is_3 :
  ∀ (pages rev1 rev2 : ℕ) (total : ℝ),
      pages = 100 →
      rev1 = 30 →
      rev2 = 20 →
      total = 710 →
      total_cost pages rev1 rev2 3 = total :=
by
  intros pages rev1 rev2 total pages_eq rev1_eq rev2_eq total_eq
  sorry

end revision_cost_is_3_l691_691566


namespace smallest_prime_with_conditions_l691_691401

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n
def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def reverse_digits (n : ℕ) : ℕ := 
  let tens := n / 10 
  let units := n % 10 
  units * 10 + tens

theorem smallest_prime_with_conditions : 
  ∃ (p : ℕ), is_prime p ∧ 20 ≤ p ∧ p < 30 ∧ (reverse_digits p) < 100 ∧ is_composite (reverse_digits p) ∧ p = 23 :=
by
  sorry

end smallest_prime_with_conditions_l691_691401


namespace total_volume_of_barrel_l691_691971

-- Define the total volume of the barrel and relevant conditions.
variable (x : ℝ) -- total volume of the barrel

-- State the given condition about the barrel's honey content.
def condition := (0.7 * x - 0.3 * x = 30)

-- Goal to prove:
theorem total_volume_of_barrel : condition x → x = 75 :=
by
  sorry

end total_volume_of_barrel_l691_691971


namespace ratio_of_AQ_BQ_l691_691611

variables {A B C Q : Type} -- Points A, B, C, Q are of some type

-- Define the conditions of the problem
variables [right_angle : ∠C = 90°] 
variables [angle_BAC : ∠BAC = 30°] 
variables [length_AB : AB = 6] 
variables [point_on_AB : Q ∈ [A, B]] 
variables [angle_condition : ∠AQC = 2 * ∠ACQ]
variables [length_CQ : CQ = 2]

-- Statement that matches the conditions
theorem ratio_of_AQ_BQ : AQ / BQ = 20 + 3 * sqrt 31 :=
sorry

end ratio_of_AQ_BQ_l691_691611


namespace average_interest_rate_is_3_75_percent_l691_691328

variables (x : ℝ) (total_principal : ℝ) (interest_rate_3 : ℝ) (interest_rate_5 : ℝ) (interest_3 : ℝ) (interest_5 : ℝ)
variables (total_interest : ℝ) (average_rate : ℝ)

-- Given conditions
def amounts_invested_at_different_rates (total_principal : ℝ) (x : ℝ) : Prop :=
  total_principal = 6000 ∧ (x = 2250 ∧ interest_rate_5 = 0.05 ∧ interest_rate_3 = 0.03)

def equal_annual_returns (x : ℝ) : Prop := 
  interest_3 = 0.03 * (6000 - x) ∧ interest_5 = 0.05 * x ∧ interest_3 = interest_5

-- To prove
theorem average_interest_rate_is_3_75_percent :
  ∀ (x : ℝ) (total_principal : ℝ),
    amounts_invested_at_different_rates total_principal x →
    equal_annual_returns x →
    average_rate = (interest_3 + interest_5) / total_principal →
    average_rate = 0.0375 :=
by
  intros
  sorry

end average_interest_rate_is_3_75_percent_l691_691328


namespace minimum_value_of_x_plus_y_l691_691819

theorem minimum_value_of_x_plus_y
  (x y : ℝ)
  (h1 : x > y)
  (h2 : y > 0)
  (h3 : 1 / (x - y) + 8 / (x + 2 * y) = 1) :
  x + y = 25 / 3 :=
sorry

end minimum_value_of_x_plus_y_l691_691819


namespace solution_set_l691_691383

noncomputable def f : ℝ → ℝ := sorry -- Assume the existence of f satisfying given conditions

lemma f_conditions :
  ∀ x : ℝ, f'' x > 1 - f x ∧ f 0 = 6 ∧ (∀ x, (f' x) = (deriv f x)) := sorry

theorem solution_set :
  { x : ℝ | e^x * f x > e^x + 5 } = set.Ioi (0) :=
by
  intro x
  simp only [set.mem_set_of_eq, set.mem_Ioi]
  split
  { intro h
    -- Use given conditions
    have h_f := f_conditions x,
    sorry },
  { intro h
    -- Proof that e^x * f x > e^x + 5 given x > 0
    sorry }

end solution_set_l691_691383


namespace solve_equation_l691_691173

-- Define the equation as a function f
def f (x : ℝ) : ℝ :=
  1/x + 1/(x+2) - 1/(x+4) - 1/(x+6) - 1/(x+8) - 1/(x+10) + 1/(x+12) + 1/(x+14)

-- Define the theorem to state the solutions to the equation f(x) = 0
theorem solve_equation :
  (f (-7) = 0) ∧
  (f (-7 + sqrt (19 + 6 * sqrt 5)) = 0 ∨ f (-7 - sqrt (19 + 6 * sqrt 5)) = 0 ∨ f (-7 + sqrt (19 - 6 * sqrt 5)) = 0 ∨ f (-7 - sqrt (19 - 6 * sqrt 5)) = 0) := by
  -- Proof will be inserted here
  sorry

end solve_equation_l691_691173


namespace incorrect_comparison_tan_138_tan_143_l691_691707

theorem incorrect_comparison_tan_138_tan_143 :
  ¬ (Real.tan (Real.pi * 138 / 180) > Real.tan (Real.pi * 143 / 180)) :=
by sorry

end incorrect_comparison_tan_138_tan_143_l691_691707


namespace part1_part2_l691_691420

variables {p m : ℝ}

-- Definitions for given problem conditions
def parabola_eq (y x : ℝ) (p : ℝ) : Prop := y^2 = 4 * p * x
def line_bc_eq (x y : ℝ) : Prop := 4 * x + y - 20 = 0
def centroid_eq_focus (x₁ x₂ x₃ y₁ y₂ y₃ p : ℝ) : Prop :=
  (x₁ + x₂ + x₃) / 3 = p ∧ (y₁ + y₂ + y₃) / 3 = 0

-- Tangent line and perpendicular condition
def tangent_slope (m : ℝ) : ℝ := 8 / m
def perpendicular_slope (m : ℝ) : ℝ := - m / 8
def angle_lambda_eq_one (m : ℝ) : Prop :=
  let k := tangent_slope m in
  let mn_slope := perpendicular_slope m in
  ∃ λ : ℝ, λ = 1

-- Proof statements
theorem part1 : ∃ (p : ℝ), ∀ (x y : ℝ), 
  centroid_eq_focus x₁ x₂ x₃ y₁ y₂ y₃ p ∧ line_bc_eq x₂ y₂ → 
  parabola_eq y x 4 :=
sorry

theorem part2 : ∃ (λ : ℝ), ∀ (m : ℝ),
  angle_lambda_eq_one m :=
sorry

end part1_part2_l691_691420


namespace part1_even_function_part2_min_value_l691_691757

variable {a x : ℝ}

def f (x a : ℝ) : ℝ := x^2 + |x - a| + 1

theorem part1_even_function (h : a = 0) : 
  ∀ x : ℝ, f x 0 = f (-x) 0 :=
by
  -- This statement needs to be proved to show that f(x) is even when a = 0
  sorry

theorem part2_min_value (h : true) : 
  (a > (1/2) → ∃ x : ℝ, f x a = a + (3/4)) ∧
  (a ≤ -(1/2) → ∃ x : ℝ, f x a = -a + (3/4)) ∧
  ((- (1/2) < a ∧ a ≤ (1/2)) → ∃ x : ℝ, f x a = a^2 + 1) :=
by
  -- This statement needs to be proved to show the different minimum values of the function
  sorry

end part1_even_function_part2_min_value_l691_691757


namespace find_r_l691_691661

theorem find_r (r : ℝ) (AB AD BD : ℝ) (circle_radius : ℝ) (main_circle_radius : ℝ) :
  main_circle_radius = 2 →
  circle_radius = r →
  AB = 2 * r →
  AD = 2 * r →
  BD = 4 + 2 * r →
  (2 * r)^2 + (2 * r)^2 = (4 + 2 * r)^2 →
  r = 4 :=
by 
  intros h_main_radius h_circle_radius h_AB h_AD h_BD h_pythagorean
  sorry

end find_r_l691_691661


namespace equation_of_C_min_length_of_AB_l691_691749

open Real 

-- Define the conditions for Ellipse C
def EllipseC (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

noncomputable def right_focus := (sqrt 2, 0)
noncomputable def inclination := π / 4

-- Calculate the equation of Ellipse C
theorem equation_of_C :
  EllipseC x y ↔ (x^2 / 4 + y^2 / 2 = 1) := sorry

-- Additional conditions for Question 2
variable (A : ℝ × ℝ)
variable (B : ℝ × ℝ)
variable (O : ℝ × ℝ)

-- Given A = (t, 2), B is on ellipse C, OA ⟂ OB
def A_on_line_y_equals_2 : Prop := A.2 = 2
def B_on_ellipse_C : Prop := EllipseC B.1 B.2
def O : (0, 0)
def OA_perp_OB (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

-- Find minimum length of segment AB
theorem min_length_of_AB (hA : A_on_line_y_equals_2 (t, 2)) (hB : B_on_ellipse_C (x0, y0)) (hP : OA_perp_OB (t, 2) (x0, y0)) :
  (t, 2) = A → (x0, y0) = B → O = (0, 0) → min_length_of_AB = 4 := sorry

end equation_of_C_min_length_of_AB_l691_691749


namespace linear_equation_unique_l691_691291

theorem linear_equation_unique (x y : ℝ) : 
  (3 * x = 2 * y) ∧ 
  ¬(3 * x - 6 = x) ∧ 
  ¬(x - 1 / y = 0) ∧ 
  ¬(2 * x - 3 * y = x * y) :=
by
  sorry

end linear_equation_unique_l691_691291


namespace perpendicular_lines_l691_691786

/- 
Given lines l1: 2*x + a*y = 0 and l2: x - (a + 1)*y = 0,
prove that if l1 is perpendicular to l2, then a = -2 or a = 1.
-/

theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, 2 * x + a * y = 0 → x - (a + 1) * y = 0 → is_perpendicular 2 a 1 (a + 1)) → (a = -2 ∨ a = 1) :=
by 
  sorry

/-- Helper definition to check if lines are perpendicular based on slopes -/
def is_perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

end perpendicular_lines_l691_691786


namespace proof_problem_l691_691751

variable {n : ℕ}
variable {mi : Fin n → ℝ}
variable {p : ℕ}

theorem proof_problem
  (h1 : ∀ i, 0 < mi i)
  (h2 : 2 ≤ p)
  (h3 : ∑ i, 1 / (1 + (mi i) ^ p) = 1) :
  (∏ i, mi i) ≥ (n - 1)^(n / p) :=
sorry

end proof_problem_l691_691751


namespace beth_sells_half_of_coins_l691_691359

theorem beth_sells_half_of_coins (x y : ℕ) (h₁ : x = 125) (h₂ : y = 35) : (x + y) / 2 = 80 :=
by
  sorry

end beth_sells_half_of_coins_l691_691359


namespace proof_theorem_l691_691164

-- Definitions based on identified conditions and problem
def f (x : ℝ) : ℝ := (2 * x + 1) / (x - 1)

-- The Lean theorem statement encapsulating the mathematical proof problem
theorem proof_theorem : 
  (∃! x, f x = 0) ∧
  (∀ y, y ∈ (set.range f) ↔ y ≠ 2) ∧
  (∀ x, f (2 - x) = -f x + 4) := 
sorry

end proof_theorem_l691_691164


namespace ana_salary_after_changes_l691_691685

-- Definitions based on conditions in part (a)
def initial_salary : ℝ := 2000
def raise_factor : ℝ := 1.20
def cut_factor : ℝ := 0.80

-- Statement of the proof problem
theorem ana_salary_after_changes : 
  (initial_salary * raise_factor * cut_factor) = 1920 :=
by
  sorry

end ana_salary_after_changes_l691_691685


namespace satisfies_condition_A_satisfies_condition_B_satisfies_condition_C_satisfies_condition_D_final_answer_is_ABD_l691_691961

noncomputable def f_A : ℝ → ℝ := λ x, -x⁻¹
noncomputable def f_B : ℝ → ℝ := λ x, (exp x - exp (-x)) / 2
noncomputable def f_C : ℝ → ℝ := λ x, 1 / (x^4 + 1)
noncomputable def f_D : ℝ → ℝ := λ x, Real.log (x + Real.sqrt (x^2 + 1))

theorem satisfies_condition_A (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) :
  (x1 - x2) * (f_A x2 - f_A x1) < 0 := sorry

theorem satisfies_condition_B (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) :
  (x1 - x2) * (f_B x2 - f_B x1) < 0 := sorry

theorem satisfies_condition_C (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) :
  ¬((x1 - x2) * (f_C x2 - f_C x1) < 0) := sorry

theorem satisfies_condition_D (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) :
  (x1 - x2) * (f_D x2 - f_D x1) < 0 := sorry

theorem final_answer_is_ABD :
  (∀ x1 x2, 0 < x1 → 0 < x2 → (x1 - x2) * (f_A x2 - f_A x1) < 0) ∧
  (∀ x1 x2, 0 < x1 → 0 < x2 → (x1 - x2) * (f_B x2 - f_B x1) < 0) ∧
  (∀ x1 x2, 0 < x1 → 0 < x2 → ¬((x1 - x2) * (f_C x2 - f_C x1) < 0)) ∧
  (∀ x1 x2, 0 < x1 → 0 < x2 → (x1 - x2) * (f_D x2 - f_D x1) < 0) := sorry

end satisfies_condition_A_satisfies_condition_B_satisfies_condition_C_satisfies_condition_D_final_answer_is_ABD_l691_691961


namespace parabola_distance_l691_691587

theorem parabola_distance :
  ∃ (P F : Real × Real), 
    (F = (4, 0)) ∧ -- focus of parabola
    (∃ A : Real × Real, 
      (A = (-4, 8)) ∧ -- foot of perpendicular from P 
      (P = (4, 8)) ∧ -- point P on the parabola
      (P.1, P.2) ∈ ({ p | p.2 ^ 2 = 16 * p.1 } : Set (ℝ × ℝ)) -- P satisfies parabola equation
    ) ∧
    (abs (sqrt ((P.1 - F.1) ^ 2 + (P.2 - F.2) ^ 2)) = 8) := -- distance between P and F
by {
  -- The proof goes here
  sorry
}

end parabola_distance_l691_691587


namespace ideal_matching_sets_count_is_27_l691_691139

def I : Set ℕ := {1, 2, 3, 4, 5, 6}

def ideal_matching_sets_count (A B : Set ℕ) : Prop :=
  A ∩ B = {1, 3, 5} ∧ A ⊆ I ∧ B ⊆ I

theorem ideal_matching_sets_count_is_27 : ∃ A B : Set ℕ, ideal_matching_sets_count A B → (Finset.univ.filter (λ (A B : Set ℕ), ideal_matching_sets_count A B)).card = 27 :=
by
  sorry

end ideal_matching_sets_count_is_27_l691_691139


namespace arithmetic_to_geometric_sequence_l691_691094

/-- Given the conditions that a_1 = 1, 
    a_2, a_4 - 2, a_6 form a geometric sequence, 
    and {a_n} is an arithmetic sequence with common difference d ≠ 0, 
    prove that the general formula for {a_n} is a_n = 3n - 2,
    and for b_n = 3^(a_n - 1), the sum of the first n terms S_n is (1/26)(27^n - 1). -/
theorem arithmetic_to_geometric_sequence
    (a : ℕ → ℤ) (b : ℕ → ℤ) (S : ℕ → ℤ)
    (h1 : a 1 = 1)
    (h2 : ∃ d : ℤ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d)
    (h3 : ∀ (a_2 a_4_minus_2 a_6 : ℤ), (a 2, a 4 - 2, a 6) = (a_2, a_4_minus_2, a_6) →
          a_4_minus_2^2 = a_2 * a_6)
    (h4 : ∀ n : ℕ, b n = 3^(a n - 1))
    (h5 : ∀ n : ℕ, b n = 3^(3n - 3))
    (h6 : ∀ n : ℕ, S n = (27^n - 1) / 26) :

    (∀ n : ℕ, a n = 3 * n - 2) ∧ (∀ n : ℕ, S n = (27^n - 1) / 26) :=
by 
  sorry

end arithmetic_to_geometric_sequence_l691_691094


namespace find_number_announced_eight_l691_691603

def numbers_picked {n : ℕ} (a : Fin n → ℕ) (total_sum : ℕ) : Prop :=
  (∑ i, a i) = total_sum 

def average_constraint {n : ℕ} (a : Fin n → ℕ) : Prop :=
  ∀ i : Fin n, 
  let j1 := if i.val = 0 then n - 1 else i.val - 1 in
  let j2 := (i.val + 1) % n in
  2 * i.val = a ⟨j1, sorry⟩ + a ⟨j2, sorry⟩

theorem find_number_announced_eight : 
  ∃ a : Fin 12 → ℕ, 
  numbers_picked a 78 ∧ 
  average_constraint a ∧ 
  a 7 = 10 :=
sorry

end find_number_announced_eight_l691_691603


namespace min_omega_value_l691_691761

theorem min_omega_value (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, sin (ω * x + π / 3) = -sin (ω * x - (4 * π * ω) / 5 + π / 3)) → 
  ω = 5 / 4 := by
  sorry

end min_omega_value_l691_691761


namespace rate_per_meter_proof_l691_691729

-- Defining the necessary constants and functions
def diameter : ℝ := 18
def total_cost : ℝ := 141.37
def π : ℝ := 3.14159

-- Calculate the circumference
def circumference : ℝ := π * diameter

-- Rate per meter calculation
def rate_per_meter : ℝ := total_cost / circumference

-- Theorem stating the required proof
theorem rate_per_meter_proof : rate_per_meter = 2.5 :=
  by
  -- Add the steps necessary to provide the proof for that rate_per_meter equals 2.5
  sorry

end rate_per_meter_proof_l691_691729


namespace fewer_hours_l691_691942

def Vb : ℝ := 20
def Va : ℝ := Vb + 5
def D : ℝ := 300

def Tb : ℝ := D / Vb
def Ta : ℝ := D / Va

theorem fewer_hours : Tb - Ta = 3 := by
  sorry

end fewer_hours_l691_691942


namespace fewer_buses_than_cars_l691_691924

theorem fewer_buses_than_cars
  (bus_to_car_ratio : ℕ := 1)
  (cars_on_river_road : ℕ := 65)
  (cars_per_bus : ℕ := 13) :
  cars_on_river_road - (cars_on_river_road / cars_per_bus) = 60 :=
by
  sorry

end fewer_buses_than_cars_l691_691924


namespace people_later_than_yoongi_l691_691648

variable (total_students : ℕ) (people_before_yoongi : ℕ)

theorem people_later_than_yoongi
    (h1 : total_students = 20)
    (h2 : people_before_yoongi = 11) :
    total_students - (people_before_yoongi + 1) = 8 := 
sorry

end people_later_than_yoongi_l691_691648


namespace maximum_PC_length_l691_691528

variable {A B C P: Type}
variable {AP BP a α: ℝ}

-- Assuming the conditions from the problem
def is_equilateral_triangle (ABC: Triangle) : Prop :=
  ABC.is_equilateral

def distance (P1 P2: Point) : ℝ := Point.distance P1 P2

def meets_conditions (P: Point) (A B: Point) (AP_length BP_length a_limit: ℝ) : Prop :=
  distance A P = AP_length ∧
  distance B P = BP_length ∧
  distance A B < a_limit

-- Formulating the maximization goal as a proof statement
theorem maximum_PC_length (ABC: Triangle) (P: Point)
  (h_eq: is_equilateral_triangle ABC)
  (h_cond: meets_conditions P A B 2 3 5) :
  ∃ C: Point, distance P C = (1 / 2) * sqrt (29 - α^2 + 2 * sqrt (3 * (26 * a^2 - a^4 - 25))) := sorry

end maximum_PC_length_l691_691528


namespace simplify_trigonometric_expression_l691_691900

noncomputable def trigonometric_simplification : Real :=
  (cos(5 * Real.pi / 180)^2 - sin(5 * Real.pi / 180)^2) / (sin(40 * Real.pi / 180) * cos(40 * Real.pi / 180))

theorem simplify_trigonometric_expression : trigonometric_simplification = 2 := by
  sorry

end simplify_trigonometric_expression_l691_691900


namespace smallest_palindrome_both_bases_is_33_l691_691708

def is_palindrome_base (n : ℕ) (b : ℕ) : Prop :=
  let digits := List.unfold (λ n, if n = 0 then none else some (n % b, n / b)) n
  digits = digits.reverse

def smallest_palindrome_both_bases : ℕ :=
  well_founded.fix (
    inv_image.wf nat.lt_wf (λ x, x)) (
    λ n f, if 10 < n ∧ is_palindrome_base n 3 ∧ is_palindrome_base n 5 then n else f (n+1)) 

theorem smallest_palindrome_both_bases_is_33 : smallest_palindrome_both_bases = 33 :=
sorry

end smallest_palindrome_both_bases_is_33_l691_691708


namespace peter_contains_five_l691_691886

theorem peter_contains_five (N : ℕ) (hN : N > 0) :
  ∃ k : ℕ, ∀ m : ℕ, m ≥ k → ∃ i : ℕ, 5 ≤ 10^i * (N * 5^m / 10^i) % 10 :=
sorry

end peter_contains_five_l691_691886


namespace vincent_books_cost_l691_691619

theorem vincent_books_cost :
  let num_animals := 10
  let num_outer_space := 1
  let num_trains := 3
  let total_books := num_animals + num_outer_space + num_trains
  let total_spent := 224
  let cost_per_book := total_spent / total_books
  cost_per_book = 16 :=
by
  let num_animals := 10
  let num_outer_space := 1
  let num_trains := 3
  let total_books := num_animals + num_outer_space + num_trains
  let total_spent := 224
  let cost_per_book := total_spent / total_books
  show cost_per_book = 16
  sorry

end vincent_books_cost_l691_691619


namespace points_in_square_disk_cover_l691_691418

theorem points_in_square_disk_cover (points : Fin 51 → ℝ × ℝ) (h_points : ∀ i, 0 ≤ points i.1 ∧ points i.1 ≤ 1 ∧ 0 ≤ points i.2 ∧ points i.2 ≤ 1) :
  ∃ (c : ℝ × ℝ), ∃ (r : ℝ), r = 1 / 7 ∧ (∑ i in Fin 51, if dist (points i) c ≤ r then 1 else 0) ≥ 3 :=
by
  sorry

end points_in_square_disk_cover_l691_691418


namespace tangent_line_at_one_unique_zero_of_f_exists_lower_bound_of_f_l691_691747

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + x * Real.exp x - Real.exp 1

-- Part (Ⅰ)
theorem tangent_line_at_one (h_a : a = 0) : ∃ m b : ℝ, ∀ x : ℝ, 2 * Real.exp 1 * x - y - 2 * Real.exp 1 = 0 := sorry

-- Part (Ⅱ)
theorem unique_zero_of_f (h_a : a > 0) : ∃! t : ℝ, f a t = 0 := sorry

-- Part (Ⅲ)
theorem exists_lower_bound_of_f (h_a : a < 0) : ∃ m : ℝ, ∀ x : ℝ, f a x ≥ m := sorry

end tangent_line_at_one_unique_zero_of_f_exists_lower_bound_of_f_l691_691747


namespace find_f_l691_691460

theorem find_f (f : ℝ → ℝ) (h : ∀ x : ℝ, f (Real.sqrt x + 4) = x + 8 * Real.sqrt x) :
  ∀ (x : ℝ), x ≥ 4 → f x = x^2 - 16 :=
by
  sorry

end find_f_l691_691460


namespace infinite_initial_values_l691_691858

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x - x^2

-- Define the sequence x_n = f(x_(n-1))
def sequence (x0 : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then x0 else f (sequence x0 (n - 1))

-- Since the problem states "finite number of different values", we need to use finiteness
def takes_finite_values (x0 : ℝ) : Prop :=
  ∃ (S : Set ℝ), Set.Finite S ∧ ∀ (n : ℕ), sequence x0 n ∈ S

-- The statement to prove
theorem infinite_initial_values :
  ∃ (X : Set ℝ), Set.Infinite X ∧ ∀ x0 ∈ X, takes_finite_values x0 :=
sorry

end infinite_initial_values_l691_691858


namespace find_t_l691_691058

theorem find_t (t : ℝ) (h : real.sqrt (3 * real.sqrt (t - 3)) = real.root 4 (10 - t)) : t = 3.7 := sorry

end find_t_l691_691058


namespace gray_area_is_50pi_l691_691104

noncomputable section

-- Define the radii of the inner and outer circles
def R_inner : ℝ := 2.5
def R_outer : ℝ := 3 * R_inner

-- Area of circles
def A_inner : ℝ := Real.pi * R_inner^2
def A_outer : ℝ := Real.pi * R_outer^2

-- Define width of the gray region
def gray_width : ℝ := R_outer - R_inner

-- Gray area calculation
def A_gray : ℝ := A_outer - A_inner

-- The theorem stating the area of the gray region
theorem gray_area_is_50pi :
  gray_width = 5 → A_gray = 50 * Real.pi := by
  -- Here we assume the proof continues
  sorry

end gray_area_is_50pi_l691_691104


namespace correct_propositions_l691_691348

/- Defining the properties and notions used in the problem -/
def skew (a b : Line) : Prop := ¬∃ p, p ∈ a ∧ p ∈ b
def perpendicular (a : Line) (α : Plane) : Prop := ∀ (p q : Point), p ∈ a ∧ p ∈ α ∧ q ∈ α → p = q ∨ p - q ⊥ α
def parallel (a : Line) (α : Plane) : Prop := ∀ (p q : Point), p ∈ a ∧ q ∈ α → ∃ (r : Line), r ⊂ α ∧ (p - q) ⊂ r

/- We need to prove that propositions 2 and 3 are correct and propositions 1 and 4 are incorrect -/
theorem correct_propositions (a b l : Line) (α : Plane) :
  (¬∃ p, p ∈ a ∧ p ∈ b) ∧ (¬∃ p, p ∈ a ∧ p ∈ b) ∧ perpendicular a α ∧ parallel a α ∧ parallel b α ∧ perpendicular l α →
  (¬∃ p, p ∈ a ∧ p ∈ b) ∧ perpendicular a α ∧ ¬perpendicular b α ∧ (parallel a α ∧ parallel b α ∧ perpendicular l α → perpendicular l a ∧ perpendicular l b) :=
sorry

end correct_propositions_l691_691348


namespace smallest_integer_l691_691283

-- Given positive integer M such that
def satisfies_conditions (M : ℕ) : Prop :=
  M % 6 = 5 ∧
  M % 7 = 6 ∧
  M % 8 = 7 ∧
  M % 9 = 8 ∧
  M % 10 = 9 ∧
  M % 11 = 10 ∧
  M % 13 = 12

-- The main theorem to prove
theorem smallest_integer (M : ℕ) (h : satisfies_conditions M) : M = 360359 :=
  sorry

end smallest_integer_l691_691283


namespace total_trip_time_is_5_hours_l691_691963

-- Define the constants and speeds
def speed_A_to_B : ℝ := 90 -- in kilometers per hour
def speed_B_to_A : ℝ := 160 -- in kilometers per hour
def time_A_to_B_minutes : ℝ := 192 -- in minutes

-- Convert time from A to B to hours
def time_A_to_B_hours : ℝ := time_A_to_B_minutes / 60

-- Calculate the distance from A to B
def distance_A_to_B : ℝ := speed_A_to_B * time_A_to_B_hours

-- Calculate the time to drive back to A-ville
def time_B_to_A_hours : ℝ := distance_A_to_B / speed_B_to_A

-- Calculate the total time for the trip
def total_trip_time_hours : ℝ := time_A_to_B_hours + time_B_to_A_hours

-- Theorem to prove total time is 5 hours
theorem total_trip_time_is_5_hours : total_trip_time_hours = 5 := by
  sorry

end total_trip_time_is_5_hours_l691_691963


namespace min_value_of_reciprocal_sum_l691_691434

theorem min_value_of_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hln : Real.ln (x + y) = 0) :
  (1 / x + 1 / y) ≥ 4 :=
by
  sorry

end min_value_of_reciprocal_sum_l691_691434


namespace subset_count_l691_691437

theorem subset_count {A : Set ℕ} (hA : A = {1, 2}) : 
  (∃ B : Set ℕ, A ⊆ {1, 2} ∧ (A ∪ B = {1, 2})) ∧ (finset.powerset {1, 2}).card = 4 :=
by
  sorry

end subset_count_l691_691437


namespace find_original_number_l691_691323

theorem find_original_number (x : ℤ) (h : 3 * (2 * x + 9) = 57) : x = 5 := by
  sorry

end find_original_number_l691_691323


namespace sum_floor_value_l691_691176

theorem sum_floor_value (x : Fin 2008 → ℝ)
  (h : ∀ n : Fin 2008, x n + (n + 1).val = (∑ k : Fin 2008, x k) + 2009) :
  (⌊ |∑ n : Fin 2008, x n| ⌋ : ℝ) = 1005 :=
by sorry

end sum_floor_value_l691_691176


namespace midpoint_arc_equidistant_l691_691221

open EuclideanGeometry

-- Definitions of the points and triangle's conditions
variables (A B C M N K P Q S : Point)
variables (Ω : Circle)

/-- Given triangle ABC inscribed in circle Ω with AB > BC. Points M and N lie on 
sides AB and BC respectively such that AM = CN. Lines MN and AC intersect at K.
P is the incenter of triangle AMK and Q is the excenter of triangle CNK touching side CN.
Prove that the midpoint of arc ABC of circle Ω is equidistant from P and Q. -/
theorem midpoint_arc_equidistant (h_cycle : ∀ (X Y Z : Point), (is_cyclic Ω X Y Z)) 
  (h_points : is_triangle AB > BC) (hM : on_line M A B) (hN : on_line N B C)
  (h_eq : dist A M = dist C N) (h_K : on_intersection (line MN) (line AC) K)
  (hP : is_incenter P (triangle A M K)) (hQ : is_excenter Q (triangle C N K) CN)
  (h_S : is_midpoint_of_arc S (arc A B C Ω)) :
  dist S P = dist S Q := 
sorry

end midpoint_arc_equidistant_l691_691221


namespace centipede_socks_shoes_order_l691_691656

theorem centipede_socks_shoes_order : 
  (number_of_valid_orders : ℕ) (total_legs : ℕ) (items_per_leg : ℕ) (factorial : ℕ → ℕ) 
  (constraints_probability : ℕ → ℚ) (calc_permutations : ℕ → ℚ) :
  total_legs = 10 →
  items_per_leg = 2 →
  factorial 20 = Nat.factorial 20 →
  constraints_probability total_legs = (1 / 2) ^ total_legs →
  calc_permutations (factorial (total_legs * items_per_leg)) =
    factorial (total_legs * items_per_leg) / constraints_probability total_legs →
  number_of_valid_orders = fracNat (factorial (total_legs * items_per_leg)) (2 ^ total_legs)
  := sorry

end centipede_socks_shoes_order_l691_691656


namespace multiples_of_six_units_digit_six_l691_691798

theorem multiples_of_six_units_digit_six (n : ℕ) : 
  (number_of_six_less_150 n) = 25 
where 
  number_of_six_less_150 (n : ℕ) := 
    ∃ M : ℕ, M * 6 = n ∧ n < 150 
:= 
  sorry

end multiples_of_six_units_digit_six_l691_691798


namespace find_AO_l691_691088

variables (A B C D O : Type)
variables [quadrilateral A B C D] [perpendicular (line AC) (line DC)] [perpendicular (line DB) (line AB)]
variables [line_perpendicular B D (line AD)] [line_intersect_at O (line AC) (perpendicular_from B (line AD))]
variables (AB OC AO : ℝ) (h1 : AB = 4) (h2 : OC = 6)

theorem find_AO :
  AO = 2 :=
sorry

end find_AO_l691_691088


namespace polynomial_expansion_l691_691364

theorem polynomial_expansion (x : ℝ) : 
  (1 - x^3) * (1 + x^4 - x^5) = 1 - x^3 + x^4 - x^5 - x^7 + x^8 :=
by
  sorry

end polynomial_expansion_l691_691364


namespace coordinates_of_P_l691_691075

structure Point (α : Type) [LinearOrderedField α] :=
  (x : α)
  (y : α)

def in_fourth_quadrant {α : Type} [LinearOrderedField α] (P : Point α) : Prop :=
  P.x > 0 ∧ P.y < 0

def distance_to_axes_is_4 {α : Type} [LinearOrderedField α] (P : Point α) : Prop :=
  abs P.x = 4 ∧ abs P.y = 4

theorem coordinates_of_P {α : Type} [LinearOrderedField α] (P : Point α) :
  in_fourth_quadrant P ∧ distance_to_axes_is_4 P → P = ⟨4, -4⟩ :=
by
  sorry

end coordinates_of_P_l691_691075


namespace rounded_sum_probability_l691_691322

theorem rounded_sum_probability :
  let x : ℝ := sorry,
      condition1 : 0 ≤ x ∧ x ≤ 3.5 := sorry,
      sum_conditions : (x.round + (3.5 - x).round = 4) := sorry
  in probability (sum_conditions) = 3 / 7 := sorry

end rounded_sum_probability_l691_691322


namespace eq_satisfies_exactly_four_points_l691_691914

theorem eq_satisfies_exactly_four_points : ∀ (x y : ℝ), 
  (x^2 - 4)^2 + (y^2 - 4)^2 = 0 ↔ 
  (x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2) ∨ (x = 2 ∧ y = -2) ∨ (x = -2 ∧ y = -2) := 
by
  sorry

end eq_satisfies_exactly_four_points_l691_691914


namespace Christina_weekly_distance_l691_691369

/-- 
Prove that Christina covered 74 kilometers that week given the following conditions:
1. Christina walks 7km to school every day from Monday to Friday.
2. She returns home covering the same distance each day.
3. Last Friday, she had to pass by her friend, which is another 2km away from the school in the opposite direction from home.
-/
theorem Christina_weekly_distance : 
  let distance_to_school := 7
  let days_school := 5
  let extra_distance_Friday := 2
  let daily_distance := 2 * distance_to_school
  let total_distance_from_Monday_to_Thursday := 4 * daily_distance
  let distance_on_Friday := daily_distance + 2 * extra_distance_Friday
  total_distance_from_Monday_to_Thursday + distance_on_Friday = 74 := 
by
  sorry

end Christina_weekly_distance_l691_691369


namespace isosceles_triangle_segments_no_common_intersection_l691_691513

theorem isosceles_triangle_segments_no_common_intersection
  (ABC : Triangle)
  (isosceles : is_isosceles ABC)
  (n : ℕ)
  (h_n_is_pos : n > 1)
  (AB_parts : ∀ (i : ℕ), i < n → Point)
  (BC_parts : ∀ (j : ℕ), j < n+1 → Point)
  (A_segments : ∀ (j : ℕ), j < n → Segment (vertex A ABC) (BC_parts j))
  (C_segments : ∀ (i : ℕ), i < n-1 → Segment (vertex C ABC) (AB_parts i))
  (BM : Segment (vertex B ABC) (midpoint A C ABC)) :
  ¬ ∃ O, (⋂ i < n, A_segments i).contains O ∧ (⋂ i < n-1, C_segments i).contains O ∧ (BM.contains O) := sorry

end isosceles_triangle_segments_no_common_intersection_l691_691513


namespace f_has_one_zero_l691_691464

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x + 2

theorem f_has_one_zero (a : ℝ) (ha : a ≥ 2) :
  ∃! x : ℝ, x > 0 ∧ f 2 x = 0 :=
begin
  sorry
end

end f_has_one_zero_l691_691464


namespace rectangle_area_trisection_problem_l691_691097
-- Import all necessary modules

-- Define the problem conditions as a Lean theorem
theorem rectangle_area_trisection_problem 
  (ABCD : Type)
  [IsRectangle ABCD]
  (angle_D_trisected : Trisected D G H)
  (on_AB : PointOnLine G AB)
  (on_BC : PointOnLine H BC)
  (AG_length : Length AG = 4)
  (BH_length : Length BH = 3) : 
  Area ABCD = 36 := by sorry

end rectangle_area_trisection_problem_l691_691097


namespace number_of_correct_statements_l691_691459

-- Define the statements as conditions
def statement1 : Prop :=
  ∀ (histogram : Type) (class_width : Type), 
    ∀ (area : histogram → class_width → ℝ),
      ¬(∀ (h : histogram) (cw : class_width), area h cw = cw)

def statement2 : Prop :=
  ∀ (x_mean y_mean : ℝ) (line : ℝ → ℝ),
    line x_mean = y_mean

def statement3 : Prop :=
  ∀ (ξ : ℝ → ℝ) (P : ℝ → ℝ),
    (∀ (μ σ : ℝ), ξ = λ x, (1 / (σ * sqrt (2 * pi))) * exp (-(x - μ)^2 / (2 * σ^2))
     → ξ 1 = 1 / 2)

def statement4 : Prop :=
  ∀ (X Y : Type) (k K2 : ℝ),
    k = K2 → 
      ¬(k > 0 → ∃ (relationship : Prop), relationship)

-- Define the main proof problem
theorem number_of_correct_statements :
  (¬statement1) ∧ statement2 ∧ statement3 ∧ (¬statement4) ↔ 2 = 2 :=
by
  sorry

end number_of_correct_statements_l691_691459


namespace salary_net_change_l691_691206

theorem salary_net_change (S : ℝ) : 
  let increased_salary := S * 1.15 in
  let final_salary := increased_salary * 0.85 in
  (final_salary - S) / S = -0.0225 :=
by
  sorry

end salary_net_change_l691_691206


namespace C_received_amount_l691_691514

variable {Investment_A : ℝ}
variable {Investment_B : ℝ}
variable {Investment_C : ℝ}
variable {Total_Profit : ℝ}

-- The ratio of investments
axiom ratio_A_C : Investment_A / Investment_C = 3 / 2
axiom ratio_A_B : Investment_A / Investment_B = 3 / 1

-- The total profit amount
axiom total_profit : Total_Profit = 60000

-- Define the amount received by C
noncomputable def amount_received_by_C :=
  (Investment_C / (Investment_A + Investment_B + Investment_C)) * Total_Profit

-- The proof statement
theorem C_received_amount : amount_received_by_C = 20000 :=
  sorry

end C_received_amount_l691_691514


namespace inequality_of_triangle_l691_691106

variable {A B C M : Type}
variable [metric_space A] [metric_space B] [metric_space C] [metric_space M]
variable (a b c m : A) (bc ac ab : ℝ)
variable (S : ℝ)

theorem inequality_of_triangle
  (h1 : point_in_triangle a b c m)
  (h2 : area_of_triangle a b c = S)
  : 4 * S <= dist a m * bc + dist b m * ac + dist c m * ab :=
by 
  sorry

end inequality_of_triangle_l691_691106


namespace least_distinct_values_l691_691320

theorem least_distinct_values (n mode_count : ℕ) (unique_mode : ℕ) :
  n = 2023 ∧ mode_count = 15 ∧ unique_mode = 1 →
  ∃ least_distinct_count, least_distinct_count = 146 := 
by
  intro h
  have h1 : n = 2023 := h.1
  have h2 : mode_count = 15 := h.2.1
  have h3 : unique_mode = 1 := h.2.2
  use 146
  sorry

end least_distinct_values_l691_691320


namespace students_brought_two_plants_l691_691223

theorem students_brought_two_plants 
  (a1 a2 a3 a4 a5 : ℕ) (p1 p2 p3 p4 p5 : ℕ)
  (h1 : a1 + a2 + a3 + a4 + a5 = 20)
  (h2 : a1 * p1 + a2 * p2 + a3 * p3 + a4 * p4 + a5 * p5 = 30)
  (h3 : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
        p3 ≠ p4 ∧ p3 ≠ p5 ∧ p4 ≠ p5)
  : ∃ a : ℕ, a = 1 ∧ (∃ i : ℕ, p1 = 2 ∨ p2 = 2 ∨ p3 = 2 ∨ p4 = 2 ∨ p5 = 2) :=
sorry

end students_brought_two_plants_l691_691223


namespace number_of_different_ways_to_travel_l691_691363

-- Define the conditions
def number_of_morning_flights : ℕ := 2
def number_of_afternoon_flights : ℕ := 3

-- Assert the question and the answer
theorem number_of_different_ways_to_travel : 
  (number_of_morning_flights * number_of_afternoon_flights) = 6 :=
by
  sorry

end number_of_different_ways_to_travel_l691_691363


namespace circle_line_intersection_shortest_chord_l691_691025

theorem circle_line_intersection (k : ℝ) :
  let circle_eq := (x y : ℝ) → x^2 + y^2 - 6 * x - 8 * y + 21 = 0
  let line_eq := (x y : ℝ) → k * x - y - 4 * k + 3 = 0
  (∃ x y : ℝ, circle_eq x y ∧ line_eq x y) → ∃! (x y : ℝ), circle_eq x y ∧ line_eq x y :=
sorry

theorem shortest_chord (k : ℝ) :
  let circle_center := (3, 4)
  let radius := 2
  let line_eq := (x y : ℝ) → k * x - y - 4 * k + 3 = 0
  let distance_from_center := |k + 1| / real.sqrt (1 + k^2)
  let max_distance := real.sqrt 2
  distance_from_center = max_distance → k = 1 ∧ ∃! (chord_length : ℝ), chord_length = 2 * real.sqrt (2^2 - max_distance^2) :=
sorry

end circle_line_intersection_shortest_chord_l691_691025


namespace old_clock_24_hours_l691_691350

theorem old_clock_24_hours :
  (22 * 66) = ((24 * 60) + 12) :=
by
  calc
    22 * 66 = 1452     : by norm_num
    ...     = 24 * 60 + 12 : by norm_num

end old_clock_24_hours_l691_691350


namespace max_area_parallelogram_l691_691203

theorem max_area_parallelogram
    (P : ℝ)
    (a b : ℝ)
    (h1 : P = 60)
    (h2 : a = 3 * b)
    (h3 : P = 2 * a + 2 * b) :
    (a * b ≤ 168.75) :=
by
  -- We prove that given the conditions, the maximum area is 168.75 square units.
  sorry

end max_area_parallelogram_l691_691203


namespace beth_sold_l691_691361

theorem beth_sold {initial_coins additional_coins total_coins sold_coins : ℕ} 
  (h_init : initial_coins = 125)
  (h_add : additional_coins = 35)
  (h_total : total_coins = initial_coins + additional_coins)
  (h_sold : sold_coins = total_coins / 2) :
  sold_coins = 80 := 
sorry

end beth_sold_l691_691361


namespace parallelogram_area_15_l691_691089

-- Define the vertices of the parallelogram
def A := (4, 4) : ℝ × ℝ
def B := (7, 4) : ℝ × ℝ
def C := (5, 9) : ℝ × ℝ
def D := (8, 9) : ℝ × ℝ

-- Function to compute the area of the parallelogram
def parallelogram_area (A B C D : ℝ × ℝ) : ℝ :=
  (B.1 - A.1) * (C.2 - A.2)

-- Statement of the problem
theorem parallelogram_area_15 : parallelogram_area A B C D = 15 := 
by 
  -- We would provide the mathematical proof here
  sorry

end parallelogram_area_15_l691_691089


namespace beth_sells_half_of_coins_l691_691357

theorem beth_sells_half_of_coins (x y : ℕ) (h₁ : x = 125) (h₂ : y = 35) : (x + y) / 2 = 80 :=
by
  sorry

end beth_sells_half_of_coins_l691_691357


namespace triangle_is_isosceles_l691_691354

variables {A B C D E F : Type}

-- Conditions
variable [triangle : Triangle A B C]
variable [equal_segments : AD = AE]
variable [intersection : Intersect CD BE F]
variable [equal_inradii : inradius (Triangle BDF) = inradius (Triangle CEF)]

-- Conclusion: Triangle ABC is isosceles
theorem triangle_is_isosceles 
  (h1 : triangle A B C)
  (h2 : AD = AE)
  (h3 : Intersect CD BE F)
  (h4 : inradius (Triangle BDF) = inradius (Triangle CEF)) :
  AB = AC :=
by
  sorry

end triangle_is_isosceles_l691_691354


namespace seventy_million_digit_number_divisible_by_239_l691_691631

theorem seventy_million_digit_number_divisible_by_239 : 
  let digit_sequences := finset.range 10000000
  let concatenated_sum := digit_sequences.sum
  concatenated_sum % 239 = 0 :=
by
  sorry

end seventy_million_digit_number_divisible_by_239_l691_691631


namespace circle_equation_l691_691933

theorem circle_equation (h k r : ℝ) (x y : ℝ) (h_center : h = 3) (k_center : k = 1) (r_radius : r = 5) :
  (x - 3) * (x - 3) + (y - 1) * (y - 1) = 25 :=
by
  have eq_r_sq : r * r = 25 := by rw [r_radius, ←sq]
  rw [h_center, k_center, r_radius]
  exact eq_r_sq
  sorry

end circle_equation_l691_691933


namespace compound_interest_time_l691_691397

theorem compound_interest_time
  (P : ℝ := 100000)
  (r : ℝ := 0.04)
  (n : ℕ := 2)
  (CI : ℝ := 8243.216)
  (A : ℝ := 108243.216)
  (hA : A = P + CI) :
  ln (A / P) / (n * ln (1 + r / n)) = 2 :=
by
  sorry

end compound_interest_time_l691_691397


namespace log2_power_function_l691_691466

theorem log2_power_function :
  ∃ (α : ℝ), (∀ x : ℝ, f x = x ^ α) ∧ f (1 / 3) = 1 / 3 ^ α ∧ (1 / 3 ^ α = sqrt 3 / 3) → log 2 (f 2) = 1 / 2 :=
by sorry

end log2_power_function_l691_691466


namespace gcd_ab_sum_a_b_pow_l691_691001

noncomputable theory

open Nat

theorem gcd_ab_sum_a_b_pow (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : gcd a b = 1) :
  ∃ d ∈ ({1, 5, 401, 2005} : set ℕ),
    gcd (a + b) ((a ^ 2005 + b ^ 2005) / (a + b)) = d :=
sorry

end gcd_ab_sum_a_b_pow_l691_691001


namespace machine_produces_rulers_l691_691311

theorem machine_produces_rulers :
  let rulers_per_minute := 8
  let total_minutes := 15
  total_rulers = rulers_per_minute * total_minutes :=
by
  let rulers_per_minute := 8
  let total_minutes := 15
  let total_rulers := rulers_per_minute * total_minutes
  have h : total_rulers = 120 := by
    sorry
  exact h

end machine_produces_rulers_l691_691311


namespace find_weight_of_silver_in_metal_bar_l691_691308

noncomputable def weight_loss_ratio_tin : ℝ := 1.375 / 10
noncomputable def weight_loss_ratio_silver : ℝ := 0.375
noncomputable def ratio_tin_silver : ℝ := 0.6666666666666664

theorem find_weight_of_silver_in_metal_bar (T S : ℝ)
  (h1 : T + S = 70)
  (h2 : T / S = ratio_tin_silver)
  (h3 : weight_loss_ratio_tin * T + weight_loss_ratio_silver * S = 7) :
  S = 15 :=
by
  sorry

end find_weight_of_silver_in_metal_bar_l691_691308


namespace constant_term_expansion_l691_691076

theorem constant_term_expansion (a : ℝ) (h : (2 + a * x) * (1 + 1/x) ^ 5 = (2 + 5 * a)) : 2 + 5 * a = 12 → a = 2 :=
by
  intro h_eq
  have h_sum : 2 + 5 * a = 12 := h_eq
  sorry

end constant_term_expansion_l691_691076


namespace arithmetic_sequence_incorrect_statement_l691_691423

theorem arithmetic_sequence_incorrect_statement:
  (∀ n: ℕ, a(2) = 7 ∧ a(4) = 15) → 
  ¬ ((∀ n: ℕ,  S(0) = 0 ∧ S(n+1) = S(n) + a(n+1)) → 
    ¬ (∀ n: ℕ, S(n+1) - S(n) > 0 → ¬ (S n < S (n + 1)))) :=
by
  intros
  sorry

end arithmetic_sequence_incorrect_statement_l691_691423


namespace find_t_l691_691055

theorem find_t (t : ℝ) (h : real.sqrt (3 * real.sqrt (t - 3)) = real.root 4 (10 - t)) : t = 3.7 := sorry

end find_t_l691_691055


namespace number_in_center_l691_691346

theorem number_in_center (a b c d e f g h i : ℕ) 
  (h_grid_filled : Multiset { a, b, c, d, e, f, g, h, i } = { 1, 2, 3, 4, 5, 6, 7, 8, 9 }) 
  (h_consecutive_adjacent : (a, b), (b, c), (d, e), (e, f), (g, h), (h, i),
                           (a, d), (b, e), (c, f), (d, g), (e, h), (f, i),
                           (a, e), (c, e), (g, e), (i, e) ∈ adjacency_rules)
  (h_corner_sum : a + c + g + i = 20) :
  e = 5 := 
sorry

end number_in_center_l691_691346


namespace rational_solution_unique_l691_691302

theorem rational_solution_unique
  (n : ℕ) (x y : ℚ)
  (hn : Odd n)
  (hx_eqn : x ^ n + 2 * y = y ^ n + 2 * x) :
  x = y :=
sorry

end rational_solution_unique_l691_691302


namespace total_plates_used_l691_691940

-- Definitions from the conditions
def number_of_people := 6
def meals_per_day_per_person := 3
def plates_per_meal_per_person := 2
def number_of_days := 4

-- Statement of the theorem
theorem total_plates_used : number_of_people * meals_per_day_per_person * plates_per_meal_per_person * number_of_days = 144 := 
by
  sorry

end total_plates_used_l691_691940


namespace Jenny_minutes_of_sleep_l691_691109

def hours_of_sleep : ℕ := 8
def minutes_per_hour : ℕ := 60

theorem Jenny_minutes_of_sleep : hours_of_sleep * minutes_per_hour = 480 := by
  sorry

end Jenny_minutes_of_sleep_l691_691109


namespace radius_coverage_l691_691676

theorem radius_coverage (d : ℝ) : 
  let square_area := 20 * 20
  let target_area := (3 / 4) * square_area
  let circle_area := π * d^2
  target_area = circle_area * 400 → d = 0.5 :=
by
  let square_area := 20 * 20
  let target_area := (3 / 4) * square_area
  let circle_area := π * d^2
  assume h : target_area = circle_area * 400
  sorry

end radius_coverage_l691_691676


namespace half_abs_diff_squares_l691_691254

theorem half_abs_diff_squares (a b : ℤ) (h₁ : a = 20) (h₂ : b = 15) : 
  (|a^2 - b^2| / 2 : ℚ) = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691254


namespace find_vector_b_magnitude_l691_691765

variable {R : Type*} [Real R]

noncomputable def vector_length (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

theorem find_vector_b_magnitude
  (a b : ℝ × ℝ)
  (h_angle : ∃ θ : ℝ, θ = 2 * Real.pi / 3)
  (h_a_mag : vector_length a = 2)
  (h_sum_mag : vector_length (a.1 + b.1, a.2 + b.2) = Real.sqrt 7)
  : vector_length b = 3 :=
sorry

end find_vector_b_magnitude_l691_691765


namespace a_minus_b_eq_neg3_l691_691451

-- Define variables
variables (a b : ℝ)

-- Define the conditions 
def symmetrical_about_x (P Q : ℝ × ℝ) := P.1 = Q.1 ∧ P.2 = -Q.2

-- Define the points
def P := (1, a)
def Q := (b, 2)

-- State the theorem to prove
theorem a_minus_b_eq_neg3
  (h : symmetrical_about_x P Q) : a - b = -3 :=
sorry

end a_minus_b_eq_neg3_l691_691451


namespace instantaneous_velocity_at_3_l691_691199

def s (t : ℝ) : ℝ := 1 - t + t^2

def velocity_at_t (t : ℝ) : ℝ := (deriv s) t

theorem instantaneous_velocity_at_3 : velocity_at_t 3 = 5 := by
  sorry

end instantaneous_velocity_at_3_l691_691199


namespace solve_for_t_l691_691063

theorem solve_for_t (t : ℝ) (h : sqrt (3 * sqrt (t - 3)) = real.sqrt (real.sqrt (real.sqrt (10 - t)))) : t = 3.7 :=
by
  sorry

end solve_for_t_l691_691063


namespace count_multiples_of_6_with_units_digit_6_l691_691800

theorem count_multiples_of_6_with_units_digit_6 : 
  ∃ (n : ℕ), n = 5 ∧ ∀ (m : ℕ), (m > 0 ∧ m < 150 ∧ m % 6 = 0 ∧ m % 10 = 6) ↔ m ∈ {6, 36, 66, 96, 126} :=
by
  sorry

end count_multiples_of_6_with_units_digit_6_l691_691800


namespace area_triangle_EFG_l691_691100

open EuclideanGeometry

variables (A B C D E F G : Point)
variable {AB : segment A B}
variable {CD : segment C D}
variable {angle : angle A D C}
variable {AF FE ED : segment A F = segment F E = segment E D}
variable {BC BG : segment B C = 4 * segment B G}

def isosceles_trapezoid (A B C D : Point) : Prop := 
  B.y = D.y ∧ angle A D C = 45°

noncomputable def triangle (A B C : Point) : Type := 
  {area : ℝ // area ≥ 0}

theorem area_triangle_EFG :
  isosceles_trapezoid A B C D →
  AF = FE →
  FE = ED →
  BC = 4 * BG →
  AB = 4 →
  CD = 12 →
  EFG.area = 4 :=
sorry

end area_triangle_EFG_l691_691100


namespace count_positive_integers_m_l691_691478

theorem count_positive_integers_m :
  ∃ m_values : Finset ℕ, m_values.card = 4 ∧ ∀ m ∈ m_values, 
    ∃ k : ℕ, k > 0 ∧ (7 * m + 2 = m * k + 2 * m) := 
sorry

end count_positive_integers_m_l691_691478


namespace convex_polyhedron_inscribed_sphere_l691_691102

theorem convex_polyhedron_inscribed_sphere (P : Polyhedron) (S' : ℝ) (R : ℝ) (V : ℝ)
  (h_inscribed_sphere : P.hasInscribedSphere R) 
  (h_surface_area : P.surfaceArea = S')
  (h_volume : P.volume = V) : V = (1 / 3) * S' * R :=
sorry

end convex_polyhedron_inscribed_sphere_l691_691102


namespace new_average_after_increase_l691_691570

theorem new_average_after_increase (S : Finset ℝ) (n : ℝ) (h₁ : S.card = 10) (h₂ : S.sum / 10 = 6.2) : 
  (S.sum + 4) / 10 = 6.6 :=
by
  sorry

end new_average_after_increase_l691_691570


namespace square_root_25_pm5_l691_691596

-- Define that a number x satisfies the equation x^2 = 25
def square_root_of_25 (x : ℝ) : Prop := x * x = 25

-- The theorem states that the square root of 25 is ±5
theorem square_root_25_pm5 : ∀ x : ℝ, square_root_of_25 x ↔ x = 5 ∨ x = -5 :=
by
  intros x
  sorry

end square_root_25_pm5_l691_691596


namespace distance_traveled_by_second_hand_l691_691313

theorem distance_traveled_by_second_hand (r : ℝ) (minutes : ℝ) (h1 : r = 10) (h2 : minutes = 45) :
  (2 * Real.pi * r) * (minutes / 1) = 900 * Real.pi := by
  -- Given:
  -- r = length of the second hand = 10 cm
  -- minutes = 45
  -- To prove: distance traveled by the tip = 900π cm
  sorry

end distance_traveled_by_second_hand_l691_691313


namespace no_solution_2023_l691_691846

theorem no_solution_2023 (a b c : ℕ) (h₁ : a + b + c = 2023) (h₂ : (b + c) ∣ a) (h₃ : (b - c + 1) ∣ (b + c)) : false :=
by
  sorry

end no_solution_2023_l691_691846


namespace book_pair_count_l691_691050

theorem book_pair_count (M F B : ℕ) (hM : M = 4) (hF : F = 3) (hB : B = 3) :
  (M * F) + (M * B) + (F * B) = 33 :=
by
  -- substitute M, F, B with 4, 3, 3 respectively
  rw [hM, hF, hB]
  -- compute the product terms and sum them
  calc
    4 * 3 + 4 * 3 + 3 * 3 = 12 + 4 * 3 + 3 * 3 : by rfl
                      ... = 12 + 12 + 3 * 3 : by rw [mul_comm]
                      ... = 12 + 12 + 9 : by rfl
                      ... = 33 : by rfl

end book_pair_count_l691_691050


namespace half_abs_diff_squares_l691_691236

theorem half_abs_diff_squares (a b : ℤ) (ha : a = 20) (hb : b = 15) : 
  (|a^2 - b^2| / 2) = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691236


namespace distance_between_parallel_lines_l691_691839

theorem distance_between_parallel_lines (a b c : ℝ) (d_ab d_ac : ℝ) 
  (hab : a // b) (hbc : b // c) (h_ab : d_ab = 5) (h_ac : d_ac = 2) : 
  d_ab = d_ac + 3 ∨ d_ab = d_ac + 7 := 
by 
  intro a b c d_ab d_ac hab hbc h_ab h_ac 
  sorry

end distance_between_parallel_lines_l691_691839


namespace polynomial_integer_values_l691_691958

theorem polynomial_integer_values (a b c d : ℤ) 
    (h0 : d ∈ ℤ) 
    (h1 : a + b + c + d ∈ ℤ) 
    (h2 : -a + b - c + d ∈ ℤ) 
    (h3 : 8a + 4b + 2c + d ∈ ℤ) : 
    ∀ x : ℤ, (ax^3 + bx^2 + cx + d) ∈ ℤ := 
by 
  sorry

end polynomial_integer_values_l691_691958


namespace elsie_money_l691_691957

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

theorem elsie_money : 
  compound_interest 2500 0.04 20 = 5477.81 :=
by 
  sorry

end elsie_money_l691_691957


namespace glen_pop_l691_691532

/-- In the village of Glen, the total population can be formulated as 21h + 6c
given the relationships between people, horses, sheep, cows, and ducks.
We need to prove that 96 cannot be expressed in the form 21h + 6c for
non-negative integers h and c. -/
theorem glen_pop (h c : ℕ) : 21 * h + 6 * c ≠ 96 :=
by
sorry

end glen_pop_l691_691532


namespace product_of_slopes_parallelogram_condition_l691_691518

variable {m : ℝ} (hm : 0 < m)

def ellipse (x y : ℝ) : Prop := 9 * x^2 + y^2 = m^2

variable {k b : ℝ}

def line (x y : ℝ) := y = k * x + b

variable (A B M P : ℝ × ℝ) 

-- A and B are intersection points of line l and the ellipse C
-- Midpoint of segment AB is M
def midpoint (A B M : ℝ × ℝ) : Prop := M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- M is midpoint of A and B, which lie on both the ellipse and the line
axiom A_on_ellipse : ellipse A.1 A.2
axiom B_on_ellipse : ellipse B.1 B.2
axiom A_on_line : line A.1 A.2
axiom B_on_line : line B.1 B.2
axiom M_is_midpoint : midpoint A B M

-- (1) Prove the product of slopes
theorem product_of_slopes
  (k_l : ℝ) (k_om : ℝ)
  (slope_l : k_l = k)
  (OM_slope : k_om = -9 / k) :
  k_l * k_om = -9 :=
sorry

-- (2) Prove conditions for parallelogram
def through_point (x y : ℝ) := y = k * x + (m * (3 - k)) / 3

variable (OP_midpoint : ℝ × ℝ)

axiom l_through_point : line (m / 3) m
axiom P_on_ellipse : ellipse P.1 P.2
axiom OP_midpoint_is_midpoint : midpoint O P OP_midpoint

-- Prove whether OAPB can be a parallelogram
theorem parallelogram_condition
  (slope_values : {k : ℝ // k = 4 - real.sqrt 7 ∨ k = 4 + real.sqrt 7}) :
  ∃ k : ℝ, k ∈ slope_values ∧ ∀ A B P : ℝ × ℝ, 
  (A_on_ellipse ∧ B_on_ellipse ∧ M_is_midpoint ∧ through_point (m / 3) m ∧ midpoint O P OP_midpoint) → is_parallelogram O A P B :=
sorry

noncomputable def is_parallelogram (O A P B : ℝ × ℝ) : Prop := sorry

end product_of_slopes_parallelogram_condition_l691_691518


namespace man_l691_691988

theorem man's_speed_against_current :
  ∀ (V_down V_c V_m V_up : ℝ),
    (V_down = 15) →
    (V_c = 2.8) →
    (V_m = V_down - V_c) →
    (V_up = V_m - V_c) →
    V_up = 9.4 :=
by
  intros V_down V_c V_m V_up
  intros hV_down hV_c hV_m hV_up
  sorry

end man_l691_691988


namespace gcd_lcm_identity_l691_691598

def GCD (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := a * (b / GCD a b)

theorem gcd_lcm_identity (a b c : ℕ) :
    (LCM a (LCM b c))^2 / (LCM a b * LCM b c * LCM c a) = (GCD a (GCD b c))^2 / (GCD a b * GCD b c * GCD c a) :=
by
  sorry

end gcd_lcm_identity_l691_691598


namespace solution_set_l691_691867

noncomputable def f : ℝ → ℝ := sorry
def is_periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f(x) = f(x + p)
def is_even (f : ℝ → ℝ) := ∀ x, f(x) = f(-x)
def strictly_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f(x) > f(y)

axiom f_cond1 : is_even f
axiom f_cond2 : is_periodic f 2
axiom f_cond3 : strictly_decreasing_on f {x | 0 ≤ x ∧ x ≤ 1}
axiom f_cond4 : f π = 1
axiom f_cond5 : f (2 * π) = 2

theorem solution_set :
  {x | 1 ≤ x ∧ x ≤ 2 ∧ 1 ≤ f x ∧ f x ≤ 2} = {x | (π - 2) ≤ x ∧ x ≤ (8 - 2 * π)} :=
sorry

end solution_set_l691_691867


namespace sum_coordinates_D_eq_7_l691_691887

/-- Points A = (4,8), B = (2,2), C = (6,4), and D = (a,b) lie in the first quadrant
    and are the vertices of quadrilateral ABCD.
    The quadrilateral formed by joining the midpoints of AB, BC, CD, and DA is a rhombus.
    Prove that the sum of the coordinates of point D is 7. -/
theorem sum_coordinates_D_eq_7 {a b : ℝ} (h1 : 0 < a) (h2 : 0 < b)
    (H : let A := (4, 8)
             B := (2, 2)
             C := (6, 4)
             D := (a, b)
             M_AB := ((4 + 2) / 2, (8 + 2) / 2)
             M_BC := ((2 + 6) / 2, (2 + 4) / 2)
             M_CD := ((6 + a) / 2, (4 + b) / 2)
             M_DA := ((a + 4) / 2, (b + 8) / 2)
         in (M_AB.1 - M_BC.1)^2 + (M_AB.2 - M_BC.2)^2 = 
            (M_BC.1 - M_CD.1)^2 + (M_BC.2 - M_CD.2)^2 ∧
            (M_CD.1 - M_DA.1)^2 + (M_CD.2 - M_DA.2)^2 = 
            (M_DA.1 - M_AB.1)^2 + (M_DA.2 - M_AB.2)^2) :
    a + b = 7 :=
begin
  sorry
end

end sum_coordinates_D_eq_7_l691_691887


namespace point_on_positive_x_axis_l691_691519

def point (m : ℝ) : ℝ × ℝ :=
  (m ^ 2 + Real.pi, 0)

theorem point_on_positive_x_axis (m : ℝ) : 
  let p := point m in 
  p.1 > 0 ∧ p.2 = 0 := 
by 
  sorry

end point_on_positive_x_axis_l691_691519


namespace min_sum_of_distances_l691_691225

variable (p : ℝ) (h : p > 0)

theorem min_sum_of_distances (p : ℝ) (h : p > 0) :
  ∃ (A B C D : ℝ × ℝ), (|AB_dist A B h + CD_dist C D h = 16 * p) :=
by
  assume p h,
  sorry -- Proof construction goes here

end min_sum_of_distances_l691_691225


namespace mary_initially_selected_10_l691_691355

-- Definitions based on the conditions
def price_apple := 40
def price_orange := 60
def avg_price_initial := 54
def avg_price_after_putting_back := 48
def num_oranges_put_back := 5

-- Definition of Mary_initially_selected as the total number of pieces of fruit initially selected by Mary
def Mary_initially_selected (A O : ℕ) := A + O

-- Theorem statement
theorem mary_initially_selected_10 (A O : ℕ) 
  (h1 : (price_apple * A + price_orange * O) / (A + O) = avg_price_initial)
  (h2 : (price_apple * A + price_orange * (O - num_oranges_put_back)) / (A + O - num_oranges_put_back) = avg_price_after_putting_back) : 
  Mary_initially_selected A O = 10 := 
sorry

end mary_initially_selected_10_l691_691355


namespace concurrency_of_lines_l691_691549

noncomputable def orthocenter (ABC : Triangle) : Point := sorry -- Assume we have the orthocenter function

variable {A B C D E F P Q R : Point}
variable {ABC : Triangle}
variable (is_acute_angled : ABC.isAcuteAngled)
variable (altitude_AD: line_through A D)
variable (altitude_BE: line_through B E)
variable (altitude_CF: line_through C F)
variable (perpendicular_from_A_to_EF : line_perpendicular_to EF A P)
variable (perpendicular_from_B_to_FD : line_perpendicular_to FD B Q)
variable (perpendicular_from_C_to_DE : line_perpendicular_to DE C R)

theorem concurrency_of_lines :
  concurrent (line_through A P) (line_through B Q) (line_through C R) :=
sorry

end concurrency_of_lines_l691_691549


namespace fixed_point_a1_b2_range_of_a_range_of_b_l691_691461

-- Fix the parameters and conditions
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + (b - 1)

-- 1. The fixed point of f(x) where a = 1 and b = 2
theorem fixed_point_a1_b2 : 
  ∃ x : ℝ, f 1 2 x = x :=
by 
  use -1
  unfold f
  linarith

-- 2. The range of values for a such that f(x) has two distinct fixed points for any real b
theorem range_of_a :
  ∀ b : ℝ, ∃! x1 x2 : ℝ, x1 ≠ x2 ∧ f a b x1 = x1 ∧ f a b x2 = x2 ↔ 0 < a ∧ a < 1 :=
by 
  have h : ∀ b : ℝ, (b + 1)^2 - 4 * a * (b - 1) > 0,
  -- Proof of the discriminant condition
  sorry
  split
  { intro h1
    -- Discriminant is indefinite
    sorry }
  { intro h2
    use 0
    -- Derive and prove result for 0 < a < 1
    sorry }

-- 3. The range of values for b under fixed a ∈ (0, 1) such that f(x₁) + x₂ logic holds
theorem range_of_b (a : ℝ) :
  a ∈ (0, 1) → 
  ∃! b : ℝ, (∃! x1 x2 : ℝ, x1 ≠ x2 ∧ f a b x1 + x2 = x1 + x2 = - a / (2 * a^2 + 1)) ↔ 0 < b ∧ b < 1 / 3 :=
by 
  intro ha
  have h : (0 < a ∧ a < 1) → (0 < b ∧ b < 1 / 3),
  sorry
  exact sorry

end fixed_point_a1_b2_range_of_a_range_of_b_l691_691461


namespace range_of_fx_l691_691442

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^k

theorem range_of_fx (k : ℝ) (x : ℝ) (h1 : k < -1) (h2 : x ∈ Set.Ici (0.5)) :
  Set.Icc (0 : ℝ) 2 = {y | ∃ x, f x k = y ∧ x ∈ Set.Ici 0.5} :=
sorry

end range_of_fx_l691_691442


namespace bridge_length_correct_l691_691343

def train_length : ℝ := 360
def train_speed_kmph : ℝ := 45
def time_seconds : ℝ := 40

/-- The conversion factor from kilometers per hour to meters per second. -/
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * 1000 / 3600

/-- The speed of the train in meters per second. -/
def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

/-- The distance traveled by the train in the given time. -/
def distance_travelled : ℝ := train_speed_mps * time_seconds

/-- The length of the bridge. -/
def bridge_length : ℝ := distance_travelled - train_length

theorem bridge_length_correct : bridge_length = 140 := by
  -- insert the proof here
  sorry

end bridge_length_correct_l691_691343


namespace total_students_l691_691638

-- Condition 1: 20% of students are below 8 years of age.
-- Condition 2: The number of students of 8 years of age is 72.
-- Condition 3: The number of students above 8 years of age is 2/3 of the number of students of 8 years of age.

variable {T : ℝ} -- Total number of students

axiom cond1 : 0.20 * T = (T - (72 + (2 / 3) * 72))
axiom cond2 : 72 = 72
axiom cond3 : (T - 72 - (2 / 3) * 72) = 0

theorem total_students : T = 150 := by
  -- Proof goes here
  sorry

end total_students_l691_691638


namespace incorrect_option_l691_691744

-- Definitions and conditions from the problem
def p (x : ℝ) : Prop := (x - 2) * Real.sqrt (x^2 - 3*x + 2) ≥ 0
def q (k : ℝ) : Prop := ∀ x : ℝ, k * x^2 - k * x - 1 < 0

-- The Lean 4 statement to verify the problem
theorem incorrect_option :
  (¬ ∃ x, p x) ∧ (∃ k, q k) ∧
  (∀ k, -4 < k ∧ k ≤ 0 → q k) →
  (∃ x, ¬p x) :=
  by
  sorry

end incorrect_option_l691_691744


namespace bisect_perimeter_l691_691372

-- Definitions for necessary conditions
variable {A B C : Point}
variable [Triangle ABC]
variable {σ_B σ_C ω_B ω_C : Circle}
variable {B0 C0 : Point}

-- Define excircles and their reflections
def incircle_B (A B C : Point) : Circle := sorry
def incircle_C (A B C : Point) : Circle := sorry
def reflect (σ : Circle) (M : Point) : Circle := sorry

-- Given conditions
axiom excenters_and_reflections :
  σ_B = incircle_B A B C ∧
  σ_C = incircle_C A B C ∧
  ω_B = reflect σ_B (midpoint A C) ∧
  ω_C = reflect σ_C (midpoint A B)

-- The theorem statement
theorem bisect_perimeter 
  (P Q : Point)
  (h_intersection: P ∈ ω_B ∩ ω_C ∧ Q ∈ ω_B ∩ ω_C) : 
  bisects_perimeter_of_triangle (line_through P Q) ABC :=
sorry

end bisect_perimeter_l691_691372


namespace domain_φ_range_f_le_g_l691_691783

def f (x : ℝ) : ℝ := Real.log (x - 1) / Real.log 2
def g (x : ℝ) : ℝ := Real.log (6 - 2 * x) / Real.log 2
def φ (x : ℝ) : ℝ := f x + g x

theorem domain_φ : ∀ x : ℝ, 1 < x ∧ x < 3 ↔ (1 < x ∧ φ x = f x + g x) := by 
  intro x
  split
  by
    intro h
    exact and.intro h.left (congrArg (λ x, f x + g x) h.right)
  by 
    intro h
    exact and.intro h.left h.right
  sorry

theorem range_f_le_g : ∀ x : ℝ, (1 < x ∧ x <= 7/3) ↔ (1 < x ∧ f x ≤ g x) := by 
  intro x
  split
  by
    intro h
    exact and.intro h.left (le_trans (le_of_lt h.left) h.right)
  by
    intro h
    simp [f, g] at h
    sorry

end domain_φ_range_f_le_g_l691_691783


namespace mixedGasTemperature_is_correct_l691_691482

noncomputable def mixedGasTemperature (V₁ V₂ p₁ p₂ T₁ T₂ : ℝ) : ℝ := 
  (p₁ * V₁ + p₂ * V₂) / ((p₁ * V₁) / T₁ + (p₂ * V₂) / T₂)

theorem mixedGasTemperature_is_correct :
  mixedGasTemperature 2 3 3 4 400 500 = 462 := by
    sorry

end mixedGasTemperature_is_correct_l691_691482


namespace find_integer_mod_l691_691730

theorem find_integer_mod (n : ℤ) (h1 : n ≡ 8657 [MOD 15]) (h2 : 0 ≤ n ∧ n ≤ 14) : n = 2 :=
by
  sorry

end find_integer_mod_l691_691730


namespace max_value_sine_cosine_l691_691077

/-- If the maximum value of the function f(x) = 4 * sin x + a * cos x is 5, then a = ±3. -/
theorem max_value_sine_cosine (a : ℝ) :
  (∀ x : ℝ, 4 * Real.sin x + a * Real.cos x ≤ 5) →
  (∃ x : ℝ, 4 * Real.sin x + a * Real.cos x = 5) →
  a = 3 ∨ a = -3 :=
by
  sorry

end max_value_sine_cosine_l691_691077


namespace find_abk_c_g_minimum_value_l691_691033

-- Definitions from conditions
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f (x)
def f (x : ℝ) := a * x ^ 3 + b * x + c
def tangent_slope_at_one := 3 * a + b
def is_perpendicular (m1 m2 : ℝ) := m1 * m2 = -1

-- Known values
def f'' (x : ℝ) := 6 * a
def f_min_third_derivative_value := 12

-- New function definition for g(x)
def g (x : ℝ) := f x / (x ^ 2)

-- Lean proof statement
theorem find_abk_c_g_minimum_value
  (a b c : ℝ)
  (h1 : is_odd_function f)
  (h2 : is_perpendicular tangent_slope_at_one (-1/18))
  (h3 : f'' 1 = f_min_third_derivative_value) :
  (a = 2) ∧ (b = 12) ∧ (c = 0) ∧ ∃ x > 0, g x = 4 * real.sqrt 6 :=
begin
  sorry
end

end find_abk_c_g_minimum_value_l691_691033


namespace range_of_a_l691_691429

noncomputable def f (a x : ℝ) : ℝ := (a / x) - Real.exp x
noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem range_of_a (a : ℝ) 
  (h : ∀ x1 ∈ Icc (1 / 2) 2, ∃ x2 ∈ Icc (1 / 2) 2, f a x1 - g x2 ≥ 1) :
  a ∈ Set.Ici (2 * Real.exp 2 - 2) :=
by
  sorry

end range_of_a_l691_691429


namespace tangent_line_condition_l691_691465

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 - b * x^2

theorem tangent_line_condition (a b : ℝ) (h1 : f 1 a b = 1) (h2 : deriv (f x a b) 1 = -1) :
  ∀ x ∈ Icc (-(1/2 : ℝ)) (3/2 : ℝ), f x (-1) (-1) ∈ Icc (-(9/8 : ℝ)) (3/8 : ℝ) :=
begin
  intro x,
  intro hx,
  sorry
end

end tangent_line_condition_l691_691465


namespace combination_simplify_l691_691373

theorem combination_simplify : (Nat.choose 6 2) + 3 = 18 := by
  sorry

end combination_simplify_l691_691373


namespace range_of_m_l691_691820

theorem range_of_m (m : ℝ) : (∃ x : ℝ, |x - 1| + |x + m| ≤ 4) → -5 ≤ m ∧ m ≤ 3 :=
by
  intro h
  sorry

end range_of_m_l691_691820


namespace domain_of_f_l691_691279

noncomputable def f (x : ℝ) : ℝ := (x + 6) / Real.sqrt (x^2 - 5*x + 6)

theorem domain_of_f : 
  {x : ℝ | x^2 - 5*x + 6 > 0} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_f_l691_691279


namespace correct_choice_is_B_l691_691582

noncomputable def f (x : ℝ) : ℝ := (2*x - x^2) * Real.exp x

lemma solution_set_f_pos : {x : ℝ | f x > 0} = {x | 0 < x ∧ x < 2} := sorry

lemma f_has_local_extrema :
  ∃ x₁ x₂ : ℝ, (IsLocalMin f x₁ ∧ IsLocalMax f x₂) := sorry

theorem correct_choice_is_B : {x : ℝ | f x > 0} = {x | 0 < x ∧ x < 2} ∧ 
  ∃ x₁ x₂ : ℝ, (IsLocalMin f x₁ ∧ IsLocalMax f x₂) :=
by { split, exact solution_set_f_pos, exact f_has_local_extrema }

end correct_choice_is_B_l691_691582


namespace solve_for_t_l691_691064

theorem solve_for_t (t : ℝ) (h : sqrt (3 * sqrt (t - 3)) = real.sqrt (real.sqrt (real.sqrt (10 - t)))) : t = 3.7 :=
by
  sorry

end solve_for_t_l691_691064


namespace min_value_proof_l691_691913

noncomputable def min_value (m n : ℝ) : ℝ := 
  if 4 * m + n = 1 ∧ (m > 0 ∧ n > 0) then (4 / m + 1 / n) else 0

theorem min_value_proof : ∃ m n : ℝ, 4 * m + n = 1 ∧ m > 0 ∧ n > 0 ∧ min_value m n = 25 :=
by
  -- stating the theorem conditionally 
  -- and expressing that there exists values of m and n
  sorry

end min_value_proof_l691_691913


namespace right_triangle_tangent_length_l691_691831

theorem right_triangle_tangent_length (DE DF : ℝ) (h1 : DE = 7) (h2 : DF = Real.sqrt 85)
  (h3 : ∀ (EF : ℝ), DE^2 + EF^2 = DF^2 → EF = 6): FQ = 6 :=
by
  sorry

end right_triangle_tangent_length_l691_691831


namespace time_difference_l691_691211

/-- The time on a digital clock is 5:55. We need to calculate the number
of minutes that will pass before the clock next shows a time with all digits identical,
which is 11:11. -/
theorem time_difference : 
  let t1 := 5 * 60 + 55  -- Time 5:55 in minutes past midnight
  let t2 := 11 * 60 + 11 -- Time 11:11 in minutes past midnight
  in t2 - t1 = 316 := 
by 
  let t1 := 5 * 60 + 55 
  let t2 := 11 * 60 + 11 
  have h : t2 - t1 = 316 := by sorry
  exact h

end time_difference_l691_691211


namespace problem_statement_l691_691628

theorem problem_statement : 25 * 15 * 9 * 5.4 * 3.24 = 3 ^ 10 := 
by 
  sorry

end problem_statement_l691_691628


namespace half_abs_diff_squares_l691_691233

theorem half_abs_diff_squares (a b : ℤ) (ha : a = 20) (hb : b = 15) : 
  (|a^2 - b^2| / 2) = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691233


namespace minimum_value_fx_cos_2x0_given_fx0_l691_691031

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x + 2 * cos x ^ 2 - 1

theorem minimum_value_fx :
  ∃ a ∈ Icc (0 : ℝ) (Real.pi / 2), ∀ x ∈ Icc (0 : ℝ) (Real.pi / 2), f x ≥ f a ∧ f a = -1 :=
sorry

theorem cos_2x0_given_fx0 :
  ∀ x0 ∈ Icc (Real.pi / 4) (Real.pi / 2), f x0 = 6 / 5 -> cos (2 * x0) = (3 - 4 * sqrt 3) / 10 :=
sorry

end minimum_value_fx_cos_2x0_given_fx0_l691_691031


namespace total_books_l691_691693

theorem total_books (B : ℕ) (s : ℕ) (hB : B = 20)
  (hs : s = B + (B / 4)) : B + s = 45 :=
by { rw [hB, hs], norm_num; sorry }

end total_books_l691_691693


namespace intersection_point_C1_C2_l691_691833

-- Define the parametric equations for curve C1
def parametric_C1 (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π/2) : ℝ × ℝ :=
  (sqrt 5 * cos θ, sqrt 5 * sin θ)

-- Define the parametric equations for curve C2
def parametric_C2 (t : ℝ) : ℝ × ℝ :=
  (1 - sqrt 2 / 2 * t, - sqrt 2 / 2 * t)

-- Prove the intersection point of curves C1 and C2 is (2, 1)
theorem intersection_point_C1_C2 : ∃ (x y : ℝ), 
  (∃ (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π/2), (x, y) = parametric_C1 θ hθ) ∧
  (∃ (t : ℝ), (x, y) = parametric_C2 t) ∧
  (x = 2 ∧ y = 1) :=
by sorry

end intersection_point_C1_C2_l691_691833


namespace remainder_of_N_eq_4101_l691_691850

noncomputable def N : ℕ :=
  20 + 3^(3^(3+1) - 13)

theorem remainder_of_N_eq_4101 : N % 10000 = 4101 := by
  sorry

end remainder_of_N_eq_4101_l691_691850


namespace domain_g_l691_691389

-- Define the function g(t)
def g (t : ℝ) : ℝ := 1 / ((t - 2)^2 + (t + 3)^2)

-- Prove that the domain of g(t) is all real numbers
theorem domain_g (t : ℝ) : ∀ t : ℝ, ((t - 2)^2 + (t + 3)^2) ≠ 0 :=
by
  intros t
  have h : (t - 2)^2 + (t + 3)^2 = 2 * t^2 + 2 * t + 13,
  -- Simplify and expand the polynomials
  sorry  -- this is where the simplification proof would go

  show (t - 2)^2 + (t + 3)^2 ≠ 0 from
  calc
    (t - 2)^2 + (t + 3)^2 = 2 * t^2 + 2 * t + 13 : by { exact h }
    ... ≠ 0 : by { simp, linarith }

end domain_g_l691_691389


namespace worksheets_turned_in_l691_691342

def initial_worksheets : ℕ := 34
def graded_worksheets : ℕ := 7
def remaining_worksheets : ℕ := initial_worksheets - graded_worksheets
def current_worksheets : ℕ := 63

theorem worksheets_turned_in :
  current_worksheets - remaining_worksheets = 36 :=
by
  sorry

end worksheets_turned_in_l691_691342


namespace coefficient_of_third_term_l691_691581

noncomputable def binomial_coefficient (n k : ℕ) : ℚ :=
  nat.choose n k

noncomputable def general_term (n a b : ℚ) (r : ℕ) : ℚ :=
  binomial_coefficient n r * (a^(n - r)) * (b^r)

theorem coefficient_of_third_term :
  general_term 5 1 (1 / 2) 2 = 5 / 2 := by
  sorry

end coefficient_of_third_term_l691_691581


namespace inequality_solution_l691_691208

theorem inequality_solution (x : ℝ) : 3 * x + 2 ≥ 5 ↔ x ≥ 1 :=
by sorry

end inequality_solution_l691_691208


namespace quadrilaterals_with_circumcenter_count_eq4_l691_691409

-- Definitions for each type of quadrilateral
def is_square (q : Quadrilateral) : Prop := is_regular q
def is_rectangle (q : Quadrilateral) : Prop := 
  is_equiangular q ∧ ¬is_regular q
def is_rhombus_non_square (q : Quadrilateral) : Prop := 
  is_equilateral q ∧ ¬is_equiangular q
def is_general_parallelogram (q : Quadrilateral) : Prop := 
  is_parallelogram q ∧ ¬is_equilateral q ∧ ¬is_equiangular q
def is_isosceles_trapezoid_non_parallelogram (q : Quadrilateral) : Prop := 
  is_isosceles_trapezoid q ∧ ¬is_parallelogram q
def is_special_kite (q : Quadrilateral) : Prop := 
  is_kite q ∧ (exists d1 d2, d1 ⊥ d2 ∧ is_bisector d1 d2 q)

-- The main theorem
theorem quadrilaterals_with_circumcenter_count_eq4 :
  ∃ sq rt rns par it pk, 
  (is_square sq ∧ has_circumcenter sq) ∧
  (is_rectangle rt ∧ has_circumcenter rt) ∧
  (is_rhombus_non_square rns ∧ ¬has_circumcenter rns) ∧
  (is_general_parallelogram par ∧ ¬has_circumcenter par) ∧
  (is_isosceles_trapezoid_non_parallelogram it ∧ has_circumcenter it) ∧
  (is_special_kite pk ∧ has_circumcenter pk) ∧
  ∃ lst, lst = [sq, rt, it, pk] ∧ list.length lst = 4 :=
begin
  sorry
end

end quadrilaterals_with_circumcenter_count_eq4_l691_691409


namespace probability_A_first_vehicle_equal_half_l691_691007

-- Definitions based on conditions
def students : Set String := {"A", "B", "C", "D"}
def vehicles : Set Nat := {1, 2}

-- Function to determine all ways of distributing students into two vehicles
def all_distributions : Finset (Set (String × Nat)) :=
  {s | (∃ (a b : String), {(a, 1), (b, 1), (a, 2), (b, 2)} = students × vehicles)}.to_finset

-- Function to determine distributions where A is in the first vehicle
def distributions_with_A_in_first : Finset (Set (String × Nat)) :=
  {s | ∃ (b : String), {(("A", 1), (b, 1), ("A", 2), (b, 2))} = students × vehicles}.to_finset

-- Total number of distributions
def total_distributions := all_distributions.card

-- Number of distributions where A is in the first vehicle
def favorable_distributions := distributions_with_A_in_first.card

-- Probability that A is in the first vehicle
noncomputable def probability_A_in_first_vehicle : ℚ :=
  favorable_distributions / total_distributions

-- Proof statement
theorem probability_A_first_vehicle_equal_half :
  probability_A_in_first_vehicle = (1 / 2) :=
by
  -- Place to insert proof
  sorry

end probability_A_first_vehicle_equal_half_l691_691007


namespace limit_ln_ex_power_l691_691374

noncomputable def limit_expr (x : ℝ) : ℝ :=
  (Real.log(ex) ^ 2) ^ (1 / (x ^ 2 + 1))

theorem limit_ln_ex_power (x : ℝ) : 
  filter.tendsto (λ x, limit_expr x) (nhds 1) (nhds 1) := sorry

end limit_ln_ex_power_l691_691374


namespace c_is_perfect_square_or_not_even_c_cannot_be_even_l691_691750

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem c_is_perfect_square_or_not_even 
  (a b c : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b))
  (h_odd : c % 2 = 1) : is_perfect_square c :=
sorry

theorem c_cannot_be_even 
  (a b c : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b))
  (h_even : c % 2 = 0) : false :=
sorry

end c_is_perfect_square_or_not_even_c_cannot_be_even_l691_691750


namespace smallest_value_l691_691775

variable (a : ℝ) (h : a > 1)

def p := Real.sqrt (1993 * a) - Real.sqrt (1993 * a - 1)
def q := Real.sqrt (1993 * a - 1) - Real.sqrt (1993 * a)
def r := Real.sqrt (1993 * a) - Real.sqrt (1993 * a + 1)
def s := Real.sqrt (1993 * a + 1) - Real.sqrt (1993 * a)

theorem smallest_value : q < p ∧ q < r ∧ q < s := 
by
  sorry

end smallest_value_l691_691775


namespace part_I_B_does_not_have_propertyP_part_I_C_has_propertyP_part_II_T_has_propertyP_l691_691470

open Set

def propertyP (A : Set ℕ) (S : Set ℕ) (n : ℕ) : Prop := 
  ∃ m : ℕ, m ∈ Icc(1, n) ∧ ∀ s1 s2 ∈ S, s1 ≠ s2 → |s1 - s2| ≠ m

theorem part_I_B_does_not_have_propertyP (n : ℕ) (h_n : n = 10) :
  ¬ propertyP (Icc 1 (2 * n)) {x | x ∈ Icc 1 (2 * n) ∧ x > 9} 10 :=
  sorry

theorem part_I_C_has_propertyP (n : ℕ) (h_n : n = 10) :
  propertyP (Icc 1 (2 * n)) {x | ∃ k : ℕ, k > 0 ∧ x = 3 * k - 1 ∧ x ∈ Icc 1 (2 * n)} 10 :=
  sorry

theorem part_II_T_has_propertyP (A S : Set ℕ) (n : ℕ) (hS : propertyP A S n) :
  propertyP A ({x | ∃ s ∈ S, x = (2 * n + 1) - s}) n :=
  sorry

end part_I_B_does_not_have_propertyP_part_I_C_has_propertyP_part_II_T_has_propertyP_l691_691470


namespace time_difference_l691_691213

/-- The time on a digital clock is 5:55. We need to calculate the number
of minutes that will pass before the clock next shows a time with all digits identical,
which is 11:11. -/
theorem time_difference : 
  let t1 := 5 * 60 + 55  -- Time 5:55 in minutes past midnight
  let t2 := 11 * 60 + 11 -- Time 11:11 in minutes past midnight
  in t2 - t1 = 316 := 
by 
  let t1 := 5 * 60 + 55 
  let t2 := 11 * 60 + 11 
  have h : t2 - t1 = 316 := by sorry
  exact h

end time_difference_l691_691213


namespace find_prism_height_l691_691848

variables (base_side_length : ℝ) (density : ℝ) (weight : ℝ) (height : ℝ)

-- Assume the base_side_length is 2 meters, density is 2700 kg/m³, and weight is 86400 kg
def given_conditions := (base_side_length = 2) ∧ (density = 2700) ∧ (weight = 86400)

-- Define the volume based on weight and density
noncomputable def volume (density weight : ℝ) : ℝ := weight / density

-- Define the area of the base
def base_area (side_length : ℝ) : ℝ := side_length * side_length

-- Define the height of the prism
noncomputable def prism_height (volume base_area : ℝ) : ℝ := volume / base_area

-- The proof statement
theorem find_prism_height (h : ℝ) : given_conditions base_side_length density weight → prism_height (volume density weight) (base_area base_side_length) = h :=
by
  intros h_cond
  sorry

end find_prism_height_l691_691848


namespace Tetrahedron_Sphere_Equal_Area_l691_691536

-- Define the main problem conditions and establish the proof requirement.
theorem Tetrahedron_Sphere_Equal_Area :
  ∃ (T S: Type) (tetrahedron : T) (sphere: S) (plane : ℝ^3) (area : ℝ),
    (∀ plane_parallel : ℝ^3, (plane_parallel || plane) → 
    area_intersection tetrahedron plane_parallel = area_intersection sphere plane_parallel) :=
sorry

end Tetrahedron_Sphere_Equal_Area_l691_691536


namespace half_abs_diff_squares_l691_691232

theorem half_abs_diff_squares (a b : ℤ) (ha : a = 20) (hb : b = 15) : 
  (|a^2 - b^2| / 2) = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691232


namespace find_p_l691_691702

theorem find_p (p q : ℝ) (hp : 0 < p) (hq : 0 < q)
  (h : let x1 := (-p + Real.sqrt (p^2 - 4 * q)) / 2
       let x2 := (-p - Real.sqrt (p^2 - 4 * q)) / 2
       in abs (x1 - x2) = 1) :
  p = Real.sqrt (4 * q + 1) :=
by
  sorry

end find_p_l691_691702


namespace coefficient_x2_expansion_l691_691728

theorem coefficient_x2_expansion :
  let f := fun n : ℕ => (1 + x)^n
  let coefficients := (List.range 2 (9 + 1)).map (λ n, binomial n 2)
  coefficients.sum = 120 := 
by
  sorry

end coefficient_x2_expansion_l691_691728


namespace number_of_ways_to_divide_friends_to_teams_l691_691805

theorem number_of_ways_to_divide_friends_to_teams :
  (let n := 8 in
   let k := 4 in
   k^n = 65536) :=
by
  let n := 8
  let k := 4
  have h1 : k^n = 4^8 := by rfl
  have h2 : 4^8 = 65536 := by sorry
  exact eq.trans h1 h2

end number_of_ways_to_divide_friends_to_teams_l691_691805


namespace identical_digits_time_l691_691216

theorem identical_digits_time (h : ∀ t, t = 355 -> ∃ u, u = 671 ∧ u - t = 316) : 
  ∃ u, u = 671 ∧ u - 355 = 316 := 
by sorry

end identical_digits_time_l691_691216


namespace cost_of_old_car_l691_691690

theorem cost_of_old_car (C_old C_new : ℝ): 
  C_new = 2 * C_old → 
  1800 + 2000 = C_new → 
  C_old = 1900 :=
by
  intros H1 H2
  sorry

end cost_of_old_car_l691_691690


namespace hemisphere_contains_at_least_four_points_l691_691427

-- Define the problem
theorem hemisphere_contains_at_least_four_points (S : Type) [sphere S] 
  (P : Fin₅ → S) : ∃ H : hemisphere S, ∃ (subset : Fin₅ → Prop), (∀ i, subset i → H.contains (P i)) ∧ (card {i | subset i} ≥ 4) :=
sorry

end hemisphere_contains_at_least_four_points_l691_691427


namespace largest_six_digit_number_l691_691623

def initial_number := "778157260669103"
def digits_to_remove := 9
def required_length := 6

theorem largest_six_digit_number : 
  (∃ (remaining_digits : List Char), 
    remaining_digits.length = required_length ∧ 
    List.is_subsequence remaining_digits (initial_number.to_list) ∧ 
    879103 = remaining_digits.foldl (λ acc d, acc * 10 + (d.to_char_digit).get) 0) := 
by {
  sorry
}

end largest_six_digit_number_l691_691623


namespace trig_identity_l691_691600

theorem trig_identity :
  cos (43 * Real.pi / 180) * cos (13 * Real.pi / 180) + sin (43 * Real.pi / 180) * sin (13 * Real.pi / 180) =
  √3 / 2 :=
by
  sorry

end trig_identity_l691_691600


namespace count_multiples_of_6_with_units_digit_6_l691_691799

theorem count_multiples_of_6_with_units_digit_6 : 
  ∃ (n : ℕ), n = 5 ∧ ∀ (m : ℕ), (m > 0 ∧ m < 150 ∧ m % 6 = 0 ∧ m % 10 = 6) ↔ m ∈ {6, 36, 66, 96, 126} :=
by
  sorry

end count_multiples_of_6_with_units_digit_6_l691_691799


namespace percentage_x_is_correct_l691_691764

-- Define the initial conditions
def y_ratio : Float := 2 / 10
def z_ratio : Float := 3 / 10
def water_ratio : Float := 5 / 10
def initial_weight_y : Float := 6
def evaporated_z : Float := 1
def added_weight_y : Float := 2

-- Define the initial amounts of liquids x, z, and water in the solution
def initial_x : Float := (y_ratio * initial_weight_y)
def initial_z : Float := (z_ratio * initial_weight_y)
def initial_water : Float := (water_ratio * initial_weight_y)

-- Define the remaining amounts after evaporation
def remaining_x : Float := initial_x
def remaining_z : Float := initial_z - evaporated_z
def remaining_water : Float := initial_water
def remaining_total : Float := remaining_x + remaining_z + remaining_water

-- Define the added amounts of liquids x, z, and water in the solution y
def added_x : Float := (y_ratio * added_weight_y)
def added_z : Float := (z_ratio * added_weight_y)
def added_water : Float := (water_ratio * added_weight_y)

-- Define the new total amounts after adding the new solution y
def new_x : Float := remaining_x + added_x
def new_z : Float := remaining_z + added_z
def new_water : Float := remaining_water + added_water
def new_total : Float := new_x + new_z + new_water

-- Define the percentage of liquid x in the new solution
def percentage_x : Float := (new_x / new_total) * 100

-- Statement to be proved
theorem percentage_x_is_correct : percentage_x ≈ 22.86 := by
  sorry

end percentage_x_is_correct_l691_691764


namespace least_number_with_remainder_4_l691_691641

theorem least_number_with_remainder_4 : ∃ n : ℕ, n = 184 ∧ 
  (∀ d ∈ [5, 9, 12, 18], (n - 4) % d = 0) ∧
  (∀ m : ℕ, (∀ d ∈ [5, 9, 12, 18], (m - 4) % d = 0) → m ≥ n) :=
by
  sorry

end least_number_with_remainder_4_l691_691641


namespace min_abs_sum_l691_691736

theorem min_abs_sum (x : ℝ) : 
  has_le.le 9 (abs (x - 3) + abs (x - 1) + abs (x + 6)) := 
sorry

end min_abs_sum_l691_691736


namespace range_of_m_l691_691784

def quadratic_inequality_solution_set (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) (x : ℝ) : Prop :=
  ax^2 - ax - 2a^2 > 1 → x ∈ Ioo -a (2 * a)

def function_domain (a m : ℝ) (ha : a > 0) : Prop :=
  ∀ x : ℝ, ∃ z : ℝ, (a⁻¹)^(x^2 + 2 * m * x - m) - 1 = z ∧ z ≥ 0

theorem range_of_m (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) :
  ∃ m_range : set ℝ, (∀ m ∈ m_range, function_domain a m ha1) ∧ m_range = Icc -1 0 :=
sorry

end range_of_m_l691_691784


namespace work_days_l691_691484

theorem work_days (m r d : ℕ) (h : 2 * m * d = 2 * (m + r) * (md / (m + r))) : d = md / (m + r) :=
by
  sorry

end work_days_l691_691484


namespace candy_problem_minimum_candies_l691_691935

theorem candy_problem_minimum_candies : ∃ (N : ℕ), N > 1 ∧ N % 2 = 1 ∧ N % 3 = 1 ∧ N % 5 = 1 ∧ N = 31 :=
by
  sorry

end candy_problem_minimum_candies_l691_691935


namespace base_8_addition_l691_691395

def digit_add (d1 d2 c h : ℕ) : ℕ × ℕ := 
  let sum := d1 + d2 + c
  (sum % h, sum / h)

theorem base_8_addition :
  ∀ (h : ℕ), 
  let col1 := digit_add 2 1 0 h,
      col2 := digit_add 4 2 col1.2 h,
      col3 := digit_add 3 4 col2.2 h,
      col4 := digit_add 5 6 col3.2 h
  in h = 8 → 
     col1.1 = 3 ∧ 
     col2.1 = 6 ∧ 
     col3.1 = 2 ∧ col3.2 = 1 ∧ 
     col4.1 = 4 ∧ col4.2 = 1 := 
by
  intro h
  simp [digit_add]
  assume : h = 8
  rw this
  simp
  sorry

end base_8_addition_l691_691395


namespace fraction_a_over_d_l691_691809

-- Defining the given conditions as hypotheses
variables (a b c d : ℚ)

-- Conditions
axiom h1 : a / b = 20
axiom h2 : c / b = 5
axiom h3 : c / d = 1 / 15

-- Goal to prove
theorem fraction_a_over_d : a / d = 4 / 15 :=
by
  sorry

end fraction_a_over_d_l691_691809


namespace constant_term_in_expansion_max_binomial_term_at_x_equals_4_l691_691027

open Nat BigOperators

-- Definitions and Hypotheses
variables {x : ℝ} {n : ℕ}
variable (general_expansion : (ℝ → ℝ) → ℕ → ℝ)
variable (ratio_condition : general_expansion = (λ (f : ℝ → ℝ) (n : ℕ), ∑ k in finset.range (n + 1), (n.choose k) * (√x ^ (n - k)) * ((2 / x^2) ^ k)))

-- Proof Problems
theorem constant_term_in_expansion
  (h : (ratio_condition (λ x, sqrt x + 2 / x^2) 10)) :
  general_expansion sqrt x + 2 / x^2 10 = 180 :=
sorry

theorem max_binomial_term_at_x_equals_4
  (h : (ratio_condition (λ x, sqrt x + 2 / x^2) 10)) :
  general_expansion (λ x, sqrt x + 2 / x^2) 4 = 63 / 256 :=
sorry

end constant_term_in_expansion_max_binomial_term_at_x_equals_4_l691_691027


namespace intersecting_lines_set_equals_plane_l691_691954

noncomputable def skew_lines {P : Type*} [EuclideanGeometry P] (a b : Line P) :=
¬ ∃ (p : P), p ∈ a ∧ p ∈ b ∧ (∃ q : P, q ∈ a ∧ q ≠ p ∧ q ∈ b ∧ q ≠ p)

theorem intersecting_lines_set_equals_plane {P : Type*} [EuclideanGeometry P]
  (a b : Line P) (ha_skew : skew_lines a b) :
  ∃ (α : Plane P), (∀ (p : P), (∃ (l : Line P), l ∈ PlaneLines α ∧ ∃ p1 ∈ l, p1 ∈ a ∧ l.is_parallel_to b) ↔ p ∈ PlanePoints α) :=
by
  sorry

end intersecting_lines_set_equals_plane_l691_691954


namespace average_salary_l691_691594

def A_salary : ℝ := 9000
def B_salary : ℝ := 5000
def C_salary : ℝ := 11000
def D_salary : ℝ := 7000
def E_salary : ℝ := 9000
def number_of_people : ℝ := 5
def total_salary : ℝ := A_salary + B_salary + C_salary + D_salary + E_salary

theorem average_salary : (total_salary / number_of_people) = 8200 := by
  sorry

end average_salary_l691_691594


namespace find_expression_and_intervals_l691_691030

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

theorem find_expression_and_intervals (ω ϕ : ℝ) (hω : ω > 0) (hϕ : 0 < ϕ ∧ ϕ < Real.pi / 2) 
  (distance_axes : ∀ x, (x + (Real.pi / 2) / ω) - x = Real.pi / 2) 
  (zero_of_f : f (Real.pi / 3) = 0) : 
  f = (λ x, 2 * Real.sin (2 * x + Real.pi / 3)) ∧ 
  (∀ x, x ∈ set.Icc (0 : ℝ) (3 * Real.pi / 2) → 
    (f.deriv x > 0 ↔ 
      (x ∈ set.Icc (0 : ℝ) (Real.pi / 12) ∨ 
      x ∈ set.Icc (7 * Real.pi / 12) (13 * Real.pi / 12)))) :=
by
  -- proofs go here
  sorry

end find_expression_and_intervals_l691_691030


namespace find_t_l691_691060

theorem find_t (t : ℝ) (h : sqrt (3 * sqrt (t - 3)) = (10 - t)^(1/4)) : t = 3.7 :=
sorry

end find_t_l691_691060


namespace bens_old_car_cost_l691_691691

theorem bens_old_car_cost :
  ∃ (O N : ℕ), N = 2 * O ∧ O = 1800 ∧ N = 1800 + 2000 ∧ O = 1900 :=
by 
  sorry

end bens_old_car_cost_l691_691691


namespace statement_A_correct_statement_D_correct_l691_691714

-- Conditions from the problem
def richter_scale (A_max A_0 : ℝ) : ℝ := log10 (A_max / A_0)
def earthquake_energy (M : ℝ) : ℝ := 10^(4.8 + 1.5*M)

-- Proof that Statement A is correct
theorem statement_A_correct (A_max A_0 : ℝ) (M : ℝ) : 
  richter_scale A_max A_0 + 1 = richter_scale (10 * A_max) A_0 := 
sorry

-- Proof that Statement D is correct
theorem statement_D_correct (A_max A_0 : ℝ) (M : ℝ) :
  earthquake_energy (richter_scale (100 * A_max) A_0) = 1000 * earthquake_energy (richter_scale A_max A_0) := 
sorry

end statement_A_correct_statement_D_correct_l691_691714


namespace correct_fill_is_C_l691_691646

-- Definitions of the fill-in-the-blank conditions
def fill_blank_options : list (string × string × string) :=
  [ ("A", "", "") -- for placeholder purpose, Lean expects all branches to have similar structures
  , ("B", "a", "the")
  , ("C", "a", "")
  , ("D", "the", "the")
  ]

-- The original question's statement, reformulated
def check_answer (first_fill second_fill : string) : Prop :=
  (first_fill = "a" ∧ second_fill = "")

-- Prove that "C" is correct
theorem correct_fill_is_C : check_answer (fill_blank_options.get! 2).2 (fill_blank_options.get! 2).3 :=
by
  sorry

end correct_fill_is_C_l691_691646


namespace num_primes_of_396_l691_691479

-- Define 396 and its prime factorization
def n : ℕ := 396
def prime_factors : List ℕ := [2, 3, 11]

-- Define the number of unique prime factors
def num_prime_factors (n : ℕ) : ℕ :=
  prime_factors.to_finset.card

-- The main theorem statement
theorem num_primes_of_396 : num_prime_factors n = 3 := sorry

end num_primes_of_396_l691_691479


namespace Amanda_distance_l691_691347

theorem Amanda_distance : 
  let south_distance := 12
  let west_distance := 40
  let north_distance := 20
  let net_south_north := south_distance - north_distance
  let net_west_east := west_distance
  sqrt (net_south_north ^ 2 + net_west_east ^ 2) = sqrt 1664 := 
by
  sorry

end Amanda_distance_l691_691347


namespace total_number_of_workers_l691_691908

theorem total_number_of_workers 
  (W : ℕ) 
  (avg_all : ℕ) 
  (n_technicians : ℕ) 
  (avg_technicians : ℕ) 
  (avg_non_technicians : ℕ) :
  avg_all * W = avg_technicians * n_technicians + avg_non_technicians * (W - n_technicians) →
  avg_all = 8000 →
  n_technicians = 7 →
  avg_technicians = 12000 →
  avg_non_technicians = 6000 →
  W = 21 :=
by 
  intro h1 h2 h3 h4 h5
  sorry

end total_number_of_workers_l691_691908


namespace parity_of_f_min_value_of_f_min_value_of_f_l691_691857

open Real

def f (a x : ℝ) := x^2 + abs (x - a) + 1

theorem parity_of_f (a : ℝ) :
  (∀ x : ℝ, f 0 x = f 0 (-x)) ∧ (∀ x : ℝ, f a x ≠ f a (-x) ∧ f a x ≠ -f a x) ↔ a = 0 :=
by sorry

theorem min_value_of_f (a : ℝ) (h : a ≤ -1/2) : 
  ∀ x : ℝ, x ≥ a → f a x ≥ f a (-1/2) :=
by sorry

theorem min_value_of_f' (a : ℝ) (h : -1/2 < a) : 
  ∀ x : ℝ, x ≥ a → f a x ≥ f a a :=
by sorry

end parity_of_f_min_value_of_f_min_value_of_f_l691_691857


namespace choose_sphere_tetrahedron_plane_to_intersect_in_equal_areas_l691_691534

noncomputable def equal_intersection_areas
  (AB CD : Line)
  (I : Line)
  (S : ℝ)
  (r : ℝ)
  (hTetrahedron : Tetrahedron)
  (hSphere : Sphere) : Prop :=
  let M_AB := midpoint AB
  let M_CD := midpoint CD
  let hI := line_segment I M_AB M_CD
  let hRect := rectangle_area (cross_section hTetrahedron M_AB M_CD)
  let r := real.sqrt (S / real.pi)
  let hLengthI := length I = 2 * r
  in
  ∀ (plane : Plane), is_parallel plane (reference_plane M_AB M_CD) → 
    (area (plane ∩ hTetrahedron) = area (plane ∩ hSphere))

theorem choose_sphere_tetrahedron_plane_to_intersect_in_equal_areas 
  (AB CD : Line)
  (I : Line)
  (S : ℝ)
  (r : ℝ)
  (hTetrahedron : Tetrahedron)
  (hSphere : Sphere) 
  (hAB_CD_perpendicular : is_perpendicular AB CD)
  (hI_midpoints : I = line_segment (midpoint AB) (midpoint CD))
  (hRect_area : rectangle_area (cross_section hTetrahedron (midpoint AB) (midpoint CD)) = S)
  (hradius : r = real.sqrt (S / real.pi))
  (hI_length : length I = 2 * r)
  : equal_intersection_areas AB CD I S r hTetrahedron hSphere := 
  sorry

end choose_sphere_tetrahedron_plane_to_intersect_in_equal_areas_l691_691534


namespace relationship_among_a_b_c_l691_691974

noncomputable def a := 20.3
noncomputable def b := 2 ^ 10.35
noncomputable def c := Real.log2 1.2

theorem relationship_among_a_b_c : b > a ∧ a > c := by
  -- Proof is omitted, just shows that hypothesis and conclusion types are correct.
  sorry

end relationship_among_a_b_c_l691_691974


namespace solve_for_t_l691_691066

theorem solve_for_t (t : ℝ) (h : sqrt (3 * sqrt (t - 3)) = real.sqrt (real.sqrt (real.sqrt (10 - t)))) : t = 3.7 :=
by
  sorry

end solve_for_t_l691_691066


namespace find_a_l691_691462

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  (1 + Real.cos (2 * x)) / (4 * Real.sin (Real.pi / 2 + x)) - 
  a * Real.sin (x / 2) * Real.cos (Real.pi - x / 2)

theorem find_a (a : ℝ) :
  (∀ x, f x a ≤ 2) → (∃ a : ℝ, a = √15 ∨ a = -√15) :=
begin
  sorry
end

end find_a_l691_691462


namespace probability_jerry_at_four_l691_691110

theorem probability_jerry_at_four :
  let total_flips := 8
  let coordinate := 4
  let total_possible_outcomes := 2 ^ total_flips
  let favorable_outcomes := Nat.choose total_flips (total_flips / 2 + coordinate / 2)
  let P := favorable_outcomes / total_possible_outcomes
  let a := 7
  let b := 64
  ∃ (a b : ℕ), Nat.gcd a b = 1 ∧ P = a / b ∧ a + b = 71
:= sorry

end probability_jerry_at_four_l691_691110


namespace minimum_value_expression_l691_691500

theorem minimum_value_expression 
  (a b : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_eq : 1 / a + 1 / b = 1) : 
  (∃ (x : ℝ), x = (1 / (a-1) + 9 / (b-1)) ∧ x = 6) :=
sorry

end minimum_value_expression_l691_691500


namespace extreme_value_at_1_minimum_b_l691_691781

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- Statement for Part (1)
theorem extreme_value_at_1 (a b : ℝ) : 
  (f 1 a b = 10) → ((f' = (3 + 2 * a + b) = 0) ∧ (f 1 a b = 1 + a + b + a * a)) → b = -11 := 
sorry

-- Statement for Part (2)
theorem minimum_b (a b : ℝ) :
  (∀ a ∈ set.Ici (-4), ∀ x ∈ set.Icc 0 2, 3*x^2 + 2*a*x + b ≥ 0) 
  → b ≥ 16/3 :=
sorry

end extreme_value_at_1_minimum_b_l691_691781


namespace probability_angie_carlos_opposite_l691_691353

-- Definitions as per conditions
section seating_problem

universes u

def seated_at_random : Prop := ∃ (positions : finset (fin 4)) (people : finset (fin 4)), 
  positions.card = 4 ∧ people.card = 4

def angie_position : fin 4 := 0

def carlos_opposite_angie (positions : finset (fin 4)) (carlos_position : fin 4) : Prop :=
  angie_position ≠ carlos_position ∧ (carlos_position + 2) % 4 = angie_position

-- Problem statement
theorem probability_angie_carlos_opposite : 
  seated_at_random → 
  (∃ (positions : finset (fin 4)) (people : finset (fin 4)) (carlos_position : fin 4), 
     carlos_opposite_angie positions carlos_position) → 
  (1 / 3 : ℚ) := 
sorry

end seating_problem

end probability_angie_carlos_opposite_l691_691353


namespace sqrt_sum_eq_nine_l691_691068

theorem sqrt_sum_eq_nine (x : ℝ) (h : Real.sqrt (7 + x) + Real.sqrt (28 - x) = 9) :
  (7 + x) * (28 - x) = 529 :=
sorry

end sqrt_sum_eq_nine_l691_691068


namespace circle_values_of_a_l691_691186

theorem circle_values_of_a (a : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 + 2*a*x + 2*a*y + 2*a^2 + a - 1 = 0) ↔ (a = -1 ∨ a = 0) :=
by
  sorry

end circle_values_of_a_l691_691186


namespace half_abs_diff_squares_l691_691256

theorem half_abs_diff_squares (a b : ℤ) (h₁ : a = 20) (h₂ : b = 15) : 
  (|a^2 - b^2| / 2 : ℚ) = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691256


namespace nth_inequality_l691_691773

theorem nth_inequality (n : ℕ) :
  (∑ i in Finset.range (2^(n + 1) - 1), (1 : ℚ) / (i + 1)) > (n + 1) / 2 :=
sorry

end nth_inequality_l691_691773


namespace mad_hatter_must_secure_at_least_70_percent_l691_691834

theorem mad_hatter_must_secure_at_least_70_percent :
  ∀ (N : ℕ) (uM uH uD : ℝ) (α : ℝ),
    uM = 0.2 ∧ uH = 0.25 ∧ uD = 0.3 → 
    uM + α * 0.25 ≥ 0.25 + (1 - α) * 0.25 ∧
    uM + α * 0.25 ≥ 0.3 + (1 - α) * 0.25 →
    α ≥ 0.7 :=
by
  intros N uM uH uD α h hx
  sorry 

end mad_hatter_must_secure_at_least_70_percent_l691_691834


namespace sum_of_binomial_coefficients_l691_691758

noncomputable def integral_value : ℝ :=
  ∫ x in 1..16, (1 / real.sqrt x)

theorem sum_of_binomial_coefficients :
  let n := integral_value in
  n = 6 →
  (∑ i in finset.range (n + 1), nat.choose n i) = 64 :=
by 
  intro n h
  rw h
  have binom_sum : (∑ i in finset.range (6 + 1), nat.choose 6 i) = 2^6 := 
  by norm_num
  exact binom_sum

end sum_of_binomial_coefficients_l691_691758


namespace smallest_coefficient_term_l691_691099

variables {x y : ℝ}

def expansion_term (n k : ℕ) : ℝ :=
  if k % 2 = 0 then (-1)^(k/2) * binomial n k * x^(n-k) * y^k
  else binomial n k * x^(n-k) * y^k

theorem smallest_coefficient_term :
  (∃ k : ℕ, 1 ≤ k + 1 ∧ k + 1 ≤ 11 ∧ 
  (∀ j : ℕ, 1 ≤ j ≤ 11 → 
    if expansion_term 10 j < expansion_term 10 k then k + 1 = 6)) :=
sorry

end smallest_coefficient_term_l691_691099


namespace find_k_range_and_line_eq_l691_691419

open Real

-- Definitions
def circle (C : Type) (x y : C) : Bool := (x - 6)^2 + y^2 = 20

def line (l : Type) (k : ℝ) (x y : l) : Bool := y = k * x

-- Proof problem statement
theorem find_k_range_and_line_eq (k : ℝ) (A B : ℝ × ℝ) :
  circle ℝ A.1 A.2 ∧ circle ℝ B.1 B.2 ∧ line ℝ k A.1 A.2 ∧ line ℝ k B.1 B.2 ∧ A ≠ B →
  (-sqrt 5 / 2 < k ∧ k < sqrt 5 / 2) ∧ (B.1 = 2 * A.1 → (k = 1 ∨ k = -1)) :=
by
  sorry

end find_k_range_and_line_eq_l691_691419


namespace cos_difference_statement_l691_691414

variable (θ : ℝ)

-- Defining the given condition
def sin_eq_condition : Prop :=
  sin (3 * π - θ) = (√5 / 2) * sin (π / 2 + θ)

-- The statement to be proven
theorem cos_difference_statement (h : sin_eq_condition θ) :
  cos (θ - π / 3) = ±(1 / 3 + √15 / 6) :=
sorry

end cos_difference_statement_l691_691414


namespace member_age_greater_than_zero_l691_691508

def num_members : ℕ := 23
def avg_age : ℤ := 0
def age_range : Set ℤ := {x | x ≥ -20 ∧ x ≤ 20}
def num_negative_members : ℕ := 5

theorem member_age_greater_than_zero :
  ∃ n : ℕ, n ≤ 18 ∧ (avg_age = 0 ∧ num_members = 23 ∧ num_negative_members = 5 ∧ ∀ age ∈ age_range, age ≥ -20 ∧ age ≤ 20) :=
sorry

end member_age_greater_than_zero_l691_691508


namespace half_abs_diff_squares_l691_691274

theorem half_abs_diff_squares : 
  let a := 20 in
  let b := 15 in
  let square (x : ℕ) := x * x in
  let abs_diff := abs (square a - square b) in
  abs_diff / 2 = 87.5 := 
by
  let a := 20
  let b := 15
  let square (x : ℕ) := x * x
  let abs_diff := abs (square a - square b)
  have h1 : square a = 400 := by native_decide
  have h2 : square b = 225 := by native_decide
  have h3 : abs_diff = abs (400 - 225) := by simp [square, h1, h2]
  have h4 : abs_diff = 175 := by simp [h3]
  have h5 : abs_diff / 2 = 87.5 := by norm_num [h4]
  exact h5

end half_abs_diff_squares_l691_691274


namespace remainder_sum_first_150_div_10000_l691_691950
open Nat

theorem remainder_sum_first_150_div_10000 :
  ((∑ i in Finset.range 151, i) % 10000) = 1325 :=
sorry

end remainder_sum_first_150_div_10000_l691_691950


namespace derivative_of_x_plus_cos_x_l691_691583

theorem derivative_of_x_plus_cos_x :
  (deriv (λ x : ℝ, x + real.cos x)) = (λ x, 1 - real.sin x) := 
sorry

end derivative_of_x_plus_cos_x_l691_691583


namespace calculate_original_goose_eggs_l691_691511

def hatched := 4 / 7
def survived_first_month := 3 / 5
def survived_next_three_months := 7 / 10
def survived_six_months_after := 5 / 8
def survived_first_half_year_two := 2 / 3
def survived_second_half_year_two := 4 / 5
def num_survived_two_years := 200

noncomputable def original_eggs :=
  (num_survived_two_years / survived_second_half_year_two / survived_first_half_year_two /
    survived_six_months_after / survived_next_three_months / survived_first_month / hatched).ceil

theorem calculate_original_goose_eggs :
  original_eggs = 2503 :=
by
  sorry

end calculate_original_goose_eggs_l691_691511


namespace beth_sold_l691_691362

theorem beth_sold {initial_coins additional_coins total_coins sold_coins : ℕ} 
  (h_init : initial_coins = 125)
  (h_add : additional_coins = 35)
  (h_total : total_coins = initial_coins + additional_coins)
  (h_sold : sold_coins = total_coins / 2) :
  sold_coins = 80 := 
sorry

end beth_sold_l691_691362


namespace first_representation_second_representation_third_representation_l691_691292

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

end first_representation_second_representation_third_representation_l691_691292


namespace Tetrahedron_Sphere_Equal_Area_l691_691537

-- Define the main problem conditions and establish the proof requirement.
theorem Tetrahedron_Sphere_Equal_Area :
  ∃ (T S: Type) (tetrahedron : T) (sphere: S) (plane : ℝ^3) (area : ℝ),
    (∀ plane_parallel : ℝ^3, (plane_parallel || plane) → 
    area_intersection tetrahedron plane_parallel = area_intersection sphere plane_parallel) :=
sorry

end Tetrahedron_Sphere_Equal_Area_l691_691537


namespace sphere_radius_l691_691766

theorem sphere_radius 
  (r h1 h2 : ℝ)
  (A1_eq : 5 * π = π * (r^2 - h1^2))
  (A2_eq : 8 * π = π * (r^2 - h2^2))
  (h1_h2_eq : h1 - h2 = 1) : r = 3 :=
by
  sorry

end sphere_radius_l691_691766


namespace a2_a4_a6_a8_a10_a12_sum_l691_691052

theorem a2_a4_a6_a8_a10_a12_sum :
  ∀ (x : ℝ), 
    (1 + x + x^2)^6 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7 + a8 * x^8 + a9 * x^9 + a10 * x^10 + a11 * x^11 + a12 * x^12 →
    a2 + a4 + a6 + a8 + a10 + a12 = 364 :=
sorry

end a2_a4_a6_a8_a10_a12_sum_l691_691052


namespace interest_equality_l691_691823

-- Definitions based on the conditions
def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := P * r * t

-- Constants for the problem
def P1 : ℝ := 200 -- 200 Rs is the principal of the first case
def r1 : ℝ := 0.1 -- 10% converted to a decimal
def t1 : ℝ := 12 -- 12 years

def P2 : ℝ := 1000 -- Correct answer for the other amount
def r2 : ℝ := 0.12 -- 12% converted to a decimal
def t2 : ℝ := 2 -- 2 years

-- Theorem stating that the interest generated is the same
theorem interest_equality : 
  simple_interest P1 r1 t1 = simple_interest P2 r2 t2 :=
by 
  -- Skip the proof since it is not required
  sorry

end interest_equality_l691_691823


namespace maximize_root_product_l691_691709

theorem maximize_root_product :
  (∃ k : ℝ, ∀ x : ℝ, 6 * x^2 - 5 * x + k = 0 ∧ (25 - 24 * k ≥ 0)) →
  ∃ k : ℝ, k = 25 / 24 :=
by
  sorry

end maximize_root_product_l691_691709


namespace christina_total_weekly_distance_l691_691367

-- Definitions based on conditions
def daily_distance_to_school : ℕ := 7
def daily_round_trip : ℕ := 2 * daily_distance_to_school
def days_per_week : ℕ := 5
def extra_trip_distance_one_way : ℕ := 2
def extra_trip_round_trip : ℕ := 2 * extra_trip_distance_one_way

-- Theorem statement
theorem christina_total_weekly_distance :
  let weekly_distance := daily_round_trip * days_per_week in
  let final_week_distance := weekly_distance + extra_trip_round_trip in
  final_week_distance = 74 := by
sorry

end christina_total_weekly_distance_l691_691367


namespace evaluate_expression_l691_691390

def improper_fraction (n : Int) (a : Int) (b : Int) : Rat :=
  n + (a : Rat) / b

def expression (x : Rat) : Rat :=
  (x * 1.65 - x + (7 / 20) * x) * 47.5 * 0.8 * 2.5

theorem evaluate_expression : 
  expression (improper_fraction 20 94 95) = 1994 := 
by 
  sorry

end evaluate_expression_l691_691390


namespace mark_cells_19x19_unique_subgrids_l691_691107

open Function Set

theorem mark_cells_19x19_unique_subgrids :
  ∃ (marking : ℕ → ℕ → Prop),
  (∀ (i j : ℕ), i < 19 → j < 19 → (marking i j = false ∨ marking i j = true)) ∧
  ∀ (i j : ℕ), i + 9 < 19 → j + 9 < 19 →
  ∃ k l, k + 9 < 19 ∧ l + 9 < 19 ∧ (k ≠ i ∨ l ≠ j) ∧ 
  (∑ x in range 10, ∑ y in range 10, if marking (i + x) (j + y) then 1 else 0 ≠
   ∑ x in range 10, ∑ y in range 10, if marking (k + x) (l + y) then 1 else 0) :=
begin
  sorry
end

end mark_cells_19x19_unique_subgrids_l691_691107


namespace angle_PQT_is_30_degrees_l691_691888

-- Define the hexagon and its properties
def regular_hexagon (P Q R S T U : Type) :=
  ∀ A B C : Type, A ≠ B → B ≠ C → C ≠ A 

-- Define the isosceles triangle within the hexagon
def isosceles_triangle {α : Type} (A B C : α) [metric_space α] [inner_product_space ℝ α] :=
  dist A B = dist A C

noncomputable def measure_of_angle_PQT (P Q R S T U : Type) [metric_space Type] [inner_product_space ℝ Type] (hhex : regular_hexagon P Q R S T U) : ℝ :=
  30 

theorem angle_PQT_is_30_degrees
  {P Q R S T U : Type} [metric_space Type] [inner_product_space ℝ Type]
  (hhex : regular_hexagon P Q R S T U) :
  measure_of_angle_PQT P Q R S T U hhex = 30 :=
by
  sorry

end angle_PQT_is_30_degrees_l691_691888


namespace cos_inequality_l691_691572

theorem cos_inequality (x y : ℝ) : cos (x^2) + cos (y^2) - cos (x * y) < 3 := 
by sorry

end cos_inequality_l691_691572


namespace same_number_of_digits_l691_691562

theorem same_number_of_digits (n : ℕ) : 
  let a := 1974 ^ n in 
  let b := 1974 ^ n + 2 ^ n in 
  (nat.digits 10 a).length = (nat.digits 10 b).length :=
sorry

end same_number_of_digits_l691_691562


namespace diff_of_squares_525_475_l691_691284

theorem diff_of_squares_525_475 : 525^2 - 475^2 = 50000 := by
  sorry

end diff_of_squares_525_475_l691_691284


namespace classify_triangles_by_angles_l691_691222

-- Define the basic types and properties for triangles and their angle classifications
def acute_triangle (α β γ : ℝ) : Prop :=
  α < 90 ∧ β < 90 ∧ γ < 90

def right_triangle (α β γ : ℝ) : Prop :=
  α = 90 ∨ β = 90 ∨ γ = 90

def obtuse_triangle (α β γ : ℝ) : Prop :=
  α > 90 ∨ β > 90 ∨ γ > 90

-- Problem: Classify triangles by angles and prove that the correct classification is as per option A
theorem classify_triangles_by_angles :
  (∀ (α β γ : ℝ), acute_triangle α β γ ∨ right_triangle α β γ ∨ obtuse_triangle α β γ) :=
sorry

end classify_triangles_by_angles_l691_691222


namespace range_of_a_l691_691871

def proposition_p (a : ℝ) : Prop :=
  ∀ m : ℝ, -1 ≤ m ∧ m ≤ 1 → a^2 - 5 * a - 3 ≥ real.sqrt (m^2 + 8)

def domain_of_f (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 4 * x + a^2 > 0

theorem range_of_a (a : ℝ) :
  (proposition_p a ∨ domain_of_f a) ∧ ¬(proposition_p a ∧ domain_of_f a) →
  a ∈ set.Icc (-2 : ℝ) (-1 : ℝ) ∪ set.Ioo (2 : ℝ) (6 : ℝ) :=
by
  sorry

end range_of_a_l691_691871


namespace number_leaves_remainder_five_l691_691219

theorem number_leaves_remainder_five (k : ℕ) (n : ℕ) (least_num : ℕ) 
  (h₁ : least_num = 540)
  (h₂ : ∀ m, m % 12 = 5 → m ≥ least_num)
  (h₃ : n = 107) 
  : 540 % 107 = 5 :=
by sorry

end number_leaves_remainder_five_l691_691219


namespace fib_100_mod_7_l691_691845

-- Define the Fibonacci sequence modulo 7
def fib : ℕ → ℤ
| 0     := 1
| 1     := 1
| n + 2 := (fib (n + 1) + fib n) % 7

-- Prove the remainder when the 100th term of the Fibonacci sequence is divided by 7
theorem fib_100_mod_7 : fib 100 % 7 = 3 := 
by sorry

end fib_100_mod_7_l691_691845


namespace range_of_a_zeros_of_g_l691_691782

-- Definitions for the original functions f and g and their corresponding conditions
noncomputable def f (x a : ℝ) : ℝ := x * Real.log x - (a / 2) * x^2

noncomputable def g (x x2 a : ℝ) : ℝ := f x a - (x2 / 2)

-- Proving the range of a
theorem range_of_a (h : ∃ x1 x2 : ℝ, x1 < x2 ∧ x1 * Real.log x1 - (a / 2) * x1^2 = 0 ∧ x2 * Real.log x2 - (a / 2) * x2^2 = 0) :
  0 < a ∧ a < 1 := 
sorry

-- Proving the number of zeros of g based on the value of a
theorem zeros_of_g (a : ℝ) (x1 x2 : ℝ) (h : x1 < x2 ∧ x1 * Real.log x1 - (a / 2) * x1^2 = 0 ∧ x2 * Real.log x2 - (a / 2) * x2^2 = 0) :
  (0 < a ∧ a < 3 / Real.exp 2 → ∃ x3 x4, x3 ≠ x4 ∧ g x3 x2 a = 0 ∧ g x4 x2 a = 0) ∧
  (a = 3 / Real.exp 2 → ∃ x3, g x3 x2 a = 0) ∧
  (3 / Real.exp 2 < a ∧ a < 1 → ∀ x, g x x2 a ≠ 0) :=
sorry

end range_of_a_zeros_of_g_l691_691782


namespace tanks_fill_l691_691190

theorem tanks_fill
  (c : ℕ) -- capacity of each tank
  (h1 : 300 < c) -- first tank is filled with 300 liters, thus c > 300
  (h2 : 450 < c) -- second tank is filled with 450 liters, thus c > 450
  (h3 : (45 : ℝ) / 100 = (450 : ℝ) / c) -- second tank is 45% filled, thus 0.45 * c = 450
  (h4 : 300 + 450 < 2 * c) -- the two tanks have the same capacity, thus they must have enough capacity to be filled more than 750 liters
  : c - 300 + (c - 450) = 1250 :=
sorry

end tanks_fill_l691_691190


namespace min_sum_of_arithmetic_seq_l691_691422

variables {a_n : ℕ → ℝ} (S_n : ℕ → ℝ) (d : ℝ)

-- Given Conditions
def arithmetic_sequence (a_n : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a_n (n+1) = a_n n + d

def sum_of_first_n_terms (a_n : ℕ → ℝ) (n : ℕ) :=
  (n : ℝ) * (a_n 1) + (n * (n - 1) * d / 2)

-- The specific conditions from the problem
def conditions :=
  a_n 1 = -9 ∧ (a_n 1 + 7 * d) + (a_n 1 + d) = -2

-- The goal statement which needs to be proven
theorem min_sum_of_arithmetic_seq : 
  conditions a_n d → 
  (∃ n : ℕ, S_n n = (n:ℝ) * a_n 1 + (n * (n-1) * d / 2) ∧ (S_n n) = n^2 - 10 * n ∧ n = 5) :=
begin
  sorry
end

end min_sum_of_arithmetic_seq_l691_691422


namespace half_abs_diff_squares_l691_691245

theorem half_abs_diff_squares : (1 / 2) * |20^2 - 15^2| = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691245


namespace volume_of_box_l691_691351

noncomputable def volume_expression (y : ℝ) : ℝ :=
  (15 - 2 * y) * (12 - 2 * y) * y

theorem volume_of_box (y : ℝ) :
  volume_expression y = 4 * y^3 - 54 * y^2 + 180 * y :=
by
  sorry

end volume_of_box_l691_691351


namespace total_plates_used_l691_691939

-- Definitions from the conditions
def number_of_people := 6
def meals_per_day_per_person := 3
def plates_per_meal_per_person := 2
def number_of_days := 4

-- Statement of the theorem
theorem total_plates_used : number_of_people * meals_per_day_per_person * plates_per_meal_per_person * number_of_days = 144 := 
by
  sorry

end total_plates_used_l691_691939


namespace min_pipes_needed_l691_691997

theorem min_pipes_needed 
  (h : ℝ) 
  (V_8inch : ℝ) 
  (V_3inch : ℝ) 
  (radius_8inch : ℝ = 4) 
  (radius_3inch : ℝ = 1.5) 
  (volume_8inch : V_8inch = π * (radius_8inch ^ 2) * h) 
  (volume_3inch : V_3inch = π * (radius_3inch ^ 2) * h) : 
  ∃ (n : ℕ), V_8inch ≤ n * V_3inch ∧ n = 8 := 
sorry

end min_pipes_needed_l691_691997


namespace largest_T_value_l691_691118

theorem largest_T_value : ∀ (a b c : ℝ), (1 ≤ a) ∧ (a ≤ 2) ∧ (1 ≤ b) ∧ (b ≤ 2) ∧ (1 ≤ c) ∧ (c ≤ 2) → 
  (let T := (a - b) ^ 2018 + (b - c) ^ 2018 + (c - a) ^ 2018 in T ≤ 2) := 
by 
  intros a b c h 
  let T := (a - b) ^ 2018 + (b - c) ^ 2018 + (c - a) ^ 2018 
  sorry

end largest_T_value_l691_691118


namespace area_less_than_circumference_l691_691992

-- Conditions
def sum_of_dice (d1 d2 : ℕ) : ℕ := d1 + d2

def probability (event : set (ℕ × ℕ)) : ℚ :=
  (event.card : ℚ) / 36

-- The formalized problem statement
theorem area_less_than_circumference (d1 d2 : ℕ) (h1 : 1 ≤ d1) (h2 : d1 ≤ 6) (h3 : 1 ≤ d2) (h4 : d2 ≤ 6) :
  probability {pair | let d := sum_of_dice pair.1 pair.2 in d < 4} = 1 / 12 :=
sorry

end area_less_than_circumference_l691_691992


namespace remainder_sum_first_150_div_10000_l691_691952

theorem remainder_sum_first_150_div_10000 :
  (∑ i in Finset.range 151, i) % 10000 = 1325 :=
by
  sorry

end remainder_sum_first_150_div_10000_l691_691952


namespace mod_calculation_l691_691696

theorem mod_calculation : 
  (nat.mod_exp (nat.gcd_nat 1 29) 27 29) + (nat.mod_exp (nat.gcd_nat 1 29) (2 * 27) 29) + (nat.mod_exp 7 27 29) % 29 = 13 := sorry

end mod_calculation_l691_691696


namespace total_receipts_correct_l691_691344

def cost_adult_ticket : ℝ := 5.50
def cost_children_ticket : ℝ := 2.50
def number_of_adults : ℕ := 152
def number_of_children : ℕ := number_of_adults / 2

def receipts_from_adults : ℝ := number_of_adults * cost_adult_ticket
def receipts_from_children : ℝ := number_of_children * cost_children_ticket
def total_receipts : ℝ := receipts_from_adults + receipts_from_children

theorem total_receipts_correct : total_receipts = 1026 := 
by
  -- Proof omitted, proof needed to validate theorem statement.
  sorry

end total_receipts_correct_l691_691344


namespace floor_log2_sum_correct_l691_691737

open Real

noncomputable def floor_log2_sum : ℕ :=
  (Finset.range 65).sum (λ n, Int.floor (log 2 n.toReal))

theorem floor_log2_sum_correct : floor_log2_sum = 264 := 
by {
  sorry
}

end floor_log2_sum_correct_l691_691737


namespace rational_numbers_in_interval_l691_691726

noncomputable def representation_x (n : ℕ) (a : Fin n → ℕ) : ℚ :=
  ∑ i in Finset.range n, (a ⟨i, by linarith⟩ : ℚ) / (∏ j in Finset.Icc (i + 1) n, (a ⟨j, by linarith⟩ : ℚ))

theorem rational_numbers_in_interval (x : ℚ) (n : ℕ) (a : Fin n → ℕ)
    (h₀ : a ⟨0, by linarith⟩ = 1) (h₁ : ∀ i, a ⟨i, by linarith⟩ < a ⟨i + 1, by linarith⟩ ∧ 0 < a ⟨i, by linarith⟩) :
  x = representation_x n a → x ∈ Set.Icc ℚ 0 1 := by
  sorry

end rational_numbers_in_interval_l691_691726


namespace trigonometric_identity_l691_691574

open Real

theorem trigonometric_identity :
  sin (72 * pi / 180) * cos (12 * pi / 180) - cos (72 * pi / 180) * sin (12 * pi / 180) = sqrt 3 / 2 :=
by
  sorry

end trigonometric_identity_l691_691574


namespace quadratic_function_explicit_formula_l691_691023

open Real

variable (f : ℝ → ℝ)

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

noncomputable def is_maximum_on (f : ℝ → ℝ) (M : ℝ) (I : set ℝ) : Prop :=
  ∀ x ∈ I, f x ≤ M

def is_root (f : ℝ → ℝ) (r : ℝ) : Prop :=
  f r = 0

theorem quadratic_function_explicit_formula
  (h_quad : is_quadratic f)
  (h_roots : is_root f 0 ∧ is_root f 4)
  (h_max : is_maximum_on f 12 (Icc (-1) 5)) :
  f = λ x, -3*(x - 2)^2 + 12 :=
sorry

end quadratic_function_explicit_formula_l691_691023


namespace complex_arg_solution_l691_691552

noncomputable def complex_arg_problem (z1 z2 : ℂ) : Prop :=
  complex.abs z1 = 3 ∧ complex.abs z2 = 5 ∧ complex.abs (z1 + z2) = 7 → complex.arg ((z2 / z1) ^ 3) = Real.pi

-- Lean statement for the problem
theorem complex_arg_solution (z1 z2 : ℂ) (h1: complex.abs z1 = 3) (h2: complex.abs z2 = 5) (h3: complex.abs (z1 + z2) = 7) : 
  complex.arg ((z2 / z1) ^ 3) = Real.pi :=
sorry

end complex_arg_solution_l691_691552


namespace walk_impossible_100x101_walk_possible_100x100_l691_691149

-- Definitions for part (a)
def board_a (m n : ℕ) := matrix (fin m) (fin n) bool
def lower_left_free_a (b : board_a 100 101) := b ! ⟨0, by linarith⟩ ! ⟨0, by linarith⟩ = false
def upper_right_free_a (b : board_a 100 101) := b ! ⟨99, by linarith⟩ ! ⟨100, by linarith⟩ = false
def domino (b : board_a 100 101) (i j : fin 100) (k l : fin 101) := b ! i ! j = true ∧ b ! k ! l = true

noncomputable def not_always_possible_walk_a : Prop :=
  ¬ ∀ (b : board_a 100 101), lower_left_free_a b ∧ upper_right_free_a b ∧
  (∀ i j k l, domino b i j k l → abs (i.value - k.value) + abs (j.value - l.value) = 1) →
  ∃ path : list (fin 100 × fin 101), path ≠ []

-- Definitions for part (b)
def board_b (m n : ℕ) := matrix (fin m) (fin n) bool
def lower_left_free_b (b : board_b 100 100) := b ! ⟨0, by linarith⟩ ! ⟨0, by linarith⟩ = false 
def upper_right_free_b (b : board_b 100 100) := b ! ⟨99, by linarith⟩ ! ⟨99, by linarith⟩ = false
def domino (b : board_b 100 100) (i j : fin 100) (k l : fin 100) := b ! i ! j = true ∧ b ! k ! l = true

noncomputable def always_possible_walk_b : Prop :=
  ∀ (b : board_b 100 100), lower_left_free_b b ∧ upper_right_free_b b ∧
  (∀ i j k l, domino b i j k l → abs (i.value - k.value) + abs (j.value - l.value) = 1) →
  ∃ path : list (fin 100 × fin 100), path ≠ []

-- Statements for Lean
theorem walk_impossible_100x101 : not_always_possible_walk_a := sorry
theorem walk_possible_100x100 : always_possible_walk_b := sorry

end walk_impossible_100x101_walk_possible_100x100_l691_691149


namespace exists_divisor_in_range_l691_691126

variable (n k : Nat)

def odd_positive_integer (m : Nat) : Prop := (m % 2 = 1) ∧ (0 < m)
def positive_integer (m : Nat) : Prop := 0 < m
def number_of_divisors_leq_k_is_odd (n k : Nat) : Prop :=
  Nat.Odd ((Finset.filter (fun d => d≤k) (Finset.Icc 1 (2 * n))).card)

theorem exists_divisor_in_range (n k : Nat) (hn : odd_positive_integer n)
  (hk : positive_integer k) (hd : number_of_divisors_leq_k_is_odd n k) :
  ∃ d, d ∈ (Finset.Icc 1 (2 * n)) ∧ k < d ∧ d ≤ 2 * k :=
sorry

end exists_divisor_in_range_l691_691126


namespace largest_divisor_36_l691_691496

theorem largest_divisor_36 (n : ℕ) (h : n > 0) (h_div : 36 ∣ n^3) : 6 ∣ n := 
sorry

end largest_divisor_36_l691_691496


namespace number_of_pupils_in_class_l691_691330

-- Defining the conditions
def wrongMark : ℕ := 79
def correctMark : ℕ := 45
def averageIncreasedByHalf : ℕ := 2  -- Condition representing average increased by half

-- The goal is to prove the number of pupils is 68
theorem number_of_pupils_in_class (n S : ℕ) (h1 : wrongMark = 79) (h2 : correctMark = 45)
(h3 : averageIncreasedByHalf = 2) 
(h4 : S + (wrongMark - correctMark) = (3 / 2) * S) :
  n = 68 :=
  sorry

end number_of_pupils_in_class_l691_691330


namespace ratio_of_white_to_yellow_balls_l691_691339

theorem ratio_of_white_to_yellow_balls 
  (original_total : ℕ)
  (whites_initial : ℕ)
  (yellows_initial : ℕ)
  (extra_yellows : ℕ)
  (whites_after : ℕ)
  (yellows_after : ℕ)
  (gcd_val : ℕ)
  (whites_ratio : ℕ)
  (yellows_ratio : ℕ) :
  original_total = 288 ∧ extra_yellows = 90 ∧ gcd 144 234 = 18 ∧
  whites_initial = original_total / 2 ∧ yellows_initial = original_total / 2 ∧
  whites_after = whites_initial ∧ yellows_after = yellows_initial + extra_yellows ∧
  whites_ratio = whites_after / gcd_val ∧ yellows_ratio = yellows_after / gcd_val →
  whites_ratio = 8 ∧ yellows_ratio = 13 :=
by
  intros,
  sorry

end ratio_of_white_to_yellow_balls_l691_691339


namespace orthogonal_matrix_sum_of_squares_l691_691491

open BigOperators

-- Define the 3x3 matrix and the properties of the orthogonal matrix
def B (x y z p q r s t u : ℝ) := ![![x, y, z], ![p, q, r], ![s, t, u]]

variables {x y z p q r s t u : ℝ}

-- State that B^T = B^(-1) with the given x
theorem orthogonal_matrix_sum_of_squares 
  (h1 : x = 1 / 3) 
  (h2 : (B x y z p q r s t u)ᵀ = (B x y z p q r s t u)⁻¹) 
  : x^2 + y^2 + z^2 + p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 3 := 
sorry

end orthogonal_matrix_sum_of_squares_l691_691491


namespace num_tables_l691_691667

/-- Given conditions related to tables, stools, and benches, we want to prove the number of tables -/
theorem num_tables 
  (t s b : ℕ) 
  (h1 : s = 8 * t)
  (h2 : b = 2 * t)
  (h3 : 3 * s + 6 * b + 4 * t = 816) : 
  t = 20 := 
sorry

end num_tables_l691_691667


namespace rectangle_original_area_l691_691934

theorem rectangle_original_area (L L' A : ℝ) 
  (h1: A = L * 10)
  (h2: L' * 10 = (4 / 3) * A)
  (h3: 2 * L' + 2 * 10 = 60) : A = 150 :=
by 
  sorry

end rectangle_original_area_l691_691934


namespace polynomial_no_integer_roots_l691_691177

-- Lean 4 statement for the math proof problem
theorem polynomial_no_integer_roots
  (n : ℕ)
  (a : ℕ → ℤ)
  (P : ℤ → ℤ)
  (hP : ∀ x, P x = ∑ i in finset.range (n+1), a i * x ^ (n-i))
  (hP0_odd : P 0 % 2 = 1)
  (hP1_odd : P 1 % 2 = 1) :
  ¬ ∃ x : ℤ, P x = 0 :=
sorry

end polynomial_no_integer_roots_l691_691177


namespace minimum_value_of_y_at_l691_691960

noncomputable def y (x : ℝ) : ℝ := x * 2^x

theorem minimum_value_of_y_at :
  ∃ x : ℝ, (∀ x' : ℝ, y x ≤ y x') ∧ x = -1 / Real.log 2 :=
by
  sorry

end minimum_value_of_y_at_l691_691960


namespace work_rate_together_l691_691655

theorem work_rate_together :
  let work_rate_A := (1 : ℚ) / 12
  let work_rate_B := (1 : ℚ) / 6
  let work_rate_C := (1 : ℚ) / 18
  work_rate_A + work_rate_B + work_rate_C = 11 / 36 := by
  let work_rate_A := (1 : ℚ) / 12
  let work_rate_B := (1 : ℚ) / 6
  let work_rate_C := (1 : ℚ) / 18
  calc
    work_rate_A + work_rate_B + work_rate_C = (1 : ℚ) / 12 + (1 : ℚ) / 6 + (1 : ℚ) / 18 : rfl
    ... = 3 / 36 + 6 / 36 + 2 / 36 : by
      congr; {field_simp [work_rate_A, work_rate_B, work_rate_C]}; apply_rat_cast_eq_field
    ... = 11 / 36 : by sorry

end work_rate_together_l691_691655


namespace third_term_binomial_expansion_l691_691821

theorem third_term_binomial_expansion (x : ℝ) : 
  (∑ k in finset.range (7), binomial 6 k * x^(6 - k) * 2^k) = 64 → 
  binomial 6 2 * x^4 * 2^2 = 60 * x^4 :=
by {
  intros h,
  sorry
}

end third_term_binomial_expansion_l691_691821


namespace magnitude_of_parallel_vector_l691_691475

theorem magnitude_of_parallel_vector {x : ℝ} 
  (h_parallel : 2 / x = -1 / 3) : 
  (Real.sqrt (x^2 + 3^2)) = 3 * Real.sqrt 5 := 
sorry

end magnitude_of_parallel_vector_l691_691475


namespace firetruck_reachable_area_l691_691510

-- Definitions based on given conditions
def truck_speed_on_road : ℝ := 60  -- miles per hour on the road
def truck_speed_on_sand : ℝ := 10  -- miles per hour on the sand
def time_minutes : ℝ := 8
def time_hours : ℝ := time_minutes / 60

-- Distance reachable in 8 minutes
def distance_on_road : ℝ := truck_speed_on_road * time_hours
def distance_on_sand : ℝ := truck_speed_on_sand * time_hours

-- Total area calculation to be proven
def area_reachable : ℝ := 267.12  -- This should match with (6678 / 25)

-- Main theorem
theorem firetruck_reachable_area : area_reachable = 267.12 :=
by
  sorry

end firetruck_reachable_area_l691_691510


namespace seq_sum_result_l691_691521

variable {a_n : ℕ → ℝ} -- definition for the arithmetic sequence

-- defining the arithmetic sequence property
def arithmetic_seq (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n m : ℕ, a (n + 1) = a n + d

-- assumption: sum of specific terms in the sequence
def seq_sum_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 4 + a 9 + a 11 = 32

-- The theorem to prove
theorem seq_sum_result (a : ℕ → ℝ)
  [ar_seq : arithmetic_seq a]
  (sum_cond : seq_sum_condition a) :
  a 6 + a 7 = 16 :=
sorry

end seq_sum_result_l691_691521


namespace train_crossing_time_l691_691999

theorem train_crossing_time
  (length_train : ℕ)
  (speed_train_kmph : ℕ)
  (total_length : ℕ)
  (htotal_length : total_length = 225)
  (hlength_train : length_train = 150)
  (hspeed_train_kmph : speed_train_kmph = 45) : 
  (total_length / (speed_train_kmph * 1000 / 3600)) = 18 := by 
  sorry

end train_crossing_time_l691_691999


namespace arrange_squares_l691_691129

theorem arrange_squares (n : ℕ) (h: n ≥ 5) : 
  ∃ (arrangement : list (ℕ × ℕ × ℕ × ℕ)), 
    (∀ (sq ∈ arrangement), sq.1 = sq.2 ∧ sq.3 = sq.4 ∧ sq.1 < sq.2 < ... < sq.n) ∧ 
    (∀ (sq1 sq2 ∈ arrangement), sq1 ≠ sq2 → 
       (sq1.1 = sq2.3 ∧ sq1.2 = sq2.4) ∨ 
       (sq1.3 = sq2.1 ∧ sq1.4 = sq2.2)) := 
sorry

end arrange_squares_l691_691129


namespace Christina_weekly_distance_l691_691370

/-- 
Prove that Christina covered 74 kilometers that week given the following conditions:
1. Christina walks 7km to school every day from Monday to Friday.
2. She returns home covering the same distance each day.
3. Last Friday, she had to pass by her friend, which is another 2km away from the school in the opposite direction from home.
-/
theorem Christina_weekly_distance : 
  let distance_to_school := 7
  let days_school := 5
  let extra_distance_Friday := 2
  let daily_distance := 2 * distance_to_school
  let total_distance_from_Monday_to_Thursday := 4 * daily_distance
  let distance_on_Friday := daily_distance + 2 * extra_distance_Friday
  total_distance_from_Monday_to_Thursday + distance_on_Friday = 74 := 
by
  sorry

end Christina_weekly_distance_l691_691370


namespace worker_y_defective_rate_l691_691713

noncomputable def y_f : ℚ := 0.1666666666666668
noncomputable def d_x : ℚ := 0.005 -- converting percentage to decimal
noncomputable def d_total : ℚ := 0.0055 -- converting percentage to decimal

theorem worker_y_defective_rate :
  ∃ d_y : ℚ, d_y = 0.008 ∧ d_total = ((1 - y_f) * d_x + y_f * d_y) :=
by
  sorry

end worker_y_defective_rate_l691_691713


namespace cuboid_vertices_on_sphere_surface_area_l691_691746

-- Defining the cuboid dimensions and surface area result
def cuboid_length (a : ℝ) := 2 * a
def cuboid_width (a : ℝ) := a
def cuboid_height (a : ℝ) := a

def body_diagonal (a : ℝ) := Real.sqrt ((cuboid_length a) ^ 2 + (cuboid_width a) ^ 2 + (cuboid_height a) ^ 2)
def sphere_diameter (a : ℝ) := body_diagonal a
def sphere_radius (a : ℝ) := sphere_diameter a / 2

-- Defining the expected surface area of the sphere
def expected_surface_area (a : ℝ) := 4 * Real.pi * (sphere_radius a) ^ 2

-- The problem statement to prove
theorem cuboid_vertices_on_sphere_surface_area (a : ℝ) : 
  expected_surface_area a = 6 * Real.pi * a ^ 2 := by
  sorry

end cuboid_vertices_on_sphere_surface_area_l691_691746


namespace catalytic_divisibility_l691_691550

noncomputable def catalan (k : ℕ) : ℚ :=
  1 / (k + 1) * (Nat.choose (2 * k) k)

theorem catalytic_divisibility (p : ℕ) [Fact p.prime] :
  ∑ n in Finset.range p \ {0} | (\sum k in Finset.range (p - 1), catalan k * n^k) = p / 2 :=
by
  -- Proof omitted
  sorry

end catalytic_divisibility_l691_691550


namespace multiples_of_six_units_digit_six_l691_691796

theorem multiples_of_six_units_digit_six (n : ℕ) : 
  (number_of_six_less_150 n) = 25 
where 
  number_of_six_less_150 (n : ℕ) := 
    ∃ M : ℕ, M * 6 = n ∧ n < 150 
:= 
  sorry

end multiples_of_six_units_digit_six_l691_691796


namespace part1_part2_l691_691125

variable {f : ℝ → ℝ}

-- Condition 1: f is an odd function
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Condition 2: ∀ a b ∈ ℝ, (a + b ≠ 0) → (f(a) + f(b))/(a + b) > 0
def positiveQuotient (f : ℝ → ℝ) : Prop :=
  ∀ a b, a + b ≠ 0 → (f a + f b) / (a + b) > 0

-- Sub-problem (1): For any a, b ∈ ℝ, a > b ⟹ f(a) > f(b)
theorem part1 (h_odd : isOddFunction f) (h_posQuot : positiveQuotient f) (a b : ℝ) (h : a > b) : f a > f b :=
  sorry

-- Sub-problem (2): If f(9^x - 2 * 3^x) + f(2 * 9^x - k) > 0 for any x ∈ [0, ∞), then k < 1
theorem part2 (h_odd : isOddFunction f) (h_posQuot : positiveQuotient f) :
  (∀ x : ℝ, 0 ≤ x → f (9^x - 2 * 3^x) + f (2 * 9^x - k) > 0) → k < 1 :=
  sorry

end part1_part2_l691_691125


namespace point_on_inverse_proportion_l691_691768

theorem point_on_inverse_proportion :
  ∀ (k x y : ℝ), 
    (∀ (x y: ℝ), (x = -2 ∧ y = 6) → y = k / x) →
    k = -12 →
    y = k / x →
    (x = 1 ∧ y = -12) :=
by
  sorry

end point_on_inverse_proportion_l691_691768


namespace find_p_l691_691018

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables (a b : V) (p : ℝ)
variables (A B C D : V)

-- Conditions
noncomputable def AB : V := 2 • a + p • b
noncomputable def BC : V := a + b
noncomputable def CD : V := a - 2 • b
noncomputable def BD : V := BC + CD

-- Collinearity implies AB is scalar multiple of BD
theorem find_p (non_collinear : ¬collinear ℝ ({a, b} : set V))
               (AB_def : AB = 2 • a + p • b)
               (BC_def : BC = a + b)
               (CD_def : CD = a - 2 • b)
               (collinear_points : collinear ℝ ({A, B, D} : set V)) :
  p = -1 := sorry

end find_p_l691_691018


namespace pizza_topping_combinations_l691_691571

theorem pizza_topping_combinations :
  (Nat.choose 7 3) = 35 :=
sorry

end pizza_topping_combinations_l691_691571


namespace white_balls_count_l691_691657

theorem white_balls_count (total_balls green_balls yellow_balls red_balls purple_balls : ℕ)
  (h_total : total_balls = 100)
  (h_green : green_balls = 20)
  (h_yellow : yellow_balls = 10)
  (h_red : red_balls = 17)
  (h_purple : purple_balls = 3)
  (h_prob : (total_balls - (red_balls + purple_balls)) / total_balls = 0.8) :
  (total_balls - (green_balls + yellow_balls + red_balls + purple_balls)) = 50 :=
by
  sorry

end white_balls_count_l691_691657


namespace angle_negative_225_in_second_quadrant_l691_691956

def inSecondQuadrant (angle : Int) : Prop :=
  angle % 360 > -270 ∧ angle % 360 <= -180

theorem angle_negative_225_in_second_quadrant :
  inSecondQuadrant (-225) :=
by
  sorry

end angle_negative_225_in_second_quadrant_l691_691956


namespace modified_pyramid_volume_l691_691331

theorem modified_pyramid_volume (s h : ℝ) (V : ℝ) (hV : V = (1/3) * s^2 * h) (hV_value : V = 60) : 
  let new_V := (1/3) * (3 * s)^2 * (2 * h) in new_V = 1080 := 
by 
  sorry

end modified_pyramid_volume_l691_691331


namespace num_multiples_divisible_by_12_15_18_between_2000_3000_l691_691049

theorem num_multiples_divisible_by_12_15_18_between_2000_3000 :
  (∃ (n : ℕ), n = 5 ∧ ∀ (m : ℕ), (2000 ≤ m ∧ m ≤ 3000) → (m % 12 = 0) → (m % 15 = 0) → (m % 18 = 0) → (↑n = (between ?-1 and ?-2 -> {
    sorry -- This requires computing the number of multiples of LCM in the interval
  })
  
  ) :=
sorry

end num_multiples_divisible_by_12_15_18_between_2000_3000_l691_691049


namespace maximal_dwarfs_in_hats_l691_691651

def circle_of_dwarfs (n: ℕ) := (n = 99) ∧
  (∀ (mask: ℕ → Prop),
    (∀ i, mask i → mask (i + 1) % n = false) →
    (∀ i, mask i → mask (i + 49) % n = false) →
    ∃ m, (∀ i, mask i → mask (i + 3) % n = mask (i + 6) % n = false) →
          (m ≤ 33))

theorem maximal_dwarfs_in_hats : circle_of_dwarfs 99 :=
by {
  sorry
}

end maximal_dwarfs_in_hats_l691_691651


namespace perpendicular_bisector_eq_line_l_eq_l691_691042

section
variables {A B P : ℝ × ℝ}
-- Define points A, B, and P
def A : ℝ × ℝ := (8, -6)
def B : ℝ × ℝ := (2, 2)
def P : ℝ × ℝ := (2, -3)

-- Midpoint of A and B
def M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Slope of line AB
def slope_AB : ℝ := (A.2 - B.2) / (A.1 - B.1)

-- Slope of the perpendicular bisector
def slope_perpendicular_bisector : ℝ := -1 / slope_AB

-- Expected equations
def eq_perpendicular_bisector := ∀ x y : ℝ, 3 * x - 4 * y = 23
def eq_line_l := ∀ x y : ℝ, 4 * x + 3 * y = -1

-- Theorem: Equation of the perpendicular bisector of segment AB
theorem perpendicular_bisector_eq : eq_perpendicular_bisector :=
by sorry

-- Theorem: Equation of the line passing through P(2, -3) and parallel to line AB
theorem line_l_eq : eq_line_l :=
by sorry
end

end perpendicular_bisector_eq_line_l_eq_l691_691042


namespace no_positive_integers_exist_for_system_l691_691161

theorem no_positive_integers_exist_for_system :
  ¬ ∃ (m n k : ℕ) (hm : 0 < m) (hn : 0 < n) (hk : 0 < k), 
    ∀ (x y : ℝ), 
      ((x + 1) ^ 2 + y ^ 2 = (m : ℝ) ^ 2) ∧ 
      ((x - 1) ^ 2 + y ^ 2 = (n : ℝ) ^ 2) ∧ 
      (x ^ 2 + (y - real.sqrt 3) ^ 2 = (k : ℝ) ^ 2) := 
  sorry

end no_positive_integers_exist_for_system_l691_691161


namespace car_city_mileage_l691_691981

theorem car_city_mileage (h c t : ℝ) 
  (h_eq : h * t = 462)
  (c_eq : (h - 15) * t = 336) 
  (c_def : c = h - 15) : 
  c = 40 := 
by 
  sorry

end car_city_mileage_l691_691981


namespace function_inequality_l691_691494

theorem function_inequality
  (f : ℝ → ℝ)
  (h₁ : ∀ x, x > 0 → f x > 0)
  (h₂ : ∀ x y, x > 0 → y > 0 → f(x + f y) ≥ f(x + y) + f y) :
  ∀ x, x > 0 → f x > x :=
by
  sorry

end function_inequality_l691_691494


namespace equation_of_line_bisecting_chord_l691_691312

theorem equation_of_line_bisecting_chord
  (P : ℝ × ℝ) 
  (A B : ℝ × ℝ)
  (P_bisects_AB : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (P_on_ellipse : 3 * P.1^2 + 4 * P.2^2 = 24)
  (A_on_ellipse : 3 * A.1^2 + 4 * A.2^2 = 24)
  (B_on_ellipse : 3 * B.1^2 + 4 * B.2^2 = 24) :
  ∃ (a b c : ℝ), a * P.2 + b * P.1 + c = 0 ∧ a = 2 ∧ b = -3 ∧ c = 7 :=
by 
  sorry

end equation_of_line_bisecting_chord_l691_691312


namespace probability_of_prime_ball_l691_691145

-- Establishing the context of the problem
def balls : Finset ℕ := {1, 2, 3, 5, 6, 7, 8, 9, 10}
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

-- Stating the main theorem
theorem probability_of_prime_ball :
  (∑ b in balls.filter is_prime, (1 : ℚ)) / (∑ b in balls, (1 : ℚ)) = 4 / 9 := by
  sorry

end probability_of_prime_ball_l691_691145


namespace sum_inequality_l691_691615

theorem sum_inequality (n : ℕ) (a : ℕ → ℕ) 
  (distinct : ∀ i j, i ≠ j → a i ≠ a j) 
  (positive : ∀ i, 0 < a i) :
  (∑ k in finset.range n, (a k / (k+1)^2)) ≥ (∑ k in finset.range n, 1 / (k+1)) := 
sorry

end sum_inequality_l691_691615


namespace enclosed_area_eq_three_l691_691579

open IntervalReal

noncomputable def enclosed_area : ℝ :=
  3 * ∫ x in (0 : ℝ)..(Real.pi / 2), Real.cos x

theorem enclosed_area_eq_three :
  enclosed_area = 3 :=
by
  rw [enclosed_area]
  refine mul_eq_mul_right_iff.mpr (Or.inl _)
  simp
  norm_num
  sorry

end enclosed_area_eq_three_l691_691579


namespace monotonic_intervals_exist_x0_sqrt_sum_l691_691742

noncomputable def f (x a b : ℝ) : ℝ := x + a * Real.sin x + b * Real.log x

def f_prime (x a b : ℝ) : ℝ := 1 + a * Real.cos x + b / x

-- Part 1
theorem monotonic_intervals (a b : ℝ) 
  (ha : a = 0) (hb : b = -1) : 
  (∀ x, 1 < x → f_prime x a b > 0) 
  ∧ (∀ x, 0 < x ∧ x < 1 → f_prime x a b < 0) :=
by
  sorry

-- Part 2
theorem exist_x0 (a : ℝ) (b : ℝ)
  (ha: a = -1/2) (hb: b ≠ 0)
  (h: ∀ x, f_prime x a b > 0) : 
  ∃ x0, f x0 a b < -1 :=
by
  sorry

-- Part 3
theorem sqrt_sum (a b x1 x2 : ℝ)
  (h0: 0 < a ∧ a < 1) (h1 : b < 0) 
  (h2 : 0 < x1) (h3 : 0 < x2) 
  (h4: f x1 a b = f x2 a b) (h5: x1 ≠ x2) : 
  Real.sqrt x1 + Real.sqrt x2 > 2 * Real.sqrt (-b / (a + 1)) :=
by
  sorry

end monotonic_intervals_exist_x0_sqrt_sum_l691_691742


namespace final_mixture_percentages_l691_691977

theorem final_mixture_percentages (
    initial_volume : ℝ, initial_alcohol_percentage : ℝ, initial_methanol_percentage : ℝ,
    added_alcohol : ℝ, added_methanol : ℝ, added_water : ℝ) :
  initial_volume = 60 →
  initial_alcohol_percentage = 0.05 →
  initial_methanol_percentage = 0.10 →
  added_alcohol = 4.5 →
  added_methanol = 6.5 →
  added_water = 3 →
  let initial_water_percentage := 1 - initial_alcohol_percentage - initial_methanol_percentage,
      initial_alcohol := initial_alcohol_percentage * initial_volume,
      initial_methanol := initial_methanol_percentage * initial_volume,
      initial_water := initial_water_percentage * initial_volume,
      total_alcohol := initial_alcohol + added_alcohol,
      total_methanol := initial_methanol + added_methanol,
      total_water := initial_water + added_water,
      total_volume := total_alcohol + total_methanol + total_water,
      final_alcohol_percentage := (total_alcohol / total_volume) * 100,
      final_methanol_percentage := (total_methanol / total_volume) * 100,
      final_water_percentage := (total_water / total_volume) * 100
  in final_alcohol_percentage = 10.14 ∧
     final_methanol_percentage = 16.89 ∧
     final_water_percentage = 72.97 := by
  sorry

end final_mixture_percentages_l691_691977


namespace problem_q_value_l691_691043

theorem problem_q_value (p q : ℝ) (hpq : 1 < p ∧ p < q) 
  (h1 : 1 / p + 1 / q = 1) 
  (h2 : p * q = 8) : 
  q = 4 + 2 * real.sqrt 2 :=
by sorry

end problem_q_value_l691_691043


namespace length_of_each_train_l691_691613

theorem length_of_each_train (L : ℝ) (V : ℝ) (t : ℝ) (rel_speed : ℝ) (d : ℝ) :
  V = 80 ∧ t = 4.499640028797696 ∧ rel_speed = 2 * V ∧ d = rel_speed * (t / 3600) →
  L = d / 2 ∧ L * 1000 ≈ 99.992556 :=
by
  intros h
  cases h with hv h
  cases h with ht h
  cases h with hrs h
  cases h with hd h
  sorry

end length_of_each_train_l691_691613


namespace remaining_students_average_l691_691826

theorem remaining_students_average :
  ∀ (total_students : ℕ) (total_average : ℝ) (group1_students : ℕ) (group1_average : ℝ) (group2_students : ℕ) (group2_average : ℝ) (group3_students : ℕ) (group3_average : ℝ),
    total_students = 120 ∧ total_average = 84 ∧
    group1_students = 30 ∧ group1_average = 96 ∧
    group2_students = 24 ∧ group2_average = 75 ∧
    group3_students = 15 ∧ group3_average = 90 →
    ((120 * 84) - (30 * 96) - (24 * 75) - (15 * 90)) / (120 - 30 - 24 - 15) = 79.41 :=
begin
  intros total_students total_average group1_students group1_average group2_students group2_average group3_students group3_average,
  rintros ⟨h_total_students, h_total_average, h_group1_students, h_group1_average, h_group2_students, h_group2_average, h_group3_students, h_group3_average⟩,
  sorry
end

end remaining_students_average_l691_691826


namespace count_friendly_sets_with_max_8_l691_691969

def cards : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}

def is_friendly_set (S : Finset ℕ) : Prop :=
  S.card = 4 ∧ S.max' (by simp [Finset.nonempty_of_card_eq_four]) ≤ 8 ∧ S.sum id = 24

def friendly_sets : Finset (Finset ℕ) :=
  (cards.powerset.filter is_friendly_set)

theorem count_friendly_sets_with_max_8 : friendly_sets.card = 8 :=
sorry

end count_friendly_sets_with_max_8_l691_691969


namespace train_passing_time_l691_691806

-- Define the conditions: length of the train, speed of the train in km/hr, and the converted speed in m/s
def train_length : ℝ := 100
def speed_km_hr : ℝ := 72
def speed_m_s : ℝ := (speed_km_hr * 1000) / 3600

-- The theorem we need to prove
theorem train_passing_time : train_length / speed_m_s = 5 :=
by
  -- sorry is put here to indicate that the proof is omitted
  sorry

end train_passing_time_l691_691806


namespace sum_distances_red_to_A_eq_blue_to_B_l691_691649

-- Definitions based on conditions from step a)
variable {A B : ℝ}  -- A and B are points on the real line representing the endpoints of segment AB
variable (midpoint : ℝ) -- midpoint is the midpoint of segment AB
variable (points : Finset ℝ)  -- points is the set of 200 points on segment AB
variable (redPoints : Finset ℝ) -- redPoints is the set of red points
variable (bluePoints : Finset ℝ) -- bluePoints is the set of blue points

-- Conditions as assumptions
axiom symmetrically_arranged : ∀ p ∈ points, (midpoint - p) ∈ points ∧ (midpoint + p) ∈ points
axiom red_blue_partition : redPoints ∪ bluePoints = points ∧ redPoints ∩ bluePoints = ∅
axiom count_red_points : redPoints.card = 100
axiom count_blue_points : bluePoints.card = 100

-- Sum of distances from a set of points to a specific point
noncomputable def sum_distances (pts : Finset ℝ) (point : ℝ) : ℝ :=
  ∑ p in pts, |p - point|

-- The main theorem to be proven
theorem sum_distances_red_to_A_eq_blue_to_B :
  sum_distances redPoints A = sum_distances bluePoints B := 
sorry

end sum_distances_red_to_A_eq_blue_to_B_l691_691649


namespace base_any_number_l691_691495

open Nat

theorem base_any_number (n k : ℕ) (h1 : k ≥ 0) (h2 : (30 ^ k) ∣ 929260) (h3 : n ^ k - k ^ 3 = 1) : true :=
by
  sorry

end base_any_number_l691_691495


namespace yoyo_cost_theorem_l691_691878

def yoyo_cost : ℕ := 24

theorem yoyo_cost_theorem (whistle_cost total_cost : ℕ) 
  (h_whistle : whistle_cost = 14) 
  (h_total : total_cost = 38) : 
  yoyo_cost = total_cost - whistle_cost := by
  rw [h_whistle, h_total]
  exact rfl

#eval yoyo_cost_theorem 14 38 rfl rfl

end yoyo_cost_theorem_l691_691878


namespace committee_probability_l691_691314

theorem committee_probability :
  let total_ways := Nat.choose 30 6
  let all_boys_ways := Nat.choose 12 6
  let all_girls_ways := Nat.choose 18 6
  let complementary_prob := (all_boys_ways + all_girls_ways) / total_ways
  let desired_prob := 1 - complementary_prob
  desired_prob = (574287 : ℚ) / 593775 :=
by
  sorry

end committee_probability_l691_691314


namespace factory_employees_l691_691506

def num_employees (n12 n14 n17 : ℕ) : ℕ := n12 + n14 + n17

def total_cost (n12 n14 n17 : ℕ) : ℕ := 
    (200 * 12 * 8) + (40 * 14 * 8) + (n17 * 17 * 8)

theorem factory_employees (n17 : ℕ) 
    (h_cost : total_cost 200 40 n17 = 31840) : 
    num_employees 200 40 n17 = 300 := 
by 
    sorry

end factory_employees_l691_691506


namespace basketball_scores_possible_l691_691978

theorem basketball_scores_possible (baskets two three total_scores : ℕ) 
    (h_baskets: baskets = 7) 
    (h_two: two = 2) 
    (h_three: three = 3) 
    (h_min: 7 * two = 14) 
    (h_max: 7 * three = 21)
    (h_total_scores: total_scores = 8) : 
    ∃ (possible_scores : ℕ → Prop), 
    (∀ n, possible_scores n → 14 ≤ n ∧ n ≤ 21) ∧ 
    (∀ n, 14 ≤ n ∧ n ≤ 21 → possible_scores n) ∧ 
    (cardinality (set_of possible_scores) = total_scores) :=
begin
  sorry
end

end basketball_scores_possible_l691_691978


namespace ganesh_ram_together_l691_691412

theorem ganesh_ram_together (G R S : ℝ) (h1 : G + R + S = 1 / 16) (h2 : S = 1 / 48) : (G + R) = 1 / 24 :=
by
  sorry

end ganesh_ram_together_l691_691412


namespace complex_quadrant_l691_691816

theorem complex_quadrant
  (z : ℂ) (h : (z - 1) * (complex.I) = 1 + 2 * complex.I) :
  ∃ (a b : ℝ), z = a + b * complex.I ∧
    a > 0 ∧ b < 0 :=
sorry

end complex_quadrant_l691_691816


namespace relationship_of_inequalities_l691_691303

theorem relationship_of_inequalities (a b : ℝ) : 
  ¬ (∀ a b : ℝ, (a > b) → (a^2 > b^2)) ∧ 
  ¬ (∀ a b : ℝ, (a^2 > b^2) → (a > b)) := 
by 
  sorry

end relationship_of_inequalities_l691_691303


namespace product_of_nonreal_roots_l691_691400

def polynomial := polynomial ℂ

theorem product_of_nonreal_roots :
  let p := (X^4 - 4*X^3 + 6*X^2 - 4*X - 2047 : polynomial) in
  (∏ z in p.roots.to_finset.filter (λ x, x.im ≠ 0), z) = 257 :=
begin
  sorry 
end

end product_of_nonreal_roots_l691_691400


namespace cake_division_l691_691653

-- Given a triangle ΔABC
variables {A B C : Type}

-- Define the centroid M of triangle ΔABC
def centroid (ΔABC : A × B × C) : Type := sorry

-- Define the triangles formed by the cut through the centroid
def sibling_division (ΔABC : A × B × C) (M : centroid ΔABC) : (A × B × C) × (A × B × C) := sorry

-- Suppose the brother points to the centroid M
-- Prove the fractions of the cake the sister and brother receive
theorem cake_division (ΔABC : A × B × C) (M : centroid ΔABC) :
  ∃ (part_sister part_brother : set (A × B × C)),
    part_sister ∪ part_brother = ΔABC ∧
    part_sister ∩ part_brother = ∅ ∧
    (area part_sister / area ΔABC = 5 / 9) ∧
    (area part_brother / area ΔABC = 4 / 9) := sorry

end cake_division_l691_691653


namespace total_length_of_scale_l691_691335

theorem total_length_of_scale 
  (n : ℕ) (len_per_part : ℕ) 
  (h_n : n = 5) 
  (h_len_per_part : len_per_part = 25) :
  n * len_per_part = 125 :=
by
  sorry

end total_length_of_scale_l691_691335


namespace ln2_approximation_l691_691113

noncomputable def integral_ln2 := ∫ x in 1..2, 1 / x

-- The midpoint and trapezoidal approximation definitions
def midpoint_rule_approx (n : ℕ) (a b : ℝ) (f : ℝ → ℝ) : ℝ :=
  let h := (b - a) / n in
  h * ∑ i in Finset.range n, f (a + i * h)

def trapezoidal_rule_approx (n : ℕ) (a b : ℝ) (f : ℝ → ℝ) : ℝ :=
  let h := (b - a) / n in
  h * (f a / 2 + ∑ i in Finset.range (n - 1), f (a + (i + 1) * h) + f b / 2)

-- Problem statement obligations
theorem ln2_approximation :
  integral_ln2 = Real.log 2 ∧
  abs (midpoint_rule_approx 10 1 2 (λ x, 1 / x) - Real.log 2) ≤ 0.05 ∧
  abs (trapezoidal_rule_approx 10 1 2 (λ x, 1 / x) - Real.log 2) ≤ 0.0017 := sorry

end ln2_approximation_l691_691113


namespace min_true_statements_in_circle_l691_691647

theorem min_true_statements_in_circle (n : ℕ) (h : n = 16) (heights : Fin n → ℕ) 
  (H_distinct : Function.Injective heights) 
  (H_statements : ∀ i, heights (i + 1) > heights (i - 1) → false) : 
  ∃ (A : Fin n → Prop), (∃ i : Fin n, A i) ∧ (∃ j : Fin n, A j) := 
by
  sorry

end min_true_statements_in_circle_l691_691647


namespace max_value_of_PQ_l691_691592

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 12)
noncomputable def g (x : ℝ) := Real.sqrt 3 * Real.cos (2 * x - Real.pi / 12)

theorem max_value_of_PQ (t : ℝ) : abs (f t - g t) ≤ 2 :=
by sorry

end max_value_of_PQ_l691_691592


namespace AK_times_BM_eq_r_squared_l691_691892

-- Define the geometry of the problem
variables (K L M N A B O : Point)
variables (r : ℝ)
variables (circle : Circle(O, r))

-- Conditions: quadrilateral KLMN is inscribed and circumscribed
axiom KLMN_inscribed : CyclicQuadrilateral K L M N
axiom KLMN_circumscribed : TangentialQuadrilateral K L M N

-- Conditions: Points A and B are tangency points of the incircle
axiom A_tangency : TangencyPoint(circle, K, L, A)
axiom B_tangency : TangencyPoint(circle, M, N, B)

-- Define the proof goal
theorem AK_times_BM_eq_r_squared : AK * BM = r^2 := 
by {
  sorry -- Proof to be derived
}

end AK_times_BM_eq_r_squared_l691_691892


namespace tank_capacity_l691_691634

theorem tank_capacity (C : ℕ) 
  (leak_rate : C / 4 = C / 4)               -- Condition: Leak rate is C/4 litres per hour
  (inlet_rate : 6 * 60 = 360)                -- Condition: Inlet rate is 360 litres per hour
  (net_emptying_rate : C / 12 = (360 - C / 4))  -- Condition: Net emptying rate for 12 hours
  : C = 1080 := 
by 
  -- Conditions imply that C = 1080 
  sorry

end tank_capacity_l691_691634


namespace half_abs_diff_squares_l691_691234

theorem half_abs_diff_squares (a b : ℤ) (ha : a = 20) (hb : b = 15) : 
  (|a^2 - b^2| / 2) = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691234


namespace problem_l691_691563

noncomputable def f (x : ℝ) : ℝ := 3 / (x + 1)

theorem problem (x : ℝ) (h1 : 3 ≤ x) (h2 : x ≤ 5) :
  (∀ x₁ x₂, 3 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 5 → f x₁ > f x₁) ∧
  f 3 = 3/4 ∧
  f 5 = 1/2 :=
by sorry

end problem_l691_691563


namespace tan_pi_over_4_plus_alpha_l691_691444

theorem tan_pi_over_4_plus_alpha (α : ℝ) 
  (h : Real.tan (Real.pi / 4 + α) = 2) : 
  1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 5 / 7 := 
by {
  sorry
}

end tan_pi_over_4_plus_alpha_l691_691444


namespace distance_between_parallel_lines_l691_691584

theorem distance_between_parallel_lines 
  (x y : ℝ) 
  (line1 : x + y - 2 = 0) 
  (line2 : x + y + 1 = 0) :
  distanceBetweenLines line1 line2 = (Real.sqrt 2) / 2 := 
sorry

def distanceBetweenLines (line1 line2 : x + y + c = 0) : ℝ :=
  |c1 - c2| / Real.sqrt (a^2 + b^2) where
  (a1, b1, c1) = (1, 1, -2)
  (a2, b2, c2) = (1, 1, -1)

end distance_between_parallel_lines_l691_691584


namespace find_a200_l691_691929

def seq (a : ℕ → ℕ) : Prop :=
a 1 = 1 ∧ ∀ n ≥ 1, a (n + 1) = a n + 2 * a n / n

theorem find_a200 (a : ℕ → ℕ) (h : seq a) : a 200 = 20100 :=
sorry

end find_a200_l691_691929


namespace half_abs_diff_squares_l691_691271

theorem half_abs_diff_squares : 
  let a := 20 in
  let b := 15 in
  let square (x : ℕ) := x * x in
  let abs_diff := abs (square a - square b) in
  abs_diff / 2 = 87.5 := 
by
  let a := 20
  let b := 15
  let square (x : ℕ) := x * x
  let abs_diff := abs (square a - square b)
  have h1 : square a = 400 := by native_decide
  have h2 : square b = 225 := by native_decide
  have h3 : abs_diff = abs (400 - 225) := by simp [square, h1, h2]
  have h4 : abs_diff = 175 := by simp [h3]
  have h5 : abs_diff / 2 = 87.5 := by norm_num [h4]
  exact h5

end half_abs_diff_squares_l691_691271


namespace half_abs_diff_of_squares_l691_691253

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 20) (h2 : b = 15) :
  (1/2 : ℚ) * |(a:ℚ)^2 - (b:ℚ)^2| = 87.5 := by
  sorry

end half_abs_diff_of_squares_l691_691253


namespace find_original_number_l691_691324

theorem find_original_number (x : ℤ) (h : 3 * (2 * x + 9) = 57) : x = 5 := by
  sorry

end find_original_number_l691_691324


namespace norm_projection_l691_691789

variables (v w : ℝ^3)
variable (θ : ℝ)
variable (norm_v : ∥v∥ = 5)
variable (norm_w : ∥w∥ = 8)
variable (angle_vw : θ = Real.pi / 6)

#check (abs (inner v w) / 8 = sqrt(3)/2 * 5)
#check (inner v w = |20 * sqrt(3)| -> 20 * sqrt(3) / 8 = 5 * sqrt(3) / 2 )
theorem norm_projection (n_v : ∥v∥ = 5) (n_w : ∥w∥ = 8) (θ : ℝ) (angle_vw : θ = real.pi / 6) : 
  ∥(v • w) / ∥w∥∥ = (5 * sqrt(3) / 2) :=
sorry

end norm_projection_l691_691789


namespace simplify_expression_l691_691171

variable (x : ℝ)

theorem simplify_expression : (2 * x + 20) + (150 * x + 20) = 152 * x + 40 :=
by
  sorry

end simplify_expression_l691_691171


namespace reciprocal_of_one_is_one_l691_691926

def is_reciprocal (x y : ℝ) : Prop := x * y = 1

theorem reciprocal_of_one_is_one : is_reciprocal 1 1 := 
by
  sorry

end reciprocal_of_one_is_one_l691_691926


namespace intervals_of_decrease_max_min_values_a_minus_2_l691_691776

noncomputable def f (x a : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

theorem intervals_of_decrease (a : ℝ) : 
  ∀ (x : ℝ), (f x a < f (x - 10^(-10)) a) → x < -1 ∨ x > 3 := sorry

noncomputable def f_a_minus_2 (x : ℝ) : ℝ := f x (-2)

theorem max_min_values_a_minus_2 : 
  f_a_minus_2 (-2) = 0 ∧ f_a_minus_2 2 = 20 ∧ f_a_minus_2 (-1) = -7 := sorry

end intervals_of_decrease_max_min_values_a_minus_2_l691_691776


namespace sin_half_angle_range_l691_691453

theorem sin_half_angle_range {A B C D E F : Type*}
  {a b c h : ℝ} (triangle_ABC : euclidean_geometry.triangle A B C)
  (incircle_touches : euclidean_geometry.incident_point_triple (euclidean_geometry.incircle A B C) E F)
  (altitude_AD : euclidean_geometry.altitude A B C D)
  (AE_add_AF_eq_AD : euclidean_geometry.E_add_F_eq_A D E F) :
  let A_half := euclidean_geometry.angle_half A B C in
  A_half.sin_angle ∈ Set.Ico (3 / 5) (Real.sqrt 2 / 2) :=
sorry

end sin_half_angle_range_l691_691453


namespace proof_problem_l691_691051

variable (x z : Real)

def condition1 : Prop :=
  sin x / cos z + sin z / cos x = 2

def condition2 : Prop :=
  cos x / sin z + cos z / sin x = 4

theorem proof_problem (h1 : condition1 x z) (h2 : condition2 x z) : 
  tan x / tan z + tan z / tan x = 2 :=
sorry

end proof_problem_l691_691051


namespace solve_for_y_l691_691483

theorem solve_for_y (y : ℝ) (h : 2^log2 7 = 3 * y + 4) : y = 1 :=
by 
  sorry

end solve_for_y_l691_691483


namespace find_t_l691_691059

theorem find_t (t : ℝ) (h : sqrt (3 * sqrt (t - 3)) = (10 - t)^(1/4)) : t = 3.7 :=
sorry

end find_t_l691_691059


namespace concurrency_lines_l691_691371

/- Given conditions -/
variables {w w1 w2 : Circle}
variables {O1 O2 D E F A B : Point}
variables {t : Line}

-- Conditions definitions
axiom circles_tangency : tangent w1 w2 D
axiom internal_tangency_w1 : tangent_in w1 w E
axiom internal_tangency_w2 : tangent_in w2 w F
axiom common_tangent : common_tangent w1 w2 t
axiom diameter_perpendicular : perpendicular (Line.mk A B) t
axiom same_side_t : same_side t A E O1

-- The theorem to prove
theorem concurrency_lines :
  concurrent Line.mk A O1  Line.mk B O2 Line.mk E F t := 
sorry

end concurrency_lines_l691_691371


namespace totalWeightAlF3_is_correct_l691_691625

-- Define the atomic weights of Aluminum and Fluorine
def atomicWeightAl : ℝ := 26.98
def atomicWeightF : ℝ := 19.00

-- Define the number of atoms of Fluorine in Aluminum Fluoride (AlF3)
def numFluorineAtoms : ℕ := 3

-- Define the number of moles of Aluminum Fluoride
def numMolesAlF3 : ℕ := 7

-- Calculate the molecular weight of Aluminum Fluoride (AlF3)
noncomputable def molecularWeightAlF3 : ℝ :=
  atomicWeightAl + (numFluorineAtoms * atomicWeightF)

-- Calculate the total weight of the given moles of AlF3
noncomputable def totalWeight : ℝ :=
  molecularWeightAlF3 * numMolesAlF3

-- Theorem stating the total weight of 7 moles of AlF3
theorem totalWeightAlF3_is_correct : totalWeight = 587.86 := sorry

end totalWeightAlF3_is_correct_l691_691625


namespace identical_digits_time_l691_691214

theorem identical_digits_time (h : ∀ t, t = 355 -> ∃ u, u = 671 ∧ u - t = 316) : 
  ∃ u, u = 671 ∧ u - 355 = 316 := 
by sorry

end identical_digits_time_l691_691214


namespace cube_painted_four_faces_l691_691677

theorem cube_painted_four_faces (n : ℕ) (hn : n ≠ 0) (h : (4 * n^2) / (6 * n^3) = 1 / 3) : n = 2 :=
by
  have : 4 * n^2 = 4 * n^2 := by rfl
  sorry

end cube_painted_four_faces_l691_691677


namespace pieces_left_after_fourth_day_l691_691873

def total_pieces := 2000
def first_day_remaining := total_pieces - (0.10 * total_pieces)
def second_day_remaining := first_day_remaining - (0.25 * first_day_remaining)
def third_day_remaining := second_day_remaining - (0.30 * second_day_remaining)
def fourth_day_remaining := third_day_remaining - (0.40 * third_day_remaining)

theorem pieces_left_after_fourth_day : fourth_day_remaining = 567 := 
sorry

end pieces_left_after_fourth_day_l691_691873


namespace smallest_n_digits_l691_691860

-- Definitions for the conditions
def divisible_by (m n : ℕ) : Prop := ∃ k, m = n * k
def is_perfect_fourth_power (x : ℕ) : Prop := ∃ k, x = k^4
def is_perfect_cube (x : ℕ) : Prop := ∃ k, x = k^3

-- The smallest positive integer n that meets the conditions
def smallest_n : ℕ := (2^6 * 3^6 * 5^6)

-- Statement to be proved
theorem smallest_n_digits :
  (divisible_by smallest_n 30) ∧
  (is_perfect_fourth_power (smallest_n^2)) ∧
  (is_perfect_cube (smallest_n^4)) ∧
  (nat.digits 10 smallest_n).length = 9 :=
by {
  sorry
}

end smallest_n_digits_l691_691860


namespace find_t_l691_691057

theorem find_t (t : ℝ) (h : real.sqrt (3 * real.sqrt (t - 3)) = real.root 4 (10 - t)) : t = 3.7 := sorry

end find_t_l691_691057


namespace sin_cos_value_l691_691415

noncomputable def tan_plus_pi_div_two_eq_two (θ : ℝ) : Prop :=
  Real.tan (θ + Real.pi / 2) = 2

theorem sin_cos_value (θ : ℝ) (h : tan_plus_pi_div_two_eq_two θ) :
  Real.sin θ * Real.cos θ = -2 / 5 :=
sorry

end sin_cos_value_l691_691415


namespace distance_inequality_l691_691119

-- Given the setup with specified distances and points
variables (l l' : ℝ) (A B C : ℝ) (a b c : ℝ)

-- Definitions for conditions
def B_midpoint_of_AC (A B C : ℝ) : Prop := B = (A + C) / 2
def distances_from_l_prime (A B C l' a b c : ℝ) : Prop := 
  ∀ (A B C : ℝ), A = a ∧ B = b ∧ C = c

-- Main theorem to prove
theorem distance_inequality 
  (h1 : B_midpoint_of_AC A B C)
  (h2 : distances_from_l_prime A B C l' a b c) :
  b ≤ sqrt ((a^2 + c^2) / 2) ∧ (b = sqrt ((a^2 + c^2) / 2) → ∀ l l', parallel l l') :=
sorry

end distance_inequality_l691_691119


namespace number_of_non_congruent_rectangles_l691_691673

theorem number_of_non_congruent_rectangles (w h : ℕ) (hp : 2 * (w + h) = 64) : 
  ∃ (n : ℕ), n = 16 ∧ ∀ (x y : ℕ), (2 * (x + y) = 64 → (x, y) ≠ (y, x) ∧ x ≠ y → n = 16) :=
by 
  -- The proof is omitted
  sorry

end number_of_non_congruent_rectangles_l691_691673


namespace stratified_sampling_l691_691336

-- Definitions for conditions given in the problem
def total_students : ℕ := 1000
def freshmen : ℕ := 400
def sophomores : ℕ := 340
def juniors : ℕ := 260
def sample_size : ℕ := 50

-- The Lean statement of the math proof problem
theorem stratified_sampling :
  (freshmen.to_nat * sample_size / total_students = 20) ∧
  (sophomores.to_nat * sample_size / total_students = 17) ∧
  (juniors.to_nat * sample_size / total_students = 13) :=
sorry

end stratified_sampling_l691_691336


namespace total_liters_needed_to_fill_two_tanks_l691_691188

theorem total_liters_needed_to_fill_two_tanks (capacity : ℕ) (liters_first_tank : ℕ) (liters_second_tank : ℕ) (percent_filled : ℕ) :
  liters_first_tank = 300 → 
  liters_second_tank = 450 → 
  percent_filled = 45 → 
  capacity = (liters_second_tank * 100) / percent_filled → 
  1000 - 300 = 700 → 
  1000 - 450 = 550 → 
  700 + 550 = 1250 :=
by sorry

end total_liters_needed_to_fill_two_tanks_l691_691188


namespace children_eating_porridge_today_l691_691829

theorem children_eating_porridge_today
  (eat_every_day : ℕ)
  (eat_every_other_day : ℕ)
  (ate_yesterday : ℕ) :
  eat_every_day = 5 →
  eat_every_other_day = 7 →
  ate_yesterday = 9 →
  (eat_every_day + (eat_every_other_day - (ate_yesterday - eat_every_day)) = 8) :=
by
  intros h1 h2 h3
  sorry

end children_eating_porridge_today_l691_691829


namespace half_abs_diff_squares_l691_691275

theorem half_abs_diff_squares : 
  let a := 20 in
  let b := 15 in
  let square (x : ℕ) := x * x in
  let abs_diff := abs (square a - square b) in
  abs_diff / 2 = 87.5 := 
by
  let a := 20
  let b := 15
  let square (x : ℕ) := x * x
  let abs_diff := abs (square a - square b)
  have h1 : square a = 400 := by native_decide
  have h2 : square b = 225 := by native_decide
  have h3 : abs_diff = abs (400 - 225) := by simp [square, h1, h2]
  have h4 : abs_diff = 175 := by simp [h3]
  have h5 : abs_diff / 2 = 87.5 := by norm_num [h4]
  exact h5

end half_abs_diff_squares_l691_691275


namespace angle_BED_is_62_5_l691_691502

-- Definitions for conditions
variables {A B C D E : Type}
variables [Geometry A B C]
variables (α β γ : ℝ) (α_BED : ℝ)

-- Angles in triangle ABC
def angle_A : ℝ := 40
def angle_C : ℝ := 85
def angle_B : ℝ := 180 - angle_A - angle_C

-- Isosceles conditions from the problem
def DB_eq_BE : Prop := DB = BE
def DE_eq_AC : Prop := DE = AC

-- Prove that the required angle condition holds
theorem angle_BED_is_62_5 :
  angle_B = 55 ∧ DB_eq_BE ∧ DE_eq_AC → α_BED = 62.5 :=
by
  intros h,
  sorry

end angle_BED_is_62_5_l691_691502


namespace small_boxes_in_big_box_l691_691658

theorem small_boxes_in_big_box (total_candles : ℕ) (candles_per_small : ℕ) (total_big_boxes : ℕ) 
  (h1 : total_candles = 8000) 
  (h2 : candles_per_small = 40) 
  (h3 : total_big_boxes = 50) :
  (total_candles / candles_per_small) / total_big_boxes = 4 :=
by
  sorry

end small_boxes_in_big_box_l691_691658


namespace seating_arrangements_336_l691_691606

-- Given: There are seven seats in a row.
-- Given: Four people are to be seated.
-- Given: Exactly two adjacent seats are unoccupied.
-- Given: Individuals A and B do not sit next to each other.
-- Prove: The number of different seating arrangements is 336.

theorem seating_arrangements_336 :
  ∃ (P4 : Finset (Fin 7 → Fin 7))
    (P3 : Finset (Fin 6 → Fin 6))
    (P2 : Finset (Fin 2 → Fin 2)),
  (P4.card = (4!)) ∧
    (P3.card = (3!)) ∧
    (P2.card = (2!)) ∧
    (5 * 4 * (P4.card) - 2 * (P3.card) * 4 * 3 = 336) :=
by
  let P4 := Finset.univ.filter (λ (f: Fin 7 → Fin 7),
    list.pairwise (≠) ((finset.range 7).filter_map (f.to_emb_domain).values))
  let P3 := Finset.univ.filter (λ (f: Fin 6 → Fin 6),
    list.pairwise (≠) ((finset.range 6).filter_map (f.to_emb_domain).values))
  let P2 := Finset.univ.filter (λ (f: Fin 2 → Fin 2),
    list.pairwise (≠) ((finset.range 2).filter_map (f.to_emb_domain).values))
  use [P4, P3, P2]
  split
  · sorry
  split
  · sorry
  split
  · sorry
  · sorry

end seating_arrangements_336_l691_691606


namespace units_digit_b2010_l691_691547

def b : ℕ → ℕ
| 0     := 1
| (n+1) := 0  -- This will be defined properly by the condition

axiom b_positive (n : ℕ) : n > 0 → b n > 0

axiom b_equation (n : ℕ) : n > 0 → n * (b (n + 1))^2 - 2 * (b n)^2 - (2 * n - 1) * b (n + 1) * b n = 0

def units_digit (x : ℕ) : ℕ := x % 10

theorem units_digit_b2010 : units_digit (b 2010) = 2 := sorry

end units_digit_b2010_l691_691547


namespace helium_balloon_height_l691_691792

theorem helium_balloon_height :
    let total_budget := 200
    let cost_sheet := 42
    let cost_rope := 18
    let cost_propane := 14
    let cost_per_ounce_helium := 1.5
    let height_per_ounce := 113
    let amount_spent := cost_sheet + cost_rope + cost_propane
    let money_left_for_helium := total_budget - amount_spent
    let ounces_helium := money_left_for_helium / cost_per_ounce_helium
    let total_height := height_per_ounce * ounces_helium
    total_height = 9492 := sorry

end helium_balloon_height_l691_691792


namespace range_of_a_l691_691471

noncomputable def A : Set ℝ := {x | Real.log x / Real.log 2 ≤ 2}
noncomputable def B (a : ℝ) : Set ℝ := Set.Iio a

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : 4 < a :=
by
  have : A = {x | 0 < x ∧ x ≤ 4} := 
    sorry -- Details to justify that A = {x | 0 < x ∧ x ≤ 4}
  rw this at h
  have : ∃ y, 4 = y ∧ A ⊆ Set.Iio a :=
    sorry -- Details to show that 4 is the upper bound of A and is a subset of an open interval (∞, a)
  sorry -- Details to conclude that 4 < a.

end range_of_a_l691_691471


namespace sandy_shopping_l691_691168

theorem sandy_shopping (T : ℝ) (h : 0.70 * T = 217) : T = 310 := sorry

end sandy_shopping_l691_691168


namespace preceding_in_base3_l691_691808

def base3_to_decimal (n : String) : Nat :=
  n.foldl (λ (acc : Nat) (d : Char) =>
    acc * 3 + (d.toNat - '0'.toNat)) 0

def decimal_to_base3 (n : Nat) : String :=
  if n = 0 then "0"
  else let rec aux (n : Nat) (acc : String) :=
    if n = 0 then acc else aux (n / 3) (Char.ofNat (n % 3 + '0'.toNat) :: acc)
  aux n ""

theorem preceding_in_base3 (N : String) (H : N = "2101") : 
  decimal_to_base3 (base3_to_decimal N - 1) = "2100" := 
by
  sorry

end preceding_in_base3_l691_691808


namespace derivative_at_two_l691_691034

noncomputable def f (a : ℝ) (g : ℝ) (x : ℝ) : ℝ := a * x^3 + g * x^2 + 3

theorem derivative_at_two (a f_prime_2 : ℝ) (h_deriv_at_1 : deriv (f a f_prime_2) 1 = -5) :
  deriv (f a f_prime_2) 2 = -5 := by
  sorry

end derivative_at_two_l691_691034


namespace half_abs_diff_squares_l691_691270

theorem half_abs_diff_squares : 
  let a := 20 in
  let b := 15 in
  let square (x : ℕ) := x * x in
  let abs_diff := abs (square a - square b) in
  abs_diff / 2 = 87.5 := 
by
  let a := 20
  let b := 15
  let square (x : ℕ) := x * x
  let abs_diff := abs (square a - square b)
  have h1 : square a = 400 := by native_decide
  have h2 : square b = 225 := by native_decide
  have h3 : abs_diff = abs (400 - 225) := by simp [square, h1, h2]
  have h4 : abs_diff = 175 := by simp [h3]
  have h5 : abs_diff / 2 = 87.5 := by norm_num [h4]
  exact h5

end half_abs_diff_squares_l691_691270


namespace area_ratio_equilateral_triangles_l691_691719

theorem area_ratio_equilateral_triangles
  (ABC DEF : Type)
  [equilateral_triangle ABC]
  [equilateral_triangle DEF]
  (inscribed : inscribed_triangle DEF ABC)
  (perpendicular : perp DE BC) :
  area DEF / area ABC = 1 / 3 :=
sorry

end area_ratio_equilateral_triangles_l691_691719


namespace bc_dot_ap_l691_691843

theorem bc_dot_ap (A B C P D : Type)
  [vector_space ℝ A B C P D]
  (AB AC BC AP AD DP : ℝ)
  (hAB : AB = 4)
  (hAC : AC = 3)
  (hBC : BC = (AC - AB))
  (hAP : AP = AD + DP)
  (hDP_perp : DP ∙ BC = 0)
  (hAD : AD = 1/2 * (AB + AC)) :
  BC ∙ AP = -7/2 :=
sorry

end bc_dot_ap_l691_691843


namespace oprah_years_to_reduce_collection_l691_691883

theorem oprah_years_to_reduce_collection (initial_cars final_cars average_cars_per_year : ℕ) (h1 : initial_cars = 3500) (h2 : final_cars = 500) (h3 : average_cars_per_year = 50) : 
  (initial_cars - final_cars) / average_cars_per_year = 60 := 
by
  rw [h1, h2, h3]
  sorry

end oprah_years_to_reduce_collection_l691_691883


namespace intersection_sum_x_coordinates_mod_17_l691_691148

theorem intersection_sum_x_coordinates_mod_17 :
  ∃ x : ℤ, (∃ y₁ y₂ : ℤ, (y₁ ≡ 7 * x + 3 [ZMOD 17]) ∧ (y₂ ≡ 13 * x + 4 [ZMOD 17]))
       ∧ x ≡ 14 [ZMOD 17]  :=
by
  sorry

end intersection_sum_x_coordinates_mod_17_l691_691148


namespace union_complement_A_B_l691_691038

-- Definitions based on conditions
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | x < 6}
def C_R (S : Set ℝ) : Set ℝ := {x | ¬ (x ∈ S)}

-- The proof problem statement
theorem union_complement_A_B :
  (C_R B ∪ A = {x | 0 ≤ x}) :=
by 
  sorry

end union_complement_A_B_l691_691038


namespace difference_in_zits_l691_691915

variable (avgZitsSwanson : ℕ := 5)
variable (avgZitsJones : ℕ := 6)
variable (numKidsSwanson : ℕ := 25)
variable (numKidsJones : ℕ := 32)
variable (totalZitsSwanson : ℕ := avgZitsSwanson * numKidsSwanson)
variable (totalZitsJones : ℕ := avgZitsJones * numKidsJones)

theorem difference_in_zits :
  totalZitsJones - totalZitsSwanson = 67 := by
  sorry

end difference_in_zits_l691_691915


namespace general_form_identity_expression_simplification_l691_691881

section
variable (a b x y : ℝ)

theorem general_form_identity : (a + b) * (a^2 - a * b + b^2) = a^3 + b^3 :=
by
  sorry

theorem expression_simplification : (x + y) * (x^2 - x * y + y^2) - (x - y) * (x^2 + x * y + y^2) = 2 * y^3 :=
by
  sorry
end

end general_form_identity_expression_simplification_l691_691881


namespace total_bill_correct_l691_691880

noncomputable def total_bill (n : ℕ) 
  (split_equally : ∀ k, k ∈ (fin 9) → nat)
  (tom_forgot : split_equally 8 = 3 + split_equally 0) : Prop :=
n = 216

theorem total_bill_correct :
  ∃ n : ℕ, total_bill n (λ k, (3 : ℕ)) (by sorry) :=
sorry

end total_bill_correct_l691_691880


namespace half_abs_diff_of_squares_l691_691246

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 20) (h2 : b = 15) :
  (1/2 : ℚ) * |(a:ℚ)^2 - (b:ℚ)^2| = 87.5 := by
  sorry

end half_abs_diff_of_squares_l691_691246


namespace total_pikes_l691_691565

theorem total_pikes (x : ℝ) (h : x = 4 + (1/2) * x) : x = 8 :=
sorry

end total_pikes_l691_691565


namespace theater_ticket_lineup_l691_691196

theorem theater_ticket_lineup (eight_people : Fin 8)
  (two_standing_together : (Fin 8) × (Fin 8))
  (h : two_standing_together.1 ≠ two_standing_together.2) : 
  ∃ n : ℕ, n = 10080 := by
  let group_of_two := [two_standing_together.1, two_standing_together.2]
  have grouped_people := (8 - 1) -- 7! permutations of groups
  let arrange_two := 2 -- 2! permutations within the group
  have total_ways := (grouped_people.factors.prod * arrange_two.factors.prod)
  exact ⟨ total_ways, sorry ⟩

end theater_ticket_lineup_l691_691196


namespace min_value_of_expression_l691_691752

theorem min_value_of_expression (x y : ℝ) (h_range : 1 < x ∧ 1 < y) (h_condition : x * y - 2 * x - y + 1 = 0) : 
    (∀ (f : ℝ), f = (3 / 2) * x^2 + y^2 → f ≥ 15) :=
begin
  sorry
end

end min_value_of_expression_l691_691752


namespace identical_digits_time_l691_691215

theorem identical_digits_time (h : ∀ t, t = 355 -> ∃ u, u = 671 ∧ u - t = 316) : 
  ∃ u, u = 671 ∧ u - 355 = 316 := 
by sorry

end identical_digits_time_l691_691215


namespace remainder_sum_first_150_div_10000_l691_691953

theorem remainder_sum_first_150_div_10000 :
  (∑ i in Finset.range 151, i) % 10000 = 1325 :=
by
  sorry

end remainder_sum_first_150_div_10000_l691_691953


namespace sin_inequality_l691_691132

-- Let α, β, and γ be the angles of a triangle.
variables (α β γ : ℝ)

-- Considering α, β, and γ as angles of a triangle:
-- α + β + γ = π (sum of angles in a triangle)
axiom angle_sum (h : α + β + γ = π)

-- Prove the inequality:
theorem sin_inequality (h : α + β + γ = π) : 
  sin ((α / 2) + β) + sin ((β / 2) + γ) + sin ((γ / 2) + α) > sin α + sin β + sin γ := 
sorry

end sin_inequality_l691_691132


namespace trajectory_of_point_M_l691_691770

def ellipse (x y : ℝ) : Prop := (x^2 / 6) + (y^2 / 2) = 1

def left_focus : (ℝ × ℝ) := (-2, 0)

def trajectory_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

def within_range (x0 : ℝ) (y0 : ℝ) : Prop :=
  let sqrt6 := Real.sqrt 6 in
  (-sqrt6 ≤ x0 ∧ x0 ≤ sqrt6) ∧
  (Real.sqrt 6 / 2 - 1 ≤ Real.sqrt ((x0 + 1)^2 + y0^2) ∧ Real.sqrt ((x0 + 1)^2 + y0^2) ≤ Real.sqrt 6 + 2)

theorem trajectory_of_point_M (x y : ℝ) :
  (∃ P A B : ℝ × ℝ, ellipse P.1 P.2 ∧ ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧
    ∃ m : ℝ, ∀ x0 y0 : ℝ, left_focus = (-2, 0) ∧
      AB passes through left_focus ∧
      (x0 = m * y0 - 2) ∧
      (x0 ≠ 0) ∧
      (M (x,y) ∧ O (0,0) ∧ (vector (0,0) • arrangement (x,y)) = 0)
      → within_range (x0) (y0)
  :=
sorry

end trajectory_of_point_M_l691_691770


namespace crescent_moon_falcata_area_l691_691378

-- Definitions of conditions:
def circle_area (r : ℝ) : ℝ := π * r^2
def quarter_circle_area (r : ℝ) : ℝ := (1/4) * circle_area r
def semicircle_area (r : ℝ) : ℝ := (1/2) * circle_area r

-- Main Problem Statement:
theorem crescent_moon_falcata_area :
  let area_large_quarter_circle := quarter_circle_area 4
  let area_small_semicircle := semicircle_area 1
  area_large_quarter_circle - area_small_semicircle = (7 / 2) * π :=
by {
  sorry -- Proof to be completed
}

end crescent_moon_falcata_area_l691_691378


namespace polynomial_property_l691_691869

theorem polynomial_property (p : ℕ → ℕ) :
  p(1) = 3 ∧ p(2) = 8 ∧ p(3) = 15 ∧ p(4) = 24 → (∀ x, p x = x^2 + 2*x + 1) ∧ p(5) = 36 :=
begin
  sorry
end

end polynomial_property_l691_691869


namespace tan_double_angle_l691_691004

-- Define the given condition
def tan_alpha_condition (α : ℝ) : Prop := Real.tan α = 2

-- Define the target statement to prove
theorem tan_double_angle (α : ℝ) (h : tan_alpha_condition α) : Real.tan (2 * α) = -4/3 :=
by
  -- the proof steps go here
  sorry

end tan_double_angle_l691_691004


namespace remainder_500th_T_l691_691852

-- Define the sequence T as described in the problem
def T : ℕ → ℕ
| n := sorry

-- Define the binary representation criterion condition
def binary_has_six_ones (n : ℕ) : Prop := (nat.popcount n) = 6

-- Define a statement proving that the 500th element of T modulo 500 is 198
theorem remainder_500th_T :
  (T 499) % 500 = 198 :=
begin
  -- Assuming T is a strictly increasing sequence of positive integers
  -- whose binary representation has exactly six '1's.
  sorry
end

end remainder_500th_T_l691_691852


namespace min_toothpicks_to_remove_l691_691220

/-- 
The proof problem: Given a figure made from 30 toothpicks containing over 25 triangles, prove that 
the minimum number of toothpicks that must be removed so that no triangles remain is 10. 
-/
theorem min_toothpicks_to_remove (toothpicks : ℕ) (triangles : ℕ) (h1 : toothpicks = 30) (h2 : triangles > 25) : ∃ (min_remove : ℕ), min_remove = 10 ∧ (∀ n, n < min_remove → some_triangles_remain n) :=
by
  sorry

end min_toothpicks_to_remove_l691_691220


namespace half_absolute_difference_of_squares_l691_691266

-- Defining the variables a and b involved in the problem
def a : ℤ := 20
def b : ℤ := 15

-- Statement to prove the solution
theorem half_absolute_difference_of_squares : 
    (1 / 2 : ℚ) * (abs ((a ^ 2) - (b ^ 2))) = 87.5 :=
by
    -- Proof omitted
    sorry

end half_absolute_difference_of_squares_l691_691266


namespace cos_gamma_l691_691546

variables (α β γ : ℝ)
noncomputable def cos_α := 2 / 5
noncomputable def cos_β := 1 / 4

def proof_problem : Prop :=
  cos_α ^ 2 + cos_β ^ 2 + cos γ ^ 2 = 1 → cos γ = (√311) / 20

theorem cos_gamma (h : proof_problem) : cos γ = (√311) / 20 :=
  sorry

end cos_gamma_l691_691546


namespace greatest_integer_is_8_l691_691622

theorem greatest_integer_is_8 {a b : ℤ} (h_sum : a + b + 8 = 21) : max a (max b 8) = 8 :=
by
  sorry

end greatest_integer_is_8_l691_691622


namespace jar_weight_percentage_l691_691601

noncomputable def weight_of_empty_jar : Type := ℝ
noncomputable def weight_of_coffee_beans : Type := ℝ

variables (J : weight_of_empty_jar) (B : weight_of_coffee_beans)

axiom jar_and_beans_weight : 0.60 * (J + B) = J + 0.5 * B

theorem jar_weight_percentage :
  J / (J + B) * 100 = 20 :=
begin
  -- Proof omitted
  sorry
end

end jar_weight_percentage_l691_691601


namespace solve_for_t_l691_691065

theorem solve_for_t (t : ℝ) (h : sqrt (3 * sqrt (t - 3)) = real.sqrt (real.sqrt (real.sqrt (10 - t)))) : t = 3.7 :=
by
  sorry

end solve_for_t_l691_691065


namespace general_formula_for_an_sum_of_bn_l691_691424

variables (a : ℕ → ℤ) (S : ℕ → ℤ) (b : ℕ → ℝ) (T : ℕ → ℝ)

-- Defining the conditions
def is_arithmetic_sequence := ∀ n, a (n + 1) - a n = a 1 - a 0

def sum_of_first_n_terms (n : ℕ) := S n = n * (a 1 + a n) / 2

def S3_condition := S 3 = -15

def is_geometric_sequence_not_one := ∀ n, (a (n + 1) + 1) * (a (n + 1) + 1) = (a n + 1) * (a (n + 2) + 1) → (a (n + 1) + 1) ≠ (a n + 1)

def b_definition (n : ℕ) := b n = 1 / (S n)

-- Proof statement for part 1
theorem general_formula_for_an 
  (harith : is_arithmetic_sequence a) 
  (sum_cond : sum_of_first_n_terms a S) 
  (s3_cond : S3_condition S) 
  (geo_cond : is_geometric_sequence_not_one a) :
  ∀ n, a n = -2 * n - 1 := 
sorry

-- Proof statement for part 2
theorem sum_of_bn 
  (harith : is_arithmetic_sequence a) 
  (sum_cond : sum_of_first_n_terms a S) 
  (s3_cond : S3_condition S) 
  (geo_cond : is_geometric_sequence_not_one a) 
  (b_def : b_definition b S) :
  ∀ n, T n = - 3 / 4 + (2 * n + 3) / (2 * (n + 1) * (n + 2)) := 
sorry

end general_formula_for_an_sum_of_bn_l691_691424


namespace problem_proof_l691_691753

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  (x^2 / 2 + y^2 = 1)

noncomputable def point_P := (0:ℝ, 1:ℝ)

noncomputable def sum_distances_to_foci (f1 f2 P : ℝ × ℝ) : ℝ :=
  dist P f1 + dist P f2

noncomputable def sum_distances_condition : Prop :=
  sum_distances_to_foci (sqrt 2, 0) (-(sqrt 2), 0) point_P = 2 * sqrt 2

noncomputable def point_A (m n : ℝ) : Prop :=
  (m = 4 / 3) ∧ (n = 1 / 3 ∨ n = -1 / 3)

noncomputable def condition_S_equals_half_S (xA yA xB yB xM yM : ℝ) : Prop :=
  xA = 4 / 3 ∧ (yA = 1 / 3 ∨ yA = -1 / 3)

theorem problem_proof :
  ellipse_eq 0 1 ∧ sum_distances_condition ∧ 
  ∃ (xA yA : ℝ), ellipse_eq xA yA ∧ condition_S_equals_half_S xA yA _ _ 4 (4 * yA / (4/3) + 1) :=
by
  sorry

end problem_proof_l691_691753


namespace annual_salt_requirement_l691_691682

-- Definition to capture the conditions in the problem
def total_months : ℕ := 2 * 12 + 8
def total_salt : ℝ := 16
def one_year_months : ℕ := 12
def monthly_salt_requirement (total_salt : ℝ) (total_months : ℕ) : ℝ :=
  total_salt / total_months

-- The proof statement
theorem annual_salt_requirement : 
  monthly_salt_requirement total_salt total_months * one_year_months = 6 := 
by 
  -- Prove the statement using the conditions
  sorry

end annual_salt_requirement_l691_691682


namespace number_of_workers_l691_691907

open Real

theorem number_of_workers (W : ℝ) 
    (average_salary_workers average_salary_technicians average_salary_non_technicians : ℝ)
    (h1 : average_salary_workers = 8000)
    (h2 : average_salary_technicians = 12000)
    (h3 : average_salary_non_technicians = 6000)
    (h4 : average_salary_workers * W = average_salary_technicians * 7 + average_salary_non_technicians * (W - 7)) :
    W = 21 :=
by
  rw [h1, h2, h3] at h4
  simp at h4
  linarith

end number_of_workers_l691_691907


namespace expression_I_expression_II_l691_691698

-- Problem (I)
theorem expression_I :
    (9/4 : ℚ)^(1/2) - (9.6)^0 - (27/8 : ℚ)^(-2/3) + (2/3 : ℚ)^2 = 1/2 := by
  sorry

-- Problem (II)
theorem expression_II :
    log 5 25 + log 10 (1/100) + log e (sqrt e) + (2^(log 2 3) : ℝ) = 7/2 := by
  sorry

end expression_I_expression_II_l691_691698


namespace smallest_n_is_138_l691_691201

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m, m ∣ p → m = 1 ∨ m = p

def smallest_prime_product (n x y : ℕ) : Prop :=
  n = x * y * (10 * x + y) ∧
  n ≥ 100 ∧ n < 1000 ∧
  x < 10 ∧ y < 10 ∧
  x % 2 = 0 ∧ is_prime x ∧ 
  y % 2 = 1 ∧ is_prime y ∧ 
  is_prime (10 * x + y) 

theorem smallest_n_is_138 : ∃ n, (∃ x y, smallest_prime_product n x y) ∧ ∀ m, (∃ x y, smallest_prime_product m x y) → n ≤ m := 
  exists.intro 138 
  (and.intro
    (exists.intro 2 
      (exists.intro 3 
        (and.intro
          (eq.refl (2 * 3 * 23)) 
          (and.intro
            (nat.le_refl 100)
            (and.intro
              (nat.lt_of_le_and_lt (nat.zero_lt_succ 1) (nat.lt_of_sub_le_sub 2))
              (and.intro
                (nat.zero_lt_succ 8)
                (and.intro
                  (nat.zero_lt_succ 8)
                  (and.intro rfl
                    (and.intro (nat.prime_two) 
                      (and.intro (eq.refl 1) 
                        (nat.prime_def 3)) (nat.prime_def 2)))))))) sorry

end smallest_n_is_138_l691_691201


namespace range_of_f_when_a_neg_2_is_0_to_4_and_bounded_range_of_a_if_f_bounded_by_4_l691_691029

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 + a * Real.sin x - Real.cos x ^ 2

theorem range_of_f_when_a_neg_2_is_0_to_4_and_bounded :
  (∀ x : ℝ, 0 ≤ f (-2) x ∧ f (-2) x ≤ 4) :=
sorry

theorem range_of_a_if_f_bounded_by_4 :
  (∀ x : ℝ, abs (f a x) ≤ 4) → (-2 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_f_when_a_neg_2_is_0_to_4_and_bounded_range_of_a_if_f_bounded_by_4_l691_691029


namespace exists_polynomial_l691_691968

def is_primitive_lattice_point (x y : ℤ) : Prop :=
  Int.gcd x y = 1

def satisfies_equation (n : ℕ) (a : Fin n → ℤ) (x y : ℤ) : Prop :=
  ∑ i in Finset.range (n + 1), a ⟨i, sorry⟩ * x^(n - i) * y^i = 1

theorem exists_polynomial (S : Finset (ℤ × ℤ)) (h_primitive : ∀ (p : ℤ × ℤ), p ∈ S → is_primitive_lattice_point p.fst p.snd) : 
  ∃ (n : ℕ) (a : Fin n → ℤ), ∀ (p : ℤ × ℤ), p ∈ S → satisfies_equation n a p.fst p.snd := 
by
  sorry

end exists_polynomial_l691_691968


namespace imaginary_part_of_l691_691975

theorem imaginary_part_of (i : ℂ) (h : i.im = 1) : (1 + i) ^ 5 = -14 - 4 * i := by
  sorry

end imaginary_part_of_l691_691975


namespace oprah_years_to_reduce_collection_l691_691884

theorem oprah_years_to_reduce_collection (initial_cars final_cars average_cars_per_year : ℕ) (h1 : initial_cars = 3500) (h2 : final_cars = 500) (h3 : average_cars_per_year = 50) : 
  (initial_cars - final_cars) / average_cars_per_year = 60 := 
by
  rw [h1, h2, h3]
  sorry

end oprah_years_to_reduce_collection_l691_691884


namespace balloon_height_l691_691794

theorem balloon_height :
  let initial_money : ℝ := 200
  let cost_sheet : ℝ := 42
  let cost_rope : ℝ := 18
  let cost_tank_and_burner : ℝ := 14
  let helium_price_per_ounce : ℝ := 1.5
  let lift_per_ounce : ℝ := 113
  let remaining_money := initial_money - cost_sheet - cost_rope - cost_tank_and_burner
  let ounces_of_helium := remaining_money / helium_price_per_ounce
  let height := ounces_of_helium * lift_per_ounce
  height = 9492 :=
by
  sorry

end balloon_height_l691_691794


namespace distinct_real_numbers_satisfying_system_l691_691384

theorem distinct_real_numbers_satisfying_system :
  ∃! (x y z : ℝ),
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  (x^2 + y^2 = -x + 3 * y + z) ∧
  (y^2 + z^2 = x + 3 * y - z) ∧
  (x^2 + z^2 = 2 * x + 2 * y - z) ∧
  ((x = 0 ∧ y = 1 ∧ z = -2) ∨ (x = -3/2 ∧ y = 5/2 ∧ z = -1/2)) :=
sorry

end distinct_real_numbers_satisfying_system_l691_691384


namespace traffic_volume_range_traffic_volume_max_l691_691996

theorem traffic_volume_range (v : ℝ) (hv : v > 0) :
  (let y := 25 * v / (v^2 - 5 * v + 16) in
   y ≥ 5 ↔ 2 ≤ v ∧ v ≤ 8) :=
sorry

theorem traffic_volume_max (v : ℝ) (hv : v > 0) :
  ∃ (vmax : ℝ) (ymax : ℝ), 
  (let y := 25 * v / (v^2 - 5 * v + 16) in
   (v = 4 ∧ y = 25 / 3) ∧
   ∀ w, (let yw := 25 * w / (w^2 - 5 * w + 16) in yw ≤ 25 / 3)) :=
sorry

end traffic_volume_range_traffic_volume_max_l691_691996


namespace binom_coeff_computation_l691_691488

-- Definition of binomial coefficient for real x and nonnegative integer k
def binom (x : ℝ) (k : ℕ) : ℝ :=
  if k = 0 then 1 else (x * binom (x - 1) (k - 1)) / k

-- The main theorem
theorem binom_coeff_computation :
  (binom 1.5 2015 * 4^2015) / (binom 4030 2015) = - (1 / 4030) :=
by
  sorry

end binom_coeff_computation_l691_691488


namespace common_number_is_eleven_l691_691937

theorem common_number_is_eleven 
  (a b c d e f g h i : ℝ)
  (H1 : (a + b + c + d + e) / 5 = 7)
  (H2 : (e + f + g + h + i) / 5 = 10)
  (H3 : (a + b + c + d + e + f + g + h + i) / 9 = 74 / 9) :
  e = 11 := 
sorry

end common_number_is_eleven_l691_691937


namespace round_to_nearest_thousandth_l691_691228

theorem round_to_nearest_thousandth : 
  (Real.round (0.05019 * 1000) * 0.001 = 0.050) :=
by
  sorry

end round_to_nearest_thousandth_l691_691228


namespace triangle_area_inequality_l691_691116

noncomputable def area {α : Type} [EuclideanGeometry α] (T : Triangle α) : Real := sorry

theorem triangle_area_inequality (α : Type) [EuclideanGeometry α] 
  (A B C H : α) (H_A H_B H_C : α)
  (h_acute : AcuteTriangle A B C)
  (hH : Orthocenter H A B C)
  (hH_A : SecondIntersectionOfCircumcircleWithAltitude H_A A B C)
  (hH_B : SecondIntersectionOfCircumcircleWithAltitude H_B B A C)
  (hH_C : SecondIntersectionOfCircumcircleWithAltitude H_C C A B) :
  area (Triangle.mk H_A H_B H_C) ≤ area (Triangle.mk A B C) := 
sorry

end triangle_area_inequality_l691_691116


namespace exam_items_count_l691_691874

-- Definitions based on the problem conditions
variable (X : ℕ) -- Number of items in the exam
variable (Lyssa_incorrect_pct : ℝ) (Precious_mistakes : ℕ) (Lyssa_more_correct : ℕ) -- Additional parameters introduced in conditions

-- Conditions from the problem
def condition1 : Prop := Lyssa_incorrect_pct = 0.2
def condition2 : Prop := Precious_mistakes = 12
def condition3 : Prop := Lyssa_more_correct = 3

-- Mathematical conditions based on solution steps
def condition4 : Prop := 0.8 * (X : ℝ) = ((X : ℝ) - 12) + 3

-- The theorem we want to prove
theorem exam_items_count (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : X = 45 :=
by
  -- Place the proof here
  sorry

end exam_items_count_l691_691874


namespace find_new_length_l691_691897

def initial_length_cm : ℕ := 100
def erased_length_cm : ℕ := 24
def final_length_cm : ℕ := 76

theorem find_new_length : initial_length_cm - erased_length_cm = final_length_cm := by
  sorry

end find_new_length_l691_691897


namespace min_num_cubes_l691_691979

/--
Given a box with dimensions 98 inches long, 77 inches wide, and 35 inches deep,
prove that the smallest number of cubes of different sizes needed to completely fill the box is 770.
-/
theorem min_num_cubes (L W D : ℕ) (hL : L = 98) (hW : W = 77) (hD : D = 35) : 
  ∃ n : ℕ, n = 770 ∧ 
           ∃ (cubes : list ℕ), 
           (list.sum cubes = 770) ∧ 
           (∀ c ∈ cubes, c > 0) ∧ 
           (∀ i j, i ≠ j → cubes.nth i ≠ cubes.nth j) :=
sorry

end min_num_cubes_l691_691979


namespace rhombus_area_l691_691674

theorem rhombus_area (d1 d2 : ℝ) (K : ℝ) (h1 : d1 = 3 * d2) (h2 : 4 * (sqrt((d2 / 2)^2 + (d1 / 2)^2)) = 40) :
  K = 60 :=
by
  sorry

end rhombus_area_l691_691674


namespace height_from_point_to_plane_eq_2_l691_691840

open Real EuclideanGeometry

variables (A B D P : Point)
variables (AB AD AP : Vect3)
variables (AB_val AD_val AP_val : ℝ × ℝ × ℝ)

-- Given conditions
def AB := (4 : ℝ, -2, 3)
def AD := (-4 : ℝ, 1, 0)
def AP := (-6 : ℝ, 2, -8)

-- Heights
def height (P : Point) (plane : Vect3 × Vect3) := EuclideanGeometry.dist P (plane.1, plane.2)

-- Statement to prove
theorem height_from_point_to_plane_eq_2 (h : P.distance (plane_of_points A B D) = 2) : 
  height P (AB, AD) = 2 := 
sorry

end height_from_point_to_plane_eq_2_l691_691840


namespace find_t_l691_691061

theorem find_t (t : ℝ) (h : sqrt (3 * sqrt (t - 3)) = (10 - t)^(1/4)) : t = 3.7 :=
sorry

end find_t_l691_691061


namespace all_logarithmic_functions_are_monotonic_exists_integer_divisible_by_2_and_5_exists_real_x_with_log_base_2_greater_than_0_l691_691710

open Real Int

/-- Prove that all logarithmic functions are monotonic -/
theorem all_logarithmic_functions_are_monotonic :
  ∀ a : ℝ, (λ x : ℝ, log a x) is Monotonic :=
  sorry

/-- Prove that there exists an integer that is divisible by both 2 and 5 -/
theorem exists_integer_divisible_by_2_and_5 :
  ∃ x : ℤ, 2 ∣ x ∧ 5 ∣ x :=
  sorry

/-- Prove that there exists an x in the real numbers such that log base 2 of x is greater than 0 -/
theorem exists_real_x_with_log_base_2_greater_than_0 :
  ∃ x : ℝ, log 2 x > 0 :=
  sorry

end all_logarithmic_functions_are_monotonic_exists_integer_divisible_by_2_and_5_exists_real_x_with_log_base_2_greater_than_0_l691_691710


namespace FI_squared_l691_691517

theorem FI_squared (ABCD : Type*)
  [metric_space ABCD] [normed_group ABCD] [normed_space ℝ ABCD]
  (A B C D E H F G I J : ABCD)
  (area_AEH : ℝ) (area_BFIE : ℝ) (area_DHJG : ℝ) (area_FCGJI : ℝ)
  (side_length_sq : ℝ)
  (h1 : AE = AH)
  (h2 : ⊥ FI EH)
  (h3 : ⊥ GJ EH)
  (h4 : area_AEH = 4)
  (h5 : area_BFIE = 5)
  (h6 : area_DHJG = 6)
  (h7 : area_FCGJI = 3)
  (side_length : side_length_sq = 18) :
  FI^2 = 12 :=
sorry

end FI_squared_l691_691517


namespace problem_4_4_l691_691898

-- Define the polar coordinate equation condition
def polar_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

-- Define the parametric equations of the line
def parametric_line (t θ : ℝ) : ℝ × ℝ := (2 + t * Real.cos θ, 1 + t * Real.sin θ)

-- Define the rectangular coordinate equation of the curve C1
def rectangular_equation (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 4

-- Define the rectangular coordinates M
def point_M (x y : ℝ) : Prop := x = 2 ∧ y = 1

-- Assertion that we need to prove based on given conditions
theorem problem_4_4
  (ρ θ x y t_A t_B : ℝ)
  (h1 : polar_equation ρ θ)
  (h2 : (x, y) = (ρ * Real.cos θ, ρ * Real.sin θ))
  (h3 : parametric_line t_A θ = (2 + t_A * Real.cos θ, 1 + t_A * Real.sin θ))
  (h4 : parametric_line t_B θ = (2 + t_B * Real.cos θ, 1 + t_B * Real.sin θ))
  (h5 : point_M 2 1)
  (ha : t_A = -2 * t_B) :
  rectangular_equation x y ∧
  (∃ θ, θ = Real.asin (sqrt 6 / 4) ∨ θ = -Real.asin (sqrt 6 / 4) ∨ θ = Real.acos (sqrt 10 / 4) ∨ θ = -Real.acos (sqrt 10 / 4)) ∧
  (∀ x y, (y = -sqrt 15 / 5 * x + 2 * sqrt 15 / 5 + 1) ∨ (y = sqrt 15 / 5 * x - 2 * sqrt 15 / 5 + 1)) :=
by
  -- We are skipping the proof as instructed
  sorry

end problem_4_4_l691_691898


namespace Razorback_total_revenue_l691_691904

def t_shirt_price : ℕ := 51
def t_shirt_discount : ℕ := 8
def hat_price : ℕ := 28
def hat_discount : ℕ := 5
def t_shirts_sold : ℕ := 130
def hats_sold : ℕ := 85

def discounted_t_shirt_price : ℕ := t_shirt_price - t_shirt_discount
def discounted_hat_price : ℕ := hat_price - hat_discount

def revenue_from_t_shirts : ℕ := t_shirts_sold * discounted_t_shirt_price
def revenue_from_hats : ℕ := hats_sold * discounted_hat_price

def total_revenue : ℕ := revenue_from_t_shirts + revenue_from_hats

theorem Razorback_total_revenue : total_revenue = 7545 := by
  unfold total_revenue
  unfold revenue_from_t_shirts
  unfold revenue_from_hats
  unfold discounted_t_shirt_price
  unfold discounted_hat_price
  unfold t_shirts_sold
  unfold hats_sold
  unfold t_shirt_price
  unfold t_shirt_discount
  unfold hat_price
  unfold hat_discount
  sorry

end Razorback_total_revenue_l691_691904


namespace lambda_range_l691_691425

theorem lambda_range (a : ℕ → ℝ) (λ : ℝ)
  (h1 : ∀ n : ℕ, 0 < n → a n = n^2 + λ * n)
  (h2 : ∀ n : ℕ, 0 < n → a (n + 1) > a n)
  : λ > -3 
  :=
sorry

end lambda_range_l691_691425


namespace beth_sold_l691_691360

theorem beth_sold {initial_coins additional_coins total_coins sold_coins : ℕ} 
  (h_init : initial_coins = 125)
  (h_add : additional_coins = 35)
  (h_total : total_coins = initial_coins + additional_coins)
  (h_sold : sold_coins = total_coins / 2) :
  sold_coins = 80 := 
sorry

end beth_sold_l691_691360


namespace P_on_QR_l691_691882

theorem P_on_QR (P Q R : ℝ × ℝ) 
  (h : ∀ X : ℝ × ℝ, dist P X < dist Q X ∨ dist P X < dist R X) : 
  ∃ t ∈ Icc (0:ℝ) 1, P = t • Q + (1 - t) • R :=
by
  -- Proof
  sorry  -- Proof omitted

end P_on_QR_l691_691882


namespace lunch_break_duration_l691_691885

theorem lunch_break_duration :
  ∃ (L : ℝ), 0.6 * 60 = L ∧ 
  ∀ (p d m : ℝ),
    (p + d + m) * (8 - (L / 60)) = 0.6 ∧
    (d + m) * (4 - (L / 60)) = 0.2 ∧
    p * (9 - (L / 60)) = 0.2 :=
by
  let L := 36
  use L
  split
  { norm_num }
  intros p d m
  split
  { norm_cast, sorry }
  split
  { norm_cast, sorry }
  { norm_cast, sorry }

end lunch_break_duration_l691_691885


namespace Hanna_erasers_l691_691790

theorem Hanna_erasers :
  ∀ (H R T : ℕ), 
    (H = 2 * R) →
    (R = 5 - 3) →
    (T = 20) →
    (T / 2 = 10) →
    H = 4 :=
by
  intros H R T
  assume h1 h2 h3 h4
  have tanya_red : ℕ := T / 2
  have rachel_erasers : ℕ := tanya_red / 2 - 3
  rw [h2] at rachel_erasers
  rw [h3, h4] at tanya_red
  have hanna_erasers := 2 * rachel_erasers
  rw [h1]
  sorry

end Hanna_erasers_l691_691790


namespace value_of_2019_tenths_l691_691286

-- Define the concept of "tenth"
def tenth (x : ℝ) : ℝ := x * (1 / 10)

-- Prove that 2019 tenths equals 201.9
theorem value_of_2019_tenths : tenth 2019 = 201.9 :=
by
  sorry

end value_of_2019_tenths_l691_691286


namespace find_smallest_n_modulo_l691_691735

theorem find_smallest_n_modulo :
  ∃ n : ℕ, n > 0 ∧ (2007 * n) % 1000 = 837 ∧ n = 691 :=
by
  sorry

end find_smallest_n_modulo_l691_691735


namespace correlation_coefficient_approx_advertising_investment_to_exceed_70k_l691_691822

/-
Definitions based on conditions in part (a):
- Monthly advertising investment and sales data
- Given sums and square roots
- Definition of linear regression

Proof problem statement:
- Prove the correlation coefficient is approximately 0.99.
- Prove the required monthly advertising investment to exceed specified sales volume.
-/

def monthly_advertising_investment_and_sales : List (ℕ × ℕ) :=
  [(1, 28), (2, 32), (3, 35), (4, 45), (5, 49), (6, 52), (7, 60)]

def sum_of_products : ℝ := 150
def sum_of_squares_y : ℝ := 820
def sqrt_1435_approx : ℝ := 37.88

noncomputable def regression_line : ℝ → ℝ := λ x, (75 / 14) * x + (151 / 7)

theorem correlation_coefficient_approx :
  let n := (monthly_advertising_investment_and_sales.length : ℝ),
      x_bar := 4,
      y_bar := 43 in 
  let r := sum_of_products / (sqrt 28 * sqrt 820) in
  |r - 0.99| < 0.01 := by
      -- skipped proof
  sorry

theorem advertising_investment_to_exceed_70k :
  ∃ x, regression_line x > 70 :=
  by
  -- skipped proof
  sorry

end correlation_coefficient_approx_advertising_investment_to_exceed_70k_l691_691822


namespace brian_shirts_are_correct_l691_691576

variable (Shirts : Type) [DivisionRing Shirts] [DecidableEq Shirts] [AddMonoid Shirts] 
variable (Steven_shirts Andrew_shirts Brian_shirts : Shirts)

constant (has_four_times_as_many : Steven_shirts = 4 * Andrew_shirts)
constant (has_six_times_as_many : Andrew_shirts = 6 * Brian_shirts)
constant (steven_has : Steven_shirts = 72)

theorem brian_shirts_are_correct :
  ∃ (Brian_shirts : Shirts), Steven_shirts = 72 ∧ Steven_shirts = 4 * Andrew_shirts ∧ Andrew_shirts = 6 * Brian_shirts → Brian_shirts = 3 :=
by
  sorry

end brian_shirts_are_correct_l691_691576


namespace matrix_power_application_l691_691121

variable (B : Matrix (Fin 2) (Fin 2) ℝ)
variable (v : Fin 2 → ℝ := ![4, -3])

theorem matrix_power_application :
  (B.mulVec v = ![8, -6]) →
  (B ^ 4).mulVec v = ![64, -48] :=
by
  intro h
  sorry

end matrix_power_application_l691_691121


namespace simplify_expression_l691_691170

variable (x : ℝ)

theorem simplify_expression : (2 * x + 20) + (150 * x + 20) = 152 * x + 40 :=
by
  sorry

end simplify_expression_l691_691170


namespace derivative_f_l691_691398

noncomputable def f (x : ℝ) : ℝ :=
  sqrt x + (1 / 3) * arctan (sqrt x) + (8 / 3) * arctan (sqrt x / 2)

theorem derivative_f (x : ℝ) (h : x > 0) :
  deriv f x = (3 * x^2 + 16 * x + 32) / (6 * sqrt x * (x + 1) * (x + 4)) :=
by sorry

end derivative_f_l691_691398


namespace part_a_part_b_l691_691849

noncomputable def triangle_properties (a b c: ℝ) (h: ∃ A B C: ℝ × ℝ, 
  ∠BAC = 90 ∧ BC = a ∧ AC = b ∧ AB = c ∧
  ∃ I: ℝ × ℝ, is_incenter I ABC ∧ 
  ∃ d: ℝ → ℝ × ℝ, passes_through I d ∧ 
  ∃ P: ℝ × ℝ, intersects d AB P ∧ 
  ∃ Q: ℝ × ℝ, intersects d AC Q) : Prop :=
  b * (PB / PA) + c * (QC / QA) = a

theorem part_a (a b c: ℝ) (h: ∃ A B C: ℝ × ℝ, 
  ∠BAC = 90 ∧ BC = a ∧ AC = b ∧ AB = c ∧
  ∃ I: ℝ × ℝ, is_incenter I ABC ∧ 
  ∃ d: ℝ → ℝ × ℝ, passes_through I d ∧ 
  ∃ P: ℝ × ℝ, intersects d AB P ∧ 
  ∃ Q: ℝ × ℝ, intersects d AC Q
) : b * (PB / PA) + c * (QC / QA) = a := 
by 
  -- proof here
  sorry

noncomputable def minimize_expression (a b c: ℝ) (h: ∃ A B C: ℝ × ℝ, 
  ∠BAC = 90 ∧ BC = a ∧ AC = b ∧ AB = c ∧
  ∃ I: ℝ × ℝ, is_incenter I ABC ∧ 
  ∃ d: ℝ → ℝ × ℝ, passes_through I d ∧ 
  ∃ P: ℝ × ℝ, intersects d AB P ∧ 
  ∃ Q: ℝ × ℝ, intersects d AC Q) : ℝ :=
  1

theorem part_b (a b c: ℝ) (h: ∃ A B C: ℝ × ℝ, ∠BAC = 90 ∧ BC = a ∧ AC = b ∧ AB = c ∧ 
  ∃ I: ℝ × ℝ, is_incenter I ABC ∧ 
  ∃ d: ℝ → ℝ × ℝ, passes_through I d ∧ 
  ∃ P: ℝ × ℝ, intersects d AB P ∧ 
  ∃ Q: ℝ × ℝ, intersects d AC Q
) : minimize_expression a b c h = 1 := 
by 
  -- proof here
  sorry

end part_a_part_b_l691_691849


namespace min_intersection_points_l691_691605

theorem min_intersection_points (n : ℕ) (N : ℕ) (h_n : n = 2008) (hTangent : ∀ i j, i ≠ j → ¬ tangent(i, j))
  (hIntersects : ∀ i, ∃ c1 c2 c3, intersects i c1 ∧ intersects i c2 ∧ intersects i c3) : N = 3012 :=
sorry

end min_intersection_points_l691_691605


namespace actual_average_height_correct_l691_691580

-- Define the conditions
def average_height_initial : ℝ := 185
def number_of_boys : ℝ := 75

-- Recorded and actual heights in cm
def recorded_height_boy1 : ℝ := 170
def actual_height_boy1 : ℝ := 140

def recorded_height_boy2 : ℝ := 195
def actual_height_boy2 : ℝ := 165

def recorded_height_boy3 : ℝ := 160
def actual_height_boy3 : ℝ := 190

-- Conversion from inches to cm
def recorded_height_boy4_inches : ℝ := 70
def actual_height_boy4_inches : ℝ := 64
def inch_to_cm : ℝ := 2.54

def recorded_height_boy4 : ℝ := recorded_height_boy4_inches * inch_to_cm
def actual_height_boy4 : ℝ := actual_height_boy4_inches * inch_to_cm

-- Total initial height
def total_initial_height : ℝ := number_of_boys * average_height_initial

-- Total difference in height correction
def total_difference : ℝ := 
  (recorded_height_boy1 - actual_height_boy1) + 
  (recorded_height_boy2 - actual_height_boy2) + 
  (actual_height_boy3 - recorded_height_boy3) +
  (recorded_height_boy4 - actual_height_boy4)

-- Corrected total height
def total_corrected_height : ℝ := total_initial_height - total_difference

-- The actual average height 
def actual_average_height := total_corrected_height / number_of_boys

-- The Lean 4 statement to express the proof problem
theorem actual_average_height_correct : 
  actual_average_height ≈ 184.00 :=
by
  sorry

end actual_average_height_correct_l691_691580


namespace faculty_after_reduction_l691_691675

-- Define the original number of faculty members
def originalFaculty : ℕ := 260
-- Define the reduction percentage (as a fraction)
def reductionPercentage : ℚ := 0.25
-- Define the expected number of faculty members after reduction
def expectedFacultyAfterReduction : ℕ := 195

-- Theorem stating the number of faculty members after the reduction
theorem faculty_after_reduction : 
  originalFaculty - (originalFaculty * reductionPercentage).natAbs = expectedFacultyAfterReduction := by
  sorry

end faculty_after_reduction_l691_691675


namespace cafe_table_count_l691_691182

theorem cafe_table_count (chairs_7 : ℕ) (chairs_10 : ℕ) (people_per_table : ℕ) (tables_needed : ℕ) :
  chairs_7 = 310 ∧ chairs_10 = 3 * 7^2 + 1 * 7^1 + 0 * 7^0 ∧ people_per_table = 3 →
  tables_needed = Int.ceil (chairs_10 / people_per_table) :=
by
  intros
  sorry

end cafe_table_count_l691_691182


namespace reeya_average_l691_691298

theorem reeya_average (s1 s2 s3 s4 s5 : ℕ) (h1 : s1 = 65) (h2 : s2 = 67) (h3 : s3 = 76) (h4 : s4 = 82) (h5 : s5 = 85) :
  (s1 + s2 + s3 + s4 + s5) / 5 = 75 := by
  sorry

end reeya_average_l691_691298


namespace half_absolute_difference_of_squares_l691_691267

-- Defining the variables a and b involved in the problem
def a : ℤ := 20
def b : ℤ := 15

-- Statement to prove the solution
theorem half_absolute_difference_of_squares : 
    (1 / 2 : ℚ) * (abs ((a ^ 2) - (b ^ 2))) = 87.5 :=
by
    -- Proof omitted
    sorry

end half_absolute_difference_of_squares_l691_691267


namespace find_t_l691_691056

theorem find_t (t : ℝ) (h : real.sqrt (3 * real.sqrt (t - 3)) = real.root 4 (10 - t)) : t = 3.7 := sorry

end find_t_l691_691056


namespace f_119_5_l691_691135

noncomputable def f : ℝ → ℝ :=
  sorry

-- Definition of even function
axiom f_even : ∀ x : ℝ, f (-x) = f x

-- Given functional equation
axiom f_eqn : ∀ x : ℝ, f x = -1 / f (x - 3)

-- Specific interval definition
axiom f_interval : ∀ x : ℝ, x ∈ set.Icc (-3) (-2) → f x = 4 * x

/-- Prove that f(119.5) = 1 / 10 given the conditions on f. -/
theorem f_119_5 : f 119.5 = 1 / 10 :=
  sorry

end f_119_5_l691_691135


namespace sin_alpha_half_plus_cos_alpha_half_l691_691754

theorem sin_alpha_half_plus_cos_alpha_half {α : ℝ} (h₁ : sin α = 1 / 3) (h₂ : 2 * π < α) (h₃ : α < 3 * π) : 
  sin (α / 2) + cos (α / 2) = - (2 * sqrt 3) / 3 := 
by 
  sorry

end sin_alpha_half_plus_cos_alpha_half_l691_691754


namespace cos_double_angle_l691_691074

theorem cos_double_angle : 
  (∑ n in Finset.range 1000, (cos θ)^(2 * n) = 10) → cos (2 * θ) = 4 / 5 :=
by
  sorry

end cos_double_angle_l691_691074


namespace students_with_B_l691_691084

theorem students_with_B (students_jacob : ℕ) (students_B_jacob : ℕ) (students_smith : ℕ) (ratio_same : (students_B_jacob / students_jacob : ℚ) = 2 / 5) : 
  ∃ y : ℕ, (y / students_smith : ℚ) = 2 / 5 ∧ y = 12 :=
by 
  use 12
  sorry

end students_with_B_l691_691084


namespace area_quadrilateral_BEFC_l691_691644

/-- 
Given:
1. An isosceles triangle ABC with AB = AC = 3 cm and BC = 4 cm.
2. Extend BC to D such that CD = BC.
3. E is the midpoint of AB.
4. A circle with center E passing through B intersects AC at point F above C.
Prove:
The area of quadrilateral BEFC is approximately 3.27 cm².
-/
theorem area_quadrilateral_BEFC :
  ∀ (A B C D E F : Point) (radius : ℝ)
  (h_isosceles : AB = AC)
  (h_AB_AC_eq_3 : AB = 3)
  (h_BC_eq_4 : BC = 4)
  (h_CD_eq_BC : CD = BC)
  (h_E_midpoint_AB : E = midpoint A B)
  (h_circle_center_E_pass_B : distance E B = radius)
  (h_F_on_circle : distance E F = radius)
  (h_F_on_AC : F ∈ line_through A C),
  area_quadrilateral B E F C ≈ 3.27 :=
begin
  -- Definitions of points and segments based on the conditions
  let A := Point.mk 0 0,
  let B := Point.mk 3 0,
  let C := Point.mk (-3/2) (2 * sqrt 5),
  let D := Point.mk (3/2) (2 * sqrt 5),
  let E := midpoint A B,
  let F := ???

  -- Constraints from the conditions
  have h1 : distance A B = 3 := by sorry,
  have h2 : distance B C = 4 := by sorry,
  have h3 : distance A C = 3 := by sorry,
  have h4 : midpoint A B = E := by sorry,
  have h5 : distance E B = 3 / 2 := by sorry,
  have h6 : distance E F = 3 / 2 := by sorry,
  have h7 : F ∈ line_through A C := by sorry,

  -- Claim 
  claim_area : area_quadrilateral B E F C ≈ 3.27 := sorry,
end

end area_quadrilateral_BEFC_l691_691644


namespace cars_sold_on_third_day_l691_691980

theorem cars_sold_on_third_day (a b total c : ℕ) (h1 : a = 14) (h2 : b = 16) (h3 : total = 57) :
  c = total - (a + b) → c = 27 :=
by
  intros h
  rw [h1, h2, h3]
  rw [Nat.add_comm, Nat.add_assoc, Nat.sub_add_cancel]
  exact h
  { rw [Nat.le_add_left] sorry }

end cars_sold_on_third_day_l691_691980


namespace total_liters_needed_to_fill_two_tanks_l691_691187

theorem total_liters_needed_to_fill_two_tanks (capacity : ℕ) (liters_first_tank : ℕ) (liters_second_tank : ℕ) (percent_filled : ℕ) :
  liters_first_tank = 300 → 
  liters_second_tank = 450 → 
  percent_filled = 45 → 
  capacity = (liters_second_tank * 100) / percent_filled → 
  1000 - 300 = 700 → 
  1000 - 450 = 550 → 
  700 + 550 = 1250 :=
by sorry

end total_liters_needed_to_fill_two_tanks_l691_691187


namespace hyperbola_standard_eqn1_hyperbola_standard_eqn2_l691_691405

noncomputable def hyperbola_eq1 : Prop :=
  ∃ (a b : ℝ), 
    a ≠ 0 ∧ b ≠ 0 ∧ 
    (∀ (x y : ℝ), (y^2 / a^2) - (x^2 / b^2) = 1 → x^2 / 27 + y^2 / 36 = 1) ∧ 
    ((sqrt 15)^2 * b^2 = 1 ∧ 4^2 * a^2 = 1) ∧ 
    (a, b) = (2, sqrt 5) ∧
    (y^2 / 4) - (x^2 / 5) = 1

theorem hyperbola_standard_eqn1 : hyperbola_eq1 := 
  sorry

noncomputable def hyperbola_eq2 : Prop :=
  ∃ (a b : ℝ), 
    a ≠ 0 ∧ b ≠ 0 ∧ 
    (∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 → 2 * x ± 3 * y = 0) ∧ 
    ((sqrt 6)^2 * a^2 = 1 ∧ 2^2 * b^2 = 1) ∧ 
    (a, b) = (3, sqrt 4/3) ∧
    (3y^2 / 4) - (x^2 / 3) = 1 

theorem hyperbola_standard_eqn2 : hyperbola_eq2 := 
  sorry

end hyperbola_standard_eqn1_hyperbola_standard_eqn2_l691_691405


namespace rate_of_current_l691_691209

theorem rate_of_current (c : ℝ) (speed_still_water : ℝ) (distance_downstream : ℝ) (time_minutes : ℝ) 
  (h1 : speed_still_water = 42)
  (h2 : distance_downstream = 34.47)
  (h3 : time_minutes = 44) :
  c = 5.005 :=
by
  let time_hours := time_minutes / 60
  have time_hours_eq : time_hours = 11 / 15 := by 
    dsimp [time_hours, h3] 
    norm_num
  let effective_speed := speed_still_water + c
  have effective_speed_eq : effective_speed = 42 + c := by 
    rw [h1] 
    rfl
  have eqn := distance_downstream = effective_speed * time_hours
  rw [effective_speed_eq, time_hours_eq] at eqn
  sorry

end rate_of_current_l691_691209


namespace matrix_transformation_l691_691394

variable (a b c d e f g h i : ℝ)
def M := Matrix.of ![
  ![3, 0, 0],
  ![0, 0, 1],
  ![0, 1, 0]
]
def N := Matrix.of ![
  ![a, b, c],
  ![d, e, f],
  ![g, h, i]
]
def Target := Matrix.of ![
  ![3 * a, 3 * b, 3 * c],
  ![g, h, i],
  ![d, e, f]
]

theorem matrix_transformation :
  M ⬝ N = Target :=
sorry

end matrix_transformation_l691_691394


namespace parallel_lines_regular_ngon_l691_691701

def closed_n_hop_path (n : ℕ) (a : Fin (n + 1) → Fin n) : Prop :=
∀ i j : Fin n, a (i + 1) + a i = a (j + 1) + a j → i = j

theorem parallel_lines_regular_ngon (n : ℕ) (a : Fin (n + 1) → Fin n):
  (Even n → ∃ i j : Fin n, i ≠ j ∧ a (i + 1) + a i = a (j + 1) + a j) ∧
  (Odd n → ¬(∃ i j : Fin n, i ≠ j ∧ a (i + 1) + a i = a (j + 1) + a j ∧ ∀ k l : Fin n, k ≠ l → a (k + 1) + k ≠ a (l + 1) + l)) :=
by
  sorry

end parallel_lines_regular_ngon_l691_691701


namespace find_sum_l691_691341

def sumInterest (P R : ℝ) : ℝ := (P * R * 10) / 100

theorem find_sum (R : ℝ): 
  (sumInterest 200 R) = (sumInterest 200 (R + 15)) - 300 :=
by
  sorry

end find_sum_l691_691341


namespace traffic_safety_team_eq_twice_fire_l691_691832

-- Define initial members in the teams
def t0 : ℕ := 8
def f0 : ℕ := 7

-- Define the main theorem
theorem traffic_safety_team_eq_twice_fire (x : ℕ) : t0 + x = 2 * (f0 - x) :=
by sorry

end traffic_safety_team_eq_twice_fire_l691_691832


namespace arrangement_methods_count_l691_691153

-- Define the original and additional programs
def original_programs : ℕ := 4
def additional_programs : ℕ := 2
def slots : ℕ := original_programs + 1

-- Define the proposition that proves the arrangement methods equal to 30
theorem arrangement_methods_count : 
  (Nat.choose (slots + additional_programs - 1) additional_programs) * factorial(additional_programs) = 30 :=
by
  sorry

end arrangement_methods_count_l691_691153


namespace not_perfect_square_l691_691920

theorem not_perfect_square (n : ℤ) : ¬ ∃ (m : ℤ), 4*n + 3 = m^2 := 
by 
  sorry

end not_perfect_square_l691_691920


namespace Q_returns_distance_l691_691983

noncomputable def Q_travel_distance (a b c r : ℝ) (triangle_perimeter : ℝ) : ℝ :=
  let k := (triangle_perimeter - 3 * r) / triangle_perimeter 
  let new_perimeter := k * triangle_perimeter 
  in new_perimeter

theorem Q_returns_distance :
  Q_travel_distance 13 14 15 2 42 = 220 / 7 := by
  sorry

end Q_returns_distance_l691_691983


namespace angle_between_vectors_cosine_l691_691854

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

theorem angle_between_vectors_cosine (h : ∥a - b∥ = ∥b∥) :
  let θ := real.angle a (a + 3 • b) in
  real.cos (θ) = (5 * (inner a b)) / (∥a∥ * ∥a + 3 • b∥) :=
by sorry

end angle_between_vectors_cosine_l691_691854


namespace find_x_l691_691070

theorem find_x : 
  ∀ x : ℝ, (1 / (x + 4) + 1 / (x - 4) = 1 / (x - 4)) → x = 1 / 2 := 
by 
  sorry

end find_x_l691_691070


namespace product_sequence_l691_691300

theorem product_sequence :
  ∏ k in Finset.range 98, (k * (k + 2)) / ((k + 1) ^ 2) = 50 / 99 :=
  sorry

end product_sequence_l691_691300


namespace scenery_photos_correct_l691_691141

-- Define the problem conditions
def animal_photos := 10
def flower_photos := 3 * animal_photos
def photos_total := 45
def scenery_photos := flower_photos - 10

-- State the theorem
theorem scenery_photos_correct : scenery_photos = 20 ∧ animal_photos + flower_photos + scenery_photos = photos_total := by
  sorry

end scenery_photos_correct_l691_691141


namespace rectangle_ratio_constant_l691_691205

theorem rectangle_ratio_constant (length width : ℝ) (d k : ℝ)
  (h1 : length/width = 5/2)
  (h2 : 2 * (length + width) = 28)
  (h3 : d^2 = length^2 + width^2)
  (h4 : (length * width) = k * d^2) :
  k = (10/29) := by
  sorry

end rectangle_ratio_constant_l691_691205


namespace fraction_pow_four_result_l691_691948

theorem fraction_pow_four_result (x : ℚ) (h : x = 1 / 4) : x ^ 4 = 390625 / 100000000 :=
by sorry

end fraction_pow_four_result_l691_691948


namespace integer_part_power_divisible_l691_691551

theorem integer_part_power_divisible (k : ℕ) (hk : 0 < k) :
  ∀ n : ℕ, k ∣ ⌊(k + 1/2 + real.sqrt (k^2 + 1/4))^n⌋ := 
sorry

end integer_part_power_divisible_l691_691551


namespace minimum_value_expression_l691_691329

theorem minimum_value_expression (a b c h1 h2 h3 k1 k2 k3 S : ℝ) (α : ℝ) 
  (M_inside_triangle : ∀ x y z : ℝ, 0 < x ∧ 0 < y ∧ 0 < z -> 0 < x + y + z) 
  (height_condition : 2 * S = a * h1 ∧ 2 * S = b * h2 ∧ 2 * S = c * h3) 
  (distance_condition : 2 * S = a * k1 + b * k2 + c * k3) 
  (alpha_condition : 1 ≤ α) :
  (\left(\frac{k1}{h1}\right)^α + \left(\frac{k2}{h2}\right)^α + \left(\frac{k3}{h3}\right)^α) ≥ \frac{1}{3^(α-1)} :=
sorry

end minimum_value_expression_l691_691329


namespace count_odd_numbers_l691_691477

theorem count_odd_numbers :
  let start := 251
  let end := 549
  ∃ n : ℕ, n = 150 ∧ ∀ k, 0 ≤ k ∧ k ≤ 149 → start + 2 * k = end → n = k + 1 :=
by
  sorry

end count_odd_numbers_l691_691477


namespace probability_correct_l691_691712

-- Define the problem conditions.
def num_balls : ℕ := 8
def possible_colors : ℕ := 2

-- Probability calculation for a specific arrangement (either configuration of colors).
def probability_per_arrangement : ℚ := (1/2) ^ num_balls

-- Number of favorable arrangements with 4 black and 4 white balls.
def favorable_arrangements : ℕ := Nat.choose num_balls 4

-- The required probability for the solution.
def desired_probability : ℚ := favorable_arrangements * probability_per_arrangement

-- The proof statement to be provided.
theorem probability_correct :
  desired_probability = 35 / 128 := 
by
  sorry

end probability_correct_l691_691712


namespace least_possible_value_of_smallest_integer_l691_691299

theorem least_possible_value_of_smallest_integer :
  ∀ (A B C D : ℕ), A < B → B < C → C < D → (A + B + C + D) / 4 = 70 → D = 90 → A ≥ 13 :=
by
  intros A B C D h₁ h₂ h₃ h₄ h₅
  sorry

end least_possible_value_of_smallest_integer_l691_691299


namespace number_of_mappings_l691_691473

-- Define the sets A and B
def A : Finset ℝ := {a | ∃ i : Fin 100, a = a i}.to_finset
def B : Finset ℝ := {b | ∃ i : Fin 50, b = b i}.to_finset

-- Define the mapping condition
def mapping_condition (f : ℝ → ℝ) : Prop :=
  ∀ a1 a2 ∈ A, a1 ≤ a2 → f a1 ≤ f a2 ∧ ∀ b ∈ B, ∃ a ∈ A, f a = b

-- Prove the number of such mappings
theorem number_of_mappings : ∃ (f : ℝ → ℝ), mapping_condition f →
  ∃ (ways : ℤ), ways = nat.choose 99 49 :=
sorry

end number_of_mappings_l691_691473


namespace min_x_plus_y_l691_691443

-- Define the conditions given in the problem
variables (x y : ℝ)
hypothesis h1 : xy = 2 * x + y + 2
hypothesis h2 : x > 1

-- State the theorem to prove
theorem min_x_plus_y : ∃ (m : ℝ), (∀ x y : ℝ, xy = 2 * x + y + 2 → x > 1 → x + y ≥ m) ∧ m = 7 :=
by
  sorry

end min_x_plus_y_l691_691443


namespace q_value_l691_691046

theorem q_value (p q : ℝ) (hpq1 : 1 < p) (hpql : p < q) (hq_condition : (1 / p) + (1 / q) = 1) (hpq2 : p * q = 8) : q = 4 + 2 * Real.sqrt 2 :=
  sorry

end q_value_l691_691046


namespace find_x_l691_691490

theorem find_x
  (p q : ℝ)
  (h1 : 3 / p = x)
  (h2 : 3 / q = 18)
  (h3 : p - q = 0.33333333333333337) :
  x = 6 :=
sorry

end find_x_l691_691490


namespace area_of_triangle_AEF_l691_691522

theorem area_of_triangle_AEF {A B C E F G H : Type} [euclidean_space A] [inhabited A] 
  (is_isosceles : ∃ x : A, x = B ∧ x = C) 
  (AB_eq_AC : AB = AC) 
  (BC_eq_30 : BC = 30)
  (square EFGH : inscribed_square_in_triangle)  
  (side_length_square : EF = 12) : 
  ∃ area : ℝ, area = 48 := 
sorry

end area_of_triangle_AEF_l691_691522


namespace Roe_total_savings_l691_691894

-- Define savings amounts per period
def savings_Jan_to_Jul : Int := 7 * 10
def savings_Aug_to_Nov : Int := 4 * 15
def savings_Dec : Int := 20

-- Define total savings for the year
def total_savings : Int := savings_Jan_to_Jul + savings_Aug_to_Nov + savings_Dec

-- Prove that Roe's total savings for the year is $150
theorem Roe_total_savings : total_savings = 150 := by
  -- Proof goes here
  sorry

end Roe_total_savings_l691_691894


namespace sum_inequality_l691_691433

theorem sum_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : sqrt x + sqrt y + sqrt z = 1) : 
  (x^2 + y * z) / sqrt (2 * x^2 * (y + z)) + 
  (y^2 + z * x) / sqrt (2 * y^2 * (z + x)) + 
  (z^2 + x * y) / sqrt (2 * z^2 * (x + y)) ≥ 1 := 
by 
  sorry

end sum_inequality_l691_691433


namespace general_formula_a_general_formula_b_l691_691122

noncomputable def S : ℕ → ℕ
| 1 => 1
| n+1 => S n + (2 * (n + 1) - 1)

noncomputable def a : ℕ → ℕ
| 1 => 1
| n+1 => 2 * (n + 1) - 1

noncomputable def b : ℕ → ℕ
| 1 => 1
| n+1 => let b_n := b n in b_n + (2 * n - 1) * 2^(n-1)

theorem general_formula_a (n : ℕ) : a n = 2 * n - 1 := by
  -- Proof goes here
  sorry

theorem general_formula_b (n : ℕ) : b n = (2 * n - 5) * 2^(n-1) + 4 := by
  -- Proof goes here
  sorry

end general_formula_a_general_formula_b_l691_691122


namespace mosquito_shadow_speed_l691_691668

theorem mosquito_shadow_speed
  (v : ℝ) (t : ℝ) (h : ℝ) (cos_theta : ℝ) (v_shadow : ℝ)
  (hv : v = 0.5) (ht : t = 20) (hh : h = 6) (hcos_theta : cos_theta = 0.6) :
  v_shadow = 0 ∨ v_shadow = 0.8 :=
  sorry

end mosquito_shadow_speed_l691_691668


namespace skew_lines_parallelepiped_l691_691319

-- Define the problem conditions
def parallelepiped (V : Finset Point) : Prop :=
  V.card = 8 ∧ -- V is a set of 8 vertices
  ∀ p q : Point, p ∈ V ∧ q ∈ V → p ≠ q → Line p q

-- Number of ways to choose 4 vertices out of 8
def C_8_4 : ℕ := Nat.choose 8 4

-- Number of pairs of skew lines
def skew_line_pairs : ℕ := 3 * (C_8_4 - 12)

-- The theorem stating the problem
theorem skew_lines_parallelepiped (V : Finset Point) (hV : parallelepiped V) : skew_line_pairs = 174 :=
by
  sorry

end skew_lines_parallelepiped_l691_691319


namespace legs_heads_multiple_l691_691509

theorem legs_heads_multiple (D C : ℕ) : 
  let L := 2 * D + 4 * C + 24 in
  let H := D + C + 6 in
  L = 2 * H + 12 :=
by
  let buffalo_legs := 4 * 6
  have L_eq : L = 2 * D + 4 * C + buffalo_legs := by rfl
  let buffalo_heads := 6
  have H_eq : H = D + C + buffalo_heads := by rfl
  have leg_condition : ∀ m, L = m * H + 12 → m = 2 := sorry
  exact leg_condition 2 (by rw [L_eq, H_eq, mul_add, add_assoc, mul_assoc, add_right_comm, add_sub_cancel_left])


end legs_heads_multiple_l691_691509


namespace half_abs_diff_of_squares_l691_691251

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 20) (h2 : b = 15) :
  (1/2 : ℚ) * |(a:ℚ)^2 - (b:ℚ)^2| = 87.5 := by
  sorry

end half_abs_diff_of_squares_l691_691251


namespace bread_last_days_l691_691607

def total_consumption_per_member_breakfast : ℕ := 4
def total_consumption_per_member_snacks : ℕ := 3
def total_consumption_per_member : ℕ := total_consumption_per_member_breakfast + total_consumption_per_member_snacks
def family_members : ℕ := 6
def daily_family_consumption : ℕ := family_members * total_consumption_per_member
def slices_per_loaf : ℕ := 10
def total_loaves : ℕ := 5
def total_bread_slices : ℕ := total_loaves * slices_per_loaf

theorem bread_last_days : total_bread_slices / daily_family_consumption = 1 :=
by
  sorry

end bread_last_days_l691_691607


namespace eccentricity_squared_l691_691026

def ellipse_eccentricity (a b c : ℝ) (h : a > b ∧ b > 0) : ℝ :=
  sqrt (1 - b^2 / a^2)

theorem eccentricity_squared (a b : ℝ) (h : a > b ∧ b > 0)
  (h1 : ∃ P : ℝ × ℝ, P.fst = c ∧ P.snd = b^2 / a)
  (h2 : ∃ Q : ℝ × ℝ, Q.fst = c / 3 ∧ Q.snd = 2 * b^2 / (3 * a))
  (h3 : let F1P := (2 * c, b^2 / a), F2Q := (-2 * c / 3, 2 * b^2 / (3 * a))
        in F1P.1 * F2Q.1 + F1P.2 * F2Q.2 = 0)
  : ellipse_eccentricity a b c h ^ 2 = 2 - sqrt 3 :=
sorry

end eccentricity_squared_l691_691026


namespace simplify_expression_l691_691169

variable (x : ℝ)

theorem simplify_expression : (2 * x + 20) + (150 * x + 20) = 152 * x + 40 :=
by
  sorry

end simplify_expression_l691_691169


namespace range_of_f_on_0_to_pi_l691_691032

noncomputable def f (x : ℝ) := (1 / 2) * x - Real.sin x

theorem range_of_f_on_0_to_pi :
  set.Icc (f (π / 3)) (f π) = set.Icc ((π / 6) - (Real.sqrt 3 / 2)) (π / 2) :=
by
  sorry

end range_of_f_on_0_to_pi_l691_691032


namespace monkey2_peach_count_l691_691085

noncomputable def total_peaches : ℕ := 81
def monkey1_share (p : ℕ) : ℕ := (5 * p) / 6
def remaining_after_monkey1 (p : ℕ) : ℕ := p - monkey1_share p
def monkey2_share (p : ℕ) : ℕ := (5 * remaining_after_monkey1 p) / 9
def remaining_after_monkey2 (p : ℕ) : ℕ := remaining_after_monkey1 p - monkey2_share p
def monkey3_share (p : ℕ) : ℕ := remaining_after_monkey2 p

theorem monkey2_peach_count : monkey2_share total_peaches = 20 :=
by
  sorry

end monkey2_peach_count_l691_691085


namespace proof_statements_l691_691137

/-- The domain of the function f(x) is ℝ -/
axiom domain_f_real : ∀ x : ℝ, f(x) ∈ ℝ

/-- The functional equations given in the problem -/
axiom functional_eq1 : ∀ x : ℝ, f(1 + x) = -f(1 - x)
axiom functional_eq2 : ∀ x : ℝ, f(2 + x) = f(2 - x)

/-- The definition of f(x) for x ∈ [1, 2] -/
axiom f_in_range : ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → f(x) = a * x^2 + b

/-- The given equation f(0) + f(3) = 6 -/
axiom f_at_points : f(0) + f(3) = 6

noncomputable def verify_statements := 
  (∀ x : ℝ, f(x + 4) = f(x)) ∧
  (a = -2) ∧
  (b = 2)

theorem proof_statements : verify_statements :=
by
  -- Proof goes here
  sorry

end proof_statements_l691_691137


namespace a_alone_days_l691_691633

-- Definitions based on the given conditions
def combined_days := 10
def b_alone_days := 20

-- Main statement to prove
theorem a_alone_days : 
  (1 / combined_days - 1 / b_alone_days)⁻¹ = 20 := 
by
  sorry

end a_alone_days_l691_691633


namespace best_estimate_volume_l691_691317

noncomputable def total_volume : ℝ := 50
noncomputable def divisions : ℕ := 4
noncomputable def volume_per_division := total_volume / (divisions : ℝ)
noncomputable def parts_full := 3
noncomputable def estimated_volume := parts_full * volume_per_division

theorem best_estimate_volume : 36 < estimated_volume ∧ estimated_volume < 37.5 :=
by
  have h1 : volume_per_division = 12.5 := by sorry
  have h2 : estimated_volume = 3 * 12.5 := by sorry
  have h3 : 36 < 37.5 := by norm_num
  show 36 < estimated_volume ∧ estimated_volume < 37.5
  from sorry

end best_estimate_volume_l691_691317


namespace triangle_O1AO2_angles_l691_691558

-- Definitions of the problem conditions
variables (θ : ℝ) (r₁ r₂ : ℝ)
variable (h1 : θ > 0)
variable (h2 : θ + 2*θ = 180)
variable (h3 : r₂/r₁ = sqrt 3)

-- The angles of the triangle O1AO2
theorem triangle_O1AO2_angles
  (θ60 : θ = 60)
  (r1r2 : r₂ = r₁ * sqrt 3) :
  (angle O1 A O2 = 90 ∧ angle O1 O2 A = 45 ∧ angle O1 O2 A = 45) ∨ 
  (angle O1 A O2 = 90 ∧ angle O1 O2 A = arctan 3 ∧ angle O1 O2 A = arccot 3) :=
sorry

end triangle_O1AO2_angles_l691_691558


namespace sum_a6_a7_l691_691486

theorem sum_a6_a7 (a : ℕ → ℝ) (d : ℝ):
  (∀ n, a (n + 1) = a n + d) ∧ (a 2 + a 3 + a 10 + a 11 = 48) → (a 6 + a 7 = 24) :=
begin
  sorry
end

end sum_a6_a7_l691_691486


namespace balls_placement_l691_691480

theorem balls_placement :
  (∃ (f : Fin 5 → Fin 3), (f 0 = f 1)) →
  (counting_theorem f) = 81 :=
sorry

end balls_placement_l691_691480


namespace angle_AFG_is_39_l691_691150

-- Definition of a regular pentagon
structure Pentagon (A B C D E : Type) :=
  (is_regular : ∀ (x y : Type), x ≠ y → ((angle x y) = 108))

-- Definition of an equilateral triangle
structure EquilateralTriangle (A E F : Type) :=
  (angle_AEF : angle F A E = 60)

-- Definition of a square
structure Square (A B H G : Type) :=
  (angle_BAG : angle B A G = 90)

-- Given assumptions
variables (A B C D E F G H : Type)
variables [Pentagon A B C D E] [EquilateralTriangle A E F] [Square A B H G]

-- Problem statement
theorem angle_AFG_is_39 :
  angle A F G = 39 := by
  sorry

end angle_AFG_is_39_l691_691150


namespace half_abs_diff_squares_l691_691255

theorem half_abs_diff_squares (a b : ℤ) (h₁ : a = 20) (h₂ : b = 15) : 
  (|a^2 - b^2| / 2 : ℚ) = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691255


namespace find_ab_cosB_find_sin_2B_minus_pi_over_6_l691_691082

-- Definition of triangle sides and given conditions
variables {a b c : ℝ} {A B : ℝ}

-- Given conditions
def conditions : Prop :=
  a - b = 2 ∧ c = 4 ∧ sin A = 2 * sin B

-- First part of the proof: Finding a, b, and cos B
theorem find_ab_cosB (h : conditions) : 
  a = 4 ∧ b = 2 ∧ cos B = 7 / 8 := 
sorry

-- Second part of the proof: Finding sin(2B - π/6)
theorem find_sin_2B_minus_pi_over_6 (h : conditions) : 
  sin (2 * B - π / 6) = (21 * sqrt 5 - 17) / 64 :=
sorry

end find_ab_cosB_find_sin_2B_minus_pi_over_6_l691_691082


namespace Xiaoming_age_l691_691294

theorem Xiaoming_age (x : ℕ) (h1 : x = x) (h2 : x + 18 = 2 * (x + 6)) : x = 6 :=
sorry

end Xiaoming_age_l691_691294


namespace max_value_a3b_b3a_l691_691732

theorem max_value_a3b_b3a (a b : ℝ) (h : a^2 + b^2 = 1) : 
  ∃ (M : ℝ), (∀ x y : ℝ, x^2 + y^2 = 1 → x^3 * y - y^3 * x ≤ M) ∧ M = 1 / 4 :=
by
  use 1/4
  split
  . intros x y hxy
    -- proof omitted, would show that x^3 * y - y^3 * x ≤ 1/4
    sorry
  . refl

end max_value_a3b_b3a_l691_691732


namespace find_sum_l691_691197

-- Definitions for matrices A and B
noncomputable def matrixA (a b c d : ℝ) : Matrix 3 3 ℝ :=
  ![![a, 1, b], ![2, 2, 3], ![c, 5, d]]

noncomputable def matrixB (e f g h : ℝ) : Matrix 3 3 ℝ :=
  ![[-5, e, -11], ![f, -13, g], ![2, h, 4]]

-- Definitions for the question conditions
def matrices_are_inverses {a b c d e f g h : ℝ} : Prop :=
  matrixA a b c d.mul (matrixB e f g h) = Matrix.one 3

-- Main theorem statement
theorem find_sum {a b c d e f g h : ℝ} 
  (h_inv : matrices_are_inverses) :
  a + b + c + d + e + f + g + h = 45 := sorry

end find_sum_l691_691197


namespace find_ratio_of_area_ine_abc_l691_691096

-- We need to assume the typical properties of a geometric isosceles triangle and an incenter
variables {α β γ : Type} [field α] [field β] [field γ]
variable (A B C D E N I : α)
variables (triangle_ABC : triangle A B C)
variables (isosceles_ABC : isosceles triangle_ABC B C)
variables (angle_BI_A_half : bisector A B I)
variables (angle_CI_A_half : bisector A C I)
variable (midpoint_N : midpoint N A B)
variable (midpoint_BC : midpoint D B C)
variable (angle_bisector_AD : bisector A D (line B C))
variable (angle_bisector_CE : bisector C E (line A B))

theorem find_ratio_of_area_ine_abc:
  area (triangle I N E) = (1/24) * area (triangle A B C) :=
sorry

end find_ratio_of_area_ine_abc_l691_691096


namespace collinear_D_incenter_centroid_of_equal_triangle_areas_l691_691890

theorem collinear_D_incenter_centroid_of_equal_triangle_areas
  (tetrahedron : Type*)
  (A B C D : tetrahedron)
  (area_ABD area_BCD area_CAD : ℝ)
  (h_equal_areas : area_ABD = area_BCD ∧ area_BCD = area_CAD) :
  ∃ (I G : tetrahedron),
  (incenter tetrahedron I) ∧
  (centroid tetrahedron G) ∧
  collinear {D, I, G} :=
  sorry

end collinear_D_incenter_centroid_of_equal_triangle_areas_l691_691890


namespace num_multiples_of_6_with_units_digit_6_lt_150_l691_691803

theorem num_multiples_of_6_with_units_digit_6_lt_150 : 
  ∃ (n : ℕ), (n = 5) ∧ (∀ m : ℕ, (m < 150) → (m % 6 = 0) → (m % 10 = 6) → (∃ k : ℕ, m = 6 * (5 * k + 1) ∧ 6 * (5 * k + 1) < 150)) :=
by 
  have multiples : list ℕ := [6, 36, 66, 96, 126] 
  apply Exists.intro 5
  split
  {
    exact rfl,
  }
  {
    intros m h1 h2 h3 
    sorry
  }

end num_multiples_of_6_with_units_digit_6_lt_150_l691_691803


namespace length_of_CD_l691_691156

theorem length_of_CD (x y: ℝ) (h1: 5 * x = 3 * y) (u v: ℝ) (h2: u = x + 3) (h3: v = y - 3) (h4: 7 * u = 4 * v) : x + y = 264 :=
by
  sorry

end length_of_CD_l691_691156


namespace fertilizer_production_bounds_l691_691660

theorem fertilizer_production_bounds :
  ∀ (x : ℕ)
  (max_workers : ℕ)
  (max_hours_worker : ℕ)
  (hours_per_bag : ℕ)
  (min_bags : ℕ)
  (kg_per_bag : ℕ)
  (raw_material_in_tons : ℕ),
  max_workers = 200 →
  max_hours_worker = 2100 →
  hours_per_bag = 4 →
  min_bags = 80000 →
  kg_per_bag = 20 →
  raw_material_in_tons = (800 + 1200 - 200) →
  (min_bags ≤ x ∧ x ≤ min (max_workers * max_hours_worker / hours_per_bag) (raw_material_in_tons * 1000 / kg_per_bag)) :=
begin
  sorry
end

end fertilizer_production_bounds_l691_691660


namespace find_possible_solutions_l691_691392

noncomputable def function_f (f : ℕ+ → ℝ) :=
  (∀ x : ℕ+, f (x + 22) = f x) ∧
  (∀ x y : ℕ+, f (x^2 * y) = (f x)^2 * f y)

theorem find_possible_solutions :
  ∃ (f : ℕ+ → ℝ), function_f f ∧ (set.count (set_of (λ f, function_f f)) = 13) :=
sorry

end find_possible_solutions_l691_691392


namespace pen_average_price_l691_691976

theorem pen_average_price (pens_purchased pencils_purchased : ℕ) (total_cost pencil_avg_price : ℝ)
  (H0 : pens_purchased = 30) (H1 : pencils_purchased = 75) 
  (H2 : total_cost = 690) (H3 : pencil_avg_price = 2) :
  (total_cost - (pencils_purchased * pencil_avg_price)) / pens_purchased = 18 :=
by
  rw [H0, H1, H2, H3]
  sorry

end pen_average_price_l691_691976


namespace find_x_l691_691650

theorem find_x (x : ℝ) (h : 2500 - 1002 / x = 2450) : x = 20.04 :=
by 
  sorry

end find_x_l691_691650


namespace max_magnitude_l691_691019

variables (a b : ℝ^3) (t : ℝ)
hypothesis ha : ∥a∥ = 1
hypothesis hb : ∥b∥ = 1
hypothesis angle_ab : real.angle a b = real.pi / 3
hypothesis t_in : -1 ≤ t ∧ t ≤ 1

theorem max_magnitude : ∃ t : ℝ, -1 ≤ t ∧ t ≤ 1 ∧ ∥a + t • b∥ = √3 :=
sorry

end max_magnitude_l691_691019


namespace simplify_f_alpha_find_value_l691_691304

-- Definition of f for the given expression in Problem 1
def f (α : ℝ) : ℝ := 
    (sin (π + α) * sin (2 * π - α) * cos (-π - α) * cos (π / 2 + α)) / 
    (sin (3 * π + α) * cos (π - α) * cos (3 * π / 2 + α))

-- Statement for Problem 1
theorem simplify_f_alpha (α : ℝ) : f(α) = sin(α) :=
by
  sorry

-- Statement for Problem 2
theorem find_value : 
  (2 * cos (10 * real.pi / 180) - sin (20 * real.pi / 180)) / sin (70 * real.pi / 180) = sqrt 3 :=
by
  sorry

end simplify_f_alpha_find_value_l691_691304


namespace sym_diff_A_B_l691_691410

def set_diff (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}
def sym_diff (M N : Set ℝ) : Set ℝ := set_diff M N ∪ set_diff N M

def A : Set ℝ := {x | -1 ≤ x ∧ x < 1}
def B : Set ℝ := {x | x < 0}

theorem sym_diff_A_B :
  sym_diff A B = {x | x < -1} ∪ {x | 0 ≤ x ∧ x < 1} := by
  sorry

end sym_diff_A_B_l691_691410


namespace cost_per_book_l691_691618

theorem cost_per_book (num_animal_books : ℕ) (num_space_books : ℕ) (num_train_books : ℕ) (total_cost : ℕ) 
                      (h1 : num_animal_books = 10) (h2 : num_space_books = 1) (h3 : num_train_books = 3) (h4 : total_cost = 224) :
  (total_cost / (num_animal_books + num_space_books + num_train_books) = 16) :=
by sorry

end cost_per_book_l691_691618


namespace category_D_cost_after_discount_is_correct_l691_691111

noncomputable def total_cost : ℝ := 2500
noncomputable def percentage_A : ℝ := 0.30
noncomputable def percentage_B : ℝ := 0.25
noncomputable def percentage_C : ℝ := 0.20
noncomputable def percentage_D : ℝ := 0.25
noncomputable def discount_A : ℝ := 0.03
noncomputable def discount_B : ℝ := 0.05
noncomputable def discount_C : ℝ := 0.07
noncomputable def discount_D : ℝ := 0.10

noncomputable def cost_before_discount_D : ℝ := total_cost * percentage_D
noncomputable def discount_amount_D : ℝ := cost_before_discount_D * discount_D
noncomputable def cost_after_discount_D : ℝ := cost_before_discount_D - discount_amount_D

theorem category_D_cost_after_discount_is_correct : cost_after_discount_D = 562.5 := 
by 
  sorry

end category_D_cost_after_discount_is_correct_l691_691111


namespace half_abs_diff_squares_l691_691258

theorem half_abs_diff_squares (a b : ℤ) (h₁ : a = 20) (h₂ : b = 15) : 
  (|a^2 - b^2| / 2 : ℚ) = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691258


namespace sum_ijk_divisibility_l691_691902

theorem sum_ijk_divisibility (p : ℕ) (hp : Nat.Prime p) (hp_gt : p > 3) :
  let S := ∑ i in Finset.Ico 2 (p - 1), ∑ j in Finset.Ico (i + 1) (p - 1), ∑ k in Finset.Ico (j + 1) (p - 1), i * j * k
  (S + 1) % p = 0 :=
by
  sorry

end sum_ijk_divisibility_l691_691902


namespace least_times_to_eat_l691_691637

theorem least_times_to_eat (A B C : ℕ) (h1 : A = (9 * B) / 5) (h2 : B = C / 8) : 
  A = 2 ∧ B = 1 ∧ C = 8 :=
sorry

end least_times_to_eat_l691_691637


namespace chose_number_l691_691807

theorem chose_number (x : ℝ) (h : (x / 12)^2 - 240 = 8) : x = 24 * Real.sqrt 62 :=
sorry

end chose_number_l691_691807


namespace teacher_a_probability_correct_total_assessments_three_correct_l691_691568

noncomputable def teacher_a_pass_probability : ℚ :=
  let p_a : ℚ := 2/3 in
  p_a + (1 - p_a) * p_a

theorem teacher_a_probability_correct : teacher_a_pass_probability = 8/9 :=
by
  -- Proof will be filled in later
  sorry

noncomputable def total_assessments_three_probability : ℚ :=
  let p_a : ℚ := 2/3 in
  let p_b : ℚ := 1/2 in
  (p_a * (1 - p_b)) + ((1 - p_a) * p_b)

theorem total_assessments_three_correct : total_assessments_three_probability = 1/2 :=
by
  -- Proof will be filled in later
  sorry

end teacher_a_probability_correct_total_assessments_three_correct_l691_691568


namespace horner_method_correctness_l691_691944

theorem horner_method_correctness :
  let f : ℤ → ℤ := λ x, x^6 + 6 * x^4 + 9 * x^2 + 208 in
  ∃ v_2, f (-4) = ((((((-4) * (-4)) + 6) * (-4))) + 9) * (-4) + 208 ∧ v_2 = 22 :=
begin
  -- The polynomial definition and Horner's method evaluation in Lean
  let poly := λ x : ℤ, x^6 + 6 * x^4 + 9 * x^2 + 208,
  let x := (-4 : ℤ),
  let v_0 := 1,
  let v_1 := v_0 * x,
  let v_2 := v_1 * x + 6,
  let v_3 := v_2 * x,
  let v_4 := v_3 * x + 9,
  let v_5 := v_4 * x,
  let v_6 := v_5 * x + 208,
  use v_2,
  split,
  {
    exact poly x = v_6,
  },
  {
    exact v_2 = 22,
  }
end

end horner_method_correctness_l691_691944


namespace hyperbola_equation_l691_691467

-- Conditions
def hyperbola (a b : ℝ) := a > 0 ∧ b > 0 ∧ (a^2 + b^2 = 25) ∧ (b = 2 * a)

-- Proof statement
theorem hyperbola_equation :
  ∀ a b : ℝ, hyperbola a b → (a = sqrt 5 ∧ b = 2 * sqrt 5) → 
  (∀ x y : ℝ, (x^2 / 5) - (y^2 / 20) = (x^2 / a^2) - (y^2 / b^2)) :=
by
  intros a b h_eqn h_asympt
  sorry

end hyperbola_equation_l691_691467


namespace opposite_face_of_black_is_orange_l691_691315

constants (Color : Type) (O Y B P V K : Color)

def cube_views := 
  ∃ (top1 front1 right1 top2 front2 right2 top3 front3 right3 : Color),
    top1 = O ∧ front1 = B ∧ right1 = P ∧
    top2 = O ∧ front2 = V ∧ right2 = P ∧
    top3 = O ∧ front3 = Y ∧ right3 = P ∧
    (top1 = top2 ∧ top2 = top3) ∧
    (right1 = right2 ∧ right2 = right3)

theorem opposite_face_of_black_is_orange : cube_views → (∃ top bottom : Color, bottom = K ∧ top = O) :=
by
  sorry

end opposite_face_of_black_is_orange_l691_691315


namespace area_calculation_l691_691318

variable (x : ℝ)

def area_large_rectangle : ℝ := (2 * x + 9) * (x + 6)
def area_rectangular_hole : ℝ := (x - 1) * (2 * x - 5)
def area_square : ℝ := (x + 3) ^ 2
def area_remaining : ℝ := area_large_rectangle x - area_rectangular_hole x - area_square x

theorem area_calculation : area_remaining x = -x^2 + 22 * x + 40 := by
  sorry

end area_calculation_l691_691318


namespace diameter_of_large_circle_l691_691716

theorem diameter_of_large_circle (
  radius_small: ℝ
  (hr: radius_small = 5) 
):
  ∃ (diameter_large: ℝ), diameter_large = 10 * (3 + Real.sqrt 3) :=
by
  sorry

end diameter_of_large_circle_l691_691716


namespace average_age_when_youngest_born_l691_691307

noncomputable def total_age_of_group (n : ℕ) (average_age : ℕ) : ℕ :=
  n * average_age

noncomputable def total_age_excluding_youngest (total_age : ℕ) (youngest_age : ℕ) : ℕ :=
  total_age - youngest_age

noncomputable def average_age_excluding_youngest (total_age_excluding_youngest : ℕ) (remaining_people : ℕ) : ℚ :=
  (total_age_excluding_youngest : ℚ) / remaining_people

theorem average_age_when_youngest_born (average_age_of_seven : ℕ) (youngest_age : ℕ) :
  let total_age := total_age_of_group 7 average_age_of_seven,
      total_age_excl_youngest := total_age_excluding_youngest total_age youngest_age,
      average_age_of_remaining := average_age_excluding_youngest total_age_excl_youngest 6
  in average_age_of_seven = 30 ∧ youngest_age = 4 → average_age_of_remaining ≈ 34.33 :=
by
  intros
  sorry

end average_age_when_youngest_born_l691_691307


namespace sum_of_inradii_eq_799_l691_691117

-- Definitions and assumptions as per the conditions
def is_convex_quadrilateral (A B C D : Type) : Prop := sorry
def right_angle (A B C : Type) := sorry

def incircle_radius (A B C : Type) (P : Type) : ℝ := sorry
def points_touch (A B C : Type) (P : Type) (BD : Type) : Prop := sorry

noncomputable def sum_of_radii {A B C D P Q : Type}
  (is_convex : is_convex_quadrilateral A B C D)
  (angle_DAB : right_angle D A B)
  (angle_BDC : right_angle B D C)
  (AD_len : ℝ)
  (PQ_len : ℝ)
  (AD_eq : AD_len = 999)
  (PQ_eq : PQ_len = 200)
  (P_between_BQ : P ≠ Q)
  (touch_P : points_touch A B D P BD)
  (touch_Q : points_touch B D C Q BD) : ℝ :=
  incircle_radius A B D P + incircle_radius B D C Q

theorem sum_of_inradii_eq_799 {A B C D P Q BD : Type}
  (is_convex : is_convex_quadrilateral A B C D)
  (angle_DAB : right_angle D A B)
  (angle_BDC : right_angle B D C)
  (AD_len : ℝ)
  (PQ_len : ℝ)
  (AD_eq : AD_len = 999)
  (PQ_eq : PQ_len = 200)
  (P_between_BQ : P ≠ Q)
  (touch_P : points_touch A B D P BD)
  (touch_Q : points_touch B D C Q BD) :
  sum_of_radii is_convex angle_DAB angle_BDC AD_len PQ_len AD_eq PQ_eq P_between_BQ touch_P touch_Q = 799 := sorry

end sum_of_inradii_eq_799_l691_691117


namespace mon_decreasing_interval_l691_691497

noncomputable def f (a x : ℝ) := x^3 - a * x^2 - x + 6

theorem mon_decreasing_interval {a : ℝ} :
  (∀ x ∈ set.Ioo (0 : ℝ) 1, deriv (λ x, f a x) x ≤ 0) ↔ a ≥ 1 :=
sorry

end mon_decreasing_interval_l691_691497


namespace polynomial_with_given_roots_l691_691700

theorem polynomial_with_given_roots :
  ∃ (p : Polynomial ℝ), p = Polynomial.C (1 : ℝ) * (Polynomial.X + 2) * 
  (Polynomial.X - (Polynomial.C((4 : ℝ) : Complex) - Polynomial.C((3 : ℝ) * Complex.I))) *
  (Polynomial.X - (Polynomial.C((4 : ℝ) : Complex) + Polynomial.C((3 : ℝ) * Complex.I))) ∧
  (p.map Polynomial.realToComplex).toReal = Polynomial.C (1 : ℝ) * 
  (Polynomial.X^3 - Polynomial.C (6 : ℝ) * Polynomial.X^2 + Polynomial.C (9 : ℝ) * Polynomial.X + Polynomial.C (50 : ℝ)) :=
by
  sorry

end polynomial_with_given_roots_l691_691700


namespace part_a_l691_691636

theorem part_a :
  ¬ ∃ (f : fin 6 × fin 6 → bool), 
    (∀ i, f i) ∧           -- all squares are covered
    (∀ i j, i ≠ j → ¬(f i ∧ f j)) ∧ -- no overlaps
    (∀ (p : fin 6 × fin 6), (∃ i, p.snd = nat.succ i ∧ 
      (∀ x, f (p.fst, x)))) -- no sticking out
:=
  sorry

end part_a_l691_691636


namespace equidistant_perpendiculars_l691_691003

open EuclideanGeometry

variables {A B C G X Y : Point} 

theorem equidistant_perpendiculars (h : centroid A B C G) (hXY : line_through G X Y)
  (h_perp_A : is_perpendicular (line X Y) (line A (foot (line X Y) A)))
  (h_perp_B : is_perpendicular (line X Y) (line B (foot (line X Y) B)))
  (h_perp_C : is_perpendicular (line X Y) (line C (foot (line X Y) C))) :
  distance A (foot (line X Y) A) + distance B (foot (line X Y) B) = distance C (foot (line X Y) C) := 
sorry

end equidistant_perpendiculars_l691_691003


namespace IMB_angle_l691_691115

theorem IMB_angle (A B C M I : Type)
  [triangle : triangle ABC]
  (H_right : ∠ B = 90)
  (H_C : ∠ C = 30)
  (H_M_midpoint : midpoint M A C)
  (H_I_incenter : incenter I ABC) :
  ∠ IMB = 15 :=
by
  sorry

end IMB_angle_l691_691115


namespace area_ratio_of_triangles_l691_691120

-- Define the pentagon PQRST with given properties
variables (P Q R S T : Type) [ConvexPentagon P Q R S T] 
variables (PQ RT QR PS QS PT : ℝ)
variables (angle_PQR : ℝ)

-- Conditions
def conditions : Prop :=
  PQ = 4 ∧
  QR = 7 ∧
  PT = 21 ∧
  PQ * ‖RT ‖ = 0 ∧ -- PQ ⊥ RT indicating parallelism
  QR * ‖PS ‖ = 0 ∧ -- QR ⊥ PS indicating parallelism
  QS * ‖PT ‖ = 0 ∧ -- QS ⊥ PT indicating parallelism
  angle_PQR = 100

-- Statement of the problem proving m+n = 232
theorem area_ratio_of_triangles (m n : ℕ) (h : Nat.Coprime m n) (m_plus_n : ℕ) 
  (h_eq : m + n = 232) : 
  (∃ m n, Nat.Coprime m n ∧ m + n = 232) := by
  sorry

-- Definition of the problem with the conditions
noncomputable def problem_statement := 
  ∀ (PQRST : ConvexPentagon P Q R S T),
    conditions P Q R S T PQ QR PT ∧ ∃ (m n : ℕ), 
      Nat.Coprime m n ∧ m + n = 232

#eval problem_statement

end area_ratio_of_triangles_l691_691120


namespace pyramid_problem_solution_l691_691932

-- Defining the constants and inputs
def side_length : ℝ := 4 * real.sqrt 3
def angle_DAB : ℝ := real.arctan (real.sqrt (37 / 3))

-- Defining the problems
def angle_between_BA1_AC1 (BA1 AC1 : ℝ) : ℝ := real.arccos (11 / 32)
def distance_between_BA1_AC1 (distance : ℝ) : ℝ := 36 / (real.sqrt 301)
def radius_of_sphere : ℝ := 2

-- The main theorem that encapsulates the entire problem and solution
theorem pyramid_problem_solution :
    ∃ (BA1 AC1 distance : ℝ), 
    angle_between_BA1_AC1 BA1 AC1 = real.arccos (11 / 32) ∧ 
    distance_between_BA1_AC1 distance = 36 / (real.sqrt 301) ∧ 
    radius_of_sphere = 2 :=
sorry

end pyramid_problem_solution_l691_691932


namespace area_of_circle_l691_691706

theorem area_of_circle (x y : ℝ) :
  (x^2 + y^2 - 8*x - 6*y = -9) → 
  (∃ (R : ℝ), (x - 4)^2 + (y - 3)^2 = R^2 ∧ π * R^2 = 16 * π) :=
by
  sorry

end area_of_circle_l691_691706


namespace fraction_of_A_eq_l691_691982

noncomputable def fraction_A (A B C T : ℕ) : ℚ :=
  A / (T - A)

theorem fraction_of_A_eq :
  ∃ (A B C T : ℕ), T = 360 ∧ A = B + 10 ∧ B = 2 * (A + C) / 7 ∧ T = A + B + C ∧ fraction_A A B C T = 1 / 3 :=
by
  sorry

end fraction_of_A_eq_l691_691982


namespace tanks_fill_l691_691189

theorem tanks_fill
  (c : ℕ) -- capacity of each tank
  (h1 : 300 < c) -- first tank is filled with 300 liters, thus c > 300
  (h2 : 450 < c) -- second tank is filled with 450 liters, thus c > 450
  (h3 : (45 : ℝ) / 100 = (450 : ℝ) / c) -- second tank is 45% filled, thus 0.45 * c = 450
  (h4 : 300 + 450 < 2 * c) -- the two tanks have the same capacity, thus they must have enough capacity to be filled more than 750 liters
  : c - 300 + (c - 450) = 1250 :=
sorry

end tanks_fill_l691_691189


namespace determine_b_l691_691332

noncomputable def b_value (b : ℝ) (m : ℝ) : Prop :=
  b = -2 * real.sqrt (3 / 20) ∧
  (∀ x, (x^2 + b*x + 1/5 = (x + m)^2 + 1/20)) ∧ m = -real.sqrt(3 / 20)

theorem determine_b :
  ∃ b : ℝ,
  ∃ m : ℝ,
  b_value b m := by
  sorry

end determine_b_l691_691332


namespace lawrence_walking_speed_l691_691542

theorem lawrence_walking_speed :
  let distance := 4
  let time := (4 : ℝ) / 3
  let speed := distance / time
  speed = 3 := 
by
  sorry

end lawrence_walking_speed_l691_691542


namespace sandwiches_count_l691_691388

def total_sandwiches : ℕ :=
  let meats := 12
  let cheeses := 8
  let condiments := 5
  meats * (Nat.choose cheeses 2) * condiments

theorem sandwiches_count : total_sandwiches = 1680 := by
  sorry

end sandwiches_count_l691_691388


namespace extreme_value_at_one_max_min_on_interval_l691_691463

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) := 2 * x^3 + a * x^2 + b * x + 1

theorem extreme_value_at_one (a b : ℝ) :
  (f 1 a b = -6) ∧ (deriv (f x a b) 1 = 0) ↔ (a = 3) ∧ (b = -12) :=
by
  sorry

theorem max_min_on_interval :
  let f' := f x 3 -12
  let I := Set.Icc (-2 : ℝ) (2 : ℝ) in
  (∀ x ∈ I, f' x ≤ 21) ∧ (∃ x ∈ I, f' x = 21) ∧ (∀ x ∈ I, f' x ≥ -6) ∧ (∃ x ∈ I, f' x = -6) :=
by
  let f' := λ x, 2 * x^3 + 3 * x^2 - 12 * x + 1
  sorry

end extreme_value_at_one_max_min_on_interval_l691_691463


namespace max_a_value_l691_691499

theorem max_a_value (a : ℝ) (h : ∃ x : ℝ, Real.sin x ≥ a) : a ≤ 1 :=
by
  have h1 : -1 ≤ Real.sin 0 := by
    apply Real.sin,
  have h2 : Real.sin 0 ≤ 1 := by
    apply Real.sin,
  sorry

end max_a_value_l691_691499


namespace area_of_field_condition_l691_691178

-- Define the given conditions
def cost_of_fencing : ℝ := 6466.70
def rate_per_meter : ℝ := 4.90

-- Define the question as a hypothesis
def area_of_field_approx : ℝ := 13.854

theorem area_of_field_condition (C : ℝ) (A : ℝ) : 
  (Cost_eq : cost_of_fencing = rate_per_meter * C) → 
  (Circumference_eq : C = 2 * Real.pi * (A / (2 * Real.pi))^0.5) →
  ∃ area_in_hectares, area_in_hectares = A / 10000 ∧ abs (area_in_hectares - 13.854) < 0.001 :=
begin
  sorry
end

end area_of_field_condition_l691_691178


namespace harmonic_bounds_l691_691865

variable {n : ℕ}

noncomputable def S_n (n : ℕ) := ∑ i in Finset.range (n+1), (1/(i+1))

theorem harmonic_bounds (n : ℕ) (h : 2 < n) : 
    (S_n n) > n * (n+1)^(1/(n:ℝ)) - n  ∧ 
    (S_n n) < n - (n-1) * n^(1/(n-1:ℝ)) :=
by
  sorry

end harmonic_bounds_l691_691865


namespace club_selection_ways_l691_691984

theorem club_selection_ways (n : ℕ) (hn : n = 18) :
  let secretary_choices := n,
      remaining := n - 1,
      president_choices := Nat.choose remaining 2
  in
    secretary_choices * president_choices = 2448 := by
  sorry

end club_selection_ways_l691_691984


namespace more_math_than_reading_l691_691893

def pages_reading := 4
def pages_math := 7

theorem more_math_than_reading : pages_math - pages_reading = 3 :=
by
  sorry

end more_math_than_reading_l691_691893


namespace rhombus_inscribed_circle_radius_l691_691595

noncomputable def sin := Real.sin -- Ensure sin function is recognized as non-computable in Lean

theorem rhombus_inscribed_circle_radius
  (a : ℝ) (theta : ℝ)
  (h_side : a = 8)
  (h_angle : theta = Real.pi / 6) : 
  ∃ (r : ℝ), r = 2 :=
by
  -- Define height function using the sine definition
  let h := a * sin theta
  -- state the height in specific scenarios
  have h_h : h = 4 :=
    by
      rw [h_side, h_angle]
      have sin_val : sin (Real.pi / 6) = 1 / 2 := by sorry
      rw [sin_val]
      norm_num
  -- radius is radius is because h is height
  use h / 2
  rw [h_h]
  norm_num

end rhombus_inscribed_circle_radius_l691_691595


namespace find_angle_l691_691431

/-
  Given point P(√3/2, -1/2) is on the terminal side of angle θ,
  and θ ∈ [0, 2π), then the value of θ is 11π/6.
-/

theorem find_angle (θ : ℝ) 
    (hP : ∃ (x y : ℝ), x = sqrt 3 / 2 ∧ y = -1 / 2 ∧ x = cos θ ∧ y = sin θ)
    (hθ : 0 ≤ θ ∧ θ < 2 * π) : θ = 11 * π / 6 := 
begin
  sorry
end

end find_angle_l691_691431


namespace tractors_planting_rate_l691_691722

theorem tractors_planting_rate (total_acres : ℕ) (total_days : ℕ) 
    (tractors_first_team : ℕ) (days_first_team : ℕ)
    (tractors_second_team : ℕ) (days_second_team : ℕ)
    (total_tractor_days : ℕ) :
    total_acres = 1700 →
    total_days = 5 →
    tractors_first_team = 2 →
    days_first_team = 2 →
    tractors_second_team = 7 →
    days_second_team = 3 →
    total_tractor_days = (tractors_first_team * days_first_team) + (tractors_second_team * days_second_team) →
    total_acres / total_tractor_days = 68 :=
by
  -- proof can be filled in later
  intros
  sorry

end tractors_planting_rate_l691_691722


namespace frog_climbs_out_l691_691985

theorem frog_climbs_out (h: 19 > 0) (climb: 3 > 0) (slide: 2 > 0) (e: 3 - 2 > 0):
  let effective_climb := 3 - 2 in
  let remaining_meters := 19 - 3 in
  let days_needed := remaining_meters / effective_climb in
  days_needed + 1 = 17 := 
by 
  sorry

end frog_climbs_out_l691_691985


namespace b_2019_l691_691138

noncomputable def a_seq (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  if n = 1 then 2 else
  2 * (Finset.range (n - 1)).sum (λ i, a_seq i) - (Finset.range n).sum (λ i, a_seq i)

def b_seq (n : ℕ) : ℕ := a_seq (n + 1) - a_seq n

theorem b_2019 : b_seq 2019 = 5 * 2 ^ 1008 :=
  sorry

end b_2019_l691_691138


namespace half_absolute_difference_of_squares_l691_691265

-- Defining the variables a and b involved in the problem
def a : ℤ := 20
def b : ℤ := 15

-- Statement to prove the solution
theorem half_absolute_difference_of_squares : 
    (1 / 2 : ℚ) * (abs ((a ^ 2) - (b ^ 2))) = 87.5 :=
by
    -- Proof omitted
    sorry

end half_absolute_difference_of_squares_l691_691265


namespace new_train_travel_distance_l691_691989

-- Definitions of the trains' travel distances
def older_train_distance : ℝ := 180
def new_train_additional_distance_ratio : ℝ := 0.50

-- Proof that the new train can travel 270 miles
theorem new_train_travel_distance
: new_train_additional_distance_ratio * older_train_distance + older_train_distance = 270 := 
by
  sorry

end new_train_travel_distance_l691_691989


namespace problem_l691_691769

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

theorem problem
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_geom : geometric_sequence a q)
  (h1 : a 0 + a 1 = 4 / 9)
  (h2 : a 2 + a 3 + a 4 + a 5 = 40) :
  (a 6 + a 7 + a 8) / 9 = 117 :=
sorry

end problem_l691_691769


namespace universal_set_condition_l691_691962

theorem universal_set_condition (A B S : Set) (h_universal: ∀ x : S, x ∈ S) (h_inter: A ∩ B = S) : A = B ∧ B = S :=
by
  sorry

end universal_set_condition_l691_691962


namespace maximum_airlines_l691_691504
open Classical

theorem maximum_airlines (n : ℕ) (h_n : n = 120) :
  let total_pairs := n * (n - 1) / 2 in
  total_pairs = 7140 → ∃ k, k = 60 ∧ (n - 1) * k ≤ total_pairs :=
by
  intros total_pairs h_pairs
  use 60
  split
  · refl
  · sorry

end maximum_airlines_l691_691504


namespace net_amount_spent_correct_l691_691875

def trumpet_cost : ℝ := 145.16
def song_book_revenue : ℝ := 5.84
def net_amount_spent : ℝ := 139.32

theorem net_amount_spent_correct : trumpet_cost - song_book_revenue = net_amount_spent :=
by
  sorry

end net_amount_spent_correct_l691_691875


namespace geometric_sequence_sum_l691_691748

theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℚ),
  (∀ n, 3 * a (n + 1) + a n = 0) ∧
  a 2 = -2/3 ∧
  (a 0 + a 1 + a 2 + a 3 + a 4) = 122/81 :=
sorry

end geometric_sequence_sum_l691_691748


namespace find_omega_max_min_values_l691_691760

-- Condition definitions
def f (x : ℝ) (ω : ℝ) : ℝ := 4 * cos (ω * x) * sin (ω * x + π / 6) + 1
def adjacent_zeros_dist (x1 x2 : ℝ) (d : ℝ) := abs (x1 - x2) = d

-- Problem (Ⅰ): Prove that ω = 1 given certain conditions.
theorem find_omega (x1 x2 : ℝ) (h1 : adjacent_zeros_dist x1 x2 π) (h2 : ∀ x, f x 1 = 0) (h3 : 1 > 0) :
  ∃ ω > 0, ∀ x, f x ω = 0 ∧ abs (x1 - x2) = π → ω = 1 :=
sorry

-- Problem (Ⅱ): Prove the max and min values of f(x) in the given interval.
theorem max_min_values :
  ∃ max_val min_val, 
    (∀ x ∈ Set.Icc (-π / 6) (π / 4), 2 * sin (2 * x + π / 6) + 2 ≤ max_val) ∧
    (∀ x ∈ Set.Icc (-π / 6) (π / 4), 2 * sin (2 * x + π / 6) + 2 ≥ min_val) ∧
    max_val = 4 ∧ min_val = 1 :=
sorry

end find_omega_max_min_values_l691_691760


namespace fastest_growth_rate_l691_691681

theorem fastest_growth_rate :
  ∀ (x : ℝ), 
    (∀ y : ℝ, 
      (y = 50 → deriv (fun x : ℝ => 50) x = 0) ∧
      (y = 1000 * x → deriv (fun x : ℝ => 1000 * x) x = 1000) ∧
      (y = log x → deriv (fun x : ℝ => log x) x = 1 / x) ∧
      (y = (1 / 1000) * exp x → deriv (fun x : ℝ => (1 / 1000) * exp x) x = (exp x) / 1000)) →
    (deriv (fun x : ℝ => (1 / 1000) * exp x) x > deriv (fun x : ℝ => 50) x) ∧
    (deriv (fun x : ℝ => (1 / 1000) * exp x) x > deriv (fun x : ℝ => 1000 * x) x) ∧
    (deriv (fun x : ℝ => (1 / 1000) * exp x) x > deriv (fun x : ℝ => log x) x) :=
by {
  intros,
  sorry -- Proof not required
}

end fastest_growth_rate_l691_691681


namespace sum_of_coordinates_l691_691817

theorem sum_of_coordinates : 
  (∃ g h : ℝ → ℝ, g 2 = 5 ∧ ∀ x, h x = (g x)^2) → 2 + (g 2)^2 = 27 :=
by
  sorry

end sum_of_coordinates_l691_691817


namespace period_of_f_number_of_zeros_range_of_lambda_l691_691035

-- Proof Problem 1
theorem period_of_f (f : ℝ → ℝ) (λ : ℝ) (h : ∀ x > 0, f(x) = λ * log(x + 1) - sin x) : 
  (∀ x > 0, f(x) = f(x + 2 * π)) ↔ λ = 0 := sorry

-- Proof Problem 2
theorem number_of_zeros (f : ℝ → ℝ) (λ : ℝ) (h : ∀ x, f(x) = log(x + 1) - sin x) (hλ : λ = 1) :
  set.finite (set_of (x ∈ Icc (π/2) (to_real ⊤) ∧ f(x) = 0)) ∧ 
  (set.card (set_of (x ∈ Icc (π/2) (to_real ⊤) ∧ f(x) = 0)) = 1) := sorry

-- Proof Problem 3
theorem range_of_lambda (f : ℝ → ℝ) (λ : ℝ) (h : ∀ x, f(x) = λ * log(x + 1) - sin x) :
  (∀ x ∈ Icc 0 π, f(x) ≥ 2 * (1 - exp x)) ↔ λ ∈ Icc (-1 : ℝ) to_real ⊤ := sorry

end period_of_f_number_of_zeros_range_of_lambda_l691_691035


namespace fraction_eaten_correct_l691_691218

def initial_nuts : Nat := 30
def nuts_left : Nat := 5
def eaten_nuts : Nat := initial_nuts - nuts_left
def fraction_eaten : Rat := eaten_nuts / initial_nuts

theorem fraction_eaten_correct : fraction_eaten = 5 / 6 := by
  sorry

end fraction_eaten_correct_l691_691218


namespace base_8_addition_l691_691723

theorem base_8_addition (X Y : ℕ) (h1 : Y + 2 % 8 = X % 8) (h2 : X + 3 % 8 = 2 % 8) : X + Y = 12 := by
  sorry

end base_8_addition_l691_691723


namespace parallelogram_area_l691_691853

open Matrix

def v : Fin 2 → ℝ := ![7, -4]
def w : Fin 2 → ℝ := ![13, -3]

theorem parallelogram_area :
  let m := Matrix.of_vec v w in
  abs (Matrix.det m) = 31 :=
by
  sorry

end parallelogram_area_l691_691853


namespace find_c_l691_691069

theorem find_c (b c : ℝ) (h : (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c * x + 8)) : c = 17 / 3 := 
by
  -- Add the necessary assumptions and let Lean verify these assumptions.
  have b_eq : 3 * b = 8 := sorry
  have b_val : b = 8 / 3 := sorry
  have h_coeff : c = b + 3 := sorry
  exact h_coeff.trans (by rw [b_val]; norm_num)

end find_c_l691_691069


namespace green_green_pairs_count_l691_691687

theorem green_green_pairs_count
  (total_students : ℕ)
  (red_shirts : ℕ)
  (green_shirts : ℕ)
  (total_pairs : ℕ)
  (red_red_pairs : ℕ)
  (h1 : total_students = 140)
  (h2 : red_shirts = 60)
  (h3 : green_shirts = 80)
  (h4 : total_pairs = 70)
  (h5 : red_red_pairs = 10)
  : 2 * (green_shirts - (red_shirts - 2 * red_red_pairs)) = 40 :=
by {
  rw [h1, h2, h3, h4, h5],
  sorry
}

end green_green_pairs_count_l691_691687


namespace half_abs_diff_of_squares_l691_691249

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 20) (h2 : b = 15) :
  (1/2 : ℚ) * |(a:ℚ)^2 - (b:ℚ)^2| = 87.5 := by
  sorry

end half_abs_diff_of_squares_l691_691249


namespace f_at_2_l691_691577

-- Define f as a quadratic function.
variable {f : ℝ → ℝ}

-- Assume f satisfies the given conditions
axiom quadratic_fn (x : ℝ) : ∃ a b c : ℝ, f(x) = a * x^2 + b * x + c
axiom inverse_fn (x : ℝ) : f(x) = 3 * by have hf := inverse_fn _; exact (1 / 3) * x - 5/3  -- Assume reasonable inverse form
axiom f_at_1 : f 1 = 5

-- Prove that f(2)=8
theorem f_at_2 : f 2 = 8 := sorry

end f_at_2_l691_691577


namespace find_Y_value_l691_691734

theorem find_Y_value : ∃ Y : ℤ, 80 - (Y - (6 + 2 * (7 - 8 - 5))) = 89 ∧ Y = -15 := by
  sorry

end find_Y_value_l691_691734


namespace half_abs_diff_squares_l691_691243

theorem half_abs_diff_squares : (1 / 2) * |20^2 - 15^2| = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691243


namespace subset_P_Q_l691_691787

def P := {x : ℝ | x > 1}
def Q := {x : ℝ | x^2 - x > 0}

theorem subset_P_Q : P ⊆ Q :=
by
  sorry

end subset_P_Q_l691_691787


namespace b_range_l691_691005

open Real

section
variable {f : ℝ → ℝ}

noncomputable def is_odd_function_on_ℝ_with_period_4 (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 4) = f x)

noncomputable def f_definition (x b : ℝ) : ℝ := ln (x^2 - x + b)
noncomputable def num_zeros_in_interval (f : ℝ → ℝ) (a b : ℝ) : ℕ := (∑ k in Icc a b, if f k = 0 then 1 else 0)

theorem b_range (h₁ : is_odd_function_on_ℝ_with_period_4 f)
                (h₂ : ∀ x ∈ Ioo 0 2, f x = f_definition x) 
                (h₃ : num_zeros_in_interval f (-2) 2 = 5) :
  (1 / 4 : ℝ) < b ∧ b ≤ 1 ∨ b = 5 / 4 :=
sorry
end

end b_range_l691_691005


namespace minimum_f_value_l691_691408

-- Defining the function f(a, b) based on the conditions specified
def f (a b : ℕ) : ℕ :=
  if a ≠ b ∧ a < 2012 ∧ b < 2012 then
    (Finset.filter (λ k, (((a * k) % 2012) > ((b * k) % 2012))) (Finset.range 2012)).card
  else 0

-- The theorem statement that needs to be proven.
theorem minimum_f_value : ∃ S, (∀ a b, a ≠ b ∧ a < 2012 ∧ b < 2012 → f(a, b) ≥ S) ∧ 
                              (∃ a b, a ≠ b ∧ a < 2012 ∧ b < 2012 ∧ f(a, b) = S) := 
begin
  use 502,
  -- Proof to be filled in
  sorry
end

end minimum_f_value_l691_691408


namespace range_of_m_l691_691435

theorem range_of_m (m : ℝ) (p : |m + 1| ≤ 2) (q : ¬(m^2 - 4 ≥ 0)) : -2 < m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l691_691435


namespace polynomial_solution_l691_691725

variable {R : Type*} [CommRing R]

theorem polynomial_solution (p : Polynomial R) :
  (∀ (a b c : R), 
    p.eval (a + b - 2 * c) + p.eval (b + c - 2 * a) + p.eval (c + a - 2 * b)
      = 3 * p.eval (a - b) + 3 * p.eval (b - c) + 3 * p.eval (c - a)
  ) →
  ∃ (a1 a2 : R), p = Polynomial.C a2 * Polynomial.X^2 + Polynomial.C a1 * Polynomial.X :=
by
  sorry

end polynomial_solution_l691_691725


namespace find_a_l691_691430

variable (a : ℝ)
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (1, 0)
def P (x : ℝ) : ℝ × ℝ := (x, Real.log x / Real.log a)

-- Define the function f(x) = x - log_a(x) + 1
def f (x : ℝ) : ℝ := x - Real.log x / Real.log a + 1

-- The main statement to be proved
theorem find_a (a > 0) (h : ∀ x > 0, f x ≥ 2 ∧ ∃ x, f x = 2) : a = Real.exp 1 :=
sorry

end find_a_l691_691430


namespace squares_arrangement_l691_691130

noncomputable def arrangement_possible (n : ℕ) (cond : n ≥ 5) : Prop :=
  ∃ (position : ℕ → ℕ × ℕ),
    (∀ i, 1 ≤ i ∧ i ≤ n → 
        ∃ j k, j ≠ k ∧ 
             dist (position i) (position j) = 1 ∧
             dist (position i) (position k) = 1)

theorem squares_arrangement (n : ℕ) (hn : n ≥ 5) :
  arrangement_possible n hn :=
  sorry

end squares_arrangement_l691_691130


namespace find_a_l691_691028

def f (a x : ℝ) : ℝ := a * Real.exp x + x^2 - 8 * x

theorem find_a (a : ℝ) (h : (fun x => a * Real.exp x + x^2 - 8 * x).deriv 0 = -4) :
  a = 4 := by
  sorry

end find_a_l691_691028


namespace square_position_2007_l691_691911

def square_transformations (n : ℕ) : string :=
  let cycle := ["ABCD", "DABC", "CBAD", "DCBA"] in
  cycle[n % 4]

theorem square_position_2007 : square_transformations 2007 = "DCBA" :=
by
  sorry

end square_position_2007_l691_691911


namespace inequality_for_positive_reals_l691_691159

variable (a b c : ℝ)

theorem inequality_for_positive_reals (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := by
  sorry

end inequality_for_positive_reals_l691_691159


namespace perfect_square_of_odd_sigma_l691_691621

theorem perfect_square_of_odd_sigma (N : ℕ)
  (h1 : N > 0)
  (h2 : ∃ (p : ℕ → ℕ) (e : ℕ → ℕ) (k : ℕ), 
        N = (List.prod (List.map (λ i, (p i) ^ (e i)) (List.range k))) ∧
        (∀ i < k, Nat.prime (p i) ∧ (e i) > 0)) 
  (h3 : ∃ (e : ℕ → ℕ) (k : ℕ), 
        σ N = List.prod (List.map (λ i, (e i + 1)) (List.range k)) ∧
        ∀ i < k, (e i) > 0) :
  ∃ M : ℕ, N = M * M :=
sorry

end perfect_square_of_odd_sigma_l691_691621


namespace coeff_of_x5_in_expansion_l691_691524

theorem coeff_of_x5_in_expansion : 
  let expr := (2 + sqrt x - (x ^ 2018 / 2017)) ^ 12 in
  (coeff expr 5) = 264 :=
sorry

end coeff_of_x5_in_expansion_l691_691524


namespace exists_odd_midpoint_l691_691870

open Real

variable {p : Fin 1993 → (ℤ × ℤ)}

-- Condition 1: There are 1993 distinct points with integer coordinates
def points_distinct (p : Fin 1993 → (ℤ × ℤ)) : Prop :=
  ∀ i j, i ≠ j → p i ≠ p j

-- Condition 2: Each segment does not contain other lattice points
def no_lattice_points (p : Fin 1993 → (ℤ × ℤ)) : Prop :=
  ∀ i, i < 1992 → ∀ k, k ≠ 0 → k ≠ 1 → (k : ℝ) * (fst (p ⟨i, sorry⟩) - fst (p ⟨i+1, sorry⟩), snd (p ⟨i, sorry⟩) - snd (p ⟨i+1, sorry⟩)) ∉ ℤ × ℤ

-- Proof Goal: There exists at least one segment containing Q satisfying the given conditions
theorem exists_odd_midpoint (p : Fin 1993 → (ℤ × ℤ)) 
  (h1 : points_distinct p) 
  (h2 : no_lattice_points p) : 
  ∃ i, i < 1992 ∧ 
    let m := (fst (p ⟨i, sorry⟩) + fst (p ⟨i+1, sorry⟩)) / 2,
        n := (snd (p ⟨i, sorry⟩) + snd (p ⟨i+1, sorry⟩)) / 2 in 
    odd (2 * m) ∧ odd (2 * n) := 
sorry

end exists_odd_midpoint_l691_691870


namespace sin_x0_min_l691_691305

-- Define the function f
def f (x : ℝ) : ℝ := Real.sin x - (Real.cos x) ^ 2 + 1

-- Define the condition: x0 is the minimum point of the function f
def is_min_point (f : ℝ → ℝ) (x0 : ℝ) : Prop := ∀ x : ℝ, f x0 ≤ f x

-- Define the math problem to prove
theorem sin_x0_min (x0 : ℝ) (h : is_min_point f x0) : Real.sin x0 = -1/2 :=
sorry

end sin_x0_min_l691_691305


namespace value_of_f_at_neg_2pi_over_3_l691_691912

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom periodic_function : ∀ x : ℝ, f (x + π / 2) = f (x - π / 2)
axiom tan_on_interval : ∀ x : ℝ, -π / 2 < x ∧ x < 0 → f x = Real.tan x

theorem value_of_f_at_neg_2pi_over_3 : f (-2 * π / 3) = -Real.sqrt 3 := by
  sorry

end value_of_f_at_neg_2pi_over_3_l691_691912


namespace christina_total_weekly_distance_l691_691368

-- Definitions based on conditions
def daily_distance_to_school : ℕ := 7
def daily_round_trip : ℕ := 2 * daily_distance_to_school
def days_per_week : ℕ := 5
def extra_trip_distance_one_way : ℕ := 2
def extra_trip_round_trip : ℕ := 2 * extra_trip_distance_one_way

-- Theorem statement
theorem christina_total_weekly_distance :
  let weekly_distance := daily_round_trip * days_per_week in
  let final_week_distance := weekly_distance + extra_trip_round_trip in
  final_week_distance = 74 := by
sorry

end christina_total_weekly_distance_l691_691368


namespace elyse_passing_threshold_l691_691717

def total_questions : ℕ := 90
def programming_questions : ℕ := 20
def database_questions : ℕ := 35
def networking_questions : ℕ := 35
def programming_correct_rate : ℝ := 0.8
def database_correct_rate : ℝ := 0.5
def networking_correct_rate : ℝ := 0.7
def passing_percentage : ℝ := 0.65

theorem elyse_passing_threshold :
  let programming_correct := programming_correct_rate * programming_questions
  let database_correct := database_correct_rate * database_questions
  let networking_correct := networking_correct_rate * networking_questions
  let total_correct := programming_correct + database_correct + networking_correct
  let required_to_pass := passing_percentage * total_questions
  total_correct = required_to_pass → 0 = 0 :=
by
  intro _h
  sorry

end elyse_passing_threshold_l691_691717


namespace difference_of_squares_l691_691597

theorem difference_of_squares (x y : ℕ) (h1 : x + y = 26) (h2 : x * y = 168) : x^2 - y^2 = 52 := by
  sorry

end difference_of_squares_l691_691597


namespace m_1_sufficient_but_not_necessary_l691_691743

def lines_parallel (m : ℝ) : Prop :=
  let l1_slope := -m
  let l2_slope := (2 - 3 * m) / m
  l1_slope = l2_slope

theorem m_1_sufficient_but_not_necessary (m : ℝ) (h₁ : lines_parallel m) : 
  (m = 1) → (∃ m': ℝ, lines_parallel m' ∧ m' ≠ 1) :=
sorry

end m_1_sufficient_but_not_necessary_l691_691743


namespace polar_curve_symmetry_l691_691529

noncomputable def is_symmetric_about_line {ρ θ : ℝ} (polar_eq : ρ = 4 * Real.sin (θ - Real.pi / 3)) : Prop :=
  ∀ (ρ₁ ρ₂ : ℝ) (θ₁ θ₂ : ℝ), (ρ₁ = 4 * Real.sin (θ₁ - Real.pi / 3)) → 
                             (ρ₂ = 4 * Real.sin (θ₂ - Real.pi / 3)) → 
                             (θ₁ = 5 * Real.pi / 6 - θ₂)

theorem polar_curve_symmetry :
  is_symmetric_about_line (polar_eq := 4 * Real.sin (θ - Real.pi / 3)) :=
  sorry

end polar_curve_symmetry_l691_691529


namespace laura_circles_ways_l691_691114

def is_integer_fraction (a b c d : ℕ) : Prop :=
  (a + b + c) % d = 0

def valid_combinations : list (ℕ × ℕ × ℕ × ℕ) :=
  [(2, 5, 11, 3), (2, 5, 11, 6)]

theorem laura_circles_ways :
  { comb : ℕ × ℕ × ℕ × ℕ // comb ∈ valid_combinations } = 2 :=
by
  sorry

end laura_circles_ways_l691_691114


namespace square_consecutive_odd_sum_l691_691891

theorem square_consecutive_odd_sum (n : ℕ) (h : 1 < n) : ∑ k in Finset.range n + 1, (2 * k - 1) = n^2 := by
  sorry

end square_consecutive_odd_sum_l691_691891


namespace average_cd_l691_691593

theorem average_cd (c d : ℝ) (h : (4 + 6 + 8 + c + d) / 5 = 18) : (c + d) / 2 = 36 := 
by
  -- The proof goes here
  sorry

end average_cd_l691_691593


namespace rainfall_second_week_l691_691967

theorem rainfall_second_week (r1 r2 : ℝ) (h1 : r1 + r2 = 40) (h2 : r2 = 1.5 * r1) : r2 = 24 :=
by
  sorry

end rainfall_second_week_l691_691967


namespace equation_represents_two_lines_l691_691288

def coefficients (k : ℝ) : Prop :=
  let a := 1
  let h := 1 / 2
  let b := -6
  let g := -10
  let f := -10
  let c := k
  det (matrix.of ![
    [a, h, g],
    [h, b, f],
    [g, f, c]
  ]) = 0

theorem equation_represents_two_lines (k : ℝ) :
  coefficients k ↔ k = 80 :=
begin
  sorry
end

end equation_represents_two_lines_l691_691288


namespace symmetric_point_about_origin_l691_691183

theorem symmetric_point_about_origin (P Q : ℤ × ℤ) (h : P = (-2, -3)) : Q = (2, 3) :=
by
  sorry

end symmetric_point_about_origin_l691_691183


namespace number_of_valid_permutations_l691_691476

-- Define the digits and the count of occurrences of each digit
def digits := [3, 1, 0, 3]

-- Define a predicate that ensures the number is a four-digit number and does not start with 0
def is_valid_number (n : List ℕ) : Prop :=
  n.length = 4 ∧ n.head ≠ 0

-- Define the main theorem
theorem number_of_valid_permutations : 
  (List.permutations digits).count is_valid_number = 9 := 
sorry

end number_of_valid_permutations_l691_691476


namespace bens_old_car_cost_l691_691692

theorem bens_old_car_cost :
  ∃ (O N : ℕ), N = 2 * O ∧ O = 1800 ∧ N = 1800 + 2000 ∧ O = 1900 :=
by 
  sorry

end bens_old_car_cost_l691_691692


namespace binary_multiplication_l691_691733

theorem binary_multiplication :
  let b1 := 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0,  -- 1101_2 in decimal
      b2 := 1 * 2^2 + 1 * 2^1 + 1 * 2^0,           -- 111_2 in decimal
      product := 1 * 2^6 + 0 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0  -- 1001111_2 in decimal
  in
  b1 * b2 = product := by
  sorry

end binary_multiplication_l691_691733


namespace div_81_expr_is_integer_l691_691889

theorem div_81_expr_is_integer (n : ℤ) : (1 / 81 : ℚ) * (10 ^ n - 1) - (n / 9 : ℚ) ∈ ℤ := 
sorry

end div_81_expr_is_integer_l691_691889


namespace rectangle_area_ratio_k_l691_691925

theorem rectangle_area_ratio_k (d : ℝ) (l w : ℝ) (h1 : l / w = 5 / 2) (h2 : d^2 = l^2 + w^2) :
  ∃ k : ℝ, k = 10 / 29 ∧ (l * w = k * d^2) :=
by {
  -- proof steps will go here
  sorry
}

end rectangle_area_ratio_k_l691_691925


namespace compare_a_b_l691_691133

theorem compare_a_b (x : ℝ) : let a := 3 * x^2 - x + 1
                               b := 2 * x^2 + x
                           in a ≥ b :=
by
  sorry

end compare_a_b_l691_691133


namespace smallest_two_digit_prime_with_conditions_l691_691404

open Nat

def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def is_composite (n : ℕ) : Prop := n ≥ 2 ∧ ∃ m, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

def reverse_digits (n : ℕ) : ℕ :=
  let digits := to_digits 10 n
  of_digits 10 digits.reverse

noncomputable def smallest_prime_with_conditions : ℕ :=
  23 -- The smallest prime satisfying the conditions, verified manually

theorem smallest_two_digit_prime_with_conditions :
  ∃ p, is_prime p ∧ (10 ≤ p ∧ p < 100) ∧ (p % 100 / 10 = 2) ∧ is_composite (reverse_digits p) ∧ 
  (∀ q, is_prime q ∧ (10 ≤ q ∧ q < 100) ∧ (q % 100 / 10 = 2) ∧ is_composite (reverse_digits q) → p ≤ q) := 
begin
  use 23,
  split, {
    dsimp [is_prime],
    split,
    { exact dec_trivial }, -- 23 ≥ 2
    { intros m hm,
      interval_cases m; simp [hm],
    },
  },
  split, {
    split,
    { norm_num }, -- 10 ≤ 23
    { norm_num }, -- 23 < 100
  },
  split, {
    norm_num, -- tens digit is 2
  },
  split, {
    dsimp [is_composite, reverse_digits],
    use 2,
    split,
    { exact dec_trivial, }, -- 10 ≤ 32
    split,
    { exact dec_trivial, }, -- 2 ≠ 1
    { exact dec_trivial },  -- 2 ≠ 32
  },
  intros q hq,
  rcases hq with ⟨prm_q, ⟨hq_low, hq_high⟩, t_d, q_comp⟩,
  rw t_d at *,
  linarith,
end

end smallest_two_digit_prime_with_conditions_l691_691404


namespace expression_never_equals_33_l691_691560

theorem expression_never_equals_33 (x y : ℤ) : 
  x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 := 
sorry

end expression_never_equals_33_l691_691560


namespace inequality_x2_y2_l691_691759

theorem inequality_x2_y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) : 
  |x^2 + y^2| / (x + y) < |x^2 - y^2| / (x - y) :=
sorry

end inequality_x2_y2_l691_691759


namespace sixteenth_term_l691_691146

theorem sixteenth_term :
  (-1)^(16+1) * Real.sqrt (3 * (16 - 1)) = -3 * Real.sqrt 5 :=
by sorry

end sixteenth_term_l691_691146


namespace Robert_diff_C_l691_691557

/- Define the conditions as hypotheses -/
variables (C : ℕ) -- Assuming the number of photos Claire has taken as a natural number.

-- Lisa has taken 3 times as many photos as Claire.
def Lisa_photos := 3 * C

-- Robert has taken the same number of photos as Lisa.
def Robert_photos := Lisa_photos C -- which will be 3 * C

-- Proof of the difference.
theorem Robert_diff_C : (Robert_photos C) - C = 2 * C :=
by
  sorry

end Robert_diff_C_l691_691557


namespace cylindrical_tank_water_depth_l691_691316

noncomputable def tankWaterDepth : ℝ :=
  let r := 3
  let h := 2
  let tankLength := 10
  let A_segment := (9 : ℝ) * real.arccos (1 / 3) - real.sqrt 8
  let V := A_segment * tankLength
  V / (9 * real.pi)

theorem cylindrical_tank_water_depth :
  let r := 3
  let h := 2
  let tankLength := 10
  let A_segment := (9 : ℝ) * real.arccos (1 / 3) - real.sqrt 8
  let V := A_segment * tankLength
  (V / (9 * real.pi)).round = 3 :=
by
  let r := 3
  let h := 2
  let tankLength := 10
  let A_segment := (9 : ℝ) * real.arccos (1 / 3) - real.sqrt 8
  let V := A_segment * tankLength
  show (V / (9 * real.pi)).round = 3
  sorry

end cylindrical_tank_water_depth_l691_691316


namespace inscribed_square_inequality_l691_691972

open Real

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def area (a b c : ℝ) : ℝ := 
  let s := semiperimeter a b c
  (s * (s - a) * (s - b) * (s - c)).sqrt

theorem inscribed_square_inequality 
  (a b c x y z : ℝ)
  (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0)
  (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let s := semiperimeter a b c in
  let F := area a b c in
  (1/x) + (1/y) + (1/z) ≤ s * (2 + sqrt 3) / (2 * F) :=
  sorry

end inscribed_square_inequality_l691_691972


namespace imaginary_part_conjugate_l691_691590

open Complex

-- Define the problem: Prove that the imaginary part of the conjugate of \(\frac{1-i}{1+i}\) is \(1\)
theorem imaginary_part_conjugate :
  let z := (1 - I) / (1 + I)
  imaginary_part (conj z) = 1 :=
by
  -- Proof details would go here
  sorry

end imaginary_part_conjugate_l691_691590


namespace inradius_of_regular_triangular_pyramid_l691_691010

noncomputable def triangle_pyramid_inradius : ℝ := 
  let base_side_length := 1
  let height := Real.sqrt 2 
  Real.sqrt 2 / 6 

theorem inradius_of_regular_triangular_pyramid :
  (base_side_length = 1) ∧ (height = Real.sqrt 2) → triangle_pyramid_inradius = Real.sqrt 2 / 6 := 
by
  intros
  unfold triangle_pyramid_inradius
  sorry

end inradius_of_regular_triangular_pyramid_l691_691010


namespace jake_fewer_peaches_than_steven_l691_691539

theorem jake_fewer_peaches_than_steven :
  ∀ (jill steven jake : ℕ),
    jill = 87 →
    steven = jill + 18 →
    jake = jill + 13 →
    steven - jake = 5 :=
by
  intros jill steven jake hjill hsteven hjake
  sorry

end jake_fewer_peaches_than_steven_l691_691539


namespace angle_QIS_69_degrees_l691_691105

theorem angle_QIS_69_degrees 
  (P Q R S T U I : Type) 
  [IsTriangle P Q R] 
  [AngleBisectors P S Q, Q T R, R U P]
  [Incenter I P Q R S T U] 
  (h1 : ∡ P R Q = 42) : 
  ∡ Q I S = 69 :=
by
  sorry

end angle_QIS_69_degrees_l691_691105


namespace triangle_is_isosceles_l691_691469

theorem triangle_is_isosceles (a b c : ℝ) 
  (h1 : a^2 - 4 * b = 7)
  (h2 : b^2 - 4 * c = -6)
  (h3 : c^2 - 6 * a = -18) : 
  (a = b ∨ a = c ∨ b = c) :=
by {
  have : (a - 3)^2 + (b - 2)^2 + (c - 2)^2 = 0,
  {
    sorry
  },
  have h4 : a = 3 ∧ b = 2 ∧ c = 2,
  {
    sorry
  },
  cases h4 with ha hb hc,
  left,
  exact hb.symm,
}

end triangle_is_isosceles_l691_691469


namespace half_abs_diff_squares_l691_691260

theorem half_abs_diff_squares (a b : ℤ) (h₁ : a = 20) (h₂ : b = 15) : 
  (|a^2 - b^2| / 2 : ℚ) = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691260


namespace combined_height_l691_691877

/-- Given that Mr. Martinez is two feet taller than Chiquita and Chiquita is 5 feet tall, prove that their combined height is 12 feet. -/
theorem combined_height (h_chiquita : ℕ) (h_martinez : ℕ) 
  (h1 : h_chiquita = 5) (h2 : h_martinez = h_chiquita + 2) : 
  h_chiquita + h_martinez = 12 :=
by sorry

end combined_height_l691_691877


namespace ice_cream_scoops_l691_691678

theorem ice_cream_scoops (total_money : ℝ) (spent_on_restaurant : ℝ) (remaining_money : ℝ) 
  (cost_per_scoop_after_discount : ℝ) (remaining_each : ℝ) 
  (initial_savings : ℝ) (service_charge_percent : ℝ) (restaurant_percent : ℝ) 
  (ice_cream_discount_percent : ℝ) (money_each : ℝ) :
  total_money = 400 ∧
  spent_on_restaurant = 320 ∧
  remaining_money = 80 ∧
  cost_per_scoop_after_discount = 5 ∧
  remaining_each = 8 ∧
  initial_savings = 200 ∧
  service_charge_percent = 0.20 ∧
  restaurant_percent = 0.80 ∧
  ice_cream_discount_percent = 0.10 ∧
  money_each = 5 → 
  ∃ (scoops_per_person : ℕ), scoops_per_person = 5 :=
by
  sorry

end ice_cream_scoops_l691_691678


namespace exists_points_with_distance_one_l691_691541

open Set Real

noncomputable def regionK (K : Set (ℝ × ℝ)) : Prop :=
  Convex ℝ K ∧ (∃ (L : List (Set (ℝ × ℝ))), Union L = boundary K ∧ ∀ l ∈ L, ∃ a b : ℝ × ℝ, l = segment ℝ a b) ∧ measure_univ K ≥ π / 4

theorem exists_points_with_distance_one {K : Set (ℝ × ℝ)} (hK : regionK K) :
  ∃ (P Q : ℝ × ℝ), P ∈ K ∧ Q ∈ K ∧ dist P Q = 1 :=
sorry

end exists_points_with_distance_one_l691_691541


namespace half_abs_diff_squares_l691_691231

theorem half_abs_diff_squares (a b : ℤ) (ha : a = 20) (hb : b = 15) : 
  (|a^2 - b^2| / 2) = 87.5 :=
by
  sorry

end half_abs_diff_squares_l691_691231


namespace quadratic_has_two_distinct_real_roots_l691_691078

theorem quadratic_has_two_distinct_real_roots (m : ℝ) : 
  (∃ (x y : ℝ), x ≠ y ∧ x^2 - 2 * x + m = 0 ∧ y^2 - 2 * y + m = 0) ↔ m < 1 :=
sorry

end quadratic_has_two_distinct_real_roots_l691_691078


namespace hexagon_ratio_l691_691544

noncomputable def ratio_of_hexagon_areas (s : ℝ) : ℝ :=
  let area_ABCDEF := (3 * Real.sqrt 3 / 2) * s^2
  let side_smaller := (3 * s) / 2
  let area_smaller := (3 * Real.sqrt 3 / 2) * side_smaller^2
  area_smaller / area_ABCDEF

theorem hexagon_ratio (s : ℝ) : ratio_of_hexagon_areas s = 9 / 4 :=
by
  sorry

end hexagon_ratio_l691_691544


namespace average_age_of_boys_l691_691180

def boys_age_proportions := (3, 5, 7)
def eldest_boy_age := 21

theorem average_age_of_boys : 
  ∃ (x : ℕ), 7 * x = eldest_boy_age ∧ (3 * x + 5 * x + 7 * x) / 3 = 15 :=
by
  sorry

end average_age_of_boys_l691_691180


namespace most_stable_shooting_performance_l691_691002

-- Given conditions
def variance_A := 0.5
def variance_B := 0.6
def variance_C := 0.9
def variance_D := 1.0

-- Proof problem statement
theorem most_stable_shooting_performance : 
  variance_A < variance_B ∧ variance_B < variance_C ∧ variance_C < variance_D → 
  "A has the most stable shooting performance" :=
by
  sorry

end most_stable_shooting_performance_l691_691002


namespace range_of_x_l691_691585

theorem range_of_x (x p : ℝ) (h₀ : 0 ≤ p ∧ p ≤ 4) :
  x^2 + p * x > 4 * x + p - 3 → (x < 1 ∨ x > 3) :=
sorry

end range_of_x_l691_691585


namespace inequality_proof_l691_691564

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (b + c)) + (1 / (a + c)) + (1 / (a + b)) ≥ 9 / (2 * (a + b + c)) :=
by
  sorry

end inequality_proof_l691_691564


namespace ending_number_l691_691632

theorem ending_number (h : ∃ n, 3 * n = 99 ∧ n = 33) : ∃ m, m = 99 :=
by
  sorry

end ending_number_l691_691632


namespace solve_for_y_l691_691172

theorem solve_for_y (y : Real) (h : (50 ^ 2) ^ 2 * 5 ^ (y - 4) = 10 ^ y) : y = 13 := by
  sorry

end solve_for_y_l691_691172


namespace correct_calculation_B_l691_691627

theorem correct_calculation_B :
  (∀ (a : ℕ), 3 * a^3 * 2 * a^2 ≠ 6 * a^6) ∧
  (∀ (x : ℕ), 2 * x^2 * 3 * x^2 = 6 * x^4) ∧
  (∀ (x : ℕ), 3 * x^2 * 4 * x^2 ≠ 12 * x^2) ∧
  (∀ (y : ℕ), 5 * y^3 * 3 * y^5 ≠ 8 * y^8) →
  (∀ (x : ℕ), 2 * x^2 * 3 * x^2 = 6 * x^4) := 
by
  sorry

end correct_calculation_B_l691_691627


namespace max_min_values_exists_k_l691_691740

-- Definitions for vectors and conditions
def vec_a (θ : ℝ) : ℝ × ℝ := (Real.cos (3 * θ / 2), Real.sin (3 * θ / 2))
def vec_b (θ : ℝ) : ℝ × ℝ := (Real.cos (θ / 2), -Real.sin (θ / 2))
def θ_in_range (θ : ℝ) : Prop := 0 ≤ θ ∧ θ ≤ Real.pi / 3

-- Dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Magnitude of a vector
def magnitude (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1 ^ 2 + u.2 ^ 2)

-- Part I: Maximum and Minimum Values
theorem max_min_values (θ : ℝ) (hθ : θ_in_range θ) :
  let a := vec_a θ
  let b := vec_b θ
  let numerator := dot_product a b
  let denominator := magnitude (a + b)
  let frac := numerator / denominator
  frac ≤ 1 / 2 ∧ frac ≥ -1 / 2 :=
by sorry

-- Part II: Existence of k
theorem exists_k (θ : ℝ) (hθ : θ_in_range θ) :
  ∃ k : ℝ, magnitude (k • vec_a θ + vec_b θ) = Real.sqrt 3 * magnitude (vec_a θ - k • vec_b θ) :=
by sorry

end max_min_values_exists_k_l691_691740


namespace triangle_inequality_l691_691868

variables {l_a l_b l_c m_a m_b m_c h_n m_n h_h_n m_m_p : ℝ}

-- Assuming some basic properties for the variables involved (all are positive in their respective triangle context)
axiom pos_l_a : 0 < l_a
axiom pos_l_b : 0 < l_b
axiom pos_l_c : 0 < l_c
axiom pos_m_a : 0 < m_a
axiom pos_m_b : 0 < m_b
axiom pos_m_c : 0 < m_c
axiom pos_h_n : 0 < h_n
axiom pos_m_n : 0 < m_n
axiom pos_h_h_n : 0 < h_h_n
axiom pos_m_m_p : 0 < m_m_p

theorem triangle_inequality :
  (h_n / m_n) + (h_n / h_h_n) + (l_c / m_m_p) > 1 :=
sorry

end triangle_inequality_l691_691868


namespace alpha_half_quadrant_l691_691016

open Real

theorem alpha_half_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * π - π / 2 < α ∧ α < 2 * k * π) :
  (∃ k1 : ℤ, (2 * k1 + 1) * π - π / 4 < α / 2 ∧ α / 2 < (2 * k1 + 1) * π) ∨
  (∃ k2 : ℤ, 2 * k2 * π - π / 4 < α / 2 ∧ α / 2 < 2 * k2 * π) :=
sorry

end alpha_half_quadrant_l691_691016


namespace median_eq_right_triangle_m_l691_691101

-- Define the points A and B
def A : (ℝ × ℝ) := (-2, 0)
def B : (ℝ × ℝ) := (0, 4)

-- Define the line equation of points on which C lies as a predicate
def on_line_C (m n : ℝ) : Prop := m - 3 * n - 3 = 0

-- Part 1: Given m=3, prove the median equation.
theorem median_eq (m n : ℝ) (h_C : on_line_C m n) (hm : m = 3) :
  ∃ k y, (k=y) ∧ (k=2) ∧ (x + k*y - 3 = 0) :=
by {
  sorry
}

-- Part 2: If triangle ABC is a right triangle, prove possible values of m.
theorem right_triangle_m (m n : ℝ) (h_C : on_line_C m n) :
  (∃ a b c, a^2 + b^2=c^2) → (m = 0 ∨ m = 6) :=
by {
  sorry
}

end median_eq_right_triangle_m_l691_691101


namespace arrange_squares_l691_691128

theorem arrange_squares (n : ℕ) (h: n ≥ 5) : 
  ∃ (arrangement : list (ℕ × ℕ × ℕ × ℕ)), 
    (∀ (sq ∈ arrangement), sq.1 = sq.2 ∧ sq.3 = sq.4 ∧ sq.1 < sq.2 < ... < sq.n) ∧ 
    (∀ (sq1 sq2 ∈ arrangement), sq1 ≠ sq2 → 
       (sq1.1 = sq2.3 ∧ sq1.2 = sq2.4) ∨ 
       (sq1.3 = sq2.1 ∧ sq1.4 = sq2.2)) := 
sorry

end arrange_squares_l691_691128


namespace value_of_y_when_x_is_zero_l691_691037

noncomputable def quadratic_y (h x : ℝ) : ℝ := -(x + h)^2

theorem value_of_y_when_x_is_zero :
  ∀ (h : ℝ), (∀ x, x < -3 → quadratic_y h x < quadratic_y h (-3)) →
            (∀ x, x > -3 → quadratic_y h x < quadratic_y h (-3)) →
            quadratic_y h 0 = -9 :=
by
  sorry

end value_of_y_when_x_is_zero_l691_691037


namespace acute_triangle_area_range_l691_691855

theorem acute_triangle_area_range
  (a b c : ℝ) (A B C : ℝ)
  (h_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (h_sum_angles : A + B + C = π)
  (h_angle_A : A = π / 3)
  (h_side_a : a = 2 * √3)
  (h_law_of_sines_b : b = a * (sin B) / (sin A))
  (h_law_of_sines_c : c = a * (sin C) / (sin A)) :
  2 * √3 < (1 / 2) * b * c * sin A ∧ (1 / 2) * b * c * sin A ≤ 3 * √3 :=
sorry

end acute_triangle_area_range_l691_691855


namespace addition_correct_l691_691306

-- Define the integers involved
def num1 : ℤ := 22
def num2 : ℤ := 62
def result : ℤ := 84

-- Theorem stating the relationship between the given numbers
theorem addition_correct : num1 + num2 = result :=
by {
  -- proof goes here
  sorry
}

end addition_correct_l691_691306


namespace ribbon_segment_length_l691_691293

theorem ribbon_segment_length :
  ∀ (ribbon_length : ℚ) (segments : ℕ), ribbon_length = 4/5 → segments = 3 → 
  (ribbon_length / segments) = 4/15 :=
by
  intros ribbon_length segments h1 h2
  sorry

end ribbon_segment_length_l691_691293


namespace lunks_needed_for_bananas_l691_691053

theorem lunks_needed_for_bananas (trade_rate : ℚ) (bananas_rate : ℚ) (bananas_required : ℚ) (conversion_rate: ℚ) :
  (bananas_required = 20) ∧ 
  (bananas_rate = 6 / 5) ∧ 
  (trade_rate = 3 / 2) ∧ 
  (conversion_rate = (3 / 2) * (6 / 5)) →
  let kunks_needed := bananas_required * bananas_rate in
  let lunks_needed := kunks_needed * trade_rate in
  lunks_needed = 36 := by
  sorry

end lunks_needed_for_bananas_l691_691053
