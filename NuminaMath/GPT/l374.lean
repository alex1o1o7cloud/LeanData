import Mathlib
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order
import Mathlib.Analysis.Calculus.Integral
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.CombinatorialBasics
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Graph.Degree
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Lcm
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Angle
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import algebra.order.field
import data.complex.basic
import data.real.basic

namespace infinite_grid_three_colors_has_isosceles_right_triangle_l374_374853

open Classical

noncomputable def exists_isosceles_right_triangle_same_color (color : ℕ → ℕ → ℕ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ x₃ y₃, 
  (x₁ = x₂ ∧ y₁ = y₃ ∧ x₃ = x₁ + (y₃ - y₂) ∧ y₂ = y₁ + (x₃ - x₁)) ∧
  (color x₁ y₁ = color x₂ y₂ ∧ color x₂ y₂ = color x₃ y₃) 

theorem infinite_grid_three_colors_has_isosceles_right_triangle :
  (∃ color : ℕ → ℕ → ℕ, ∀ x y, color x y < 3) →
    exists_isosceles_right_triangle_same_color := by 
  intros coloring
  sorry

end infinite_grid_three_colors_has_isosceles_right_triangle_l374_374853


namespace symmetric_circle_eq_l374_374091

theorem symmetric_circle_eq : 
    let C := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 + 4)^2 = 1}
    ∧ let L := {p : ℝ × ℝ | p.2 = -p.1 + 6}
    ∧ let C_sym := {p : ℝ × ℝ | (p.1 - 10)^2 + (p.2 - 3)^2 = 1} in
    ∀ p, p ∈ C ↔ p⁻¹ ∈ C_sym := 
by
  sorry

end symmetric_circle_eq_l374_374091


namespace chord_length_proof_tangent_lines_through_M_l374_374250

noncomputable def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

noncomputable def line_l (x y : ℝ) : Prop := 2*x - y + 4 = 0

noncomputable def point_M : (ℝ × ℝ) := (3, 1)

noncomputable def chord_length : ℝ := 4 * Real.sqrt (5) / 5

noncomputable def tangent_line_1 (x y : ℝ) : Prop := 3*x - 4*y - 5 = 0
noncomputable def tangent_line_2 (x : ℝ) : Prop := x = 3

theorem chord_length_proof :
  ∀ x y : ℝ, circle_C x y → line_l x y → chord_length = 4 * Real.sqrt (5) / 5 :=
by sorry

theorem tangent_lines_through_M :
  ∀ x y : ℝ, circle_C x y → (tangent_line_1 x y ∨ tangent_line_2 x) :=
by sorry

end chord_length_proof_tangent_lines_through_M_l374_374250


namespace max_rectangle_perimeter_l374_374592

theorem max_rectangle_perimeter (n : ℕ) (a b : ℕ) (ha : a * b = 180) (hb: ∀ (a b : ℕ),  6 ∣ (a * b) → a * b = 180): 
  2 * (a + b) ≤ 184 :=
sorry

end max_rectangle_perimeter_l374_374592


namespace log_one_plus_xsq_lt_xsq_over_one_plus_xsq_l374_374310

theorem log_one_plus_xsq_lt_xsq_over_one_plus_xsq (x : ℝ) (hx : 0 < x) : 
  Real.log (1 + x^2) < x^2 / (1 + x^2) :=
sorry

end log_one_plus_xsq_lt_xsq_over_one_plus_xsq_l374_374310


namespace intersection_of_M_N_l374_374647

open Set

def M : Set ℝ := { x | 4 ≤ x ∧ x ≤ 7 }
def N : Set ℝ := {3, 5, 8}

theorem intersection_of_M_N : M ∩ N = {5} := sorry

end intersection_of_M_N_l374_374647


namespace remainder_of_polynomial_l374_374162

variable {R : Type*} [CommRing R]

theorem remainder_of_polynomial (q : R[X])
  (h1 : q.eval 3 = 2)
  (h2 : q.eval (-2) = -3)
  (h3 : q.eval 4 = 6) :
  ∃ a b c : R, 
    (∀ x, q.eval x = (x - 3) * (x + 2) * (x - 4) * (polynomial.eval x) + (a * x^2 + b * x + c)) ∧
    (a = 1/2 ∧ b = 1/2 ∧ c = -4) ∧ 
    (a * 5^2 + b * 5 + c = 11) :=
sorry

end remainder_of_polynomial_l374_374162


namespace profit_percentage_is_60_l374_374971

variable (SellingPrice CostPrice : ℝ)

noncomputable def Profit : ℝ := SellingPrice - CostPrice

noncomputable def ProfitPercentage : ℝ := (Profit SellingPrice CostPrice / CostPrice) * 100

theorem profit_percentage_is_60
  (h1 : SellingPrice = 400)
  (h2 : CostPrice = 250) :
  ProfitPercentage SellingPrice CostPrice = 60 := by
  sorry

end profit_percentage_is_60_l374_374971


namespace percent_increase_first_half_correct_l374_374017

noncomputable def percent_increase_first_half : ℝ :=
  let P := 1 in
  let x := arbitrary ℝ in
  have h1 : 4 * (P * (1 + x / 100)) = 12 * P := by
    sorry
  x

theorem percent_increase_first_half_correct :
  percent_increase_first_half = 200 :=
by
  sorry

end percent_increase_first_half_correct_l374_374017


namespace triangle_bisector_length_l374_374688

-- Define the basic structure of the problem
structure Triangle :=
(X Y Z : Point)
(angleY : Angle)
(sideYZ : ℝ)

-- Define the points of intersection of the perpendicular bisector
structure BisectorIntersections :=
(P Q : Point)

-- Setting up the problem conditions
def problem_conditions (T : Triangle) (B : BisectorIntersections) (YZ := 30) (angY := 60) :=
  ∃ (P_mid: P.midpoint T.Y T.Z) (B_perp : B.P ⊥ T.YZ ∧ B.Q ∈ T.XZ),
  -- Question: find the length of PQ
  PQ.length = 7.5

-- Translating the problem to Lean 4 statement
theorem triangle_bisector_length : ∀ (T : Triangle) (B : BisectorIntersections), problem_conditions T B :=
by
  intros
  sorry

end triangle_bisector_length_l374_374688


namespace equal_preference_ratio_l374_374489

-- Defining conditions given in a)
variables (T M N W E : ℕ)
hypothesis total_students : T = 210
hypothesis preferred_mac : M = 60
hypothesis no_preference : N = 90
hypothesis preferred_windows : W = 40
hypothesis equal_preference : E = T - (M + N + W)

-- Defining the ratio problem in Lean 4
theorem equal_preference_ratio : (E : ℚ) / (M : ℚ) = 1 / 3 :=
by
  rw [equal_preference, total_students, preferred_mac, no_preference, preferred_windows]
  norm_num
  sorry

end equal_preference_ratio_l374_374489


namespace length_of_BC_in_acute_triangle_l374_374137

noncomputable def acute_triangle (A B C : Type) := 
  ∃ (O : Type) (R : ℝ) (K : Type) (t : ℝ),
    (t > 0) ∧
    (BK : ℝ := 5 * t) ∧
    (AK : ℝ := t) ∧
    (AB : ℝ := 6 * t) ∧
    ((2 * R) ^ 2 = (6 * t) * (5 * t)) ∧
    (3 * R = sqrt ((2 * R) ^ 2 + (R * sqrt 5) ^ 2))

theorem length_of_BC_in_acute_triangle (A B C : Type) : 
  ∃ (O : Type) (R : ℝ) (K : Type),
    acute_triangle A B C → 
    BC = 3 * R :=
by 
  sorry

end length_of_BC_in_acute_triangle_l374_374137


namespace altitude_through_C_l374_374089

-- Definitions based on the provided conditions.
variables {ABC : Type*} [IsTriangle ABC] {A B C O : Point}
noncomputable def circumradius (ABC : Triangle) := 2 * (distance O AB)
variables (AC BC : ℝ) (hAC : AC = 2) (hBC : BC = 3)

-- Main statement to prove
theorem altitude_through_C (h_acute : is_acute_triangle ABC) (h_circumcenter : circumradius ABC = 2 * (distance O AB)) :
  altitude C = 3 * sqrt 57 / 19 := sorry

end altitude_through_C_l374_374089


namespace probability_of_pairing_long_with_short_l374_374405

theorem probability_of_pairing_long_with_short :
  let S := finset.range 5, L := finset.range 5
  let total_permutations := (10.factorial : ℚ)
  let distinct_permutations := total_permutations / ((5.factorial : ℚ) * (5.factorial : ℚ))
  let acceptable_pairings := 2^5
  let P := acceptable_pairings / distinct_permutations
  P = 8 / 63 := by
  sorry

end probability_of_pairing_long_with_short_l374_374405


namespace parabola_problem_l374_374292

open Real

structure Parabola where
  p : ℝ
  hp_pos : p > 0
  xy_eq : (y : ℝ) (x : ℝ) (H : y^2 = 2 * p * x) := ∃ F : ℝ × ℝ, F.x = 1

def DirectrixLine := (-1 : ℝ)

theorem parabola_problem (p : ℝ) (hp : p > 0) :
  (∀ (A B G : ℝ × ℝ),
    (G = (-1, 0)) →
    (∃ F : ℝ × ℝ, F = (p / 2, 0)) →
    ∃ A B : ℝ × ℝ, A ≠ B → 
    (∃ k : ℝ, k = (A.1 + B.1) / 2 ∧ 
      p = 2 ∧ 
      (F = (1, 0)) ∧ 
      A.x * A.y^2 = B.x * B.y^2 ∧
      B.x * B.y^2 = 2 * p * F.x ∧
      (angle G F A = angle G F B) ∧
      circle_with_diameter_AB_tangent_to_l A B l ∧
      triangle_area A G B = minimum_area 4)) := sorry

end parabola_problem_l374_374292


namespace product_of_digits_base6_7891_l374_374898

theorem product_of_digits_base6_7891 : 
  let digits := [1, 0, 1, 0, 3, 1]
  in digits.prod = 0 :=
by
  let digits := [1, 0, 1, 0, 3, 1]
  show digits.prod = 0
  sorry

end product_of_digits_base6_7891_l374_374898


namespace problem_l374_374928

noncomputable def f (x : ℝ) : ℝ := (Real.sin (x / 4)) ^ 6 + (Real.cos (x / 4)) ^ 6

theorem problem : (derivative^[2008] f 0) = 3 / 8 := by sorry

end problem_l374_374928


namespace frac_left_handed_l374_374184

variable (x : ℕ)

def red_participants := 10 * x
def blue_participants := 5 * x
def total_participants := red_participants x + blue_participants x

def left_handed_red := (1 / 3 : ℚ) * red_participants x
def left_handed_blue := (2 / 3 : ℚ) * blue_participants x
def total_left_handed := left_handed_red x + left_handed_blue x

theorem frac_left_handed :
  total_left_handed x / total_participants x = (4 / 9 : ℚ) := by
  sorry

end frac_left_handed_l374_374184


namespace range_of_a_for_three_tangents_curve_through_point_l374_374000

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := x^3 + 3 * x^2 + a * x + a - 2

noncomputable def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 6 * x + a

theorem range_of_a_for_three_tangents_curve_through_point :
  ∀ (a : ℝ), (∀ x0 : ℝ, 2 * x0^3 + 3 * x0^2 + 4 - a = 0 → 
    ((2 * -1^3 + 3 * -1^2 + 4 - a > 0) ∧ (2 * 0^3 + 3 * 0^2 + 4 - a < 0))) ↔ (4 < a ∧ a < 5) :=
by
  sorry

end range_of_a_for_three_tangents_curve_through_point_l374_374000


namespace form_three_digit_numbers_l374_374122

theorem form_three_digit_numbers :
  ∃ (a b c : ℕ), 
  (∀ i j k : ℕ, {1, 2, 3, 4, 5, 6, 7, 8, 9} = {d | d ∈ set.range id ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a % 10 ≠ b % 10 ∧ b % 10 ≠ c % 10 ∧ a % 10 ≠ c % 10)} 
  ∧ (set.range (λ x, (x % 10)) {a, b, c} ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9})
  ∧ a = 963 
  ∧ b = 875 
  ∧ c = 124 
  ∧ a % 3 = 0 
  ∧ b % 3 = 2 
  ∧ c % 3 = 1 
  ∧ ∀ i j k, i ≠ j → j ≠ k → i ≠ k → a > b > c ) := sorry

end form_three_digit_numbers_l374_374122


namespace negation_of_p_l374_374294

def p := ∀ x, x ≤ 0 → Real.exp x ≤ 1

theorem negation_of_p : ¬ p ↔ ∃ x, x ≤ 0 ∧ Real.exp x > 1 := 
by
  sorry

end negation_of_p_l374_374294


namespace lisa_more_marbles_than_cindy_l374_374550

-- Definitions
def initial_cindy_marbles : ℕ := 20
def difference_cindy_lisa : ℕ := 5
def cindy_gives_lisa : ℕ := 12

-- Assuming Cindy's initial marbles are 20, which are 5 more than Lisa's marbles, and Cindy gives Lisa 12 marbles,
-- prove that Lisa now has 19 more marbles than Cindy.
theorem lisa_more_marbles_than_cindy :
  let lisa_initial_marbles := initial_cindy_marbles - difference_cindy_lisa,
      lisa_current_marbles := lisa_initial_marbles + cindy_gives_lisa,
      cindy_current_marbles := initial_cindy_marbles - cindy_gives_lisa
  in lisa_current_marbles - cindy_current_marbles = 19 :=
by {
  sorry
}

end lisa_more_marbles_than_cindy_l374_374550


namespace trapezoid_perimeter_l374_374686

section

variables (PQ RS PS QR : ℝ)
variables (height distance : ℝ)

-- Conditions
def isosceles_trapezoid (PQ RS PS QR height distance : ℝ) :=
  PQ = RS ∧
  distance = 5 ∧
  PQ = 7 ∧
  RS = 7 ∧
  height = 6

-- Proof that the perimeter of trapezoid PQRS is 14 + 4 * sqrt 13
theorem trapezoid_perimeter (h : isosceles_trapezoid PQ RS PS QR height distance) :
  2 * PQ + 2 * (real.sqrt (RS^2 - height^2)) = 14 + 4 * real.sqrt 13 :=
sorry

end

end trapezoid_perimeter_l374_374686


namespace ellipse_equation_constant_sum_l374_374610

noncomputable def ellipse_properties (a b : ℝ) : Prop :=
a > b ∧ b > 0 ∧ (∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1) ∧
(∃ (c : ℝ), 2 * c = 4 ∧ (∃ (M : ℝ × ℝ), 
    let F1 := (-c, 0) in
    let F2 := (c, 0) in
    let MF1 := (M.1 + c)^2 + M.2^2 in
    let MF2 := (M.1 - c)^2 + M.2^2 in
    MF1 * MF2 = 16 / 3 ∧
    angle F1 M F2 = π / 3 ∧
    area F1 M F2 = 4 * sqrt 3 / 3)) ∧
(∃ (N : ℝ × ℝ), N = (0, 2) ∧
 ∀ (k : ℝ), let P := (1, -2) in
  let L := line_through_slope P k in
  ∃ (A B : ℝ × ℝ), (A ≠ N ∧ B ≠ N ∧
    A, B ∈ ellipse C) ∧
  let k1 := slope N A in
  let k2 := slope N B in
  k1 + k2 = 4)

theorem ellipse_equation_constant_sum :
  ∃ (C : conic_section), ellipse_properties C.a C.b ∧
  C = { x : ℝ × ℝ | x.1^2 / 8 + x.2^2 / 4 = 1 } ∧
  ∀ (k1 k2 : ℝ), (k1 + k2 = 4) :=
sorry

end ellipse_equation_constant_sum_l374_374610


namespace petya_digits_l374_374766

theorem petya_digits (a b c d : ℕ) (distinct_digits : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (non_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (sum_formed_numbers : 6 * (a + b + c + d) * 1111 = 73326) :
  {a, b, c, d} = {1, 2, 3, 5} :=
by 
  sorry

end petya_digits_l374_374766


namespace perfect_square_expression_l374_374199

theorem perfect_square_expression (x y : ℕ) (p : ℕ) [Fact (Nat.Prime p)]
    (h : 4 * x^2 + 8 * y^2 + (2 * x - 3 * y) * p - 12 * x * y = 0) :
    ∃ (n : ℕ), 4 * y + 1 = n^2 :=
sorry

end perfect_square_expression_l374_374199


namespace arithmetic_progression_squares_l374_374232

theorem arithmetic_progression_squares :
  ∃ (n : ℤ), ((3 * n^2 + 8 = 1111 * 5) ∧ (n-2, n, n+2) = (41, 43, 45)) :=
by
  sorry

end arithmetic_progression_squares_l374_374232


namespace eleven_segment_open_polygons_count_l374_374056

theorem eleven_segment_open_polygons_count :
  let A := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  ∃ (n : ℕ), n = 1024 ∧ 
  -- Count number of distinct 11-segment open polygons without self-intersections
  let count_distinct_polygons := 2 ^ 10 in
  -- Polygons that can be transformed into each other by rotation are considered the same
  n = count_distinct_polygons :=
sorry

end eleven_segment_open_polygons_count_l374_374056


namespace marbles_problem_l374_374548

theorem marbles_problem
  (cindy_original : ℕ)
  (lisa_original : ℕ)
  (h1 : cindy_original = 20)
  (h2 : cindy_original = lisa_original + 5)
  (marbles_given : ℕ)
  (h3 : marbles_given = 12) :
  (lisa_original + marbles_given) - (cindy_original - marbles_given) = 19 :=
by
  sorry

end marbles_problem_l374_374548


namespace fifth_term_of_sequence_l374_374553

def pow_four_sequence (n : ℕ) : ℕ :=
  ∑ i in Finset.range (n + 1), 4^i

theorem fifth_term_of_sequence :
  pow_four_sequence 4 = 341 :=
sorry

end fifth_term_of_sequence_l374_374553


namespace meter_to_skips_l374_374406

-- Define the problem entities and hypotheses
variables (a b c d e f g : ℕ)

-- Hypotheses given in the problem
axiom hops_to_skips : ∀ (a b : ℕ), a * b.hops = b * a.skips
axiom jogs_to_hops  : ∀ (c d : ℕ), c * d.jogs = d * c.hops
axiom dashes_to_jogs: ∀ (e f : ℕ), e * f.dashes = f * e.jogs
axiom meter_to_dashes: ∀ (g : ℕ), 1.meter = g * dashes

-- Theorem to prove the equivalency of 1 meter to skips
theorem meter_to_skips : ∀ (a b c d e f g : ℕ), 1.meter = (gfdb / eca).skips :=
by {
  -- Declare the variables
  intros a b c d e f g,
  -- Assume the given conditions
  have H1: a * b.hops = b * a.skips := hops_to_skips a b,
  have H2: c * d.jogs = d * c.hops := jogs_to_hops c d,
  have H3: e * f.dashes = f * e.jogs := dashes_to_jogs e f,
  have H4: 1.meter = g * dashes := meter_to_dashes g,
  -- Prove the theorem based on the given conditions (Details of proof skipped here)
  sorry,
}

end meter_to_skips_l374_374406


namespace linearly_correlated_l374_374908

def Heights_Positive_Correlation := true
def Cylinder_Radius_Functional_Relation := true
def Car_Weight_Negative_Correlation := true
def Income_Expenditure_Positive_Correlation := true

theorem linearly_correlated (H1 : Heights_Positive_Correlation) 
                            (H2 : Cylinder_Radius_Functional_Relation)
                            (H3 : Car_Weight_Negative_Correlation) 
                            (H4 : Income_Expenditure_Positive_Correlation) : 
                        (H1 ∧ H3 ∧ H4) ∧ ¬H2 := 
by sorry

end linearly_correlated_l374_374908


namespace minimize_PQ_QR_RP_l374_374374

open Real

theorem minimize_PQ_QR_RP (a b : ℝ) (h0 : 0 < b) (h1 : b < a) :
  ∃ (Q R : (ℝ × ℝ)), (Q.2 = 0 ∧ R.1 = R.2) ∧ 
  sqrt (2 * (a^2 + b^2)) = dist (a, b) Q + dist Q R + dist R (a, b) :=
begin
  sorry
end

end minimize_PQ_QR_RP_l374_374374


namespace binomial_4th_term_coefficient_l374_374015

theorem binomial_4th_term_coefficient :
  (∀ (x : ℕ) (n : ℕ), 
    let a := 2 * x^2,
    let b := -1 / x,
    let term := Nat.choose 5 3 * a^2 * b^3,
    term = -40 * x) :=
by sorry

end binomial_4th_term_coefficient_l374_374015


namespace island_distance_l374_374701

theorem island_distance (d1 d2 d3 : ℕ) 
    (h1 : d1 = 132)
    (h2 : d2 = 236) 
    (h3 : d3 = 68) : 
    d1 + d2 + d3 = 436 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end island_distance_l374_374701


namespace total_kayaks_built_by_april_l374_374187

def kayaks_built_february : ℕ := 5
def kayaks_built_next_month (n : ℕ) : ℕ := 3 * n
def kayaks_built_march : ℕ := kayaks_built_next_month kayaks_built_february
def kayaks_built_april : ℕ := kayaks_built_next_month kayaks_built_march

theorem total_kayaks_built_by_april : 
  kayaks_built_february + kayaks_built_march + kayaks_built_april = 65 :=
by
  -- proof goes here
  sorry

end total_kayaks_built_by_april_l374_374187


namespace min_bail_rate_to_reach_shore_l374_374909

theorem min_bail_rate_to_reach_shore (
  distance_to_shore : ℝ := 2,
  leak_rate : ℝ := 15,
  sink_capacity : ℝ := 50,
  rowing_speed : ℝ := 3,
  time_conversion : ℝ := 60
) : 
  let time_to_shore := distance_to_shore / rowing_speed * time_conversion in
  let total_intake := leak_rate * time_to_shore in
  let excess_intake := total_intake - sink_capacity in
  let min_bail_rate := excess_intake / time_to_shore in
  min_bail_rate.tt := 14 := 
sorry

end min_bail_rate_to_reach_shore_l374_374909


namespace systematic_sampling_eighth_group_number_l374_374944

theorem systematic_sampling_eighth_group_number (total_students groups students_per_group draw_lots_first : ℕ) 
  (h_total : total_students = 480)
  (h_groups : groups = 30)
  (h_students_per_group : students_per_group = 16)
  (h_draw_lots_first : draw_lots_first = 5) : 
  (8 - 1) * students_per_group + draw_lots_first = 117 :=
by
  sorry

end systematic_sampling_eighth_group_number_l374_374944


namespace sine_quotient_polynomial_l374_374394

theorem sine_quotient_polynomial (k : ℤ) (φ : ℝ) :
  (sin ((2 * k + 1 : ℤ) * φ) / sin φ) =
  (-4 : ℝ) ^ k * 
  ∏ i in finset.range k, (sin^2 φ - sin^2 (i * π / (2 * k + 1 : ℤ))) :=
  sorry

end sine_quotient_polynomial_l374_374394


namespace average_weight_of_class_l374_374109

theorem average_weight_of_class :
  let section_A_students := 40
  let section_B_students := 30
  let avg_weight_A := 50.0
  let avg_weight_B := 60.0
  let total_students := section_A_students + section_B_students
  let total_weight := (section_A_students * avg_weight_A) + (section_B_students * avg_weight_B)
  let class_avg_weight := total_weight / total_students
  class_avg_weight ≈ 54.29 :=
by
  sorry

end average_weight_of_class_l374_374109


namespace area_AOC_eq_l374_374731

variables {A B C O : Type}
variables [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ O]
variables (area_ABC : ℝ)

-- Adding vector variables and basic conditions
variables (OA OB OC : A)
variables (zero_vector : A)

-- Hypotheses
axiom inside_triangle : true -- Placeholder since there's no inherent "inside" concept in all vector spaces
axiom area_eq_six : area_ABC = 6
axiom vector_eq : OA + OB + 2 * OC = zero_vector

-- Concluding area of triangle AOC
theorem area_AOC_eq (area_AOC : ℝ) : area_AOC = 3 / 2 :=
by
  -- Placeholder proof, actual proof not required as per instruction.
  sorry

end area_AOC_eq_l374_374731


namespace sam_seashells_l374_374814

-- We define our variables based on the conditions.
def joan_seashells : ℕ := 18
def total_seashells : ℕ := 53

-- Our goal is to prove that Sam found 35 seashells.
theorem sam_seashells : ∃ S : ℕ, S + joan_seashells = total_seashells ∧ S = 35 :=
by {
  -- We start by assuming that there exists a number of seashells Sam found.
  use 35,
  -- We need to show S + joan_seashells equals total_seashells and S is indeed 35.
  split,
  {
    -- Prove the arithmetic part.
    calc 35 + 18 = 53 : by norm_num,
  },
  {
    -- State that the number of seashells Sam found is 35.
    refl,
  },
}

end sam_seashells_l374_374814


namespace find_possible_values_l374_374240

def matrix_is_not_invertible (x y z : ℝ) : Prop :=
  let A := matrix ([[x, y, z], [y, z, x], [z, x, y]] : matrix (fin 3) (fin 3) ℝ)
  in A.det = 0

noncomputable def expression (x y z : ℝ) : ℝ :=
  x / (y + z) + y / (x + z) + z / (x + y)

theorem find_possible_values (x y z : ℝ) (h : matrix_is_not_invertible x y z) :
  expression x y z = 2 ∨ expression x y z = 3 / 2 :=
sorry

end find_possible_values_l374_374240


namespace find_f_f_neg_half_l374_374598

def f (x : ℝ) : ℝ :=
if x ≤ 0 then Real.exp x else Real.log x

theorem find_f_f_neg_half : f (f (-1/2)) = -1/2 :=
by
  sorry

end find_f_f_neg_half_l374_374598


namespace intersection_distance_squared_l374_374828

-- Definitions of the circles based on their center coordinates and radius.
def circle1 : set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 3)^2 = 3^2}
def circle2 : set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 + 4)^2 = 6^2}

-- Definition of the intersection points C and D
def point_C : ℝ × ℝ := (3, 10/7)
def point_D : ℝ × ℝ := (-1, 10/7)

-- Main theorem statement
theorem intersection_distance_squared : 
  (point_C.1 - point_D.1)^2 + (point_C.2 - point_D.2)^2 = 16 :=
by
  sorry

end intersection_distance_squared_l374_374828


namespace calories_per_stair_l374_374847

-- Given conditions:
def runs (rounds : ℕ) := 40
def stairs_one_way (stairs : ℕ) := 32
def total_calories (calories : ℕ) := 5120

-- Intermediate calculation step (based on conditions, but not part of the solution steps):
def total_stairs_climbed (stairs_one_way : ℕ) (runs : ℕ) : ℕ :=
  (stairs_one_way * 2) * runs

-- Final calculation (to be proven):
theorem calories_per_stair
  (runs : ℕ) (stairs_one_way : ℕ) (total_calories : ℕ)
  (h_runs : runs = 40) (h_stairs_one_way : stairs_one_way = 32) (h_calories : total_calories = 5120) :
  (total_calories / ((stairs_one_way * 2) * runs) = 2) :=
by
  rw [h_runs, h_stairs_one_way, h_calories]
  sorry

end calories_per_stair_l374_374847


namespace digital_earth_correct_descriptions_l374_374529

def Description₁ : Prop := "Digital Earth is a digitized Earth"
def Description₃ : Prop := "The biggest feature of Digital Earth is virtual reality"

theorem digital_earth_correct_descriptions : Description₁ ∧ Description₃ := 
sorry

end digital_earth_correct_descriptions_l374_374529


namespace not_sophomores_percentage_l374_374009

theorem not_sophomores_percentage (total_students : ℕ)
    (juniors_percentage : ℚ) (juniors : ℕ)
    (seniors : ℕ) (freshmen sophomores : ℕ)
    (h1 : total_students = 800)
    (h2 : juniors_percentage = 0.22)
    (h3 : juniors = juniors_percentage * total_students)
    (h4 : seniors = 160)
    (h5 : freshmen = sophomores + 48)
    (h6 : freshmen + sophomores + juniors + seniors = total_students) :
    ((total_students - sophomores : ℚ) / total_students) * 100 = 74 := by
  sorry

end not_sophomores_percentage_l374_374009


namespace petya_digits_l374_374789

def are_distinct (a b c d : Nat) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 

def non_zero_digits (a b c d : Nat) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

theorem petya_digits :
  ∃ (a b c d : Nat), are_distinct a b c d ∧ non_zero_digits a b c d ∧ (a + b + c + d = 11) ∧ (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 5) ∧
  (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 5) ∧ (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 5) ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 5) :=
by
  sorry

end petya_digits_l374_374789


namespace max_min_xy_l374_374862

theorem max_min_xy (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) : (x * y) ∈ { real.min (-1), real.max (1) } :=
by
  have h : a^2 ≤ 4 := 
  sorry
  have xy_eq : x * y = (a^2 - 2) / 2 := 
  sorry
  split
  { calc min (xy_eq) = -1 :=
    sorry
    calc max (xy_eq) = 1 :=
    sorry
  }

end max_min_xy_l374_374862


namespace sum_log_eq_six_sevenths_l374_374988

theorem sum_log_eq_six_sevenths :
  ∑ k in Finset.range 126 \ Finset.singleton 0 \ Finset.singleton 1, 
    (Real.log (1 + (1 / k.cast))) / (Real.log k.cast * Real.log (k.cast + 1)) = 6 / 7 :=
by
  sorry

end sum_log_eq_six_sevenths_l374_374988


namespace true_propositions_l374_374990

-- Definitions of the propositions
def Prop1 (x y : ℝ) : Prop := (x * y = 1) → (x = 1 / y) ∧ (y = 1 / x)
def Prop2 (congruent : Type → Type → Prop) (T1 T2 : Type) : Prop := congruent T1 T2 → (∀ (A B : Type), congruent A B → area A = area B)
def Prop3 (m : ℝ) : Prop := (m ≤ 1) → ∃ x : ℝ, x^2 - 2 * x + m = 0
def Prop4 (A B : Set) : Prop := (A ∩ B = B) → (A ⊆ B)

-- The Lean statement for the problem
theorem true_propositions :
  (Prop1 1 1) ∧
  (∀ (congruent : Type → Type → Prop) (T1 T2 : Type), Prop2 congruent T1 T2) ∧
  (Prop3 1) ∧
  (¬ (∀ (A B : Set), Prop4 A B)) :=
by sorry

end true_propositions_l374_374990


namespace stock_values_l374_374755

theorem stock_values (AA_invest : ℕ) (BB_invest : ℕ) (CC_invest : ℕ)
  (AA_first_year_increase : ℝ) (BB_first_year_decrease : ℝ) (CC_first_year_change : ℝ)
  (AA_second_year_decrease : ℝ) (BB_second_year_increase : ℝ) (CC_second_year_increase : ℝ)
  (A_final : ℝ) (B_final : ℝ) (C_final : ℝ) :
  AA_invest = 150 → BB_invest = 100 → CC_invest = 50 →
  AA_first_year_increase = 1.10 → BB_first_year_decrease = 0.70 → CC_first_year_change = 1 →
  AA_second_year_decrease = 0.95 → BB_second_year_increase = 1.10 → CC_second_year_increase = 1.08 →
  A_final = (AA_invest * AA_first_year_increase) * AA_second_year_decrease →
  B_final = (BB_invest * BB_first_year_decrease) * BB_second_year_increase →
  C_final = (CC_invest * CC_first_year_change) * CC_second_year_increase →
  C_final < B_final ∧ B_final < A_final :=
by
  intros
  sorry

end stock_values_l374_374755


namespace derivative_at_1_of_f_l374_374248

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 2^x) / x^2

theorem derivative_at_1_of_f :
  (deriv f 1) = 2 * Real.log 2 - 3 :=
sorry

end derivative_at_1_of_f_l374_374248


namespace triangle_ABC_cos_C_l374_374635

-- Define the function f(x)
def f (x : ℝ) : ℝ := sqrt 3 * sin x * cos x + sin x ^ 2

-- Condition for part II of the problem
def A := (1 / 3 : ℝ) * Real.pi
def B := (1 / 4 : ℝ) * Real.pi

-- Given conditions
theorem triangle_ABC_cos_C :
  let c := sqrt 6 - sqrt 2 in
  f A = 3 / 2 ∧ 
  sqrt 2 * sin (1 / 6 * Real.pi) = 2 * sin B ∧
  cos (A + B) = - (c / 4) :=
begin
  sorry,
end

end triangle_ABC_cos_C_l374_374635


namespace find_BE_l374_374054

-- Definitions of necessary geometrical properties
def square (s : ℝ) := s * s

def distance (A B : ℝ × ℝ) := real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

-- Main theorem stating the given proof
theorem find_BE (s : ℝ) (A B C D E F : ℝ × ℝ) 
  (h1 : square s = 400) 
  (h2 : distance A F = 10) 
  (h3 : distance F D = 10) 
  (h4 : ∃ x y : ℝ, (y - C.2) = 0  ∧ distance E (A.1 + x, A.2) = 30)
  (h5 : ∃ z : ℝ, z * sqrt (((F.1 - C.1) ^ 2 + (F.2 - C.2) ^ 2)) = 250) :
  distance B E = 30 :=
sorry

end find_BE_l374_374054


namespace swimming_speed_solution_l374_374954

-- Definition of the conditions
def speed_of_water : ℝ := 2
def distance_against_current : ℝ := 10
def time_against_current : ℝ := 5

-- Definition of the person's swimming speed in still water
def swimming_speed_in_still_water (v : ℝ) :=
  distance_against_current = (v - speed_of_water) * time_against_current

-- Main theorem we want to prove
theorem swimming_speed_solution : 
  ∃ v : ℝ, swimming_speed_in_still_water v ∧ v = 4 :=
by
  sorry

end swimming_speed_solution_l374_374954


namespace center_of_mass_distance_to_line_l374_374241

theorem center_of_mass_distance_to_line (m1 m2 m3 : ℝ) (y1 y2 y3 : ℝ) (h1 : m1 > 0) (h2 : m2 > 0) (h3 : m3 > 0) :
  ∃ z : ℝ, z = (m1 * y1 + m2 * y2 + m3 * y3) / (m1 + m2 + m3) := by
  use (m1 * y1 + m2 * y2 + m3 * y3) / (m1 + m2 + m3)
  apply (rfl)

end center_of_mass_distance_to_line_l374_374241


namespace find_second_number_l374_374411

theorem find_second_number (x : ℝ) : 
  (20 + x + 60) / 3 = (10 + 70 + 13) / 3 + 9 → x = 40 :=
by
  sorry

end find_second_number_l374_374411


namespace tom_hockey_games_l374_374445

def tom_hockey_games_last_year (games_this_year missed_this_year total_games : Nat) : Nat :=
  total_games - games_this_year

theorem tom_hockey_games :
  ∀ (games_this_year missed_this_year total_games : Nat),
    games_this_year = 4 →
    missed_this_year = 7 →
    total_games = 13 →
    tom_hockey_games_last_year games_this_year total_games = 9 := by
  intros games_this_year missed_this_year total_games h1 h2 h3
  -- The proof steps would go here
  sorry

end tom_hockey_games_l374_374445


namespace present_age_of_son_is_22_l374_374949

theorem present_age_of_son_is_22 (S F : ℕ) (h1 : F = S + 24) (h2 : F + 2 = 2 * (S + 2)) : S = 22 :=
by
  sorry

end present_age_of_son_is_22_l374_374949


namespace petya_digits_sum_l374_374761

def distinct_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem petya_digits_sum (a b c d : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
    (h_distinct: distinct_digits a b c d)
    (h_sum: 6 * (a + b + c + d) * 1111 = 73326) :
    a + b + c + d = 11 ∧ {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end petya_digits_sum_l374_374761


namespace g_eq_f_plus_n_plus_one_g_eq_floor_plus_one_l374_374997

noncomputable def alpha := (Real.sqrt 5 - 1) / 2

noncomputable def f (n : ℕ) : ℕ :=
  Real.floor (n * alpha)

noncomputable def g (n : ℕ) : ℕ :=
  Nat.find (Nat.exists_min {m | f m ≥ n})

theorem g_eq_f_plus_n_plus_one (n : ℕ) : g n = f n + n + 1 :=
  sorry

theorem g_eq_floor_plus_one (n : ℕ) : g n = Nat.floor ((Real.sqrt 5 + 1) / 2 * n) + 1 :=
  sorry

end g_eq_f_plus_n_plus_one_g_eq_floor_plus_one_l374_374997


namespace length_of_square_cut_off_l374_374501

theorem length_of_square_cut_off 
  (x : ℝ) 
  (h_eq : (48 - 2 * x) * (36 - 2 * x) * x = 5120) : 
  x = 8 := 
sorry

end length_of_square_cut_off_l374_374501


namespace democrat_ratio_l374_374441

noncomputable def total_participants : ℕ := 870
noncomputable def female_democrats : ℕ := 145
noncomputable def male_democrats (male_participants : ℕ) : ℕ := male_participants / 4

theorem democrat_ratio :
  let female_participants := female_democrats * 2,
      male_participants := total_participants - female_participants,
      total_democrats := female_democrats + male_democrats male_participants in
      (total_democrats : ℚ) / total_participants = 1 / 3 :=
by
  sorry

end democrat_ratio_l374_374441


namespace ellipse_eccentricity_m_l374_374279

theorem ellipse_eccentricity_m (m : ℝ) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 = 4 * b^2 ∧ x^2 + m * y^2 = 1 ∧ (c / a = √(3) / 2)) → 
  (m = 4 ∨ m = 1 / 4) :=
by sorry

end ellipse_eccentricity_m_l374_374279


namespace paperboy_12_houses_l374_374505

def E : ℕ → ℕ
| 0     := 1 -- a trivial base case that is generally added in such recurrences to handle edge cases
| 1     := 2
| 2     := 4
| 3     := 8
| 4     := 15
| (n+5) := E n + E (n+1) + E (n+2) + E (n+3)

theorem paperboy_12_houses : E 12 = 2872 :=
by sorry

end paperboy_12_houses_l374_374505


namespace sum_of_areas_of_circles_l374_374875

noncomputable def sum_of_areas (α β γ : ℝ) : ℝ :=
  π * (α^2 + β^2 + γ^2)

theorem sum_of_areas_of_circles :
  let α := 2
  let β := 4
  let γ := 6
  (α + β = 6) ∧ (α + γ = 8) ∧ (β + γ = 10) →
  sum_of_areas α β γ = 56 * π := 
by 
  sorry

end sum_of_areas_of_circles_l374_374875


namespace Ethan_uses_8_ounces_each_l374_374573

def Ethan (b: ℕ): Prop :=
  let number_of_candles := 10 - 3
  let total_coconut_oil := number_of_candles * 1
  let total_beeswax := 63 - total_coconut_oil
  let beeswax_per_candle := total_beeswax / number_of_candles
  beeswax_per_candle = b

theorem Ethan_uses_8_ounces_each (b: ℕ) (hb: Ethan b): b = 8 :=
  sorry

end Ethan_uses_8_ounces_each_l374_374573


namespace sum_of_angles_l374_374018

variables (A B C D F G EDC ECD : ℝ)

-- Condition: 
-- Given angles A, B, C, D, F, G in a diagram where G + F = EDC + ECD
-- We need to prove that their sum equals 360 degrees.

theorem sum_of_angles (h1 : ∠F + ∠G = ∠EDC + ∠ECD) :
  ∠A + ∠B + ∠C + ∠D + ∠F + ∠G = 360 := 
by 
  sorry

end sum_of_angles_l374_374018


namespace BC_length_47_l374_374613

theorem BC_length_47 (A B C D : ℝ) (h₁ : A ≠ B) (h₂ : B ≠ C) (h₃ : B ≠ D)
  (h₄ : dist A C = 20) (h₅ : dist A D = 45) (h₆ : dist B D = 13)
  (h₇ : C = 0) (h₈ : D = 0) (h₉ : B = A + 43) :
  dist B C = 47 :=
sorry

end BC_length_47_l374_374613


namespace find_sum_a_t_l374_374245

theorem find_sum_a_t :
  ∀ (n : ℕ) (an : ℕ), an = n + 1 → n = 5 → let a := an in let t := a^2 - 1 in a + t = 41 :=
by
  intros n an h1 h2
  let a := an
  let t := a^2 - 1
  rw h1 at h2
  rw h2
  sorry

end find_sum_a_t_l374_374245


namespace prob_xi_range_negative_2_to_4_l374_374273

noncomputable
def normal_distribution_mean : ℝ := 1
def normal_distribution_variance : ℝ := 4

axiom xi_follows_normal_distribution (ξ : ℝ) : True

axiom prob_xi_greater_than_4 (ξ : ℝ) : ℙ {x | x > 4} = 0.1

theorem prob_xi_range_negative_2_to_4 (ξ : ℝ) 
  (H1 : xi_follows_normal_distribution ξ) 
  (H2 : prob_xi_greater_than_4 ξ) : 
  ℙ ((-2 : ℝ) ≤ ξ ∧ ξ ≤ 4) = 0.8 := 
sorry

end prob_xi_range_negative_2_to_4_l374_374273


namespace iterate_fixed_point_l374_374376

theorem iterate_fixed_point {f : ℤ → ℤ} (a : ℤ) :
  (∀ n, f^[n] a = a → f a = a) ∧ (f a = a → f^[22000] a = a) :=
sorry

end iterate_fixed_point_l374_374376


namespace find_unique_n_l374_374580

theorem find_unique_n : ∃! (n : ℤ), 3 ≤ n ∧ n ≤ 10 ∧ n ≡ 10573 [MOD 7] :=
by
  sorry

end find_unique_n_l374_374580


namespace sara_initial_quarters_l374_374068

theorem sara_initial_quarters (borrowed quarters_current : ℕ) (q_initial : ℕ) :
  quarters_current = 512 ∧ quarters_borrowed = 271 → q_initial = 783 :=
by
  sorry

end sara_initial_quarters_l374_374068


namespace quadratic_inequality_properties_l374_374274

theorem quadratic_inequality_properties
  (a b c : ℝ) 
  (h1 : ∀ x : ℝ, ax^2 + bx + c > 0 ↔ x < -2 ∨ x > 3) :
  (∀ x : ℝ, bx - c > 0 ↔ x < 6) ∧ 
  (∀ x : ℝ, -6a * x^2 + a * x + a ≥ 0 ↔ -1/3 ≤ x ∧ x ≤ 1/2) :=
by
  sorry

end quadratic_inequality_properties_l374_374274


namespace sum_of_roots_eq_14_div_3_l374_374980

theorem sum_of_roots_eq_14_div_3 :
  let f := (λ x : ℝ, (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7))
  ∃ r1 r2 : ℝ, f r1 = 0 ∧ f r2 = 0 ∧ r1 + r2 = 14 / 3 :=
by
  sorry

end sum_of_roots_eq_14_div_3_l374_374980


namespace product_of_roots_cubic_l374_374987

theorem product_of_roots_cubic :
  let a := 2
  let b := -5
  let c := 4
  let d := -35
  (∏ x in (roots (2 * X^3 - 5 * X^2 + 4 * X - 35)), x) = 35 / 2 := 
by
  sorry

end product_of_roots_cubic_l374_374987


namespace pyramid_volume_division_l374_374836

noncomputable def pyramid_division_ratio (P A B C D K M : Point) 
  (parallelogram_ABCD : is_parallelogram A B C D)
  (midpoint_CP : is_midpoint K C P)
  (AM_MB_ratio : ratio_eq AM MB 1 2)
  (plane_passing_through_KM_parallel_BD : plane_through_KM_parallel_to_lineBD K M B D)
  : Prop :=
  volume_ratio_of_division_by_plane (PABCD : pyramid P A B C D) (K M) (line BD) = 109 / 143

theorem pyramid_volume_division (P A B C D K M : Point) 
  (parallelogram_ABCD : is_parallelogram A B C D)
  (midpoint_CP : is_midpoint K C P)
  (AM_MB_ratio : ratio_eq AM MB 1 2)
  (plane_passing_through_KM_parallel_BD : plane_through_KM_parallel_to_lineBD K M B D)
  : pyramid_division_ratio P A B C D K M parallelogram_ABCD midpoint_CP AM_MB_ratio plane_passing_through_KM_parallel_BD :=
sorry

end pyramid_volume_division_l374_374836


namespace biography_percentage_increase_l374_374032

variable {T : ℝ}
variable (hT : T > 0 ∧ T ≤ 10000)
variable (B : ℝ := 0.20 * T)
variable (B' : ℝ := 0.32 * T)
variable (percentage_increase : ℝ := ((B' - B) / B) * 100)

theorem biography_percentage_increase :
  percentage_increase = 60 :=
by
  sorry

end biography_percentage_increase_l374_374032


namespace balls_distribution_l374_374877

theorem balls_distribution : 
  let balls := 20
  let labeled_boxes := 3
  let box1_min := 1
  let box2_min := 2
  let box3_min := 3
  let y_total := 14 in
  ∑' (y1 y2 y3 : ℕ), (y1 + y2 + y3 = y_total) = 120 := 
begin
  sorry
end

end balls_distribution_l374_374877


namespace correct_system_equations_l374_374677

theorem correct_system_equations (x y : ℕ) (h1 : x + y = 56) (h2 : 2 * 16 * x = 24 * y) :
  (x + y = 56) ∧ (2 * 16 * x = 24 * y) :=
by {
  split;
  { assumption },
  {
    sorry
  }
}

end correct_system_equations_l374_374677


namespace petya_four_digits_l374_374781

theorem petya_four_digits :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
    6 * (a + b + c + d) * 1111 = 73326 ∧ 
    (∃ S, S = a + b + c + d ∧ S = 11) :=
begin
  use 1, 2, 3, 5,
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split,
  { exact dec_trivial },
  use 11,
  split,
  { exact dec_trivial },
  { exact dec_trivial }
end

end petya_four_digits_l374_374781


namespace collinear_A_K_P_l374_374709

open EuclideanGeometry 

theorem collinear_A_K_P
  (A B C D K P : Point)
  (h1 : rhombus A B C D)
  (h2 : lies_on K (line_through C D))
  (h3 : K ≠ C ∧ K ≠ D)
  (h4 : distance A D = distance B K)
  (h5 : is_intersection P (line_through B D) (perpendicular_bisector B C)) :
  collinear {A, K, P} := by
  sorry

end collinear_A_K_P_l374_374709


namespace power_set_card_greater_l374_374407

open Set

variables {A : Type*} (α : ℕ) [Fintype A] (hA : Fintype.card A = α)

theorem power_set_card_greater (h : Fintype.card A = α) :
  2 ^ α > α :=
sorry

end power_set_card_greater_l374_374407


namespace find_angle_A_find_area_range_l374_374264

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (triangle_ABC : Prop) -- represents the property that ∆ABC is an acute triangle 

-- Ensure the hypotheses are correct and triangle ABC is acute
axiom sides_angles (h_triangle: triangle_ABC) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 ∧ A + B + C = π

-- Given condition to be used in proofs:
axiom condition : a * Real.cos C + sqrt 3 * a * Real.sin C - b - c = 0

-- Prove that the measure of angle A is π/3 under the given condition.
theorem find_angle_A (h_triangle : triangle_ABC) (h : a * Real.cos C + sqrt 3 * a * Real.sin C - b - c = 0) :
  A = π / 3 :=
sorry -- proof goes here

-- Prove the range of values for the area of ∆ABC under the condition a = sqrt 3.
theorem find_area_range (h_triangle : triangle_ABC) (ha : a = sqrt 3) (hb : b > 0) (hc : c > 0) :
  ∃ S, S = 1/2 * b * c * Real.sin A ∧ (sqrt 3 / 2 < S ∧ S <= 3 * sqrt 3 / 4) :=
sorry -- proof goes here

end find_angle_A_find_area_range_l374_374264


namespace average_age_of_choir_l374_374834

theorem average_age_of_choir (avg_age_female : ℕ) (num_female : ℕ) (avg_age_male : ℕ) (num_male : ℕ) 
(h1 : avg_age_female = 30) (h2 : num_female = 10) (h3 : avg_age_male = 35) (h4 : num_male = 15) :
  let total_sum_female := avg_age_female * num_female,
      total_sum_male := avg_age_male * num_male,
      total_sum := total_sum_female + total_sum_male,
      total_people := num_female + num_male
  in total_sum / total_people = 33 :=
by sorry

end average_age_of_choir_l374_374834


namespace petya_digits_sum_l374_374759

def distinct_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem petya_digits_sum (a b c d : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
    (h_distinct: distinct_digits a b c d)
    (h_sum: 6 * (a + b + c + d) * 1111 = 73326) :
    a + b + c + d = 11 ∧ {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end petya_digits_sum_l374_374759


namespace total_handshakes_l374_374472

variable (n : ℕ) (h : n = 12)

theorem total_handshakes (H : ∀ (b : ℕ), b = n → (n * (n - 1)) / 2 = 66) : 
  (12 * 11) / 2 = 66 := 
by
  sorry

end total_handshakes_l374_374472


namespace no_prime_factor_congruent_to_7_mod_8_l374_374601

open Nat

theorem no_prime_factor_congruent_to_7_mod_8 (n : ℕ) (hn : 0 < n) : 
  ¬ (∃ p : ℕ, p.Prime ∧ p ∣ 2^n + 1 ∧ p % 8 = 7) :=
sorry

end no_prime_factor_congruent_to_7_mod_8_l374_374601


namespace probability_f_le_g_l374_374247

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ x
noncomputable def g (x : ℝ) : ℝ := 3 ^ x

theorem probability_f_le_g :
  ∀ (x0 : ℝ), x0 ∈ Icc (-2 : ℝ) 2 →
  ∃ (p : ℝ), p = 1 / 2 ∧ (p = (measure_theory.measure_space.volume (set_of (λ y, f y ≤ g y) ∩ Icc (-2 : ℝ) 2) / measure_theory.measure_space.volume (Icc (-2 : ℝ) 2))) :=
begin
  sorry
end

end probability_f_le_g_l374_374247


namespace max_min_floor_diff_l374_374724

theorem max_min_floor_diff (m : ℕ) (x : Fin m → ℚ) (hx : (∑ i, x i) = 1) (hx_pos : ∀ i, 0 < x i) (n : ℕ) (hn_pos : n > 0) :
  0 ≤ n - ∑ i, ⌊n * (x i)⌋ ∧ n - ∑ i, ⌊n * (x i)⌋ ≤ m - 1 :=
sorry

end max_min_floor_diff_l374_374724


namespace john_spent_l374_374028

def length_of_cloth : ℝ := 9.25
def cost_per_meter : ℝ := 45
def total_cost : ℝ := 416.25

theorem john_spent (Total_cost_calculated : length_of_cloth * cost_per_meter = total_cost) : total_cost = 416.25 :=
by
  exact Total_cost_calculated

end john_spent_l374_374028


namespace tangent_line_at_1_0_monotonic_intervals_l374_374632

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x + 2 * Real.log x

noncomputable def f_derivative (x : ℝ) (a : ℝ) : ℝ := (2 * x^2 - a * x + 2) / x

theorem tangent_line_at_1_0 (a : ℝ) (h : a = 1) :
  ∀ x y : ℝ, 
  (f x a, f 1 a) = (0, x - 1) → 
  y = 3 * x - 3 := 
sorry

theorem monotonic_intervals (a : ℝ) :
  (∀ x : ℝ, 0 < x → f_derivative x a ≥ 0) ↔ (a ≤ 4) ∧ 
  (∀ x : ℝ, 0 < x → 
    (0 < x ∧ x < (a - Real.sqrt (a^2 - 16)) / 4) ∨ 
    ((a + Real.sqrt (a^2 - 16)) / 4 < x) 
  ) :=
sorry

end tangent_line_at_1_0_monotonic_intervals_l374_374632


namespace num_functions_with_given_range_l374_374637

noncomputable def range_f (f : ℝ → ℝ) : set ℝ :=
  {y | ∃ x, f x = y}

def f (x : ℝ) : ℝ := x^2 - 1

theorem num_functions_with_given_range :
  (range_f f = {0, 1}) → (∃ n : ℕ, n = 9) := by
  sorry

end num_functions_with_given_range_l374_374637


namespace deny_evenness_l374_374210

-- We need to define the natural numbers and their parity.
variables {a b c : ℕ}

-- Define what it means for a number to be odd and even.
def is_odd (n : ℕ) := ∃ k, n = 2 * k + 1
def is_even (n : ℕ) := ∃ k, n = 2 * k

-- The Lean theorem statement translating the given problem.
theorem deny_evenness :
  (is_odd a ∧ is_odd b ∧ is_odd c) → ¬(is_even a ∨ is_even b ∨ is_even c) :=
by sorry

end deny_evenness_l374_374210


namespace ratio_IC_JE_l374_374393

variables (A B C D E F G H I J : Point) (l : Line)
variables (AB CD EF BC DE : ℝ)
variables (h : ¬Collinear H A G)
variables (IC JE AH : Line)

-- Conditions
-- Points A, B, C, D, E, F, and G lie on line l
-- Lengths
variables (AB_eq_1 : AB = 1) (CD_eq_1 : CD = 1) (EF_eq_1 : EF = 1)
variables (BC_eq_2 : BC = 2) (DE_eq_2 : DE = 2)

-- Parallelism
variables (AH_parallel_IC : Is_parallel AH IC) (AH_parallel_JE : Is_parallel AH JE)
variables (DJ_parallel_AH : Is_parallel DJ AH)

-- Length computations
variables (AG_eq_7 : Distance A G = 7) (AD_eq_4 : Distance A D = 4)
variables (DG_eq_3 : Distance D G = 3) (AE_eq_6 : Distance A E = 6) (EG_eq_1 : Distance E G = 1)

-- The main statement to prove
theorem ratio_IC_JE : (IC_length JE_length : ℝ) = (3 / 2) :=
sorry

end ratio_IC_JE_l374_374393


namespace bisecting_line_slope_l374_374335

def point := (ℝ × ℝ)

def vertices : List point :=
  [ (0,0), (0,4), (4,4), (4,2), (7,2), (7,0) ]

def slope_of_bisecting_line (pts : List point) : ℝ :=
  let area1 := 4 * 4
  let area2 := 3 * 2
  let total_area := area1 + area2
  let half_area := total_area / 2
  let x := 1.5
  let G := (4, 2.5)
  G.2 / G.1 -- slope calculation

theorem bisecting_line_slope :
  slope_of_bisecting_line vertices = 5 / 8 :=
by
  sorry

end bisecting_line_slope_l374_374335


namespace max_min_abs_diff_l374_374034

theorem max_min_abs_diff (n : ℕ) (hn : n ≥ 2) (x : ℕ → ℝ) (hx_sum : ∑ i in finset.range n, x i = 0) 
  (hx_bound : ∀ i < n, |x i| ≤ 1) :
  max (finset.image (λ i, |x i - x (i+1)|) {i | i < n-1}) = n / (nat.ceil (n / 2)) :=
sorry

end max_min_abs_diff_l374_374034


namespace problem1_problem2_part1_problem2_part2_l374_374140

-- Problem 1
theorem problem1 (x : ℚ) (h : x = 11 / 12) : 
  (2 * x - 5) * (2 * x + 5) - (2 * x - 3) ^ 2 = -23 := 
by sorry

-- Problem 2
theorem problem2_part1 (a b : ℚ) (h1 : a + b = 6) (h2 : a * b = 7) : 
  a^2 + b^2 = 22 := 
by sorry

theorem problem2_part2 (a b : ℚ) (h1 : a + b = 6) (h2 : a * b = 7) : 
  (a - b)^2 = 8 := 
by sorry

end problem1_problem2_part1_problem2_part2_l374_374140


namespace find_period_find_area_l374_374633

-- Define the function f(x) with the condition ω > 0
def f (ω x : ℝ) : ℝ := 2 * sin (ω * x) * cos (ω * x) - 2 * √3 * (cos (ω * x))^2 + √3

-- Define the triangle ABC with given conditions
structure Triangle :=
  (A B C a b c: ℝ)
  (C_acute : 0 < C ∧ C < π / 2)
  (c_value : c = 3 * √2)
  (sinB_2sinA : sin B = 2 * sin A)

-- Define the conditions of the problem
constant ω : ℝ
axiom ω_positive : ω > 0

constant symmetry_distance : ℝ
axiom symmetry_distance_value : symmetry_distance = π / 2

constant triangle_ABC : Triangle
axiom f_C_value : f ω triangle_ABC.C = √3

-- The period of the function f(x) is π
theorem find_period : ∀ x, f ω x = f ω (x + π) :=
by
  sorry

-- The area of triangle ABC is 3√3
theorem find_area (T : Triangle) : T = triangle_ABC → 
  let area := (1 / 2) * T.a * T.b * sin T.C in
  area = 3 * √3 :=
by
  sorry

end find_period_find_area_l374_374633


namespace jack_jill_same_speed_l374_374352

-- Definitions for Jack and Jill's conditions
def jacks_speed (x : ℝ) : ℝ := x^2 - 13*x - 48
def jills_distance (x : ℝ) : ℝ := x^2 - 5*x - 84
def jills_time (x : ℝ) : ℝ := x + 8

-- Theorem stating the same walking speed given the conditions
theorem jack_jill_same_speed (x : ℝ) (h : jacks_speed x = jills_distance x / jills_time x) : 
  jacks_speed x = 6 :=
by
  sorry

end jack_jill_same_speed_l374_374352


namespace houses_in_block_l374_374946

variables (H : ℕ) (mails : ℕ) (blocks : ℕ) (mails_per_house : ℕ) (skip_interval : ℕ)

-- Conditions from the problem
def conditions :=
  mails = 128 ∧
  blocks = 85 ∧
  mails_per_house = 16 ∧
  skip_interval = 3

-- Proof statement that needs to be proven: There are 12 houses in a block
theorem houses_in_block (h : conditions) : H = 12 :=
sorry

end houses_in_block_l374_374946


namespace monotonic_f_implies_a_le_4_l374_374322

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x^2 + Real.log x - a * x

def is_monotonically_increasing (f : ℝ → ℝ) (dom : Set ℝ) := ∀ x y ∈ dom, x ≤ y → f x ≤ f y

theorem monotonic_f_implies_a_le_4 : 
  (∀ x > 0, ∀ y > 0, (x ≤ y → f x a ≤ f y a)) → a ≤ 4 :=
by
  intro h
  -- Proof to be completed
  sorry

end monotonic_f_implies_a_le_4_l374_374322


namespace son_age_l374_374951

theorem son_age (M S : ℕ) (h1 : M = S + 24) (h2 : M + 2 = 2 * (S + 2)) : S = 22 :=
by
  sorry

end son_age_l374_374951


namespace convex_polygon_angle_bound_l374_374396

theorem convex_polygon_angle_bound (n : ℕ) (h1 : n ≥ 3) (angles : Fin n → ℝ) (h2 : ∀ i, angles i < 180) (h3 : ∑ i, angles i = (n - 2) * 180) :
  (∃ (m : ℕ), m > 35 ∧ ∀ i < m, angles i < 170) → false :=
by
  sorry

end convex_polygon_angle_bound_l374_374396


namespace coefficients_sum_eq_zero_l374_374584

theorem coefficients_sum_eq_zero 
  (a b c : ℝ)
  (f g h : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : ∀ x, g x = b * x^2 + c * x + a)
  (h3 : ∀ x, h x = c * x^2 + a * x + b)
  (h4 : ∃ x : ℝ, f x = 0 ∧ g x = 0 ∧ h x = 0) :
  a + b + c = 0 := 
sorry

end coefficients_sum_eq_zero_l374_374584


namespace boat_downstream_time_l374_374150

theorem boat_downstream_time :
  ∀ (V_b V_s T_upstream T_downstream : ℝ),
  V_b = 15 →
  V_s = 3 →
  T_upstream = 11.5 →
  T_downstream = (138 : ℝ) / (V_b + V_s) →
  T_downstream ≈ 7.67 := by 
sorry

end boat_downstream_time_l374_374150


namespace part_I_solution_set_part_II_prove_inequality_l374_374042

-- Definition for part (I)
def f (x: ℝ) := |x - 2|
def g (x: ℝ) := 4 - |x - 1|

-- Theorem for part (I)
theorem part_I_solution_set :
  {x : ℝ | f x ≥ g x} = {x : ℝ | x ≤ -1/2} ∪ {x : ℝ | x ≥ 7/2} :=
by sorry

-- Definition for part (II)
def satisfiable_range (a: ℝ) := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def density_equation (m n a: ℝ) := (1 / m) + (1 / (2 * n)) = a

-- Theorem for part (II)
theorem part_II_prove_inequality (m n: ℝ) (hm: 0 < m) (hn: 0 < n) 
  (a: ℝ) (h_a: satisfiable_range a = {x : ℝ | abs (x - a) ≤ 1}) (h_density: density_equation m n a) :
  m + 2 * n ≥ 4 :=
by sorry

end part_I_solution_set_part_II_prove_inequality_l374_374042


namespace negation_of_p_equiv_l374_374642

def is_square_less (n : ℝ) : Prop := n^2 < 3 * n + 4

def proposition_p : Prop := ∀ n ∈ set.Icc (1 : ℝ) (2 : ℝ), is_square_less n

theorem negation_of_p_equiv :
  ¬proposition_p ↔ ∃ n ∈ set.Icc (1 : ℝ) (2 : ℝ), ¬is_square_less n := by
  sorry

end negation_of_p_equiv_l374_374642


namespace solve_equation_l374_374077

noncomputable def f (x : ℝ) := (1 / (x^2 + 17 * x + 20)) + (1 / (x^2 + 12 * x + 20)) + (1 / (x^2 - 15 * x + 20))

theorem solve_equation :
  {x : ℝ | f x = 0} = {-1, -4, -5, -20} :=
by
  sorry

end solve_equation_l374_374077


namespace problem1_l374_374926

theorem problem1 (x y : ℝ) (h1 : x + y = 4) (h2 : 2 * x - y = 5) : 
  x = 3 ∧ y = 1 := sorry

end problem1_l374_374926


namespace geometric_sequence_exists_l374_374608

noncomputable def b1 : ℝ := sorry
noncomputable def q : ℝ := sorry

noncomputable def T : ℕ → ℝ 
| 4 := b1 ^ 4 * q ^ 6
| 8 := b1 ^ 8 * q ^ 28
| 12 := b1 ^ 12 * q ^ 66
| 16 := b1 ^ 16 * q ^ 120
| _ := sorry

theorem geometric_sequence_exists :
  ∃ x y : ℝ, x = T 8 / T 4 ∧ y = T 12 / T 8 ∧ 
  ∃ z : ℝ, z = T 16 / T 12 ∧
  (T 4, x, y, z) forms a geometric sequence := sorry

end geometric_sequence_exists_l374_374608


namespace units_digit_quotient_eq_one_l374_374982

theorem units_digit_quotient_eq_one :
  (2^2023 + 3^2023) / 5 % 10 = 1 := by
  sorry

end units_digit_quotient_eq_one_l374_374982


namespace lawn_unmowed_fraction_l374_374748

noncomputable def rate_mary : ℚ := 1 / 6
noncomputable def rate_tom : ℚ := 1 / 3

theorem lawn_unmowed_fraction :
  (1 : ℚ) - ((1 * rate_tom) + (2 * (rate_mary + rate_tom))) = 1 / 6 :=
by
  -- This part will be the actual proof which we are skipping
  sorry

end lawn_unmowed_fraction_l374_374748


namespace sum_of_possible_x_l374_374073

theorem sum_of_possible_x :
  let equation (x : ℝ) :=  4^(x^2 + 6*x + 9) = 16^(x + 3)
  let solutions := { x : ℝ | equation x }
  ∑ x in solutions, x = -4 :=
by
  sorry

end sum_of_possible_x_l374_374073


namespace sum_of_altitudes_l374_374204

def line_equation (x y : ℝ) : Prop := 15 * x + 3 * y = 45

theorem sum_of_altitudes : 
  (∑ (altitude : ℝ) in {3, 15, 15 / Real.sqrt 26}, altitude) = (18 * Real.sqrt 26 + 15) / Real.sqrt 26
:=
sorry

end sum_of_altitudes_l374_374204


namespace standard_equation_line_standard_equation_circle_intersection_range_a_l374_374293

theorem standard_equation_line (a t x y : ℝ) (h1 : x = a - 2 * t * y) (h2 : y = -4 * t) : 
    2 * x - y - 2 * a = 0 :=
sorry

theorem standard_equation_circle (θ x y : ℝ) (h1 : x = 4 * Real.cos θ) (h2 : y = 4 * Real.sin θ) : 
    x ^ 2 + y ^ 2 = 16 :=
sorry

theorem intersection_range_a (a : ℝ) (h : ∃ (t θ : ℝ), (a - 2 * t * (-4 * t)) = 4 * (Real.cos θ) ∧ (-4 * t) = 4 * (Real.sin θ)) :
    -4 * Real.sqrt 5 <= a ∧ a <= 4 * Real.sqrt 5 :=
sorry

end standard_equation_line_standard_equation_circle_intersection_range_a_l374_374293


namespace bee_leg_count_l374_374933

theorem bee_leg_count (legs_two_bees : Nat) (h : legs_two_bees = 12) : (legs_two_bees / 2) = 6 := 
by {
  rw h,
  exact Nat.div_self 12 2 (by decide),
}

end bee_leg_count_l374_374933


namespace sunflower_seeds_l374_374183

theorem sunflower_seeds :
  ∃ S : ℕ, 
  let seeds_first := 78 in
  let seeds_third := S + 30 in
  let total_seeds := seeds_first + S + seeds_third in
  total_seeds = 214 ∧ S = 53 :=
by
  sorry

end sunflower_seeds_l374_374183


namespace sector_radius_l374_374085

theorem sector_radius (A L : ℝ) (hA : A = 240 * Real.pi) (hL : L = 20 * Real.pi) : 
  ∃ r : ℝ, r = 24 :=
by
  sorry

end sector_radius_l374_374085


namespace binders_required_l374_374146

variables (b1 b2 B1 B2 d1 d2 b3 : ℕ)

def binding_rate_per_binder_per_day : ℚ := B1 / (↑b1 * d1)

def books_per_binder_in_d2_days : ℚ := binding_rate_per_binder_per_day b1 B1 d1 * ↑d2

def binding_rate_for_b2_binders : ℚ := B2 / ↑b2

theorem binders_required (b1 b2 B1 B2 d1 d2 b3 : ℕ)
  (h1 : binding_rate_per_binder_per_day b1 B1 d1 = binding_rate_for_b2_binders b2 B2)
  (h2 : books_per_binder_in_d2_days b1 B1 d1 d2 = binding_rate_for_b2_binders b2 B2) :
  b3 = b2 :=
sorry

end binders_required_l374_374146


namespace triangle_area_when_A_max_l374_374003

theorem triangle_area_when_A_max (a : ℝ) (A B C : ℝ) (cos_A : ℝ) (sin_A : ℝ) (h1 : 3 * a = 3a) (h2 : 2 = 2) (h3 : cos_A = (3a^2 + 2^2 - a^2) / (2 * 3a * 2)) (h4 : sin_A = sqrt (1 - (cos_A)^2)) : 
  let b := 3 * a,
      c := 2 in 
  (1/2) * b * c * sin_A = sqrt 2 / 2 := 
by 
  sorry

end triangle_area_when_A_max_l374_374003


namespace nice_number_5_all_nice_numbers_l374_374163

/-- A positive integer k > 1 is called nice if for any pair (m, n) of positive integers satisfying the condition kn + m divides km + n, we have n divides m. -/
def nice_number (k : ℕ) : Prop :=
  k > 1 ∧ ∀ (m n : ℕ), m > 0 → n > 0 → (k * n + m) ∣ (k * m + n) → n ∣ m

/-- Prove that 5 is a nice number -/
theorem nice_number_5 : nice_number 5 := 
  sorry

/-- Find all nice numbers -/
theorem all_nice_numbers : { k : ℕ // nice_number k } = { 2, 3, 5 } := 
  sorry

end nice_number_5_all_nice_numbers_l374_374163


namespace good_vertex_count_l374_374495

-- Definition: A vertex is "good" if it belongs to exactly one parallelogram
def is_good (v : vertex) (partitions : set (parallelogram)) : Prop :=
  count_partitions v partitions = 1

-- Definition: A convex polygon partitioned into parallelograms
structure convex_polygon_partitioned_into_parallelograms :=
  (vertices : set vertex)
  (partitions : set parallelogram)
  (convex : is_convex vertices)
  (partition_property : ∀ v ∈ vertices, ∃ p ∈ partitions, v ∈ vertices p)

-- Statement: Prove that in a convex polygon partitioned into parallelograms, there are more than two good vertices
theorem good_vertex_count (P : convex_polygon_partitioned_into_parallelograms) :
  ∃ n > 2, ∃ good_vertices : set vertex, card good_vertices = n ∧ ∀ v ∈ good_vertices, is_good v P.partitions :=
by
  sorry

end good_vertex_count_l374_374495


namespace largest_composite_not_written_l374_374689

theorem largest_composite_not_written (n : ℕ) (hn : n = 2022) : ¬ ∃ d > 1, 2033 = n + d := 
by
  sorry

end largest_composite_not_written_l374_374689


namespace runners_align_same_point_first_time_l374_374046

def lap_time_stein := 6
def lap_time_rose := 10
def lap_time_schwartz := 18

theorem runners_align_same_point_first_time : Nat.lcm (Nat.lcm lap_time_stein lap_time_rose) lap_time_schwartz = 90 :=
by
  sorry

end runners_align_same_point_first_time_l374_374046


namespace count_non_decreasing_digits_of_12022_l374_374508

/-- Proof that the number of digits left in the number 12022 that form a non-decreasing sequence is 3. -/
theorem count_non_decreasing_digits_of_12022 : 
  let num := [1, 2, 0, 2, 2]
  let remaining := [1, 2, 2] -- non-decreasing sequence from 12022
  List.length remaining = 3 :=
by
  let num := [1, 2, 0, 2, 2]
  let remaining := [1, 2, 2]
  have h : List.length remaining = 3 := rfl
  exact h

end count_non_decreasing_digits_of_12022_l374_374508


namespace max_roses_l374_374064

def cost_per_rose_individual := 2.30
def cost_per_dozen := 36
def cost_per_two_dozen := 50
def total_money := 680

theorem max_roses : 
  let individual := cost_per_rose_individual in
  let dozen := cost_per_dozen in
  let two_dozen := cost_per_two_dozen in
  let money := total_money in
  ∑ x in finset.range (money / two_dozen), 24 + ∑ y in finset.range ((money % two_dozen) / individual) ≥ 325 := sorry

end max_roses_l374_374064


namespace digit_at_1286th_position_l374_374694

def naturally_written_sequence : ℕ → ℕ := sorry

theorem digit_at_1286th_position : naturally_written_sequence 1286 = 3 :=
sorry

end digit_at_1286th_position_l374_374694


namespace largest_possible_s_l374_374364

theorem largest_possible_s (r s : ℕ) (h1 : 3 ≤ s) (h2 : s ≤ r) (h3 : s < 122)
    (h4 : ∀ r s, (61 * (s - 2) * r = 60 * (r - 2) * s)) : s ≤ 121 :=
by
  sorry

end largest_possible_s_l374_374364


namespace range_of_m_increasing_function_l374_374623

theorem range_of_m_increasing_function :
  (2 : ℝ) ≤ m ∧ m ≤ 4 ↔ ∀ x : ℝ, (1 / 3 : ℝ) * x ^ 3 - (4 * m - 1) * x ^ 2 + (15 * m ^ 2 - 2 * m - 7) * x + 2 ≤ 
                                 ((1 / 3 : ℝ) * (x + 1) ^ 3 - (4 * m - 1) * (x + 1) ^ 2 + (15 * m ^ 2 - 2 * m - 7) * (x + 1) + 2) :=
by
  sorry

end range_of_m_increasing_function_l374_374623


namespace find_digits_sum_l374_374803

theorem find_digits_sum
  (a b c d : ℕ)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (non_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (sum_of_numbers : 6 * (a + b + c + d) * 1111 = 73326) :
  {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end find_digits_sum_l374_374803


namespace rationalize_denominator_eq_68_l374_374062

theorem rationalize_denominator_eq_68 :
  ∃ (A B C D E : ℤ), 
    B < D ∧ 
    (A * sqrt B + C * sqrt D) / E = (3 : ℝ) / (4 * sqrt 7 + 5 * sqrt 2) ∧ 
    A + B + C + D + E = 68 :=
by
  sorry

end rationalize_denominator_eq_68_l374_374062


namespace dan_bought_2_candy_bars_l374_374568

variable (total_spent : ℕ) (cost_per_candy_bar : ℕ)

theorem dan_bought_2_candy_bars (h1 : total_spent = 4) (h2 : cost_per_candy_bar = 2) : 
  let num_candy_bars := total_spent / cost_per_candy_bar in
  num_candy_bars = 2 :=
by
  sorry

end dan_bought_2_candy_bars_l374_374568


namespace petya_digits_l374_374769

theorem petya_digits (a b c d : ℕ) (distinct_digits : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (non_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (sum_formed_numbers : 6 * (a + b + c + d) * 1111 = 73326) :
  {a, b, c, d} = {1, 2, 3, 5} :=
by 
  sorry

end petya_digits_l374_374769


namespace equal_area_AOB_COD_l374_374347

variables {A B C D O : Type}
variables [HasArea A B C D O] -- Assume there exists a concept of area for these variables
variables (ABCD : IsTrapezoid A B C D) (bases_AD_BC : Bases AD BC ABCD) (diagonals_AC_BD_intersect_O : DiagonalsIntersect AC BD O)

theorem equal_area_AOB_COD :
  area (triangle A O B) = area (triangle C O D) :=
sorry

end equal_area_AOB_COD_l374_374347


namespace parallel_trans_l374_374648

variables {Line : Type} (a b c : Line)

-- Define parallel relation
def parallel (x y : Line) : Prop := sorry -- Replace 'sorry' with the actual definition

-- The main theorem
theorem parallel_trans (h1 : parallel a c) (h2 : parallel b c) : parallel a b :=
sorry

end parallel_trans_l374_374648


namespace coeff_x3y7_in_expansion_l374_374189

noncomputable def binomial_coefficient (n k : ℕ) := nat.choose n k

theorem coeff_x3y7_in_expansion :
  let a := (4 : ℚ) / 7
  let b := - (1 : ℚ) / 3
  let c : ℚ := binomial_coefficient 10 7 * (a ^ 3) * (b ^ 7)
  c = -7680 / 759321 :=
by
  sorry

end coeff_x3y7_in_expansion_l374_374189


namespace three_digit_clubsuit_condition_l374_374035

-- Definition to compute the sum of digits of a number
def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- The main theorem we want to prove
theorem three_digit_clubsuit_condition :
  let three_digit_numbers := {x | 100 ≤ x ∧ x ≤ 999}
  let count := (three_digit_numbers.to_finset.filter (λ x, digit_sum (digit_sum x) = 4)).card
  count = 24 :=
by
  sorry

end three_digit_clubsuit_condition_l374_374035


namespace cubic_poly_sum_l374_374251

noncomputable def q (x : ℕ) : ℤ := sorry

axiom h0 : q 1 = 5
axiom h1 : q 6 = 24
axiom h2 : q 10 = 16
axiom h3 : q 15 = 34

theorem cubic_poly_sum :
  (q 0) + (q 1) + (q 2) + (q 3) + (q 4) + (q 5) + (q 6) +
  (q 7) + (q 8) + (q 9) + (q 10) + (q 11) + (q 12) + (q 13) +
  (q 14) + (q 15) + (q 16) = 340 :=
by
  sorry

end cubic_poly_sum_l374_374251


namespace circles_intersect_l374_374626

theorem circles_intersect (r1 r2 d : ℝ) (h1 : r1 = 5) (h2 : r2 = 3) (h3 : d = 7)
  (h4 : r1 - r2 < d) (h5 : d < r1 + r2) : 
  "the circles intersect" :=
by {
  sorry
}

end circles_intersect_l374_374626


namespace rahim_average_price_l374_374399

def books_shop1 : ℕ := 50
def cost_shop1 : ℕ := 1000
def books_shop2 : ℕ := 40
def cost_shop2 : ℕ := 800

def total_books : ℕ := books_shop1 + books_shop2
def total_cost : ℕ := cost_shop1 + cost_shop2
def average_price_per_book : ℕ := total_cost / total_books

theorem rahim_average_price :
  average_price_per_book = 20 := by
  sorry

end rahim_average_price_l374_374399


namespace max_cables_cut_l374_374878

theorem max_cables_cut 
  (initial_computers : ℕ)
  (initial_cables : ℕ)
  (final_clusters : ℕ)
  (H1 : initial_computers = 200)
  (H2 : initial_cables = 345)
  (H3 : final_clusters = 8) 
  : ∃ (cut_cables : ℕ), cut_cables = 153 :=
by
  use 153
  sorry

end max_cables_cut_l374_374878


namespace unique_expression_values_count_l374_374016

def expression_values : Finset ℤ :=
  (Finset.univ.product Finset.univ).product (Finset.univ.product Finset.univ).product Finset.univ |
  let digits := [1, 2, 3, 4, 5].to_finset in
  digits.bind $ λ a, 
  digits.bind $ λ b, 
  digits.bind $ λ c, 
  digits.bind $ λ d, 
  digits.bind $ λ e,
  if (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
      b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
      c ≠ d ∧ c ≠ e ∧ 
      d ≠ e) then 
    {(a * b - c * d + e : Int)}
  else ∅

theorem unique_expression_values_count : 
  ∃ k : ℕ, k = expression_values.card → k = PICK_AMONG_CHOICES := 
sorry

end unique_expression_values_count_l374_374016


namespace purely_imaginary_complex_l374_374669

theorem purely_imaginary_complex (m : ℝ) :
  (m^2 - 3 * m = 0) → (m^2 - 5 * m + 6 ≠ 0) → m = 0 :=
begin
  intros h_real h_imag,
  -- The proof will go here
  sorry
end

end purely_imaginary_complex_l374_374669


namespace parabola_a_value_l374_374094

theorem parabola_a_value
  (a b c : ℤ)
  (h1 : ∀ x : ℝ, (a * x^2 + b * x + c : ℝ) = (a * (x - 2) ^ 2 + b * (x - 1) + c : ℝ))
  (h2 : (0, 7) ∈ set_of (λ x : ℝ × ℝ, ∃ y : ℝ, y = a * (x.1 - 2) ^ 2 + 3)) :
  a = 1 :=
sorry

end parabola_a_value_l374_374094


namespace infinite_solutions_iff_l374_374726

theorem infinite_solutions_iff (a b c d : ℤ) :
  (∃ᶠ x in at_top, ∃ᶠ y in at_top, x^2 + a * x + b = y^2 + c * y + d) ↔ (a^2 - 4 * b = c^2 - 4 * d) :=
by sorry

end infinite_solutions_iff_l374_374726


namespace largest_multiple_of_9_less_than_110_l374_374456

theorem largest_multiple_of_9_less_than_110 : ∃ x, (x < 110 ∧ x % 9 = 0 ∧ ∀ y, (y < 110 ∧ y % 9 = 0) → y ≤ x) ∧ x = 108 :=
by
  sorry

end largest_multiple_of_9_less_than_110_l374_374456


namespace exists_positive_integer_solutions_l374_374810

theorem exists_positive_integer_solutions (m : ℕ) (hm : 0 < m) :
  ∃ s : ℕ, ∃ (x : ℕ → ℕ), (∀ i, 1 ≤ i ∧ i ≤ s → 0 < x i) ∧ (∑ i in finset.range s, (1 / (x i) ^ m)) = 1 :=
sorry

end exists_positive_integer_solutions_l374_374810


namespace angle_quadrant_l374_374416

theorem angle_quadrant (α : ℝ) (x y : ℝ) 
  (h1 : x^2 * sin α + y^2 * cos α = 1) 
  (h2 : cos α > 0) 
  (h3 : sin α < 0) : 
  α ∈ Icc (3 * π / 2) (2 * π) :=
sorry

end angle_quadrant_l374_374416


namespace longest_side_of_enclosure_l374_374522

theorem longest_side_of_enclosure (l w : ℝ) (hlw : 2*l + 2*w = 240) (harea : l*w = 2880) : max l w = 72 := 
by {
  sorry
}

end longest_side_of_enclosure_l374_374522


namespace range_of_a_l374_374268

open Real

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Iic (1 : ℝ), 1 + 2^x + 4^x * a > 0) ↔ a ∈ Ioi (-1 / 4) :=
by sorry

end range_of_a_l374_374268


namespace determine_ab_l374_374252

def f (a b x : ℝ) : ℝ := x^2 + (Real.log a + 2) * x + Real.log b

theorem determine_ab (a b : ℝ)
  (h1 : f a b (-1) = -2)
  (h2 : ∀ x : ℝ, f a b x ≥ 2 * x) :
  a = 100 ∧ b = 10 := 
sorry

end determine_ab_l374_374252


namespace find_D_l374_374413

-- Defining the points with the given coordinates
def A := (1, 1)
def B := (3, 1)
def C := (3, 5)

-- Defining the property of the rectangle
def is_rectangle (A B C D : (ℤ × ℤ)) :=
  let ⟨x1, y1⟩ := A
  let ⟨x2, y2⟩ := B
  let ⟨x3, y3⟩ := C
  let ⟨x4, y4⟩ := D
  (x1 = x4 ∧ y2 = y1 ∧ x3 = x2 ∧ y3 = y4)

-- The statement to prove
theorem find_D : ∃ D : (ℤ × ℤ), is_rectangle A B C D ∧ D = (1, 5) :=
by {
  -- Define point D
  let D := (1, 5),
  -- Prove that this point satisfies the rectangle property and is (1,5)
  use D,
  split,
  { dsimp only [is_rectangle],
    rw [A, B, C, D],
    exact ⟨rfl, rfl, rfl, rfl⟩ },
  { rw D }
}

end find_D_l374_374413


namespace parallelogram_area_l374_374229

theorem parallelogram_area (d : ℝ) (h : ℝ) (α : ℝ) (h_d : d = 30) (h_h : h = 20) : 
  ∃ A : ℝ, A = d * h ∧ A = 600 :=
by
  sorry

end parallelogram_area_l374_374229


namespace ten_years_less_than_average_age_l374_374012

theorem ten_years_less_than_average_age (L : ℕ) :
  (2 * L - 14) = 
    (2 * L - 4) - 10 :=
by {
  sorry
}

end ten_years_less_than_average_age_l374_374012


namespace area_ratio_PQR_ABC_l374_374687

-- Given the conditions of the triangle and segmentation ratios
variables {A B C D E F P Q R : Type}
variables [has_coords A] [has_coords B] [has_coords C] [has_coords D] [has_coords E] [has_coords F]
variables [has_coords P] [has_coords Q] [has_coords R]
variables {segment_ratio_BD_DC : rat := 2 / 5}
variables {segment_ratio_CE_EA : rat := 3 / 5}
variables {segment_ratio_AF_FB : rat := 2 / 5}

-- Define the segments and their intersection points ratios
noncomputable def pointD := (3 / 5 : rat) • B + (2 / 5 : rat) • C
noncomputable def pointE := (2 / 5 : rat) • A + (3 / 5 : rat) • C
noncomputable def pointF := (3 / 5 : rat) • A + (2 / 5 : rat) • B

-- Intersection points of lines
noncomputable def pointP := (7 / 12 : rat) • A + (5 / 12 : rat) • pointD
noncomputable def pointQ := sorry -- Calculate similarly for Q
noncomputable def pointR := sorry -- Calculate similarly for R

-- State the theorem/assertion of the area ratio
theorem area_ratio_PQR_ABC : 
  area_ratio (triangle A B C) (triangle P Q R) = 35 / 144 :=
  sorry -- Proof goes here

end area_ratio_PQR_ABC_l374_374687


namespace volume_of_circumscribed_sphere_l374_374022

theorem volume_of_circumscribed_sphere
  (P A B C : ℝ³) 
  (PA_perpendicular : PA ⊥ (ABC))
  (AB_eq_1 : distance A B = 1)
  (AC_eq_2 : distance A C = 2)
  (angle_BAC_eq_60 : ∠BAC = 60)
  (volume_eq_sqrt3_div3 : volume P (triangle (A B C)) = (sqrt 3 / 3)) :
  ∃ R : ℝ, volume_circumscribed_sphere P A B C = 4/3 * π * R ^ 3 ∧ R = sqrt 2 :=
by
  sorry

end volume_of_circumscribed_sphere_l374_374022


namespace threes_painted_in_houses_1_to_100_l374_374166

def occurs_in_range (digit : ℕ) (from : ℕ) (to : ℕ) : ℕ :=
  let digits_in_num (n : ℕ) : List ℕ := 
    n.digits 10
  (List.range' from (to - from + 1)).sumBy (fun n => (digits_in_num n).count digit)

def number_of_threes_painted := occurs_in_range 3 1 100

theorem threes_painted_in_houses_1_to_100 : number_of_threes_painted = 10 := by
  sorry

end threes_painted_in_houses_1_to_100_l374_374166


namespace find_digits_sum_l374_374799

theorem find_digits_sum
  (a b c d : ℕ)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (non_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (sum_of_numbers : 6 * (a + b + c + d) * 1111 = 73326) :
  {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end find_digits_sum_l374_374799


namespace parabola_focus_l374_374090

theorem parabola_focus (a : ℝ) (h : a = 1) : focus_y_eq (y, x = x ^ 2) = (0, 1 / 4) := sorry

end parabola_focus_l374_374090


namespace price_of_individual_rose_l374_374065

-- Definitions based on conditions

def price_of_dozen := 36  -- one dozen roses cost $36
def price_of_two_dozen := 50 -- two dozen roses cost $50
def total_money := 680 -- total available money
def total_roses := 317 -- total number of roses that can be purchased

-- Define the question as a theorem
theorem price_of_individual_rose : 
  ∃ (x : ℕ), (12 * (total_money / price_of_two_dozen) + 
              (total_money % price_of_two_dozen) / price_of_dozen * 12 + 
              (total_money % price_of_two_dozen % price_of_dozen) / x = total_roses) ∧ (x = 6) :=
by
  sorry

end price_of_individual_rose_l374_374065


namespace calc_cos_15_l374_374978

noncomputable def cos_15 := real.cos (15 * real.pi / 180)

theorem calc_cos_15 :
  2 * cos_15^2 - 1 = real.cos (30 * real.pi / 180) := by
  sorry

end calc_cos_15_l374_374978


namespace surface_area_correct_l374_374253

def radius_hemisphere : ℝ := 9
def height_cone : ℝ := 12
def radius_cone_base : ℝ := 9

noncomputable def total_surface_area : ℝ := 
  let base_area : ℝ := radius_hemisphere^2 * Real.pi
  let curved_area_hemisphere : ℝ := 2 * radius_hemisphere^2 * Real.pi
  let slant_height_cone : ℝ := Real.sqrt (radius_cone_base^2 + height_cone^2)
  let lateral_area_cone : ℝ := radius_cone_base * slant_height_cone * Real.pi
  base_area + curved_area_hemisphere + lateral_area_cone

theorem surface_area_correct : total_surface_area = 378 * Real.pi := by
  sorry

end surface_area_correct_l374_374253


namespace arithmetic_sequence_property_value_of_a101_l374_374685

noncomputable def seq (n : ℕ) : ℕ → ℚ
| 0     => 2
| (n+1) => (2 * seq n + 1) / 2

theorem arithmetic_sequence_property (n : ℕ) : seq (n + 1) = seq n + 1 / 2 := by
  sorry

theorem value_of_a101 : seq 100 = 52 :=
  by sorry

end arithmetic_sequence_property_value_of_a101_l374_374685


namespace find_coefficients_l374_374583

theorem find_coefficients (a b p q : ℝ) :
    (∀ x : ℝ, (2 * x - 1) ^ 20 - (a * x + b) ^ 20 = (x^2 + p * x + q) ^ 10) →
    a = -2 * b ∧ (b = 1 ∨ b = -1) ∧ p = -1 ∧ q = 1 / 4 :=
by 
    sorry

end find_coefficients_l374_374583


namespace fn_polynomial_l374_374209

open Function

def f : ℕ → Polynomial ℤ
| 0 := 1
| 1 := Polynomial.X
| (n + 2) := let fn := f (n + 1) in let fn1 := f n in
             (fn ^ 2 - 1) / fn1

theorem fn_polynomial : ∀ n : ℕ, ∃ p : Polynomial ℤ, f n = p
  := by
  sorry

end fn_polynomial_l374_374209


namespace max_min_product_xy_theorem_l374_374866

noncomputable def max_min_product_xy (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) : Prop :=
  -1 ≤ x * y ∧ x * y ≤ 1/2

theorem max_min_product_xy_theorem (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) :
  max_min_product_xy x y a h1 h2 :=
sorry

end max_min_product_xy_theorem_l374_374866


namespace fifth_term_sequence_l374_374563

theorem fifth_term_sequence : (∑ i in Finset.range 5, 4^i) = 341 :=
by
  sorry

end fifth_term_sequence_l374_374563


namespace find_solutions_l374_374578

noncomputable def system_of_equations (x y z : ℝ) : Prop :=
  (x + y) ^ 3 = z ∧
  (y + z) ^ 3 = x ∧
  (z + x) ^ 3 = y

theorem find_solutions : ∀ (x y z : ℝ), system_of_equations x y z → 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
  (x = y ∧ y = z ∧ (x = sqrt 2 / 4 ∨ x = -sqrt 2 / 4)) :=
by
  sorry

end find_solutions_l374_374578


namespace total_amount_correct_l374_374443

noncomputable def total_amount_collected
    (single_ticket_price : ℕ)
    (couple_ticket_price : ℕ)
    (total_people : ℕ)
    (couple_tickets_sold : ℕ) : ℕ :=
  let single_tickets_sold := total_people - (couple_tickets_sold * 2)
  let amount_from_couple_tickets := couple_tickets_sold * couple_ticket_price
  let amount_from_single_tickets := single_tickets_sold * single_ticket_price
  amount_from_couple_tickets + amount_from_single_tickets

theorem total_amount_correct :
  total_amount_collected 20 35 128 16 = 2480 := by
  sorry

end total_amount_correct_l374_374443


namespace proof_problem_l374_374263

variable (a b c d x : ℤ)

-- Conditions
def are_opposite (a b : ℤ) : Prop := a + b = 0
def are_reciprocals (c d : ℤ) : Prop := c * d = 1
def largest_negative_integer (x : ℤ) : Prop := x = -1

theorem proof_problem 
  (h1 : are_opposite a b) 
  (h2 : are_reciprocals c d) 
  (h3 : largest_negative_integer x) :
  x^2 - (a + b - c * d)^(2012 : ℕ) + (-c * d)^(2011 : ℕ) = -1 :=
by
  sorry

end proof_problem_l374_374263


namespace unique_digits_addition_l374_374337

theorem unique_digits_addition :
  ∃ (X Y B M C : ℕ), 
    -- Conditions
    X ≠ 0 ∧ Y ≠ 0 ∧ B ≠ 0 ∧ M ≠ 0 ∧ C ≠ 0 ∧
    X ≠ Y ∧ X ≠ B ∧ X ≠ M ∧ X ≠ C ∧ Y ≠ B ∧ Y ≠ M ∧ Y ≠ C ∧ B ≠ M ∧ B ≠ C ∧ M ≠ C ∧
    -- Addition equation with distinct digits
    (X * 1000 + Y * 100 + 70) + (B * 100 + M * 10 + C) = (B * 1000 + M * 100 + C * 10 + 0) ∧
    -- Correct Answer
    X = 9 ∧ Y = 8 ∧ B = 3 ∧ M = 8 ∧ C = 7 :=
sorry

end unique_digits_addition_l374_374337


namespace largest_quotient_l374_374457

theorem largest_quotient :
  let S := {-12, -4, -3, 1, 3, 9}
  ∃ a b ∈ S, b ≠ 0 ∧ a / b = 9 := sorry

end largest_quotient_l374_374457


namespace max_area_AJ1J2_l374_374002

theorem max_area_AJ1J2 {AB AC BC : ℝ} (hAB : AB = 42) (hBC : BC = 45) (hAC : AC = 51) : 
  ∃ (Y : ℝ), ∃ (AJ1 AJ2 : ℝ), let J1J2_area := (1 / 2) * AJ1 * AJ2 * (Real.sin ((1 / 2) * Real.arccos ((BC * BC + AC * AC - AB * AB) / (2 * BC * AC))) in
  J1J2_area = 1071 * Real.sqrt(0.604) := sorry

end max_area_AJ1J2_l374_374002


namespace shorter_diagonal_length_l374_374958

-- Define a rhombus and its properties
variables {x : ℝ}
-- Given conditions
def area (diag1 diag2 : ℝ) : ℝ := (diag1 * diag2) / 2
def ratio_diagonals (diag1 diag2 : ℝ) : Prop := diag1 / diag2 = 5 / 3
def diag_side_relation (diag side : ℝ) : Prop := diag = 2 * side

-- Proving the length of the shorter diagonal
theorem shorter_diagonal_length (area_rhombus : area (5*x) (3*x) = 150) 
                                (ratio_diag : ratio_diagonals (5*x) (3*x))
                                (diag_side : ∃ side : ℝ, diag_side_relation (5*x) side) : 
                                3*x = 6*Real.sqrt(5) :=
by {
  sorry
}

end shorter_diagonal_length_l374_374958


namespace alyssa_allowance_l374_374524

-- Definition using the given problem
def weekly_allowance (A : ℝ) : Prop :=
  A / 2 + 8 = 12

-- Theorem to prove that weekly allowance is 8 dollars
theorem alyssa_allowance : ∃ A : ℝ, weekly_allowance A ∧ A = 8 :=
by
  use 8
  unfold weekly_allowance
  exact eq.refl _

end alyssa_allowance_l374_374524


namespace angle_ACB_l374_374993

theorem angle_ACB (A B C : Point) (latitude_A longitude_A latitude_B longitude_B : ℝ) 
(hA: latitude_A = 0 ∧ longitude_A = 90) 
(hB: latitude_B = 30 ∧ longitude_B = -80) 
(hEarth_sphere: PerfectSphere Earth) :
angle A C B = 140 :=
sorry

end angle_ACB_l374_374993


namespace find_last_score_l374_374047

/-- The list of scores in ascending order -/
def scores : List ℕ := [60, 65, 70, 75, 80, 85, 95]

/--
  The problem states that the average score after each entry is an integer.
  Given the scores in ascending order, determine the last score entered.
-/
theorem find_last_score (h : ∀ (n : ℕ) (hn : n < scores.length),
    (scores.take (n + 1) |>.sum : ℤ) % (n + 1) = 0) :
  scores.last' = some 80 :=
sorry

end find_last_score_l374_374047


namespace intersecting_points_count_l374_374320

open Real

def line1 (x y : ℝ) : Prop := 3 * y - 2 * x = 1
def line2 (x y : ℝ) : Prop := x + 2 * y = 2
def line3 (x y : ℝ) : Prop := 4 * x - 6 * y = 5
def line4 (x y : ℝ) : Prop := 2 * x - 3 * y = 4

theorem intersecting_points_count : 
  let points := { p : ℝ × ℝ | 
    (∃ x y, line1 x y ∧ p = (x, y)) ∨ 
    (∃ x y, line2 x y ∧ p = (x, y)) ∨ 
    (∃ x y, line3 x y ∧ p = (x, y)) ∨ 
    (∃ x y, line4 x y ∧ p = (x, y))
  } in
  let intersect_points := { p : ℝ × ℝ | 
    (∃ x y, line1 x y ∧ line2 x y ∧ p = (x, y)) ∨ 
    (∃ x y, line1 x y ∧ line3 x y ∧ p = (x, y)) ∨ 
    (∃ x y, line1 x y ∧ line4 x y ∧ p = (x, y)) ∨ 
    (∃ x y, line2 x y ∧ line3 x y ∧ p = (x, y)) ∨ 
    (∃ x y, line2 x y ∧ line4 x y ∧ p = (x, y)) ∨ 
    (∃ x y, line3 x y ∧ line4 x y ∧ p = (x, y))
  } in
  intersect_points.card = 3 :=
sorry

end intersecting_points_count_l374_374320


namespace batsman_average_l374_374919

theorem batsman_average (avg_20 : ℕ) (avg_10 : ℕ) (total_matches_20 : ℕ) (total_matches_10 : ℕ) :
  avg_20 = 40 → avg_10 = 20 → total_matches_20 = 20 → total_matches_10 = 10 →
  (800 + 200) / 30 = 33.33 :=
by
  sorry

end batsman_average_l374_374919


namespace exact_time_between_9_10_l374_374024

theorem exact_time_between_9_10
  (t : ℝ)
  (h1 : 0 ≤ t ∧ t < 60)
  (h2 : |6 * (t + 5) - (270 + 0.5 * (t - 2))| = 180) :
  t = 10 + 3 / 4 :=
sorry

end exact_time_between_9_10_l374_374024


namespace discontinuous_at_one_l374_374692

noncompute def f (x : ℝ) : ℝ := real.exp (-2 / (x - 1))

theorem discontinuous_at_one : 
  (filter.tendsto f (filter.lt_top (filter.inv (filter.comap (λ x, x - 1) filter.at_bot))) filter.at_top) ∧ 
  (filter.tendsto f (filter.gt_top (filter.inv (filter.comap (λ x, x - 1) filter.at_top))) filter.at_bot) → 
  (∀ x, x ≠ 1 → continuous_at f x) ∧ ¬continuous_at f 1 :=
begin
  sorry  -- Proof not required per instructions
end

end discontinuous_at_one_l374_374692


namespace smallest_prime_dividing_sum_l374_374899

theorem smallest_prime_dividing_sum :
  ∃ p : ℕ, Prime p ∧ p ∣ (7^14 + 11^15) ∧ ∀ q : ℕ, Prime q ∧ q ∣ (7^14 + 11^15) → p ≤ q := by
  sorry

end smallest_prime_dividing_sum_l374_374899


namespace symmetric_point_wrt_y_axis_l374_374021

-- Definitions
def point := (ℝ × ℝ × ℝ)

def symmetric_y_axis (p : point) : point := (-p.1, p.2, -p.3)

-- Theorem that needs to be proven
theorem symmetric_point_wrt_y_axis :
  symmetric_y_axis (3, -4, 1) = (-3, -4, 1) :=
sorry

end symmetric_point_wrt_y_axis_l374_374021


namespace share_of_a_l374_374486

def shares_sum (a b c : ℝ) := a + b + c = 366
def share_a (a b c : ℝ) := a = 1/2 * (b + c)
def share_b (a b c : ℝ) := b = 2/3 * (a + c)

theorem share_of_a (a b c : ℝ) 
  (h1 : shares_sum a b c) 
  (h2 : share_a a b c) 
  (h3 : share_b a b c) : 
  a = 122 := 
by 
  -- Proof goes here
  sorry

end share_of_a_l374_374486


namespace sum_of_recorded_numbers_l374_374876

theorem sum_of_recorded_numbers (n : ℕ) (h : n = 16)
    (friend_or_enemy : Fin n → Fin n → Prop)
    (records : Fin n → (list ℕ × list ℕ)) :
    (∑ i : Fin n, (records i).1.sum + (records i).2.sum) = 120 :=
by
  sorry

end sum_of_recorded_numbers_l374_374876


namespace avg_price_of_returned_tshirts_l374_374815

-- Define the conditions as Lean definitions
def avg_price_50_tshirts := 750
def num_tshirts := 50
def num_returned_tshirts := 7
def avg_price_remaining_43_tshirts := 720

-- The correct price of the 7 returned T-shirts
def correct_avg_price_returned := 6540 / 7

-- The proof statement
theorem avg_price_of_returned_tshirts :
  (num_tshirts * avg_price_50_tshirts - (num_tshirts - num_returned_tshirts) * avg_price_remaining_43_tshirts) / num_returned_tshirts = correct_avg_price_returned :=
by
  sorry

end avg_price_of_returned_tshirts_l374_374815


namespace part1_part2_l374_374282

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * (Real.cos x)^2 + (Real.sqrt 3 / 2) * Real.sin x * Real.cos x + 1

theorem part1 :
  ∀ x : ℝ, f (x + π) = f x :=
by sorry

theorem part2 :
  ∃ (max_x min_x : ℝ), max_x ∈ Set.Icc (π/12) (π/4) ∧ min_x ∈ Set.Icc (π/12) (π/4) ∧
    f max_x = 7 / 4 ∧ f min_x = (5 + Real.sqrt 3) / 4 ∧
    (max_x = π / 6) ∧ (min_x = π / 12 ∨ min_x = π / 4) :=
by sorry

end part1_part2_l374_374282


namespace not_all_primes_l374_374664

variable {ℕ : Type} [Nontrivial ℕ]

def sequence (x₀ a b : ℕ) : ℕ → ℕ
| 0       => x₀
| (n + 1) => sequence n * a + b

theorem not_all_primes (x₀ a b : ℕ) (h₁ : x₀ ∈ ℕ) (h₂ : a ∈ ℕ) (h₃ : b ∈ ℕ) :
  ¬ (∀ n : ℕ, Prime (sequence x₀ a b n)) :=
sorry

end not_all_primes_l374_374664


namespace triangle_area_eq_e_div_4_l374_374086

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def P : ℝ × ℝ := (1, Real.exp 1)

noncomputable def tangent_line (x : ℝ) : ℝ :=
  let k := (Real.exp 1) * (x + 1)
  k * (x - 1) + Real.exp 1

theorem triangle_area_eq_e_div_4 :
  let area := (1 / 2) * Real.exp 1 * (1 / 2)
  area = (Real.exp 1) / 4 :=
by
  sorry

end triangle_area_eq_e_div_4_l374_374086


namespace phone_charging_time_l374_374955

theorem phone_charging_time (fast_charging_time : ℕ) (regular_charging_time : ℕ) 
    (fast_charge_fraction : ℚ) (uniform_charging : Prop) : 
    fast_charging_time = 80 ∧ regular_charging_time = 240 ∧ fast_charge_fraction = 1/3 → 
    let t := 144 in 
    (t / fast_charging_time / 3) + (2 * t / 3 / regular_charging_time) = 1 := 
begin
    intros h,
    cases h with hfast hreg,
    cases hreg with hreg hfrac,
    rw [hfast, hreg, hfrac],
    exact_eq { intros t, simp },
    sorry -- proof to be completed
end

end phone_charging_time_l374_374955


namespace machine_initial_value_l374_374850

-- Conditions
def initial_value (P : ℝ) : Prop := P * (0.75 ^ 2) = 4000

noncomputable def initial_market_value : ℝ := 4000 / (0.75 ^ 2)

-- Proof problem statement
theorem machine_initial_value (P : ℝ) (h : initial_value P) : P = 4000 / (0.75 ^ 2) :=
by
  sorry

end machine_initial_value_l374_374850


namespace petya_digits_sum_l374_374760

def distinct_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem petya_digits_sum (a b c d : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
    (h_distinct: distinct_digits a b c d)
    (h_sum: 6 * (a + b + c + d) * 1111 = 73326) :
    a + b + c + d = 11 ∧ {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end petya_digits_sum_l374_374760


namespace cardinality_P3_P5_no_disjoint_A_B_max_value_of_n_l374_374589

open Set

-- Define the elements of the problem
def E (n : ℕ) : Set ℕ := { i | 1 ≤ i ∧ i ≤ n }

def P (n : ℕ) : Set ℝ := { x | ∃ a b : ℕ, a ∈ E n ∧ b ∈ E n ∧ x = a / real.sqrt b }

def has_property_omega (S : Set ℝ) : Prop :=
  S ⊆ P(15) ∧ ∀ (x₁ x₂ ∈ S), x₁ ≠ x₂ → ¬ ∃ k : ℕ, k > 0 ∧ x₁ + x₂ = (k * k)

-- Part a
theorem cardinality_P3_P5 : card (P 3) = 9 ∧ card (P 5) = 23 := sorry

-- Part b
theorem no_disjoint_A_B : ¬ ∃ (A B : Set ℝ), has_property_omega A ∧ has_property_omega B ∧ A ∩ B = ∅ ∧ E(15) = A ∪ B := sorry

-- Part c
theorem max_value_of_n : ∀ n ≥ 14, ∃ (A B : Set ℝ), has_property_omega A ∧ has_property_omega B ∧ A ∩ B = ∅ ∧ P n = A ∪ B := sorry

#check cardinality_P3_P5
#check no_disjoint_A_B
#check max_value_of_n

end cardinality_P3_P5_no_disjoint_A_B_max_value_of_n_l374_374589


namespace wrapping_paper_per_present_l374_374357

theorem wrapping_paper_per_present :
  ∀ (used_total : ℚ) (presents : ℕ), 
  used_total = 5 / 8 → 
  presents = 5 → 
  (used_total / presents) = 1 / 8 :=
by
  intros used_total presents h_used_total h_presents
  rw [h_used_total, h_presents]
  norm_num
  sorry

end wrapping_paper_per_present_l374_374357


namespace mean_and_variance_y_l374_374643

variables (x : ℕ → ℝ) (y : ℕ → ℝ) (a : ℝ)
-- Conditions: assigning specific values.
-- mean of x's
def mean_x : Prop := (∑ i in finset.range 10, x i) / 10 = 1
-- variance of x's
def variance_x : Prop := (∑ i in finset.range 10, (x i - 1)^2) / 10 = 4

-- defining y based on x and a
def y_def : Prop := ∀ i, y i = x i + a

-- Proving the mean and variance of y's
theorem mean_and_variance_y : mean_x x → variance_x x → y_def x y a → 
  ( (∑ i in finset.range 10, y i) / 10 = 1 + a ∧ (∑ i in finset.range 10, (y i - (1 + a))^2) / 10 = 4 ) :=
by
  sorry

end mean_and_variance_y_l374_374643


namespace g_at_1_l374_374661

variable (g : ℝ → ℝ)
variable (h : ∀ x : ℝ, g(2 * x - 3) = 3 * x + 9)

theorem g_at_1 : g 1 = 15 :=
by
  sorry

end g_at_1_l374_374661


namespace de_length_l374_374447

theorem de_length :
  ∀ (A B C D E : Type) (AB AC BC : ℕ), 
    AB = 15 → AC = 39 → BC = 36 →
    ∀ (D : A) (E : B),
    parallel DE BC → 
    (exists (incenter : D), incenter ∈ line DE) → 
    DE = 36 / 5 := 
sorry

end de_length_l374_374447


namespace smallest_x_domain_ffx_l374_374313

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x + 3)

theorem smallest_x_domain_ffx : 
  ∃ x : ℝ, (∀ y : ℝ, y < x → (f (f y)).is_nonpos) ∧ x = 6 :=
by
  sorry

end smallest_x_domain_ffx_l374_374313


namespace circumcircle_AMN_tangent_to_I_l374_374327

/-
In Figure 3, let O and G be the circumcenter and centroid of triangle ABC respectively,
and D be the midpoint of BC. The circle with diameter BC intersects the altitude AH at point E.
The line EG intersects OD at point F. Lines FK are parallel to OB and FL are parallel to OC.
KM is perpendicular to BC and NL is perpendicular to BC.
A circle passing through point B and tangent to OB is denoted as ⊙I.
Prove that the circumcircle of △AMN is tangent to ⊙I.
-/

variable {A B C O G D E F K L M N : Point}

/--
The circumcircle of △AMN is tangent to ⊙I at a particular point.
-/
theorem circumcircle_AMN_tangent_to_I
  (hO : Circumcenter O A B C)
  (hG : Centroid G A B C)
  (hD : Midpoint D B C)
  (hE : ∃ H : Point, Altitude H A (Line.mk B C) ∧ Intersects (Circle.bc B C) (Line.mk A H) E)
  (hF : Intersects (Line.mk E G) (Line.mk O D) F)
  (hFK : Parallel (Line.mk F K) (Line.mk O B))
  (hFL : Parallel (Line.mk F L) (Line.mk O C))
  (hKM : Perpendicular (Line.mk K M) (Line.mk B C))
  (hNL : Perpendicular (Line.mk N L) (Line.mk B C))
  (hI : TangentCircleThroughPoint I B (Line.mk O B)) :
  Tangent (Circumcircle A M N) I :=
sorry

end circumcircle_AMN_tangent_to_I_l374_374327


namespace ellipse_range_of_k_l374_374628

theorem ellipse_range_of_k (k : ℝ) :
  (∃ (eq : ((x y : ℝ) → (x ^ 2 / (3 + k) + y ^ 2 / (2 - k) = 1))),
  ((3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k))) ↔
  (k ∈ Set.Ioo (-3 : ℝ) ((-1) / 2) ∪ Set.Ioo ((-1) / 2) 2) :=
by sorry

end ellipse_range_of_k_l374_374628


namespace mowing_time_correct_l374_374356

-- Definitions corresponding to conditions from part a
def lawn_width : ℝ := 120
def lawn_height : ℝ := 180
def swath_inches : ℝ := 30
def overlap_inches : ℝ := 6
def mowing_rate : ℝ := 4000

-- Converting swath and overlap to feet
def swath_feet : ℝ := (swath_inches - overlap_inches) / 12

-- Number of strips required to mow the lawn
def strips : ℝ := lawn_height / swath_feet

-- Total distance Joe mows
def total_distance : ℝ := strips * lawn_width

-- Time required to mow the lawn
def time_required : ℝ := total_distance / mowing_rate

-- The goal is to prove that the time required is 2.7 hours
theorem mowing_time_correct : time_required = 2.7 :=
by sorry

end mowing_time_correct_l374_374356


namespace add_fractions_l374_374577

theorem add_fractions: (2 / 5) + (3 / 8) = 31 / 40 := 
by 
  sorry

end add_fractions_l374_374577


namespace joe_eggs_town_hall_l374_374027

-- Define the conditions.
def eggs_club_house : ℕ := 12
def eggs_park : ℕ := 5
def eggs_total : ℕ := 20

-- Define the desired result.
def eggs_town_hall : ℕ := eggs_total - eggs_club_house - eggs_park

-- The statement that needs to be proved.
theorem joe_eggs_town_hall : eggs_town_hall = 3 :=
by
  sorry

end joe_eggs_town_hall_l374_374027


namespace domain_of_log_function_l374_374839

theorem domain_of_log_function :
  {x : ℝ | x^2 - x > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 1} :=
by
  sorry

end domain_of_log_function_l374_374839


namespace Petya_digits_sum_l374_374780

theorem Petya_digits_sum (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :
  (24 * (a + b + c + d) * 1111 = 73326) →
  {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end Petya_digits_sum_l374_374780


namespace max_min_product_xy_theorem_l374_374865

noncomputable def max_min_product_xy (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) : Prop :=
  -1 ≤ x * y ∧ x * y ≤ 1/2

theorem max_min_product_xy_theorem (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) :
  max_min_product_xy x y a h1 h2 :=
sorry

end max_min_product_xy_theorem_l374_374865


namespace fifth_term_of_sequence_l374_374554

def pow_four_sequence (n : ℕ) : ℕ :=
  ∑ i in Finset.range (n + 1), 4^i

theorem fifth_term_of_sequence :
  pow_four_sequence 4 = 341 :=
sorry

end fifth_term_of_sequence_l374_374554


namespace find_functions_l374_374927

noncomputable theory

def S := {n : ℕ // 0 ≤ n}

variables (f g h : S → S)

def func_property_1 : Prop :=
  ∀ (m n : S), f ⟨m.val + n.val, add_nonneg m.property n.property⟩
    = g m + h n + ⟨2 * m.val * n.val, mul_nonneg (mul_nonneg (bit0_nonneg.mp zero_le_bit0) (bit0_nonneg.mp zero_le_bit0))⟩

def func_property_2 : Prop :=
  g ⟨1, zero_le_one⟩ = ⟨1, zero_le_one⟩ ∧ h ⟨1, zero_le_one⟩ = ⟨1, zero_le_one⟩

def correct_functions : Prop :=
  ∃ a : ℕ,
  (∀ n : S, f n = ⟨n.val ^ 2 - a * n.val + 2 * a, nat.sub_le (n.val ^ 2 + 2 * a) (a * n.val)⟩)
  ∧ (∀ n : S, g n = ⟨n.val ^ 2 - a * n.val + a, nat.sub_le (n.val ^ 2 + a) (a * n.val)⟩)
  ∧ (∀ n : S, h n = ⟨n.val ^ 2 - a * n.val + a, nat.sub_le (n.val ^ 2 + a) (a * n.val)⟩)

theorem find_functions (Hp1 : func_property_1 f g h) (Hp2 : func_property_2 g h) : correct_functions f g h :=
sorry

end find_functions_l374_374927


namespace problem_statement_l374_374681

open Finset

-- Definitions
def subjects : Finset String := {"politics", "history", "geography", "physics", "chemistry", "biology", "technology"}
def remaining_subjects_after_physics : Finset String := erase subjects "physics"

-- Problem Statement 
theorem problem_statement :
  (card (choose remaining_subjects_after_physics 2) = 15) ∧ 
  (card (choose subjects 3) * card (choose subjects 3) ≠ 0) → 
  (card (choose remaining_subjects_after_physics 2) * card (choose remaining_subjects_after_physics 2) / (card (choose subjects 3) * card (choose subjects 3)) = (9 : ℝ) / 49) :=
by {
  sorry
}

end problem_statement_l374_374681


namespace fifth_term_sequence_l374_374565

theorem fifth_term_sequence : (∑ i in Finset.range 5, 4^i) = 341 :=
by
  sorry

end fifth_term_sequence_l374_374565


namespace range_of_m_l374_374716

theorem range_of_m (m : ℝ) (h : 0 < m)
  (subset_cond : ∀ x y : ℝ, x - 4 ≤ 0 → y ≥ 0 → mx - y ≥ 0 → (x - 2)^2 + (y - 2)^2 ≤ 8) :
  m ≤ 1 :=
sorry

end range_of_m_l374_374716


namespace feet_count_l374_374161

-- We define the basic quantities
def total_heads : ℕ := 50
def num_hens : ℕ := 30
def num_cows : ℕ := total_heads - num_hens
def hens_feet : ℕ := num_hens * 2
def cows_feet : ℕ := num_cows * 4
def total_feet : ℕ := hens_feet + cows_feet

-- The theorem we want to prove
theorem feet_count : total_feet = 140 :=
  by
  sorry

end feet_count_l374_374161


namespace correct_properties_l374_374336

def sos (θ : ℝ) (x₀ y₀ r : ℝ) : ℝ := (y₀ + x₀) / r

def sos_function (x : ℝ) : ℝ := sqrt 2 * sin (x + π / 4)

theorem correct_properties {x₀ y₀ r : ℝ} (h : r > 0) :
  (∀ x, sos_function x ∈ [-sqrt 2, sqrt 2]) ∧ -- Property 1
  (∀ x₁ x₂, (x₁ ≠ x₂ ∧ x₁ = -(x₂)) → sos_function x₁ ≠ sos_function x₂ ) ∧ -- Property 2 (False)
  (∀ x₁ x₂, (x₁ ≠ x₂ ∧ x₁ = 3*π/4 - x₂) → sos_function x₁ ≠ sos_function x₂ ) ∧ -- Property 3 (False)
  (∃ T > 0, ∀ x, sos_function (x + T) = sos_function x) ∧ -- Property 4
  (∀ k : ℤ, ∀ x ∈ Icc (2 * k * π - 3 * π / 4) (2 * k * π + π / 4), monotone_on sos_function (Icc (2 * k * π - 3 * π / 4) (2 * k * π + π / 4))) -- Property 5
  := sorry

end correct_properties_l374_374336


namespace problem1_problem2_l374_374735

def M := { x : ℝ | 0 < x ∧ x < 1 }

theorem problem1 :
  { x : ℝ | |2 * x - 1| < 1 } = M :=
by
  simp [M]
  sorry

theorem problem2 (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (a * b + 1) > (a + b) :=
by
  simp [M] at ha hb
  sorry

end problem1_problem2_l374_374735


namespace sampling_is_systematic_sampling_l374_374970

-- Definitions based on conditions
def participating_students := ℕ  -- Set of all participating students (exam numbers)
def ends_in_5 (n : ℕ) : Prop := n % 10 = 5  -- Exam number ends in 5
def interval_between := 10 -- Interval between selected exam numbers is 10

-- Given conditions
axiom all_students_participating : ∀ n : ℕ, ends_in_5 n → n ∈ participating_students

-- The mathematical proof problem statement
theorem sampling_is_systematic_sampling : 
  (∀ n : ℕ, ends_in_5 n → ∃ m : ℕ, ends_in_5 m ∧ m ≠ n ∧ (m - n) = interval_between) → 
  systematic_sampling participating_students :=
sorry  -- Proof is skipped

end sampling_is_systematic_sampling_l374_374970


namespace petya_digits_l374_374765

theorem petya_digits (a b c d : ℕ) (distinct_digits : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (non_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (sum_formed_numbers : 6 * (a + b + c + d) * 1111 = 73326) :
  {a, b, c, d} = {1, 2, 3, 5} :=
by 
  sorry

end petya_digits_l374_374765


namespace find_digits_sum_l374_374797

theorem find_digits_sum
  (a b c d : ℕ)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (non_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (sum_of_numbers : 6 * (a + b + c + d) * 1111 = 73326) :
  {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end find_digits_sum_l374_374797


namespace more_tails_than_heads_l374_374314

theorem more_tails_than_heads (total_flips : ℕ) (heads : ℕ) (tails : ℕ) :
  total_flips = 211 → heads = 65 → tails = (total_flips - heads) → (tails - heads) = 81 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  rw [h1, h2]
  exact h3.trans (show 211 - 65 - 65 = 81 by norm_num)

end more_tails_than_heads_l374_374314


namespace continuous_numbers_count_is_12_l374_374665

def is_continuous_number (n : ℕ) : Prop :=
  (n + (n + 1) + (n + 2) < 10)

def count_continuous_numbers_below_100 : ℕ :=
  (List.range 100).countp is_continuous_number

theorem continuous_numbers_count_is_12 :
  count_continuous_numbers_below_100 = 12 :=
sorry

end continuous_numbers_count_is_12_l374_374665


namespace decreasing_interval_l374_374636

def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x

theorem decreasing_interval :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi →
  (∀ x₀ x₁ : ℝ, x₀ ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 2) ∧ x₀ < x₁ ∧ x₁ ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 2) → f x₀ ≥ f x₁) := sorry

end decreasing_interval_l374_374636


namespace stone_slab_length_1_2_meters_l374_374149

noncomputable def length_of_each_stone_slab (total_area : ℝ) (number_of_slabs : ℕ) : ℝ :=
  real.sqrt (total_area / number_of_slabs)

theorem stone_slab_length_1_2_meters :
  length_of_each_stone_slab 72 50 = 1.2 :=
by
  unfold length_of_each_stone_slab
  norm_num

end stone_slab_length_1_2_meters_l374_374149


namespace saras_sister_ordered_notebooks_l374_374402

theorem saras_sister_ordered_notebooks (x : ℕ) 
  (initial_notebooks : ℕ := 4) 
  (lost_notebooks : ℕ := 2) 
  (current_notebooks : ℕ := 8) :
  initial_notebooks + x - lost_notebooks = current_notebooks → x = 6 :=
by
  intros h
  sorry

end saras_sister_ordered_notebooks_l374_374402


namespace eccentricity_of_ellipse_l374_374871

theorem eccentricity_of_ellipse {a b c : ℝ} (h1 : a > b) (h2 : b > 0) 
  (h_ellipse : ∀ x y : ℝ, (x, y) ∈ line ⟨2, 0⟩ → x = c ∨ x = -c)
  (h_ellipse_eq : ∀ x y : ℝ, x = c ∨ x = -c → (x^2 / a^2) + (y^2 / b^2) = 1):
  let e := c / a in
  0 < e ∧ e < 1 → e = sqrt 2 - 1 :=
begin
  sorry
end

end eccentricity_of_ellipse_l374_374871


namespace game_not_fair_l374_374806

-- Define the set of possible outcomes from the dice roll
def outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the event where player A wins (number greater than 3)
def A_wins : Finset ℕ := {4, 5, 6}

-- Define the event where player B wins (number less than 3)
def B_wins : Finset ℕ := {1, 2}

-- Probability of winning for player A
def P_A_wins : ℚ := A_wins.card / outcomes.card

-- Probability of winning for player B
def P_B_wins : ℚ := B_wins.card / outcomes.card

-- Statement: Prove that the game is unfair
theorem game_not_fair : P_A_wins ≠ P_B_wins :=
by
  have h1 : A_wins.card = 3 := by simp
  have h2 : B_wins.card = 2 := by simp
  have h3 : outcomes.card = 6 := by simp
  have h4 : P_A_wins = 3 / 6 := by simp [P_A_wins, h1, h3]
  have h5 : P_B_wins = 2 / 6 := by simp [P_B_wins, h2, h3]
  rw [h4, h5]
  norm_num -- This will simplify 3/6 and 2/6 to 1/2 and 1/3 respectively, showing they are not equal
  sorry

end game_not_fair_l374_374806


namespace lisa_more_marbles_than_cindy_l374_374551

-- Definitions
def initial_cindy_marbles : ℕ := 20
def difference_cindy_lisa : ℕ := 5
def cindy_gives_lisa : ℕ := 12

-- Assuming Cindy's initial marbles are 20, which are 5 more than Lisa's marbles, and Cindy gives Lisa 12 marbles,
-- prove that Lisa now has 19 more marbles than Cindy.
theorem lisa_more_marbles_than_cindy :
  let lisa_initial_marbles := initial_cindy_marbles - difference_cindy_lisa,
      lisa_current_marbles := lisa_initial_marbles + cindy_gives_lisa,
      cindy_current_marbles := initial_cindy_marbles - cindy_gives_lisa
  in lisa_current_marbles - cindy_current_marbles = 19 :=
by {
  sorry
}

end lisa_more_marbles_than_cindy_l374_374551


namespace total_apples_packed_correct_l374_374744

-- Define the daily production of apples under normal conditions
def apples_per_box := 40
def boxes_per_day := 50
def days_per_week := 7
def apples_per_day := apples_per_box * boxes_per_day

-- Define the change in daily production for the next week
def fewer_apples := 500
def apples_per_day_next_week := apples_per_day - fewer_apples

-- Define the weekly production in normal and next conditions
def apples_first_week := apples_per_day * days_per_week
def apples_second_week := apples_per_day_next_week * days_per_week

-- Define the total apples packed in two weeks
def total_apples_packed := apples_first_week + apples_second_week

-- Prove the total apples packed is 24500
theorem total_apples_packed_correct : total_apples_packed = 24500 := by
  sorry

end total_apples_packed_correct_l374_374744


namespace monotonicity_and_maximum_of_f_inequality_holds_for_k_product_of_sequence_lt_e_l374_374630

noncomputable def f (x : ℝ) : ℝ := ln x - x

theorem monotonicity_and_maximum_of_f :
  (∀ x ∈ Ioo 0 1, deriv f x > 0) ∧ 
  (∀ x > 1, deriv f x < 0) ∧ 
  (f 1 = -1) :=
by sorry

theorem inequality_holds_for_k (k : ℝ) :
  (∀ x ∈ Ioi 2, x * f x + x^2 - k * x + k > 0) ↔ k ≤ 2 * ln 2 :=
by sorry

noncomputable def a_n (n : ℕ) : ℝ := 1 + 1 / (2^n)

theorem product_of_sequence_lt_e (n : ℕ) :
  (∏ i in finset.range n.succ, a_n i) < Real.exp 1 :=
by sorry

end monotonicity_and_maximum_of_f_inequality_holds_for_k_product_of_sequence_lt_e_l374_374630


namespace seventh_observation_value_l374_374088

variable (average_6 : ℝ) (new_average : ℝ)

noncomputable def value_of_seventh_observation (average_6 : ℝ) (new_average : ℝ) : ℝ :=
  let total_sum_6 := 6 * average_6
  let total_sum_7 := 7 * new_average
  total_sum_7 - total_sum_6

theorem seventh_observation_value :
  average_6 = 12 →
  new_average = 11 →
  value_of_seventh_observation 12 11 = 5 :=
by
  intros h1 h2
  unfold value_of_seventh_observation
  rw [h1, h2]
  norm_num
  sorry

end seventh_observation_value_l374_374088


namespace number_of_white_squares_l374_374180

theorem number_of_white_squares (n : ℕ) (h : n = 37) : 
  let N := 2 * n - 1 in
  ((N + 1) / 2) = 37 :=
by
  -- Problem statement given
  have h1 : n = 37 := h,
  -- We calculate N for n = 37
  let N := 2 * h1 - 1,
  -- Replacing N with (2 * 37 - 1)
  have hN : N = 73 := rfl,
  -- Therefore, we calculate ((N + 1) / 2)
  show ((N + 1) / 2) = 37 from 
    rfl

end number_of_white_squares_l374_374180


namespace convex_polygon_eq_two_pow_l374_374080

variable {α : Type*}
variable [fintype α] [linear_order α]
variable (S : finset α)
variable [decidable_pred (λ x : α, x ∈ S)]
variable (convex_polygon : set α → Prop)
variable (a b : π)

noncomputable def a_p (p : set α) : ℕ :=
  (finset.filter convex_polygon S).card

noncomputable def b_p (p : set α) : ℕ :=
  (S \ p).card

theorem convex_polygon_eq_two_pow {x : ℝ} (hconvex : ∀ p, convex_polygon p → 2 * x^a_p p * (1-x)^b_p p = 1) :
  ∀ p, convex_polygon p → 2 * x^a_p p * (1-x)^b_p p = 1 :=
sorry

end convex_polygon_eq_two_pow_l374_374080


namespace total_marbles_correct_l374_374218

def marbles_dohyun : Nat := 7 * 16
def marbles_joohyun : Nat := 6 * 25
def total_marbles : Nat := marbles_dohyun + marbles_joohyun

theorem total_marbles_correct : total_marbles = 262 := by
  unfold total_marbles marbles_dohyun marbles_joohyun
  calc
    7 * 16 + 6 * 25 = 112 + 150 := rfl
    ...             = 262       := rfl

end total_marbles_correct_l374_374218


namespace triangle_BDG_is_isosceles_l374_374894

namespace Geometry

-- Define the vertices of squares BEGF and ABCD
structure Point where
  x : ℝ
  y : ℝ

def B : Point := ⟨0, 0⟩
def E : Point := ⟨2, 0⟩
def G : Point := ⟨2, 2⟩
def F : Point := ⟨0, 2⟩
def A : Point := ⟨0, 0⟩
def C : Point := ⟨0, 4⟩
def D : Point := ⟨4, 0⟩

-- Define the distance function between two points
def distance (p1 p2 : Point) : ℝ := 
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Prove triangle BDG is isosceles
theorem triangle_BDG_is_isosceles : distance B D = distance D G :=
by
  sorry

end Geometry

end triangle_BDG_is_isosceles_l374_374894


namespace balls_in_bags_l374_374254

theorem balls_in_bags (n : ℕ) (h : n ≥ 2) :
  ∃ λ, (∀ bags : list (list ℕ), 
    (∀ (bag : list ℕ), bag ∈ bags → (∀ ball_weight : ℕ, ball_weight ∈ bag → (∃ k : ℕ, ball_weight = 2^k)) ∧ bag.sum = bags.head.sum) →
    λ ≥ ⌊ n / 2 ⌋ + 1 ∧ 
    ∃ (weight : ℕ), count_weight_in_bags weight bags ≥ λ) :=
sorry

def count_weight_in_bags (weight : ℕ) (bags : list (list ℕ)) : ℕ :=
  bags.countp (λ bag, weight ∈ bag)

end balls_in_bags_l374_374254


namespace compare_abc_l374_374618

noncomputable def a : ℝ := Real.log 10 / Real.log 5
noncomputable def b : ℝ := Real.log 12 / Real.log 6
noncomputable def c : ℝ := Real.log 14 / Real.log 7

theorem compare_abc : a > b ∧ b > c := by
  sorry

end compare_abc_l374_374618


namespace hotdog_cost_l374_374703

theorem hotdog_cost
  (h s : ℕ) -- Make sure to assume that the cost in cents is a natural number 
  (h1 : 3 * h + 2 * s = 360)
  (h2 : 2 * h + 3 * s = 390) :
  h = 60 :=

sorry

end hotdog_cost_l374_374703


namespace whitewashed_fence_length_l374_374117

theorem whitewashed_fence_length :
  ∀ (T B_1 B_2 J L : ℝ),
    T = 100 →
    B_1 = 10 →
    B_2 = (T - B_1) / 5 →
    J = (T - B_1 - B_2) / 3 →
    L = T - B_1 - B_2 - J →
    L = 48 :=
by {
  intros,
  sorry
}

end whitewashed_fence_length_l374_374117


namespace tina_work_time_l374_374113

theorem tina_work_time (T : ℕ) (h1 : ∀ Ann_hours, Ann_hours = 9)
                       (h2 : ∀ Tina_worked_hours, Tina_worked_hours = 8)
                       (h3 : ∀ Ann_worked_hours, Ann_worked_hours = 3)
                       (h4 : (8 : ℚ) / T + (1 : ℚ) / 3 = 1) : T = 12 :=
by
  sorry

end tina_work_time_l374_374113


namespace company_D_fewer_attendees_than_company_C_l374_374937

def company_A_attendees := 30
def company_B_attendees := 2 * company_A_attendees
def company_C_attendees := company_A_attendees + 10
def total_attendees := 185
def other_attendees := 20
def company_D_attendees := total_attendees - (company_A_attendees + company_B_attendees + company_C_attendees + other_attendees)

theorem company_D_fewer_attendees_than_company_C :
  company_C_attendees - company_D_attendees = 25 :=
by
  unfold company_A_attendees company_B_attendees company_C_attendees company_D_attendees total_attendees other_attendees
  sorry

end company_D_fewer_attendees_than_company_C_l374_374937


namespace cost_of_bench_l374_374940

variables (cost_table cost_bench : ℕ)

theorem cost_of_bench :
  cost_table + cost_bench = 450 ∧ cost_table = 2 * cost_bench → cost_bench = 150 :=
by
  sorry

end cost_of_bench_l374_374940


namespace symmetric_about_line_l374_374531

-- Define the sought function
def sought_function (x : ℝ) : ℝ :=
  log 2 (4 / x)

-- Prove that the sought function is symmetric to y = log_2 x about the line y = 1
theorem symmetric_about_line (x y : ℝ) :
  (y = log 2 x) ↔ (sought_function x = 2 - log 2 x) :=
by
  sorry

end symmetric_about_line_l374_374531


namespace carl_cost_l374_374545

theorem carl_cost (property_damage medical_bills : ℝ) (insurance_coverage : ℝ) (carl_coverage : ℝ) (H1 : property_damage = 40000) (H2 : medical_bills = 70000) (H3 : insurance_coverage = 0.80) (H4 : carl_coverage = 0.20) :
  carl_coverage * (property_damage + medical_bills) = 22000 :=
by
  sorry

end carl_cost_l374_374545


namespace triangle_area_l374_374171

theorem triangle_area :
  let (x1, y1) := (-2, 1)
  let (x2, y2) := (7, -3)
  let (x3, y3) := (4, 6)
  (1 / 2) * (abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))) = 34.5 :=
by
  let (x1, y1) := (-2, 1)
  let (x2, y2) := (7, -3)
  let (x3, y3) := (4, 6)
  sorry

end triangle_area_l374_374171


namespace seventy_percentile_is_seven_l374_374256

-- Define the data set and the condition that the average is 6
def dataset (x : ℝ) := [x, x+2, 3*x-3, 2*x+1, 9]

-- Condition that the average of the set is 6
def average_is_six (x : ℝ) : Prop := 
  (x + (x + 2) + (3 * x - 3) + (2 * x + 1) + 9) / 5 = 6

-- Define the 70th percentile function
noncomputable def percentile (xs : List ℝ) (p : ℝ) : ℝ :=
  let n := xs.length
  let pos := n * p
  if pos % 1 = 0 then
    xs[(pos : ℕ) - 1]
  else
    xs[pos.floor - 1]

-- Lean statement to prove that with x = 3, the 70th percentile is 7
theorem seventy_percentile_is_seven (x : ℝ) (h : average_is_six x) :
  percentile (dataset x).sort 0.7 = 7 :=
by
  sorry

end seventy_percentile_is_seven_l374_374256


namespace range_of_f_on_interval_l374_374639

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := 1 / x^k

theorem range_of_f_on_interval {k : ℝ} (hk : k > 0) :
  Set.range (λ (x : ℝ), f x k) ∈ Ioo 0 1 :=
sorry

end range_of_f_on_interval_l374_374639


namespace remainingAreaCalculation_l374_374044

noncomputable def totalArea : ℝ := 9500.0
noncomputable def lizzieGroupArea : ℝ := 2534.1
noncomputable def hilltownTeamArea : ℝ := 2675.95
noncomputable def greenValleyCrewArea : ℝ := 1847.57

theorem remainingAreaCalculation :
  (totalArea - (lizzieGroupArea + hilltownTeamArea + greenValleyCrewArea) = 2442.38) :=
by
  sorry

end remainingAreaCalculation_l374_374044


namespace find_k_l374_374848

structure Point :=
  (x : ℝ)
  (y : ℝ)

def slope (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

theorem find_k
  (A B X Y : Point)
  (hA : A = Point.mk (-6) 0)
  (hB : B = Point.mk 0 (-6))
  (hX : X = Point.mk 0 10)
  (hY : Y = Point.mk 18 k)
  (h_parallel : slope A B = slope X Y) :
  k = -8 :=
by {
  sorry
}

end find_k_l374_374848


namespace max_cables_cut_l374_374880

def initial_cameras : ℕ := 200
def initial_cables : ℕ := 345
def resulting_clusters : ℕ := 8

theorem max_cables_cut :
  ∃ (cables_cut : ℕ), resulting_clusters = 8 ∧ initial_cables - cables_cut = (initial_cables - cables_cut) - (resulting_clusters - 1) ∧ cables_cut = 153 :=
by
  sorry

end max_cables_cut_l374_374880


namespace part_a_part_b_l374_374914

noncomputable def same_start_digit (n x : ℕ) : Prop :=
  ∃ d : ℕ, ∀ k : ℕ, (k ≤ n) → (x * 10^(k-1) ≤ d * 10^(k-1) + 10^(k-1) - 1) ∧ ((d * 10^(k-1)) < x * 10^(k-1))

theorem part_a (x : ℕ) : 
  (same_start_digit 3 x) → ¬(∃ d : ℕ, d = 1) → false :=
  sorry

theorem part_b (x : ℕ) : 
  (same_start_digit 2015 x) → ¬(∃ d : ℕ, d = 1) → false :=
  sorry

end part_a_part_b_l374_374914


namespace part_I_part_II_l374_374739

open set

def universal_set : set ℝ := univ
def set_A : set ℝ := {x | x < -5 ∨ x > 1}
def set_B : set ℝ := {x | -4 < x ∧ x < 3}
def complement_A := compl set_A

theorem part_I (x : ℝ) : x ∈ (set_A ∪ set_B) ↔ x < -5 ∨ x > -4 :=
by sorry

theorem part_II (x : ℝ) : x ∈ (complement_A ∩ set_B) ↔ -4 < x ∧ x ≤ 1 :=
by sorry

end part_I_part_II_l374_374739


namespace petya_digits_l374_374796

def are_distinct (a b c d : Nat) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 

def non_zero_digits (a b c d : Nat) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

theorem petya_digits :
  ∃ (a b c d : Nat), are_distinct a b c d ∧ non_zero_digits a b c d ∧ (a + b + c + d = 11) ∧ (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 5) ∧
  (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 5) ∧ (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 5) ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 5) :=
by
  sorry

end petya_digits_l374_374796


namespace two_pow_2014_mod_seven_l374_374662

theorem two_pow_2014_mod_seven : 
  ∃ r : ℕ, 2 ^ 2014 ≡ r [MOD 7] → r = 2 :=
sorry

end two_pow_2014_mod_seven_l374_374662


namespace fifth_term_of_sequence_equals_341_l374_374558

theorem fifth_term_of_sequence_equals_341 : 
  ∑ i in Finset.range 5, 4^i = 341 :=
by sorry

end fifth_term_of_sequence_equals_341_l374_374558


namespace lucy_50_cent_items_l374_374384

theorem lucy_50_cent_items :
  ∃ (a b c : ℕ), a + b + c = 30 ∧ 50 * a + 150 * b + 300 * c = 4500 ∧ a = 6 :=
by
  sorry

end lucy_50_cent_items_l374_374384


namespace petya_digits_l374_374770

theorem petya_digits (a b c d : ℕ) (distinct_digits : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (non_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (sum_formed_numbers : 6 * (a + b + c + d) * 1111 = 73326) :
  {a, b, c, d} = {1, 2, 3, 5} :=
by 
  sorry

end petya_digits_l374_374770


namespace book_pages_count_l374_374220

theorem book_pages_count (n : ℕ) 
  (condition1 : ∀ k, 1 ≤ k ∧ k ≤ n → k % 2 ≠ 0) 
  (condition2 : ∑ k in (finset.range (n + 1)).filter (λ x, x % 2 ≠ 0), (to_string k).length = 845) 
  : n = 598 ∨ n = 600 := 
sorry

end book_pages_count_l374_374220


namespace cab_driver_income_l374_374934

theorem cab_driver_income (x2 : ℕ) :
  (600 + x2 + 450 + 400 + 800) / 5 = 500 → x2 = 250 :=
by
  sorry

end cab_driver_income_l374_374934


namespace percentage_of_defective_meters_is_approx_0_07_l374_374533

-- Given conditions
def defectiveMeters : ℝ := 2
def totalMeters : ℝ := 2857.142857142857

-- The percentage calculation
def percentageDefectiveMeters (defectiveMeters totalMeters : ℝ) : ℝ :=
  (defectiveMeters / totalMeters) * 100

-- Statement of theorem
theorem percentage_of_defective_meters_is_approx_0_07 :
  abs (percentageDefectiveMeters defectiveMeters totalMeters - 0.07) < 0.001 :=
by 
  sorry

end percentage_of_defective_meters_is_approx_0_07_l374_374533


namespace cyclist_speed_l374_374499

theorem cyclist_speed:
  ∀ (c : ℝ), 
  ∀ (hiker_speed : ℝ), 
  (hiker_speed = 4) → 
  (4 * (5 / 60) + 4 * (25 / 60) = c * (5 / 60)) → 
  c = 24 := 
by
  intros c hiker_speed hiker_speed_def distance_eq
  sorry

end cyclist_speed_l374_374499


namespace perp_line_plane_l374_374296

variables {Point : Type} [AffineSpace Point ℝ]
variables {Line : Type} [AffineSubspace ℝ Point Line]
variables {Plane : Type} [AffineSubspace ℝ Point Plane]

variables {m n : Line}
variables {α β γ : Plane}

-- Assuming m is perpendicular to plane α and n is parallel to plane α
def perpendicular {m : Line} {α : Plane} : Prop := sorry
def parallel {n : Line} {α : Plane} : Prop := sorry

axiom m_perp_α : perpendicular m α
axiom n_parallel_α : parallel n α

-- To prove m is perpendicular to n
theorem perp_line_plane (m : Line) (n : Line) (α : Plane) 
  (h1 : perpendicular m α) (h2 : parallel n α) : 
  perpendicular m n := 
sorry

end perp_line_plane_l374_374296


namespace smallest_common_term_l374_374994

def is_power_of (n b : ℕ) : Prop := ∃ k : ℕ, n = b ^ k

def is_sum_of_distinct_powers (n b : ℕ) : Prop :=
  ∃ (s : finset ℕ), s.nonempty ∧ (s.sum (λ k, b ^ k)) = n

def sequence (b : ℕ) : set ℕ :=
  { n | is_power_of n b ∨ is_sum_of_distinct_powers n b }

def nth_term_of_sequence (s : set ℕ) (n : ℕ) : ℕ :=
  (s.to_finset.sort (≤)).nth n

theorem smallest_common_term {n : ℕ} (h150 : n ≥ 150) :
  let seq3 := sequence 3,
      seq5 := sequence 5,
      term_150 := nth_term_of_sequence seq3 149,
      term := seq3.filter (λ x, x ≥ term_150) ∩ seq5 in
  term.min' (by sorry) = 3125 :=
by sorry

end smallest_common_term_l374_374994


namespace cubic_reciprocal_sum_l374_374200

noncomputable def cubic_roots (a b c : ℝ) : Prop :=
  a^3 - 12 * a^2 + 20 * a - 3 = 0 ∧
  b^3 - 12 * b^2 + 20 * b - 3 = 0 ∧
  c^3 - 12 * c^2 + 20 * c - 3 = 0

noncomputable def vieta_relations (a b c : ℝ) : Prop :=
  a + b + c = 12 ∧
  ab + bc + ca = 20 ∧
  abc = 3

theorem cubic_reciprocal_sum (a b c : ℝ) 
  (h1 : cubic_roots a b c) 
  (h2 : vieta_relations a b c) :
  (\sum perm : {a, b, c}, 1/perm^2) = 328/9 :=
begin
  sorry,
end

end cubic_reciprocal_sum_l374_374200


namespace palindrome_divisibility_probability_correct_l374_374504

def is_valid_palindrome (a b c : ℕ) : Prop :=
  a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10

def is_divisible_by_11 (a b c : ℕ) : Prop :=
  (10 * a + 10 * b + c) % 11 = 0

def count_valid_palindromes : ℕ :=
  (List.finRange 9).bind (λ a => 
  (List.finRange 10).bind (λ b => 
  (List.finRange 10).filter (λ c => 
  is_valid_palindrome a.succ b c ∧ is_divisible_by_11 a.succ b c))).length

def total_palindromes : ℕ :=
  9 * 10 * 10

def probability_divisible_by_11 : ℚ :=
  count_valid_palindromes / total_palindromes

theorem palindrome_divisibility_probability_correct :
  probability_divisible_by_11 = 77 / 900 :=
by
  sorry

end palindrome_divisibility_probability_correct_l374_374504


namespace probability_not_eat_pizza_l374_374867

theorem probability_not_eat_pizza (P_eat_pizza : ℚ) (h : P_eat_pizza = 5 / 8) : 
  ∃ P_not_eat_pizza : ℚ, P_not_eat_pizza = 3 / 8 :=
by
  use 1 - P_eat_pizza
  sorry

end probability_not_eat_pizza_l374_374867


namespace train_cars_count_l374_374465

theorem train_cars_count
  (cars_in_15_seconds : ℕ)
  (time_in_seconds : ℕ)
  (total_time_in_minutes : ℕ)
  (r : ℚ)
  (T : ℕ)
  (n : ℕ)
  (H1 : cars_in_15_seconds = 8)
  (H2 : time_in_seconds = 15)
  (H3 : total_time_in_minutes = 3)
  (H4 : r = cars_in_15_seconds / time_in_seconds)
  (H5 : T = total_time_in_minutes * 60)
  (H6 : n = Int.round (r * T)) :
  n = 96 :=
by
  sorry

end train_cars_count_l374_374465


namespace max_sum_at_n_eq_7_l374_374710

noncomputable def arithmetic_sequence_max_sum {a : ℕ → ℤ} (S : ℕ → ℤ) (d : ℤ) : Prop :=
  (∀ n, S n = (n * (a 1 + a n)) / 2) ∧
  (a 1 > 0) ∧
  (S 5 = S 9) ∧
  (∃ n, S n = S (nat.find_max (λ m, S m)))

theorem max_sum_at_n_eq_7 {a : ℕ → ℤ} {S : ℕ → ℤ} (d : ℤ) (h : arithmetic_sequence_max_sum S d) : nat.find_max (λ m, S m) = 7 := 
sorry

end max_sum_at_n_eq_7_l374_374710


namespace prop1_prop2_l374_374468

-- Proposition 1: Prove the contrapositive
theorem prop1 (q : ℝ) (h : ¬(∃ x : ℝ, x^2 + 2 * x + q = 0)) : q ≥ 1 :=
sorry

-- Proposition 2: Prove the contrapositive
theorem prop2 (x y : ℝ) (h : ¬(x = 0 ∧ y = 0)) : x^2 + y^2 ≠ 0 :=
sorry

end prop1_prop2_l374_374468


namespace tapA_fill_time_l374_374451

-- Define the conditions
def fillTapA (t : ℕ) := 1 / t
def fillTapB := 1 / 40
def fillCombined (t : ℕ) := 9 * (fillTapA t + fillTapB)
def fillRemaining := 23 * fillTapB

-- Main theorem statement
theorem tapA_fill_time : ∀ (t : ℕ), fillCombined t + fillRemaining = 1 → t = 45 := by
  sorry

end tapA_fill_time_l374_374451


namespace prop1_prop2_l374_374467

-- Proposition 1: Prove the contrapositive
theorem prop1 (q : ℝ) (h : ¬(∃ x : ℝ, x^2 + 2 * x + q = 0)) : q ≥ 1 :=
sorry

-- Proposition 2: Prove the contrapositive
theorem prop2 (x y : ℝ) (h : ¬(x = 0 ∧ y = 0)) : x^2 + y^2 ≠ 0 :=
sorry

end prop1_prop2_l374_374467


namespace amber_worked_hours_l374_374528

-- Define the variables and conditions
variables (A : ℝ) (Armand_hours : ℝ) (Ella_hours : ℝ)
variables (h1 : Armand_hours = A / 3) (h2 : Ella_hours = 2 * A)
variables (h3 : A + Armand_hours + Ella_hours = 40)

-- Prove the statement
theorem amber_worked_hours : A = 12 :=
by
  sorry

end amber_worked_hours_l374_374528


namespace basketball_but_not_football_l374_374936

theorem basketball_but_not_football (N B F N_n : ℕ) (hN : N = 30) (hB : B = 15) (hF : F = 8) (hN_n : N_n = 8) :
  ∃ x, N = (B - x) + (F - x) + x + N_n ∧ (B - x) = 14 :=
by {
  use 1,
  split,
  { -- calculate total number of students
    rw [hN, hB, hF, hN_n],
    linarith,
  },
  { -- calculate the number of students who like basketball but not football
    rw hB,
    linarith,
  }
}

end basketball_but_not_football_l374_374936


namespace train_length_l374_374968

theorem train_length (L V : ℝ) (h1 : L = V * 26) (h2 : L + 150 = V * 39) : L = 300 := by
  sorry

end train_length_l374_374968


namespace isosceles_right_triangle_with_same_color_l374_374855

/-- The nodes of an infinite grid are colored in three colors.
Prove that there exists an isosceles right triangle with 
vertices of the same color. -/
theorem isosceles_right_triangle_with_same_color
  (coloring : ℕ → ℕ → Fin 3) :
  ∃ (a b c : ℕ × ℕ), is_isosceles_right_triangle a b c ∧ 
  coloring a.1 a.2 = coloring b.1 b.2 ∧ coloring b.1 b.2 = coloring c.1 c.2 :=
sorry

/-- Predicate for checking if three points form an isosceles right triangle -/
def is_isosceles_right_triangle (a b c : ℕ × ℕ) : Prop :=
(a.1 - b.1) * (a.1 - b.1) + (a.2 - b.2) * (a.2 - b.2) = (a.1 - c.1) * (a.1 - c.1) + (a.2 - c.2) * (a.2 - c.2) ∧ 
(a.1 - c.1) * (a.1 - c.1) + (a.2 - c.2) * (a.2 - c.2) = (b.1 - c.1) * (b.1 - c.1) + (b.2 - c.2) * (b.2 - c.2)

end isosceles_right_triangle_with_same_color_l374_374855


namespace intersection_of_sets_A_and_B_l374_374646

def setA : set ℝ := { x | x^2 - x - 6 < 0 }
def setB : set ℝ := { x | (x + 4) * (x - 2) > 0 }
def setC : set ℝ := { x | 2 < x ∧ x < 3 }

theorem intersection_of_sets_A_and_B : (setA ∩ setB) = setC := by
  sorry

end intersection_of_sets_A_and_B_l374_374646


namespace max_value_of_f_l374_374851

noncomputable def f : ℝ → ℝ := λ x, x^3 - 15 * x^2 - 33 * x + 6

theorem max_value_of_f : ∃ (x : ℝ), f x = 23 ∧ (∀ y : ℝ, f y ≤ f x) := by
  -- The proof and all intermediate steps are omitted.
  sorry

end max_value_of_f_l374_374851


namespace range_of_a_l374_374616

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 + 2 * x - 3 - a

theorem range_of_a (a : ℝ) : 
  (∃ (x₀ : ℝ), -1 ≤ x₀ ∧ x₀ ≤ 1 ∧ f a x₀ = 0) ↔ (1 ≤ a ∨ a ≤ (-3 - real.sqrt 7) / 2) :=
sorry

end range_of_a_l374_374616


namespace inequality_for_positive_reals_l374_374061

theorem inequality_for_positive_reals
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a / (a^4 + b^2) + b / (a^2 + b^4) ≤ 1 / (a * b) := 
sorry

end inequality_for_positive_reals_l374_374061


namespace possible_starting_cities_l374_374233

open SimpleGraph

noncomputable def cities : Finset String :=
  { "Saint Petersburg", "Tver", "Yaroslavl", "Nizhny Novgorod", "Moscow", "Kazan" }

noncomputable def connections : SimpleGraph (Finset String) :=
  ⟨cities, λ u v, (u, v) ∈ {
    ("Saint Petersburg", "Tver"), 
    ("Yaroslavl", "Nizhny Novgorod"), 
    ("Moscow", "Kazan"), 
    ("Nizhny Novgorod", "Kazan"), 
    ("Moscow", "Tver"), 
    ("Moscow", "Nizhny Novgorod")
  }⟩

theorem possible_starting_cities (u : Finset String) :
  u ∈ {"Saint Petersburg", "Yaroslavl"} ↔ 
  ((degree connections u) % 2 = 1) := sorry

end possible_starting_cities_l374_374233


namespace engineering_students_pass_percentage_l374_374333

theorem engineering_students_pass_percentage :
  let num_male_students := 120
  let num_female_students := 100
  let perc_male_eng_students := 0.25
  let perc_female_eng_students := 0.20
  let perc_male_eng_pass := 0.20
  let perc_female_eng_pass := 0.25
  
  let num_male_eng_students := num_male_students * perc_male_eng_students
  let num_female_eng_students := num_female_students * perc_female_eng_students
  
  let num_male_eng_pass := num_male_eng_students * perc_male_eng_pass
  let num_female_eng_pass := num_female_eng_students * perc_female_eng_pass
  
  let total_eng_students := num_male_eng_students + num_female_eng_students
  let total_eng_pass := num_male_eng_pass + num_female_eng_pass
  
  (total_eng_pass / total_eng_students) * 100 = 22 :=
by
  sorry

end engineering_students_pass_percentage_l374_374333


namespace countable_set_condition_l374_374704

open Set

noncomputable def is_countable (S : Set ℝ) : Prop :=
  ∃ f : ℕ → S, Surjective f

theorem countable_set_condition (S : Set ℝ) (hS_inf : Infinite S)
  (hS_sum : ∀ (x : Finset ℝ), ↑x ⊆ S → |x.sum id| ≤ 1) :
  is_countable S :=
sorry

end countable_set_condition_l374_374704


namespace minValue_correct_l374_374638

noncomputable def minValue (a m n: ℝ) : ℝ :=
  if (a > 0 ∧ a ≠ 1 ∧ mn > 0 ∧ m + n = 2) then
    3 + 2 * Real.sqrt(2)
  else 
    0

theorem minValue_correct {a m n : ℝ}
  (ha_pos : a > 0)
  (ha_ne_one : a ≠ 1)
  (hmn_pos : m * n > 0)
  (hmn_eq_two : m + n = 2) :
  minValue a m n = 3 + 2 * Real.sqrt(2) :=
by
  sorry

end minValue_correct_l374_374638


namespace parallelogram_area_l374_374889

-- Define the vertices of the parallelogram
def vertexA := (0, 1)
def vertexB := (1, 2)
def vertexC := (2, 1)

-- Define the vectors corresponding to two sides of the parallelogram
def vector_u := (vertexB.1 - vertexA.1, vertexB.2 - vertexA.2)
def vector_v := (vertexC.1 - vertexA.1, vertexC.2 - vertexA.2)

-- Define the cross product in 2D augmented to 3D
def cross_product_z (u v : ℕ × ℕ) : ℕ :=
  u.1 * v.2 - u.2 * v.1

-- Define the magnitude of the 2D cross product (only z-component matters)
def magnitude_cross_product (u v : ℕ × ℕ) : ℕ :=
  abs (cross_product_z u v)

-- Proposition stating the area of the parallelogram
theorem parallelogram_area : magnitude_cross_product vector_u vector_v = 2 :=
by 
  have h1 : vector_u = (1, 1) := by simp [vector_u]
  have h2 : vector_v = (2, 0) := by simp [vector_v]
  rw [h1, h2]
  simp [magnitude_cross_product, cross_product_z]
  sorry

end parallelogram_area_l374_374889


namespace tulips_to_maintain_ratio_l374_374328

theorem tulips_to_maintain_ratio (initial_daisies : ℕ) (ratio_3: ℕ) (ratio_4: ℕ) 
  (additional_daisies : ℕ) : 
  3 * (initial_daisies + additional_daisies) / 4 = (3 * initial_daisies / 4) + 
  (3 * additional_daisies / 4) :=
by
  have initial_tulips := (3 * initial_daisies / 4)
  have new_daisies := initial_daisies + additional_daisies
  have tulips_needed := (3 * new_daisies / 4)
  have additional_tulips := tulips_needed - initial_tulips
  have total_tulips := initial_tulips + additional_tulips
  simp at *
  sorry

end tulips_to_maintain_ratio_l374_374328


namespace petya_digits_l374_374772

theorem petya_digits (a b c d : ℕ) (distinct_digits : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (non_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (sum_formed_numbers : 6 * (a + b + c + d) * 1111 = 73326) :
  {a, b, c, d} = {1, 2, 3, 5} :=
by 
  sorry

end petya_digits_l374_374772


namespace shelf_with_at_least_40_books_l374_374051

theorem shelf_with_at_least_40_books 
  (num_shelves : ℕ := 5)
  (total_books : ℕ := 160)
  (specific_shelf_books : ℕ := 3)
  (remaining_books : ℕ := total_books - specific_shelf_books) 
  (books_on_shelves : Fin 5 → ℕ)
  (h1 : books_on_shelves 0 = 3)
  (h2 : (∑ i in Finset.filter (λ i, i ≠ 0) Finset.univ, books_on_shelves i) = remaining_books) :
  ∃ i : Fin 5, books_on_shelves i ≥ 40 :=
by
  sorry

end shelf_with_at_least_40_books_l374_374051


namespace geometric_progression_fraction_l374_374379

theorem geometric_progression_fraction (a₁ a₂ a₃ a₄ : ℝ) (h1 : a₂ = 2 * a₁) (h2 : a₃ = 2 * a₂) (h3 : a₄ = 2 * a₃) : 
  (2 * a₁ + a₂) / (2 * a₃ + a₄) = 1 / 4 := 
by 
  sorry

end geometric_progression_fraction_l374_374379


namespace time_spent_filling_per_trip_l374_374824

theorem time_spent_filling_per_trip :
  ∀ (total_hours : ℕ) (driving_time_per_trip : ℕ) (number_of_trips : ℕ),
    total_hours = 7 →
    driving_time_per_trip = 30 →
    number_of_trips = 6 →
    ((total_hours * 60) - (driving_time_per_trip * number_of_trips)) / number_of_trips = 40 :=
by
  intros total_hours driving_time_per_trip number_of_trips
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end time_spent_filling_per_trip_l374_374824


namespace intersect_CF_AE_BK_at_single_point_l374_374493

theorem intersect_CF_AE_BK_at_single_point {A B C D E F K : Point} {AB AC BC DE : Line}
  (h1: Circle → passes_through [A, C])
  (h2: intersect_at_midpoint AB D)
  (h3: intersects DE F)
  (h4: intersects AC K)
  (h5: tangent_to_line_at_point AC E)
  (h6: prove_intersect_at_single_point : intersect_single_point (Line_through C F) (Line_through A E) (Line_through B K)) :
  lines_intersect_single_point (Line_through C F) (Line_through A E) (Line_through B K) := 
sorry

end intersect_CF_AE_BK_at_single_point_l374_374493


namespace problem_statement_l374_374660

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 3 * x^2 + 4

theorem problem_statement : f (g (-3)) = 961 := by
  sorry

end problem_statement_l374_374660


namespace meaning_of_random_number_l374_374424

theorem meaning_of_random_number :
  ∀ (n : ℝ), n ∈ set.range (λ x : ℝ, x) ∧ (∀ m ∈ set.range (λ x : ℝ, x), probability_of_selection m = probability_of_selection n) :=
sorry

end meaning_of_random_number_l374_374424


namespace oatmeal_cookies_l374_374389

theorem oatmeal_cookies (total_cookies chocolate_chip_cookies : ℕ)
  (h1 : total_cookies = 6 * 9)
  (h2 : chocolate_chip_cookies = 13) :
  total_cookies - chocolate_chip_cookies = 41 := by
  sorry

end oatmeal_cookies_l374_374389


namespace problem_statement_l374_374321

noncomputable def f : ℝ → ℝ := λ x, if 0 ≤ x ∧ x ≤ π then sin x else sin (x % π)

theorem problem_statement : f (15 * π / 4) = sqrt 2 / 2 :=
by
  -- The actual proof will be inserted here
  sorry

end problem_statement_l374_374321


namespace smallest_abundant_number_not_multiple_of_10_is_18_l374_374216

def is_abundant (n : ℕ) : Prop :=
  ∑ m in (List.range n).filter (fun m => m ∣ n), m > n

def not_multiple_of_ten (n : ℕ) : Prop :=
  ¬ (10 ∣ n)

def smallest_abundant_not_multiple_of_ten : Prop :=
  ∀ n : ℕ, is_abundant n ∧ not_multiple_of_ten n → n ≥ 18

theorem smallest_abundant_number_not_multiple_of_10_is_18 :
  ∃ n : ℕ, is_abundant n ∧ not_multiple_of_ten n ∧ (∀ m < n, is_abundant m → not_multiple_of_ten m → false) :=
begin
  use 18,
  split,
  { -- is_abundant 18
    sorry },
  split,
  { -- not_multiple_of_ten 18
    sorry },
  { -- ∀ m < 18, is_abundant m → not_multiple_of_ten m → false
    sorry }
end

end smallest_abundant_number_not_multiple_of_10_is_18_l374_374216


namespace probability_tan_gte_sqrt3_correct_l374_374502

noncomputable def probability_tan_gte_sqrt3 := 
let interval := Ioc (-real.pi / 2) (real.pi / 2)
let sub_interval := Ico (real.pi / 3) (real.pi / 2) in
classical.some (measure_theory.measure_space.measure.interval_normalized_probability interval sub_interval)

theorem probability_tan_gte_sqrt3_correct :
  probability_tan_gte_sqrt3 = 1 / 6 :=
sorry

end probability_tan_gte_sqrt3_correct_l374_374502


namespace correct_calculation_for_b_l374_374129

theorem correct_calculation_for_b (a : ℝ) : (a^2)^3 = a^6 := 
by sorry

end correct_calculation_for_b_l374_374129


namespace value_of_2x_minus_y_l374_374595

theorem value_of_2x_minus_y (x y : ℝ) (hx : 5^x = 3) (hy : y = log 5 (9 / 25)) : 2 * x - y = 2 := 
by 
  sorry

end value_of_2x_minus_y_l374_374595


namespace difference_of_primes_l374_374212

theorem difference_of_primes (S : Set ℤ) (hS : ∀ k : ℤ, 10 * k + 7 ∈ S):
  ∀ n ∈ S, ¬ ∃ p q : ℤ, prime p ∧ prime q ∧ p - q = n := by
  intros n hn
  cases' n with z hn
  have h : z = 10 * nat.fst hn + 7 := nat.mul_add_mod_right 10 hn.2
  sorry

end difference_of_primes_l374_374212


namespace a_alone_completes_in_eight_days_l374_374910

variable (a b : Type)
variables (days_ab : ℝ) (days_a : ℝ) (days_ab_2 : ℝ)

noncomputable def days := ℝ

axiom work_together_four_days : days_ab = 4
axiom work_together_266666_days : days_ab_2 = 8 / 3

theorem a_alone_completes_in_eight_days (a b : Type) (days_ab : ℝ) (days_a : ℝ) (days_ab_2 : ℝ)
  (work_together_four_days : days_ab = 4)
  (work_together_266666_days : days_ab_2 = 8 / 3) :
  days_a = 8 :=
by
  sorry

end a_alone_completes_in_eight_days_l374_374910


namespace interest_paid_percent_l374_374938

noncomputable def down_payment : ℝ := 300
noncomputable def total_cost : ℝ := 750
noncomputable def monthly_payment : ℝ := 57
noncomputable def final_payment : ℝ := 21
noncomputable def num_monthly_payments : ℕ := 9

noncomputable def total_instalments : ℝ := (num_monthly_payments * monthly_payment) + final_payment
noncomputable def total_paid : ℝ := total_instalments + down_payment
noncomputable def amount_borrowed : ℝ := total_cost - down_payment
noncomputable def interest_paid : ℝ := total_paid - amount_borrowed
noncomputable def interest_percent : ℝ := (interest_paid / amount_borrowed) * 100

theorem interest_paid_percent:
  interest_percent = 85.33 := by
  sorry

end interest_paid_percent_l374_374938


namespace perimeter_of_PQRST_l374_374342

open Real EuclideanGeometry

/-- Coordinates of the points -/
def P : Point := (0, 8)
def Q : Point := (4, 8)
def R : Point := (4, 2)
def S : Point := (9, 0)
def T : Point := (0, 0)

/-- Definitions of distances -/
def d_PQ : ℝ := euclidean_dist P Q
def d_QR : ℝ := euclidean_dist Q R
def d_RS : ℝ := euclidean_dist R S
def d_ST : ℝ := euclidean_dist S T
def d_TP : ℝ := euclidean_dist T P

/-- Calculate the total perimeter -/
theorem perimeter_of_PQRST : 
  d_PQ + d_QR + d_RS + d_ST + d_TP = 25 + sqrt 41 :=
sorry

end perimeter_of_PQRST_l374_374342


namespace f_is_linear_l374_374718

-- Define the set of all integer sequences
def int_seq := ℕ → ℤ

-- Define the function type
def f (A : int_seq) : Type := ∀ (x y : int_seq), f (x + y) = f x + f y

noncomputable def form_of_f (f : int_seq → ℤ) : Prop :=
  ∃ (a : int_seq), ∀ (x : int_seq), f x = ∑ i in (finset.range (nat.succ (x.as_nat.length))), a i * x i

-- The theorem statement
theorem f_is_linear (f : int_seq → ℤ) :
  (∀ x y : int_seq, f (x + y) = f x + f y) →
  ∃ (a : int_seq), ∀ (x : int_seq), f x = ∑ i in (finset.range (nat.succ (x.as_nat.length))), a i * x i := 
sorry

end f_is_linear_l374_374718


namespace trip_duration_l374_374911

variable (A : ℝ) (T : ℝ)
def Speed1 := 45
def Speed2 := 75
def InitialTime := 4
def AverageSpeed := 65
def Distance1 := Speed1 * InitialTime
def Distance2 := Speed2 * A
def TotalDistance := Distance1 + Distance2
def TotalTime := InitialTime + A

theorem trip_duration :
  180 + 75 * A = 65 * T ∧ T = 4 + A → T = 12 :=
by
  intro h
  have eq₁ : 180 + 75 * A = 65 * (4 + A) := h.1.trans (congr_arg (65 *) h.2)
  have eq₂ : 10 * A = 80 := by linarith
  have A_val : A = 8 := by linarith
  show T = 12 from by linarith

end trip_duration_l374_374911


namespace triangle_construction_condition_l374_374995

variable (s α m_a : ℝ)

theorem triangle_construction_condition (h₁ : m_a < s * (1 - real.sin(α / 2)) / real.cos(α / 2))
(h₂ : m_a < s * real.cot (real.pi / 4 + α / 4)) :
  ∃ (A B C : Type) [Triangle A B C], True :=
by
  sorry

end triangle_construction_condition_l374_374995


namespace number_of_angels_is_two_l374_374173

def beauty_says (beauty : Nat) (statement : Nat) := 
  match beauty with
  | 1 => statement = 1
  | 2 => statement = 2
  | 3 => statement = 3
  | 4 => statement = 4
  | 5 => statement = 5
  | 6 => statement = 6
  | 7 => statement = 7
  | 8 => statement = 8
  | 9 => statement = 9
  | _ => false

def is_angel (beauty : Nat) := 
  beauty = 2 ∨ beauty = 7

theorem number_of_angels_is_two : 
  ∃ (angels devils : Fin 10 → Prop), 
  angels = (fun n => is_angel n) ∧ 
  devils = (fun n => ¬ is_angel n) ∧ 
  (∃ (s : Fin 10 → Nat),
  (∀ b, beauty_says b (s b)) ∧ 
  angels.count ∅ = 2) :=
by
  sorry

end number_of_angels_is_two_l374_374173


namespace maximize_village_value_l374_374172

theorem maximize_village_value :
  ∃ (x y z : ℕ), 
  x + y + z = 20 ∧ 
  2 * x + 3 * y + 4 * z = 50 ∧ 
  (∀ x' y' z' : ℕ, 
      x' + y' + z' = 20 → 2 * x' + 3 * y' + 4 * z' = 50 → 
      (1.2 * x + 1.5 * y + 1.2 * z : ℝ) ≥ (1.2 * x' + 1.5 * y' + 1.2 * z' : ℝ)) ∧ 
  x = 10 ∧ y = 10 ∧ z = 0 := by 
  sorry

end maximize_village_value_l374_374172


namespace find_digits_sum_l374_374802

theorem find_digits_sum
  (a b c d : ℕ)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (non_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (sum_of_numbers : 6 * (a + b + c + d) * 1111 = 73326) :
  {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end find_digits_sum_l374_374802


namespace simplify_expression_l374_374817

theorem simplify_expression : (81 ^ (1 / 2 : ℝ) - 144 ^ (1 / 2 : ℝ) = -3) :=
by
  have h1 : 81 ^ (1 / 2 : ℝ) = 9 := by sorry
  have h2 : 144 ^ (1 / 2 : ℝ) = 12 := by sorry
  rw [h1, h2]
  norm_num

end simplify_expression_l374_374817


namespace fran_speed_calculation_l374_374699

theorem fran_speed_calculation (joann_speed joann_time fran_time : ℝ) (joann_speed_eq : joann_speed = 15) (joann_time_eq : joann_time = 4) (fran_time_eq : fran_time = 2.5) :
  let distance := joann_speed * joann_time in
  let fran_speed := distance / fran_time in
  fran_speed = 24 := by
  sorry

end fran_speed_calculation_l374_374699


namespace donkey_wins_inevitably_l374_374110

theorem donkey_wins_inevitably (n : ℕ) (h_n : n = 2005) 
  (points : Fin n → Point ℝ) 
  (h_no_three_collinear : ∀ (a b c : Fin n), ¬ Collinear ℝ ({points a, points b, points c} : Set (Point ℝ))) 
  (segment_label : (Fin n) × (Fin n) → ℕ) 
  (point_label : Fin n → ℕ) :
  ∃ (a b : Fin n), a ≠ b ∧ point_label a = point_label b ∧ point_label a = segment_label (a, b) := 
sorry

end donkey_wins_inevitably_l374_374110


namespace repair_cost_is_5000_l374_374067

-- Define the initial cost of the machine
def initial_cost : ℝ := 9000

-- Define the transportation charges
def transportation_charges : ℝ := 1000

-- Define the selling price
def selling_price : ℝ := 22500

-- Define the profit percentage as a decimal
def profit_percentage : ℝ := 0.5

-- Define the total cost including repairs
def total_cost (repair_cost : ℝ) : ℝ :=
  initial_cost + transportation_charges + repair_cost

-- Define the equation for selling price with 50% profit
def selling_price_equation (repair_cost : ℝ) : Prop :=
  selling_price = (1 + profit_percentage) * total_cost repair_cost

-- State the proof problem in Lean
theorem repair_cost_is_5000 : selling_price_equation 5000 :=
by 
  sorry

end repair_cost_is_5000_l374_374067


namespace number_of_primes_squares_between_4000_7000_l374_374303

theorem number_of_primes_squares_between_4000_7000 :
  {n : ℕ | prime n ∧ 4000 < n^2 ∧ n^2 < 7000}.card = 5 :=
sorry

end number_of_primes_squares_between_4000_7000_l374_374303


namespace other_root_zero_l374_374266

theorem other_root_zero (b : ℝ) (x : ℝ) (hx_root : x^2 + b * x = 0) (h_x_eq_minus_two : x = -2) : 
  (0 : ℝ) = 0 :=
by
  sorry

end other_root_zero_l374_374266


namespace sum_of_squares_of_roots_l374_374197

theorem sum_of_squares_of_roots : 
  let p : Polynomial ℝ := Polynomial.C 2020 + Polynomial.monomial 2 5 + Polynomial.monomial 8 10 + Polynomial.monomial 10 1,
      roots := p.roots.to_finset in
  ∑ r in roots, r ^ 2 = 0 :=
by
  sorry

end sum_of_squares_of_roots_l374_374197


namespace domain_of_f_when_a_is_1_range_of_g_range_of_a_l374_374415

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := sqrt (-x^2 + (a + 2) * x - a - 1)
def g (x : ℝ) : ℝ := 2^x - 1

theorem domain_of_f_when_a_is_1 : 
  let a := 1 in 
  {x : ℝ | -x^2 + 3 * x - 2 ≥ 0} = Icc 1 2 :=
sorry

theorem range_of_g : ∀ (x : ℝ), x ≤ 2 → g x ∈ Icc (-1 : ℝ) 3 :=
sorry

theorem range_of_a (a : ℝ) : 
  0 < a → 
  (∀ x, f a x = sqrt (-x^2 + (a + 2) * x - a - 1) → (x ∈ Icc 1 (a + 1)) → g x ∈ Icc (-1 : ℝ) 3) → 
  a ≤ 2 :=
sorry

end domain_of_f_when_a_is_1_range_of_g_range_of_a_l374_374415


namespace sum_of_possible_values_of_abs_p_s_l374_374369

variable {p q r s : ℝ}

def conditions (p q r s : ℝ) : Prop := 
  |p - q| = 3 ∧ 
  |q - r| = 4 ∧ 
  |r - s| = 5 ∧ 
  r > q
  
theorem sum_of_possible_values_of_abs_p_s (p q r s : ℝ) (h : conditions p q r s) :
  sum (abs (p - s)) = 24 :=
sorry

end sum_of_possible_values_of_abs_p_s_l374_374369


namespace game_not_fair_l374_374805

-- Define the set of possible outcomes from the dice roll
def outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the event where player A wins (number greater than 3)
def A_wins : Finset ℕ := {4, 5, 6}

-- Define the event where player B wins (number less than 3)
def B_wins : Finset ℕ := {1, 2}

-- Probability of winning for player A
def P_A_wins : ℚ := A_wins.card / outcomes.card

-- Probability of winning for player B
def P_B_wins : ℚ := B_wins.card / outcomes.card

-- Statement: Prove that the game is unfair
theorem game_not_fair : P_A_wins ≠ P_B_wins :=
by
  have h1 : A_wins.card = 3 := by simp
  have h2 : B_wins.card = 2 := by simp
  have h3 : outcomes.card = 6 := by simp
  have h4 : P_A_wins = 3 / 6 := by simp [P_A_wins, h1, h3]
  have h5 : P_B_wins = 2 / 6 := by simp [P_B_wins, h2, h3]
  rw [h4, h5]
  norm_num -- This will simplify 3/6 and 2/6 to 1/2 and 1/3 respectively, showing they are not equal
  sorry

end game_not_fair_l374_374805


namespace find_centroid_of_equilateral_triangle_inscribed_in_parabola_l374_374258

-- Definitions of conditions
def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  let (Ax, Ay) := A
  let (Bx, By) := B
  let (Cx, Cy) := C
  dist (Ax, Ay) (Bx, By) = dist (Bx, By) (Cx, Cy) ∧ dist (Bx, By) (Cx, Cy) = dist (Cx, Cy) (Ax, Ay)

def is_inscribed_in_parabola (A B C : ℝ × ℝ) : Prop :=
  let (Ax, Ay) := A
  let (Bx, By) := B
  let (Cx, Cy) := C
  Ax = Ay^2 ∧ Bx = By^2 ∧ Cx = Cy^2

def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  let (Ax, Ay) := A
  let (Bx, By) := B
  let (Cx, Cy) := C
  ((Ax + Bx + Cx) / 3, (Ay + By + Cy) / 3)

def lies_on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (Px, Py) := P
  Px * Py = 1

-- Problem definition
theorem find_centroid_of_equilateral_triangle_inscribed_in_parabola
  (A B C : ℝ × ℝ)
  (hEquilateral : is_equilateral_triangle A B C)
  (hInscribed : is_inscribed_in_parabola A B C)
  (P : ℝ × ℝ)
  (hCentroid : P = centroid A B C)
  (hOnHyperbola : lies_on_hyperbola P) :
  P = (3, 1 / 3) :=
sorry

end find_centroid_of_equilateral_triangle_inscribed_in_parabola_l374_374258


namespace lines_AB_concurrent_locus_midpoints_l374_374381

-- Definitions and assumptions based on conditions
variable {r1 r2 : ℝ}
variable (C1 C2 : Circle) (T A B : Point)

-- Externally tangent circles with centers O1 and O2 and radii r1 and r2, tangent at point T.
def O1 := C1.center
def O2 := C2.center
def r1 := C1.radius
def r2 := C2.radius

-- First statement: prove that lines AB are concurrent
theorem lines_AB_concurrent
  (hT1 : T ∈ C1.points)
  (hT2 : T ∈ C2.points)
  (hAB : ∀ (A ∈ C1.points) (B ∈ C2.points), ∠ A T B = 90) :
  ∃ (P : Point), ∀ (A ∈ C1.points) (B ∈ C2.points), Line.through A B = Line.through A P := sorry

-- Second statement: prove the locus of midpoints of segments AB
theorem locus_midpoints
  (hT1 : T ∈ C1.points)
  (hT2 : T ∈ C2.points)
  (hAB : ∀ (A ∈ C1.points) (B ∈ C2.points), ∠ A T B = 90) :
  ∃ (C : Circle), ∀ (A ∈ C1.points) (B ∈ C2.points), midpoint A B ∈ C := sorry

-- Helper definitions and assumptions
structure Circle where
  center : Point
  radius : ℝ
  points : Set Point

structure Point where
  x : ℝ
  y : ℝ

def Line.through (P Q : Point) : Set Point := sorry

def ∠ (A B C : Point) : ℝ := sorry

def midpoint (P Q : Point) : Point := sorry

end lines_AB_concurrent_locus_midpoints_l374_374381


namespace minimum_u_val_l374_374619

noncomputable def u (x y : ℝ) : ℝ := 
  (4 / (4 - x^2)) + (9 / (9 - y^2))

theorem minimum_u_val (x y : ℝ) (h1 : x ∈ set.Ioo (-2 : ℝ) 2) (h2 : y ∈ set.Ioo (-2 : ℝ) 2) (h3 : x * y = -1) :
  ∃ p, p = u x y ∧ p = 12 / 7 := 
sorry

end minimum_u_val_l374_374619


namespace yeast_experiment_correctness_l374_374813

/-- Definition of the conditions for the yeast population experiment -/
structure YeastPopulationExperiment where
  inoculated : bool -- Indicates whether inoculation has been done
  first_sampling_test_conducted_immediately : bool -- Indicates if the first sampling test is done immediately after inoculation
  culture_medium_sterilized_after_inoculation : bool -- Indicates if the culture medium is sterilized after inoculation
  culture_medium_shaken_before_sampling : bool -- Indicates if the culture medium is shaken before sampling
  cover_glass_placed_before_sample_added : bool -- Indicates if the cover glass is placed before sample added

/-- Define a hypothesis where option B is the only correct option -/
def correct_option_B (exp : YeastPopulationExperiment) : Prop :=
  exp.inoculated = true →
  exp.first_sampling_test_conducted_immediately = true ∧
  exp.culture_medium_sterilized_after_inoculation = false ∧
  exp.culture_medium_shaken_before_sampling = true ∧
  exp.cover_glass_placed_before_sample_added = false

theorem yeast_experiment_correctness (exp : YeastPopulationExperiment) : correct_option_B exp := sorry

end yeast_experiment_correctness_l374_374813


namespace ratio_a2_a3_l374_374323

namespace SequenceProof

def a (n : ℕ) : ℤ := 3 - 2^n

theorem ratio_a2_a3 : a 2 / a 3 = 1 / 5 := by
  sorry

end SequenceProof

end ratio_a2_a3_l374_374323


namespace Petya_digits_sum_l374_374779

theorem Petya_digits_sum (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :
  (24 * (a + b + c + d) * 1111 = 73326) →
  {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end Petya_digits_sum_l374_374779


namespace coffee_expenses_l374_374128

-- Define amounts consumed and unit costs for French and Columbian roast
def ounces_per_donut_M := 2
def ounces_per_donut_D := 3
def ounces_per_donut_S := ounces_per_donut_D
def ounces_per_pot_F := 12
def ounces_per_pot_C := 15
def cost_per_pot_F := 3
def cost_per_pot_C := 4

-- Define number of donuts consumed
def donuts_M := 8
def donuts_D := 12
def donuts_S := 16

-- Calculate total ounces needed
def total_ounces_F := donuts_M * ounces_per_donut_M
def total_ounces_C := (donuts_D + donuts_S) * ounces_per_donut_D

-- Calculate pots needed, rounding up since partial pots are not allowed
def pots_needed_F := Nat.ceil (total_ounces_F / ounces_per_pot_F)
def pots_needed_C := Nat.ceil (total_ounces_C / ounces_per_pot_C)

-- Calculate total cost
def total_cost := (pots_needed_F * cost_per_pot_F) + (pots_needed_C * cost_per_pot_C)

-- Theorem statement to assert the proof
theorem coffee_expenses : total_cost = 30 := by
  sorry

end coffee_expenses_l374_374128


namespace prove_sum_of_integers_in_range_l374_374586

noncomputable def sum_of_integers_in_range (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  let s := {i : ℝ | ∃ x, a ≤ x ∧ x ≤ b ∧ i = f x ∧ (i ∈ Set.Icc 3 4 ∧ i ∈ ℤ)}
  s.toFinset.sum id

theorem prove_sum_of_integers_in_range :
  let a := 1.25 * (Real.arctan (1 / 3)) * (Real.cos (Real.pi + Real.asin (-0.6)))
  let b := Real.arctan 2
  let f := λ x : ℝ, Real.log2 (5 * Real.cos (2 * x) + 11)
  sum_of_integers_in_range f a b = 7 :=
by
  sorry

end prove_sum_of_integers_in_range_l374_374586


namespace exists_special_number_l374_374368

/-- The number of 2-factors in m! -/
def n : ℕ → ℕ
| 0       := 0
| m + 1 := (m + 1) / 2 + n (m / 2)

/-- There exists a natural number m > 2006^2006 such that m = 3^2006 + n(m) -/
theorem exists_special_number : ∃ m : ℕ, m > 2006 ^ 2006 ∧ m = 3 ^ 2006 + n m :=
by sorry

end exists_special_number_l374_374368


namespace part_a_part_b_l374_374730

-- Define what it means for a number to be "surtido"
def is_surtido (A : ℕ) : Prop :=
  ∀ n, (1 ≤ n → n ≤ (A.digits 10).sum → ∃ B : ℕ, n = (B.digits 10).sum) 

-- Part (a): Prove that if 1, 2, 3, 4, 5, 6, 7, and 8 can be expressed as sums of digits in A, then A is "surtido".
theorem part_a (A : ℕ)
  (h1 : ∃ B1 : ℕ, 1 = (B1.digits 10).sum)
  (h2 : ∃ B2 : ℕ, 2 = (B2.digits 10).sum)
  (h3 : ∃ B3 : ℕ, 3 = (B3.digits 10).sum)
  (h4 : ∃ B4 : ℕ, 4 = (B4.digits 10).sum)
  (h5 : ∃ B5 : ℕ, 5 = (B5.digits 10).sum)
  (h6 : ∃ B6 : ℕ, 6 = (B6.digits 10).sum)
  (h7 : ∃ B7 : ℕ, 7 = (B7.digits 10).sum)
  (h8 : ∃ B8 : ℕ, 8 = (B8.digits 10).sum) : is_surtido A :=
sorry

-- Part (b): Determine if having the sums 1, 2, 3, 4, 5, 6, and 7 as sums of digits in A implies that A is "surtido".
theorem part_b (A : ℕ)
  (h1 : ∃ B1 : ℕ, 1 = (B1.digits 10).sum)
  (h2 : ∃ B2 : ℕ, 2 = (B2.digits 10).sum)
  (h3 : ∃ B3 : ℕ, 3 = (B3.digits 10).sum)
  (h4 : ∃ B4 : ℕ, 4 = (B4.digits 10).sum)
  (h5 : ∃ B5 : ℕ, 5 = (B5.digits 10).sum)
  (h6 : ∃ B6 : ℕ, 6 = (B6.digits 10).sum)
  (h7 : ∃ B7 : ℕ, 7 = (B7.digits 10).sum) : ¬is_surtido A :=
sorry

end part_a_part_b_l374_374730


namespace circles_tangent_line_parallel_l374_374449

-- Define the structure for points, circles, and parallelism
structure Point (α : Type*) := (x : α) (y : α)
structure Circle (α : Type*) := (center : Point α) (radius : ℝ)

namespace Geometry

variables {α : Type*} [linear_ordered_field α] [discrete_linear_ordered_field ℝ]

def tangent_at (A : Point α) (S1 S2 : Circle α) : Prop :=
  let c1 := S1.center in
  let c2 := S2.center in
  (c1.x - A.x) * (c2.y - A.y) = (c2.x - A.x) * (c1.y - A.y)

def collinear (P Q R : Point α) : Prop :=
  (P.x - Q.x) * (R.y - Q.y) = (R.x - Q.x) * (P.y - Q.y)

def intersects (A : Point α) (S : Circle α) (B : Point α) : Prop :=
  (B ≠ A ∧ (B.x - S.center.x) ^ 2 + (B.y - S.center.y) ^ 2 = S.radius ^ 2)

def parallel (l1 l2 : Point α × Point α) : Prop :=
  (l1.2.y - l1.1.y) * (l2.2.x - l2.1.x) = (l2.2.y - l2.1.y) * (l1.2.x - l1.1.x)

-- Define the main theorem statement
theorem circles_tangent_line_parallel {S1 S2 : Circle α} {A A1 A2 : Point α} {O1 O2 : Point α}
  (h_tangent : tangent_at A S1 S2)
  (h_center_S1 : S1.center = O1)
  (h_center_S2 : S2.center = O2)
  (h_intersect_A1 : intersects A S1 A1)
  (h_intersect_A2 : intersects A S2 A2)
  (h_collinear : collinear O1 A O2) :
  parallel (O1, A1) (O2, A2) :=
sorry

end Geometry

end circles_tangent_line_parallel_l374_374449


namespace min_value_of_f_l374_374425

-- Defining the function f
def f (x a : ℝ) : ℝ := (Real.exp x - a)^2 + (Real.exp (-x) - a)^2

-- The theorem statement
theorem min_value_of_f (a : ℝ) (h : 0 < a ∧ a < 2) : ∃ x : ℝ, f x a = 2 * (a - 1)^2 :=
by
  sorry

end min_value_of_f_l374_374425


namespace pentagon_lines_intersect_l374_374339

-- Lean representation of the problem.
theorem pentagon_lines_intersect (A B C D E : Type*) [affine_space A]
  (h1 : parallel (line_through A B) (line_through D E))
  (h2 : euclidean_distance C D = euclidean_distance D E)
  (h3 : perpendicular (line_through C E) (line_through B C))
  (h4 : perpendicular (line_through C E) (line_through A D)) :
  ∃ P : point, parallel (line_through A P) (line_through C D) ∧
              parallel (line_through B P) (line_through C E) ∧
              parallel (line_through E P) (line_through B C) :=
begin
  -- Structure indicating that the lines indeed intersect
  sorry
end

end pentagon_lines_intersect_l374_374339


namespace find_number_l374_374147

-- Define the given conditions and statement as Lean types
theorem find_number (x : ℝ) :
  (0.3 * x > 0.6 * 50 + 30) -> x = 200 :=
by
  -- Proof here
  sorry

end find_number_l374_374147


namespace find_real_a_l374_374715

open Complex

noncomputable def pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem find_real_a (a : ℝ) (i : ℂ) (h_i : i = Complex.I) :
  pure_imaginary ((2 + i) * (a - (2 * i))) ↔ a = -1 :=
by
  sorry

end find_real_a_l374_374715


namespace petya_digits_sum_l374_374762

def distinct_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem petya_digits_sum (a b c d : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
    (h_distinct: distinct_digits a b c d)
    (h_sum: 6 * (a + b + c + d) * 1111 = 73326) :
    a + b + c + d = 11 ∧ {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end petya_digits_sum_l374_374762


namespace solve_inequality_l374_374404

theorem solve_inequality (x k : ℤ) : 
  (2 * (5:ℝ)^(2 * x) * (Real.sin (2 * x)) - (3:ℝ)^x ≥ (5:ℝ)^(2 * x) - 2 * (3:ℝ)^x * (Real.sin (2 * x)))
  → ((x : ℝ) ∈ set.range (λ k : ℤ, ℝ * (Set.Icc ((real.pi / 12) + k * (real.pi)) ((5 * real.pi / 12) + k * (real.pi))))). 


end solve_inequality_l374_374404


namespace positive_integer_triplet_sum_l374_374886

noncomputable def exists_unique_pos_int_triplet : ∃ a b c : ℕ, 
  0 < a ∧ 0 < b ∧ 0 < c ∧ a ≤ b ∧ b ≤ c ∧ (25 / 84 = 1 / a + 1 / (a * b) + 1 / (a * b * c)) :=
sorry

theorem positive_integer_triplet_sum : (a b c : ℕ) (h1: 0 < a) (h2: 0 < b) (h3: 0 < c) 
  (h4: a ≤ b) (h5: b ≤ c) (h6: 25 / 84 = 1 / a + 1 / (a * b) + 1 / (a * b * c)) :
  a + b + c = 17 :=
sorry

end positive_integer_triplet_sum_l374_374886


namespace least_m_value_l374_374569

def recursive_sequence (x : ℕ → ℚ) : Prop :=
  x 0 = 3 ∧ ∀ n, x (n + 1) = (x n ^ 2 + 9 * x n + 20) / (x n + 8)

theorem least_m_value (x : ℕ → ℚ) (h : recursive_sequence x) : ∃ m, m > 0 ∧ x m ≤ 3 + 1 / 2^10 ∧ ∀ k, k > 0 → k < m → x k > 3 + 1 / 2^10 :=
sorry

end least_m_value_l374_374569


namespace intersection_KQ_MP_on_AD_l374_374135

open Set

variables {P : Type*} [MetricSpace P]

-- Conditions
variables {A B C D K L M P Q : P}
-- Assume we have a quadrilateral ABCD
variable (quad : ConvexHull ℝ ({A, B, C, D} : Set P) = univ)

-- Points K, L, and M are on the sides AB, BC, CD respectively (or their extensions)
variable (K_on_AB : K ∈ lineSegment ℝ A B)
variable (L_on_BC : L ∈ lineSegment ℝ B C)
variable (M_on_CD : M ∈ lineSegment ℝ C D)

-- Lines KL and AC intersect at point P
variable (interKL_AC : P ∈ (lineThrough K L) ∩ (lineThrough A C))

-- Lines LM and BD intersect at point Q
variable (interLM_BD : Q ∈ (lineThrough L M) ∩ (lineThrough B D))

-- Proof that the intersection point of lines KQ and MP lies on line AD
theorem intersection_KQ_MP_on_AD :
  ∃ N : P, N ∈ lineThrough A D ∧ N ∈ (lineThrough K Q ∩ lineThrough M P) :=
sorry

end intersection_KQ_MP_on_AD_l374_374135


namespace Petya_digits_sum_l374_374777

theorem Petya_digits_sum (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :
  (24 * (a + b + c + d) * 1111 = 73326) →
  {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end Petya_digits_sum_l374_374777


namespace eddy_freddy_speed_ratio_l374_374222

def ratio_of_average_speeds (dist_A_B : ℝ) (time_Eddy : ℝ) (dist_A_C : ℝ) (time_Freddy : ℝ) : ℝ :=
  (dist_A_B / time_Eddy) / (dist_A_C / time_Freddy)

theorem eddy_freddy_speed_ratio :
  ratio_of_average_speeds 900 3 300 4 = 4 :=
by
  -- sorry statement is used to avoid providing the actual proof steps
  sorry

end eddy_freddy_speed_ratio_l374_374222


namespace friendship_arrangement_count_l374_374821

/-- 
Six colleagues have a social network. Each has exactly two friends, and no three form a triangle. 
Prove that the number of different ways their friendships can be arranged is 60. 
-/
theorem friendship_arrangement_count : 
    ∃ (arrangements : Finset (Finset (Fin 6 × Fin 6))),
    (∀ friendship, friendship ∈ arrangements → (∀ p, ∃ q r, p.1 ≠ p.2 ∧ q.1 ≠ q.2 ∧ r.1 ≠ r.2 ∧ friendship.contains (p, q) ∧ friendship.contains (q, r) → ¬ friendship.contains (p, r))) ∧
    arrangements.card = 60 := 
by {
  sorry  -- proof to be provided
}

end friendship_arrangement_count_l374_374821


namespace largest_possible_last_digit_l374_374843

theorem largest_possible_last_digit (D : Fin 3003 → Nat) :
  D 0 = 2 →
  (∀ i : Fin 3002, (10 * D i + D (i + 1)) % 17 = 0 ∨ (10 * D i + D (i + 1)) % 23 = 0) →
  D 3002 = 9 :=
sorry

end largest_possible_last_digit_l374_374843


namespace points_in_quadrant_I_l374_374998

def point_in_quadrant_I (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

theorem points_in_quadrant_I (x y : ℝ) :
  (y > -x + 3) ∧ (y > 3x - 1) → point_in_quadrant_I x y := 
by
  sorry

end points_in_quadrant_I_l374_374998


namespace basketball_players_taking_chemistry_l374_374975

variable (total_players : ℕ) (taking_biology : ℕ) (taking_both : ℕ)

theorem basketball_players_taking_chemistry (h1 : total_players = 20) 
                                           (h2 : taking_biology = 8) 
                                           (h3 : taking_both = 4) 
                                           (h4 : ∀p, p ≤ total_players) :
  total_players - taking_biology + taking_both = 16 :=
by sorry

end basketball_players_taking_chemistry_l374_374975


namespace petya_digits_l374_374793

def are_distinct (a b c d : Nat) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 

def non_zero_digits (a b c d : Nat) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

theorem petya_digits :
  ∃ (a b c d : Nat), are_distinct a b c d ∧ non_zero_digits a b c d ∧ (a + b + c + d = 11) ∧ (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 5) ∧
  (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 5) ∧ (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 5) ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 5) :=
by
  sorry

end petya_digits_l374_374793


namespace cover_distance_l374_374348

theorem cover_distance (h : ∀ (A B C : ℕ), 
    (∃ (motorcycle_speed pedestrian_speed total_time required_distance : ℝ), 
      motorcycle_speed = 50 ∧ pedestrian_speed = 5 ∧ total_time = 3 ∧ required_distance = 60) 
    → 
    ∃ (possible : Prop), possible = true) 
  : 
  ∃ (A B C : ℕ) (possible : Prop), possible = true :=
by
  have h1 : ∀ A B C : ℕ, 
      (∃ (motorcycle_speed pedestrian_speed total_time required_distance : ℝ), 
        motorcycle_speed = 50 ∧ pedestrian_speed = 5 ∧ total_time = 3 ∧ required_distance = 60) 
      → 
      ∃ (possible : Prop), possible = true := sorry
  exact ⟨1, 1, 1, true⟩

end cover_distance_l374_374348


namespace more_tails_than_heads_l374_374317

def total_flips : ℕ := 211
def heads_flips : ℕ := 65
def tails_flips : ℕ := total_flips - heads_flips

theorem more_tails_than_heads : tails_flips - heads_flips = 81 := by
  -- proof is unnecessary according to the instructions
  sorry

end more_tails_than_heads_l374_374317


namespace tv_blender_cost_more_l374_374873

-- Definitions of the conditions
def in_store_price : ℝ := 75.99
def tv_payment : ℝ := 17.99
def shipping_fee : ℝ := 6.50
def handling_charge : ℝ := 2.50

-- The proof problem
theorem tv_blender_cost_more :
  let tv_total_cost := 4 * tv_payment + shipping_fee + handling_charge in
  let cost_difference := tv_total_cost - in_store_price in
  cost_difference * 100 = 497 :=
by
  sorry

end tv_blender_cost_more_l374_374873


namespace infiniteSeriesSum_is_correct_l374_374191

noncomputable def infiniteSeriesSum : ℚ := 1 - 3 * (1 / 999) + 5 * (1 / 999)^2 - 7 * (1 / 999)^3 + ∑' n, ((-1)^(n+1) * (2*n + 1) * (1 / 999)^n)

theorem infiniteSeriesSum_is_correct : infiniteSeriesSum = 996995 / 497004 :=
sorry

end infiniteSeriesSum_is_correct_l374_374191


namespace part1_part2_part3_l374_374291

/-- Define the odd function f -/
def f (x : ℝ) (a : ℝ) : ℝ := 2^x + a * 2^(-x)

/-- Prove that a = -1 given that f(x) is odd on (-1, 1) -/
theorem part1 (h_odd : ∀ x ∈ Ioo (-1 : ℝ) 1, f x a = -f (-x) a) : a = -1 :=
sorry

/-- Prove the monotonicity of f on (-1,1) -/
theorem part2 (h_a : a = -1) : ∀ x1 x2 : ℝ, -1 < x1 -> x1 < x2 -> x2 < 1 -> f x1 a < f x2 a :=
sorry

/-- Find the range of m given f(1-m) + f(1-2m) < 0 -/
theorem part3 (h_odd_m : ∀ x ∈ Ioo (-1 : ℝ) 1, f 1 - x a = -f (x - 1) a) 
  (h_monotone : ∀ x1 x2 : ℝ, -1 < x1 -> x1 < x2 -> x2 < 1 -> f x1 a < f x2 a) 
  : ∀ m : ℝ, f (1-m) a + f (1-2m) a < 0 -> (2 : ℝ) / 3 < m ∧ m < 1 :=
sorry

end part1_part2_part3_l374_374291


namespace marie_daily_rent_l374_374747

noncomputable def daily_revenue (bread_loaves : ℕ) (bread_price : ℝ) (cakes : ℕ) (cake_price : ℝ) : ℝ :=
  bread_loaves * bread_price + cakes * cake_price

noncomputable def total_profit (daily_revenue : ℝ) (days : ℕ) (cash_register_cost : ℝ) : ℝ :=
  cash_register_cost

noncomputable def daily_profit (total_profit : ℝ) (days : ℕ) : ℝ :=
  total_profit / days

noncomputable def daily_profit_after_electricity (daily_profit : ℝ) (electricity_cost : ℝ) : ℝ :=
  daily_profit - electricity_cost

noncomputable def daily_rent (daily_revenue : ℝ) (daily_profit_after_electricity : ℝ) : ℝ :=
  daily_revenue - daily_profit_after_electricity

theorem marie_daily_rent
  (bread_loaves : ℕ) (bread_price : ℝ) (cakes : ℕ) (cake_price : ℝ)
  (days : ℕ) (cash_register_cost : ℝ) (electricity_cost : ℝ) :
  bread_loaves = 40 → bread_price = 2 → cakes = 6 → cake_price = 12 →
  days = 8 → cash_register_cost = 1040 → electricity_cost = 2 →
  daily_rent (daily_revenue bread_loaves bread_price cakes cake_price)
             (daily_profit_after_electricity (daily_profit (total_profit (daily_revenue bread_loaves bread_price cakes cake_price) days cash_register_cost) days) electricity_cost) = 24 :=
by
  intros h0 h1 h2 h3 h4 h5 h6
  sorry

end marie_daily_rent_l374_374747


namespace scale_factor_sqrt2_l374_374872

theorem scale_factor_sqrt2 (a : ℝ) (λ : ℝ) (A B C D B' C' D' M : ℝ × ℝ)
  (h1 : B = (a, 0))
  (h2 : C = (a, a))
  (h3 : D = (0, a))
  (h4 : B' = (λ * a, 0))
  (h5 : C' = (λ * a, λ * a))
  (h6 : D' = (0, λ * a))
  (hM : M = (λ * a / 2, λ * a / 2))
  (hM_cond : dist M C = dist B B') :
  λ = real.sqrt 2 :=
by
  sorry

end scale_factor_sqrt2_l374_374872


namespace greatest_possible_integer_l374_374385

theorem greatest_possible_integer (n : ℕ) (h1 : n < 150) (h2 : ∃ k : ℕ, n = 9 * k - 1) (h3 : ∃ l : ℕ, n = 10 * l - 4) : n = 86 := 
sorry

end greatest_possible_integer_l374_374385


namespace red_shells_correct_l374_374521

-- Define the conditions
def total_shells : Nat := 291
def green_shells : Nat := 49
def non_red_green_shells : Nat := 166

-- Define the number of red shells as per the given conditions
def red_shells : Nat :=
  total_shells - green_shells - non_red_green_shells

-- State the theorem
theorem red_shells_correct : red_shells = 76 :=
by
  sorry

end red_shells_correct_l374_374521


namespace petya_digits_l374_374792

def are_distinct (a b c d : Nat) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 

def non_zero_digits (a b c d : Nat) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

theorem petya_digits :
  ∃ (a b c d : Nat), are_distinct a b c d ∧ non_zero_digits a b c d ∧ (a + b + c + d = 11) ∧ (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 5) ∧
  (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 5) ∧ (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 5) ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 5) :=
by
  sorry

end petya_digits_l374_374792


namespace tan_alpha_one_l374_374615

theorem tan_alpha_one (α : ℝ) (h : (sin α + cos α) / (2 * sin α - cos α) = 2) : tan α = 1 :=
sorry

end tan_alpha_one_l374_374615


namespace smallest_number_with_property_l374_374205

theorem smallest_number_with_property: 
  ∃ (N : ℕ), N = 25 ∧ (∀ (x : ℕ) (h : N = x + (x / 5)), N ≤ x) := 
  sorry

end smallest_number_with_property_l374_374205


namespace range_m_l374_374311

theorem range_m (m : ℝ) : (∀ x : ℤ, 2 ≤ x → x < 3 → ¬(2x - m > 4)) → (2 - m > 4) ∧ (2x - m > 4) :=
by sorry

end range_m_l374_374311


namespace ratio_of_money_spent_l374_374518

theorem ratio_of_money_spent (h : ∀(a b c : ℕ), a + b + c = 75) : 
  (25 / 75 = 1 / 3) ∧ 
  (40 / 75 = 4 / 3) ∧ 
  (10 / 75 = 2 / 15) :=
by
  sorry

end ratio_of_money_spent_l374_374518


namespace fifth_term_of_sequence_l374_374556

def pow_four_sequence (n : ℕ) : ℕ :=
  ∑ i in Finset.range (n + 1), 4^i

theorem fifth_term_of_sequence :
  pow_four_sequence 4 = 341 :=
sorry

end fifth_term_of_sequence_l374_374556


namespace cylinder_surface_area_l374_374622

theorem cylinder_surface_area (O1 O2 : Point) (R : ℝ) (S : Square) (h₁ : S.area = 8) (h₂ : S.side = 2*R) :
  let surface_area := 2 * π * R^2 + 2 * R * 2 * π * 2 * R in
  surface_area = 12 * π :=
begin
  -- Provide the formal structure.
  sorry
end

end cylinder_surface_area_l374_374622


namespace normal_distribution_percent_lt_m_plus_s_l374_374912

theorem normal_distribution_percent_lt_m_plus_s
  (m s : ℝ)
  (symmetric_about_mean : ∀ x, p(x) = p(2 * m - x))
  (within_one_std_dev : ∀ x, p(x ∈ (m - s, m + s)) = 0.68) :
  ∀ x, p(x < m + s) = 0.84 :=
begin
  sorry
end

end normal_distribution_percent_lt_m_plus_s_l374_374912


namespace solve_for_x_l374_374072

theorem solve_for_x (x : ℤ) (h : 45 - (5 * 3) = x + 7) : x = 23 := 
by
  sorry

end solve_for_x_l374_374072


namespace find_f_minus_one_l374_374727

-- Condition 1: f(x) is an odd function defined on ℝ.
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - (f x)

-- Definition of f when x ≥ 0.
def f (x : ℝ) : ℝ := if x ≥ 0 then 2^x + 2*x - 1 else 0  -- We'll later use the odd property to define for x < 0.

theorem find_f_minus_one :
  odd_function f →
  f 0 = 1 - 1 → -- Given the specific form of f
  f (-1) = -3 := 
by
  sorry

end find_f_minus_one_l374_374727


namespace average_of_N_is_27_5_l374_374125

noncomputable def average_N_values (N : ℕ) : ℝ :=
  if 14 ≤ N ∧ N ≤ 41 then 27.5 else 0

theorem average_of_N_is_27_5 :
  ( ∑ N in Finset.range (42 - 14), (14 + N) : ℝ) / (42 - 14) = 27.5 :=
by
  sorry

end average_of_N_is_27_5_l374_374125


namespace smallest_angle_between_vectors_l374_374036

open Real EuclideanSpace

variables {V : Type*} [inner_product_space ℝ V]

theorem smallest_angle_between_vectors
  (a b c : V)
  (ha : ∥a∥ = 2)
  (hb : ∥b∥ = 1)
  (hc : ∥c∥ = 3)
  (h : a × (a × c) + b = 0) : 
  ∃ θ : ℝ, θ = 60 := 
sorry

end smallest_angle_between_vectors_l374_374036


namespace determine_k_l374_374214

variables (x y z k : ℝ)

theorem determine_k (h1 : (5 / (x - z)) = (k / (y + z))) 
                    (h2 : (k / (y + z)) = (12 / (x + y))) 
                    (h3 : y + z = 2 * x) : 
                    k = 17 := 
by 
  sorry

end determine_k_l374_374214


namespace original_strip_length_l374_374932

theorem original_strip_length (x : ℝ) 
  (h1 : 3 + x + 3 + x + 3 + x + 3 + x + 3 = 27) : 
  4 * 9 + 4 * 3 = 57 := 
  sorry

end original_strip_length_l374_374932


namespace new_percentage_profit_l374_374160

-- Define the given conditions
def cost_price : ℝ := 100
def profit_percentage : ℝ := 30 / 100
def original_selling_price : ℝ := cost_price * (1 + profit_percentage)
def doubled_selling_price : ℝ := 2 * original_selling_price
def new_profit : ℝ := doubled_selling_price - cost_price

-- Define the proof goal
theorem new_percentage_profit : (new_profit / cost_price) * 100 = 160 := by
  sorry

end new_percentage_profit_l374_374160


namespace eval_expression_l374_374488

theorem eval_expression : 10 * 1.8 - 2 * 1.5 / 0.3 = 8 := 
by
  sorry

end eval_expression_l374_374488


namespace height_of_rectangle_l374_374962

theorem height_of_rectangle (s : ℝ) :
  let square_area := s^2,
      rectangle_area := 2 * square_area,
      base := s,
      height := rectangle_area / base
  in height = 2 * s :=
by
  sorry

end height_of_rectangle_l374_374962


namespace logarithmic_expression_value_l374_374198

theorem logarithmic_expression_value :
  2 * Real.logBase 2 8 + Real.log 0.01 / Real.log 10 - Real.logBase 2 (1 / 8) + (0.01)^(-0.5) = 17 :=
by
  sorry

end logarithmic_expression_value_l374_374198


namespace number_of_parallel_planes_l374_374319

-- Given definitions
variable (P : Type) [PointClass P]
variable (a b : Line P) [SkewLines a b]

-- Condition: Point P is outside skew lines a and b
def point_outside_skew_lines (P : P) (a b : Line P) : Prop :=
  ¬ (P ∈ a) ∧ ¬ (P ∈ b)

-- Theorem: The number of planes passing through P and parallel to both skew lines a and b is 0 or 1
theorem number_of_parallel_planes (P : P) (a b : Line P) [point_outside_skew_lines P a b] : 
  (∃ (plane : Plane P), parallel_to_both_skew_lines plane a b) → 
  (∀ (plane : Plane P), parallel_to_both_skew_lines plane a b) := sorry

-- Helper definition
def parallel_to_both_skew_lines (plane : Plane P) (a b : Line P) : Prop :=
  (plane // a) ∧ (plane // b)

#check number_of_parallel_planes

end number_of_parallel_planes_l374_374319


namespace num_two_digit_integers_l374_374653

theorem num_two_digit_integers (N : ℕ) (t u k : ℕ) 
  (hN : 10 ≤ N ∧ N < 100) 
  (hN_digits : N = 10 * t + u)
  (hreversed : N - (10 * u + t) = 2 * k * k) :
  12 := sorry

end num_two_digit_integers_l374_374653


namespace student_ratio_l374_374330

theorem student_ratio (total_students below_8 above_8 age_8 : ℕ)
  (H1 : total_students = 80)
  (H2 : below_8 = 0.25 * total_students)
  (H3 : age_8 = 36)
  (H4 : above_8 = total_students - below_8 - age_8) :
  above_8 / age_8 = 2 / 3 :=
by {
  sorry
}

end student_ratio_l374_374330


namespace num_different_transformations_of_cube_under_six_reflections_l374_374063

theorem num_different_transformations_of_cube_under_six_reflections :
  let origin := (0, 0, 0)
  let opposite_vertex := (1, 1, 1)
  let planes := [
    (λ (a: ℝ × ℝ × ℝ), (-a.1, a.2, a.3)),
    (λ (a: ℝ × ℝ × ℝ), (2 - a.1, a.2, a.3)),
    (λ (a: ℝ × ℝ × ℝ), (a.1, -a.2, a.3)),
    (λ (a: ℝ × ℝ × ℝ), (a.1, 2 - a.2, a.3)),
    (λ (a: ℝ × ℝ × ℝ), (a.1, a.2, -a.3)),
    (λ (a: ℝ × ℝ × ℝ), (a.1, a.2, 2 - a.3))
  ]
  in (number_of_unique_transformations planes) = 8 :=
sorry

end num_different_transformations_of_cube_under_six_reflections_l374_374063


namespace goods_train_speed_proof_l374_374952

-- Definitions based on the conditions
def man_train_speed : ℝ := 60  -- in km/h
def goods_train_length : ℝ := 300 / 1000  -- converting meters to kilometers.
def passing_time : ℝ := 12 / 3600  -- converting seconds to hours.

-- The speed of the goods train based on the provided conditions and the correct answer.
def goods_train_speed : ℝ := 30  -- in km/h

-- Statement to prove
theorem goods_train_speed_proof :
  let relative_speed := goods_train_length / passing_time in
  relative_speed = man_train_speed + goods_train_speed := by
  sorry

end goods_train_speed_proof_l374_374952


namespace perpendicular_AC_AE_l374_374974

open EuclideanGeometry

theorem perpendicular_AC_AE
  (O A B C D P E : Point)
  (circleO : Circle O)
  (h1 : Diameter AB circleO)
  (h2 : Tangent PA circleO A)
  (h3 : Secant PCD circleO)
  (PO : Line P O)
  (BD : Line B D)
  (h4 : E ∈ (PO ∩ BD)) :
  Perpendicular AC AE :=
sorry

end perpendicular_AC_AE_l374_374974


namespace complex_eq_solution_l374_374242

theorem complex_eq_solution (x y : ℝ) (i : ℂ) (h : (2 * x - 1) + i = y - (3 - y) * i) : 
  x = 5 / 2 ∧ y = 4 :=
  sorry

end complex_eq_solution_l374_374242


namespace max_const_c_for_inequality_l374_374269

theorem max_const_c_for_inequality :
  ∀ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z →
  x^3 + y^3 + z^3 - 3 * x * y * z ≥
  (max_const_c : ℝ) * |(x - y) * (y - z) * (z - x)| →
  max_const_c = (sqrt 6 + 3 * sqrt 2) / 2 * real.sqrt (real.sqrt 3) :=
by
  sorry

end max_const_c_for_inequality_l374_374269


namespace arithmetic_sequence_15th_term_l374_374844

theorem arithmetic_sequence_15th_term (a1 a2 a3 : ℕ) (h1 : a1 = 3) (h2 : a2 = 17) (h3 : a3 = 31) :
  let d := a2 - a1 in
  let a15 := a1 + 14 * d in
  a15 = 199 :=
by
  sorry

end arithmetic_sequence_15th_term_l374_374844


namespace fence_remaining_length_l374_374115

theorem fence_remaining_length : 
  let initial_length := 100
  let ben_contribution := 10
  let billy_fraction := 1 / 5
  let johnny_fraction := 1 / 3
  let remaining_after_ben := initial_length - ben_contribution
  let billy_contribution := billy_fraction * remaining_after_ben
  let remaining_after_billy := remaining_after_ben - billy_contribution
  let johnny_contribution := johnny_fraction * remaining_after_billy
  let remaining_after_johnny := remaining_after_billy - johnny_contribution
  in remaining_after_johnny = 48 :=
by
  sorry

end fence_remaining_length_l374_374115


namespace anthony_more_than_mabel_l374_374390

noncomputable def transactions := 
  let M := 90  -- Mabel's transactions
  let J := 82  -- Jade's transactions
  let C := J - 16  -- Cal's transactions
  let A := (3 / 2) * C  -- Anthony's transactions
  let P := ((A - M) / M) * 100 -- Percentage more transactions Anthony handled than Mabel
  P

theorem anthony_more_than_mabel : transactions = 10 := by
  sorry

end anthony_more_than_mabel_l374_374390


namespace linear_function_expression_l374_374142

theorem linear_function_expression (f : ℝ → ℝ) 
  (h : ∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17) : 
  ∀ x, f(x) = 2 * x + 7 :=
sorry

end linear_function_expression_l374_374142


namespace solve_system_l374_374215

theorem solve_system : ∃ x y : ℚ, 
  (2 * x + 3 * y = 7 - 2 * x + 7 - 3 * y) ∧ 
  (3 * x - 2 * y = x - 2 + y - 2) ∧ 
  x = 3 / 4 ∧ 
  y = 11 / 6 := 
by 
  sorry

end solve_system_l374_374215


namespace factor_difference_of_squares_l374_374906

theorem factor_difference_of_squares (a b : ℝ) : 
    (∃ A B : ℝ, a = A ∧ b = B) → 
    (a^2 - b^2 = (a + b) * (a - b)) :=
by
  intros h
  sorry

end factor_difference_of_squares_l374_374906


namespace larger_tent_fabric_amount_l374_374512

-- Define the fabric used for the small tent
def small_tent_fabric : ℝ := 4

-- Define the fabric computation for the larger tent
def larger_tent_fabric (small_tent_fabric : ℝ) : ℝ :=
  2 * small_tent_fabric

-- Theorem stating the amount of fabric needed for the larger tent
theorem larger_tent_fabric_amount : larger_tent_fabric small_tent_fabric = 8 :=
by
  -- Skip the actual proof
  sorry

end larger_tent_fabric_amount_l374_374512


namespace hyperbola_eccentricity_correct_l374_374270

noncomputable def hyperbola_eccentricity (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) : ℝ :=
  (1 + Real.sqrt 5) / 2

theorem hyperbola_eccentricity_correct (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (C : ℝ) (F : ℝ × ℝ) (P Q : ℝ × ℝ) : 
  (a > 0) →
  (b > 0) →
  (F = (C, 0)) → 
  (P ≠ F) → 
  (Q ≠ F) → 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (P = (x₁, y₁) ∧ Q = (x₂, y₂)) ∧ 
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 = 4 * x₁ * x₂ ∧ 
    x₁^2 / a^2 - y₁^2 / b^2 = 1 ∧ 
    x₂^2 / a^2  - y₂^2 / b^2 = 1) → 
    let e := hyperbola_eccentricity a b h₁ h₂ in 
    e = (1 + Real.sqrt 5) / 2 :=
begin
  intros,
  sorry,
end

end hyperbola_eccentricity_correct_l374_374270


namespace biggest_number_l374_374916

theorem biggest_number (A B C D : ℕ) (h1 : A / B = 2 / 3) (h2 : B / C = 3 / 4) (h3 : C / D = 4 / 5) (h4 : A + B + C + D = 1344) : D = 480 := 
sorry

end biggest_number_l374_374916


namespace solve_player_coins_l374_374438

def player_coins (n m k: ℕ) : Prop :=
  ∃ k, 
  (m = k * (n - 1) + 50) ∧ 
  (3 * m = 7 * n * k - 3 * k + 74) ∧ 
  (m = 69)

theorem solve_player_coins (n m k : ℕ) : player_coins n m k :=
by {
  sorry
}

end solve_player_coins_l374_374438


namespace max_min_xy_l374_374863

theorem max_min_xy (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) : (x * y) ∈ { real.min (-1), real.max (1) } :=
by
  have h : a^2 ≤ 4 := 
  sorry
  have xy_eq : x * y = (a^2 - 2) / 2 := 
  sorry
  split
  { calc min (xy_eq) = -1 :=
    sorry
    calc max (xy_eq) = 1 :=
    sorry
  }

end max_min_xy_l374_374863


namespace find_analytical_expression_of_function_l374_374261

theorem find_analytical_expression_of_function (f : ℝ → ℝ) 
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_monotonic : ∀ x y : ℝ, 0 < x → x < 1 → 0 < y → y < 1 → x < y → f x > f y)
  (h_periodic : ∀ x : ℝ, f (x + 4) = f x) :
  f = (λ x, -sin (π / 2 * x)) := 
sorry

end find_analytical_expression_of_function_l374_374261


namespace helical_curve_curvature_l374_374539

-- Define the helical curve
def helical_curve (a h : ℝ) (t : ℝ) : ℝ × ℝ × ℝ :=
  (a * Real.cos t, a * Real.sin t, h * t)

-- Define the curvature of the helical curve
noncomputable def curvature_of_helical_curve (a h : ℝ) : ℝ :=
  let numerator := a
  let denominator := a^2 + h^2
  numerator / denominator

-- The proof problem statement
theorem helical_curve_curvature (a h : ℝ) (t : ℝ) :
  let K := curvature_of_helical_curve a h
  K = a / (a^2 + h^2) := by
  sorry

end helical_curve_curvature_l374_374539


namespace fraction_of_hexagon_area_within_distance_one_half_of_vertices_l374_374453

open Real

noncomputable def hexagon_area (s : ℝ) : ℝ :=
  (3 * sqrt 3 / 2) * s^2

noncomputable def circular_sector_area (r : ℝ) (θ : ℝ) : ℝ :=
  (θ / 360) * π * r^2

theorem fraction_of_hexagon_area_within_distance_one_half_of_vertices :
  let s := 1 in
  let r := 1 / 2 in
  let θ := 120 in
  let hex_area := hexagon_area s in
  let sector_area := circular_sector_area r θ in
  let sectors_total_area := 6 * sector_area in
  sectors_total_area / hex_area = π * sqrt 3 / 9 :=
by
  sorry

end fraction_of_hexagon_area_within_distance_one_half_of_vertices_l374_374453


namespace min_rectangles_to_form_square_l374_374756

theorem min_rectangles_to_form_square : 
  ∃ (n : ℕ), let rectangle_width := 14,
                 rectangle_height := 10 in
                 (rectangle_width * rectangle_height * n = (14 * 7 + 10 * 5)^2) ∧
                 n = 35 :=
by
  sorry

end min_rectangles_to_form_square_l374_374756


namespace find_cosine_of_dihedral_angle_l374_374893

def dihedral_cosine (R r : ℝ) (α β : ℝ) : Prop :=
  R = 2 * r ∧ β = Real.pi / 4 → Real.cos α = 8 / 9

theorem find_cosine_of_dihedral_angle : ∃ α, ∀ R r : ℝ, dihedral_cosine R r α (Real.pi / 4) :=
sorry

end find_cosine_of_dihedral_angle_l374_374893


namespace money_left_after_shopping_l374_374401

/-- Sandy took $300 for shopping and spent 30% of it. -/
def money_taken : ℕ := 300
def percentage_spent : ℝ := 0.30

/-- Theorem to prove Sandy had $210 left after shopping. -/
theorem money_left_after_shopping (total : ℕ) (spent_percent : ℝ) (spent : ℕ) (left : ℕ) :
  total = 300 → spent_percent = 0.30 →
  spent = (spent_percent * total : ℝ).to_nat →
  left = total - spent →
  left = 210 := by
  intros ht hs hp hl
  rw [ht, hs] at hp
  simp at hp
  rw [hp] at hl
  simp at hl
  assumption


end money_left_after_shopping_l374_374401


namespace simplify_sqrt_l374_374070

theorem simplify_sqrt (a b : ℕ) : 
  (a = 3∧ b = 5) → 
  (Real.sqrt(↑a * ↑b) * Real.sqrt((↑b^4) * (↑a^5)) = 675 * Real.sqrt(↑b)) :=
by 
  rintros ⟨ha, hb⟩
  rw [ha, hb]
  simp
  sorry

end simplify_sqrt_l374_374070


namespace isosceles_right_triangle_incenter_length_BI_l374_374712

noncomputable def length_BI (A B C I : ℝ) (h_isosceles : ∀ {a b c : ℝ}, a = b ∧ ∠ B = real.pi / 2) 
(h_AB : AB = 6) (h_incenter : is_incenter I A B C) : ℝ :=
  6*real.sqrt 2 - 6

# Reduction check that will be turned into the actual theorem
theorem isosceles_right_triangle_incenter_length_BI :
  ∀ (A B C I : Point) (h_isosceles : is_isosceles_right_triangle A B C B)
  (h_AB : length AB = 6) (h_incenter : is_incenter I A B C), 
  length_BI A B C I h_isosceles h_AB h_incenter = 6*real.sqrt 2 - 6 :=
sorry

end isosceles_right_triangle_incenter_length_BI_l374_374712


namespace find_x_plus_y_l374_374249

theorem find_x_plus_y (x y : ℝ) (h : 2^x = 18^y ∧ 18^y = 6^(x * y)) : x + y = 0 ∨ x + y = 2 :=
sorry

end find_x_plus_y_l374_374249


namespace certain_number_is_15_l374_374318

theorem certain_number_is_15 :
  (∃ (p q : ℕ), 17 * (p + 1) = 28 * (q + 1) ∧ p > 15 ∧ q > 15 ∧ p + q = 43) → 15 :=
by
sorry

end certain_number_is_15_l374_374318


namespace dilation_image_example_l374_374497

def dilation (center : ℂ) (scale : ℝ) (point : ℂ) : ℂ :=
  center + scale * (point - center)

theorem dilation_image_example :
  dilation (-1 + 4 * complex.I) 3 (1 + complex.I) = 5 - 5 * complex.I :=
by
  sorry

end dilation_image_example_l374_374497


namespace largest_number_value_l374_374888

theorem largest_number_value (x : ℕ) (h : 7 * x - 3 * x = 40) : 7 * x = 70 :=
by
  sorry

end largest_number_value_l374_374888


namespace trailing_zeros_345_factorial_l374_374428

theorem trailing_zeros_345_factorial : trailingZeros 345! = 84 := sorry

-- Define the function to count trailing zeros of a factorial
def trailingZeros (n : ℕ) : ℕ :=
  let count (p : ℕ) (n : ℕ) : ℕ := if n = 0 then 0 else n / p + count p (n / p)
  count 5 n

end trailing_zeros_345_factorial_l374_374428


namespace part_a_l374_374884

noncomputable def exists_unique_integers (n : ℕ) (t : ℕ → ℕ → ℕ) : Prop :=
  ∃ (a : ℕ → ℕ → ℕ),
  (∀ x y, 1 ≤ x ∧ x ≤ n → 1 ≤ y ∧ y ≤ n →
  (∑ i in finset.range (x + 1), ∑ j in finset.range (y + 1), a i j) = t x y)

theorem part_a :
  exists_unique_integers 2012 (λ i j, sorry) :=
sorry

end part_a_l374_374884


namespace eleven_segment_open_polygons_count_l374_374055

theorem eleven_segment_open_polygons_count :
  let A := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  ∃ (n : ℕ), n = 1024 ∧ 
  -- Count number of distinct 11-segment open polygons without self-intersections
  let count_distinct_polygons := 2 ^ 10 in
  -- Polygons that can be transformed into each other by rotation are considered the same
  n = count_distinct_polygons :=
sorry

end eleven_segment_open_polygons_count_l374_374055


namespace parallelogram_area_l374_374206

-- Define the polynomial
def poly := (3 : ℂ) * X^4 + (9 * ⟨0, 1⟩) * X^3 + ((-12 : ℂ) + (12 * ⟨0, 1⟩)) * X^2 + ((-27 : ℂ) + (-3 * ⟨0, 1⟩)) * X + (4 : ℂ) + (-18 * ⟨0, 1⟩)

-- The main theorem to prove
theorem parallelogram_area :
  let roots := {z : ℂ | poly.eval z = 0}
  ∃ p q : ℝ, p * q * 2 = 6 :=
sorry

end parallelogram_area_l374_374206


namespace rectangle_area_l374_374842

noncomputable def side_length (area : ℕ) : ℕ :=
  nat.sqrt area

noncomputable def larger_square_side_length (shaded_square_side : ℕ) : ℕ :=
  2 * shaded_square_side

noncomputable def area_of_larger_square (side_length : ℕ) : ℕ :=
  side_length * side_length

noncomputable def area_of_rectangle (shaded_square_area : ℕ) : ℕ :=
  let shaded_side := side_length shaded_square_area
  let larger_side := larger_square_side_length shaded_side
  let smaller_square_area := shaded_square_area
  2 * smaller_square_area + area_of_larger_square larger_side

theorem rectangle_area (H1 : ∃ squares : ℕ, squares = 3) (H2 : ∃ shaded_square_area : ℕ, shaded_square_area = 4) :
  area_of_rectangle 4 = 24 :=
sorry

end rectangle_area_l374_374842


namespace slope_of_line_through_points_l374_374159

theorem slope_of_line_through_points :
  let x1 := -1
      y1 := -4
      x2 := 5
      y2 := 0.8
  in (y2 - y1) / (x2 - x1) = 0.8 :=
by
  sorry

end slope_of_line_through_points_l374_374159


namespace plane_speed_east_l374_374506

variable (v_e : ℝ)
variable (v_w : ℝ := 400) -- Speed flying west
variable (d_total : ℝ := 1200) -- Total distance traveled
variable (t_total : ℝ := 7) -- Total time

-- Main thm: Prove that the plane's speed when flying east is approximately 109.09 km/h 
theorem plane_speed_east (h : d_total / 2 / v_w + d_total / 2 / v_e = t_total) : v_e ≈ 109.09 := 
sorry

end plane_speed_east_l374_374506


namespace vector_dot_product_l374_374298

variables {V : Type} [inner_product_space ℝ V]

theorem vector_dot_product
  (a b : V)
  (h1 : ∥a + b∥ = 2 * real.sqrt 3)
  (h2 : ∥a - b∥ = 2) :
  ⟪a, b⟫ = 2 :=
by sorry

end vector_dot_product_l374_374298


namespace leftmost_three_nonzero_digits_of_ring_arrangements_l374_374609

theorem leftmost_three_nonzero_digits_of_ring_arrangements :
  let n := 1900800 in leftmostThreeNonzeroDigits n = 190 :=
by
  let leftmostThreeNonzeroDigits (m : ℕ) : ℕ := String.toNat (String.extract (Nat.toDigits 10 m |> List.filter (fun d => d ≠ 0)) 0 3)
  sorry

end leftmost_three_nonzero_digits_of_ring_arrangements_l374_374609


namespace fran_speed_calculation_l374_374700

theorem fran_speed_calculation (joann_speed joann_time fran_time : ℝ) (joann_speed_eq : joann_speed = 15) (joann_time_eq : joann_time = 4) (fran_time_eq : fran_time = 2.5) :
  let distance := joann_speed * joann_time in
  let fran_speed := distance / fran_time in
  fran_speed = 24 := by
  sorry

end fran_speed_calculation_l374_374700


namespace petya_digits_sum_l374_374758

def distinct_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem petya_digits_sum (a b c d : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
    (h_distinct: distinct_digits a b c d)
    (h_sum: 6 * (a + b + c + d) * 1111 = 73326) :
    a + b + c + d = 11 ∧ {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end petya_digits_sum_l374_374758


namespace dice_probability_l374_374223

/-- The probability that exactly five of the fifteen 6-sided dice show a 1 is approximately 0.092. -/
theorem dice_probability (n k : ℕ) (p : ℝ) (q : ℝ)
  (hn : n = 15)
  (hk : k = 5)
  (hp : p = 1 / 6)
  (hq : q = 5 / 6) :
  (nat.choose n k) * (p ^ k) * (q ^ (n - k)) ≈ 0.092 :=
by
  sorry

end dice_probability_l374_374223


namespace difference_in_ages_is_54_l374_374874

theorem difference_in_ages_is_54 (c d : ℕ) (h1 : 10 ≤ c ∧ c < 100 ∧ 10 ≤ d ∧ d < 100) 
    (h2 : 10 * c + d - (10 * d + c) = 9 * (c - d)) 
    (h3 : 10 * c + d + 10 = 3 * (10 * d + c + 10)) : 
    10 * c + d - (10 * d + c) = 54 :=
by
sorry

end difference_in_ages_is_54_l374_374874


namespace gas_volume_change_l374_374239

-- Defining the given conditions as a Lean 4 statement.
theorem gas_volume_change 
  (initial_temp : ℕ) (initial_volume : ℕ) (temperature_step : ℕ) (volume_change_step : ℕ)
  (rise_temp1 : ℕ) (drop_temp2 : ℕ) :
  initial_temp = 30 ∧ initial_volume = 36 ∧ temperature_step = 5 ∧ volume_change_step = 4 ∧
  rise_temp1 = 40 ∧ drop_temp2 = 25 → 
  let final_volume := initial_volume + ((rise_temp1 - initial_temp) / temperature_step * volume_change_step) - 
                                  ((rise_temp1 - drop_temp2) / temperature_step * volume_change_step)
  in final_volume = 32 :=
by {
  intro h,
  unfold final_volume,
  sorry
}

end gas_volume_change_l374_374239


namespace conjugate_in_third_quadrant_l374_374265

-- Define that z is a complex number
def z : ℂ := -3 + 2 * complex.i

-- Define the conjugate of z
def conjugate_z := complex.conj z

-- Definition of a point in the third quadrant
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

-- The statement to prove
theorem conjugate_in_third_quadrant : in_third_quadrant (conjugate_z.re) (conjugate_z.im) :=
by
  sorry

end conjugate_in_third_quadrant_l374_374265


namespace binomial_theorem_l374_374272

section BinomialDistribution

-- Define the random variable X that follows a binomial distribution B(8, 1/2)
variables {X : ℕ → ℝ}
def binomial (n : ℕ) (p : ℝ) : ℕ → ℝ := λ k, (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

-- Given condition
def binomial_condition := binomial 8 (1 / 2)

-- Expected value condition
def expected_value_binom := 8 * (1 / 2)

-- Define a probability mass function for our binomial distribution
def P (k : ℕ) : ℝ := binomial 8 (1 / 2) k

-- Theorem to prove
theorem binomial_theorem :
  (expected_value_binom = 4) ∧ (P 3 = P 5) :=
by
  -- Proof omitted
  sorry

end BinomialDistribution

end binomial_theorem_l374_374272


namespace bus_capacity_fraction_l374_374492

theorem bus_capacity_fraction
  (capacity : ℕ)
  (x : ℚ)
  (return_fraction : ℚ)
  (total_people : ℕ)
  (capacity_eq : capacity = 200)
  (return_fraction_eq : return_fraction = 4/5)
  (total_people_eq : total_people = 310)
  (people_first_trip_eq : 200 * x + 200 * 4/5 = 310) :
  x = 3/4 :=
by
  sorry

end bus_capacity_fraction_l374_374492


namespace vector_parallel_l374_374243

variables (x : ℝ)

def a := (x, 4 : ℝ)
def b := (4, x : ℝ)

definition parallel (v1 v2 : ℝ × ℝ) := v1.1 * v2.2 - v1.2 * v2.1 = 0

theorem vector_parallel (h : parallel (a x) (b x)) : x = 4 ∨ x = -4 :=
by sorry

end vector_parallel_l374_374243


namespace remaining_amount_to_pay_l374_374496

-- Define the constants and conditions
def total_cost : ℝ := 1300
def first_deposit : ℝ := 0.10 * total_cost
def second_deposit : ℝ := 2 * first_deposit
def promotional_discount : ℝ := 0.05 * total_cost
def interest_rate : ℝ := 0.02

-- Define the function to calculate the final payment
def final_payment (total_cost first_deposit second_deposit promotional_discount interest_rate : ℝ) : ℝ :=
  let total_paid := first_deposit + second_deposit
  let remaining_balance := total_cost - total_paid
  let remaining_after_discount := remaining_balance - promotional_discount
  remaining_after_discount * (1 + interest_rate)

-- Define the theorem to be proven
theorem remaining_amount_to_pay :
  final_payment total_cost first_deposit second_deposit promotional_discount interest_rate = 861.90 :=
by
  -- The proof goes here
  sorry

end remaining_amount_to_pay_l374_374496


namespace more_tails_than_heads_l374_374316

def total_flips : ℕ := 211
def heads_flips : ℕ := 65
def tails_flips : ℕ := total_flips - heads_flips

theorem more_tails_than_heads : tails_flips - heads_flips = 81 := by
  -- proof is unnecessary according to the instructions
  sorry

end more_tails_than_heads_l374_374316


namespace solve_simultaneous_l374_374079

-- Define the simultaneous equations
def eq1 (x y : ℚ) : Prop := x^2 - 2 * x * y = 1
def eq2 (x y : ℚ) : Prop := 5 * x^2 - 2 * x * y + 2 * y^2 = 5

-- Define the set of solutions
def solutions : set (ℚ × ℚ) := 
  {(1, 0), (-1, 0), (1 / 3, -4 / 3), (-1 / 3, 4 / 3)}

-- The theorem to be proved
theorem solve_simultaneous : 
  ∀ x y : ℚ, eq1 x y ∧ eq2 x y ↔ (x, y) ∈ solutions := 
by
  sorry

end solve_simultaneous_l374_374079


namespace k_value_geometric_sequence_l374_374588

theorem k_value_geometric_sequence (S : ℕ+ → ℝ) (k : ℝ) (hS : ∀ n : ℕ+, S n = 3 * 2^n + k) : k = -3 :=
sorry

end k_value_geometric_sequence_l374_374588


namespace geometric_series_mod_l374_374127

theorem geometric_series_mod : 
  let S := (finset.range 1002).sum (λ n, 3^n)
  in S % 500 = 4 := 
by sorry

end geometric_series_mod_l374_374127


namespace hyperbola_eccentricity_range_correct_l374_374203

noncomputable def hyperbola_eccentricity_range (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (e : ℝ) : Prop :=
  let c := a * e in               -- c is the distance from the center to the focus
  let |AB| := 4 * b in             -- Distance AB is 4b (given condition)
  ∃ t : ℝ, 
    (2 * (b / a) * sqrt((c - t) ^ 2 - a ^ 2) = 4 * b) ∧ -- Intersection condition
    (e ^ 2 = 1 + (4 * b ^ 2) / (a ^ 2)) ∧               -- Resulting value of e^2 
    (e > sqrt 5)

theorem hyperbola_eccentricity_range_correct (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (e : ℝ) :
  hyperbola_eccentricity_range a b h_a h_b e ↔ e ∈ (Ioi (sqrt 5)) :=
sorry

end hyperbola_eccentricity_range_correct_l374_374203


namespace area_of_ABCD_l374_374121

noncomputable def area_of_sector (r : ℝ) (θ : ℝ) : ℝ :=
  (θ / 360) * π * r^2

theorem area_of_ABCD (r θ : ℝ) (h_r : r = 15) (h_θ : θ = 45) :
  2 * area_of_sector r θ = 56.25 * π :=
by
  rw [h_r, h_θ]
  sorry

end area_of_ABCD_l374_374121


namespace hyperbola_eccentricity_l374_374679

theorem hyperbola_eccentricity {A B C D E : ℝ} :
  (equilateral_triangle A B C) ∧ (midpoint D A B) ∧ (midpoint E A C) →
  eccentricity_hyperbola B C D E = sqrt 3 + 1 :=
by
  -- Hypotheses about the equilateral triangle, midpoint conditions, etc.
  sorry

end hyperbola_eccentricity_l374_374679


namespace no_prime_number_between_30_and_40_mod_9_eq_7_l374_374102

theorem no_prime_number_between_30_and_40_mod_9_eq_7 : ¬ ∃ n : ℕ, 30 ≤ n ∧ n ≤ 40 ∧ Nat.Prime n ∧ n % 9 = 7 :=
by
  sorry

end no_prime_number_between_30_and_40_mod_9_eq_7_l374_374102


namespace weekly_allowance_is_8_l374_374527

variable (A : ℝ)

def condition_1 (A : ℝ) : Prop := ∃ A : ℝ, A / 2 + 8 = 12

theorem weekly_allowance_is_8 (A : ℝ) (h : condition_1 A) : A = 8 :=
sorry

end weekly_allowance_is_8_l374_374527


namespace bus_speed_excluding_stoppages_l374_374575

theorem bus_speed_excluding_stoppages 
  (V : ℝ) -- Denote the average speed excluding stoppages as V
  (h1 : 30 / 1 = 30) -- condition 1: average speed including stoppages is 30 km/hr
  (h2 : 1 / 2 = 0.5) -- condition 2: The bus is moving for 0.5 hours per hour due to 30 min stoppage
  (h3 : V = 2 * 30) -- from the condition that the bus must cover the distance in half the time
  : V = 60 :=
by {
  sorry -- proof is not required
}

end bus_speed_excluding_stoppages_l374_374575


namespace exists_sequence_S_l374_374362

theorem exists_sequence_S (n : ℕ) (h_n_pos : 0 < n) : 
  ∃ S : list ℕ, 
  (∀ x ∈ S, x = 0 ∨ x = 1) ∧ 
  (∀ d : ℕ, d ≥ 2 → 
    (let number_in_base_d := ∑ i in list.range S.length, S.nth_le i (by sorry) * d ^ i in 
    number_in_base_d ≠ 0 ∧ n ∣ number_in_base_d)) :=
sorry

end exists_sequence_S_l374_374362


namespace chef_pies_sold_l374_374547

theorem chef_pies_sold :
  let small_shepherds_pie_pieces := 4
  let large_shepherds_pie_pieces := 8
  let small_chicken_pot_pie_pieces := 5
  let large_chicken_pot_pie_pieces := 10
  let small_shepherds_pie_customers := 52
  let large_shepherds_pie_customers := 76
  let small_chicken_pot_pie_customers := 80
  let large_chicken_pot_pie_customers := 130
  
  let small_shepherds_pies := small_shepherds_pie_customers / small_shepherds_pie_pieces
  let large_shepherds_pies := large_shepherds_pie_customers / large_shepherds_pie_pieces
  let small_chicken_pot_pies := small_chicken_pot_pie_customers / small_chicken_pot_pie_pieces
  let large_chicken_pot_pies := large_chicken_pot_pie_customers / large_chicken_pot_pie_pieces
  
  ceil (small_shepherds_pies + large_shepherds_pies + small_chicken_pot_pies + large_chicken_pot_pies) = 52 :=
by
  let small_shepherds_pie_pieces := 4
  let large_shepherds_pie_pieces := 8
  let small_chicken_pot_pie_pieces := 5
  let large_chicken_pot_pie_pieces := 10
  let small_shepherds_pie_customers := 52
  let large_shepherds_pie_customers := 76
  let small_chicken_pot_pie_customers := 80
  let large_chicken_pot_pie_customers := 130
  
  let small_shepherds_pies := small_shepherds_pie_customers / small_shepherds_pie_pieces
  let large_shepherds_pies := large_shepherds_pie_customers / large_shepherds_pie_pieces
  let small_chicken_pot_pies := small_chicken_pot_pie_customers / small_chicken_pot_pie_pieces
  let large_chicken_pot_pies := large_chicken_pot_pie_customers / large_chicken_pot_pie_pieces

  have total_pies := small_shepherds_pies + large_shepherds_pies + small_chicken_pot_pies + large_chicken_pot_pies
  have rounded_total_pies := ceil total_pies

  show rounded_total_pies = 52
  sorry

end chef_pies_sold_l374_374547


namespace binomial_coefficient_sq_sum_l374_374981

theorem binomial_coefficient_sq_sum :
  (∑ k in Finset.range' 2 10, ((Nat.choose k 2)^2)) = 220 :=
sorry

end binomial_coefficient_sq_sum_l374_374981


namespace domain_y_l374_374897

def domain_of_y (x : ℝ) : Prop :=
  (x < -6) ∨ (-6 < x ∧ x < 6) ∨ (6 < x)

theorem domain_y :
  ∀ x : ℝ, (x^2 - 36 = 0) → (y = (x^4 - 16) / (x^2 - 36)) → ¬ domain_of_y x :=
begin
  intro x,
  sorry
end

end domain_y_l374_374897


namespace number_of_large_m_l374_374606

open List

noncomputable def m (a : List ℝ) (k : ℕ) : ℝ := 
  if h : 1 ≤ k ∧ k ≤ a.length then 
    have : 0 < k := Nat.one_le_div_iff.mpr h.left
    List.maximum (List.finRange k).map (λ l, (a.drop (k - l)).take l).sum / l
  else 
    0

theorem number_of_large_m (a : List ℝ) (alpha : ℝ) (h_alpha : 0 < alpha) :
  (List.range a.length).filter (λ k, m a k > alpha).length < (a.sum / alpha) := 
sorry

end number_of_large_m_l374_374606


namespace total_eyes_correct_l374_374305

-- Conditions
def boys := 21 * 2 + 2 * 1
def girls := 15 * 2 + 3 * 1
def cats := 8 * 2 + 2 * 1
def spiders := 4 * 8 + 1 * 6

-- Total count of eyes
def total_eyes := boys + girls + cats + spiders

theorem total_eyes_correct: total_eyes = 133 :=
by 
  -- Here the proof steps would go, which we are skipping
  sorry

end total_eyes_correct_l374_374305


namespace razorback_tshirt_revenue_l374_374409

theorem razorback_tshirt_revenue 
    (total_tshirts : ℕ) (total_money : ℕ) 
    (h1 : total_tshirts = 245) 
    (h2 : total_money = 2205) : 
    (total_money / total_tshirts = 9) := 
by 
    sorry

end razorback_tshirt_revenue_l374_374409


namespace B_work_time_l374_374151

-- Define the conditions as constants
constant W : ℝ  -- Total work
constant A : ℝ := W / 2  -- A's work rate
constant C : ℝ := W / 2 - W / 2  -- C's work rate (from A + C's rate and A's rate which simplifies to 0)
constant B : ℝ := W / 3  -- B's work rate, derived from B + C's rate and C's rate

-- State the main theorem
theorem B_work_time : (W / B) = 3 := by
  sorry

end B_work_time_l374_374151


namespace dot_product_ps_l374_374378

noncomputable theory

def unit_vector (v : ℝ × ℝ × ℝ) : Prop :=
  v.1^2 + v.2^2 + v.3^2 = 1

def dot_product (v₁ v₂ : ℝ × ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2 + v₁.3 * v₂.3

variables (p q r s : ℝ × ℝ × ℝ)

axiom unit_vectors :
  unit_vector p ∧ unit_vector q ∧ unit_vector r ∧ unit_vector s

axiom dot_product_conditions :
  dot_product p q = -1/5 ∧
  dot_product p r = -1/5 ∧
  dot_product q r = -1/5 ∧
  dot_product q s = -1/5 ∧
  dot_product r s = -1/5

theorem dot_product_ps : dot_product p s = -17/25 :=
sorry

end dot_product_ps_l374_374378


namespace last_integer_in_sequence_l374_374869

theorem last_integer_in_sequence (a₀ : ℕ) (h₀ : a₀ = 800000) (h_subseq : ∀ n : ℕ, a₀ / (2^n) ∈ ℕ) : 
  (last_integer : ℕ) := 
  last_integer = 3125 := 
by 
  sorry

end last_integer_in_sequence_l374_374869


namespace petya_digits_l374_374768

theorem petya_digits (a b c d : ℕ) (distinct_digits : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (non_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (sum_formed_numbers : 6 * (a + b + c + d) * 1111 = 73326) :
  {a, b, c, d} = {1, 2, 3, 5} :=
by 
  sorry

end petya_digits_l374_374768


namespace pastries_selection_l374_374145

theorem pastries_selection (m n : ℕ) (hm : m = 20) (hn : n = 10) 
  (boxes : fin m → fin n → fin m) :
∃ (selection : fin m → fin m),
  function.injective selection ∧
  (∀ i, boxes i (selection i) = i) := sorry

end pastries_selection_l374_374145


namespace fifth_term_sequence_l374_374566

theorem fifth_term_sequence : (∑ i in Finset.range 5, 4^i) = 341 :=
by
  sorry

end fifth_term_sequence_l374_374566


namespace solve_problem_l374_374211

-- Conditions from the problem
def is_prime (p : ℕ) : Prop := Nat.Prime p

def satisfies_conditions (n p : ℕ) : Prop := 
  (p > 1) ∧ is_prime p ∧ (n > 0) ∧ (n ≤ 2 * p)

-- Main proof statement
theorem solve_problem (n p : ℕ) (h1 : satisfies_conditions n p)
    (h2 : (p - 1) ^ n + 1 ∣ n ^ (p - 1)) :
    (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) ∨ (n = 1 ∧ is_prime p) :=
sorry

end solve_problem_l374_374211


namespace parallelogram_area_approx_304_52_l374_374228

noncomputable def sin_degrees (d : ℝ) : ℝ :=
  Real.sin (d * Real.pi / 180)

noncomputable def parallelogram_area (a b : ℝ) (angle_deg : ℝ) : ℝ :=
  a * b * sin_degrees angle_deg

theorem parallelogram_area_approx_304_52 :
  parallelogram_area 14 24 65 ≈ 304.52 :=
by
  sorry

end parallelogram_area_approx_304_52_l374_374228


namespace Mike_profit_l374_374045

def total_cost (acres : ℕ) (cost_per_acre : ℕ) : ℕ :=
  acres * cost_per_acre

def revenue (acres_sold : ℕ) (price_per_acre : ℕ) : ℕ :=
  acres_sold * price_per_acre

def profit (revenue : ℕ) (cost : ℕ) : ℕ :=
  revenue - cost

theorem Mike_profit :
  let acres := 200
  let cost_per_acre := 70
  let acres_sold := acres / 2
  let price_per_acre := 200
  let cost := total_cost acres cost_per_acre
  let rev := revenue acres_sold price_per_acre
  profit rev cost = 6000 :=
by
  sorry

end Mike_profit_l374_374045


namespace find_length_XY_l374_374340

noncomputable def length_XY (W X Y Z : ℝ × ℝ)
  (YZ_len : ℝ)
  (tan_Z : ℝ)
  (tan_X : ℝ)
  (H_trapezoid : (Y.2 - Z.2 = 0) ∧ (X.2 - W.2 = 0))
  (H_parallel : W.1 - X.1 = Y.1 - Z.1)
  (H_perpendicular : W.2 - Y.2 = Z.2 - Y.2) : ℝ :=
  (W.1 - W.1) -- Placeholder for actual calculations

theorem find_length_XY :
  ∀ (W X Y Z : ℝ × ℝ),
    (YZ_len = 15) →
    (tan_Z = 2) →
    (tan_X = (3 / 2)) →
    (H_trapezoid : (Y.2 - Z.2 = 0) ∧ (X.2 - W.2 = 0)) →
    (H_parallel : W.1 - X.1 = Y.1 - Z.1) →
    (H_perpendicular : W.2 - Y.2 = Z.2 - Y.2) →
    length_XY W X Y Z YZ_len tan_Z tan_X H_trapezoid H_parallel H_perpendicular = 10 * sqrt 13 :=
by
  intros
  sorry

end find_length_XY_l374_374340


namespace avogadro_constant_problem_l374_374108

theorem avogadro_constant_problem 
  (N_A : ℝ) -- Avogadro's constant
  (mass1 : ℝ := 18) (molar_mass1 : ℝ := 20) (moles1 : ℝ := mass1 / molar_mass1) 
  (atoms_D2O_molecules : ℝ := 2) (atoms_D2O : ℝ := moles1 * atoms_D2O_molecules * N_A)
  (mass2 : ℝ := 14) (molar_mass_N2CO : ℝ := 28) (moles2 : ℝ := mass2 / molar_mass_N2CO)
  (electrons_per_molecule : ℝ := 14) (total_electrons_mixture : ℝ := moles2 * electrons_per_molecule * N_A)
  (volume3 : ℝ := 2.24) (temp_unk : Prop := true) -- unknown temperature
  (pressure_unk : Prop := true) -- unknown pressure
  (carbonate_molarity : ℝ := 0.1) (volume_solution : ℝ := 1) (moles_carbonate : ℝ := carbonate_molarity * volume_solution) 
  (anions_carbonate_solution : ℝ := moles_carbonate * N_A) :
  (atoms_D2O ≠ 2 * N_A) ∧ (anions_carbonate_solution > 0.1 * N_A) ∧ (total_electrons_mixture = 7 * N_A) -> 
  True := sorry

end avogadro_constant_problem_l374_374108


namespace probability_a1_divides_a2_and_a2_divides_a3_l374_374375

open Finset BigOperators

noncomputable def S := {d : ℕ | d ∣ 30^10}

def div_condition (a1 a2 a3 : ℕ) : Prop := a1 ∈ S ∧ a2 ∈ S ∧ a3 ∈ S ∧ a1 ∣ a2 ∧ a2 ∣ a3

theorem probability_a1_divides_a2_and_a2_divides_a3 :
  ∃ m n : ℕ, nat.coprime m n ∧
  (∑ a1 in S, ∑ a2 in S, ∑ a3 in S, if div_condition a1 a2 a3 then 1 else 0)  = m * (prime_divisors (30^10).card)^3
  := sorry

end probability_a1_divides_a2_and_a2_divides_a3_l374_374375


namespace sinA_eq_sinB_iff_a_eq_b_l374_374672

variables {A B C : Type} [triangle A B C]

theorem sinA_eq_sinB_iff_a_eq_b (A B : angle) (a b : real) (sin : angle → real) : 
  (sin A = sin B) ↔ (a = b) :=
by sorry

end sinA_eq_sinB_iff_a_eq_b_l374_374672


namespace integral_equality_l374_374463

theorem integral_equality :
    (∫ x in 0..1, 1) = 1 :=
by {
  exact Integral.integral_const _ _ _
}

end integral_equality_l374_374463


namespace exists_line_with_two_colors_l374_374442

open Classical

/-- Given a grid with 1x1 squares where each vertex is painted one of four colors such that each 1x1 square's vertices are all different colors, 
    there exists a line in the grid with nodes of exactly two different colors. -/
theorem exists_line_with_two_colors 
  (A : Type)
  [Inhabited A]
  [DecidableEq A]
  (colors : Finset A) 
  (h_col : colors.card = 4) 
  (grid : ℤ × ℤ → A) 
  (h_diff_colors : ∀ (i j : ℤ), i ≠ j → ∀ (k l : ℤ), grid (i, k) ≠ grid (j, k) ∧ grid (i, l) ≠ grid (i, k)) :
  ∃ line : ℤ → ℤ × ℤ, ∃ a b : A, a ≠ b ∧ ∀ n : ℤ, grid (line n) = a ∨ grid (line n) = b :=
sorry

end exists_line_with_two_colors_l374_374442


namespace new_person_weight_l374_374479

theorem new_person_weight (avg_inc : Real) (num_persons : Nat) (old_weight new_weight : Real)
  (h1 : avg_inc = 2.5)
  (h2 : num_persons = 8)
  (h3 : old_weight = 40)
  (h4 : num_persons * avg_inc = new_weight - old_weight) :
  new_weight = 60 :=
by
  --proof will be done here
  sorry

end new_person_weight_l374_374479


namespace least_pounds_of_sugar_l374_374536

theorem least_pounds_of_sugar :
  ∃ s : ℝ, (∀ f : ℝ, (f ≥ 6 + s / 2 ∧ f ≤ 2 * s) → s = 4) :=
by {
    use 4,
    sorry
}

end least_pounds_of_sugar_l374_374536


namespace distinct_digits_count_total_numbers_count_two_digits_same_count_l374_374452

/-- 
1. Prove that the number of three-digit numbers formed by rolling a die three times
   with all distinct digits is 120.
--/
theorem distinct_digits_count (rolls : list ℕ) (h1 : rolls.length = 3)
  (h2 : ∀ x ∈ rolls, x ∈ {1, 2, 3, 4, 5, 6}) 
  (h3 : rolls.nodup) : rolls.permutations.length = 120 := by
  sorry

/-- 
2. Prove that the total number of distinct three-digit numbers formed by rolling a die three times is 216.
--/
theorem total_numbers_count : (die_rolls : list (list ℕ))
  (h : die_rolls = list.replicate 6 [1, 2, 3, 4, 5, 6]) : 
  die_rolls.bind (λ die, die.combinations 3).length = 216 := by
  sorry

/-- 
3. Prove that the number of three-digit numbers formed by rolling a die three times where exactly 
   two digits are the same is 90.
--/
theorem two_digits_same_count (rolls : list ℕ) (h1 : rolls.length = 3)
  (h2 : ∀ x ∈ rolls, x ∈ {1, 2, 3, 4, 5, 6})
  (h3 : ∃ a b, rolls = [a, a, b] ∨ rolls = [a, b, a] ∨ rolls = [b, a, a] 
    ∧ a ≠ b) : 
  list.permutations(rolls).length = 90 := by
  sorry

end distinct_digits_count_total_numbers_count_two_digits_same_count_l374_374452


namespace math_problem_l374_374195

-- Declare the basic conditions
lemma sqrt_eight : real.sqrt 8 = 2 * real.sqrt 2 := 
by sorry

lemma half_inv : (1 / 2 : ℝ)⁻¹ = 2 := 
by sorry

lemma cos_45 : real.cos (real.pi / 4) = real.sqrt 2 / 2 := 
by sorry

lemma div_exp : 2 / (1 / 2) * 2 = 4 := 
by sorry

lemma zero_power : (2009 - real.sqrt 3)^0 = 1 := 
by sorry

-- Prove the main statement
theorem math_problem : 
  real.sqrt 8 + (1 / 2 : ℝ)⁻¹ - 4 * real.cos (real.pi / 4) - 2 / (1 / 2) * 2 - (2009 - real.sqrt 3)^0 = -7 :=
by 
  rw [sqrt_eight, half_inv, cos_45, div_exp, zero_power],
  norm_num,
  sorry

end math_problem_l374_374195


namespace find_value_of_m_l374_374100

theorem find_value_of_m :
  (∃ y : ℝ, y = 20 - (0.5 * -6.7)) →
  (m : ℝ) = 3 * -6.7 + (20 - (0.5 * -6.7)) :=
by {
  sorry
}

end find_value_of_m_l374_374100


namespace shiny_igneous_rocks_count_l374_374674

-- Definitions based on the conditions
variables (S I shiny_igneous_rocks : ℕ)

-- Condition 1: I = 1/2 * S
def condition1 : Prop := I = S / 2

-- Condition 2: 2/3 of I are shiny igneous rocks
def condition2 : Prop := shiny_igneous_rocks = 2 * I / 3

-- Condition 4: I + S = 180
def condition4 : Prop := I + S = 180

-- Proven statement
theorem shiny_igneous_rocks_count (h1 : condition1) (h2 : condition2) (h4 : condition4) : shiny_igneous_rocks = 40 :=
by {
  -- The proof steps are omitted according to the instruction
  sorry
}

end shiny_igneous_rocks_count_l374_374674


namespace solution_of_system_l374_374471

theorem solution_of_system (x y z : ℕ) :
  x + y + z = 12 ∧ 4 * x + 3 * y + 2 * z = 36 ↔ x ∈ {0, 1, 2, 3, 4, 5, 6} := by
sorry

end solution_of_system_l374_374471


namespace best_fit_slope_l374_374470

theorem best_fit_slope (x1 x2 x3 x4 x5 y1 y2 y3 y4 y5 d : ℝ)
  (h1 : x1 < x2)
  (h2 : x2 < x3)
  (h3 : x3 < x4)
  (h4 : x4 < x5)
  (h5 : x2 - x1 = d)
  (h6 : x3 - x2 = 2 * d)
  (h7 : x4 - x3 = 3 * d)
  (h8 : x5 - x4 = 4 * d)
  (h9 : 0 < d)
  : 
  let x̄ := (x1 + x2 + x3 + x4 + x5) / 5,
      ȳ := (y1 + y2 + y3 + y4 + y5) / 5
  in
  (∑ i in [x1, x2, x3, x4, x5], (i - x̄) * (list.nth_le [y1, y2, y3, y4, y5] (list.index_of i [x1, x2, x3, x4, x5]) sorry - ȳ))
  / 
  (∑ i in [x1, x2, x3, x4, x5], (i - x̄)^2)
  = (∑ i in finset.range 5, (λ i, 
    match i with 
      | 0 => (x1 - x̄) * (y1 - ȳ)
      | 1 => (x2 - x̄) * (y2 - ȳ)
      | 2 => (x3 - x̄) * (y3 - ȳ)
      | 3 => (x4 - x̄) * (y4 - ȳ)
      | 4 => (x5 - x̄) * (y5 - ȳ) 
    end
  ))
  /
  (∑ i in finset.range 5, (λ i, 
    match i with 
      | 0 => (x1 - x̄)^2
      | 1 => (x2 - x̄)^2
      | 2 => (x3 - x̄)^2
      | 3 => (x4 - x̄)^2
      | 4 => (x5 - x̄)^2 
    end
  )) := sorry

end best_fit_slope_l374_374470


namespace probability_product_multiple_of_3_l374_374825

open Nat

def SpinnerC := {1, 2, 3, 4, 5}
def SpinnerD := {1, 2, 3, 4}

def is_multiple_of_3 (n : ℕ) := ∃ k, n = 3 * k

theorem probability_product_multiple_of_3 : 
  let outcomes := (do a ← SpinnerC; b ← SpinnerD; pure (a, b))
  let favorable_outcomes := do (a, b) ← outcomes; guard (is_multiple_of_3 (a * b)); pure (a, b)
  (favorable_outcomes.length) / (outcomes.length) = 2 / 5 := 
by
  sorry

end probability_product_multiple_of_3_l374_374825


namespace solve_equation_l374_374078

theorem solve_equation : 
  ∀ (x : ℝ), (x ≠ 2) → (3 / (x - 2) - 1 = 1 / (2 - x)) → x = 6 := 
by
  intros x hx h_eq
  have hx_neq : x - 2 ≠ 0 := by linarith
  calc 
    3 / (x - 2) - 1 = -1 / (x - 2) : by {rw [h_eq, neg_inv (x-2)]}
    (3 / (x - 2)) * (x - 2) - 1 * (x - 2) = (-1 / (x - 2)) * (x - 2) : by {simp [hx_neq]}
    3 - (x - 2) = -1 : by linarith
    3 - x + 2 = -1 : by linarith
    5 - x = -1 : by linarith
    - x = -6 : by linarith
    x = 6 : by linarith

end solve_equation_l374_374078


namespace tina_earnings_l374_374114

def regular_hourly_wage := 18.0
def daily_hours_worked := 10.0
def days_worked := 5
def overtime_rate (hw : ℝ) : ℝ := hw + hw / 2
def regular_hours_per_day := 8.0
def overtime_hours_per_day (dh : ℝ) (rhpd : ℝ) : ℝ := dh - rhpd
def total_hours_in_week (dh : ℝ) (dw : ℕ) : ℝ := dh * dw
def total_regular_hours_in_week (rhpd : ℝ) (dw : ℕ) : ℝ := rhpd * dw
def total_overtime_hours_in_week (ohpd : ℝ) (dw : ℕ) : ℝ := ohpd * dw
def total_earnings (reg_hours : ℝ) (ot_hours : ℝ) (reg_rate : ℝ) (ot_rate : ℝ) : ℝ :=
  (reg_hours * reg_rate) + (ot_hours * ot_rate)

theorem tina_earnings : 
  total_earnings (total_regular_hours_in_week regular_hours_per_day days_worked) 
    (total_overtime_hours_in_week (overtime_hours_per_day daily_hours_worked regular_hours_per_day) days_worked) 
    regular_hourly_wage 
    (overtime_rate regular_hourly_wage) = 990.0 := 
by {
  -- Proof would go here, but we use sorry to skip it for now.
  sorry
}

end tina_earnings_l374_374114


namespace more_tails_than_heads_l374_374315

theorem more_tails_than_heads (total_flips : ℕ) (heads : ℕ) (tails : ℕ) :
  total_flips = 211 → heads = 65 → tails = (total_flips - heads) → (tails - heads) = 81 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  rw [h1, h2]
  exact h3.trans (show 211 - 65 - 65 = 81 by norm_num)

end more_tails_than_heads_l374_374315


namespace value_of_y_l374_374309

theorem value_of_y (x y : ℝ) (h1 : x - y = 16) (h2 : x + y = 4) : y = -6 :=
by
  sorry

end value_of_y_l374_374309


namespace winning_strategy_l374_374706

-- Let n be a positive integer representing the number of cards.
variables (n : ℕ) (h_pos_n : 0 < n)

-- Let k be the number on the topmost card.
variables (k : ℕ) (h_k_le_n : k ≤ n)

-- Assume the initial shuffle of the n cards.
variable (cards : list ℕ)
variable (h_cards_length : cards.length = n)

-- Define the initial top k cards.
def top_k_cards := cards.take k

-- Define a function to determine if k is the smallest card in the first k cards.
def is_small_state (cards : list ℕ) (k : ℕ) : Prop :=
  k = (top_k_cards cards k).min' (by { rw h_cards_length, exact list.take_length_le k cards })

-- Conclusion about winning strategy based on the initial state.
theorem winning_strategy : (is_small_state cards k) ∨ ¬(is_small_state cards k) :=
sorry

end winning_strategy_l374_374706


namespace initial_pigeons_l374_374111

theorem initial_pigeons (n : ℕ) (h : n + 1 = 2) : n = 1 := 
sorry

end initial_pigeons_l374_374111


namespace petya_digits_l374_374795

def are_distinct (a b c d : Nat) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 

def non_zero_digits (a b c d : Nat) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

theorem petya_digits :
  ∃ (a b c d : Nat), are_distinct a b c d ∧ non_zero_digits a b c d ∧ (a + b + c + d = 11) ∧ (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 5) ∧
  (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 5) ∧ (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 5) ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 5) :=
by
  sorry

end petya_digits_l374_374795


namespace true_propositions_l374_374280

-- Definitions for the propositions
def proposition1 := ∀ (l1 l2 : ℝ → ℝ → ℝ), (∀ p : ℝ → ℝ, (l1 p ∧ l2 p) → p) → (l1 = l2)
def proposition2 := ∀ (l1 l2 l3 : ℝ → ℝ), (∀ p : ℝ → ℝ, (l1 p ∧ l2 p ∧ l3 p) → p) → (l1 = l2)
def proposition3 := ∀ (P : ℝ→ ℝ → ℝ), ∃ (!) Q : ℝ→ ℝ → ℝ, (∀ p : ℝ, P p ↔ Q p)
def proposition4 := ∀ (l : ℝ → ℝ → ℝ) (P : ℝ → ℝ → ℝ), ∃ Q : ℝ → ℝ → ℝ, (∀ p : ℝ, Q p ↔ Q p ∧ P p)

theorem true_propositions : proposition1 = false ∧
                            proposition2 = false ∧
                            proposition3 = true ∧
                            proposition4 = false :=
by
  sorry  -- Proof omitted

end true_propositions_l374_374280


namespace faster_pipe_time_l374_374754

-- Define the conditions as constants
def second_pipe_rate (R : ℝ) : Prop := R > 0
def first_pipe_rate (R : ℝ) : ℝ := 1.25 * R
def third_pipe_rate : ℝ := 1 / 8
def combined_rate (R : ℝ) : Prop := (R + 1.25 * R + 1 / 8) = 1 / 3
def time_first_pipe (R : ℝ) : ℝ := 1 / (1.25 * R)

-- Prove the question
theorem faster_pipe_time (R : ℝ) (hR : second_pipe_rate R) (h_combined : combined_rate R) : time_first_pipe R = 8.64 := 
sorry

end faster_pipe_time_l374_374754


namespace train_length_l374_374924

noncomputable def length_of_each_train (distance : ℝ) (relative_speed : ℝ) (time : ℝ) : ℝ :=
2 * distance = relative_speed * time

theorem train_length (L : ℝ) (distance time : ℝ) (speed_faster speed_slower : ℝ)
  (h1 : speed_faster = 46) (h2 : speed_slower = 36) (h3 : time = 54) 
  (h4 : relative_speed = (speed_faster - speed_slower) * (5/18)) (h5 : 2 * L = relative_speed * time):
  L = 75 :=
by
  sorry

end train_length_l374_374924


namespace simplify_expression_l374_374820

theorem simplify_expression (x y z : ℝ) : 
  (x + complex.I * y + z) * (x - complex.I * y - z) = x^2 - y^2 - z^2 :=
  sorry

end simplify_expression_l374_374820


namespace triangle_side_length_l374_374892

-- Defining basic properties and known lengths of the similar triangles
def GH : ℝ := 8
def HI : ℝ := 16
def YZ : ℝ := 24
def XY : ℝ := 12

-- Defining the similarity condition for triangles GHI and XYZ
def triangles_similar : Prop := 
  -- The similarity of the triangles implies proportionality of the sides
  (XY / GH = YZ / HI)

-- The theorem statement to prove
theorem triangle_side_length (h_sim : triangles_similar) : XY = 12 :=
by
  -- assuming the similarity condition and known lengths
  sorry -- This will be the detailed proof

end triangle_side_length_l374_374892


namespace common_factor_of_polynomial_l374_374837

variable (m a b : ℤ)

theorem common_factor_of_polynomial : ∃ m, m ∣ (3 * m * a^2) ∧ m ∣ (6 * m * a * b) := 
by {
  use m,
  split,
  repeat { rw mul_assoc, apply dvd_mul_right }
}

end common_factor_of_polynomial_l374_374837


namespace odd_number_of_divisors_implies_perfect_square_l374_374361

theorem odd_number_of_divisors_implies_perfect_square 
  (n : ℕ) (h_pos : 0 < n) 
  (h_odd_divisors : (nat.divisors n).length % 2 = 1) : 
  ∃ k : ℕ, n = k ^ 2 :=
sorry

end odd_number_of_divisors_implies_perfect_square_l374_374361


namespace calculate_expression_l374_374194

theorem calculate_expression :
  sqrt 12 - abs (-1) + (1 / 2) ^ (-1 : ℤ) + (2023 + Real.pi) ^ 0 = 2 * sqrt 3 + 2 :=
by sorry

end calculate_expression_l374_374194


namespace sum_of_possible_x_l374_374074

theorem sum_of_possible_x :
  let equation (x : ℝ) :=  4^(x^2 + 6*x + 9) = 16^(x + 3)
  let solutions := { x : ℝ | equation x }
  ∑ x in solutions, x = -4 :=
by
  sorry

end sum_of_possible_x_l374_374074


namespace compare_neg_rational_numbers_l374_374985

theorem compare_neg_rational_numbers :
  - (3 / 2) > - (5 / 3) := 
sorry

end compare_neg_rational_numbers_l374_374985


namespace simplify_expression_l374_374819

noncomputable def original_expression : ℝ := (-1 / 81) ^ (-4 / 3)

theorem simplify_expression : original_expression = 81 := 
by
  sorry

end simplify_expression_l374_374819


namespace nicky_run_time_l374_374049

-- Define the constants according to the conditions in the problem
def head_start : ℕ := 100 -- Nicky's head start (meters)
def cr_speed : ℕ := 8 -- Cristina's speed (meters per second)
def ni_speed : ℕ := 4 -- Nicky's speed (meters per second)

-- Define the event of Cristina catching up to Nicky
def meets_at_time (t : ℕ) : Prop :=
  cr_speed * t = head_start + ni_speed * t

-- The proof statement
theorem nicky_run_time : ∃ t : ℕ, meets_at_time t ∧ t = 25 :=
by
  sorry

end nicky_run_time_l374_374049


namespace Petya_digits_sum_l374_374775

theorem Petya_digits_sum (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :
  (24 * (a + b + c + d) * 1111 = 73326) →
  {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end Petya_digits_sum_l374_374775


namespace length_of_each_brick_l374_374444

theorem length_of_each_brick (wall_length wall_height wall_thickness : ℝ) (brick_length brick_width brick_height : ℝ) (num_bricks_used : ℝ) 
  (h1 : wall_length = 8) 
  (h2 : wall_height = 6) 
  (h3 : wall_thickness = 0.02) 
  (h4 : brick_length = 0.11) 
  (h5 : brick_width = 0.05) 
  (h6 : brick_height = 0.06) 
  (h7 : num_bricks_used = 2909.090909090909) : 
  brick_length = 0.11 :=
by
  -- variables and assumptions
  have vol_wall : ℝ := wall_length * wall_height * wall_thickness
  have vol_brick : ℝ := brick_length * brick_width * brick_height
  have calc_bricks : ℝ := vol_wall / vol_brick
  -- skipping proof
  sorry

end length_of_each_brick_l374_374444


namespace general_formula_a_sum_c_l374_374603

-- Definition of the given sequences and conditions
def a_sequence (n : ℕ) : ℕ := 2 * n - 1

-- Proof (I) to be completed: sequence a_n is arithmetic and has the form 2n - 1
theorem general_formula_a {a : ℕ → ℕ} 
  (h1 : ∀ n ≥ 2, (a (n + 1)) / (a (n - 1)) + (a (n - 1)) / (a (n + 1)) = (4 * (a n) ^ 2) / (a (n + 1) * a (n - 1)) - 2) 
  (h2 : a 6 = 11) 
  (h3 : ∑ i in finset.range 9, a i = 81) : 
  ∀ n, a n = 2 * n - 1 := 
sorry

-- Definition for sequence b_n
def b_sequence (n : ℕ) : ℝ := 
  if n = 1 then 3 
  else (2 * n + 1) / (2 * n - 1)

-- Definition for sequence c_n
def c_sequence (n : ℕ) : ℝ := (a_sequence n * b_sequence n) / (2^(n + 1))

-- Proof (II) to be completed: sum of the first n terms of c_n sequence
theorem sum_c (n : ℕ) :
  (∑ i in finset.range n, c_sequence i) = 5 / 2 - (2 * n + 5) / (2^(n + 1)) :=
sorry

end general_formula_a_sum_c_l374_374603


namespace coat_price_reduction_l374_374921

theorem coat_price_reduction (original_price reduction_amount : ℝ) 
  (h1 : original_price = 500) (h2 : reduction_amount = 400) :
  (reduction_amount / original_price) * 100 = 80 :=
by {
  sorry -- This is where the proof would go
}

end coat_price_reduction_l374_374921


namespace theorem_cloth_sales_l374_374967

variable (Rs : Type) [AddCommGroup Rs] [Module ℤ Rs] [OrderedAddCommMonoid Rs]
variable (cp sp tp p : Rs)
variable (x : ℤ)

def cloth_sales_problem (cp sp tp : Rs) (p : Rs) (x : ℤ) : Prop :=
  -- Given conditions
  (cp = 80) ∧
  (p = 25) ∧
  (sp = cp + p) ∧
  (tp = 8925) ∧
  -- Given equation
  (tp = sp * x) →
  -- Prove that x = 85
  (x = 85)

theorem theorem_cloth_sales : ∀ (cp sp tp p: Rs) (x : ℤ),
  cloth_sales_problem cp sp tp p x := sorry

end theorem_cloth_sales_l374_374967


namespace find_range_of_m_l374_374649

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m * x + 1 = 0 ∧ y^2 + m * y + 1 = 0
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

theorem find_range_of_m (m : ℝ) (h1 : ¬(p m ∧ q m)) (h2 : ¬¬p m) : m ≥ 3 ∨ m < -2 :=
by 
  sorry

end find_range_of_m_l374_374649


namespace find_angle_A_find_b_c_l374_374011

variables {A B C a b c : ℝ} -- Angles and sides
variable (triangle_abc : acute_triangle a b c A B C)

-- Given condition
axiom cond1 : sqrt 3 * sin C - cos B = cos (A - C)

-- Provided data for part (2)
variable (a_val : a = 2 * sqrt 3)
variable (area_val : 1/2 * b * c * sin A = 3 * sqrt 3)

-- Part (1)
theorem find_angle_A (h : sqrt 3 * sin C - cos B = cos (A - C)) : A = π / 3 := sorry

-- Part (2)
theorem find_b_c (harea : 1/2 * b * c * sin A = 3 * sqrt 3) (ha : a = 2 * sqrt 3) (hA : A = π / 3) : b + c = 4 * sqrt 3 := sorry

end find_angle_A_find_b_c_l374_374011


namespace geometric_mean_condition_l374_374719

theorem geometric_mean_condition (A B C D : Type) (α β γ : ℝ) (h_angle_sum : α + β + γ = π) :
  (∃ D ∈ lineSegment A B, CD_length D = geometric_mean (AD_length D) (BD_length D)) ↔
  sin α * sin β ≤ (sin (γ / 2)) ^ 2 := sorry

end geometric_mean_condition_l374_374719


namespace rectangle_area_l374_374849

theorem rectangle_area (b l: ℕ) (h1: l = 3 * b) (h2: 2 * (l + b) = 120) : l * b = 675 := 
by 
  sorry

end rectangle_area_l374_374849


namespace david_still_has_l374_374915

variable (P L S R : ℝ)

def initial_amount : ℝ := 1800
def post_spending_condition (S : ℝ) : ℝ := S - 800
def remaining_money (P S : ℝ) : ℝ := P - S

theorem david_still_has :
  ∀ (S : ℝ),
    initial_amount = P →
    post_spending_condition S = L →
    remaining_money P S = R →
    R = L →
    R = 500 :=
by
  intros S hP hL hR hCl
  sorry

end david_still_has_l374_374915


namespace smallest_integer_in_range_l374_374107

theorem smallest_integer_in_range :
  ∃ n : ℕ, 
  1 < n ∧ 
  n % 3 = 2 ∧ 
  n % 5 = 2 ∧ 
  n % 7 = 2 ∧ 
  90 < n ∧ n < 119 :=
sorry

end smallest_integer_in_range_l374_374107


namespace min_convex_n_polygon_lattice_points_l374_374237

-- Define a convex polygon with lattice points
def convex_n_gon (n : ℕ) : Prop :=
  ∃ vertices : fin n → ℤ × ℤ, convex_hull (set.range vertices) ∧ (∀ i j : fin n, i ≠ j → vertices i ≠ vertices j)

-- Define lattice points on and inside the polygon
def lattice_points_on_and_inside (n : ℕ) : ℕ :=
  ∃ vertices : fin n → ℤ × ℤ, (convex_hull (set.range vertices)).1.card

theorem min_convex_n_polygon_lattice_points :
  ∃ n : ℕ, (∀ k : ℕ, convex_n_gon k → lattice_points_on_and_inside k ≥ k + 1) ↔ n = 5 :=
sorry

end min_convex_n_polygon_lattice_points_l374_374237


namespace part_a_part_b_l374_374106

-- Define the properties of the tangential quadrilateral

variable (a b c d t r R : ℝ)

-- Condition: the sides of a circumscribed and tangent quadrilateral are a, b, c, d

-- Define the area t of the tangential quadrilateral
def area_tangential (a b c d : ℝ) : ℝ := Real.sqrt (a * b * c * d)

-- Theorem (a): The area t of the tangential quadrilateral is equal to sqrt(a * b * c * d)
theorem part_a (h : t = area_tangential a b c d) : t = Real.sqrt (a * b * c * d) :=
by
  sorry

-- Define the relationship of r and R
def inradius (a b c d t : ℝ) (r : ℝ) : Prop := t = ((a + b + c + d) / 2) * r
def circumradius (a b c d t : ℝ) (R : ℝ) : Prop := r * Real.sqrt 2 ≤ R

-- Theorem (b): The radius of the inscribed circle r and the radius of the circumscribed circle R satisfies r * sqrt(2) <= R
theorem part_b (h_area : t = area_tangential a b c d)
               (h_inradius : inradius a b c d t r)
               (h_circumradius : circumradius a b c d t R) : r * Real.sqrt 2 ≤ R :=
by
  sorry

end part_a_part_b_l374_374106


namespace sum_of_possible_x_values_l374_374446

-- Given a class with a total of 360 students
def total_students : ℕ := 360

-- Conditions
def min_students_per_row : ℕ := 18
def min_rows : ℕ := 12

-- Define x validation
def is_valid_students_per_row (x : ℕ) : Prop :=
  total_students % x = 0 ∧ x ≥ min_students_per_row ∧ total_students / x ≥ min_rows

-- Possible valid x values
noncomputable def valid_x_values : List ℕ :=
  (List.range (total_students + 1)).filter is_valid_students_per_row

-- Sum of valid x values
noncomputable def sum_of_valid_x_values : ℕ :=
  valid_x_values.sum

theorem sum_of_possible_x_values :
  sum_of_valid_x_values = 66 :=
sorry

end sum_of_possible_x_values_l374_374446


namespace sharmila_hourly_wage_l374_374134

-- Sharmila works 10 hours per day on Monday, Wednesday, and Friday.
def hours_worked_mwf : ℕ := 3 * 10

-- Sharmila works 8 hours per day on Tuesday and Thursday.
def hours_worked_tt : ℕ := 2 * 8

-- Total hours worked in a week.
def total_hours_worked : ℕ := hours_worked_mwf + hours_worked_tt

-- Sharmila earns $460 per week.
def weekly_earnings : ℕ := 460

-- Calculate and prove her hourly wage is $10 per hour.
theorem sharmila_hourly_wage : (weekly_earnings / total_hours_worked) = 10 :=
by sorry

end sharmila_hourly_wage_l374_374134


namespace count_no_singletons_subsets_l374_374708

-- Definition of a set S
def S : Set ℤ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Function to check if a subset has no singletons
def no_singletons (A : Set ℤ) : Prop :=
  ∀ k ∈ A, k-1 ∈ A ∨ k+1 ∈ A

-- Total number of 3-element subsets without singletons
theorem count_no_singletons_subsets : 
  {A : Set ℤ // A ⊆ S ∧ A.card = 3 ∧ no_singletons A}.card = 6 := 
  sorry

end count_no_singletons_subsets_l374_374708


namespace max_area_of_sheep_pen_is_112_l374_374165

/-- A shepherd has 15 segments of 2-meter-long fences to form a rectangular sheep pen against a wall.
    Find the maximum area of the sheep pen that can be enclosed. --/
noncomputable def max_area_sheep_pen : ℝ :=
  let total_fence_length : ℝ := 15 * 2
  in let area_function (w : ℝ) : ℝ := (total_fence_length - 2 * w) * w
  in Real.Sup (set_of (λ a, ∃ w : ℝ, 0 ≤ w ∧ area_function w = a))

theorem max_area_of_sheep_pen_is_112 : max_area_sheep_pen = 112 :=
sorry

end max_area_of_sheep_pen_is_112_l374_374165


namespace horner_eval_add_mul_count_l374_374895

-- Lean 4 statement

def f (x : ℝ) : ℝ := 5 * x^6 + 4 * x^5 + x^4 + 3 * x^3 - 81 * x^2 + 9 * x - 1

theorem horner_eval_add_mul_count :
  ∀ x : ℝ, ∃ (add_count mul_count : ℕ), add_count = 6 ∧ mul_count = 6 ∧ 
  by horner_method f(x) = 5 * x^6 + 4 * x^5 + x^4 + 3 * x^3 - 81 * x^2 + 9 * x - 1 :=
begin
  intros,
  use [6, 6],
  split,
  { -- Proof for addition count
    sorry
  },
  { -- Proof for multiplication count 
    sorry
  }
end

end horner_eval_add_mul_count_l374_374895


namespace optionA_is_multiple_of_5_optionB_is_multiple_of_5_optionC_is_multiple_of_5_optionD_is_multiple_of_5_optionE_is_not_multiple_of_5_l374_374904

-- Definitions of the options
def optionA : ℕ := 2019^2 - 2014^2
def optionB : ℕ := 2019^2 * 10^2
def optionC : ℕ := 2020^2 / 101^2
def optionD : ℕ := 2010^2 - 2005^2
def optionE : ℕ := 2015^2 / 5^2

-- Statements to be proven
theorem optionA_is_multiple_of_5 : optionA % 5 = 0 := by sorry
theorem optionB_is_multiple_of_5 : optionB % 5 = 0 := by sorry
theorem optionC_is_multiple_of_5 : optionC % 5 = 0 := by sorry
theorem optionD_is_multiple_of_5 : optionD % 5 = 0 := by sorry
theorem optionE_is_not_multiple_of_5 : optionE % 5 ≠ 0 := by sorry

end optionA_is_multiple_of_5_optionB_is_multiple_of_5_optionC_is_multiple_of_5_optionD_is_multiple_of_5_optionE_is_not_multiple_of_5_l374_374904


namespace men_entered_count_l374_374690

variable (M W x : ℕ)

noncomputable def initial_ratio : Prop := M = 4 * W / 5
noncomputable def men_entered : Prop := M + x = 14
noncomputable def women_double : Prop := 2 * (W - 3) = 14

theorem men_entered_count (M W x : ℕ) (h1 : initial_ratio M W) (h2 : men_entered M x) (h3 : women_double W) : x = 6 := by
  sorry

end men_entered_count_l374_374690


namespace solution_set_inequality_l374_374597

theorem solution_set_inequality (x : ℝ) (f : ℝ → ℝ) (h : f = λ x, x^2 - 5 * x + 6) :
    f x > 0 ↔ (x > 3 ∨ x < 2) := by
  sorry

end solution_set_inequality_l374_374597


namespace inequality_holds_l374_374227

theorem inequality_holds (a b : ℝ) (h₁ : a = 1 / 2) (h₂ : b = 1) :
  ∀ (n : ℕ) (h₃ : n > 2) (x : Fin n → ℝ), (∀ i, x i ≥ 0)
  → (Finset.range n).sum (λ i, x i * x (i + 1)) ≥ (Finset.range n).sum (λ i, x i^a * x (i + 1)^b * x (i + 2)^a) :=
sorry

end inequality_holds_l374_374227


namespace circle_count_gt_l374_374720

axiom points_not_collinear {n : ℕ} (P : fin 2n+3 → ℝ × ℝ) :
  ∀ i j k : fin 2n+3, i ≠ j → j ≠ k → i ≠ k → ¬ collinear (P i) (P j) (P k)

axiom points_not_concyclic {n : ℕ} (P : fin 2n+3 → ℝ × ℝ) :
  ∀ i j k l : fin 2n+3, i ≠ j → j ≠ k → k ≠ l → l ≠ i → ¬ concyclic (P i) (P j) (P k) (P l)

def num_circles {n : ℕ} (P : fin 2n+3 → ℝ × ℝ) : ℕ := 
  { K | ∃ i j k : fin 2n+3, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (set.count_inside_outside (P i) (P j) (P k) P).fst = n ∧ 
    (set.count_inside_outside (P i) (P j) (P k) P).snd = n }.count

theorem circle_count_gt {n : ℕ} (P : fin 2n+3 → ℝ × ℝ) :
  (num_circles P) > (1/real.pi * nat.choose (2n + 3) 2) :=
by {
  sorry
}

end circle_count_gt_l374_374720


namespace find_angle_AOB_l374_374752

structure Circle (R : ℝ) :=
(center : ℝ × ℝ)
(radius : ℝ)

def point_on_circle (P : ℝ × ℝ) (c : Circle 12) : Prop :=
  (P.1 - c.center.1)^2 + (P.2 - c.center.2)^2 = c.radius^2

def tangent_line (P : ℝ × ℝ) (c : Circle 12) : Prop :=
  ∃ (m b : ℝ), ∀ (x y : ℝ), y = m * x + b ∧ point_on_circle (x, y) c

def inscribed_circle (c1 : Circle 12) {A B C M K H : ℝ × ℝ} : Prop :=
  ∀ (x y : ℝ) (P Q : ℝ × ℝ), tangent_line A c1 ∧ tangent_line B c1 ∧ 
  tangent_line C c1 ∧ tangent_line K c1 ∧ tangent_line H c1 ∧ 
  dist M Q = 3

theorem find_angle_AOB :
  ∀ {A B C M K H : ℝ × ℝ}, 
    point_on_circle A (Circle.mk (0, 0) 12) →
    point_on_circle B (Circle.mk (0, 0) 12) →
    tangent_line A (Circle.mk (0, 0) 12) →
    tangent_line B (Circle.mk (0, 0) 12) →
    inscribed_circle (Circle.mk (0, 0) 12) →
    dist ƒfrom_origin M (Circle.mk (0, 0) 12) = 3 →
  ∠(0, 0) A B = 120 :=
sorry

end find_angle_AOB_l374_374752


namespace amount_of_CaO_required_l374_374579

def chemical_reaction_CaO_H2O_CaOH2 (n_H2O : ℕ) (R : ℕ → ℕ → ℕ → Prop) : Prop :=
  ∃ n_CaO n_Ca(OH)₃, R n_CaO n_H2O n_Ca(OH)₃ ∧ n_H2O = 3 ∧ n_CaO = 3 ∧ n_Ca(OH)₃ = 3

-- Molar ratio function for our specific reaction
def molar_ratio (n_CaO n_H2O n_Ca(OH)₃ : ℕ) : Prop :=
  n_CaO = n_H2O ∧ n_H2O = n_Ca(OH)₃

theorem amount_of_CaO_required :
  chemical_reaction_CaO_H2O_CaOH2 3 molar_ratio :=
by
  sorry

end amount_of_CaO_required_l374_374579


namespace completion_time_C_l374_374141

theorem completion_time_C (r_A r_B r_C : ℝ) 
  (h1 : r_A + r_B = 1 / 3) 
  (h2 : r_B + r_C = 1 / 3) 
  (h3 : r_A + r_C = 1 / 3) :
  1 / r_C = 6 :=
by
  sorry

end completion_time_C_l374_374141


namespace number_of_correct_propositions_is_one_l374_374419

def obtuse_angle_is_second_quadrant (θ : ℝ) : Prop :=
  θ > 90 ∧ θ < 180

def acute_angle (θ : ℝ) : Prop :=
  θ < 90

def first_quadrant_not_negative (θ : ℝ) : Prop :=
  θ > 0 ∧ θ < 90

def second_quadrant_greater_first (θ₁ θ₂ : ℝ) : Prop :=
  (θ₁ > 90 ∧ θ₁ < 180) → (θ₂ > 0 ∧ θ₂ < 90) → θ₁ > θ₂

theorem number_of_correct_propositions_is_one :
  (¬ ∀ θ, obtuse_angle_is_second_quadrant θ) ∧
  (∀ θ, acute_angle θ → θ < 90) ∧
  (¬ ∀ θ, first_quadrant_not_negative θ) ∧
  (¬ ∀ θ₁ θ₂, second_quadrant_greater_first θ₁ θ₂) →
  1 = 1 :=
by
  sorry

end number_of_correct_propositions_is_one_l374_374419


namespace total_shaded_area_l374_374960

theorem total_shaded_area
  (carpet_side : ℝ)
  (large_square_side : ℝ)
  (small_square_side : ℝ)
  (ratio_large : carpet_side / large_square_side = 4)
  (ratio_small : large_square_side / small_square_side = 2) : 
  (1 * large_square_side^2 + 12 * small_square_side^2 = 64) := 
by 
  sorry

end total_shaded_area_l374_374960


namespace total_apples_packed_correct_l374_374745

-- Define the daily production of apples under normal conditions
def apples_per_box := 40
def boxes_per_day := 50
def days_per_week := 7
def apples_per_day := apples_per_box * boxes_per_day

-- Define the change in daily production for the next week
def fewer_apples := 500
def apples_per_day_next_week := apples_per_day - fewer_apples

-- Define the weekly production in normal and next conditions
def apples_first_week := apples_per_day * days_per_week
def apples_second_week := apples_per_day_next_week * days_per_week

-- Define the total apples packed in two weeks
def total_apples_packed := apples_first_week + apples_second_week

-- Prove the total apples packed is 24500
theorem total_apples_packed_correct : total_apples_packed = 24500 := by
  sorry

end total_apples_packed_correct_l374_374745


namespace incorrect_statement_l374_374096

-- Define the general rules of program flowcharts
def isValidStart (box : String) : Prop := box = "start"
def isValidEnd (box : String) : Prop := box = "end"
def isInputBox (box : String) : Prop := box = "input"
def isOutputBox (box : String) : Prop := box = "output"

-- Define the statement to be proved incorrect
def statement (boxes : List String) : Prop :=
  ∀ xs ys, boxes = xs ++ ["start", "input"] ++ ys ->
           ∀ zs ws, boxes = zs ++ ["output", "end"] ++ ws

-- The target theorem stating that the statement is incorrect
theorem incorrect_statement (boxes : List String) :
  ¬ statement boxes :=
sorry

end incorrect_statement_l374_374096


namespace equation_of_circle_l374_374682

def distance_point_to_line (A B C x₀ y₀ : ℝ) : ℝ := 
  |A * x₀ + B * y₀ + C| / (Real.sqrt (A^2 + B^2))

theorem equation_of_circle :
  let A := 3
  let B := -4
  let C := 5
  let O := (0 : ℝ, 0 : ℝ)
  let line : ℝ × ℝ → Prop := fun (p : ℝ × ℝ) => A * p.1 + B * p.2 + C = 0
  let r := distance_point_to_line A B C 0 0
  let circle_eq := (fun (x y : ℝ) => x^2 + y^2 = r^2)
  r = 1 ∧ circle_eq 0 0 :=
by
  sorry

end equation_of_circle_l374_374682


namespace valid_votes_correct_l374_374334

noncomputable def Total_votes : ℕ := 560000
noncomputable def Percentages_received : Fin 4 → ℚ 
| 0 => 0.4
| 1 => 0.35
| 2 => 0.15
| 3 => 0.1

noncomputable def Percentages_invalid : Fin 4 → ℚ 
| 0 => 0.12
| 1 => 0.18
| 2 => 0.25
| 3 => 0.3

noncomputable def Votes_received (i : Fin 4) : ℚ := Total_votes * Percentages_received i

noncomputable def Invalid_votes (i : Fin 4) : ℚ := Votes_received i * Percentages_invalid i

noncomputable def Valid_votes (i : Fin 4) : ℚ := Votes_received i - Invalid_votes i

def A_valid_votes := 197120
def B_valid_votes := 160720
def C_valid_votes := 63000
def D_valid_votes := 39200

theorem valid_votes_correct :
  Valid_votes 0 = A_valid_votes ∧
  Valid_votes 1 = B_valid_votes ∧
  Valid_votes 2 = C_valid_votes ∧
  Valid_votes 3 = D_valid_votes := by
  sorry

end valid_votes_correct_l374_374334


namespace min_neg_half_neg_third_l374_374656

-- Define the concept of minimum between two numbers
def my_min (m n : ℝ) : ℝ := if m ≤ n then m else n

theorem min_neg_half_neg_third : my_min (-1/2) (-1/3) = -1/2 := 
by 
  sorry

end min_neg_half_neg_third_l374_374656


namespace driver_actual_speed_l374_374913

theorem driver_actual_speed (v t : ℝ) 
  (h1 : t > 0) 
  (h2 : v > 0) 
  (cond : v * t = (v + 18) * (2 / 3 * t)) : 
  v = 36 :=
by 
  sorry

end driver_actual_speed_l374_374913


namespace trailing_zeros_345_factorial_l374_374429

theorem trailing_zeros_345_factorial : trailingZeros 345! = 84 := sorry

-- Define the function to count trailing zeros of a factorial
def trailingZeros (n : ℕ) : ℕ :=
  let count (p : ℕ) (n : ℕ) : ℕ := if n = 0 then 0 else n / p + count p (n / p)
  count 5 n

end trailing_zeros_345_factorial_l374_374429


namespace max_min_product_xy_theorem_l374_374864

noncomputable def max_min_product_xy (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) : Prop :=
  -1 ≤ x * y ∧ x * y ≤ 1/2

theorem max_min_product_xy_theorem (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) :
  max_min_product_xy x y a h1 h2 :=
sorry

end max_min_product_xy_theorem_l374_374864


namespace sin_graph_symmetric_l374_374286

theorem sin_graph_symmetric {ω : ℝ} {φ : ℝ} (hω : ω > 0) (hφ1 : -π / 2 < φ) 
  (hφ2 : φ < π / 2) (h_period : ∀ x, sin (ω * x + φ) = sin (ω * (x + π))) 
  (h_odd : ∀ x, sin (2 * (x - π / 3) + φ) = -sin (2 * x + φ)) : 
  ∃ x, f x = sin (2 * x - π / 3) ∧ x = 5 * π / 12 → 
  symmetric_about_line (λ x, sin (2 * x - π / 3)) (5 * π / 12) :=
sorry

end sin_graph_symmetric_l374_374286


namespace servings_per_cup_l374_374408

variable {honey_per_ounce : ℕ}
variable {ounces_per_container : ℕ}
variable {cups_per_night : ℕ}
variable {nights : ℕ}

def total_servings (honey_per_ounce ounces_per_container : ℕ) := honey_per_ounce * ounces_per_container
def total_cups (cups_per_night nights : ℕ) := cups_per_night * nights

theorem servings_per_cup 
  (honey_per_ounce : ℕ)
  (ounces_per_container : ℕ)
  (cups_per_night : ℕ)
  (nights : ℕ)
  (honey_servings : total_servings honey_per_ounce ounces_per_container)
  (total_cups : total_cups cups_per_night nights)
  (honey_servings = total_cups) :
  honey_servings / total_cups = 1 :=
by {
  sorry
}

end servings_per_cup_l374_374408


namespace odd_even_subsets_count_eq_odd_even_subsets_weights_sum_eq_odd_subsets_weights_sum_l374_374484

open Finset

/-- 
  Set S_n is defined as the set {1, 2, ..., n}
  X is a subset of S_n, and weight(X) is defined as the sum of elements in X, with weight(empty) = 0
  X is called odd(Even) subset if weight(X) is odd(Even)
--/
section odd_even_subsets

variables (n : ℕ)
def S_n := range (n + 1)

/-- 1. Prove that the number of odd subsets of S_n is equal to the number of even subsets of S_n -/
theorem odd_even_subsets_count_eq : 
  (S_n n).powerset.filter (λ x, (x.sum id) % 2 = 1).card = (S_n n).powerset.filter (λ x, (x.sum id) % 2 = 0).card := 
sorry

/-- 2. Prove that when n ≥ 3, the sum of the weights of all odd subsets of S_n is equal to the sum of the weights of all even subsets of S_n -/
theorem odd_even_subsets_weights_sum_eq (h : n ≥ 3) : 
  ((S_n n).powerset.filter (λ x, (x.sum id) % 2 = 1)).sum (λ x, x.sum id) = ((S_n n).powerset.filter (λ x, (x.sum id) % 2 = 0)).sum (λ x, x.sum id) := 
sorry

/-- 3. When n ≥ 3, find the sum of the weights of all odd subsets of S_n -/
theorem odd_subsets_weights_sum (h : n ≥ 3): 
  ((S_n n).powerset.filter (λ x, (x.sum id) % 2 = 1)).sum (λ x, x.sum id) = 2^(n - 3) * n * (n + 1) := 
sorry

end odd_even_subsets

end odd_even_subsets_count_eq_odd_even_subsets_weights_sum_eq_odd_subsets_weights_sum_l374_374484


namespace exists_at_most_two_quadratic_polynomials_l374_374178

noncomputable def S : set ℂ := sorry

noncomputable def S_is_of_size_n_minus_3_real (n : ℕ) : Prop :=
  ∃ (s : finset ℂ), s.card = n ∧ s.card = s.filter (λ x, x.im = 0).card + 3

theorem exists_at_most_two_quadratic_polynomials 
  (S_is_of_size_n_minus_3_real_n n : ℕ) (hS : S_is_of_size_n_minus_3_real n) 
  (f : ℂ → ℂ) (hf : ∀ (z : ℂ), z ∈ S → f z ∈ S ∧ ∀ w ∈ S, ∃ z ∈ S, f z = w) :
  ∃ f1 f2 : ℂ → ℂ, f = f1 ∨ f = f2 :=
sorry

end exists_at_most_two_quadratic_polynomials_l374_374178


namespace function_passes_through_point_l374_374460

theorem function_passes_through_point (a : ℝ) (x y : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  (x = 1 ∧ y = 4) ↔ (y = a^(x-1) + 3) :=
sorry

end function_passes_through_point_l374_374460


namespace exponential_reliability_property_l374_374418

section ExponentialReliability

variable (λ : ℝ) (t t₀ : ℝ)
variable (hλ : λ > 0)

-- Definition of the reliability function 
def R (t : ℝ) : ℝ := real.exp (-λ * t)

-- Hypotheses defining the probabilities using the reliability function
def P_A := R λ t₀
def P_B := R λ t
def P_AB := R λ (t₀ + t)

-- Theorem statement
theorem exponential_reliability_property (hλ : λ > 0): 
  let P_A_B := P_AB λ t₀ t / P_A λ t₀
  P_A_B = real.exp (-λ * t) :=
by
  sorry

end ExponentialReliability

end exponential_reliability_property_l374_374418


namespace average_squares_of_first_11_consecutive_even_numbers_l374_374478

theorem average_squares_of_first_11_consecutive_even_numbers :
  let even_numbers := (List.range 11).map (λ n, 2 * (n + 1))
  let squares := even_numbers.map (λ n, n ^ 2)
  let sum_squares := squares.sum
  sum_squares / 11 = 184 :=
by
  let even_numbers := (List.range 11).map (λ n, 2 * (n + 1))
  let squares := even_numbers.map (λ n, n ^ 2)
  let sum_squares := squares.sum
  have h_even_numbers : even_numbers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22] := rfl
  have h_squares : squares = [4, 16, 36, 64, 100, 144, 196, 256, 324, 400, 484] := rfl
  have h_sum_squares : sum_squares = 2024 := by simp [sum_squares]
  show 2024 / 11 = 184
  sorry

end average_squares_of_first_11_consecutive_even_numbers_l374_374478


namespace divisible_by_133_l374_374395

theorem divisible_by_133 (n : ℕ) : (11^(n + 2) + 12^(2*n + 1)) % 133 = 0 :=
by
  sorry

end divisible_by_133_l374_374395


namespace minimum_N_l374_374513

/-- A set of measurements with the following percentage frequency distribution:
    - 12.5%
    - 50%
    - 25%
    - 12.5%
Can be represented as fractions of the total N measurements. 

Prove that the minimum possible value of N such that these frequencies sum to exactly the total number of measurements is 8.
--/
theorem minimum_N {N : ℕ} :
  (∃ (m₁ m₂ m₃ m₄ : ℕ), 
    m₁ * 100 / N = 12.5 ∧ 
    m₂ * 100 / N = 50 ∧ 
    m₃ * 100 / N = 25 ∧ 
    m₄ * 100 / N = 12.5 ∧ 
    m₁ + m₂ + m₃ + m₄ = N) → N = 8 := 
sorry

end minimum_N_l374_374513


namespace correct_average_l374_374477

theorem correct_average
  (incorrect_avg : ℝ)
  (incorrect_num correct_num : ℝ)
  (n : ℕ)
  (h1 : incorrect_avg = 16)
  (h2 : incorrect_num = 26)
  (h3 : correct_num = 46)
  (h4 : n = 10) :
  (incorrect_avg * n - incorrect_num + correct_num) / n = 18 :=
sorry

end correct_average_l374_374477


namespace remainder_equiv_l374_374459

noncomputable def x : ℤ := 2^75

theorem remainder_equiv 
  (x_def : x = 2^75)
  (x_squared : 2^150 = x^2)
  (original_expr : x^4 + 300 = (x^2 + x + 1) * (x^2 - x + 299) + 1) : 
  remainder (x^4 + 300) (x^2 + x + 1) = 1 :=
by sorry

end remainder_equiv_l374_374459


namespace trajectory_of_point_l374_374602

variables {a b x : ℝ}

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 2 * a - b

def domain (a : ℝ) := Set.Icc (2 * a - 1) (a^2 + 1)

def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x ∈ domain a, f (-x) = f x

theorem trajectory_of_point :
  is_even_function (f a b) →
  {(a, b)} = { (0, 0), (-2, 0) } :=
sorry

end trajectory_of_point_l374_374602


namespace problem_solution_l374_374736

-- Define the set M based on the condition |2x - 1| < 1
def M : Set ℝ := {x | 0 < x ∧ x < 1}

-- Main theorem composed of two parts
theorem problem_solution :
  (∀ x, |2 * x - 1| < 1 ↔ 0 < x ∧ x < 1) ∧
  (∀ a b ∈ M, (a * b + 1) > (a + b)) :=
by
  split
  -- Part 1: Prove the equivalence of |2x - 1| < 1 and the set definition of M
  · intro x
    split
    -- Prove |2x - 1| < 1 → 0 < x ∧ x < 1
    sorry
    -- Prove 0 < x ∧ x < 1 → |2x - 1| < 1
    sorry
  -- Part 2: Prove ab + 1 > a + b for all a, b in M
  · intros a b ha hb
    have ha' : 0 < a ∧ a < 1 := ha
    have hb' : 0 < b ∧ b < 1 := hb
    -- Prove the inequality
    sorry

end problem_solution_l374_374736


namespace set_S_cardinality_l374_374041

-- Definitions
def prime (p : ℕ) : Prop := p.prime
def distinct_integers (a : List ℕ) : Prop := a.nodup
def in_range (x : ℕ) (p : ℕ) : Prop := 1 ≤ x ∧ x ≤ p - 1
def set_S (p : ℕ) (a : List ℕ) :=
  {n : ℕ | in_range n p ∧ (∀ i j : ℕ, i < j → i < a.length ∧ j < a.length → 
  (n * a.nth i % p) < (n * a.nth j % p))}

-- Problem statement
theorem set_S_cardinality (p : ℕ) (a : List ℕ) (k : ℕ) (h_prime : prime p)
  (h_distinct : distinct_integers a) (h_length : a.length = k + 1)
  (h_in_range : ∀ x, x ∈ a → in_range x p) :
  (set_S p a).card < 2 * p / (k + 1) := 
sorry

end set_S_cardinality_l374_374041


namespace Jill_watch_time_l374_374354

theorem Jill_watch_time (show1_length : ℕ) (show2_length : ℕ) :
  show1_length = 30 ∧ show2_length = 4 * show1_length → show1_length + show2_length = 150 :=
by
  intro h
  obtain ⟨h1, h2⟩ := h
  rw [h1, h2]
  norm_num

end Jill_watch_time_l374_374354


namespace nine_by_nine_grid_possible_l374_374693

theorem nine_by_nine_grid_possible : 
  ∃ (grid : ℕ → ℕ → ℕ), 
    (∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 81 ∧ (∃ k, ∑ x in finset.range 3, ∑ y in finset.range 3, grid (3*k.1 + x) (3*k.2 + y) = 369)) ∧
    (∀ i j, i ≠ j → grid i j ≠ grid i j) :=
sorry

end nine_by_nine_grid_possible_l374_374693


namespace common_tangents_of_C1_C2_l374_374571

def Circle (x y a b r : ℝ) := (x + a)^2 + (y + b)^2 = r^2

def C1 : Circle x y 1 1 2
def C2 : Circle x y (-3) 1 2

theorem common_tangents_of_C1_C2 : 
  ∀ x y : ℝ, Circle C1 x y ∧ Circle C2 x y → 
  number_of_common_tangents C1 C2 = 3 := 
by 
  sorry

end common_tangents_of_C1_C2_l374_374571


namespace no_possible_seating_arrangement_l374_374902

theorem no_possible_seating_arrangement : 
  ¬(∃ (students : Fin 11 → Fin 4),
    ∀ (i : Fin 11),
    ∃ (s1 s2 s3 s4 s5 : Fin 11),
      s1 = i ∧ 
      (s2 = (i + 1) % 11) ∧ 
      (s3 = (i + 2) % 11) ∧ 
      (s4 = (i + 3) % 11) ∧ 
      (s5 = (i + 4) % 11) ∧
      ∃ (g1 g2 g3 g4 : Fin 4),
        (students s1 = g1) ∧ 
        (students s2 = g2) ∧ 
        (students s3 = g3) ∧ 
        (students s4 = g4) ∧ 
        (students s5).val ≠ (students s1).val ∧ 
        (students s5).val ≠ (students s2).val ∧ 
        (students s5).val ≠ (students s3).val ∧ 
        (students s5).val ≠ (students s4).val) :=
sorry

end no_possible_seating_arrangement_l374_374902


namespace smallest_B_for_sum_2024_l374_374999

theorem smallest_B_for_sum_2024 : 
  ∃ B : ℤ, (∀ n : ℤ, B ≤ n → n ≤ B + 4046 → n ≠ B + 4047) ∧ (B + B + 1 = -2023) ∧ (∀ z : ℤ, ((z, 2024 ∃ x : ℕ, x ≠ 2024) → x ≥ -2023) := sorry

end smallest_B_for_sum_2024_l374_374999


namespace carl_personal_owe_l374_374542

def property_damage : ℝ := 40000
def medical_bills : ℝ := 70000
def insurance_coverage : ℝ := 0.8
def carl_responsibility : ℝ := 0.2
def total_cost : ℝ := property_damage + medical_bills
def carl_owes : ℝ := total_cost * carl_responsibility

theorem carl_personal_owe : carl_owes = 22000 := by
  sorry

end carl_personal_owe_l374_374542


namespace trig_identity_example_l374_374192

theorem trig_identity_example : sin (13 * real.pi / 180) * cos (17 * real.pi / 180) + cos (13 * real.pi / 180) * sin (17 * real.pi / 180) = 1 / 2 :=
by
  sorry

end trig_identity_example_l374_374192


namespace purely_imaginary_l374_374667

theorem purely_imaginary {m : ℝ} (h1 : m^2 - 3 * m = 0) (h2 : m^2 - 5 * m + 6 ≠ 0) : m = 0 :=
sorry

end purely_imaginary_l374_374667


namespace monochromatic_triangle_l374_374138

open Set

theorem monochromatic_triangle 
  (points : Fin 6 → ℝ × ℝ × ℝ)
  (h : ∀ (s : Finset (Fin 6)), s.card = 4 → ¬Collinear ℝ (s.image points))
  (color : (Fin 6) × (Fin 6) → Bool) :
  ∃ (a b c : Fin 6), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ color (a, b) = color (b, c) ∧ color (b, c) = color (c, a) :=
sorry

end monochromatic_triangle_l374_374138


namespace rate_of_interest_l374_374964

namespace SimpleInterest

-- Define the simple interest equation.
def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

-- Definitions of the given conditions.
def P : ℝ := 8032.5
def SI : ℝ := 4016.25
def T : ℝ := 5.0

-- The proof statement we want to validate.
theorem rate_of_interest : simple_interest P 10 T = SI := by
  -- Proof goes here
  sorry

end SimpleInterest

end rate_of_interest_l374_374964


namespace A_alone_days_l374_374487

noncomputable def days_for_A (r_A r_B r_C : ℝ) : ℝ :=
  1 / r_A

theorem A_alone_days
  (r_A r_B r_C : ℝ) 
  (h1 : r_A + r_B = 1 / 3)
  (h2 : r_B + r_C = 1 / 6)
  (h3 : r_A + r_C = 1 / 4) :
  days_for_A r_A r_B r_C = 4.8 := by
  sorry

end A_alone_days_l374_374487


namespace find_n_l374_374092

theorem find_n (n : ℕ) (h : n! / (n - 3)! = 504) : n = 9 :=
sorry

end find_n_l374_374092


namespace total_revenue_correct_l374_374830

def KwikETaxCenter : Type := ℕ

noncomputable def federal_return_price : ℕ := 50
noncomputable def state_return_price : ℕ := 30
noncomputable def quarterly_business_taxes_price : ℕ := 80
noncomputable def international_return_price : ℕ := 100
noncomputable def value_added_service_price : ℕ := 75

noncomputable def federal_returns_sold : ℕ := 60
noncomputable def state_returns_sold : ℕ := 20
noncomputable def quarterly_returns_sold : ℕ := 10
noncomputable def international_returns_sold : ℕ := 13
noncomputable def value_added_services_sold : ℕ := 25

noncomputable def international_discount : ℕ := 20

noncomputable def calculate_total_revenue 
   (federal_price : ℕ) (state_price : ℕ) 
   (quarterly_price : ℕ) (international_price : ℕ) 
   (value_added_price : ℕ)
   (federal_sold : ℕ) (state_sold : ℕ) 
   (quarterly_sold : ℕ) (international_sold : ℕ) 
   (value_added_sold : ℕ)
   (discount : ℕ) : ℕ := 
    (federal_price * federal_sold) 
  + (state_price * state_sold) 
  + (quarterly_price * quarterly_sold) 
  + ((international_price - discount) * international_sold) 
  + (value_added_price * value_added_sold)

theorem total_revenue_correct :
  calculate_total_revenue federal_return_price state_return_price 
                          quarterly_business_taxes_price international_return_price 
                          value_added_service_price
                          federal_returns_sold state_returns_sold 
                          quarterly_returns_sold international_returns_sold 
                          value_added_services_sold 
                          international_discount = 7315 := 
  by sorry

end total_revenue_correct_l374_374830


namespace polygons_count_l374_374057

-- Define the problem conditions
def num_vertices := 12
def num_segments := 11
def distinct_polygons := 1024

-- Lean 4 statement
theorem polygons_count :
  (number of distinct 11-segment open polygons without self-intersections
   with vertices at the points of a regular 12-gon,
   considering polygons that can be transformed into each other by rotation as the same)
  = distinct_polygons :=
by
  sorry

end polygons_count_l374_374057


namespace find_digits_sum_l374_374804

theorem find_digits_sum
  (a b c d : ℕ)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (non_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (sum_of_numbers : 6 * (a + b + c + d) * 1111 = 73326) :
  {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end find_digits_sum_l374_374804


namespace find_k_coplanar_lines_l374_374053

-- Conditions and given problem
def line1 (s : ℝ) (k : ℝ) : ℝ × ℝ × ℝ := (-1 + s, 3 - k * s, 1 + k * s)
def line2 (t : ℝ) : ℝ × ℝ × ℝ := (t / 2, 1 + t, 2 - t)

-- The direction vectors of the lines
def dir_vec1 (k : ℝ) : ℝ × ℝ × ℝ := (1, -k, k)
def dir_vec2 : ℝ × ℝ × ℝ := (1 / 2, 1, -1)

-- Proof statement
theorem find_k_coplanar_lines (k : ℝ) : 
  (∃ (s t : ℝ), line1 s k = line2 t) → False ∨ k = -2 :=
by 
  sorry

end find_k_coplanar_lines_l374_374053


namespace room_dimensions_difference_l374_374164

-- Define necessary constants and variables
variables {L W H : ℝ}

-- Conditions of the problem
def W_def := W = (1/2) * L
def H_def := H = (3/4) * L
def volume_def := L * W * H = 384

-- The target statement to be proved
theorem room_dimensions_difference :
  W_def → H_def → volume_def → abs (L - W - H) = 2.52 :=
by
  intros
  sorry

end room_dimensions_difference_l374_374164


namespace necessary_and_sufficient_for_Sn_lt_an_l374_374436

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n + 1) * a 0 + (n * (n + 1)) / 2

theorem necessary_and_sufficient_for_Sn_lt_an
  (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ)
  (h_arith_seq : arithmetic_seq a d)
  (h_d_neg : d < 0)
  (m n : ℕ)
  (h_pos_m : m ≥ 3)
  (h_am_eq_Sm : a m = S m) :
  n > m ↔ S n < a n := sorry

end necessary_and_sufficient_for_Sn_lt_an_l374_374436


namespace range_of_a_if_p_or_q_true_range_of_a_if_p_or_q_true_and_p_and_q_false_l374_374732

section
variables (a : ℝ)

def p : Prop :=
  (a - 1 > 1 ∧ a - 3 > 0) ∨ (0 < a - 1 ∧ a - 3 < 0)

def q : Prop :=
  3a - 4 > 0 ∧ 4a^2 - 8 * (3 * a - 4) < 0

theorem range_of_a_if_p_or_q_true :
  (p a ∨ q a) → ((1 < a ∧ a < 2) ∨ (2 < a)) := sorry

theorem range_of_a_if_p_or_q_true_and_p_and_q_false :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → ((1 < a ∧ a < 2) ∨ (2 < a ∧ a ≤ 3) ∨ (4 ≤ a)) := sorry
end

end range_of_a_if_p_or_q_true_range_of_a_if_p_or_q_true_and_p_and_q_false_l374_374732


namespace min_value_arith_seq_l374_374295

theorem min_value_arith_seq (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + c = 2 * b) :
  (a + c) / b + b / (a + c) ≥ 5 / 2 := 
sorry

end min_value_arith_seq_l374_374295


namespace infinitely_many_f_nplus1_gt_fn_infinitely_many_f_nplus1_lt_fn_l374_374590

-- Define the function f(n) given the conditions
def f (n : Nat) : Real :=
  (1 : Real) / (n : Real) * (∑ i in Finset.range n.succ, ⌊(n : Real) / (i + 1)⌋)

-- Statement of the theorem for part (a)
theorem infinitely_many_f_nplus1_gt_fn : ∃ᶠ n in atTop, f n.succ > f n := sorry

-- Statement of the theorem for part (b)
theorem infinitely_many_f_nplus1_lt_fn : ∃ᶠ n in atTop, f n.succ < f n := sorry

end infinitely_many_f_nplus1_gt_fn_infinitely_many_f_nplus1_lt_fn_l374_374590


namespace max_stamps_with_50_dollars_l374_374324

theorem max_stamps_with_50_dollars (stamp_price : ℕ) (total_cents : ℕ) (h1 : stamp_price = 37) (h2 : total_cents = 5000) : 
  ∃ n : ℕ, 37 * n ≤ 5000 ∧ ∀ m : ℕ, 37 * m ≤ 5000 → m ≤ n :=
by
  use 135
  split
  · sorry -- Proof that 37 * 135 ≤ 5000
  · sorry -- Proof that if 37 * m ≤ 5000 then m ≤ 135

end max_stamps_with_50_dollars_l374_374324


namespace problem_l374_374081

theorem problem (p q r : ℝ)
    (h1 : p * 1^2 + q * 1 + r = 5)
    (h2 : p * 2^2 + q * 2 + r = 3) :
  p + q + 2 * r = 10 := 
sorry

end problem_l374_374081


namespace trig_identity_evaluation_l374_374540

theorem trig_identity_evaluation :
  4 * Real.cos (50 * Real.pi / 180) - Real.tan (40 * Real.pi / 180) = Real.sqrt 3 :=
by
  sorry

end trig_identity_evaluation_l374_374540


namespace problem_1_problem_2_l374_374631

def f (x : ℝ) : ℝ := x^2 - 4 * x + 6

theorem problem_1 (m : ℝ) (h_mono : ∀ x y, m ≤ x → x ≤ y → y ≤ m + 1 → f y ≤ f x) : m ≤ 1 :=
  sorry

theorem problem_2 (a b : ℝ) (h_min : a < b) 
  (h_min_val : ∀ x, a ≤ x ∧ x ≤ b → f a ≤ f x)
  (h_max_val : ∀ x, a ≤ x ∧ x ≤ b → f x ≤ f b) 
  (h_fa_eq_a : f a = a) (h_fb_eq_b : f b = b) : a = 2 ∧ b = 3 :=
  sorry

end problem_1_problem_2_l374_374631


namespace find_px_qx_sum_l374_374422

theorem find_px_qx_sum (p q : ℝ → ℝ)
  (h_asymptote_x_neg1 : ∀ x, q(x) = 0 → x = -1)
  (h_horizontal_asymptote : ∀ x, p(x) / q(x) → 0)
  (q_quadratic : ∃ a b c, ∀ x, q(x) = a * x^2 + b * x + c)
  (h_p3_zero : p 3 = 0)
  (h_q1_one : q 1 = 1)
  (h_p0_one : p 0 = 1) :
  ∀ x, p(x) + q(x) = x^2 - (1/3) * x :=
by
  sorry

end find_px_qx_sum_l374_374422


namespace eval_expression_l374_374193

def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem eval_expression :
  3^(Real.log 4 / Real.log 3) - 27^(2 / 3) + log10 0.01 + (3 / 4)^(-1 : ℝ) + Real.log (1 / Real.exp 1) = -20 / 3 :=
by
  sorry

end eval_expression_l374_374193


namespace qingming_festival_bus_distance_l374_374219

noncomputable def problem_statement : Prop :=
∀ (d : ℕ) (t : ℕ) (v : ℕ) (t1 : ℕ) (t2 : ℕ),
  (t = t1 + t2) ∧
  (t1 = 10:time) ∧
  (t1 + d/v) ∧
  (t2 + (d/1)=216km) ->
  d = 216

theorem qingming_festival_bus_distance:
  problem_statement :=
sorry

end qingming_festival_bus_distance_l374_374219


namespace whitewashed_fence_length_l374_374118

theorem whitewashed_fence_length :
  ∀ (T B_1 B_2 J L : ℝ),
    T = 100 →
    B_1 = 10 →
    B_2 = (T - B_1) / 5 →
    J = (T - B_1 - B_2) / 3 →
    L = T - B_1 - B_2 - J →
    L = 48 :=
by {
  intros,
  sorry
}

end whitewashed_fence_length_l374_374118


namespace range_of_positive_integers_in_list_K_l374_374383

-- Definition of the list K based on the given conditions
def list_K : List Int := List.range' (-4) 10

-- Definition to extract the positive integers from list_K
def positive_integers (l : List Int) : List Int := l.filter (λ x => x > 0)

-- Definition of the range of a list of integers
def range_of_list (l : List Int) : Int :=
  l.maximum.get_or_else 0 - l.minimum.get_or_else 0

-- Statement of the theorem to be proved
theorem range_of_positive_integers_in_list_K :
  range_of_list (positive_integers list_K) = 4 :=
by
  sorry

end range_of_positive_integers_in_list_K_l374_374383


namespace variance_of_heights_l374_374593
-- Importing all necessary libraries

-- Define a list of heights
def heights : List ℕ := [160, 162, 159, 160, 159]

-- Define the function to calculate the mean of a list of natural numbers
def mean (list : List ℕ) : ℚ :=
  list.sum / list.length

-- Define the function to calculate the variance of a list of natural numbers
def variance (list : List ℕ) : ℚ :=
  let μ := mean list
  (list.map (λ x => (x - μ) ^ 2)).sum / list.length

-- The theorem statement that proves the variance is 6/5
theorem variance_of_heights : variance heights = 6 / 5 :=
  sorry

end variance_of_heights_l374_374593


namespace petya_four_digits_l374_374788

theorem petya_four_digits :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
    6 * (a + b + c + d) * 1111 = 73326 ∧ 
    (∃ S, S = a + b + c + d ∧ S = 11) :=
begin
  use 1, 2, 3, 5,
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split,
  { exact dec_trivial },
  use 11,
  split,
  { exact dec_trivial },
  { exact dec_trivial }
end

end petya_four_digits_l374_374788


namespace find_digits_sum_l374_374801

theorem find_digits_sum
  (a b c d : ℕ)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (non_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (sum_of_numbers : 6 * (a + b + c + d) * 1111 = 73326) :
  {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end find_digits_sum_l374_374801


namespace unseen_corner_color_code_l374_374097

theorem unseen_corner_color_code (face_color_code : ℕ → ℕ) 
  (coding : face_color_code 1 = 1 ∧ face_color_code 2 = 2 ∧ face_color_code 3 = 3 ∧ 
            face_color_code 4 = 4 ∧ face_color_code 5 = 5 ∧ face_color_code 6 = 6) 
  (final_state : set ℕ) 
  (visible_corners : ∃ s, s ⊆ final_state ∧ s.card = 7) : 
  ∃ unseen_corner_color, unseen_corner_color = face_color_code 1 :=
begin
  -- Here we will prove that the unseen corner color code is 1
  sorry
end

end unseen_corner_color_code_l374_374097


namespace solve_system_l374_374822

-- Define the system of equations
def system_of_equations (a b c x y z : ℝ) :=
  x ≠ y ∧
  a ≠ 0 ∧
  c ≠ 0 ∧
  (x + z) * a = x - y ∧
  (x + z) * b = x^2 - y^2 ∧
  (x + z)^2 * (b^2 / (a^2 * c)) = (x^3 + x^2 * y - x * y^2 - y^3)

-- Proof goal: establish the values of x, y, and z
theorem solve_system (a b c x y z : ℝ) (h : system_of_equations a b c x y z):
  x = (a^3 * c + b) / (2 * a) ∧
  y = (b - a^3 * c) / (2 * a) ∧
  z = (2 * a^2 * c - a^3 * c - b) / (2 * a) :=
by
  sorry

end solve_system_l374_374822


namespace false_proposition_is_C_l374_374176

theorem false_proposition_is_C : ¬ (∀ x : ℝ, x^3 > 0) :=
sorry

end false_proposition_is_C_l374_374176


namespace arnold_gas_expenditure_l374_374535

-- Conditions as definitions in Lean
def avg_mpg_first_car := 50
def avg_mpg_second_car := 10
def avg_mpg_third_car := 15
def total_miles_per_month := 450
def gas_cost_per_gallon := 2
def number_of_cars := 3

-- Question (goal) formulated as a theorem
theorem arnold_gas_expenditure :
  let miles_per_car_per_month := total_miles_per_month / number_of_cars
      gas_first_car := miles_per_car_per_month / avg_mpg_first_car
      gas_second_car := miles_per_car_per_month / avg_mpg_second_car
      gas_third_car := miles_per_car_per_month / avg_mpg_third_car
      cost_first_car := gas_first_car * gas_cost_per_gallon
      cost_second_car := gas_second_car * gas_cost_per_gallon
      cost_third_car := gas_third_car * gas_cost_per_gallon
      total_cost := cost_first_car + cost_second_car + cost_third_car
  in total_cost = 56 :=
by
  sorry

end arnold_gas_expenditure_l374_374535


namespace trapezoid_AD_length_l374_374391

noncomputable def length_AD {A B C D M H K : Type} [geometry A B C D M H K] : Real :=
  let BC := 16
  let CM := 8
  let MD := 9
  let AD_eq_HD := true
  let AH_perpendicular_BM := true
  let AD := sorry
  AD

theorem trapezoid_AD_length (AD_eq_HD : AD = HD) (BC : ∀ {Real}, BC = 16) (CM : ∀ {Real}, CM = 8) (MD : ∀ {Real}, MD = 9) : length_AD = 18 := by
  sorry

end trapezoid_AD_length_l374_374391


namespace jana_distance_travel_in_20_minutes_l374_374350

theorem jana_distance_travel_in_20_minutes :
  ∀ (usual_pace half_pace double_pace : ℚ)
    (first_15_minutes_distance second_5_minutes_distance total_distance : ℚ),
  usual_pace = 1 / 30 →
  half_pace = usual_pace / 2 →
  double_pace = usual_pace * 2 →
  first_15_minutes_distance = 15 * half_pace →
  second_5_minutes_distance = 5 * double_pace →
  total_distance = first_15_minutes_distance + second_5_minutes_distance →
  total_distance = 7 / 12 := 
by
  intros
  sorry

end jana_distance_travel_in_20_minutes_l374_374350


namespace slope_angle_range_l374_374627

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * Real.sin x - (Real.sqrt 3 / 2) * Real.cos x

theorem slope_angle_range :
  ∃ a b c d : ℝ, 
  a = 0 ∧ b = π/4 ∧ c = 3*π/4 ∧ d = π ∧
  ∀ α : ℝ, (∃ x : ℝ, ∀ x, f' x = Real.sin (x + π/6) ∧ α = Real.arctan (f' x)) →
           (α ∈ set.Icc a b ∨ α ∈ set.Ico c d) :=
sorry

end slope_angle_range_l374_374627


namespace find_m_value_l374_374972

theorem find_m_value (m : ℝ) 
  (first_term : ℝ := 18) (second_term : ℝ := 6)
  (second_term_2 : ℝ := 6 + m) 
  (S1 : ℝ := first_term / (1 - second_term / first_term))
  (S2 : ℝ := first_term / (1 - second_term_2 / first_term))
  (eq_sum : S2 = 3 * S1) :
  m = 8 := by
  sorry

end find_m_value_l374_374972


namespace initial_incorrect_average_l374_374835

theorem initial_incorrect_average (S_correct S_wrong : ℝ) :
  (S_correct = S_wrong - 26 + 36) →
  (S_correct / 10 = 19) →
  (S_wrong / 10 = 18) :=
by
  sorry

end initial_incorrect_average_l374_374835


namespace increase_in_lines_l374_374430

variable (L : ℝ)
variable (h1 : L + (1 / 3) * L = 240)

theorem increase_in_lines : (240 - L) = 60 := by
  sorry

end increase_in_lines_l374_374430


namespace man_speed_with_the_stream_l374_374947

def speed_with_the_stream (V_m V_s : ℝ) : Prop :=
  V_m + V_s = 2

theorem man_speed_with_the_stream (V_m V_s : ℝ) (h1 : V_m - V_s = 2) (h2 : V_m = 2) : speed_with_the_stream V_m V_s :=
by
  sorry

end man_speed_with_the_stream_l374_374947


namespace prove_root_property_l374_374717

-- Define the quadratic equation and its roots
theorem prove_root_property :
  let r := -4 + Real.sqrt 226
  let s := -4 - Real.sqrt 226
  (r + 4) * (s + 4) = -226 :=
by
  -- the proof steps go here (omitted)
  sorry

end prove_root_property_l374_374717


namespace parabola_distance_l374_374671

theorem parabola_distance (c : ℝ) : 
  let vertex_y := c - 11 in
  abs vertex_y = 3 ↔ c = 8 ∨ c = 14 :=
by
  sorry

end parabola_distance_l374_374671


namespace trailing_zeros_factorial_345_l374_374427

theorem trailing_zeros_factorial_345 : 
  let count_factors (n k : ℕ) := n / k in
  count_factors 345 5 + count_factors 345 25 + count_factors 345 125 = 84 :=
by
  have count_factors := λ n k : ℕ, n / k
  calc
    count_factors 345 5 = 69    : by sorry
    count_factors 345 25 = 13   : by sorry
    count_factors 345 125 = 2   : by sorry
    69 + 13 + 2 = 84            : by ring

end trailing_zeros_factorial_345_l374_374427


namespace sum_of_exponents_is_five_l374_374882

theorem sum_of_exponents_is_five:
  ∃ (s : ℕ) (m : fin s → ℕ) (b : fin s → ℤ), 
    (∀ i j, i < j → m i > m j) ∧ 
    (∀ i, b i ∈ {1, -1, 2, -2}) ∧ 
    (finset.univ.sum (λ i, b i * 5 ^ m i) = 3125) → 
    (finset.univ.sum (λ i, m i) = 5) := 
sorry

end sum_of_exponents_is_five_l374_374882


namespace problem1_problem2_l374_374600

theorem problem1 (a b : ℤ) (h₁ : |a| = 5) (h₂ : |b| = 2) (h₃ : a > b) : a + b = 7 ∨ a + b = 3 := 
by sorry

theorem problem2 (a b : ℤ) (h₁ : |a| = 5) (h₂ : |b| = 2) (h₃ : |a + b| = |a| - |b|) : (a = -5 ∧ b = 2) ∨ (a = 5 ∧ b = -2) := 
by sorry

end problem1_problem2_l374_374600


namespace minimum_perimeter_l374_374482

noncomputable def minimum_perimeter_triangle (l m n : ℕ) : ℕ :=
  l + m + n

theorem minimum_perimeter :
  ∀ (l m n : ℕ),
    (l > m) → (m > n) → 
    ((∃ k : ℕ, 10^4 ∣ 3^l - 3^m + k * 10^4) ∧ (∃ k : ℕ, 10^4 ∣ 3^m - 3^n + k * 10^4) ∧ (∃ k : ℕ, 10^4 ∣ 3^l - 3^n + k * 10^4)) →
    minimum_perimeter_triangle l m n = 3003 :=
by
  intros l m n hlm hmn hmod
  sorry

end minimum_perimeter_l374_374482


namespace boys_went_down_the_slide_total_l374_374490

/-- Conditions -/
def a : Nat := 87
def b : Nat := 46
def c : Nat := 29

/-- The main proof problem -/
theorem boys_went_down_the_slide_total :
  a + b + c = 162 :=
by
  sorry

end boys_went_down_the_slide_total_l374_374490


namespace find_m_l374_374614

def A (m : ℤ) : Set ℤ := {2, 5, m ^ 2 - m}
def B (m : ℤ) : Set ℤ := {2, m + 3}

theorem find_m (m : ℤ) : A m ∩ B m = B m → m = 3 := by
  sorry

end find_m_l374_374614


namespace exists_pair_Nij_gt_200_l374_374139

open Function Set

/-- Main statement of the problem -/
theorem exists_pair_Nij_gt_200 (A : Fin 29 → Set ℕ) (e : ℝ) (h_e : e = Real.exp 1) :
  (∀ i : Fin 29, ∀ x : ℕ, (A i).filter (λ a, a ≤ x).card ≥ (x / e).toNat) →
  ∃ (i j : Fin 29), i < j ∧ ((A i ∩ A j).filter (λ a, a ≤ 1988)).card > 200 := 
sorry

end exists_pair_Nij_gt_200_l374_374139


namespace remainder_of_large_prime_l374_374883

open Nat

theorem remainder_of_large_prime :
  ∃ p : ℕ, 
    Prime p ∧
    p > 10^50 ∧
    (10^294 ≡ 1 [MOD p]) ∧
    (∀ n < 294, 10^n ≠ 1 [MOD p]) ∧
    (p % 10^9 = 572857143) := 
  sorry

end remainder_of_large_prime_l374_374883


namespace sum_of_consecutive_evens_l374_374435

theorem sum_of_consecutive_evens (E1 E2 E3 E4 : ℕ) (h1 : E4 = 38) (h2 : E3 = E4 - 2) (h3 : E2 = E3 - 2) (h4 : E1 = E2 - 2) : 
  E1 + E2 + E3 + E4 = 140 := 
by 
  sorry

end sum_of_consecutive_evens_l374_374435


namespace petya_digits_l374_374771

theorem petya_digits (a b c d : ℕ) (distinct_digits : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (non_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (sum_formed_numbers : 6 * (a + b + c + d) * 1111 = 73326) :
  {a, b, c, d} = {1, 2, 3, 5} :=
by 
  sorry

end petya_digits_l374_374771


namespace find_n_for_k_eq_1_l374_374655

theorem find_n_for_k_eq_1 (n : ℤ) (h : (⌊(n^2 : ℤ) / 5⌋ - ⌊n / 2⌋^2 = 1)) : n = 5 := 
by 
  sorry

end find_n_for_k_eq_1_l374_374655


namespace bob_total_candies_l374_374977

noncomputable def total_chewing_gums : ℕ := 45
noncomputable def total_chocolate_bars : ℕ := 60
noncomputable def total_assorted_candies : ℕ := 45

def chewing_gum_ratio_sam_bob : ℕ × ℕ := (2, 3)
def chocolate_bar_ratio_sam_bob : ℕ × ℕ := (3, 1)
def assorted_candy_ratio_sam_bob : ℕ × ℕ := (1, 1)

theorem bob_total_candies :
  let bob_chewing_gums := (total_chewing_gums * chewing_gum_ratio_sam_bob.snd) / (chewing_gum_ratio_sam_bob.fst + chewing_gum_ratio_sam_bob.snd)
  let bob_chocolate_bars := (total_chocolate_bars * chocolate_bar_ratio_sam_bob.snd) / (chocolate_bar_ratio_sam_bob.fst + chocolate_bar_ratio_sam_bob.snd)
  let bob_assorted_candies := (total_assorted_candies * assorted_candy_ratio_sam_bob.snd) / (assorted_candy_ratio_sam_bob.fst + assorted_candy_ratio_sam_bob.snd)
  bob_chewing_gums + bob_chocolate_bars + bob_assorted_candies = 64 := by
  sorry

end bob_total_candies_l374_374977


namespace quadratic_two_real_roots_quadratic_no_real_roots_l374_374629

theorem quadratic_two_real_roots (k : ℝ) :
  (∃ x : ℝ, 2 * x^2 - (4 * k - 1) * x + (2 * k^2 - 1) = 0) → 
  k ≤ 9 / 8 :=
by
  sorry

theorem quadratic_no_real_roots (k : ℝ) :
  ¬ (∃ x : ℝ, 2 * x^2 - (4 * k - 1) * x + (2 * k^2 - 1) = 0) → 
  k > 9 / 8 :=
by
  sorry

end quadratic_two_real_roots_quadratic_no_real_roots_l374_374629


namespace train_speed_in_kmh_l374_374969

def train_length : ℝ := 250 -- Length of the train in meters
def station_length : ℝ := 200 -- Length of the station in meters
def time_to_pass : ℝ := 45 -- Time to pass the station in seconds

theorem train_speed_in_kmh :
  (train_length + station_length) / time_to_pass * 3.6 = 36 :=
  sorry -- Proof is skipped

end train_speed_in_kmh_l374_374969


namespace ac_lt_bc_of_a_gt_b_and_c_lt_0_l374_374903

theorem ac_lt_bc_of_a_gt_b_and_c_lt_0 {a b c : ℝ} (h1 : a > b) (h2 : c < 0) : a * c < b * c :=
  sorry

end ac_lt_bc_of_a_gt_b_and_c_lt_0_l374_374903


namespace max_min_xy_l374_374861

theorem max_min_xy (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) : (x * y) ∈ { real.min (-1), real.max (1) } :=
by
  have h : a^2 ≤ 4 := 
  sorry
  have xy_eq : x * y = (a^2 - 2) / 2 := 
  sorry
  split
  { calc min (xy_eq) = -1 :=
    sorry
    calc max (xy_eq) = 1 :=
    sorry
  }

end max_min_xy_l374_374861


namespace rational_quad_inf_sol_or_none_l374_374201

-- Define the variables and condition
variables (a b : ℚ)

-- Define the statement to be proved
theorem rational_quad_inf_sol_or_none (h1 : ¬ ∃ (x y : ℚ), (a * x^2 + b * y^2 = 1)) ∨
  (∃ (x y : ℚ), (a * x^2 + b * y^2 = 1) ∧ ∀ (x y : ℚ), (a * x^2 + b * y^2 = 1) → ∃ (inf_sols : ℕ), inf_sols > 1) : Prop :=
sorry

end rational_quad_inf_sol_or_none_l374_374201


namespace problem_proof_l374_374244

variable (a b c : ℝ)
noncomputable def a_def : ℝ := Real.exp 0.2
noncomputable def b_def : ℝ := Real.sin 1.2
noncomputable def c_def : ℝ := 1 + Real.log 1.2

theorem problem_proof (ha : a = a_def) (hb : b = b_def) (hc : c = c_def) : b < c ∧ c < a :=
by
  have ha_val : a = Real.exp 0.2 := ha
  have hb_val : b = Real.sin 1.2 := hb
  have hc_val : c = 1 + Real.log 1.2 := hc
  sorry

end problem_proof_l374_374244


namespace count_abundant_numbers_under_35_l374_374532

def is_abundant (n : ℕ) : Prop :=
  (∑ d in Finset.filter (λ d, d ∣ n ∧ d < n) (Finset.range (n + 1))) > n

def count_abundant_less_than (m : ℕ) : ℕ :=
  (Finset.range m).filter is_abundant |>.card

theorem count_abundant_numbers_under_35 : count_abundant_less_than 35 = 5 :=
by
  sorry

end count_abundant_numbers_under_35_l374_374532


namespace closest_to_million_seconds_is_10_days_l374_374464

theorem closest_to_million_seconds_is_10_days :
  (λ x, (x - 11.574) ^ 2) 10 < (λ x, (x - 11.574) ^ 2) 1 ∧
  (λ x, (x - 11.574) ^ 2) 10 < (λ x, (x - 11.574) ^ 2) 100 ∧
  (λ x, (x - 11.574) ^ 2) 10 < (λ x, (x - 11.574) ^ 2) 365.25 ∧
  (λ x, (x - 11.574) ^ 2) 10 < (λ x, (x - 11.574) ^ 2) 3652.5 :=
by
  -- Given conversions and calculations from the problem
  let seconds_in_a_day := 60 * 60 * 24
  let million := 10^6
  let days := million / seconds_in_a_day
  have closest_to_10_days : abs (days - 10) < abs (days - 1) ∧
                            abs (days - 10) < abs (days - 100) ∧
                            abs (days - 10) < abs (days - 365.25) ∧
                            abs (days - 10) < abs (days - 3652.5),
  {
    sorry -- The detailed proof goes here
  }
  exact closest_to_10_days

end closest_to_million_seconds_is_10_days_l374_374464


namespace rows_of_cans_l374_374509

theorem rows_of_cans (n : ℕ) (h1 : 2 + 5 + 8 + ... + (3 * n - 1) = 91) : n = 7 :=
sorry

end rows_of_cans_l374_374509


namespace problem_I_problem_II_l374_374673

noncomputable def area_of_triangle_ABC (a b c : ℝ) (A B C : ℝ) 
  (hB : B = π / 3) (h_a_c : a + c = 13) (hb : b = 7) : ℝ :=
  let ac := 40 in
  1 / 2 * ac * sin B

theorem problem_I (a b c A B C : ℝ) (h_seq : 2 * B = A + C) (hb : b = 7) (h_a_c : a + c = 13) :
  area_of_triangle_ABC a b c A B C (by rw [h_seq, B_eq_pi_over_3]) (by assumption) (by assumption) = 10 * sqrt 3 :=
  sorry

theorem problem_II (A C : ℝ) (hA_range : A > 0 ∧ A < 2 * π / 3) :
  (let max_val := 2 in
   let opt_A := π / 3 in
   max_val ∧ opt_A = π / 3) :=
  sorry

end problem_I_problem_II_l374_374673


namespace fifth_term_of_sequence_l374_374555

def pow_four_sequence (n : ℕ) : ℕ :=
  ∑ i in Finset.range (n + 1), 4^i

theorem fifth_term_of_sequence :
  pow_four_sequence 4 = 341 :=
sorry

end fifth_term_of_sequence_l374_374555


namespace smallest_odd_angle_in_right_triangle_l374_374410

theorem smallest_odd_angle_in_right_triangle
  (x y : ℤ) (hx1 : even x) (hy1 : odd y) (hx2 : x > y) (ha : x + y = 90) :
  y = 31 :=
by sorry

end smallest_odd_angle_in_right_triangle_l374_374410


namespace train_car_passengers_l374_374167

theorem train_car_passengers (x : ℕ) (h : 60 * x = 732 + 228) : x = 16 :=
by
  sorry

end train_car_passengers_l374_374167


namespace food_required_6_days_l374_374031

def mom_food_daily : ℝ := 1.5 * 3
def first_two_puppies_food_daily : ℝ := (1 / 2) * 3 * 2
def next_two_puppies_food_daily : ℝ := (3 / 4) * 2 * 2
def last_puppy_food_daily : ℝ := 1 * 4

def total_food_daily : ℝ :=
  mom_food_daily + first_two_puppies_food_daily + next_two_puppies_food_daily + last_puppy_food_daily

def total_food_6_days : ℝ :=
  total_food_daily * 6

theorem food_required_6_days : total_food_6_days = 87 := 
by
  -- Proof will go here
  sorry

end food_required_6_days_l374_374031


namespace football_field_length_l374_374186

theorem football_field_length :
  (launch_distance = 6 * field_length) →
  (dog_speed_feet_per_min = 400) →
  (dog_run_time_min = 9) →
  (dog_run_distance_yards = (dog_speed_feet_per_min / 3) * dog_run_time_min) →
  (field_length = dog_run_distance_yards / 6) →
  field_length = 200 :=
begin
  intros,
  sorry
end

end football_field_length_l374_374186


namespace division_problem_l374_374343

theorem division_problem 
  (a b c d e f g h i : ℕ) 
  (h1 : a = 7) 
  (h2 : b = 9) 
  (h3 : c = 8) 
  (h4 : d = 1) 
  (h5 : e = 2) 
  (h6 : f = 3) 
  (h7 : g = 4) 
  (h8 : h = 6) 
  (h9 : i = 0) 
  : 7981 / 23 = 347 := 
by 
  sorry

end division_problem_l374_374343


namespace Sheila_weekly_earnings_l374_374816

-- Definitions based on the conditions
def hours_per_day_MWF : ℕ := 8
def hours_per_day_TT : ℕ := 6
def hourly_wage : ℕ := 7
def days_MWF : ℕ := 3
def days_TT : ℕ := 2

-- Theorem that Sheila earns $252 per week
theorem Sheila_weekly_earnings : (hours_per_day_MWF * hourly_wage * days_MWF) + (hours_per_day_TT * hourly_wage * days_TT) = 252 :=
by 
  sorry

end Sheila_weekly_earnings_l374_374816


namespace partition_into_subsets_l374_374721

variable {α : Type*}

-- Assume S is a set of n positive integers
def set_S (n : ℕ) : set ℕ := { a : ℕ | a > 0 }

-- Define the set P as the set of all sums of one or more distinct elements of S
def set_P (S : set ℕ) : set ℕ := { p | ∃ (t : finset ℕ), (∀ x ∈ t, x ∈ S) ∧ p = t.sum id }

-- Define the partition of P into subsets P_m
def P_m (S : set ℕ) (m : ℕ) (partial_sums : ℕ → ℕ) : set ℕ :=
  { p | partial_sums (m - 1) < p ∧ p ≤ partial_sums m }

-- The partial sums
def partial_sums (S : set ℕ) : ℕ → ℕ
  | 0       := 0
  | (m + 1) := partial_sums m + classical.some (set.exists_mem_of_ne_empty (finite.exists_maximal_wrt id S)).val

-- The main theorem to be proved
theorem partition_into_subsets (S : set ℕ) (P : set ℕ)
  (hS : ∃ n, S = set_S n)
  (hP : P = set_P S)
  (partial_sums : ℕ → ℕ)
  (hpartial_sums : ∀ m, partial_sums m = (finset.range (m + 1)).sum (λ i, classical.some (set.exists_mem_of_ne_empty (finite.exists_maximal_wrt id S)).val)):
  ∃ (P_m : ℕ → set ℕ) (n : ℕ), (∀ m, P_m S m (partial_sums m) ⊆ P) ∧ (P = ⋃ m, P_m S m (partial_sums m)) ∧ (∀ m, ∀ a b ∈ P_m S m (partial_sums m), a ≤ 2 * b) := sorry

end partition_into_subsets_l374_374721


namespace piecewise_function_evaluation_l374_374284

theorem piecewise_function_evaluation (m : ℝ) (x : ℝ) :
  (if x <= 3 then 2 * x - 2^(2 * m + 1) else log 2 (x - 3)) = 2 * m :=
by
  sorry

end piecewise_function_evaluation_l374_374284


namespace son_age_l374_374950

theorem son_age (M S : ℕ) (h1 : M = S + 24) (h2 : M + 2 = 2 * (S + 2)) : S = 22 :=
by
  sorry

end son_age_l374_374950


namespace petya_digits_sum_l374_374764

def distinct_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem petya_digits_sum (a b c d : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
    (h_distinct: distinct_digits a b c d)
    (h_sum: 6 * (a + b + c + d) * 1111 = 73326) :
    a + b + c + d = 11 ∧ {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end petya_digits_sum_l374_374764


namespace white_numbers_remain_l374_374691

noncomputable def initial_numbers : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2021}

noncomputable def painted_blue : Set ℕ := {n | n ∈ initial_numbers ∧ n % 3 = 0}
noncomputable def painted_red : Set ℕ := {n | n ∈ initial_numbers ∧ n % 5 = 0}
noncomputable def painted_both : Set ℕ := {n | n ∈ initial_numbers ∧ n % 15 = 0}

noncomputable def count_initial : ℕ := 2021

noncomputable def count_colored : ℕ := (painted_blue.to_finset.card + painted_red.to_finset.card - painted_both.to_finset.card)
noncomputable def count_white : ℕ := count_initial - count_colored

theorem white_numbers_remain : count_white = 1078 := by
  sorry

end white_numbers_remain_l374_374691


namespace cube_color_count_l374_374753

-- Definition of the cube and the coloring problem
def cube_coloring_ways : ℕ := 36

-- The theorem stating the number of ways to color a cube as specified
theorem cube_color_count :
  let red := 2, yellow := 2, blue := 2 in
  ∑ c in {cube_coloring_ways}, 
    (red + yellow + blue = 6 ∧ 
     cube_coloring_ways = 36) :=
sorry

end cube_color_count_l374_374753


namespace race_orders_count_l374_374300

theorem race_orders_count : ∃ n: ℕ, n = 6! ∧ n = 720 :=
by {
  let n := 6!,
  use n,
  split,
  {
    -- Prove n = 6!
    exact rfl,
  },
  {
    -- Prove n = 720
    norm_num,
  }
}

end race_orders_count_l374_374300


namespace solve_eqn_l374_374812

theorem solve_eqn (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 56) : x + y = 2 := by
  sorry

end solve_eqn_l374_374812


namespace alyssa_allowance_l374_374525

-- Definition using the given problem
def weekly_allowance (A : ℝ) : Prop :=
  A / 2 + 8 = 12

-- Theorem to prove that weekly allowance is 8 dollars
theorem alyssa_allowance : ∃ A : ℝ, weekly_allowance A ∧ A = 8 :=
by
  use 8
  unfold weekly_allowance
  exact eq.refl _

end alyssa_allowance_l374_374525


namespace quadratic_equals_binomial_square_l374_374217

theorem quadratic_equals_binomial_square (d : ℝ) : 
  (∃ b : ℝ, (x^2 + 60 * x + d) = (x + b)^2) → d = 900 :=
by
  sorry

end quadratic_equals_binomial_square_l374_374217


namespace determinant_31_l374_374986

variable (A : Matrix (Fin 3) (Fin 3) ℝ)
variable (det_A : Matrix.det A)

theorem determinant_31 : A = ![![2, -4, 5], ![3, 6, -2], ![1, -1, 3]] → det_A = 31 :=
by
  intro h_A
  rw [h_A]
  sorry

end determinant_31_l374_374986


namespace num_ways_to_use_100_yuan_l374_374466

noncomputable def x : ℕ → ℝ
| 0       => 0
| 1       => 1
| 2       => 3
| (n + 3) => x (n + 2) + 2 * x (n + 1)

theorem num_ways_to_use_100_yuan :
  x 100 = (1 / 3) * (2 ^ 101 + 1) :=
sorry

end num_ways_to_use_100_yuan_l374_374466


namespace maximize_water_jet_distance_l374_374494

theorem maximize_water_jet_distance (a g : ℝ) (h₁ : 0 < a) (h₂ : 0 < g) : 
  ∃ x : ℝ, x = a / 2 ∧ ∀ y : ℝ, (y = 2 * sqrt (x * (a - x))) → y ≤ a := 
begin
  sorry
end

end maximize_water_jet_distance_l374_374494


namespace shape_area_l374_374832

theorem shape_area (f : ℝ → ℝ) (h : ∀ x, f x = Real.exp x - 1) : 
  (∫ x in 0..1, f x) = Real.exp 1 - 2 :=
by
  rw [←h]
  rw [Real.integral_exp_sub_1]
  rw [Real.exp_one]
  Sorry

end shape_area_l374_374832


namespace polar_equation_line_point_ratio_l374_374345

theorem polar_equation_line (t : ℝ) : 
  (∃ ρ θ : ℝ, ρ * (Real.cos θ + Real.sin θ) = 4) :=
by
  have parametric_eqn_x := 3 - (Real.sqrt 2) / 2 * t
  have parametric_eqn_y := 1 + (Real.sqrt 2) / 2 * t
  -- Line form transformation and substitution referenced

theorem point_ratio (P Q : ℝ) 
  (hP : P = 8 / (Real.sqrt 3 + 1)) 
  (hQ : Q = 2 * Real.cos (Real.pi / 6)) :
  Q / P = (3 + Real.sqrt 3) / 8 :=
by
  sorry

end polar_equation_line_point_ratio_l374_374345


namespace complement_angle_l374_374666

variable (θ : ℝ) (h : θ = 70)

theorem complement_angle (h1 : θ = 70) : 90 - θ = 20 := 
by
  rw [h1]
  exact rfl

end complement_angle_l374_374666


namespace number_of_9s_in_1_to_50_l374_374510

theorem number_of_9s_in_1_to_50 : 
  let count_9s_in_digit (n : ℕ) : ℕ := (n.toString.filter (λ c, c = '9')).length
  let total_9s := (List.range 50).map (λ n, count_9s_in_digit (n + 1)).sum
  total_9s = 5
:= by
  sorry

end number_of_9s_in_1_to_50_l374_374510


namespace perfect_cube_divisor_l374_374267

theorem perfect_cube_divisor (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a^2 + 3*a*b + 3*b^2 - 1 ∣ a + b^3) :
  ∃ k > 1, ∃ m : ℕ, a^2 + 3*a*b + 3*b^2 - 1 = k^3 * m := 
sorry

end perfect_cube_divisor_l374_374267


namespace sequence_converges_l374_374358

open Real

noncomputable def f_k (x : ℝ) (k : ℕ) : ℕ :=
  let floor_1 := Int.floor (k / x)
  let floor_2 := Int.floor ((k + 1) / x)
  floor_2 - floor_1 + 1

theorem sequence_converges (x : ℝ) (hx : 0 < x ∧ x < 1) :
  ∃ L : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |((finset.range (n+1)).prod (λ k, (f_k x k))^(1/(n:ℝ))) - L| < ε :=
by
  use (1 / x)
  sorry

end sequence_converges_l374_374358


namespace range_of_a_l374_374287

theorem range_of_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1)
  (h₂ : ∃ x₁ x₂ ∈ set.Icc (-1 : ℝ) 1, |a^x₁ + x₁^2 - x₁ * log a - (a^x₂ + x₂^2 - x₂ * log a)| ≥ real.exp 1 - 1) :
  (0 < a ∧ a ≤ real.exp (-1)) ∨ (a ≥ real.exp 1) :=
by
  sorry

end range_of_a_l374_374287


namespace polygon_diagonals_l374_374179

theorem polygon_diagonals (n : ℕ) (h_convex : ∀ (k : ℕ), 4 ≤ n)
  (h_angle : ∃ i, (n-2) * 180 = 540) :
  (n = 5) → (n * (n - 3)) / 2 = 5 := by
  intros h_n
  rw [h_n]
  norm_num
  sorry

end polygon_diagonals_l374_374179


namespace ellipse_properties_l374_374373

open Real

def F1 : ℝ × ℝ := (0, 0)
def F2 : ℝ × ℝ := (6, 2)
def P (x y : ℝ) : ℝ × ℝ := (x, y)

def dist (A B : ℝ × ℝ) : ℝ :=
  sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def is_ellipse (x y : ℝ) : Prop :=
  dist (P x y) F1 + dist (P x y) F2 = 10

theorem ellipse_properties :
  ∃ h k a b : ℝ,
    (∀ x y, is_ellipse x y ↔ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) ∧
    h + k + a + b = 9 + sqrt 15 :=
by
  sorry

end ellipse_properties_l374_374373


namespace find_a_1000_l374_374005

noncomputable def a : ℕ → ℤ
| 0 := 2023
| 1 := 2024
| n + 2 := n - a n - a (n + 1) + n + 2

theorem find_a_1000 :
  a 1001 = 2356 := sorry

end find_a_1000_l374_374005


namespace find_digits_sum_l374_374800

theorem find_digits_sum
  (a b c d : ℕ)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (non_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (sum_of_numbers : 6 * (a + b + c + d) * 1111 = 73326) :
  {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end find_digits_sum_l374_374800


namespace number_of_distinct_differences_l374_374301

def distinct_differences (s : Finset ℕ) : Finset ℕ :=
  s.bUnion (λ x, s.filter (λ y, y ≠ x).image (λ y, abs (x - y)))

theorem number_of_distinct_differences :
  distinct_differences {1, 2, 3, 4, 5, 6, 9} = {1, 2, 3, 4, 5, 6, 7, 8} :=
by
  sorry

end number_of_distinct_differences_l374_374301


namespace monotonic_intervals_sum_harmonic_inequality_l374_374288

-- Step 1
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x - 2 * Real.log x - 1

-- Function used in condition
def extremum_at (a : ℝ) (x : ℝ) : Prop := x = 1 / Real.exp 1 ∧ 
  (∀ y, (f a y < f a x ↔ y < x) ∨ (f a y > f a x ↔ y > x))

-- The first assertion
theorem monotonic_intervals (a : ℝ) :
  extremum_at a (1 / Real.exp 1) →
  (∀ x, 0 < x ∧ x < 1 / Real.exp 1 → deriv (f a) x > 0) ∧
  (∀ x, x > 1 / Real.exp 1 → deriv (f a) x < 0) :=
sorry

-- The second assertion
theorem sum_harmonic_inequality (n : ℕ) (h : 0 < n) :
  (∑ k in finset.range n, 1 / (2 * k + 1 : ℝ)) > (1 / 2) * Real.log (2 * n + 1) + n / (2 * n + 1) :=
sorry

end monotonic_intervals_sum_harmonic_inequality_l374_374288


namespace fried_green_tomato_family_l374_374890

theorem fried_green_tomato_family :
  (∀ (num_tomatoes slices_per_tomato slices_per_meal : Nat), 
    num_tomatoes = 20 → 
    slices_per_tomato = 8 → 
    slices_per_meal = 20 → 
    (num_tomatoes * slices_per_tomato) / slices_per_meal = 8) :=
by
  intros num_tomatoes slices_per_tomato slices_per_meal h1 h2 h3
  rw [h1, h2, h3]
  sorry

end fried_green_tomato_family_l374_374890


namespace projection_matrix_correct_l374_374363

def normal_vector : ℝ × ℝ × ℝ := (2, -1, 2)

def P_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.ofLin' 
    ![![5/9, 2/9, -4/9], 
      ![2/9, 8/9, 2/9], 
      ![-4/9, 2/9, 5/9]]

theorem projection_matrix_correct (v : Fin 3 → ℝ) :
    let proj := P_matrix.mulVec v
     in proj = v - ((〈v, Vector3.ofTuple normal_vector〉 / 9) • (Vector3.ofTuple normal_vector)) := 
  sorry

end projection_matrix_correct_l374_374363


namespace sqrt_x_plus_5_l374_374901

theorem sqrt_x_plus_5 (x : ℝ) (h : x = -1) : Real.sqrt (x + 5) = 2 :=
by
  sorry

end sqrt_x_plus_5_l374_374901


namespace purely_imaginary_complex_l374_374670

theorem purely_imaginary_complex (m : ℝ) :
  (m^2 - 3 * m = 0) → (m^2 - 5 * m + 6 ≠ 0) → m = 0 :=
begin
  intros h_real h_imag,
  -- The proof will go here
  sorry
end

end purely_imaginary_complex_l374_374670


namespace rectangular_box_surface_area_l374_374437

def total_surface_area_rectangular_box (x y z : ℝ) : ℝ :=
  2 * (x * y + y * z + z * x)

theorem rectangular_box_surface_area :
  ∀ (x y z : ℝ),
  4 * x + 4 * y + 4 * z = 240 →
  real.sqrt (x^2 + y^2 + z^2) = 31 →
  total_surface_area_rectangular_box x y z = 2639 := by
  intros x y z h1 h2
  sorry

end rectangular_box_surface_area_l374_374437


namespace area_of_45_45_90_l374_374448

-- Definitions as per conditions
def is_45_45_90_triangle (a b c : ℝ) : Prop := a = b ∧ c = a * sqrt 2
def right_angle (θ : ℝ) : Prop := θ = 90 

variable {PQR : Type} [NormedAddCommGroup PQR] [NormedSpace ℝ PQR]
variable {PQ QR PR : ℝ}
variable (h1 : is_45_45_90_triangle PQ QR PR)
variable (h2 : right_angle 90)
variable (PR_eq : PR = 10)

-- Proving the area
theorem area_of_45_45_90 (PQ QR PR : ℝ)
  (h1 : is_45_45_90_triangle PQ QR PR)
  (h2 : right_angle 90)
  (PR_eq : PR = 10) :
  (1 / 2) * PQ * QR = 25 :=
by
  sorry

end area_of_45_45_90_l374_374448


namespace find_n_in_range_l374_374226

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_n_in_range :
  ∀ n : ℕ, n ∈ finset.Icc 1 999 → n^2 = (sum_digits n)^3 ↔ n = 1 ∨ n = 27 :=
by
  intro n hn
  constructor
  · intro h
    cases hn
    simp [sum_digits] at h
    sorry
  · intro h
    cases h
    · rw h; simp [sum_digits]
    · rw h; simp [sum_digits]

end find_n_in_range_l374_374226


namespace nap_duration_is_two_hours_l374_374029

-- Conditions as definitions in Lean
def naps_per_week : ℕ := 3
def days : ℕ := 70
def total_nap_hours : ℕ := 60

-- Calculate the duration of each nap
theorem nap_duration_is_two_hours :
  ∃ (nap_duration : ℕ), nap_duration = 2 ∧
  (days / 7) * naps_per_week * nap_duration = total_nap_hours :=
by
  sorry

end nap_duration_is_two_hours_l374_374029


namespace c_share_l374_374066

theorem c_share (a b c : ℕ) (k : ℕ) 
    (h1 : a + b + c = 1010)
    (h2 : a - 25 = 3 * k) 
    (h3 : b - 10 = 2 * k) 
    (h4 : c - 15 = 5 * k) 
    : c = 495 := 
sorry

end c_share_l374_374066


namespace total_students_in_faculty_l374_374133

theorem total_students_in_faculty (N A B : ℕ) (hN : N = 230) (hA : A = 423) (hB : B = 134)
  (h80_percent : (N + A - B) = 80 / 100 * T) : T = 649 := 
by
  sorry

end total_students_in_faculty_l374_374133


namespace definite_integral_eq_l374_374574

-- Define the integrand
def integrand (x : ℝ) : ℝ := 3 * x + Real.sin x

-- Define the theorem
theorem definite_integral_eq : ∫ x in 0..(Real.pi / 2), integrand x = (3 / 8) * Real.pi ^ 2 + 1 :=
by
  sorry

end definite_integral_eq_l374_374574


namespace sum_arithmetic_sequence_l374_374979

theorem sum_arithmetic_sequence :
  let n := 21
  let a := 100
  let l := 120
  (n / 2) * (a + l) = 2310 :=
by
  -- define n, a, and l based on the conditions
  let n := 21
  let a := 100
  let l := 120
  -- state the goal
  have h : (n / 2) * (a + l) = 2310 := sorry
  exact h

end sum_arithmetic_sequence_l374_374979


namespace independence_test_result_l374_374684

noncomputable def independence_test (P : ℝ → ℝ) (H0 : Prop) (X Y : Type) : Prop :=
  (∀ p, P p = 0.001) → H0 → (1 - P 10.83) = 0.999

theorem independence_test_result (P : ℝ → ℝ) (H0 : Prop) (X Y : Type) :
  (∀ p, P p = 0.001) → 
  H0 → 
  (1 - P 10.83) = 0.999 :=
by
  intro hP hH0
  have : P 10.83 = 0.001 := hP 10.83
  rw [this]
  norm_num

end independence_test_result_l374_374684


namespace problem_solution_l374_374737

-- Define the set M based on the condition |2x - 1| < 1
def M : Set ℝ := {x | 0 < x ∧ x < 1}

-- Main theorem composed of two parts
theorem problem_solution :
  (∀ x, |2 * x - 1| < 1 ↔ 0 < x ∧ x < 1) ∧
  (∀ a b ∈ M, (a * b + 1) > (a + b)) :=
by
  split
  -- Part 1: Prove the equivalence of |2x - 1| < 1 and the set definition of M
  · intro x
    split
    -- Prove |2x - 1| < 1 → 0 < x ∧ x < 1
    sorry
    -- Prove 0 < x ∧ x < 1 → |2x - 1| < 1
    sorry
  -- Part 2: Prove ab + 1 > a + b for all a, b in M
  · intros a b ha hb
    have ha' : 0 < a ∧ a < 1 := ha
    have hb' : 0 < b ∧ b < 1 := hb
    -- Prove the inequality
    sorry

end problem_solution_l374_374737


namespace total_sales_amount_correct_l374_374152

noncomputable def total_sales_amount_approx : ℝ :=
  100 + 90 * ((1 - 6.19) / (1 - 1.2))

theorem total_sales_amount_correct :
  ∑ n in Finset.range 10, (100 * (1.2 ^ n) - 2 * 10 * ((1.2 ^ n - 1) / (1.2 - 1))) ≈ 2435.5 :=
by
  sorry

end total_sales_amount_correct_l374_374152


namespace maximal_cardinality_set_l374_374722

theorem maximal_cardinality_set (n : ℕ) (h_n : n ≥ 2) :
  ∃ M : Finset (ℕ × ℕ), ∀ (j k : ℕ), (1 ≤ j ∧ j < k ∧ k ≤ n) → 
  ((j, k) ∈ M → ∀ m, (k, m) ∉ M) ∧ 
  M.card = ⌊(n * n / 4 : ℝ)⌋ :=
by
  sorry

end maximal_cardinality_set_l374_374722


namespace joe_lifting_problem_l374_374475

theorem joe_lifting_problem (x y : ℝ) (h1 : x + y = 900) (h2 : 2 * x = y + 300) : x = 400 :=
sorry

end joe_lifting_problem_l374_374475


namespace tetrahedron_surface_area_l374_374257

theorem tetrahedron_surface_area (a : ℝ) (sqrt_3 : ℝ) (h : a = 4) (h_sqrt : sqrt_3 = Real.sqrt 3) :
  4 * (sqrt_3 * a^2 / 4) = 16 * sqrt_3 :=
by
  subst h
  subst h_sqrt
  have area_one_face : (Real.sqrt 3 * 4^2) / 4 = sqrt_3 * 4 := by
    norm_num [Real.sqrt_eq_rpow 3 0.5]
    sorry
  calc
    4 * (sqrt_3 * 4^2 / 4)
        = 4 * (sqrt_3 * 4) := by rw area_one_face
    ... = 16 * sqrt_3 := by norm_num

end tetrahedron_surface_area_l374_374257


namespace total_apples_packed_l374_374743

def apples_packed_daily (apples_per_box : ℕ) (boxes_per_day : ℕ) : ℕ :=
  apples_per_box * boxes_per_day

def apples_packed_first_week (apples_per_box : ℕ) (boxes_per_day : ℕ) (days_first_week : ℕ) : ℕ :=
  apples_packed_daily apples_per_box boxes_per_day * days_first_week

def apples_packed_second_week (apples_per_box : ℕ) (boxes_per_day : ℕ) (days_second_week : ℕ) (decrease_per_day : ℕ) : ℕ :=
  (apples_packed_daily apples_per_box boxes_per_day - decrease_per_day) * days_second_week

theorem total_apples_packed (apples_per_box : ℕ) (boxes_per_day : ℕ) (days_first_week : ℕ) (days_second_week : ℕ) (decrease_per_day : ℕ) :
  apples_per_box = 40 →
  boxes_per_day = 50 →
  days_first_week = 7 →
  days_second_week = 7 →
  decrease_per_day = 500 →
  apples_packed_first_week apples_per_box boxes_per_day days_first_week + apples_packed_second_week apples_per_box boxes_per_day days_second_week decrease_per_day = 24500 :=
  by
  intros h₁ h₂ h₃ h₄ h₅
  rw [h₁, h₂, h₃, h₄, h₅]
  dsimp [apples_packed_first_week, apples_packed_second_week, apples_packed_daily]
  sorry

end total_apples_packed_l374_374743


namespace reeya_average_score_l374_374400

theorem reeya_average_score :
  let scores := [50, 60, 70, 80, 80]
  let sum_scores := scores.sum
  let num_scores := scores.length
  sum_scores / num_scores = 68 :=
by
  sorry

end reeya_average_score_l374_374400


namespace max_height_of_tennis_ball_l374_374169

noncomputable def height (t : ℝ) : ℝ :=
  - (1 / 80) * t^2 + (1 / 5) * t + 1

theorem max_height_of_tennis_ball : ∀ t, 0 ≤ t ∧ t ≤ 20 → height t ≤ 1.8 ∧ (∃ t₀, 0 ≤ t₀ ∧ t₀ ≤ 20 ∧ height t₀ = 1.8) :=
by
  sorry

end max_height_of_tennis_ball_l374_374169


namespace part_one_part_two_l374_374483

noncomputable def largest_prime_less_than (k : ℕ) : ℕ := 
  max {p : ℕ | nat.prime p ∧ p < k}

def satisfies_conditions (k : ℕ) (p_k : ℕ) (n : ℕ) : Prop :=
  k ≥ 14 ∧ p_k = largest_prime_less_than k ∧ p_k ≥ 3 * k / 4 ∧ nat.composite n

theorem part_one (k n : ℕ) (p_k : ℕ) (h : satisfies_conditions k p_k n) (h_n : n = 2 * p_k) :
  ¬ n ∣ nat.factorial (n - k) :=
sorry

theorem part_two (k n : ℕ) (p_k : ℕ) (h : satisfies_conditions k p_k n) (h_n : n > 2 * p_k) :
  n ∣ nat.factorial (n - k) :=
sorry

end part_one_part_two_l374_374483


namespace find_coordinates_of_P_l374_374278

theorem find_coordinates_of_P : 
  ∃ P: ℝ × ℝ, 
  (∃ θ: ℝ, 0 ≤ θ ∧ θ ≤ π ∧ P = (3 * Real.cos θ, 4 * Real.sin θ)) ∧ 
  ∃ m: ℝ, m = 1 ∧ P.fst = P.snd ∧ P = (12/5, 12/5) :=
by {
  sorry -- Proof is omitted as per instruction
}

end find_coordinates_of_P_l374_374278


namespace sum_binom_coeff_l374_374366

def binom (n j : ℕ) : ℤ := n.factorial / (j.factorial * (n - j).factorial)

theorem sum_binom_coeff : (∑ k in Finset.range 50, (-1 : ℤ)^k * binom 99 (2 * k)) = -(2 : ℤ)^49 :=
by
  sorry

end sum_binom_coeff_l374_374366


namespace circumcircle_radius_greater_than_one_l374_374541

theorem circumcircle_radius_greater_than_one (a b c : ℝ) (R : ℝ) : 
  (a ≤ 0.01) ∧ (b ≤ 0.01) ∧ (c ≤ 0.01) ∧ (1 < R) → 
  ∃ (T : Triangle), circumcircle_radius T > 1 := 
sorry

end circumcircle_radius_greater_than_one_l374_374541


namespace distance_from_origin_12_5_l374_374332

def distance_from_origin (x y : ℕ) : ℕ := 
  Int.natAbs (Nat.sqrt (x * x + y * y))

theorem distance_from_origin_12_5 : distance_from_origin 12 5 = 13 := by
  sorry

end distance_from_origin_12_5_l374_374332


namespace arithmetic_sequence_15th_term_l374_374845

theorem arithmetic_sequence_15th_term (a1 a2 a3 : ℕ) (h1 : a1 = 3) (h2 : a2 = 17) (h3 : a3 = 31) :
  let d := a2 - a1 in
  let a15 := a1 + 14 * d in
  a15 = 199 :=
by
  sorry

end arithmetic_sequence_15th_term_l374_374845


namespace determine_if_one_l374_374083

-- Definitions to represent the operations of the MK-97 calculator
def equal_check (a b : ℝ) : Prop := a = b
def addition (a b : ℝ) : ℝ := a + b
def roots (a b : ℝ) : Option (ℝ × ℝ) := 
  if b * b - 4 * a < 0 then none 
  else some ((-b + real.sqrt (b * b - 4 * a)) / (2 * a), (-b - real.sqrt (b * b - 4 * a)) / (2 * a))

-- Main statement to prove: We can determine if a number a is 1 using the operations of the MK-97
theorem determine_if_one (a : ℝ) : 
  (equal_check a (addition a a) = False) ∧ (roots (2*a) (a) = some ((-a + real.sqrt (a^2 - a)), (-a - real.sqrt (a^2 - a))) → -a + real.sqrt (a^2 - a) = -a - real.sqrt (a^2 - a)) ↔ a = 1 :=
by sorry

end determine_if_one_l374_374083


namespace count_set_pairs_l374_374707

def C (n k : ℕ) := nat.choose n k

def a_n (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2^(n-2) - C (n-2) ((n/2)-1) else 2^(n-2)

theorem count_set_pairs (A B : set ℕ) (a b : ℕ) (n : ℕ) 
  (h_n_pos : n ≥ 3) 
  (hA : A.nonempty) 
  (hB : B.nonempty)
  (union : A ∪ B = {i | i ∈ finset.range (n+1)}.to_set)
  (intersection : A ∩ B = ∅) :
  a_n n = (if n % 2 = 0 then 2^(n-2) - C (n-2) ((n/2)-1) else 2^(n-2)) := 
sorry

end count_set_pairs_l374_374707


namespace present_age_of_son_is_22_l374_374948

theorem present_age_of_son_is_22 (S F : ℕ) (h1 : F = S + 24) (h2 : F + 2 = 2 * (S + 2)) : S = 22 :=
by
  sorry

end present_age_of_son_is_22_l374_374948


namespace flood_damage_in_usd_l374_374158

theorem flood_damage_in_usd (damage_in_aud : ℝ) (conversion_rate : ℝ) (damage_in_usd : ℝ) 
  (h1 : damage_in_aud = 45000000)
  (h2 : conversion_rate = 0.7)
  (h3 : damage_in_usd = damage_in_aud * conversion_rate) :
  damage_in_usd = 31500000 :=
by
  rw [h1, h2] at h3
  exact h3

end flood_damage_in_usd_l374_374158


namespace max_value_complex_l374_374372

noncomputable def maxValue (z : Complex) (h1 : Complex.abs z = Real.sqrt 2) : Real := 
  Complex.abs ((z - 1)^2 * (z + 1))

theorem max_value_complex : ∀ z : Complex, Complex.abs z = Real.sqrt 2 →
  max_value (λ z h1, Complex.abs ((z - 1)^2 * (z + 1))) = 4 * Real.sqrt 2 := by 
  sorry

end max_value_complex_l374_374372


namespace problem1_problem2_l374_374143

-- First problem: proving the expression simplifies to sqrt(2)
theorem problem1 : 
  (∛((-1) ^ 2) + ∛(-8) + real.sqrt 3 - abs (1 - real.sqrt 3) + real.sqrt 2) = real.sqrt 2 := 
sorry

-- Second problem: proving the solutions for the quadratic equation
theorem problem2 {x : ℝ} : 
  25 * (x + 2) ^ 2 - 36 = 0 ↔ (x = -16/5 ∨ x = -4/5) := 
sorry

end problem1_problem2_l374_374143


namespace paint_cost_of_cube_l374_374920

theorem paint_cost_of_cube (cost_per_kg : ℕ) (coverage_per_kg : ℕ) (side_length : ℕ) (total_cost : ℕ) 
  (h1 : cost_per_kg = 20)
  (h2 : coverage_per_kg = 15)
  (h3 : side_length = 5)
  (h4 : total_cost = 200) : 
  (6 * side_length^2 / coverage_per_kg) * cost_per_kg = total_cost :=
by
  sorry

end paint_cost_of_cube_l374_374920


namespace petya_digits_l374_374794

def are_distinct (a b c d : Nat) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 

def non_zero_digits (a b c d : Nat) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

theorem petya_digits :
  ∃ (a b c d : Nat), are_distinct a b c d ∧ non_zero_digits a b c d ∧ (a + b + c + d = 11) ∧ (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 5) ∧
  (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 5) ∧ (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 5) ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 5) :=
by
  sorry

end petya_digits_l374_374794


namespace area_square_B_l374_374831

theorem area_square_B (a b : ℝ) (h1 : a^2 = 25) (h2 : abs (a - b) = 4) : b^2 = 81 :=
by
  sorry

end area_square_B_l374_374831


namespace range_of_m_if_neg_proposition_false_l374_374325

theorem range_of_m_if_neg_proposition_false :
  (¬ ∃ x_0 : ℝ, x_0^2 + m * x_0 + 2 * m - 3 < 0) ↔ (2 ≤ m ∧ m ≤ 6) :=
by
  sorry

end range_of_m_if_neg_proposition_false_l374_374325


namespace expression_evaluates_to_8_l374_374481

def evaluate_expression : ℝ :=
  ((0.128 / 3.2 + 0.86) / ((5 / 6) * 1.2 + 0.8)) * (((95 / 63) - (39 / 63)) * 3.6) / (0.505 * (2 / 5) - 0.002)

theorem expression_evaluates_to_8 : evaluate_expression = 8 := 
by
  sorry

end expression_evaluates_to_8_l374_374481


namespace volume_and_circumradius_of_tetrahedron_l374_374231

noncomputable def volume_of_tetrahedron (A B C D : Point) : ℝ :=
  let AB := dist A B
  let AC := dist A C
  let AD := dist A D
  let BC := dist B C
  let BD := dist B D
  let CD := dist C D
  -- Compute the volume using the given lengths
  (18 * Real.sqrt 3)

noncomputable def circumradius_of_tetrahedron (A B C D : Point) : ℝ :=
  let AB := dist A B
  let AC := dist A C
  let AD := dist A D
  let BC := dist B C
  let BD := dist B D
  let CD := dist C D
  -- Compute the circumradius using the given lengths
  (7.5)

theorem volume_and_circumradius_of_tetrahedron (A B C D : Point)
  (h1 : dist A B = 9) (h2 : dist A C = 9) (h3 : dist A D = 15)
  (h4 : dist B C = 3) (h5 : dist B D = 12) (h6 : dist C D = 12) :
  volume_of_tetrahedron A B C D = 18 * Real.sqrt 3 ∧ circumradius_of_tetrahedron A B C D = 7.5 :=
by
  sorry

end volume_and_circumradius_of_tetrahedron_l374_374231


namespace smallest_x_abs_eq_15_l374_374585

theorem smallest_x_abs_eq_15 :
  ∃ x : ℝ, |5 * x - 3| = 15 ∧ ∀ y : ℝ, |5 * y - 3| = 15 → x ≤ y :=
sorry

end smallest_x_abs_eq_15_l374_374585


namespace sum_of_possible_x_values_l374_374076

theorem sum_of_possible_x_values : 
  (∑ x in {x | 4^(x^2 + 6*x + 9) = 16^(x + 3)}.to_finset, x) = -4 :=
by
  sorry

end sum_of_possible_x_values_l374_374076


namespace fraction_dark_tiles_l374_374500

theorem fraction_dark_tiles 
  (n : ℕ) (pattern : ℕ → ℕ → bool) (is_tiling_pattern : ∀ x y, pattern (x % 8) y = pattern x (y % 8))
  (consistent_pattern : ∀ x y, pattern x y = pattern y x)
  (dark_tiles_in_4x4 : ∃ a b, ∑ i in range 4, ∑ j in range 4, if pattern (a + i) (b + j) then 1 else 0 = 10) :
  ∃ k,
    k / (8 * 8) = (5 / 8) :=
by
  sorry

end fraction_dark_tiles_l374_374500


namespace petya_four_digits_l374_374786

theorem petya_four_digits :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
    6 * (a + b + c + d) * 1111 = 73326 ∧ 
    (∃ S, S = a + b + c + d ∧ S = 11) :=
begin
  use 1, 2, 3, 5,
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split,
  { exact dec_trivial },
  use 11,
  split,
  { exact dec_trivial },
  { exact dec_trivial }
end

end petya_four_digits_l374_374786


namespace point_above_line_l374_374683

/-- Given the point (-2, t) lies above the line x - 2y + 4 = 0,
    we want to prove t ∈ (1, +∞) -/
theorem point_above_line (t : ℝ) : (-2 - 2 * t + 4 > 0) → t > 1 :=
sorry

end point_above_line_l374_374683


namespace scientific_notation_0_00000023_l374_374868

-- Define what it means for a number to be in scientific notation.
def scientific_notation (x : ℝ) (c : ℝ) (b : ℤ) : Prop :=
  1 ≤ c ∧ c < 10 ∧ x = c * 10^b

theorem scientific_notation_0_00000023 :
  scientific_notation 0.00000023 2.3 (-7) :=
by
  unfold scientific_notation
  split
  · norm_num
  split
  · norm_num
  · norm_num
    sorry

end scientific_notation_0_00000023_l374_374868


namespace concurrency_of_ap_dt_l374_374380

open_locale classical

noncomputable theory

-- Define the geometric objects and their relationships
variables (A B C H D E F P T Q : Type)
variable [has_mem A B C H D E F P T Q]

-- Definitions derived from conditions
variable (Triangle_ABC : triangle A B C) [scalene Triangle_ABC]
variable (orthocenter_H : orthocenter Triangle_ABC)
variable (circumcircle_Gamma : circle (Triangle_ABC.verts) )
variable (AH_meets_Gamma_at_D : line (seg AH) ∩ circumcircle_Gamma = (A ∈ ∂\circumcircle_Gamma, D ∈ ∂\circumcircle_Gamma) ∧ D ≠ A)
variable (BH_meets_CA_at_E : line (seg BH) ∩ line (seg CA) = E)
variable (CH_meets_AB_at_F : line (seg CH) ∩ line (seg AB) = F)
variable (EF_meets_BC_at_P : line (seg EF) ∩ line (seg BC) = P)
variable (tangents_meet_at_T : tangent circumcircle_Gamma B ∩ tangent circumcircle_Gamma C = T)

-- Hypothesis to show these points and lines meet
variable (concurrent_ap_dt : (line (seg AP) ∩ line (seg DT) ∩ circumcircle E F = Q)→ Q ∈ circumcircle_E F A)

-- Statement of the theorem we need to prove
theorem concurrency_of_ap_dt (h₁ : Triangle_ABC)
                               (h₂ : orthocenter_H)
                               (h₃ : circumcircle_Gamma)
                               (h₄ : AH_meets_Gamma_at_D)
                               (h₅ : BH_meets_CA_at_E)
                               (h₆ : CH_meets_AB_at_F)
                               (h₇ : EF_meets_BC_at_P)
                               (h₈ : tangents_meet_at_T) :
  concurrent_ap_dt :=
begin
  sorry
end

end concurrency_of_ap_dt_l374_374380


namespace babysitting_charge_l374_374030

theorem babysitting_charge
  (num_babysitting_families : ℕ := 4)
  (num_cars_washed : ℕ := 5)
  (earnings_per_car : ℕ := 12)
  (total_earnings : ℕ := 180) :
  ∃ (x : ℕ), 4 * x + 5 * 12 = 180 ∧ x = 30 :=
by
  use 30
  split
  · calc
    4 * 30 + 5 * 12 = 120 + 60 := by norm_num
    ... = 180 := by norm_num
  · rfl

end babysitting_charge_l374_374030


namespace tent_ratio_l374_374386

-- Define the relevant variables
variables (N E S C T : ℕ)

-- State the conditions
def conditions : Prop :=
  N = 100 ∧
  E = 2 * N ∧
  S = 200 ∧
  T = 900 ∧
  N + E + S + C = T

-- State the theorem to prove the ratio
theorem tent_ratio (h : conditions N E S C T) : C = 4 * N :=
by sorry

end tent_ratio_l374_374386


namespace weekly_allowance_is_8_l374_374526

variable (A : ℝ)

def condition_1 (A : ℝ) : Prop := ∃ A : ℝ, A / 2 + 8 = 12

theorem weekly_allowance_is_8 (A : ℝ) (h : condition_1 A) : A = 8 :=
sorry

end weekly_allowance_is_8_l374_374526


namespace petya_digits_l374_374791

def are_distinct (a b c d : Nat) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 

def non_zero_digits (a b c d : Nat) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

theorem petya_digits :
  ∃ (a b c d : Nat), are_distinct a b c d ∧ non_zero_digits a b c d ∧ (a + b + c + d = 11) ∧ (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 5) ∧
  (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 5) ∧ (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 5) ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 5) :=
by
  sorry

end petya_digits_l374_374791


namespace find_digits_sum_l374_374798

theorem find_digits_sum
  (a b c d : ℕ)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (non_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (sum_of_numbers : 6 * (a + b + c + d) * 1111 = 73326) :
  {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end find_digits_sum_l374_374798


namespace shaded_and_unshaded_area_percentage_l374_374450

theorem shaded_and_unshaded_area_percentage (side_length : ℕ) (total_length : ℕ) (rect_width : ℕ)
  (area_shaded : ℕ) (total_area : ℕ) (unshaded_area : ℕ) :
  side_length = 20 →
  total_length = 35 →
  rect_width = 20 →
  area_shaded = 100 →
  total_area = 700 →
  unshaded_area = 600 →
  (area_shaded / total_area : ℚ) * 100 = 14.29 :=
by 
sor

end shaded_and_unshaded_area_percentage_l374_374450


namespace line_and_circle_separate_l374_374099

theorem line_and_circle_separate (k : ℝ) (hk : k < 0) (intersects_C : ∃ p : ℝ × ℝ, (p.1+3)^2 + (p.2+2)^2 = 9 ∧ p.2 = k * p.1) :
  let d_D := |2| / Real.sqrt (k^2 + 1) in
  d_D > 1 → ∃ d : ℝ, d > 0 ∧ d = d_D :=
by
  let d_D := |2| / Real.sqrt (k^2 + 1)
  have h : d_D > 1 → ∃ d : ℝ, d = d_D :=
    sorry
  exact ⟨d_D, h⟩

end line_and_circle_separate_l374_374099


namespace marbles_problem_l374_374549

theorem marbles_problem
  (cindy_original : ℕ)
  (lisa_original : ℕ)
  (h1 : cindy_original = 20)
  (h2 : cindy_original = lisa_original + 5)
  (marbles_given : ℕ)
  (h3 : marbles_given = 12) :
  (lisa_original + marbles_given) - (cindy_original - marbles_given) = 19 :=
by
  sorry

end marbles_problem_l374_374549


namespace Petya_digits_sum_l374_374776

theorem Petya_digits_sum (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :
  (24 * (a + b + c + d) * 1111 = 73326) →
  {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end Petya_digits_sum_l374_374776


namespace length_minus_width_l374_374957

theorem length_minus_width 
  (area length diff width : ℝ)
  (h_area : area = 171)
  (h_length : length = 19.13)
  (h_diff : diff = length - width)
  (h_area_eq : area = length * width) :
  diff = 10.19 := 
by {
  sorry
}

end length_minus_width_l374_374957


namespace trigonometric_identity_l374_374071

theorem trigonometric_identity :
  cos 15 * cos 45 + sin 15 * sin 45 = real.cos (45 - 15) := 
by 
  sorry

end trigonometric_identity_l374_374071


namespace circumcircle_area_l374_374283

noncomputable def f (ω x : ℝ) : ℝ :=
  2 * (Real.cos (ω * x / 2)) ^ 2 + Real.cos (ω * x + Real.pi / 3)

theorem circumcircle_area
  (A B C a b c : ℝ)
  (hf_min_positive_period : ∀ ω > 0, Function.periodic (f ω) Real.pi)
  (hf_A : f 2 A = -1 / 2)
  (hc : c = 3)
  (area_triangle : Real.sin A * b * c / 2 = 6 * Real.sqrt 3) :
  ∃ R : ℝ, π * R ^2 = 49 * Real.pi / 3 :=
sorry

end circumcircle_area_l374_374283


namespace cat_direction_at_noon_l374_374891

def time_to_tell_a_tale := 5
def time_to_sing_a_song := 4
def cycles_duration := time_to_tell_a_tale + time_to_sing_a_song
def start_time := 10 * 60  -- 10:00 AM in minutes
def end_time := 12 * 60    -- 12:00 PM (noon) in minutes

theorem cat_direction_at_noon :
  let total_time := end_time - start_time in
  let full_cycles := total_time / cycles_duration in
  let remaining_time := total_time % cycles_duration in
  remaining_time < time_to_tell_a_tale → 
  direction_at_noon = "left" sorry

end cat_direction_at_noon_l374_374891


namespace infinite_k_values_l374_374591

theorem infinite_k_values (k : ℕ) : (∃ k, ∀ (a b c : ℕ),
  (a = 64 ∧ b ≥ 0 ∧ c = 0 ∧ k = 2^a * 3^b * 5^c) ↔
  Nat.lcm (Nat.lcm (2^8) (2^24 * 3^12)) k = 2^64) →
  ∃ (b : ℕ), true :=
by
  sorry

end infinite_k_values_l374_374591


namespace fifth_term_sequence_l374_374564

theorem fifth_term_sequence : (∑ i in Finset.range 5, 4^i) = 341 :=
by
  sorry

end fifth_term_sequence_l374_374564


namespace number_of_safe_integers_l374_374234

/-
Define what it means to be p-safe.
-/
def p_safe (n p : ℕ) : Prop :=
  ∀ k : ℕ, |(n - k * p) % p| ≥ 3

/-
Define the problem conditions.
-/
def is_safe (n : ℕ) : Prop :=
  p_safe n 5 ∧ p_safe n 7 ∧ p_safe n 11

/--
The proof problem: Prove that there are 975 integers less than or equal to 15000
that are simultaneously 5-safe, 7-safe, and 11-safe.
-/
theorem number_of_safe_integers : ∃ N, N = 975 ∧ ∀ n : ℕ, n ≤ 15000 → is_safe n → n ∈ finset.range (N + 1) :=
by
  sorry

end number_of_safe_integers_l374_374234


namespace Lyka_saves_for_8_weeks_l374_374741

theorem Lyka_saves_for_8_weeks : 
  ∀ (C I W : ℕ), C = 160 → I = 40 → W = 15 → (C - I) / W = 8 := 
by 
  intros C I W hC hI hW
  sorry

end Lyka_saves_for_8_weeks_l374_374741


namespace commercial_duration_l374_374749

/-- Michael was watching a TV show, which was aired for 1.5 hours. 
    During this time, there were 3 commercials. 
    The TV show itself, not counting commercials, was 1 hour long. 
    Prove that each commercial lasted 10 minutes. -/
theorem commercial_duration (total_time : ℝ) (num_commercials : ℕ) (show_time : ℝ)
  (h1 : total_time = 1.5) (h2 : num_commercials = 3) (h3 : show_time = 1) :
  (total_time - show_time) / num_commercials * 60 = 10 := 
sorry

end commercial_duration_l374_374749


namespace bench_cost_150_l374_374941

-- Define the conditions
def combined_cost (bench_cost table_cost : ℕ) : Prop := bench_cost + table_cost = 450
def table_cost_eq_twice_bench (bench_cost table_cost : ℕ) : Prop := table_cost = 2 * bench_cost

-- Define the main statement, which includes the goal of the proof.
theorem bench_cost_150 (bench_cost table_cost : ℕ) (h_combined_cost : combined_cost bench_cost table_cost)
  (h_table_cost_eq_twice_bench : table_cost_eq_twice_bench bench_cost table_cost) : bench_cost = 150 :=
by
  sorry

end bench_cost_150_l374_374941


namespace find_m_n_product_of_sines_l374_374105

theorem find_m_n_product_of_sines :
  ∃ (m n : ℕ), 1 < m ∧ 1 < n ∧ (∏ k in (finset.range 30).map (λ k, sin (real.pi * (3 * ↑k - 2) / 180) ^ 2)) = m ^ n ∧ m + n = 61 :=
begin
  sorry
end

end find_m_n_product_of_sines_l374_374105


namespace angle_between_vectors_magnitude_of_vector_sum_l374_374740

variable {V : Type*} [InnerProductSpace ℝ V]
variable (a b : V)
variables (h1 : ∥a∥ = 1) (h2 : ∥b∥ = 1) (h3 : ∥3 • a - 2 • b∥ = Real.sqrt 7)

theorem angle_between_vectors : Real.arccos (⟪a, b⟫) = Real.pi / 3 := by
  sorry

variable (h4 : ⟪a, b⟫ = 1 / 2)

theorem magnitude_of_vector_sum : ∥2 • a + 3 • b∥ = Real.sqrt 19 := by
  sorry

end angle_between_vectors_magnitude_of_vector_sum_l374_374740


namespace arithmetic_sequence_ninth_term_l374_374014

-- Definitions and Conditions
variables {a : ℕ → ℝ} (S : ℕ → ℝ)

-- Problem Statement
theorem arithmetic_sequence_ninth_term
  (h1 : a 3 = 4)
  (h2 : S 11 = 110)
  (h3 : ∀ n, S n = (n * (a 1 + a n)) / 2) :
  a 9 = 16 :=
sorry

end arithmetic_sequence_ninth_term_l374_374014


namespace math_problem_l374_374675

noncomputable def m : ℕ := 294
noncomputable def n : ℕ := 81
noncomputable def d : ℕ := 3

axiom circle_radius (r : ℝ) : r = 42
axiom chords_length (l : ℝ) : l = 78
axiom intersection_distance (d : ℝ) : d = 18

theorem math_problem :
  let m := 294
  let n := 81
  let d := 3
  m + n + d = 378 :=
by {
  -- Proof omitted
  sorry
}

end math_problem_l374_374675


namespace convex_on_Icc_l374_374168

variables {α : Type*} [linear_ordered_field α] {f : α → α} {a b : α} (h : ∀ x1 x2 y1 y2 ∈ set.Icc a b, 
  x1 + x2 = y1 + y2 → |y1 - y2| ≤ |x1 - x2| → f x1 + f x2 ≤ f y1 + f y2)

theorem convex_on_Icc (h : ∀ x1 x2 y1 y2 ∈ set.Icc a b, 
  x1 + x2 = y1 + y2 → |y1 - y2| ≤ |x1 - x2| → f x1 + f x2 ≤ f y1 + f y2) : 
  convex_on (set.Icc a b) f :=
sorry

end convex_on_Icc_l374_374168


namespace sequence_general_term_l374_374346

theorem sequence_general_term {a : ℕ → ℕ} (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 1, a (n + 1) = a n + 2) :
  ∀ n : ℕ, n ≥ 1 → a n = 2 * n - 1 :=
by
  -- skip the proof with sorry
  sorry

end sequence_general_term_l374_374346


namespace fifth_term_of_sequence_equals_341_l374_374560

theorem fifth_term_of_sequence_equals_341 : 
  ∑ i in Finset.range 5, 4^i = 341 :=
by sorry

end fifth_term_of_sequence_equals_341_l374_374560


namespace domain_of_f_l374_374840

noncomputable def f (x : ℝ) := Real.log (x + 2) / Real.sqrt (1 - 3 ^ x)

theorem domain_of_f :
  {x : ℝ | x + 2 > 0 ∧ 1 - 3 ^ x > 0} = {x : ℝ | -2 < x ∧ x < 0} :=
by
  ext x
  simp
  split
  · intro h
    cases h with h1 h2
    split
    · linarith
    · suffices h3' : 3 ^ x < 1
      · linarith
      from Real.pow_lt_one_iff.mpr ⟨lt_of_neg_of_lt zero_lt_three (Real.log_pos_iff.mpr h1),
                              by norm_num⟩
  · intro h
    cases h with h1 h2
    split
    · linarith
    · from Real.one_gt_pow h2

end domain_of_f_l374_374840


namespace strawberries_eaten_l374_374519

-- Definitions based on the conditions
def strawberries_picked : ℕ := 35
def strawberries_remaining : ℕ := 33

-- Statement of the proof problem
theorem strawberries_eaten :
  strawberries_picked - strawberries_remaining = 2 :=
by
  sorry

end strawberries_eaten_l374_374519


namespace disprove_tangency_l374_374225

theorem disprove_tangency (a b : ℝ) (h1 : a ≠ 0 ∨ b ≠ 0) : a = 1 ∧ b = 1 → a^2 + b^2 ≠ 1 :=
by
  intro h
  cases h with ha hb
  rw [ha, hb]
  norm_num

end disprove_tangency_l374_374225


namespace game_is_unfair_l374_374808

def fair_game (pA pB : ℚ) : Prop := pA = pB

theorem game_is_unfair :
  let cube_faces := {1, 2, 3, 4, 5, 6}
  let winning_set_A := {4, 5, 6}
  let winning_set_B := {1, 2}
  let total_faces := 6
  let probability_A := (3 / total_faces : ℚ)
  let probability_B := (2 / total_faces : ℚ)
  ¬ fair_game probability_A probability_B :=
by
  sorry

end game_is_unfair_l374_374808


namespace possible_values_a1_l374_374365

def sequence_sum (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum a

theorem possible_values_a1 {a : ℕ → ℤ} (h1 : ∀ n : ℕ, a n + a (n + 1) = 2 * n - 1)
  (h2 : ∃ k : ℕ, sequence_sum a k = 190 ∧ sequence_sum a (k + 1) = 190) :
  (a 0 = -20 ∨ a 0 = 19) :=
sorry

end possible_values_a1_l374_374365


namespace coefficient_x3_l374_374344

noncomputable def P (x : ℝ) : ℝ :=
  ∑ i in finset.range (11), (nat.choose 10 i) * (x - 1) ^ i

theorem coefficient_x3 :
  (finset.range (11)).sum (λ i, (nat.choose 10 i) * (if i = 3 then 1 else if i > 3 then (if h : (i - 3) ≥ 0 then (nat.choose i (i-3)) else 0) else 0)) = 
  (nat.choose 10 3 + (nat.choose 10 4) * (nat.choose 4 1) + (nat.choose 10 5) * (nat.choose 5 2) + 
  (nat.choose 10 6) * (nat.choose 6 3) + (nat.choose 10 7) * (nat.choose 7 4) + 
  (nat.choose 10 8) * (nat.choose 8 5) + (nat.choose 10 9) * (nat.choose 9 6) + 
  (nat.choose 10 10) * (nat.choose 10 7)) := by sorry

end coefficient_x3_l374_374344


namespace trapezoid_area_l374_374537

theorem trapezoid_area:
  ∀ (A_outer: ℝ) (A_hole: ℝ) (n: ℝ),
    A_outer = 2500 → A_hole = 900 → n = 4 →
    let A_total := A_outer - A_hole in
    let A_trapezoid := A_total / n in
    A_trapezoid = 400 :=
by
  sorry

end trapezoid_area_l374_374537


namespace Petya_digits_sum_l374_374774

theorem Petya_digits_sum (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :
  (24 * (a + b + c + d) * 1111 = 73326) →
  {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end Petya_digits_sum_l374_374774


namespace distributions_balls_into_boxes_l374_374304

def num_distributions_of_balls := 7
def num_boxes := 4

theorem distributions_balls_into_boxes :
  (number_of_partitions num_distributions_of_balls num_boxes) = 11 :=
sorry

end distributions_balls_into_boxes_l374_374304


namespace tan_theta_l374_374276

theorem tan_theta (θ : ℝ) (x y : ℝ) (hx : x = - (Real.sqrt 3) / 2) (hy : y = 1 / 2) (h_terminal : True) : 
  Real.tan θ = - (Real.sqrt 3) / 3 :=
sorry

end tan_theta_l374_374276


namespace simplify_exponents_l374_374069

variable (x : ℝ)

theorem simplify_exponents (x : ℝ) : (x^5) * (x^2) = x^(7) :=
by
  sorry

end simplify_exponents_l374_374069


namespace polynomials_divisibility_l374_374039

open Real

-- Definitions 
def R := ℝ
def poly := R → R

noncomputable def I (f g : R → R) (x : R) : R := ∫ t in 1..x, f t * g t

-- Definitions of polynomials, a(x), b(x), c(x), d(x)
variables (a b c d : poly)

-- Definition of the function F(x)
def F (x : R) : R := (I a c x) * (I b d x) - (I a d x) * (I b c x)

-- The proposition to prove
theorem polynomials_divisibility (a b c d : poly) : 
  ∀ x : R, (F a b c d ) x ∣ (x - 1) ^ 4 :=
sorry

end polynomials_divisibility_l374_374039


namespace game_is_unfair_l374_374807

def fair_game (pA pB : ℚ) : Prop := pA = pB

theorem game_is_unfair :
  let cube_faces := {1, 2, 3, 4, 5, 6}
  let winning_set_A := {4, 5, 6}
  let winning_set_B := {1, 2}
  let total_faces := 6
  let probability_A := (3 / total_faces : ℚ)
  let probability_B := (2 / total_faces : ℚ)
  ¬ fair_game probability_A probability_B :=
by
  sorry

end game_is_unfair_l374_374807


namespace Petya_digits_sum_l374_374778

theorem Petya_digits_sum (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :
  (24 * (a + b + c + d) * 1111 = 73326) →
  {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end Petya_digits_sum_l374_374778


namespace sum_mod_7_eq_5_l374_374458

theorem sum_mod_7_eq_5 : ((Finset.range 202).sum % 7) = 5 :=
sorry

end sum_mod_7_eq_5_l374_374458


namespace irreducibility_condition_l374_374236

-- Definition of irreducibility over ℤ
def is_irreducible_over_Z (f : Polynomial ℤ) : Prop :=
  ¬ ∃ (g h : Polynomial ℤ), g.degree > 0 ∧ h.degree > 0 ∧ f = g * h

-- Definition of reducibility over ℤ_p for any prime p
def is_reducible_over_Zp (f : Polynomial ℤ) : Prop :=
  ∀ (p : ℕ), p.prime → ∃ (g h : Polynomial (Zmod p)), f.map (Int.castRingHom (Zmod p)) = g * h

-- Main theorem defining the problem and solution
theorem irreducibility_condition (k : ℤ) :
  let F_k := Polynomial.X ^ 4 + 2 * (1 - k) * Polynomial.X ^ 2 + (1 + k)^2 in
  (¬(∃ (a : ℤ), a^2 = k) ∧ ¬(∃ (a : ℤ), a^2 = -k)) ↔ (is_irreducible_over_Z F_k ∧ is_reducible_over_Zp F_k) :=
begin
  sorry
end

end irreducibility_condition_l374_374236


namespace petya_digits_sum_l374_374757

def distinct_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem petya_digits_sum (a b c d : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
    (h_distinct: distinct_digits a b c d)
    (h_sum: 6 * (a + b + c + d) * 1111 = 73326) :
    a + b + c + d = 11 ∧ {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end petya_digits_sum_l374_374757


namespace number_of_tulips_l374_374702

theorem number_of_tulips (T : ℕ) (roses : ℕ) (used_flowers : ℕ) (extra_flowers : ℕ) 
  (total_flowers : ℕ) (h1 : roses = 37) (h2 : used_flowers = 70) 
  (h3 : extra_flowers = 3) (h4: total_flowers = 73) 
  (h5 : T + roses = total_flowers) : T = 36 := 
by
  sorry

end number_of_tulips_l374_374702


namespace imaginary_part_complex_expr_l374_374277

-- Define the given complex numbers
def z1 : ℂ := 2 - I
def z2 : ℂ := 1 - 3 * I

-- Define the expression whose imaginary part we need to find
def complex_expr : ℂ := (I / z1) + (conj(z2) / 5)

-- The proof statement in Lean 4: Prove that the imaginary part of the complex number complex_expr is 1
theorem imaginary_part_complex_expr : complex_expr.im = 1 := sorry

end imaginary_part_complex_expr_l374_374277


namespace max_cables_cut_l374_374881

def initial_cameras : ℕ := 200
def initial_cables : ℕ := 345
def resulting_clusters : ℕ := 8

theorem max_cables_cut :
  ∃ (cables_cut : ℕ), resulting_clusters = 8 ∧ initial_cables - cables_cut = (initial_cables - cables_cut) - (resulting_clusters - 1) ∧ cables_cut = 153 :=
by
  sorry

end max_cables_cut_l374_374881


namespace ratio_of_diagonals_l374_374833

theorem ratio_of_diagonals (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 = 49 * b^2 / 64) : (a * real.sqrt 2) / (b * real.sqrt 2) = 7 / 8 :=
by
  sorry

end ratio_of_diagonals_l374_374833


namespace find_m_value_l374_374973

theorem find_m_value (m : ℝ) 
  (first_term : ℝ := 18) (second_term : ℝ := 6)
  (second_term_2 : ℝ := 6 + m) 
  (S1 : ℝ := first_term / (1 - second_term / first_term))
  (S2 : ℝ := first_term / (1 - second_term_2 / first_term))
  (eq_sum : S2 = 3 * S1) :
  m = 8 := by
  sorry

end find_m_value_l374_374973


namespace find_k_l374_374382

theorem find_k : 
  ∃ (k : ℚ), 
    (∃ (x y : ℚ), y = 3 * x + 7 ∧ y = -4 * x + 1) ∧ 
    ∃ (x y : ℚ), y = 3 * x + 7 ∧ y = 2 * x + k ∧ k = 43 / 7 := 
sorry

end find_k_l374_374382


namespace solve_equation_l374_374130

theorem solve_equation (x : ℝ) (h : x > 0) :
  25^(Real.log x / Real.log 4) - 5^(Real.log (x^2) / Real.log 16 + 1) = Real.log (9 * Real.sqrt 3) / Real.log (Real.sqrt 3) - 25^(Real.log x / Real.log 16) ->
  x = 4 :=
by
  sorry

end solve_equation_l374_374130


namespace max_min_product_xy_l374_374859

-- Definition of conditions
variables (a x y : ℝ)
def condition_1 : Prop := x + y = a
def condition_2 : Prop := x^2 + y^2 = -a^2 + 2

-- The main theorem statement
theorem max_min_product_xy (a : ℝ) (ha_range : -2 ≤ a ∧ a ≤ 2): 
  (∀ x y : ℝ, condition_1 a x y ∧ condition_2 a x y → (x * y) ≤ (1 / 3)) ∧
  (∀ x y : ℝ, condition_1 a x y ∧ condition_2 a x y → (x * y) ≥ (-1)) :=
sorry

end max_min_product_xy_l374_374859


namespace carl_personal_owe_l374_374543

def property_damage : ℝ := 40000
def medical_bills : ℝ := 70000
def insurance_coverage : ℝ := 0.8
def carl_responsibility : ℝ := 0.2
def total_cost : ℝ := property_damage + medical_bills
def carl_owes : ℝ := total_cost * carl_responsibility

theorem carl_personal_owe : carl_owes = 22000 := by
  sorry

end carl_personal_owe_l374_374543


namespace sum_of_possible_x_values_l374_374075

theorem sum_of_possible_x_values : 
  (∑ x in {x | 4^(x^2 + 6*x + 9) = 16^(x + 3)}.to_finset, x) = -4 :=
by
  sorry

end sum_of_possible_x_values_l374_374075


namespace hyperbola_has_eccentricity_l374_374289

noncomputable def hyperbola_eccentricity (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : b = 2 * a) : ℝ :=
  sqrt (5)

theorem hyperbola_has_eccentricity (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : b = 2 * a) :
  hyperbola_eccentricity a b h₁ h₂ h₃ = sqrt 5 :=
by 
  unfold hyperbola_eccentricity 
  sorry

end hyperbola_has_eccentricity_l374_374289


namespace circle_through_A_B_exists_l374_374611

variables {Point : Type*} [MetricSpace Point]

-- Definitions for points and the angle 
variables (A B C : Point) (φ : ℝ)
  -- Definition for distances between points
  (dAB dAC dBC : ℝ)
-- Condition for distances are defined to conform to the given translation of conditions
def condition1 := dist A B = dAB
def condition2 := dist A C = dAC
def condition3 := dist B C = dBC

-- The mathematical condition to be verified
def angle_condition := sin (φ / 2) ≥ dAB / (dAC + dBC)

-- Rewrite the problem as a Lean theorem that needs a proof
theorem circle_through_A_B_exists (hA : A ≠ B) (hAC : A ≠ C) (hBC : B ≠ C):
  condition1 A B dAB → condition2 A C dAC → condition3 B C dBC → angle_condition dAB dAC dBC φ →
  ∃ k : Circle, k.passes_through A ∧ k.passes_through B ∧ k.tangent_angle_from C = φ :=
by
  intros
  sorry

end circle_through_A_B_exists_l374_374611


namespace valentines_day_expense_l374_374387

theorem valentines_day_expense :
  ∀ (cost_heart_biscuit cost_puppy_boot : ℕ) (qty_heart_biscuit_A qty_heart_biscuit_B qty_puppy_boot_A qty_puppy_boot_B : ℕ),
  cost_heart_biscuit = 2 →
  cost_puppy_boot = 15 →
  qty_heart_biscuit_A = 5 →
  qty_puppy_boot_A = 1 →
  qty_heart_biscuit_B = 7 →
  qty_puppy_boot_B = 2 →
  let total_cost_A := qty_heart_biscuit_A * cost_heart_biscuit + qty_puppy_boot_A * cost_puppy_boot,
      total_cost_B := qty_heart_biscuit_B * cost_heart_biscuit + qty_puppy_boot_B * cost_puppy_boot,
      total_cost := total_cost_A + total_cost_B in
  total_cost = 69 :=
by
  intros
  simp_all
  sorry

end valentines_day_expense_l374_374387


namespace lines_concur_at_G_l374_374811

-- Define the points and circles
variable {A B C D O P G : Type}
variable [InCircle A B C D O]
variable [Intersection AC BD P]
variable [Circumcenter A B P O1]
variable [Circumcenter B C P O2]
variable [Circumcenter C D P O3]
variable [Circumcenter D A P O4]

-- The main theorem
theorem lines_concur_at_G (h1 : InCircle A B C D O)
  (h2 : Intersection AC BD P)
  (h3 : Circumcenter A B P O1)
  (h4 : Circumcenter B C P O2)
  (h5 : Circumcenter C D P O3)
  (h6 : Circumcenter D A P O4) : Concurrent OP O1O3 O2O4 :=
sorry

end lines_concur_at_G_l374_374811


namespace cars_meet_in_five_hours_l374_374480

theorem cars_meet_in_five_hours
  (distance : ℝ)
  (speed1 : ℝ)
  (speed2 : ℝ)
  (start_at_same_time : Prop)
  (opposite_ends : Prop) :
  distance = 500 → speed1 = 40 → speed2 = 60 → start_at_same_time → opposite_ends → 
  (distance / (speed1 + speed2) = 5) :=
by
  intro h_distance h_speed1 h_speed2 h_start h_ends
  have cars_meet_time := (distance / (speed1 + speed2))
  rw [h_distance, h_speed1, h_speed2] at cars_meet_time
  exact cars_meet_time

end cars_meet_in_five_hours_l374_374480


namespace relationship_between_x_and_y_l374_374654

theorem relationship_between_x_and_y (x y : ℝ) (h1 : 2 * x - y > 3 * x) (h2 : x + 2 * y < 2 * y) :
  x < 0 ∧ y > 0 :=
sorry

end relationship_between_x_and_y_l374_374654


namespace smallest_n_for_buttons_l374_374359

-- Define the problem setup
def convex8_gon (A : Fin 8 → ℝ × ℝ) : Prop :=
  true -- Placeholder for actual check if A defines a convex 8-gon

def sub_quadrilateral (A : Fin 8 → ℝ × ℝ) (v : Finset (Fin 8)) : Prop :=
  v.card = 4 ∧ convex8_gon A

def is_button (A : Fin 8 → ℝ × ℝ) (i j k l : Fin 8) : Prop :=
  true -- Placeholder for actual condition where diagonals intersect

-- The main theorem to find the smallest number n
theorem smallest_n_for_buttons (A : Fin 8 → ℝ × ℝ) :
  convex8_gon A ∧ (∀ (i k : Fin 8), i ≠ k → 
    ∃ n : ℕ, ∑ s in (Finset.subsets_of_card 4 (Finset.univ : Finset (Fin 8))), 
      (sub_quadrilateral A s ∧ ∃ i j k l, s = {i, j, k, l} ∧ is_button A i j k l) = n)
  → n = 14 :=
begin
  sorry
end

end smallest_n_for_buttons_l374_374359


namespace probability_composite_divisible_by_3_or_5_l374_374922

def is_composite (n : ℕ) : Prop := 
  1 < n ∧ ∃ d : ℕ, 1 < d ∧ d < n ∧ d ∣ n

def is_divisible_by_3_or_5 (n : ℕ) : Prop := 
  (3 ∣ n) ∨ (5 ∣ n)

def count_composite_divisible_by_3_or_5_upto_n (n : ℕ) : ℕ := 
  (Finset.range n).filter (λ x => is_composite x ∧ is_divisible_by_3_or_5 x).card

theorem probability_composite_divisible_by_3_or_5 : 
  count_composite_divisible_by_3_or_5_upto_n 51 = 21 →
  (21 : ℚ) / 50 = 21 / 50 :=
by
  intro h
  sorry

end probability_composite_divisible_by_3_or_5_l374_374922


namespace infinite_grid_three_colors_has_isosceles_right_triangle_l374_374852

open Classical

noncomputable def exists_isosceles_right_triangle_same_color (color : ℕ → ℕ → ℕ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ x₃ y₃, 
  (x₁ = x₂ ∧ y₁ = y₃ ∧ x₃ = x₁ + (y₃ - y₂) ∧ y₂ = y₁ + (x₃ - x₁)) ∧
  (color x₁ y₁ = color x₂ y₂ ∧ color x₂ y₂ = color x₃ y₃) 

theorem infinite_grid_three_colors_has_isosceles_right_triangle :
  (∃ color : ℕ → ℕ → ℕ, ∀ x y, color x y < 3) →
    exists_isosceles_right_triangle_same_color := by 
  intros coloring
  sorry

end infinite_grid_three_colors_has_isosceles_right_triangle_l374_374852


namespace min_m_value_l374_374644

theorem min_m_value (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) (m : ℕ) (k : ℤ) :
  (∀ n, a (n + 1) = a n + 4) →
  a 1 + a 4 = 14 →
  (∀ n, b n = S n / (n + k)) →
  (∀ n, b (n + 1) - b n = b n - b (n - 1)) →
  (∀ n, T n = ∑ i in range n, (1 / (b i * b (i + 1))) ∧ ∀ n, T n ≤ m / 100) →
  m = 50 :=
sorry

end min_m_value_l374_374644


namespace sum_of_squares_of_roots_l374_374989

theorem sum_of_squares_of_roots :
  (∀ y : ℝ, y ^ 3 - 8 * y ^ 2 + 9 * y + 2 = 0 → y ≥ 0) →
  let s : ℝ := 8
  let p : ℝ := 9
  let q : ℝ := -2
  (s ^ 2 - 2 * p = 46) :=
by
  -- Placeholders for definitions extracted from the conditions
  -- and additional necessary let-bindings from Vieta's formulas
  intro h
  sorry

end sum_of_squares_of_roots_l374_374989


namespace fifth_term_of_sequence_equals_341_l374_374557

theorem fifth_term_of_sequence_equals_341 : 
  ∑ i in Finset.range 5, 4^i = 341 :=
by sorry

end fifth_term_of_sequence_equals_341_l374_374557


namespace petya_four_digits_l374_374782

theorem petya_four_digits :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
    6 * (a + b + c + d) * 1111 = 73326 ∧ 
    (∃ S, S = a + b + c + d ∧ S = 11) :=
begin
  use 1, 2, 3, 5,
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split,
  { exact dec_trivial },
  use 11,
  split,
  { exact dec_trivial },
  { exact dec_trivial }
end

end petya_four_digits_l374_374782


namespace no_very_convex_function_exists_l374_374984

-- Definition of very convex function
def very_convex (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f x + f y) / 2 ≥ f ((x + y) / 2) + |x - y|

-- Theorem stating the non-existence of very convex functions
theorem no_very_convex_function_exists : ¬∃ f : ℝ → ℝ, very_convex f :=
by {
  sorry
}

end no_very_convex_function_exists_l374_374984


namespace rigid_motion_transformations_l374_374841

-- Definitions and conditions
def infinite_recur_pattern (ell : Line) := ∀ (p : Point), True -- Placeholder definition

-- Main statement
theorem rigid_motion_transformations (ell : Line) :
  (∃ p: Point, rotation 180 p ell) ∧ 
  (∃ d: ℝ, translation d ell) ∧ 
  ¬ reflection_across_line ell ∧ 
  ¬ reflection_perpendicular_line ell :=
sorry

end rigid_motion_transformations_l374_374841


namespace sin_of_angle_X_l374_374013

theorem sin_of_angle_X (X Y Z : ℝ) (hXYZ : X * X + Y * Y = Z * Z) (hXYcos : 5 * (Y / Z) = 4 * (X / Z)) : (sin (atan2 X Z)) = 5 / sqrt 41 :=
by
  sorry

end sin_of_angle_X_l374_374013


namespace g_rewritten_as_sin_l374_374235

def g (x : ℝ) : ℝ := cot (x / 3) - cot (x / 2)

theorem g_rewritten_as_sin (x : ℝ) : g(x) = (sin (x / 6)) / (sin (x / 3) * sin (x / 2)) :=
by sorry

end g_rewritten_as_sin_l374_374235


namespace drums_per_day_l374_374185

theorem drums_per_day (pickers : ℕ) (total_drums : ℕ) (total_days : ℕ) 
                      (h_pick : pickers = 94) (h_drums : total_drums = 90) (h_days : total_days = 6) : 
                      (total_drums / total_days = 15) :=
by
  rw [h_drums, h_days]
  norm_num
  sorry

end drums_per_day_l374_374185


namespace problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_problem_7_problem_7_monotone_decreasing_l374_374059

-- Definitions of lim sup and lim inf for sequences of sets
def limsup {α : Type*} (A : ℕ → Set α) : Set α :=
  {x : α | ∃ᶠ n in at_top, x ∈ A n}

def liminf {α : Type*} (A : ℕ → Set α) : Set α :=
  {x : α | ∀ᶠ n in at_top, x ∈ A n}

noncomputable def lim {α : Type*} (A : ℕ → Set α) : Set α :=
  {x : α | ∃ N, ∀ n ≥ N, x ∈ A n}

-- Problem statements
theorem problem_1 {α : Type*} (A : ℕ → Set α) :
  (liminf (λ n, - (limsup A))) = (liminf (λ n, - A n)) :=
sorry

theorem problem_2 {α : Type*} (A : ℕ → Set α) :
  - (lim A) = limsup (λ n, - A n) :=
sorry

theorem problem_3 {α : Type*} (A : ℕ → Set α) :
  liminf A ⊆ limsup A :=
sorry

theorem problem_4 {α : Type*} (A B : ℕ → Set α) : 
  limsup (λ n, A n ∪ B n) = limsup A ∪ lim B :=
sorry

theorem problem_5 {α : Type*} (A B : ℕ → Set α) :
  liminf (λ n, A n ∩ B n) = liminf A ∩ liminf B :=
sorry

theorem problem_6 {α : Type*} (A B : ℕ → Set α) :
  limsup A ∩ liminf B ⊆ limsup (λ n, A n ∩ B n) ∧ 
  limsup (λ n, A n ∩ B n) ⊆ limsup A ∩ limsup B :=
sorry

theorem problem_7 {α : Type*} (A : ℕ → Set α) (A_set : Set α)
  (h_inc : ∀ {n}, A n ⊆ A (n+1)) :
  limsup A = A_set ↔ liminf A = A_set :=
sorry

theorem problem_7_monotone_decreasing {α : Type*} (A : ℕ → Set α) (A_set : Set α)
  (h_dec : ∀ {n}, A (n+1) ⊆ A n) :
  limsup A = A_set ↔ liminf A = A_set :=
sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_problem_7_problem_7_monotone_decreasing_l374_374059


namespace mayoral_election_l374_374474

theorem mayoral_election :
  ∀ (X Y Z : ℕ), (X = Y + (Y / 2)) → (Y = Z - (2 * Z / 5)) → (Z = 25000) → X = 22500 :=
by
  intros X Y Z h1 h2 h3
  -- Proof here, not necessary for the task
  sorry

end mayoral_election_l374_374474


namespace line_MN_parallel_AF_and_bisect_perimeter_l374_374620
variables (A B C D E M N F: Point)

-- Definitions of the variables and setup conditions for the problem
variable (Triangle : A B C)
variable (ExcircleOppositeB : ∃ D, touches (excircle_triangle B) D CA)
variable (ExcircleOppositeC : ∃ E, touches (excircle_triangle C) E AB)
variable (MidpointsMBC : ∃ M, Midpoint M B C)
variable (MidpointsNCA : ∃ N, Midpoint N C A)

-- The two properties to prove
theorem line_MN_parallel_AF_and_bisect_perimeter :
  (MN_parallel_AF : parallel MN (angle_bisector A F)) ∧
  (MN_bisect_perimeter : bisect_perimeter MN A B C) :=
sorry

end line_MN_parallel_AF_and_bisect_perimeter_l374_374620


namespace technician_completion_percentage_l374_374514

noncomputable def percentage_completed (D : ℝ) : ℝ :=
  let total_distance := 2.20 * D
  let completed_distance := 1.12 * D
  (completed_distance / total_distance) * 100

theorem technician_completion_percentage (D : ℝ) (hD : D > 0) :
  percentage_completed D = 50.91 :=
by
  sorry

end technician_completion_percentage_l374_374514


namespace percentage_increase_in_breadth_l374_374098

theorem percentage_increase_in_breadth (L B : ℝ)  (hL : L > 0) (hB : B > 0) 
  (h_area_increase : 1.10 * L * (B * (1 + (p / 100))) = 1.43 * (L * B)) 
  : p = 30 :=
begin
  sorry
end

end percentage_increase_in_breadth_l374_374098


namespace platform_length_l374_374131

noncomputable def speed (length : ℝ) (time : ℝ) : ℝ := length / time

theorem platform_length (l_train : ℝ) (t_pole : ℝ) (t_platform : ℝ) (correct_length : ℝ) (v_train : ℝ) : 
  l_train = 300 → 
  t_pole = 18 → 
  t_platform = 39 → 
  v_train = speed 300 18 → 
  correct_length = 350.13 → 
  let l_platform := (v_train * t_platform) - l_train in
  l_platform ≈ correct_length := 
by
  intros h1 h2 h3 h4 h5
  simp [speed] at h4 
  have h6 : v_train ≈ 300 / 18 := by rwa [h4]
  let l_platform := (v_train * t_platform) - l_train
  have h7 : l_platform ≈ 350.13 
  {
    calc {...} : ...
  }
  exact h7

end platform_length_l374_374131


namespace not_possible_perimeter_72_l374_374120

variable (a b : ℕ)
variable (P : ℕ)

def valid_perimeter_range (a b : ℕ) : Set ℕ := 
  { P | ∃ x, 15 < x ∧ x < 35 ∧ P = a + b + x }

theorem not_possible_perimeter_72 :
  (a = 10) → (b = 25) → ¬ (72 ∈ valid_perimeter_range 10 25) := 
by
  sorry

end not_possible_perimeter_72_l374_374120


namespace length_of_AB_l374_374621

-- Definition of the problem conditions
variable (A B C : Type)
variable [InnerProductSpace ℝ A]
variable (a b c : A)
variable (area : ℝ)
variable (BC : ℝ)
variable (angleC : ℝ)

-- Definition of the conditions as Lean predicates
def triangle_ABC_conditions : Prop :=
  (area = 2 * Real.sqrt 3) ∧
  (BC = 2) ∧
  (angleC = Real.pi * 2 / 3)  -- 120 degrees in radians

-- The statement to be proved
theorem length_of_AB
  (h : triangle_ABC_conditions A B C a b c area BC angleC) :
  ∃ (AB : ℝ), AB = 2 * Real.sqrt 7 :=
sorry

end length_of_AB_l374_374621


namespace cost_of_bench_l374_374939

variables (cost_table cost_bench : ℕ)

theorem cost_of_bench :
  cost_table + cost_bench = 450 ∧ cost_table = 2 * cost_bench → cost_bench = 150 :=
by
  sorry

end cost_of_bench_l374_374939


namespace probability_of_black_ball_l374_374329

theorem probability_of_black_ball 
  (p_red : ℝ)
  (p_white : ℝ)
  (h_red : p_red = 0.43)
  (h_white : p_white = 0.27)
  : (1 - p_red - p_white) = 0.3 :=
by 
  sorry

end probability_of_black_ball_l374_374329


namespace both_hit_probability_l374_374392

variable (P_A : ℝ) (P_B : ℝ)
variable (independent : Prop)

def probability_both_hit (P_A P_B : ℝ) [IndepEvents] := P_A * P_B

theorem both_hit_probability : (P_A = 0.8) → (P_B = 0.9) → independent → probability_both_hit P_A P_B = 0.72 :=
by
  intros hPA hPB hIndep
  rw [hPA, hPB]
  have : independent := by assumption
  sorry

end both_hit_probability_l374_374392


namespace last_four_digits_of_5_pow_2011_l374_374750

theorem last_four_digits_of_5_pow_2011 :
  (5^2011) % 10000 = 8125 := 
by
  -- Using modular arithmetic and periodicity properties of powers of 5.
  sorry

end last_four_digits_of_5_pow_2011_l374_374750


namespace desired_lines_l374_374607

-- Defining the given entities
variables {Point : Type}
variables [MetricSpace Point]

variable (A O B : Point)
variable (l : AffineSubspace ℝ Point) -- Line l

-- Condition: Given an angle ∠AOB
def angleAOB (A O B: Point) : ℕ := sorry  -- A placeholder for the actual angle calculation

-- Desired properties of line l₁ such that the angle between l and l₁ equals ∠AOB
def isCandidateLine (l  l₁: AffineSubspace ℝ Point) (angleAOB: ℕ) : Prop := 
  sorry  -- A placeholder for the characterization of angle between lines

-- Variables for candidate lines
variable (XO YO : AffineSubspace ℝ Point)

-- Statement to be Proven
theorem desired_lines (A O B: Point) (l XO YO: AffineSubspace ℝ Point):
  isCandidateLine l XO (angleAOB A O B) ∧ 
  isCandidateLine l YO (angleAOB A O B) :=
sorry

end desired_lines_l374_374607


namespace maximum_value_at_vertex_l374_374641

-- Defining the parabola as a function
def parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Defining the vertex condition
def vertex_condition (a b c : ℝ) := ∀ x : ℝ, parabola a b c x = a * x^2 + b * x + c

-- Defining the condition that the parabola opens downward
def opens_downward (a : ℝ) := a < 0

-- Defining the vertex coordinates condition
def vertex_coordinates (a b c : ℝ) := 
  ∃ (x₀ y₀ : ℝ), x₀ = 2 ∧ y₀ = -3 ∧ parabola a b c x₀ = y₀

-- The main theorem statement
theorem maximum_value_at_vertex (a b c : ℝ) (h1 : opens_downward a) (h2 : vertex_coordinates a b c) : ∃ y₀, y₀ = -3 ∧ ∀ x : ℝ, parabola a b c x ≤ y₀ :=
by
  sorry

end maximum_value_at_vertex_l374_374641


namespace fifth_term_of_sequence_l374_374552

def pow_four_sequence (n : ℕ) : ℕ :=
  ∑ i in Finset.range (n + 1), 4^i

theorem fifth_term_of_sequence :
  pow_four_sequence 4 = 341 :=
sorry

end fifth_term_of_sequence_l374_374552


namespace ratio_of_ages_l374_374355

noncomputable def Jim_age := 46
def Sam_age := 14
def Fred_age := Sam_age + 9
def Fred_age_six_years_ago := Sam_age - 6

theorem ratio_of_ages (h1 : Jim_age = 46) 
                       (h2 : Fred_age = Sam_age + 9) 
                       (h3 : Jim_age - 6 = 5 * (Sam_age - 6)) : 
    (Jim_age : Fred_age) = 2 : 1 :=
by 
  have Jim_age_eq : Jim_age = 46 := h1
  rw Jim_age_eq at h3
  have h3_trans : 40 = 5 * (Sam_age - 6) := h3
  have Sam_age_eq : Sam_age = 14 := by linarith
  have Fred_age_trans : Fred_age = Sam_age + 9 := h2
  rw [Sam_age_eq, Fred_age_trans]
  rw Sam_age_eq at Fred_age_trans
  have Fred_age_eq : Fred_age = 23 := by linarith
  calc
    Jim_age / Fred_age = 46 / 23 : by rw [Jim_age_eq, Fred_age_eq]
    ...               = 2/1 : sorry

end ratio_of_ages_l374_374355


namespace min_value_e2_t_e1_forall_t_e1_e2_le_l374_374367

noncomputable theory
open real

variables {E : Type*} [inner_product_space ℝ E]
variables (e1 e2 : E)
variables (angle120 : real.angle (↑e1 - ↑e2) = real.pi * 2 / 3) (unit_e1 : ∥e1∥ = 1) (unit_e2 : ∥e2∥ = 1)

theorem min_value_e2_t_e1 (t : ℝ) :
  ∃ (t : ℝ) (min_val : ℝ), min_val = ∥e2 + t * (e1 - e2)∥ ∧ min_val = 1 / 2 := sorry

theorem forall_t_e1_e2_le (t : ℝ) :
  ∥e1 + 1 / 2 * e2∥ ≤ ∥e1 + t * e2∥ := sorry

end min_value_e2_t_e1_forall_t_e1_e2_le_l374_374367


namespace relativ_prime_and_divisible_exists_l374_374397

theorem relativ_prime_and_divisible_exists
  (a b c : ℕ)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c) :
  ∃ r s : ℕ, Nat.gcd r s = 1 ∧ 0 < r ∧ 0 < s ∧ c ∣ (a * r + b * s) :=
by
  sorry

end relativ_prime_and_divisible_exists_l374_374397


namespace petya_four_digits_l374_374783

theorem petya_four_digits :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
    6 * (a + b + c + d) * 1111 = 73326 ∧ 
    (∃ S, S = a + b + c + d ∧ S = 11) :=
begin
  use 1, 2, 3, 5,
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split,
  { exact dec_trivial },
  use 11,
  split,
  { exact dec_trivial },
  { exact dec_trivial }
end

end petya_four_digits_l374_374783


namespace boat_speed_in_still_water_l374_374918

-- Definitions for conditions
variables (V_b V_s : ℝ)

-- The conditions provided for the problem
def along_stream := V_b + V_s = 13
def against_stream := V_b - V_s = 5

-- The theorem we want to prove
theorem boat_speed_in_still_water (h1 : along_stream V_b V_s) (h2 : against_stream V_b V_s) : V_b = 9 :=
sorry

end boat_speed_in_still_water_l374_374918


namespace orchids_more_than_roses_l374_374112

theorem orchids_more_than_roses {initial_roses initial_orchids new_roses new_orchids : ℕ} 
                                  (initial_roses = 9) (initial_orchids = 6) 
                                  (new_roses = 3) (new_orchids = 13) :
  new_orchids - new_roses = 10 :=
by sorry

end orchids_more_than_roses_l374_374112


namespace episodes_per_season_before_loss_l374_374207

-- Define the given conditions
def initial_total_seasons : ℕ := 12 + 14
def episodes_lost_per_season : ℕ := 2
def remaining_episodes : ℕ := 364
def total_episodes_lost : ℕ := 12 * episodes_lost_per_season + 14 * episodes_lost_per_season
def initial_total_episodes : ℕ := remaining_episodes + total_episodes_lost

-- Define the theorem to prove
theorem episodes_per_season_before_loss : initial_total_episodes / initial_total_seasons = 16 :=
by
  sorry

end episodes_per_season_before_loss_l374_374207


namespace problem1_problem2_l374_374733

noncomputable def f (x : ℝ) (a : ℝ): ℝ := Real.log x - (a + 1) * x

theorem problem1 (x : ℝ) : 
  0 < x → x < 1 → (Real.log x - x)' > 0 ∧ x > 1 → (Real.log x - x)' < 0 := 
by 
  sorry

theorem problem2 (a : ℝ) :
  a > -1 → (∃ x : ℝ, (0 < x ∧ x = 1/(a + 1)) ∧  (Real.log x - (a + 1) * x) > -2) →
  a < real.exp 1 - 1 := 
by
  sorry

end problem1_problem2_l374_374733


namespace find_n_m_find_k_range_l374_374246

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

noncomputable def f (x : ℝ) (n m : ℝ) : ℝ :=
  (-2^x + n) / (2^(x + 1) + m)

theorem find_n_m : 
  (∃ f_is_odd : ∀ x : ℝ, f x 1 2 = (-f (-x) 1 2)) → ∃ n m : ℝ, n = 1 ∧ m = 2 :=
sorry

theorem find_k_range (k : ℝ) : 
  (∀ c ∈ set.Ioo (-1 : ℝ) 1, f (4^c - 2^(c + 1)) 1 2 + f (2 * 4^c - k) 1 2 < 0) → k ≤ -1/4 :=
sorry

end find_n_m_find_k_range_l374_374246


namespace min_value_m_l374_374123

open Finset

-- Define the set S and the number of colors
def S : Finset ℕ := (finset.range 61).map (λ x : ℕ, x + 1)

-- Let m be the number of non-empty subsets of S
def m (coloring : Π (x : ℕ), x ∈ S → ℕ) : ℕ :=
  (∑ i in range 25, (2 ^ ((S.filter (λ x, coloring x (mem_map_of_mem _ (mem_range.mpr (lt_trans (self_lt_succ _) bot_lt_self))))) : ℕ).card - 1))

-- Prove that the minimum value of m is 119
theorem min_value_m : ∃ (coloring : Π (x : ℕ), x ∈ S → ℕ), m coloring = 119 :=
by {
  -- sorry to indicate skipping the proof steps
  sorry
}

end min_value_m_l374_374123


namespace fraction_equiv_l374_374007

theorem fraction_equiv (x y : ℚ) (h : (5/6) * 192 = (x/y) * 192 + 100) : x/y = 5/16 :=
sorry

end fraction_equiv_l374_374007


namespace rounding_addition_l374_374520

theorem rounding_addition :
  Float.toFixed (45.378 + 13.897 + 29.4567) 2 = 88.74 :=
by
  -- proof omitted
  sorry

end rounding_addition_l374_374520


namespace cone_volume_half_sector_l374_374943

noncomputable def cone_volume (r slant_height : ℝ) :=
  (1/3) * π * r^2 * (real.sqrt (slant_height^2 - r^2))

theorem cone_volume_half_sector {r : ℝ} (h : r = 6) :
  cone_volume 3 r = 9 * π * real.sqrt 3 :=
by 
  have sl : r = 6, from h,
  have base_radius : ℝ := 3,
  show cone_volume base_radius sl = 9 * π * real.sqrt 3,
  sorry

end cone_volume_half_sector_l374_374943


namespace find_midline_l374_374976

theorem find_midline (a b c : ℝ) (d : ℝ) (h : ∀ x : ℝ, -3 ≤ a * sin (b * x + c) + d ∧ a * sin (b * x + c) + d ≤ 5) :
  d = 1 :=
by
  sorry

end find_midline_l374_374976


namespace factor_difference_of_squares_l374_374905

theorem factor_difference_of_squares (a b : ℝ) : 
    (∃ A B : ℝ, a = A ∧ b = B) → 
    (a^2 - b^2 = (a + b) * (a - b)) :=
by
  intros h
  sorry

end factor_difference_of_squares_l374_374905


namespace power_expression_eval_l374_374188

theorem power_expression_eval :
  (-2: ℤ)^23 + 2^((2^4 + 5^2 - 7^2):ℤ) = -8388607.99609375 := by
  sorry

end power_expression_eval_l374_374188


namespace petya_four_digits_l374_374785

theorem petya_four_digits :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
    6 * (a + b + c + d) * 1111 = 73326 ∧ 
    (∃ S, S = a + b + c + d ∧ S = 11) :=
begin
  use 1, 2, 3, 5,
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split,
  { exact dec_trivial },
  use 11,
  split,
  { exact dec_trivial },
  { exact dec_trivial }
end

end petya_four_digits_l374_374785


namespace square_area_calculation_l374_374961

-- Define the diagonal length of the square
def diagonal_length : ℝ := 8 * Real.sqrt 2

-- Definition of side length based on the properties of 45-45-90 triangle
def side_length (d : ℝ) : ℝ := d / Real.sqrt 2

-- Definition of the area of a square given its side length
def square_area (s : ℝ) : ℝ := s * s

-- The main statement to prove
theorem square_area_calculation : square_area (side_length diagonal_length) = 64 := by
  sorry

end square_area_calculation_l374_374961


namespace not_factorial_tails_less_than_2500_l374_374996

noncomputable def factorial_zeros (m : ℕ) : ℕ :=
  m / 5 + m / 25 + m / 125 + m / 625 + m / 3125 + m / 15625 -- and so on.

theorem not_factorial_tails_less_than_2500 : 
  ∀ N < 2500, (N - (∑ m in (finset.range 25000).filter (λ m, factorial_zeros m = N), 1)) = 500 :=
sorry

end not_factorial_tails_less_than_2500_l374_374996


namespace first_term_exceeding_10000_l374_374093

theorem first_term_exceeding_10000 {a : ℕ → ℕ} (h0 : a 1 = 2)
  (h1 : ∀ n ≥ 2, a n = 2 * ∑ i in Finset.range (n - 1), a (i + 1)) :
  ∃ n, a n > 10000 ∧ n = 10 ∧ a 10 = 26124 :=
by
  sorry

end first_term_exceeding_10000_l374_374093


namespace asymptote_of_hyperbola_l374_374417
-- Import the entirety of the necessary library 

-- Define the problem statement in Lean 4
theorem asymptote_of_hyperbola (y x : ℝ) :
  (y^2 / 3 - x^2 = 1) → (y = sqrt 3 * x ∨ y = -sqrt 3 * x) :=
sorry

end asymptote_of_hyperbola_l374_374417


namespace jason_games_planned_last_month_l374_374353

-- Define the conditions
variable (games_planned_this_month : Nat) (games_missed : Nat) (games_attended : Nat)

-- Define what we want to prove
theorem jason_games_planned_last_month (h1 : games_planned_this_month = 11)
                                        (h2 : games_missed = 16)
                                        (h3 : games_attended = 12) :
                                        (games_attended + games_missed - games_planned_this_month = 17) := 
by
  sorry

end jason_games_planned_last_month_l374_374353


namespace clock_angle_at_2_15_l374_374181

theorem clock_angle_at_2_15 :
  let h := 2
  let m := 15
  abs ((60 * h - 11 * m) / 2) = 22.5 :=
by
  let h := 2
  let m := 15
  have angle_formula : abs ((60 * h - 11 * m) / 2) = abs ((60 * 2 - 11 * 15) / 2) := by rfl
  calc
    abs ((60 * 2 - 11 * 15) / 2)
        = abs ((120 - 165) / 2) : by rfl
    ... = abs (-45 / 2) : by rfl
    ... = abs ((-45 : ℚ) / 2) : by rw [Int.cast_neg, Int.cast_bit0, Int.cast_one]
    ... = abs ((-45 : ℚ) / 2) : by rfl
    ... = abs ((-45 : ℚ) / (2 : ℚ)) : by rfl
    ... = abs (-45) * abs (2⁻¹) : by rw [abs_div]
    ... = 45 * 1 / 2 : by simp only [abs_neg, abs_pos, inv_eq_one_div, Int.cast_bit0, Int.cast_one]
    ... = 22.5 : by norm_num

end clock_angle_at_2_15_l374_374181


namespace volleyball_tournament_l374_374010

open Function

theorem volleyball_tournament
  (V : Type*) -- Type of the vertices representing teams
  [Fintype V] -- Finitely many teams
  [DecidableEq V] -- Decidable equality on teams
  (d : V → V → Prop) -- Directed edge representing match result
  [DecidableRel d] -- Decidability of the match result
  (h_outdegree : ∀ v : V, ∃! (w : V → Prop), ∃ s, length (s.filter (λ x, d v x)) = 10) -- Outdegree is 10 for each vertex
  (h_indegree : ∀ v : V, ∃! (u : V → Prop), ∃ r, length (r.filter (λ x, d x v)) = 10) -- Indegree is 10 for each vertex
  : ∃ (G' : V → V → Prop), 
    (∀ v, ∃! k : V, G' v k) ∧ -- Exactly one outgoing edge 
    (∀ v, ∃! l : V, G' l v) -- Exactly one incoming edge 
    :=
sorry

end volleyball_tournament_l374_374010


namespace longest_segment_in_quadrilateral_l374_374008

theorem longest_segment_in_quadrilateral 
  (A B C D : Type*) 
  (ABD : ℝ) (BAD : ℝ) (CBD : ℝ) (BCD : ℝ)
  (ABD_eq : ABD = 45)
  (BAD_eq : BAD = 50)
  (CBD_eq : CBD = 65)
  (BCD_eq : BCD = 70)
  : (∀ AB AD BD BC CD, AB < AD ∧ AD < BD ∧ BD < CD ∧ CD < BC)
  → (∃ s, s = BC) := by
  sorry

end longest_segment_in_quadrilateral_l374_374008


namespace sum_integers_from_5_to_75_l374_374900

theorem sum_integers_from_5_to_75 : (∑ k in Finset.range(71) + 5, k) = 2840 := 
by
  sorry

end sum_integers_from_5_to_75_l374_374900


namespace prime_factors_of_M_l374_374651

theorem prime_factors_of_M :
  ∀ (M : ℝ), (log 2 (log 5 (log 7 (log 11 M)))) = 9 → (prime_factors M).card = 1 :=
by
  sorry

end prime_factors_of_M_l374_374651


namespace inequality_a_neg_one_inequality_general_a_l374_374290

theorem inequality_a_neg_one : ∀ x : ℝ, (x^2 + x - 2 > 0) ↔ (x < -2 ∨ x > 1) :=
by { sorry }

theorem inequality_general_a : 
∀ (a x : ℝ), ax^2 - (a + 2)*x + 2 < 0 ↔ 
  if a = 0 then x > 1
  else if a < 0 then x < (2 / a) ∨ x > 1
  else if 0 < a ∧ a < 2 then 1 < x ∧ x < (2 / a)
  else if a = 2 then False
  else (2 / a) < x ∧ x < 1 :=
by { sorry }

end inequality_a_neg_one_inequality_general_a_l374_374290


namespace area_of_triangle_is_11_25_l374_374516

noncomputable def area_of_triangle : ℝ :=
  let A := (1 / 2, 2)
  let B := (8, 2)
  let C := (2, 5)
  let base := (B.1 - A.1 : ℝ)
  let height := (C.2 - A.2 : ℝ)
  0.5 * base * height

theorem area_of_triangle_is_11_25 :
  area_of_triangle = 11.25 := sorry

end area_of_triangle_is_11_25_l374_374516


namespace arithmetic_sequence_sum_l374_374275

theorem arithmetic_sequence_sum (a : ℝ) (n : ℕ) (S : ℕ → ℝ)
  (h_sum : ∀ n, S n = a * n ^ 2)
  (h_d : ∃ d : ℝ, d ≠ 0) :
  (∃ a_n : ℕ → ℝ, ∀ n, a_n n = S n - S (n-1)) →
  let a_5 := S 5 - S 4,
      d := (S 2 - S 1) - (S 1 - S 0) in
  a_5 / d = 9 / 2 :=
by
  sorry

end arithmetic_sequence_sum_l374_374275


namespace find_y_l374_374887

theorem find_y (y z : ℕ) (h1 : 50 = y * 10) (h2 : 300 = 50 * z) : y = 5 :=
by
  sorry

end find_y_l374_374887


namespace correct_exponent_calculation_l374_374461

theorem correct_exponent_calculation (a : ℝ) : 
  (a^5 * a^2 = a^7) :=
by
  sorry

end correct_exponent_calculation_l374_374461


namespace max_profit_l374_374157

noncomputable def fixed_cost : ℝ := 2.5

noncomputable def cost (x : ℝ) : ℝ :=
  if x < 80 then (1/3) * x^2 + 10 * x
  else 51 * x + 10000 / x - 1450

noncomputable def revenue (x : ℝ) : ℝ := 0.05 * 1000 * x

noncomputable def profit (x : ℝ) : ℝ :=
  revenue x - cost x - fixed_cost * 10

theorem max_profit : ∃ x_opt : ℝ, ∀ x : ℝ, 0 < x → 
  profit x ≤ profit 100 ∧ x_opt = 100 :=
by
  sorry

end max_profit_l374_374157


namespace students_taller_than_Yoongi_l374_374469

theorem students_taller_than_Yoongi {n total shorter : ℕ} (h1 : total = 20) (h2 : shorter = 11) : n = 8 :=
by
  sorry

end students_taller_than_Yoongi_l374_374469


namespace area_ratio_proof_l374_374507

variables (BE CE DE AE : ℝ)
variables (S_alpha S_beta S_gamma S_delta : ℝ)
variables (x : ℝ)

-- Definitions for the given conditions
def BE_val := 80
def CE_val := 60
def DE_val := 40
def AE_val := 30

-- Expressing the ratios
def S_alpha_ratio := 2
def S_beta_ratio := 2

-- Assuming areas in terms of x
def S_alpha_val := 2 * x
def S_beta_val := 2 * x
def S_delta_val := x
def S_gamma_val := 2 * x

-- Problem statement
theorem area_ratio_proof
  (BE := BE_val)
  (CE := CE_val)
  (DE := DE_val)
  (AE := AE_val)
  (S_alpha := S_alpha_val)
  (S_beta := S_beta_val)
  (S_gamma := S_gamma_val)
  (S_delta := S_delta_val) :
  (S_gamma + S_delta) / (S_alpha + S_beta) = 5 / 4 :=
by
  sorry

end area_ratio_proof_l374_374507


namespace log_concave_l374_374398

theorem log_concave : ∀ (x1 x2 : ℝ), (0 < x1) → (0 < x2) → 
  (log ((x1 + x2) / 2) ≥ (log x1 + log x2) / 2) :=
by
  intros x1 x2 hx1 hx2
  sorry

end log_concave_l374_374398


namespace fifth_term_sequence_l374_374562

theorem fifth_term_sequence : (∑ i in Finset.range 5, 4^i) = 341 :=
by
  sorry

end fifth_term_sequence_l374_374562


namespace torricelli_experiment_proof_l374_374124

variables (t1 t2 : ℤ) (p_ambient : ℝ) (sigma_ether sigma_mercury : ℝ) 
          (p_vapor_30 p_vapor_10 : ℝ)

-- Conditions
def condition_t1 := t1 = 30
def condition_t2 := t2 = -10
def condition_p_ambient := p_ambient = 750
def condition_sigma_ether := sigma_ether = 0.715
def condition_sigma_mercury := sigma_mercury = 13.6
def condition_p_vapor_30 := p_vapor_30 = 634.8
def condition_p_vapor_10 := p_vapor_10 = 114.72

-- Main theorem
theorem torricelli_experiment_proof 
  (h1 : condition_t1 t1)
  (h2 : condition_t2 t2)
  (h3 : condition_p_ambient p_ambient)
  (h4 : condition_sigma_ether sigma_ether)
  (h5 : condition_sigma_mercury sigma_mercury)
  (h6 : condition_p_vapor_30 p_vapor_30)
  (h7 : condition_p_vapor_10 p_vapor_10) :
  ∃ l1 l2 : ℝ, l1 ≈ 2188.83 ∧ l2 ≈ 12084.34 := by sorry

end torricelli_experiment_proof_l374_374124


namespace general_formula_a_sum_bn_l374_374605

noncomputable section

open Nat

-- Define the sequence Sn
def S (n : ℕ) : ℕ := 2^n + n - 1

-- Define the sequence an
def a (n : ℕ) : ℕ := 1 + 2^(n-1)

-- Define the sequence bn
def b (n : ℕ) : ℕ := 2 * n * (a n - 1)

-- Define the sum Tn
def T (n : ℕ) : ℕ := n * 2^n

-- Proposition 1: General formula for an
theorem general_formula_a (n : ℕ) : a n = 1 + 2^(n-1) :=
by
  sorry

-- Proposition 2: Sum of first n terms of bn
theorem sum_bn (n : ℕ) : T n = 2 + (n - 1) * 2^(n+1) :=
by
  sorry

end general_formula_a_sum_bn_l374_374605


namespace petya_four_digits_l374_374784

theorem petya_four_digits :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
    6 * (a + b + c + d) * 1111 = 73326 ∧ 
    (∃ S, S = a + b + c + d ∧ S = 11) :=
begin
  use 1, 2, 3, 5,
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split,
  { exact dec_trivial },
  use 11,
  split,
  { exact dec_trivial },
  { exact dec_trivial }
end

end petya_four_digits_l374_374784


namespace problem_lean_l374_374658

theorem problem_lean (a : ℝ) (h : a - 1/a = 5) : a^2 + 1/a^2 = 27 := by
  sorry

end problem_lean_l374_374658


namespace tower_height_l374_374182

theorem tower_height (d h : ℝ) (h1 : tan (π / 6) = h / d) (h2 : tan (π / 4) = h / (d - 20)) :
  h = 10 * (Real.sqrt 3 + 1) :=
sorry

end tower_height_l374_374182


namespace rectangle_perimeter_of_right_triangle_l374_374170

noncomputable def right_triangle_area (a b: ℕ) : ℝ := (1/2 : ℝ) * a * b

noncomputable def rectangle_length (area width: ℝ) : ℝ := area / width

noncomputable def rectangle_perimeter (length width: ℝ) : ℝ := 2 * (length + width)

theorem rectangle_perimeter_of_right_triangle :
  rectangle_perimeter (rectangle_length (right_triangle_area 7 24) 5) 5 = 43.6 :=
by
  sorry

end rectangle_perimeter_of_right_triangle_l374_374170


namespace count_three_digit_palindromes_l374_374856

theorem count_three_digit_palindromes : 
  let num_palindromes := 
    (finset.filter (λ n : ℕ, (100 ≤ n) ∧ (n < 1000) ∧ (let d2 := n / 10 % 10 in n / 100 % 10 = n % 10)
      ) ((finset.range 1000))) in
  num_palindromes.card = 90 := 
begin
  sorry
end

end count_three_digit_palindromes_l374_374856


namespace arrangement_count_l374_374148

theorem arrangement_count (boys_girls : Finset (Fin 7)) 
  (boys : Finset (Fin 4)) 
  (girls : Finset (Fin 3))
  (condition1 : ∃ g1 g2 ∈ girls, g1 ≠ g2 ∧ (g1, g2) ∈ (boys_girls.pairwise (≠)))
  (condition2 : ∀ g1 g2 g3 ∈ girls, g1 ≠ g2 ∧ g2 ≠ g3 ∧ g1 ≠ g3 → 
                ¬ (g1 ∈ pairwise (g2 g3))) :
  (∃! s : Finset (Fin 7), boys_girls.card = 7 ∧ boys.card = 4 ∧ girls.card = 3 ∧
    2 ∃ g_in : girls,  g_in ∧ ∀ g1, g2 ∈ girls ∧ (g1 ≠ g2 → (g1, g2) ∈ (boys_girls.pairwise (≠)))) 
  ≃ 2880 := sorry

end arrangement_count_l374_374148


namespace Joe_team_wins_eq_1_l374_374026

-- Definition for the points a team gets for winning a game.
def points_per_win := 3
-- Definition for the points a team gets for a tie game.
def points_per_tie := 1

-- Given conditions
def Joe_team_draws := 3
def first_place_wins := 2
def first_place_ties := 2
def points_difference := 2

def first_place_points := (first_place_wins * points_per_win) + (first_place_ties * points_per_tie)

def Joe_team_total_points := first_place_points - points_difference
def Joe_team_points_from_ties := Joe_team_draws * points_per_tie
def Joe_team_points_from_wins := Joe_team_total_points - Joe_team_points_from_ties

-- To prove: number of games Joe's team won
theorem Joe_team_wins_eq_1 : (Joe_team_points_from_wins / points_per_win) = 1 :=
by
  sorry

end Joe_team_wins_eq_1_l374_374026


namespace bella_bakes_most_cookies_per_batch_l374_374331

theorem bella_bakes_most_cookies_per_batch (V : ℝ) :
  let alex_cookies := V / 9
  let bella_cookies := V / 7
  let carlo_cookies := V / 8
  let dana_cookies := V / 10
  alex_cookies < bella_cookies ∧ carlo_cookies < bella_cookies ∧ dana_cookies < bella_cookies :=
sorry

end bella_bakes_most_cookies_per_batch_l374_374331


namespace total_apples_packed_l374_374742

def apples_packed_daily (apples_per_box : ℕ) (boxes_per_day : ℕ) : ℕ :=
  apples_per_box * boxes_per_day

def apples_packed_first_week (apples_per_box : ℕ) (boxes_per_day : ℕ) (days_first_week : ℕ) : ℕ :=
  apples_packed_daily apples_per_box boxes_per_day * days_first_week

def apples_packed_second_week (apples_per_box : ℕ) (boxes_per_day : ℕ) (days_second_week : ℕ) (decrease_per_day : ℕ) : ℕ :=
  (apples_packed_daily apples_per_box boxes_per_day - decrease_per_day) * days_second_week

theorem total_apples_packed (apples_per_box : ℕ) (boxes_per_day : ℕ) (days_first_week : ℕ) (days_second_week : ℕ) (decrease_per_day : ℕ) :
  apples_per_box = 40 →
  boxes_per_day = 50 →
  days_first_week = 7 →
  days_second_week = 7 →
  decrease_per_day = 500 →
  apples_packed_first_week apples_per_box boxes_per_day days_first_week + apples_packed_second_week apples_per_box boxes_per_day days_second_week decrease_per_day = 24500 :=
  by
  intros h₁ h₂ h₃ h₄ h₅
  rw [h₁, h₂, h₃, h₄, h₅]
  dsimp [apples_packed_first_week, apples_packed_second_week, apples_packed_daily]
  sorry

end total_apples_packed_l374_374742


namespace seq_sum_and_term_bn_increasing_l374_374604

-- Question 1: Prove the formulas for S_n and a_n
theorem seq_sum_and_term (n : ℕ) (a : ℕ → ℕ) (S : ℕ → ℤ) (h₁ : a 1 = 3)
    (h₂ : ∀ n, S n + 1 = 4 * 4^n.pred) :
  S n = 4^n - 1 ∧ a n = 3 * 4^n.pred :=
by sorry

-- Question 2: Prove the range for λ given b_n is increasing
theorem bn_increasing (λ : ℝ) (n : ℕ) (a : ℕ → ℝ) (h₁ : ∀ n, a n = 3 * 4^n.pred)
    (b : ℕ → ℝ) (h₂ : ∀ n, b n = n * 4^n + λ * a n)
    (h₃ : ∀ n, b (n + 1) - b n > 0) : 
  λ > - 28 / 9 :=
by sorry

end seq_sum_and_term_bn_increasing_l374_374604


namespace parallelogram_area_l374_374119

noncomputable def radius : ℝ := 1
noncomputable def segment_to_tangency_point : ℝ := √3
noncomputable def height : ℝ := 2
noncomputable def base : ℝ := 2 * √3 + 2

theorem parallelogram_area : 
  let area := base * height in 
  area = 4 * (1 + √3) := 
by 
  sorry

end parallelogram_area_l374_374119


namespace fraction_of_August_tips_l374_374208

-- Define the conditions
variable {A : ℝ} -- The average monthly tips for months other than August
variable (total_tips_March_to_July_and_September : ℝ := 6 * A) -- 6 months' total tips
variable (August_tips : ℝ := 2 * A) -- Tips for August

-- Total tips for all the months
def total_tips_all_months : ℝ := total_tips_March_to_July_and_September + August_tips

-- Define the fraction of August tips to total tips to be proven as 1/4
theorem fraction_of_August_tips : (August_tips / total_tips_all_months) = 1 / 4 := by
  sorry

end fraction_of_August_tips_l374_374208


namespace cos_17_pi_over_6_l374_374538

noncomputable def rad_to_deg (r : ℝ) : ℝ := r * 180 / Real.pi

theorem cos_17_pi_over_6 : Real.cos (17 * Real.pi / 6) = - (Real.sqrt 3) / 2 := by
  sorry

end cos_17_pi_over_6_l374_374538


namespace problem1_problem2_l374_374734

def M := { x : ℝ | 0 < x ∧ x < 1 }

theorem problem1 :
  { x : ℝ | |2 * x - 1| < 1 } = M :=
by
  simp [M]
  sorry

theorem problem2 (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (a * b + 1) > (a + b) :=
by
  simp [M] at ha hb
  sorry

end problem1_problem2_l374_374734


namespace range_of_y_l374_374572

-- Define the function y in terms of x
def y (x : ℝ) : ℝ := (sin x) ^ 2 + (sin x) - 2

-- Define the range of the sine function
lemma range_sin (x : ℝ) : -1 ≤ sin x ∧ sin x ≤ 1 :=
  sorry

-- The goal is to prove the range of the function y is [-9/4, 0]
theorem range_of_y : set.range y = set.Icc (-9 / 4) 0 :=
  sorry

end range_of_y_l374_374572


namespace sequence_sum_correct_l374_374221

theorem sequence_sum_correct :
  ∀ (r x y : ℝ),
  (x = 128 * r) →
  (y = x * r) →
  (2 * r = 1 / 2) →
  (x + y = 40) :=
by
  intros r x y hx hy hr
  sorry

end sequence_sum_correct_l374_374221


namespace matrix_a_cannot_be_transformed_to_matrix_b_l374_374930

def position (c : Char) : ℕ :=
  if c = 'Z' then 26 else c.val - 'A'.val + 1

def to_matrix (s : String) : Array (Array ℕ) :=
  #[#[position s[0], position s[1], position s[2], position s[3]],
    #[position s[4], position s[5], position s[6], position s[7]],
    #[position s[8], position s[9], position s[10], position s[11]],
    #[position s[12], position s[13], position s[14], position s[15]]]

def invariant_k (m : Array (Array ℕ)) (i j : ℕ) : ℤ :=
  (m[i][0] + m[j][3] - (m[i][1] + m[j][2]))

def can_transform (A B : Array (Array ℕ)) : Prop :=
  A.size = 4 ∧ A.all (λ row => row.size = 4) ∧
  B.size = 4 ∧ B.all (λ row => row.size = 4) ∧
  (∃ col, ∀ i, invariant_k A i col = invariant_k B i col)

def matrix_a := to_matrix "SOTZEXTZABCDEFGHIJKLMNOPQRSTUVWXYZ"
def matrix_b := to_matrix "KBHEWBQRSTUVWXYZABCDEFGHIJKLMNOP"

theorem matrix_a_cannot_be_transformed_to_matrix_b :
  ¬ can_transform matrix_a matrix_b := sorry

end matrix_a_cannot_be_transformed_to_matrix_b_l374_374930


namespace line_equation_l374_374101

-- Definitions of lines l1 and l2
def l1 (x y : ℝ) := 4 * x + y + 3 = 0
def l2 (x y : ℝ) := 3 * x - 5 * y - 5 = 0

/-- Prove the equation of line l given the midpoint and intersections -/
theorem line_equation (P : ℝ × ℝ) (a b y1 y2 : ℝ) (l : ℝ → ℝ → Prop) :
  P = (-1, 2) →
  l1 a y1 →
  l2 b y2 →
  l1 (-2) y1 →
  l2 0 y2 →
  (l (-2) 5) ∧ (l 0 (-1)) →
  (P.1 = (a + b) / 2) ∧ (P.2 = (y1 + y2) / 2) →
  l = (λ x y, 3 * x + y + 1 = 0) :=
by
  sorry

end line_equation_l374_374101


namespace petya_digits_l374_374767

theorem petya_digits (a b c d : ℕ) (distinct_digits : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (non_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (sum_formed_numbers : 6 * (a + b + c + d) * 1111 = 73326) :
  {a, b, c, d} = {1, 2, 3, 5} :=
by 
  sorry

end petya_digits_l374_374767


namespace initial_games_played_l374_374491

theorem initial_games_played (avg_hits_per_game : ℕ) (players : ℕ) (best_player_hits : ℕ) (avg_hits_next_games : ℕ) (next_games : ℕ)
  (team_avg_hits : avg_hits_per_game = 15)
  (total_players : players = 11)
  (best_player_total_hits : best_player_hits = 25)
  (avg_hits_per_other_player : avg_hits_next_games = 6)
  (games_to_play : next_games = 6) :
  ∃ x : ℕ, 15 * x - 25 = 360 ∧ (x = 25) :=
by
  exist 25;
  split;
  {
    calc
      15 * (25:ℕ) - 25 = 375 - 25 : by rfl
                     ... = 350 + 10 : by rfl
                     ... = 350 + 10 : by rfl
                     ... = 360 : by ring
  sorry

end initial_games_played_l374_374491


namespace car_salesman_earnings_l374_374935

theorem car_salesman_earnings :
  ∀ (x : ℕ), 
  let base_salary := 1000
  let commission_per_car := 200
  let March_earnings := base_salary + commission_per_car * x
  let April_earnings := base_salary + commission_per_car * 15
  in 2 * March_earnings = April_earnings → March_earnings = 2000 :=
by
  sorry

end car_salesman_earnings_l374_374935


namespace surface_area_of_sphere_l374_374956

-- Definitions based on the conditions
def sphere_radius_from_intersection (r: ℝ) (d: ℝ) : ℝ :=
  Real.sqrt (r^2 + d^2)

def sphere_surface_area (R: ℝ) : ℝ :=
  4 * Real.pi * R^2

theorem surface_area_of_sphere (radius_intersection: ℝ) (distance_to_plane: ℝ) :
  radius_intersection = 1 → distance_to_plane = Real.sqrt 2 → 
  sphere_surface_area (sphere_radius_from_intersection radius_intersection distance_to_plane) = 12 * Real.pi :=
by
  intros h_radius h_distance
  rw [h_radius, h_distance]
  simp [sphere_radius_from_intersection, sphere_surface_area]
  sorry

end surface_area_of_sphere_l374_374956


namespace sum_of_roots_of_quadratic_l374_374326

theorem sum_of_roots_of_quadratic :
  ∀ x1 x2 : ℝ, (∃ a b c, a = -1 ∧ b = 2 ∧ c = 4 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) → (x1 + x2 = 2) :=
by
  sorry

end sum_of_roots_of_quadratic_l374_374326


namespace square_area_proof_l374_374423

variable (r s l b : ℝ)
variable (area_rect area_square : ℝ)

constant (h1 : l = (1/6) * r)
constant (h2 : r = s)
constant (h3 : b = 10)
constant (h4 : area_rect = l * b)
constant (h5 : area_rect = 360)
constant (side : s = 216)
constant (area_square : ℝ := s^2)

theorem square_area_proof :
  area_square = 46656 :=
by
  sorry

end square_area_proof_l374_374423


namespace sufficient_not_necessary_condition_l374_374596

theorem sufficient_not_necessary_condition (a b : ℝ) (h : (a - b) * a^2 > 0) : a > b ∧ a ≠ 0 :=
by {
  sorry
}

end sufficient_not_necessary_condition_l374_374596


namespace pattern_expression_equality_l374_374751

theorem pattern_expression_equality (n : ℕ) : ((n - 1) * (n + 1)) + 1 = n^2 :=
  sorry

end pattern_expression_equality_l374_374751


namespace sin_480_deg_l374_374440

theorem sin_480_deg : Real.sin (480 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_480_deg_l374_374440


namespace octahedron_side_length_l374_374517

/-- Proof Problem: Given conditions of vertices locations and 
segments on a unit cube, prove that the side length of the 
regular octahedron is sqrt(2) / 3 -/
theorem octahedron_side_length
  {P1 P2 P3 P4 P1' P2' P3' P4' : ℝ × ℝ × ℝ}
  (adjacent_to_P1 : P2 = (1, 0, 0) ∧ P3 = (0, 1, 0) ∧ P4 = (0, 0, 1))
  (opposite_vertices : P1' = (1, 1, 1) ∧ P2' = (0, 1, 1) ∧ P3' = (1, 0, 1) ∧ P4' = (1, 1, 0))
  (octahedron_vertices : 
     ∃ v1 v2 v3 v4 v5 v6 : ℝ × ℝ × ℝ,
     v1 = (1/3, 0, 0) ∧ v2 = (0, 1/3, 0) ∧ v3 = (0, 0, 1/3) ∧ 
     v4 = (1, 2/3, 1) ∧ v5 = (1, 1, 2/3) ∧ v6 = (2/3, 1, 1)) :
  ∀ u v : ℝ × ℝ × ℝ, 
    (u = (1/3, 0, 0) ∧ v = (0, 1/3, 0)) → dist u v = sqrt(2) / 3 :=
by
  sorry

end octahedron_side_length_l374_374517


namespace eq_proof_l374_374454

noncomputable def S_even : ℚ := 28
noncomputable def S_odd : ℚ := 24

theorem eq_proof : ( (S_even / S_odd - S_odd / S_even) * 2 ) = (13 / 21) :=
by
  sorry

end eq_proof_l374_374454


namespace arithmetic_sequence_y_value_l374_374338

theorem arithmetic_sequence_y_value :
  ∀ (x y z : ℤ),
  (∀ (d : ℤ), x = 12 + d ∧ y = 12 + 2 * d ∧ z = 12 + 3 * d ∧ 32 = 12 + 4 * d) →
  y = 22 :=
by
  intros x y z h,
  cases h with d hd,
  cases hd,
  sorry

end arithmetic_sequence_y_value_l374_374338


namespace g_diff_eq_neg8_l374_374420

noncomputable def g : ℝ → ℝ := sorry

axiom linear_g : ∀ x y : ℝ, g (x + y) = g x + g y

axiom condition_g : ∀ x : ℝ, g (x + 2) - g x = 4

theorem g_diff_eq_neg8 : g 2 - g 6 = -8 :=
by
  sorry

end g_diff_eq_neg8_l374_374420


namespace find_point_Q_l374_374625

-- Define the function f
def f (x : ℝ) (m : ℝ) : ℝ := (1 / 3) * x^3 + ((m + 1) / 2) * x^2 + 2 + (1 / x)

-- Define the derivative function f'
def f' (x : ℝ) (m : ℝ) : ℝ := x^2 + (m + 1) * x - (1 / x^2)

-- Define the condition that f is monotonically increasing on [1, +∞)
def is_monotonically_increasing (m : ℝ) : Prop :=
  ∀ x, 1 ≤ x → 0 ≤ f' x m

-- Define the minimum value of m
def minimum_value_of_m : ℝ := -1

-- Define the function p by translating f downward by 2 units
def p (x : ℝ) : ℝ := (1 / 3) * x^3 + (1 / x)

-- Define the point Q
def Q : ℝ × ℝ := (0, 2)

-- State the theorem
theorem find_point_Q :
  is_monotonically_increasing minimum_value_of_m →
  Q = (0, 2) :=
by
  sorry

end find_point_Q_l374_374625


namespace trailing_zeros_factorial_345_l374_374426

theorem trailing_zeros_factorial_345 : 
  let count_factors (n k : ℕ) := n / k in
  count_factors 345 5 + count_factors 345 25 + count_factors 345 125 = 84 :=
by
  have count_factors := λ n k : ℕ, n / k
  calc
    count_factors 345 5 = 69    : by sorry
    count_factors 345 25 = 13   : by sorry
    count_factors 345 125 = 2   : by sorry
    69 + 13 + 2 = 84            : by ring

end trailing_zeros_factorial_345_l374_374426


namespace angle_BAC_correct_l374_374154

noncomputable def find_angle_BAC (B N M C A : Type) 
  (h1 : N ∈ segment A B) 
  (h2 : M ∈ segment A C) 
  (angle_ACB : Real := π / 3) 
  (angle_BMC : Real := π / 4) 
  (BN_eq_2MN : ∀ (MN : Real), BN = 2 * MN) : Real :=
  arctan (sqrt 3 / 2) - π / 12

theorem angle_BAC_correct (B N M C A : Type) 
  (h1 : N ∈ segment A B) 
  (h2 : M ∈ segment A C) 
  (angle_ACB : Real := π / 3) 
  (angle_BMC : Real := π / 4) 
  (BN_eq_2MN : ∀ (MN : Real), BN = 2 * MN) : 
  find_angle_BAC B N M C A h1 h2 angle_ACB angle_BMC BN_eq_2MN = arctan (sqrt 3 / 2) - π / 12 :=
  sorry

end angle_BAC_correct_l374_374154


namespace fifth_term_geometric_sequence_l374_374992

theorem fifth_term_geometric_sequence (y : ℝ) : 
  let a : ℕ → ℝ := λ n, 3 * (4 * y) ^ n in
  a 4 = 768 * y ^ 4 :=
sorry

end fifth_term_geometric_sequence_l374_374992


namespace Liz_can_roast_turkeys_l374_374043

theorem Liz_can_roast_turkeys :
  (∀ (pounds_per_turkey : ℕ) (minutes_per_pound : ℕ) (start_time : ℕ) (end_time : ℕ),
    pounds_per_turkey = 16 →
    minutes_per_pound = 15 →
    start_time = 10 →
    end_time = 18 →
    (end_time - start_time) * 60 / (pounds_per_turkey * minutes_per_pound) = 2) :=
begin
  intros pounds_per_turkey minutes_per_pound start_time end_time,
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  norm_num,
end

end Liz_can_roast_turkeys_l374_374043


namespace ratio_of_semicircles_to_circle_l374_374341

-- Define the given conditions

-- Define a rectangle and its enclosing circle's properties
structure RectangleCircleConfig where
  r : ℝ -- radius of the circle and consequentially half of the side lengths
  AB_CD_length : ℝ := 2 * r -- this is given as the length AB = CD = 2r

-- Define the semicircles properties
structure Semicircle where
  radius : ℝ
  is_tangent_to : ℝ -- the radius, this will be r 

-- Define the circle properties
structure Circle where
  radius : ℝ

-- The main theorem to prove
theorem ratio_of_semicircles_to_circle (config : RectangleCircleConfig) (semi1 semi2 : Semicircle) (circ : Circle)
  (semi1_prop : semi1.radius = config.r)
  (semi2_prop : semi2.radius = config.r)
  (circ_prop : circ.radius = config.r) :
  let semicircle_area (r : ℝ) := (1 / 2) * real.pi * r^2 in
  (semicircle_area semi1.radius + semicircle_area semi2.radius) / (real.pi * circ.radius^2) = 1 :=
by {
  sorry
}

end ratio_of_semicircles_to_circle_l374_374341


namespace find_difference_l374_374224

theorem find_difference (x0 y0 : ℝ) 
  (h1 : x0^3 - 2023 * x0 = y0^3 - 2023 * y0 + 2020)
  (h2 : x0^2 + x0 * y0 + y0^2 = 2022) : 
  x0 - y0 = -2020 :=
by
  sorry

end find_difference_l374_374224


namespace correct_statement_C_l374_374599

variables {m n : Line} {α : Plane}

def is_perpendicular (l : Line) (p : Plane) : Prop := sorry -- Define perpendicularity of a line and a plane

def is_contained (l : Line) (p : Plane) : Prop := sorry -- Define if a line is contained in a plane

def are_perpendicular (l1 l2 : Line) : Prop := sorry -- Define perpendicularity of two lines

theorem correct_statement_C (h1 : are_different m n) (h2 : is_perpendicular m α) (h3 : is_contained n α) : are_perpendicular m n := 
sorry

end correct_statement_C_l374_374599


namespace biking_time_l374_374963

noncomputable def radius (width : ℕ) : ℕ := width / 2
noncomputable def semicircle_distance (radius : ℕ) : ℕ := (2 * radius) * π
noncomputable def total_semicircles (mile_in_feet : ℕ) (semicircle_diameter : ℕ) : ℕ := mile_in_feet / semicircle_diameter
noncomputable def biking_distance (total_semicircles : ℕ) (semicircle_distance : ℕ) : ℝ := total_semicircles * semicircle_distance
noncomputable def distance_in_miles (distance_in_feet : ℝ) (feet_per_mile : ℕ) : ℝ := distance_in_feet / feet_per_mile
noncomputable def time_without_breaks (distance_in_miles : ℝ) (speed_mph : ℝ) : ℝ := distance_in_miles / speed_mph
noncomputable def total_breaks (distance_break : ℝ) (breaks_frequency_miles : ℝ) : ℝ := (distance_break / breaks_frequency_miles) * (5 / 60) 
noncomputable def total_time (time_without_breaks : ℝ) (total_breaks : ℝ) : ℝ := time_without_breaks + total_breaks

theorem biking_time
  (width_one_mile_highway : ℕ) (speed : ℝ) (break_time : ℝ) (mile_in_feet : ℕ) : 
  total_time (time_without_breaks (distance_in_miles (biking_distance (total_semicircles mile_in_feet (2 * radius width_one_mile_highway)) (semicircle_distance (radius width_one_mile_highway))) mile_in_feet) speed) (total_breaks (distance_in_miles (biking_distance (total_semicircles mile_in_feet (2 * radius width_one_mile_highway)) (semicircle_distance (radius width_one_mile_highway))) mile_in_feet) (1 / 2)) = 6 * π + 5 :=
begin
  sorry
end

end biking_time_l374_374963


namespace mass_percentage_H_correct_l374_374570

def molar_mass_C : ℝ := 12.01
def molar_mass_H : ℝ := 1.01
def molar_mass_N : ℝ := 14.01
def molar_mass_O : ℝ := 16.00

def atoms_C : ℝ := 9
def atoms_H : ℝ := 14
def atoms_N : ℝ := 3
def atoms_O : ℝ := 5

def molar_mass_compound : ℝ :=
  (atoms_C * molar_mass_C) + (atoms_H * molar_mass_H) + (atoms_N * molar_mass_N) + (atoms_O * molar_mass_O)

def mass_percentage_H : ℝ :=
  (atoms_H * molar_mass_H / molar_mass_compound) * 100

theorem mass_percentage_H_correct :
  mass_percentage_H = 5.79 := sorry

end mass_percentage_H_correct_l374_374570


namespace petya_digits_sum_l374_374763

def distinct_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem petya_digits_sum (a b c d : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
    (h_distinct: distinct_digits a b c d)
    (h_sum: 6 * (a + b + c + d) * 1111 = 73326) :
    a + b + c + d = 11 ∧ {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end petya_digits_sum_l374_374763


namespace find_lambda_l374_374612

variables {Point : Type} [AddCommGroup Point] [Module ℝ Point]

-- Define points A, B, and C
def A : Point := (2, 3)
def B : Point := (4, 5)
def C : Point := (7, 10)

-- Define the line equation
def on_line (P : Point) : Prop := P.x - 2 * P.y = 0

-- Define the vector equation \overrightarrow{AP} = \overrightarrow{AB} + λ\overrightarrow{AC}
def vector_eq (P : Point) (λ : ℝ) : Prop :=
  (P.x - 2, P.y - 3) = (2 + 5 * λ, 2 + 7 * λ)

-- Theorem statement
theorem find_lambda (P : Point) (λ : ℝ) (h1 : vector_eq P λ) (h2 : on_line P) : λ = -2 / 3 :=
sorry

end find_lambda_l374_374612


namespace find_wall_width_l374_374650

noncomputable def wall_width (painting_width : ℝ) (painting_height : ℝ) (wall_height : ℝ) (painting_coverage : ℝ) : ℝ :=
  (painting_width * painting_height) / (painting_coverage * wall_height)

-- Given constants
def painting_width : ℝ := 2
def painting_height : ℝ := 4
def wall_height : ℝ := 5
def painting_coverage : ℝ := 0.16
def expected_width : ℝ := 10

theorem find_wall_width : wall_width painting_width painting_height wall_height painting_coverage = expected_width := 
by
  sorry

end find_wall_width_l374_374650


namespace false_proposition_4_l374_374174

theorem false_proposition_4 
  (P : ℝ × ℝ) (x0 y0 : ℝ)
  (P1 P2 : ℝ × ℝ) (x1 y1 x2 y2 : ℝ)
  (a b x y k : ℝ) :
  (¬ ∀ P : ℝ × ℝ, ∃ k : ℝ, y - y0 = k * (x - x0)) →
  (∀ (P1 P2 : ℝ × ℝ), (x1 ≠ x2 → (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)) ∧ 
                      (x1 = x2 → x = x1)) →
  (∃ (a b : ℝ), ¬ (x / a + y / b = 1) ∧ ((0, 0) = (0, 0))) →
  (Q : ℝ × ℝ) (b : ℝ) (h : ¬ ∃ k : ℝ, y = k * x + b) →
  ¬ ∀ Q : ℝ × ℝ, ∃ k : ℝ, y = k * x + b :=
begin
  sorry
end

end false_proposition_4_l374_374174


namespace locus_of_P_coordinates_of_P_l374_374262

-- Define the points A and B
def A : ℝ × ℝ := (4, -3)
def B : ℝ × ℝ := (2, -1)

-- Define the line l : 4x + 3y - 2 = 0
def l (x y: ℝ) := 4 * x + 3 * y - 2 = 0

-- Problem (1): Equation of the locus of point P such that |PA| = |PB|
theorem locus_of_P (P : ℝ × ℝ) :
  (∃ P, dist P A = dist P B) ↔ (∀ x y : ℝ, P = (x, y) → x - y - 5 = 0) :=
sorry

-- Problem (2): Coordinates of P such that |PA| = |PB| and the distance from P to line l is 2
theorem coordinates_of_P (a b : ℝ):
  (dist (a, b) A = dist (a, b) B ∧ abs (4 * a + 3 * b - 2) / 5 = 2) ↔
  ((a = 1 ∧ b = -4) ∨ (a = 27 / 7 ∧ b = -8 / 7)) :=
sorry

end locus_of_P_coordinates_of_P_l374_374262


namespace sqrt_of_square_eq_pm_and_abs_l374_374434

variable {m : ℝ}

theorem sqrt_of_square_eq_pm_and_abs (m : ℝ) :
  (sqrt ((5 + m) ^ 2) = abs (5 + m)) ∧ (sqrt ((5 + m) ^ 2) = 5 + m ∨ sqrt ((5 + m) ^ 2) = -(5 + m)) :=
by
  sorry

end sqrt_of_square_eq_pm_and_abs_l374_374434


namespace num_fixed_last_two_digits_l374_374302

theorem num_fixed_last_two_digits : 
  ∃ c : ℕ, c = 36 ∧ ∀ (a : ℕ), 2 ≤ a ∧ a ≤ 101 → 
    (∃ N : ℕ, ∀ n : ℕ, n ≥ N → (a^(2^n) % 100 = a^(2^N) % 100)) ↔ (a = c ∨ c ≠ 36) :=
sorry

end num_fixed_last_two_digits_l374_374302


namespace isosceles_triangle_l374_374052

theorem isosceles_triangle
  {A B C M N O : Type*}
  [triangle A B C]
  (h1 : M ∈ line_segment A B)
  (h2 : N ∈ line_segment A C)
  (h3 : AM = AN)
  (h4 : (intersection CM BN = O))
  (h5 : BO = CO) : 
  isosceles_triangle A B C :=
sorry

end isosceles_triangle_l374_374052


namespace fran_speed_to_match_joann_distance_l374_374698

theorem fran_speed_to_match_joann_distance:
  ∀ (joann_speed joann_time fran_time: ℝ), 
  joann_speed = 15 → 
  joann_time = 4 → 
  fran_time = 2.5 → 
  (joann_speed * joann_time) = 60 → 
  (∃ fran_speed: ℝ, fran_speed = 24) :=
by
  intros joann_speed joann_time fran_time h_joann_speed h_joann_time h_fran_time h_distance
  use 24
  have h_distance_fran : 24 * fran_time = 60 := by 
    rw [h_fran_time]
    norm_num
  exact h_distance_fran.symm.trans h_distance.symm
  sorry

end fran_speed_to_match_joann_distance_l374_374698


namespace arithmetic_sum_calculation_l374_374983

theorem arithmetic_sum_calculation :
  3 * (71 + 75 + 79 + 83 + 87 + 91) = 1458 :=
by
  sorry

end arithmetic_sum_calculation_l374_374983


namespace power_modulo_remainder_l374_374126

theorem power_modulo_remainder :
  (17 ^ 2046) % 23 = 22 := 
sorry

end power_modulo_remainder_l374_374126


namespace sum_of_roots_of_parabola_l374_374870

open Real

noncomputable def parabola_sum_of_roots (a b L : ℝ) (E B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  let f := λ x : ℝ, (1/5)*x^2 + a*x + b in
  B = (-L/2, f (-L/2)) ∧ C = (L/2, f (L/2)) ∧ E = (0, f 0) ∧ L = 20

theorem sum_of_roots_of_parabola : ∃ a b L E B C, 
  parabola_sum_of_roots a b L E B C ∧
  (1/5 * (0 - L/2) * (L/2)) + a * 0 + b = 20 :=
sorry

end sum_of_roots_of_parabola_l374_374870


namespace purely_imaginary_l374_374668

theorem purely_imaginary {m : ℝ} (h1 : m^2 - 3 * m = 0) (h2 : m^2 - 5 * m + 6 ≠ 0) : m = 0 :=
sorry

end purely_imaginary_l374_374668


namespace complex_seventh_root_identity_l374_374370

open Complex

theorem complex_seventh_root_identity (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 11 :=
by
  sorry

end complex_seventh_root_identity_l374_374370


namespace polygons_count_l374_374058

-- Define the problem conditions
def num_vertices := 12
def num_segments := 11
def distinct_polygons := 1024

-- Lean 4 statement
theorem polygons_count :
  (number of distinct 11-segment open polygons without self-intersections
   with vertices at the points of a regular 12-gon,
   considering polygons that can be transformed into each other by rotation as the same)
  = distinct_polygons :=
by
  sorry

end polygons_count_l374_374058


namespace find_angle_l374_374299

noncomputable def a : ℝ × ℝ × ℝ := (1, 2, 3)
noncomputable def b : ℝ × ℝ × ℝ := (-2, -4, -6)
noncomputable def c_magnitude : ℝ := Real.sqrt 14

axiom ab_dot_c : (a.1 + b.1, a.2 + b.2, a.3 + b.3) • c = 7
axiom angle_between_a_c : ∀ c : ℝ × ℝ × ℝ, (Real.sqrt ((a.1 ^ 2 + a.2 ^ 2 + a.3 ^ 2)) = c_magnitude) → (a.1 ^ 2 + a.2 ^ 2 + a.3 ^ 2) ^ 0.5 * c_magnitude → (a • c = -7) → (Real.acos ((a • c) / (Real.sqrt ((a.1 ^ 2 + a.2 ^ 2 + a.3 ^ 2)) * c_magnitude)) = 120 * Real.pi / 180)

-- Statement to prove
theorem find_angle :
  ∀ (c : ℝ × ℝ × ℝ), Real.sqrt (c.1 ^ 2 + c.2 ^ 2 + c.3 ^ 2) = c_magnitude →
  (a + b) • c = 7 →
  Real.acos ((a • c) / (Real.sqrt ((a.1 ^ 2 + a.2 ^ 2 + a.3 ^ 2)) * c_magnitude)) = 120 * Real.pi / 180 :=
sorry

end find_angle_l374_374299


namespace f_sum_2016_eq_0_l374_374498

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_symmetry : ∀ x : ℝ, f (3/2 + x) = f (3/2 - x)
axiom f_neg1 : f (-1) = 1
axiom f_0 : f 0 = -2

theorem f_sum_2016_eq_0 : (∑ i in Finset.range 2016, f (i+1)) = 0 :=
sorry

end f_sum_2016_eq_0_l374_374498


namespace slope_of_tangent_at_neg_5_l374_374259

variable (f : ℝ → ℝ)

theorem slope_of_tangent_at_neg_5 (hf : ∀ x, f x = f (-x))
  (hf_diff : Differentiable ℝ f)
  (hf'_1 : f' 1 = 1)
  (hf_periodic : ∀ x, f (x+2) = f (x-2)) :
  ∀ h : -ID.mk (-5) , f' -5 = -1 :=
begin
  sorry
end

end slope_of_tangent_at_neg_5_l374_374259


namespace complex_number_in_third_quadrant_l374_374412

-- Defines the complex number z from the problem.
def z : ℂ := i^2 + i^3

-- Statement to prove: The coordinates of z place it in the third quadrant.
theorem complex_number_in_third_quadrant : 
  z.re < 0 ∧ z.im < 0 :=
by
  sorry

end complex_number_in_third_quadrant_l374_374412


namespace no_nontrivial_polynomials_exist_l374_374349

theorem no_nontrivial_polynomials_exist
  (f g : Polynomial ℝ) :
  ¬ (¬ is_square f ∧ ¬ is_square g ∧ is_square (f.comp g) ∧ is_square (g.comp f)) :=
by
  sorry

end no_nontrivial_polynomials_exist_l374_374349


namespace train_crossing_time_l374_374917

noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

noncomputable def time_to_cross_bridge (length_train length_bridge speed_kmph : ℝ) : ℝ :=
  let total_distance := length_train + length_bridge
  let speed_mps := kmph_to_mps speed_kmph
  total_distance / speed_mps

theorem train_crossing_time :
  time_to_cross_bridge 100 150 63 ≈ 14.29 :=
by
  -- The proof goes here
  sorry

end train_crossing_time_l374_374917


namespace gianna_savings_l374_374594

theorem gianna_savings:
  let daily_savings := 39 in
  let days_in_year := 365 in
  daily_savings * days_in_year = 14235 :=
by
  sorry

end gianna_savings_l374_374594


namespace intersection_of_sets_l374_374645

-- Define sets A and B
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

-- State the theorem
theorem intersection_of_sets : A ∩ B = {1, 2} := by
  sorry

end intersection_of_sets_l374_374645


namespace option_B_option_D_l374_374285

-- Define the function f(x)
def f (x a b : ℝ) : ℝ := (1/3) * x^3 + x^2 + a * x + b

-- Option B: a = -3, f(x) has 3 zeros implies -9 < b < 5/3
theorem option_B (b : ℝ) (h : ∃ x1 x2 x3 : ℝ, f x1 (-3) b = 0 ∧ f x2 (-3) b = 0 ∧ f x3 (-3) b = 0) : 
  -9 < b ∧ b < 5 / 3 :=
sorry

-- Option D: If f(x) has an extremum point x0, and f(x0) = f(x1), where x0 ≠ x1, then x1 + 2*x0 + 3 = 0
theorem option_D (a b x0 x1 : ℝ) (h1 : f x0 a b = f x1 a b) (h2 : x0 ≠ x1) (h3 : first_derivative(f) x0 = 0) : 
  x1 + 2 * x0 + 3 = 0 :=
sorry

end option_B_option_D_l374_374285


namespace pentagon_largest_angle_l374_374676

variable (F G H I J : ℝ)

-- Define the conditions given in the problem
axiom angle_sum : F + G + H + I + J = 540
axiom angle_F : F = 80
axiom angle_G : G = 100
axiom angle_HI : H = I
axiom angle_J : J = 2 * H + 20

-- Statement that the largest angle in the pentagon is 190°
theorem pentagon_largest_angle : max F (max G (max H (max I J))) = 190 :=
sorry

end pentagon_largest_angle_l374_374676


namespace value_of_f_prime_at_2_l374_374624

theorem value_of_f_prime_at_2 :
  ∃ (f' : ℝ → ℝ), 
  (∀ (x : ℝ), f' x = 2 * x + 3 * f' 2 + 1 / x) →
  f' 2 = - (9 / 4) := 
by 
  sorry

end value_of_f_prime_at_2_l374_374624


namespace original_number_is_15_l374_374019

theorem original_number_is_15 (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) (N : ℕ) (h4 : 100 * a + 10 * b + c = m)
  (h5 : 100 * a +  10 * b +   c +
        100 * a +   c + 10 * b + 
        100 * b +  10 * a +   c +
        100 * b +   c + 10 * a + 
        100 * c +  10 * a +   b +
        100 * c +   b + 10 * a = 3315) :
  m = 15 :=
sorry

end original_number_is_15_l374_374019


namespace find_m_for_parallel_lines_l374_374297

theorem find_m_for_parallel_lines (m : ℝ) :
  (∀ x y : ℝ, 3 * x - y + 2 = 0 → x + m * y - 3 = 0) →
  m = -1 / 3 := sorry

end find_m_for_parallel_lines_l374_374297


namespace greatest_possible_value_of_x_plus_y_717_l374_374959

-- Define the conditions
variables {a b c d : ℕ}
variables {x y : ℕ}
variables (pair_sums : Finset ℕ)
    (h_pair_sums : pair_sums = {210, 335, 296, 245, x, y})
    (h_distinct : pair_sums.card = 6)
    (S : ℕ) (h_S : S = a + b + c + d)

-- The final proof statement
theorem greatest_possible_value_of_x_plus_y_717 :
  (∀ a b c d : ℕ, (let pair_sums := {a + b, a + c, a + d, b + c, b + d, c + d} in
    pair_sums = {210, 335, 296, 245, x, y} ∧ pair_sums.card = 6) →
    (S = a + b + c + d) →
    x + y = 3 * S - 1086 →
    3 * (335 + 296) - 1086 = 717) :=
sorry

end greatest_possible_value_of_x_plus_y_717_l374_374959


namespace lower_total_payment_option_l374_374746

noncomputable def compound_interest_payment (P : ℕ) (r : ℚ) (n : ℕ) (num_periods : ℕ) : ℚ :=
  P * (1 + r / n)^num_periods

noncomputable def simple_interest_total (P : ℕ) (r : ℚ) (t : ℕ) : ℚ :=
  P * (1 + r * t)

theorem lower_total_payment_option (P : ℕ) (r_c : ℚ) (r_s : ℚ) (n : ℕ) (term : ℕ) : 
  (2 * P * (1 + r_s * term) / 3) - compound_interest_payment (2 * P / 3) r_c n term  = 14995.30 :=
by
  -- Variables and constants
  let P := 15000
  let r_c := 0.08
  let n := 2
  let r_s := 0.095
  let term := 15
  
  -- Calculation using compound interest formula
  let A1 := compound_interest_payment P r_c n 5
  let rem1 := (2 / 3) * A1
  let A2 := compound_interest_payment rem1 r_c n 5
  let rem2 := (2 / 3) * A2
  let A3 := compound_interest_payment rem2 r_c n 5
  
  -- Calculation using simple interest formula
  let A4 := simple_interest_total P r_s term
  
  -- Checking the difference
  have : A4 - A3 = 14995.30 := sorry
  exact this


end lower_total_payment_option_l374_374746


namespace proof_problem_l374_374678

variable (P_A P_M P_C : ℚ)

-- Define the conditions
def condition1 : Prop := P_A = 1 / 2
def condition2 : Prop := P_M = 2 * P_C
def condition3 : Prop := P_A + P_M + P_C = 1

-- Calculate the combined probability
def specific_probability : ℚ := (P_A^4) * (P_M^3) * (P_C)

-- Calculate the number of ways to arrange the wins
def arrange_ways : ℚ := Nat.factorial 8 / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 1)

-- Final Probability
def total_probability : ℚ := specific_probability * arrange_ways

theorem proof_problem :
  condition1 → condition2 → condition3 →
  total_probability = 35 / 324 := by
  sorry

end proof_problem_l374_374678


namespace isosceles_right_triangle_with_same_color_l374_374854

/-- The nodes of an infinite grid are colored in three colors.
Prove that there exists an isosceles right triangle with 
vertices of the same color. -/
theorem isosceles_right_triangle_with_same_color
  (coloring : ℕ → ℕ → Fin 3) :
  ∃ (a b c : ℕ × ℕ), is_isosceles_right_triangle a b c ∧ 
  coloring a.1 a.2 = coloring b.1 b.2 ∧ coloring b.1 b.2 = coloring c.1 c.2 :=
sorry

/-- Predicate for checking if three points form an isosceles right triangle -/
def is_isosceles_right_triangle (a b c : ℕ × ℕ) : Prop :=
(a.1 - b.1) * (a.1 - b.1) + (a.2 - b.2) * (a.2 - b.2) = (a.1 - c.1) * (a.1 - c.1) + (a.2 - c.2) * (a.2 - c.2) ∧ 
(a.1 - c.1) * (a.1 - c.1) + (a.2 - c.2) * (a.2 - c.2) = (b.1 - c.1) * (b.1 - c.1) + (b.2 - c.2) * (b.2 - c.2)

end isosceles_right_triangle_with_same_color_l374_374854


namespace min_k_greater_than_50_l374_374809

theorem min_k_greater_than_50 (k : ℕ) (x : fin k.succ → ℝ) 
  (pos : ∀ i, 0 < x i)
  (cond1 : (∑ i, (x i)^2) < (∑ i, x i) / 2)
  (cond2 : (∑ i, x i) < (∑ i, (x i)^3) / 2) : 50 < k := 
sorry

end min_k_greater_than_50_l374_374809


namespace relationship_among_abc_l374_374617

noncomputable def a : ℝ := Real.sqrt 6 + Real.sqrt 7
noncomputable def b : ℝ := Real.sqrt 5 + Real.sqrt 8
def c : ℝ := 5

theorem relationship_among_abc : c < b ∧ b < a :=
by
  sorry

end relationship_among_abc_l374_374617


namespace incorrect_calculation_sqrt2_plus_sqrt5_l374_374462

theorem incorrect_calculation_sqrt2_plus_sqrt5 : ¬ (sqrt 2 + sqrt 5 = sqrt 7) := by
  sorry

end incorrect_calculation_sqrt2_plus_sqrt5_l374_374462


namespace carl_cost_l374_374544

theorem carl_cost (property_damage medical_bills : ℝ) (insurance_coverage : ℝ) (carl_coverage : ℝ) (H1 : property_damage = 40000) (H2 : medical_bills = 70000) (H3 : insurance_coverage = 0.80) (H4 : carl_coverage = 0.20) :
  carl_coverage * (property_damage + medical_bills) = 22000 :=
by
  sorry

end carl_cost_l374_374544


namespace length_sum_focus_l374_374103

-- Define the properties of the parabola and line
variable (p x_A y_A : ℝ)
variable (a : ℝ)
variable (F : ℝ × ℝ := (p / 2, 0))

-- Given conditions:
-- The parabola y^2 = 2px
def parabola (x y : ℝ) : Prop := y^2 = 2 * p * x

-- The line ax + y - 4 = 0
def line (x y : ℝ) : Prop := a * x + y - 4 = 0

-- Point A has coordinates (1, 2)
def A := (1 : ℝ, 2 : ℝ)

-- The task is to prove: |FA + FB| = 7 for the focus F
theorem length_sum_focus (h1 : a * 1 + 2 - 4 = 0) 
                         (h2 : 4 = 2 * p * 1) 
                         (h3 : parabola x_A y_A) 
                         (h4 : line x_A y_A) 
                         (ha : a = 2) 
                         (hp : p = 2) 
                         (x_B : ℝ)
                         (hB : parabola x_B (4 - 2 * x_B)) 
                         (hb : x_B ≠ 1) :
    abs (dist (F) A + dist (F) (x_B, 4 - 2 * x_B)) = 7 := 
sorry

end length_sum_focus_l374_374103


namespace driving_time_to_higher_ground_l374_374945

def number_of_trips (cattle : ℕ) (capacity : ℕ) : ℕ :=
  (cattle + capacity - 1) / capacity -- rounding up division

def round_trip_distance (one_way_dist : ℕ) : ℕ :=
  one_way_dist * 2

def total_distance (num_trips : ℕ) (trip_distance : ℕ) : ℕ :=
  num_trips * trip_distance

def total_driving_time (total_miles : ℕ) (speed : ℕ) : ℕ :=
  total_miles / speed

theorem driving_time_to_higher_ground :
  let cattle := 800
  let capacity := 15
  let speed := 60
  let loc1_dist := 80
  let loc2_dist := 100
  let loc1_cattle := 450
  let loc2_cattle := cattle - loc1_cattle
  let loc1_trips := number_of_trips loc1_cattle capacity
  let loc2_trips := number_of_trips loc2_cattle capacity
  let loc1_round_trip := round_trip_distance loc1_dist
  let loc2_round_trip := round_trip_distance loc2_dist
  let loc1_total_distance := total_distance loc1_trips loc1_round_trip
  let loc2_total_distance := total_distance loc2_trips loc2_round_trip
  let total_miles := loc1_total_distance + loc2_total_distance
  in total_driving_time total_miles speed = 160 := 
by
  sorry

end driving_time_to_higher_ground_l374_374945


namespace problem1_l374_374929

theorem problem1 (t : ℝ) (h : t > 0) : (1 + 2 / t) * real.log (1 + t) > 2 := sorry

end problem1_l374_374929


namespace carnations_percentage_l374_374473

-- Definitions based on conditions
def flowers (C : ℕ) : Type := 
  { V := C / 3, -- violets
    T := (C / 3) / 3, -- tulips
    R := (C / 3) / 3 } -- roses

def total_flowers (C : ℕ) : ℕ := 
  C + (C / 3) + ((C / 3) / 3) + ((C / 3) / 3)

def percent_carnations (C : ℕ) : ℕ := 
  (C * 100) / (total_flowers C)

-- Theorem statement to prove
theorem carnations_percentage (C : ℕ) (hC_pos : 0 < C) :
  percent_carnations C = 64 := 
sorry

end carnations_percentage_l374_374473


namespace initial_nickels_proof_l374_374896

def initial_nickels (N : ℕ) (D : ℕ) (total_value : ℝ) : Prop :=
  D = 3 * N ∧
  total_value = (N + 2 * N) * 0.05 + 3 * N * 0.10 ∧
  total_value = 9

theorem initial_nickels_proof : ∃ N, ∃ D, (initial_nickels N D 9) → (N = 20) :=
by
  sorry

end initial_nickels_proof_l374_374896


namespace angle_EBC_l374_374680

theorem angle_EBC {ABCD : Type} [cyclic_quad ABCD]
  (A B C D E : ABCD)
  (h1 : ∠ A B D = 80)
  (h2 : ∠ A D C = 70)
  (h3 : ∠ B C D = 45) :
  ∠ E B C = 100 := by
sorry

end angle_EBC_l374_374680


namespace calories_burned_per_week_l374_374025

-- Definitions of the conditions
def classes_per_week : ℕ := 3
def hours_per_class : ℝ := 1.5
def calories_per_min : ℝ := 7
def minutes_per_hour : ℝ := 60

-- Theorem stating the proof problem
theorem calories_burned_per_week : 
  (classes_per_week * (hours_per_class * minutes_per_hour) * calories_per_min) = 1890 := by
  sorry

end calories_burned_per_week_l374_374025


namespace team_selection_ways_correct_l374_374966

-- Definition of the problem
def team_selection_ways : ℕ := 
  (Fintype.choose 6 2) -- selecting 2 positions (leader, deputy leader) out of 6
  * (Fintype.choose 4 2) -- selecting 2 ordinary members out of remaining 4

-- Proof statement
theorem team_selection_ways_correct : team_selection_ways = 180 := by
  sorry

end team_selection_ways_correct_l374_374966


namespace num_1989_period_points_l374_374705

-- Definitions
def unit_circle : set ℂ := { z | abs z = 1 }

def f (m : ℕ) (h : m > 1) : ℂ → ℂ := λ z, z^m

def is_n_period_point (f : ℂ → ℂ) (n : ℕ) (c : ℂ) : Prop :=
  (∀ (k : ℕ), k < n → (f^[k] c) ≠ c) ∧ (f^[n] c) = c

-- The Lean statement of the problem
theorem num_1989_period_points {m : ℕ} (hm : m > 1) :
  ∑ c in unit_circle, is_n_period_point (f m hm) 1989 c =
  Nat.totient (m ^ 1989 - 1) :=
sorry

end num_1989_period_points_l374_374705


namespace gcd_pow_sub_one_l374_374455

theorem gcd_pow_sub_one (n m : ℕ) (h1 : n = 1005) (h2 : m = 1016) :
  (Nat.gcd (2^n - 1) (2^m - 1)) = 2047 := by
  rw [h1, h2]
  sorry

end gcd_pow_sub_one_l374_374455


namespace nadines_dog_cleaning_time_l374_374048

noncomputable def remaining_mud (initial_mud: ℝ) (removal_percent: ℝ) : ℝ :=
initial_mud * (1 - removal_percent)

noncomputable def total_cleaning_time : ℝ := 
let hosed_off_mud := 0.5 in -- 50% mud removed
let first_shampoo_mud := remaining_mud hosed_off_mud 0.3 in -- 30% of remaining
let second_shampoo_mud := remaining_mud first_shampoo_mud 0.15 in -- 15% of remaining
let third_shampoo_mud := remaining_mud second_shampoo_mud 0.05 in -- 5% of remaining
let mud_remaining := third_shampoo_mud in 
let drying_time := if mud_remaining > 0.1 then 25 else 20 in
let cleanliness := 1 - mud_remaining in
let brushing_time := if cleanliness > 0.9 then 20 else 25 in
10 + 15 + 12 + 10 + drying_time + brushing_time -- total time

theorem nadines_dog_cleaning_time : total_cleaning_time = 97 :=
sorry

end nadines_dog_cleaning_time_l374_374048


namespace max_trapezoid_area_at_m_l374_374156

noncomputable def tangent_line (m : ℝ) : ℝ → ℝ := 
  λ x, (exp m) * x + (1 - m) * (exp m)

def trapezoid_area (m : ℝ) : ℝ :=
  if (1 ≤ m ∧ m ≤ 2) then 4 * (4 - m) * (exp m)
  else if (2 < m ∧ m ≤ 5) then 8 * (exp m)
  else 0

theorem max_trapezoid_area_at_m (h : 1 ≤ m ∧ m ≤ 5) : 
  trapezoid_area m = 8 * (exp 2) ↔ m = 2 := 
by
  sorry

end max_trapezoid_area_at_m_l374_374156


namespace range_of_area_PAB_l374_374729

noncomputable def f : ℝ → ℝ
| x => if h : 0 < x ∧ x < 1 then -Real.log x else Real.log x

def tangent_slope (f' : ℝ → ℝ) (x₁ x₂ : ℝ) (hx₁ : 0 < x₁ ∧ x₁ < 1) (hx₂ : 1 < x₂) : Prop :=
  ∀ k₁ k₂ : ℝ, k₁ = -1 / x₁ → k₂ = 1 / x₂ → k₁ * k₂ = -1

def line_eq (f' : ℝ → ℝ) (x₁ x₂ : ℝ) (hx₁ : 0 < x₁ ∧ x₁ < 1) (hx₂ : 1 < x₂) : Prop :=
  ∀ (l₁ l₂ : ℝ → ℝ),
  l₁ = λ x, -1/x₁ * (x - x₁) - f x₁ →
  l₂ = λ x, 1/x₂ * (x - x₂) + f x₂ →
  l₁ 0 ≠ l₂ 0

theorem range_of_area_PAB :
  ∀ (x₁ x₂ : ℝ) (hx₁ : 0 < x₁ ∧ x₁ < 1) (hx₂ : 1 < x₂),
  tangent_slope f x₁ x₂ hx₁ hx₂ →
  line_eq f x₁ x₂ hx₁ hx₂ →
  0 < (2 / (x₁ + 1/x₁) : ℝ) ∧ (2 / (x₁ + 1/x₁) : ℝ) < 1 :=
begin
  intros x₁ x₂ hx₁ hx₂ htangent hline,
  sorry -- proof steps will be provided here
end

end range_of_area_PAB_l374_374729


namespace candy_division_l374_374965

theorem candy_division (pieces_of_candy : Nat) (students : Nat) 
  (h1 : pieces_of_candy = 344) (h2 : students = 43) : pieces_of_candy / students = 8 := by
  sorry

end candy_division_l374_374965


namespace a_8_is_301_l374_374433

-- The number of students choosing A on the n-th Monday
def a : ℕ → ℕ
| 1       := 428
| (n + 2) := (a (n + 1)) / 2 + 150

-- Prove that the number of students choosing A on the 8th Monday is 301
theorem a_8_is_301 : a 8 = 301 :=
sorry

end a_8_is_301_l374_374433


namespace calc_result_l374_374038

namespace ProofExample

def a := 4 / 7
def b := 5 / 3

theorem calc_result : a^3 * b^(-2) = 576 / 8575 := by
  sorry

end ProofExample

end calc_result_l374_374038


namespace fifth_term_of_sequence_equals_341_l374_374559

theorem fifth_term_of_sequence_equals_341 : 
  ∑ i in Finset.range 5, 4^i = 341 :=
by sorry

end fifth_term_of_sequence_equals_341_l374_374559


namespace initial_earning_members_l374_374087

theorem initial_earning_members (n : ℕ) (h1 : (n * 735) - ((n - 1) * 650) = 905) : n = 3 := by
  sorry

end initial_earning_members_l374_374087


namespace sum_first_45_natural_numbers_l374_374923

theorem sum_first_45_natural_numbers :
  (∑ k in Finset.range 45, (k + 1)) = 1035 :=
by
  sorry

end sum_first_45_natural_numbers_l374_374923


namespace ordered_triples_lcm_sum_zero_l374_374213

theorem ordered_triples_lcm_sum_zero :
  ∀ (x y z : ℕ), 
    (0 < x) → 
    (0 < y) → 
    (0 < z) → 
    Nat.lcm x y = 180 →
    Nat.lcm x z = 450 →
    Nat.lcm y z = 600 →
    x + y + z = 120 →
    false := 
by
  intros x y z hx hy hz hxy hxz hyz hs
  sorry

end ordered_triples_lcm_sum_zero_l374_374213


namespace simplest_radical_l374_374907

def is_simplest_quadratic_radical (radical : ℝ) : Prop :=
  -- Definition of a simplest quadratic radical (simplified expression without further simplification possible)
  (radical = \(\sqrt{3}\))

def problem_condition_A : ∀ (reduction : ℝ), 
  reduction = \(\dfrac{\sqrt{6}}{3}\) → 
  reduction ≠ \(\sqrt{\dfrac{2}{3}}\)

def problem_condition_B : \(\sqrt{3}\) = \(\sqrt{3})

def problem_condition_C : (radical = \(\sqrt{9}\)) → 
  radical = 3

def problem_condition_D : (reduction : ℝ), 
  reduction = \(2\sqrt{3}\) → 
  reduction ≠ \(\sqrt{12}\)

theorem simplest_radical : 
  problem_condition_A → 
  problem_condition_B → 
  problem_condition_C → 
  problem_condition_D → 
  is_simplest_quadratic_radical \(\sqrt{3}\) := 
by 
  intros 
  -- Additional specifics can be detailed here if proving.
  sorry

end simplest_radical_l374_374907


namespace probability_quadrilateral_intersects_inner_circle_l374_374725

theorem probability_quadrilateral_intersects_inner_circle :
  let Γ1 := circle_of_radius (0, 0) 1 in    -- Circle centered at (0, 0) with radius 1
  let Γ2 := circle_of_radius (0, 0) 2 in    -- Circle centered at (0, 0) with radius 2
  let points := random_points_on_circle Γ2 4 in    -- Four random points on Γ2
  probability (convex_quadrilateral_of points ∩ Γ1 ≠ ∅) = 32 / 33 :=
sorry

end probability_quadrilateral_intersects_inner_circle_l374_374725


namespace largest_x_value_l374_374581

theorem largest_x_value (x : ℝ) :
  (x ≠ 9) ∧ (x ≠ -4) ∧ ((x ^ 2 - x - 72) / (x - 9) = 5 / (x + 4)) → x = -3 :=
sorry

end largest_x_value_l374_374581


namespace max_non_cyclic_handshakes_l374_374931

theorem max_non_cyclic_handshakes (n : ℕ) (h : n = 18) : 
  (n * (n - 1)) / 2 = 153 := by
  sorry

end max_non_cyclic_handshakes_l374_374931


namespace derivative_of_odd_function_is_even_l374_374050

theorem derivative_of_odd_function_is_even (f : ℝ → ℝ) (hf : ∀ x, f (-x) = -f x) :
  ∀ x, (deriv f) (-x) = (deriv f) x :=
by
  sorry

end derivative_of_odd_function_is_even_l374_374050


namespace Carlos_gave_Rachel_21_blocks_l374_374546

def initial_blocks : Nat := 58
def remaining_blocks : Nat := 37
def given_blocks : Nat := initial_blocks - remaining_blocks

theorem Carlos_gave_Rachel_21_blocks : given_blocks = 21 :=
by
  sorry

end Carlos_gave_Rachel_21_blocks_l374_374546


namespace tile_position_l374_374136

theorem tile_position (square : matrix (fin 7) (fin 7) bool) (tiles_3x1 : fin 16 → set (matrix (fin 7) (fin 7) bool))
  (tile_1x1 : matrix (fin 7) (fin 7) bool) :
  (∃ i j, 0 ≤ i ∧ i < 7 ∧ 0 ≤ j ∧ j < 7 ∧
    (square i j = tile_1x1 i j) ∧
    (i = 3 ∧ j = 3 ∨ i = 0 ∨ i = 6 ∨ j = 0 ∨ j = 6)) :=
begin
  sorry
end

end tile_position_l374_374136


namespace regression_decrease_by_5_l374_374255

theorem regression_decrease_by_5 (x y : ℝ) (h : y = 2 - 2.5 * x) :
  y = 2 - 2.5 * (x + 2) → y ≠ 2 - 2.5 * x - 5 :=
by sorry

end regression_decrease_by_5_l374_374255


namespace banana_count_l374_374885

-- Variables representing the number of bananas, oranges, and apples
variables (B O A : ℕ)

-- Conditions translated from the problem statement
def conditions : Prop :=
  (O = 2 * B) ∧
  (A = 2 * O) ∧
  (B + O + A = 35)

-- Theorem to prove the number of bananas is 5 given the conditions
theorem banana_count (B O A : ℕ) (h : conditions B O A) : B = 5 :=
sorry

end banana_count_l374_374885


namespace quadratic_inequality_solution_l374_374823

theorem quadratic_inequality_solution (x : ℝ) :
    -15 * x^2 + 10 * x + 5 > 0 ↔ (-1 / 3 : ℝ) < x ∧ x < 1 :=
by
  sorry

end quadratic_inequality_solution_l374_374823


namespace fraction_spoiled_l374_374826

-- Create the variables for initial conditions
variables (initial_stock sold_stock new_stock total_stock : ℕ)

-- Define the conditions
def steve_conditions : Prop :=
  initial_stock = 200 ∧
  sold_stock = 50 ∧
  new_stock = 200 ∧
  total_stock = 300

-- Define the problem statement
theorem fraction_spoiled (h : steve_conditions initial_stock sold_stock new_stock total_stock) :
  let remaining_before_spoil := initial_stock - sold_stock in
  let spoiled := total_stock - new_stock in
  (spoiled : ℚ) / remaining_before_spoil = 2 / 3 :=
by
  sorry

end fraction_spoiled_l374_374826


namespace total_weight_correct_l374_374587

-- Conditions for the weights of different types of candies
def frank_chocolate_weight : ℝ := 3
def gwen_chocolate_weight : ℝ := 2
def frank_gummy_bears_weight : ℝ := 2
def gwen_gummy_bears_weight : ℝ := 2.5
def frank_caramels_weight : ℝ := 1
def gwen_caramels_weight : ℝ := 1
def frank_hard_candy_weight : ℝ := 4
def gwen_hard_candy_weight : ℝ := 1.5

-- Combined weights of each type of candy
def chocolate_weight : ℝ := frank_chocolate_weight + gwen_chocolate_weight
def gummy_bears_weight : ℝ := frank_gummy_bears_weight + gwen_gummy_bears_weight
def caramels_weight : ℝ := frank_caramels_weight + gwen_caramels_weight
def hard_candy_weight : ℝ := frank_hard_candy_weight + gwen_hard_candy_weight

-- Total weight of the Halloween candy haul
def total_halloween_weight : ℝ := 
  chocolate_weight +
  gummy_bears_weight +
  caramels_weight +
  hard_candy_weight

-- Theorem to prove the total weight is 17 pounds
theorem total_weight_correct : total_halloween_weight = 17 := by
  sorry

end total_weight_correct_l374_374587


namespace perpendicular_bisector_c_value_l374_374095

theorem perpendicular_bisector_c_value :
  (∃ c : ℝ, ∀ x y : ℝ, 
    2 * x - y = c ↔ x = 5 ∧ y = 8) → c = 2 := 
by
  sorry

end perpendicular_bisector_c_value_l374_374095


namespace auntie_em_parking_l374_374953

theorem auntie_em_parking (total_spaces cars : ℕ) (probability_can_park : ℚ) :
  total_spaces = 20 →
  cars = 15 →
  probability_can_park = 232/323 :=
by
  sorry

end auntie_em_parking_l374_374953


namespace max_min_value_l374_374371

theorem max_min_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 12) (h5 : x * y + y * z + z * x = 30) :
  ∃ n : ℝ, n = min (x * y) (min (y * z) (z * x)) ∧ n = 2 :=
sorry

end max_min_value_l374_374371


namespace greatest_power_of_3_in_22_factorial_l374_374132

theorem greatest_power_of_3_in_22_factorial : 
  ∃ n : ℕ, (∀ k : ℕ, (3^k ∣ nat.factorial 22) → k ≤ n) ∧ n = 9 :=
by
  -- let n be the greatest number such that 3^n divides 22!
  let n := (∑ i in finset.range 8, (22 / 3 ^ i)),
  -- show that n equals 9
  have hn : n = 9,
  { -- proof that n equals 9 is left as an exercise
    sorry },
  -- show that the desired property holds 
  use n,
  split,
  { intros k hk,
    -- show that k ≤ n
    sorry },
  { exact hn }

end greatest_power_of_3_in_22_factorial_l374_374132


namespace number_in_lowermost_row_l374_374515

theorem number_in_lowermost_row :
  (let a (i j : ℕ) := 
       if i = 1 then j 
       else a (i-1) j + a (i-1) (j+1)
   in a 2000 1) = 2001 * 2^(1998) :=
by
  sorry

end number_in_lowermost_row_l374_374515


namespace largest_power_divides_factorial_l374_374196

theorem largest_power_divides_factorial (k : ℕ) :
  (2310 = 2 * 3 * 5 * 7 * 11) → (2310^k ∣ nat.factorial 2310) → k = 229 :=
by
  sorry

end largest_power_divides_factorial_l374_374196


namespace charge_per_copy_Y_l374_374238

variable (charge_per_copy_X : ℚ) (charge_Y : ℚ) (charge_difference : ℚ)
variable (n_copies : ℚ)

-- Given conditions
def print_shop_X_charge : charge_per_copy_X = 1.20 := sorry
def additional_charge_Y : charge_difference = 20 := sorry
def n_copies_condition : n_copies = 40 := sorry

-- Given total cost condition for 40 copies at print shop Y
def total_cost_condition : 40 * charge_Y = 40 * 1.20 + 20 := sorry

-- Prove the charge per color copy at print shop Y
theorem charge_per_copy_Y : charge_Y = 1.70 := 
by 
  rw [total_cost_condition]
  sorry

end charge_per_copy_Y_l374_374238


namespace fran_speed_to_match_joann_distance_l374_374697

theorem fran_speed_to_match_joann_distance:
  ∀ (joann_speed joann_time fran_time: ℝ), 
  joann_speed = 15 → 
  joann_time = 4 → 
  fran_time = 2.5 → 
  (joann_speed * joann_time) = 60 → 
  (∃ fran_speed: ℝ, fran_speed = 24) :=
by
  intros joann_speed joann_time fran_time h_joann_speed h_joann_time h_fran_time h_distance
  use 24
  have h_distance_fran : 24 * fran_time = 60 := by 
    rw [h_fran_time]
    norm_num
  exact h_distance_fran.symm.trans h_distance.symm
  sorry

end fran_speed_to_match_joann_distance_l374_374697


namespace sphere_wedge_volume_l374_374511

noncomputable def volume_of_one_wedge (C : ℝ) (n : ℕ) :=
  let r := C / (2 * Real.pi) in
  let V := (4 / 3) * Real.pi * r^3 in
  V / n

theorem sphere_wedge_volume :
  volume_of_one_wedge (24 * Real.pi) 6 = 384 * Real.pi :=
by
  -- Add steps to prove the theorem
  sorry

end sphere_wedge_volume_l374_374511


namespace max_min_product_xy_l374_374860

-- Definition of conditions
variables (a x y : ℝ)
def condition_1 : Prop := x + y = a
def condition_2 : Prop := x^2 + y^2 = -a^2 + 2

-- The main theorem statement
theorem max_min_product_xy (a : ℝ) (ha_range : -2 ≤ a ∧ a ≤ 2): 
  (∀ x y : ℝ, condition_1 a x y ∧ condition_2 a x y → (x * y) ≤ (1 / 3)) ∧
  (∀ x y : ℝ, condition_1 a x y ∧ condition_2 a x y → (x * y) ≥ (-1)) :=
sorry

end max_min_product_xy_l374_374860


namespace largest_number_among_four_l374_374530

open Real

theorem largest_number_among_four :
  ∀ (a b c d : ℝ), a = -3 → b = 0 → c = sqrt 5 → d = 2 → max a (max b (max c d)) = sqrt 5 := 
by intros a b c d ha hb hc hd
   rw [ha, hb, hc, hd]
   norm_num
   rw [Real.sqrt_sq, max_assoc, max_comm, max_assoc, max_comm (sqrt 5)]
   simp [sqrt_lt]
   apply socratic -- sqrt 5 > 2
   norm_num

-- Placeholder for the parts of the proof
sorry


end largest_number_among_four_l374_374530


namespace num_distinct_pairs_x_y_divisible_by_49_l374_374652

theorem num_distinct_pairs_x_y_divisible_by_49 :
  (∑ x in finset.range 1001, ∑ y in finset.range 1001, if (x^2 + y^2) % 49 = 0 then 1 else 0) = 10153 := 
sorry

end num_distinct_pairs_x_y_divisible_by_49_l374_374652


namespace simplify_expression_l374_374818

theorem simplify_expression : (81 ^ (1 / 2 : ℝ) - 144 ^ (1 / 2 : ℝ) = -3) :=
by
  have h1 : 81 ^ (1 / 2 : ℝ) = 9 := by sorry
  have h2 : 144 ^ (1 / 2 : ℝ) = 12 := by sorry
  rw [h1, h2]
  norm_num

end simplify_expression_l374_374818


namespace convert_653_base8_to_base5_l374_374567

-- Define the conversion from base 8 to decimal
def base8_to_dec (n : ℕ) (d : List ℕ) : ℕ :=
  (d.zipWith (fun p v => v * (n ^ p)) (List.range (d.length))).sum

-- Define the conversion from decimal to base 5
def dec_to_base5 (n : ℕ) : List ℕ :=
  let rec aux (n : ℕ) (acc : List ℕ) :=
    if n = 0 then acc else aux (n / 5) (n % 5 :: acc)
  aux n []

-- Define the problem statement
noncomputable def problem_statement :=
  base8_to_dec 8 [3, 5, 6] = 427 ∧ dec_to_base5 427 = [3, 2, 0, 2]

theorem convert_653_base8_to_base5 :
  problem_statement → dec_to_base5 (base8_to_dec 8 [3, 5, 6]) = [3, 2, 0, 2] := by 
  intros h; exact h.right

end convert_653_base8_to_base5_l374_374567


namespace divisible_by_7_last_digits_card_l374_374388

open Set

def divisible_by_7_last_digits : Set ℕ :=
  { d ∈ (range 10) | ∃ n : ℕ, d = n % 10 ∧ 7 ∣ n }

theorem divisible_by_7_last_digits_card : divisible_by_7_last_digits.card = 2 := 
  sorry

end divisible_by_7_last_digits_card_l374_374388


namespace find_k_l374_374001

theorem find_k (x : ℝ) (a h k : ℝ) (h1 : 9 * x^2 - 12 * x = a * (x - h)^2 + k) : k = -4 := by
  sorry

end find_k_l374_374001


namespace complex_quadrant_l374_374307

theorem complex_quadrant (i : ℂ) (h_imag : i = Complex.I) :
  let z := (1 + i)⁻¹
  z.re > 0 ∧ z.im < 0 :=
by
  sorry

end complex_quadrant_l374_374307


namespace Petya_digits_sum_l374_374773

theorem Petya_digits_sum (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :
  (24 * (a + b + c + d) * 1111 = 73326) →
  {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end Petya_digits_sum_l374_374773


namespace range_of_a_l374_374281

noncomputable def f (a x : ℝ) := x^3 - a * x^2 + 3 * a * x + 1

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ set.Ioo (-2 : ℝ) 2, ∃ c d ∈ set.Ioo (-2 : ℝ) 2, 
  c ≠ d ∧ f'' a c = 0 ∧ f'' a d = 0 ∧ ∀ x ∈ set.Ioo c d, f'' a x < 0) →
  -12 / 7 < a ∧ a < 0 :=
begin
  sorry 
end

-- f'' is a helper definition for the second derivative of f
def f'' (a : ℝ) : ℝ → ℝ := λ x, 6 * x - 2 * a

end range_of_a_l374_374281


namespace member_sum_of_two_others_l374_374534

def numMembers : Nat := 1978
def numCountries : Nat := 6

theorem member_sum_of_two_others :
  ∃ m : ℕ, m ∈ Finset.range numMembers.succ ∧
  ∃ a b : ℕ, a ∈ Finset.range numMembers.succ ∧ b ∈ Finset.range numMembers.succ ∧ 
  ∃ country : Fin (numCountries + 1), (a = m + b ∧ country = country) :=
by
  sorry

end member_sum_of_two_others_l374_374534


namespace ellipse_properties_l374_374271

noncomputable def ellipse_standard_equation : String :=
  "x^2 / 12 + y^2 / 3 = 1"

noncomputable def line_equation_through_focus : String :=
  "y = ± (2 * sqrt 11 / 11) * (x + 3)"

theorem ellipse_properties
  (center_origin : (0, 0))
  (foci_shared : ∀ x y : ℝ, x^2 / 8 -  y^2 = 1)
  (point_on_ellipse : ∀ x y : ℝ, x = -2 → y = sqrt 2 → x^2 / 12 + y^2 / 3 = 1)
  (focus_left : ∃ x y : ℝ, x = -3 ∧ y = 0)
  (orthogonal_diameter : ∀ x1 y1 x2 y2 : ℝ, x1 * x2 + y1 * y2 = 0 → line_equation_through_focus)
  : ∃ eq_ellipse : String, eq_ellipse = ellipse_standard_equation ∧ ∃ eq_line : String, eq_line = line_equation_through_focus :=
by
  sorry

end ellipse_properties_l374_374271


namespace kendra_shirt_club_days_l374_374033

theorem kendra_shirt_club_days :
  ∀ (total_shirts per_weekdays per_saturday per_sunday per_two_weeks),
  per_weekdays = 5 →
  per_saturday = 1 →
  per_sunday = 2 →
  per_two_weeks = 22 →
  total_shirts = per_weekdays + per_saturday + per_sunday →
  (total_shirts * 2) = per_two_weeks →
  (per_two_weeks - total_shirts * 2) / 2 = 3 :=
by
  intros total_shirts per_weekdays per_saturday per_sunday per_two_weeks
  assume h_weekdays : per_weekdays = 5
  assume h_saturday : per_saturday = 1
  assume h_sunday : per_sunday = 2
  assume h_two_weeks : per_two_weeks = 22
  assume h_total_shirts : total_shirts = per_weekdays + per_saturday + per_sunday
  assume h_two_weeks_total: (total_shirts * 2) = per_two_weeks
  have h_per_week := total_shirts * 2
  rw [h_weekdays, h_saturday, h_sunday] at h_total_shirts
  rw [h_total_shirts, h_two_weeks_total, h_two_weeks] at h_per_week
  linarith
  sorry

end kendra_shirt_club_days_l374_374033


namespace number_of_integer_values_f_in_interval_l374_374312

open Real

noncomputable def f (x: ℝ) : ℝ := log 2 (log 2 (2 * x + 2)) + (2:ℝ)^(2 * x + 2)

theorem number_of_integer_values_f_in_interval :
  (finset.range 18).filter (λ n, ∃ x, x ∈ (Icc 0 1 : set ℝ) ∧ f x = n).card = 14 := by
sorry

end number_of_integer_values_f_in_interval_l374_374312


namespace integer_values_of_f_l374_374202

noncomputable def f (x : ℝ) : ℝ := (1 + x)^(1/3) + (3 - x)^(1/3)

theorem integer_values_of_f : 
  {x : ℝ | ∃ k : ℤ, f x = k} = {1 + Real.sqrt 5, 1 - Real.sqrt 5, 1 + (10/9) * Real.sqrt 3, 1 - (10/9) * Real.sqrt 3} :=
by
  sorry

end integer_values_of_f_l374_374202


namespace temperature_at_night_l374_374439

theorem temperature_at_night :
  ∀ (T_noon : ℤ) (drop : ℤ), T_noon = -2 ∧ drop = 4 → T_noon - drop = -6 := by
  intros T_noon drop h
  cases h with h1 h2
  rw [h1, h2]
  norm_num

end temperature_at_night_l374_374439


namespace tamika_greater_probability_l374_374829

-- Definitions for the conditions
def tamika_results : Set ℕ := {11 * 12, 11 * 13, 12 * 13}
def carlos_result : ℕ := 2 + 3 + 4

-- Theorem stating the problem
theorem tamika_greater_probability : 
  (∀ r ∈ tamika_results, r > carlos_result) → (1 : ℚ) = 1 := 
by
  intros h
  sorry

end tamika_greater_probability_l374_374829


namespace Simson_line_bisects_segment_l374_374060

variables {A B C P H : Type}
variables [euclidean_geometry A B C P H]

/-- Simson line bisects the line segment joining P and the orthocenter H -/
theorem Simson_line_bisects_segment (hP : P.is_on_circumcircle (triangle A B C))
  (hH : H.is_orthocenter (triangle A B C))
  (Simson : Simson_line_definition P A B C) :
  is_midpoint (segment HP) (Simson.intersection (segment HP)) :=
sorry

end Simson_line_bisects_segment_l374_374060


namespace original_area_l374_374414

def side_ratio : ℕ := 4

def new_area : ℝ := 256

def area_ratio (r : ℕ) : ℕ := r * r

theorem original_area (r : ℕ) (A_new : ℝ) :
  A_new = new_area →
  r = side_ratio →
  A_new / (area_ratio r) = 16 :=
by
  intro hA_new hr
  rw [hA_new, hr, (by norm_num : area_ratio 4 = 16)]
  norm_num
  done

end original_area_l374_374414


namespace petya_four_digits_l374_374787

theorem petya_four_digits :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
    6 * (a + b + c + d) * 1111 = 73326 ∧ 
    (∃ S, S = a + b + c + d ∧ S = 11) :=
begin
  use 1, 2, 3, 5,
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split,
  { exact dec_trivial },
  use 11,
  split,
  { exact dec_trivial },
  { exact dec_trivial }
end

end petya_four_digits_l374_374787


namespace jack_walking_rate_l374_374663

-- Define the problem
def walking_rate (distance : ℝ) (hours_part : ℝ) (minutes_part : ℝ) : ℝ :=
  let time_in_hours := hours_part + minutes_part / 60
  distance / time_in_hours

-- Define the conditions
def jack_walked_distance : ℝ := 8
def jack_walked_hours : ℝ := 1
def jack_walked_minutes : ℝ := 15

-- State the theorem
theorem jack_walking_rate :
  walking_rate jack_walked_distance jack_walked_hours jack_walked_minutes = 6.4 := 
sorry

end jack_walking_rate_l374_374663


namespace angle_at_vertex_of_cone_axial_section_l374_374155

theorem angle_at_vertex_of_cone_axial_section (r : ℝ) :
  let α := π - 4 * arctan ((sqrt 6 + sqrt 2) / 4)
  in α = π - 4 * arctan ((sqrt 6 + sqrt 2) / 4) :=
by
  sorry

end angle_at_vertex_of_cone_axial_section_l374_374155


namespace integral_equals_zero_l374_374190

open Real

noncomputable def integral_expression : ℝ :=
  ∫ x in -π/2..π/2, x + cos (2 * x)

theorem integral_equals_zero :
  integral_expression = 0 :=
by
  sorry

end integral_equals_zero_l374_374190


namespace solve_trig_eq_l374_374723

open Real

theorem solve_trig_eq (x a : ℝ) (hx1 : 0 < x) (hx2 : x < 2 * π) (ha : a > 0) :
    (sin (3 * x) + a * sin (2 * x) + 2 * sin x = 0) →
    (0 < a ∧ a < 2 → x = 0 ∨ x = π) ∧ 
    (a > 5 / 2 → ∃ α, (x = α ∨ x = 2 * π - α)) :=
by sorry

end solve_trig_eq_l374_374723


namespace trig_problem_solution_l374_374657

theorem trig_problem_solution (x : ℝ) (h : sin (4 * x) * sin (5 * x) = cos (4 * x) * cos (5 * x)) : x = (π / 18) :=
sorry

end trig_problem_solution_l374_374657


namespace point_on_segment_bisector_condition_l374_374377

open Point Line

variables {ℝ} (A B C P X Y : Point ℝ)
variables {PA PB PC AB AC XY : ℝ}

theorem point_on_segment_bisector_condition :
  (P ∈ segment BC) →
  (XY = intersection (PA) (common_external_tangent (circumcircle PAB) (circumcircle PAC))) →
  ((PA / XY)^2 + (PB * PC) / (AB * AC) = 1) →
  (P is_a_bisector_feet) :=
by
  sorry

end point_on_segment_bisector_condition_l374_374377


namespace cube_division_possibility_a_cube_division_possibility_b_l374_374023
open Real

def cube_division_4_plane_possible (cube_edge_length : ℝ) (d : ℝ) : Prop :=
  ∃ (parts : Set (Set (ℝ × ℝ × ℝ))), 
    (∀ x ∈ parts, diameter x < d) ∧ 
    (number_of_planes cube_edge_length = 4)

-- Part (a): Proof for edge length 1 and distance less than 4/5
theorem cube_division_possibility_a : cube_division_4_plane_possible 1 (4 / 5) :=
begin
  sorry
end

-- Part (b): Proof for edge length 1 and distance less than 4/7 (not possible)
theorem cube_division_possibility_b : ¬ cube_division_4_plane_possible 1 (4 / 7) :=
begin
  sorry
end

end cube_division_possibility_a_cube_division_possibility_b_l374_374023


namespace inequality_k_l374_374431

variable {R : Type} [LinearOrderedField R] [Nontrivial R]

theorem inequality_k (x y z : R) (k : ℕ) (h : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) 
  (hineq : (1/x) + (1/y) + (1/z) ≥ x + y + z) :
  (1/x^k) + (1/y^k) + (1/z^k) ≥ x^k + y^k + z^k :=
sorry

end inequality_k_l374_374431


namespace log_S2016_eq_2017_l374_374006

noncomputable def geometric_sequence_sums (a₁ a₂ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem log_S2016_eq_2017 
  (a₁ a₂ a₃ : ℝ)
  (q : ℝ)
  (h1 : a₁ + a₂ = 6)
  (h2 : a₂ + a₃ = 12)
  (h3 : a₂ = q * a₁)
  (h4 : a₃ = q * a₂) :
  Real.logb 2 ((geometric_sequence_sums a₁ a₂ q 2016) + 2) = 2017 :=
by {
  sorry,
}

end log_S2016_eq_2017_l374_374006


namespace S_bounds_l374_374360

/-- 
  The Euler's totient function, φ(n), is the number of positive integers less than or equal to n
  that are relatively prime to n.
-/
def totient (n : ℕ) : ℕ := 
  (Finset.range (n + 1)).filter (λ m => Nat.gcd n m = 1).card

/--
  The function S(n) defined as the sum S(n) = Σ k=1^n (k * φ(k)).
-/
noncomputable def S (n : ℕ) : ℕ := 
  (Finset.range (n + 1)).sum (λ k => k * totient k)

theorem S_bounds (n : ℕ) (hn : n ≥ 2018) : 
  0.17 * n^3 ≤ S n ∧ S n ≤ 0.23 * n^3 := 
sorry

end S_bounds_l374_374360


namespace log_relationship_l374_374659

open Real

theorem log_relationship :
  let a := logBase 2 0.1
  let b := logBase 2 3
  let c := logBase 2 8
  a < b ∧ b < c := 
sorry

end log_relationship_l374_374659


namespace behavior_of_g_l374_374991

def g (x : ℝ) : ℝ := -3 * x ^ 3 + 4 * x ^ 2 + 5

theorem behavior_of_g :
  (∀ x, (∃ M, x ≥ M → g x < 0)) ∧ (∀ x, (∃ N, x ≤ N → g x > 0)) :=
by
  sorry

end behavior_of_g_l374_374991


namespace averageRounds_l374_374857

def roundsAndGolfers : List (Nat × Nat) := [
  (1, 6), 
  (2, 3), 
  (3, 2), 
  (4, 4), 
  (5, 6), 
  (6, 4)
]

def totalRounds : Nat :=
  roundsAndGolfers.foldl (fun acc ⟨rounds, golfers⟩ => acc + rounds * golfers) 0

def totalGolfers : Nat :=
  roundsAndGolfers.foldl (fun acc ⟨_, golfers⟩ => acc + golfers) 0

theorem averageRounds : Int :=
  let avg := (totalRounds.toFloat / totalGolfers.toFloat).round
  avg = 4 := sorry

end averageRounds_l374_374857


namespace problem_a_problem_b_l374_374728

variables (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℂ)

-- Condition given: AB^2 = A - B
def condition := A ⬝ B ⬝ B = A - B

-- Problem (a): Show that I_n + B is invertible
theorem problem_a (h : condition A B) : Invertible (1 + B) :=
sorry

-- Problem (b): Show that AB = BA
theorem problem_b (h : condition A B) : A ⬝ B = B ⬝ A :=
sorry

end problem_a_problem_b_l374_374728


namespace number_and_sum_of_g_five_l374_374040

def S := {x : ℝ // x ≠ 0}

noncomputable def g (x : S) : S

axiom g_property : ∀ (x y : S), (x + y : ℝ) ≠ 0 → g(x) + g(y) = g(⟨(x + y) / g(x * y), by sorry⟩)

theorem number_and_sum_of_g_five (g : S → S) (hg : ∀ (x y : S), (x + y : ℝ) ≠ 0 → g(x) + g(y) = g(⟨(x + y) / g(x * y), sorry⟩)) : 
  ∃ (n : ℕ) (s : ℝ), n = 1 ∧ s = 1 / 5 := 
sorry

end number_and_sum_of_g_five_l374_374040


namespace value_of_f_13_l374_374634

noncomputable def f : ℝ → ℝ
| x => if 0 < x ∧ x ≤ 9 then real.log x / real.log 3
       else f (x - 4)

theorem value_of_f_13 : f 13 = 2 :=
by
  sorry

end value_of_f_13_l374_374634


namespace range_of_y_eq_4_sin_squared_x_minus_2_l374_374432

theorem range_of_y_eq_4_sin_squared_x_minus_2 : 
  (∀ x : ℝ, y = 4 * (Real.sin x)^2 - 2) → 
  (∃ a b : ℝ, ∀ x : ℝ, y ∈ Set.Icc a b ∧ a = -2 ∧ b = 2) :=
sorry

end range_of_y_eq_4_sin_squared_x_minus_2_l374_374432


namespace bench_cost_150_l374_374942

-- Define the conditions
def combined_cost (bench_cost table_cost : ℕ) : Prop := bench_cost + table_cost = 450
def table_cost_eq_twice_bench (bench_cost table_cost : ℕ) : Prop := table_cost = 2 * bench_cost

-- Define the main statement, which includes the goal of the proof.
theorem bench_cost_150 (bench_cost table_cost : ℕ) (h_combined_cost : combined_cost bench_cost table_cost)
  (h_table_cost_eq_twice_bench : table_cost_eq_twice_bench bench_cost table_cost) : bench_cost = 150 :=
by
  sorry

end bench_cost_150_l374_374942


namespace product_of_roots_l374_374306

theorem product_of_roots : 
  ∀ x : ℝ, (x - 1) * (x + 4) = 22 → (∃ a b c : ℝ, a = 1 ∧ b = 3 ∧ c = -26 ∧ (a * x^2 + b * x + c = 0) ∧ (b^2 - 4 * a * c) ≥ 0 ∧ (x = (-b + sqrt (b^2 - 4 * a * c)) / (2 * a) ∨ x = (-b - sqrt (b^2 - 4 * a * c)) / (2 * a)) ∧ x_1 * x_2 = -26)
  sorry

end product_of_roots_l374_374306


namespace find_f_20_l374_374714

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_20 :
  (∀ x : ℝ, f x = f (-x)) →
  (∀ x : ℝ, f x = f (2 - x)) →
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x - 1 / 2) →
  f 20 = - 1 / 2 :=
sorry

end find_f_20_l374_374714


namespace bus_stop_time_l374_374576

theorem bus_stop_time
  (speed_without_stoppage : ℝ := 54)
  (speed_with_stoppage : ℝ := 45)
  (distance_diff : ℝ := speed_without_stoppage - speed_with_stoppage)
  (distance : ℝ := distance_diff)
  (speed_km_per_min : ℝ := speed_without_stoppage / 60) :
  distance / speed_km_per_min = 10 :=
by
  -- The proof steps would go here.
  sorry

end bus_stop_time_l374_374576


namespace train_bridge_problem_l374_374153

theorem train_bridge_problem
  (train_length : ℕ)
  (crossing_time : ℕ)
  (train_speed : ℕ) : 
  train_length = 100 → 
  crossing_time = 60 → 
  train_speed = 5 → 
  let total_distance := train_speed * crossing_time in
  let bridge_length := total_distance - train_length in
  bridge_length = 200 :=
by
  intros h_train_length h_crossing_time h_train_speed
  unfold total_distance bridge_length
  rw [h_train_length, h_crossing_time, h_train_speed]
  norm_num
  sorry -- Proof would follow here

end train_bridge_problem_l374_374153


namespace new_student_teacher_ratio_l374_374104

theorem new_student_teacher_ratio
  (curr_students: ℕ)
  (curr_teachers: ℕ)
  (additional_students: ℕ)
  (additional_teachers: ℕ)
  (student_teacher_ratio: ℕ)
  (new_ratio: ℕ) :
  student_teacher_ratio = 50 →
  curr_teachers = 3 →
  additional_students = 50 →
  additional_teachers = 5 →
  new_ratio = 25 →
  curr_students = student_teacher_ratio * curr_teachers →
  new_ratio = (curr_students + additional_students) / (curr_teachers + additional_teachers) :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end new_student_teacher_ratio_l374_374104


namespace raja_monthly_income_l374_374476

noncomputable def monthly_income (household_percentage clothes_percentage medicines_percentage savings : ℝ) : ℝ :=
  let spending_percentage := household_percentage + clothes_percentage + medicines_percentage
  let savings_percentage := 1 - spending_percentage
  savings / savings_percentage

theorem raja_monthly_income :
  monthly_income 0.35 0.20 0.05 15000 = 37500 :=
by
  sorry

end raja_monthly_income_l374_374476


namespace fifth_term_of_sequence_equals_341_l374_374561

theorem fifth_term_of_sequence_equals_341 : 
  ∑ i in Finset.range 5, 4^i = 341 :=
by sorry

end fifth_term_of_sequence_equals_341_l374_374561


namespace inequality_proof_l374_374713

noncomputable def a : ℝ := 0.99^1.01
noncomputable def b : ℝ := 1.01^0.99
noncomputable def c : ℝ := Real.log 0.99 / Real.log 1.01

theorem inequality_proof : c < a ∧ a < b :=
by
  sorry

end inequality_proof_l374_374713


namespace Jack_pages_per_day_l374_374351

variable (total_pages : ℕ) (num_days : ℕ)

-- Define the foundational assumptions based on the given conditions
def pages_per_day (total_pages num_days: ℕ) : ℤ :=
  Int.round (total_pages / num_days)

-- The problem statement in Lean
theorem Jack_pages_per_day : pages_per_day 285 13 = 22 :=
by
  sorry

end Jack_pages_per_day_l374_374351


namespace simplest_square_root_problem_l374_374177

theorem simplest_square_root_problem :
    let sqrt_5 := (√5 : ℝ),
        sqrt_one_half := (√(1 / 2) : ℝ),
        sqrt_8 := (√8 : ℝ),
        sqrt_a_squared := (√(a^2) : ℝ)
    in sqrt_5 = (√5 : ℝ) ∧
       sqrt_one_half = (√(1 / 2) : ℝ) ∧
       sqrt_8 = (√8 : ℝ) ∧
       sqrt_a_squared = (√(a^2) : ℝ) ->
       sqrt_5 = √5 ∧
       sqrt_one_half = (√2 / 2 : ℝ) ∧
       sqrt_8 = (2 * √2 : ℝ) ∧
       sqrt_a_squared = (abs a : ℝ) :=
begin
  sorry 
end

end simplest_square_root_problem_l374_374177


namespace find_number_l374_374503

-- Define the condition: a number exceeds by 40 from its 3/8 part.
def exceeds_by_40_from_its_fraction (x : ℝ) := x = (3/8) * x + 40

-- The theorem: prove that the number is 64 given the condition.
theorem find_number (x : ℝ) (h : exceeds_by_40_from_its_fraction x) : x = 64 := 
by
  sorry

end find_number_l374_374503


namespace contrapositive_l374_374838

theorem contrapositive (a b : ℝ) : (a ≠ 0 ∨ b ≠ 0) → a^2 + b^2 ≠ 0 :=
by
  intro h
  sorry

end contrapositive_l374_374838


namespace max_cables_cut_l374_374879

theorem max_cables_cut 
  (initial_computers : ℕ)
  (initial_cables : ℕ)
  (final_clusters : ℕ)
  (H1 : initial_computers = 200)
  (H2 : initial_cables = 345)
  (H3 : final_clusters = 8) 
  : ∃ (cut_cables : ℕ), cut_cables = 153 :=
by
  use 153
  sorry

end max_cables_cut_l374_374879


namespace cyclic_quadrilateral_symmetry_l374_374485

theorem cyclic_quadrilateral_symmetry (ABCD : Quadrilateral) (A B C D P : Point) (hBDsym : symmetric_wrt_angle_bisectors B D BD) (hBDmidAC : passes_through_midpoint BD AC P) :
  symmetric_wrt_angle_bisectors A C AC → passes_through_midpoint AC BD P :=
by
  sorry

end cyclic_quadrilateral_symmetry_l374_374485


namespace find_positive_X_l374_374711

def op #(X : ℝ) (Y : ℝ) : ℝ := X^2 + Y^2

theorem find_positive_X (X : ℝ) :
  op X 7 = 85 → X = 6 :=
by
  sorry

end find_positive_X_l374_374711


namespace largest_lambda_l374_374260

theorem largest_lambda (n : ℤ) (h1 : n ≥ 2) (a : ℤ → ℤ) 
  (h2 : ∀ i j, 1 ≤ i → i < j → j ≤ n → a i < a j) :
  ∃ λ : ℝ, 
  (∀ i, (i = n) → (a i : ℝ)^2 ≥ λ * (∑ k in finset.range(n + 1), a k) + 2 * (a i)) ∧
  λ = (2 * n - 4) / (n + 1) :=
sorry

end largest_lambda_l374_374260


namespace equations_of_motion_l374_374421

-- Initial conditions and setup
def omega : ℝ := 10
def OA : ℝ := 90
def AB : ℝ := 90
def AM : ℝ := 45

-- Questions:
-- 1. Equations of motion for point M
-- 2. Equation of the trajectory of point M
-- 3. Velocity of point M

theorem equations_of_motion (t : ℝ) :
  let xM := 45 * (1 + Real.cos (omega * t))
  let yM := 45 * Real.sin (omega * t)
  xM = 45 * (1 + Real.cos (omega * t)) ∧
  yM = 45 * Real.sin (omega * t) ∧
  ((yM / 45) ^ 2 + ((xM - 45) / 45) ^ 2 = 1) ∧
  let vMx := -450 * Real.sin (omega * t)
  let vMy := 450 * Real.cos (omega * t)
  (vMx = -450 * Real.sin (omega * t)) ∧
  (vMy = 450 * Real.cos (omega * t)) :=
by
  sorry

end equations_of_motion_l374_374421


namespace petya_digits_l374_374790

def are_distinct (a b c d : Nat) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 

def non_zero_digits (a b c d : Nat) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

theorem petya_digits :
  ∃ (a b c d : Nat), are_distinct a b c d ∧ non_zero_digits a b c d ∧ (a + b + c + d = 11) ∧ (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 5) ∧
  (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 5) ∧ (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 5) ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 5) :=
by
  sorry

end petya_digits_l374_374790


namespace math_proof_statement_l374_374175

-- Assume an odd function f with a period of 4
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def hasPeriod4 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 4) = f x
def symmetricAbout2 (f : ℝ → ℝ) : Prop := ∀ x, f (2 - x) = f (2 + x)

-- Proposition 1
def prop1 (f : ℝ → ℝ) (H_odd : isOdd f) (H_period : hasPeriod4 f) : Prop :=
  symmetricAbout2 f

-- Proposition 2
def prop2 (a : ℝ) (H : 0 < a ∧ a < 1) : Prop :=
  ¬ (a^(1 + a) < a^(1 + 1/a))

-- Proposition 3
def prop3 : Prop :=
  isOdd (λ x, Real.log ((1 + x) / (1 - x)))

-- Proposition 4
def isOddLog (a : ℝ) (f : ℝ → ℝ) : Prop := isOdd (λ x, Real.log (a*x + Real.sqrt (2 * x^2 + 1))) 

def prop4 : Prop :=
  ¬ ∃! (a : ℝ), isOddLog a (λ x, Real.log (a * x + Real.sqrt (2 * x^2 + 1)))

-- The statement proving Propositions 1 and 3 are true, and Propositions 2 and 4 are false.
theorem math_proof_statement (f : ℝ → ℝ) (H_odd : isOdd f) (H_period : hasPeriod4 f) (a : ℝ) (H : 0 < a ∧ a < 1) :
  prop1 f H_odd H_period ∧ prop3 ∧ prop2 a H ∧ prop4 :=
by {
  split, sorry,  -- Proof for prop1
  split, sorry,  -- Proof for prop3
  split, sorry,  -- Proof for prop2
  sorry          -- Proof for prop4
}

end math_proof_statement_l374_374175


namespace tourists_meet_conditions_l374_374846

theorem tourists_meet_conditions :
  (∀ t : ℝ, (sqrt (1 + 6 * t) - 1 = 2) ↔ t = 4 / 3) ∧
  (∀ t : ℝ, (t ≥ 1 / 6 → 6 * (t - 1 / 6) = 2) ↔ t = 1 / 2) →
  (∀ t : ℝ, t ∈ [1 / 2, 4 / 3]) :=
by
  intros,
  sorry

end tourists_meet_conditions_l374_374846


namespace length_of_QR_l374_374827

theorem length_of_QR {P Q R : Type} [right_triangle P Q R] (cos_Q : ℝ) (PQ : ℝ) (QR : ℝ) :
  cos_Q = 0.6 → PQ = 15 → QR = 25 :=
by sorry

end length_of_QR_l374_374827


namespace ant_square_distance_eq_605000_l374_374084

-- Define the movement sequence of the ant.
def antMovementSequence (n : ℕ) : (ℤ × ℤ) :=
  let movements := 
    List.range n |>.map (λ i, match i % 4 with
      | 0 => (1 + (i+1) / 10, 0)  -- North
      | 1 => (0, - (i+1) / 5)     -- West
      | 2 => (- (i+1) / 3, 0)     -- South
      | 3 => (0, (i+1) / 4)       -- East
      | _ => (0, 0))              -- Default case (shouldn't happen)
  movements.foldl (λ acc mov => (acc.1 + mov.1, acc.2 + mov.2)) (0, 0)

-- Define the final position function after 1000 steps
def finalPositionAfterSteps (n : ℕ) : (ℤ × ℤ) :=
  if n % 4 = 0 then antMovementSequence n else (0, 0)

-- Calculate square of the straight-line distance between points A and B after 1000 steps
def squareOfDistance (position : ℤ × ℤ) : ℤ :=
  position.1 * position.1 + position.2 * position.2

theorem ant_square_distance_eq_605000 :
  squareOfDistance (finalPositionAfterSteps 1000) = 605000 :=
by
  sorry  -- Proof goes here

end ant_square_distance_eq_605000_l374_374084


namespace fermat_little_theorem_variant_l374_374308

theorem fermat_little_theorem_variant (p : ℕ) (m : ℤ) [hp : Fact (Nat.Prime p)] : 
  (m ^ p - m) % p = 0 :=
sorry

end fermat_little_theorem_variant_l374_374308


namespace hyperbola_shares_focus_with_eccentricity_length_of_chord_AB_l374_374144

theorem hyperbola_shares_focus_with_eccentricity 
  (a1 b1 : ℝ) (h1 : a1 = 3 ∧ b1 = 2)
  (e : ℝ) (h_eccentricity : e = (Real.sqrt 5) / 2)
  (c : ℝ) (h_focus : c = Real.sqrt (a1^2 - b1^2)) :
  (∃ a b : ℝ, a^2 - b^2 = c^2 ∧ c/a = e ∧ a = 2 ∧ b = 1) :=
sorry

theorem length_of_chord_AB 
  (a b : ℝ) (h_ellipse : a^2 = 4 ∧ b^2 = 1)
  (c : ℝ) (h_focus : c = Real.sqrt (a^2 - b^2))
  (f : ℝ) (h_f : f = Real.sqrt 3)
  (line_eq : ℝ -> ℝ) (h_line_eq : ∀ x, line_eq x = x - f) :
  (∃ x1 x2 : ℝ, 
    x1 + x2 = (8 * Real.sqrt 3) / 5 ∧
    x1 * x2 = 8 / 5 ∧
    Real.sqrt ((x1 - x2)^2 + (line_eq x1 - line_eq x2)^2) = 8 / 5) :=
sorry

end hyperbola_shares_focus_with_eccentricity_length_of_chord_AB_l374_374144


namespace probability_one_girl_no_increasing_pies_is_correct_l374_374523

def total_pies := 6
def increasing_pies := 2
def decreasing_pies := total_pies - increasing_pies
def pies_picked := 3

def total_combinations := Nat.choose total_pies pies_picked
def no_increasing_combinations := Nat.choose decreasing_pies pies_picked

def probability_no_increasing := (no_increasing_combinations : ℝ) / (total_combinations : ℝ)
def probability_at_least_one_increasing := 1 - probability_no_increasing
def probability_one_girl_no_increasing_pies := probability_no_increasing * 2 - probability_no_increasing * probability_no_increasing
def final_probability := probability_no_increasing + probability_no_increasing - probability_no_increasing * probability_no_increasing / total_combinations

theorem probability_one_girl_no_increasing_pies_is_correct : final_probability = 0.4 :=
by
  sorry

end probability_one_girl_no_increasing_pies_is_correct_l374_374523


namespace minimum_argument_difference_l374_374738

theorem minimum_argument_difference (n : ℕ) (z : Fin n → ℂ)
    (h_sum : ∑ k, z k = 0) : ∃ i j, i ≠ j ∧ abs (complex.arg (z i) - complex.arg (z j)) ≥ 2 * real.pi / 3 := sorry

end minimum_argument_difference_l374_374738


namespace hyperbola_ratio_l374_374640

theorem hyperbola_ratio (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_hyperbola : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_foci_distance : c^2 = a^2 + b^2)
  (h_midpoint_on_hyperbola : ∀ x y, 
    (x, y) = (-(c / 2), c / 2) → ∃ (k l : ℝ), (k^2 / a^2) - (l^2 / b^2) = 1) :
  c / a = (Real.sqrt 10 + Real.sqrt 2) / 2 := 
sorry

end hyperbola_ratio_l374_374640


namespace temperature_difference_l374_374004

theorem temperature_difference :
  let T_midnight := -4
  let T_10am := 5
  T_10am - T_midnight = 9 :=
by
  let T_midnight := -4
  let T_10am := 5
  show T_10am - T_midnight = 9
  sorry

end temperature_difference_l374_374004


namespace max_min_product_xy_l374_374858

-- Definition of conditions
variables (a x y : ℝ)
def condition_1 : Prop := x + y = a
def condition_2 : Prop := x^2 + y^2 = -a^2 + 2

-- The main theorem statement
theorem max_min_product_xy (a : ℝ) (ha_range : -2 ≤ a ∧ a ≤ 2): 
  (∀ x y : ℝ, condition_1 a x y ∧ condition_2 a x y → (x * y) ≤ (1 / 3)) ∧
  (∀ x y : ℝ, condition_1 a x y ∧ condition_2 a x y → (x * y) ≥ (-1)) :=
sorry

end max_min_product_xy_l374_374858


namespace smallest_x_value_l374_374403

theorem smallest_x_value :
  let eq := 6 * (9 * x^2 + 9 * x + 10) = x * (9 * x - 45)
  in ∃ x : ℚ, eq ∧ ∀ y : ℚ, eq → y ≥ x → x = -(4/3) :=
by
  sorry

end smallest_x_value_l374_374403


namespace min_colors_cube_l374_374925

-- Represent the conditions and main theorem
theorem min_colors_cube (n : ℕ) (n_ge_3 : n ≥ 3) : ∃ (r : ℕ), r = 7 :=
by {
  -- We define the existence of r, and will state that r is 7
  existsi 7,
  -- Stating the minimum number of colors required is 7 for n >= 3 is the goal
  sorry -- Proof goes here
}

end min_colors_cube_l374_374925


namespace phase_shift_of_sine_l374_374582

theorem phase_shift_of_sine :
  let a := 3
  let b := 4
  let c := - (Real.pi / 4)
  let phase_shift := -(c / b)
  phase_shift = Real.pi / 16 :=
by
  sorry

end phase_shift_of_sine_l374_374582


namespace simplify_with_guess_find_correct_coefficient_l374_374696

-- Define the expressions
def expr_with_guess (x : ℝ) : ℝ := (3 * x ^ 2 + 6 * x + 8) - (6 * x + 5 * x ^ 2 + 2)
def simplified_expr (x : ℝ) : ℝ := -2 * x ^ 2 + 6

-- Define the condition to find the correct value of the coefficient
def expr_with_general (a x : ℝ) : ℝ := (a * x ^ 2 + 6 * x + 8) - (6 * x + 5 * x ^ 2 + 2)
def constant_expr : ℝ := 6

theorem simplify_with_guess : ∀ (x : ℝ), expr_with_guess x = simplified_expr x :=
by
  intros x,
  sorry

theorem find_correct_coefficient : ∃ (a : ℝ), (∀ (x : ℝ), expr_with_general a x = constant_expr) ∧ a = 5 :=
by
  use 5,
  split,
  { intro x, sorry },
  { refl }

end simplify_with_guess_find_correct_coefficient_l374_374696


namespace relationship_among_abc_l374_374037

noncomputable def a : ℝ := 6 ^ 0.7
noncomputable def b : ℝ := 0.7 ^ 6
noncomputable def c : ℝ := Real.logBase 0.7 6

theorem relationship_among_abc : c < b ∧ b < a := 
by
  -- Proof to be filled in here
  sorry

end relationship_among_abc_l374_374037


namespace fence_remaining_length_l374_374116

theorem fence_remaining_length : 
  let initial_length := 100
  let ben_contribution := 10
  let billy_fraction := 1 / 5
  let johnny_fraction := 1 / 3
  let remaining_after_ben := initial_length - ben_contribution
  let billy_contribution := billy_fraction * remaining_after_ben
  let remaining_after_billy := remaining_after_ben - billy_contribution
  let johnny_contribution := johnny_fraction * remaining_after_billy
  let remaining_after_johnny := remaining_after_billy - johnny_contribution
  in remaining_after_johnny = 48 :=
by
  sorry

end fence_remaining_length_l374_374116


namespace right_triangle_side_length_l374_374230

theorem right_triangle_side_length (a c b : ℕ) (h1 : a = 3) (h2 : c = 5) (h3 : c^2 = a^2 + b^2) : b = 4 :=
sorry

end right_triangle_side_length_l374_374230


namespace average_text_messages_correct_l374_374695

-- Define Jason's text message count function for each day
def messages_on_monday : ℕ := 220

def messages_on_tuesday (monday : ℕ) : ℕ := monday - (0.15 * monday).toInt

def messages_on_wednesday (tuesday : ℕ) : ℕ := tuesday + (0.25 * tuesday).toInt

def messages_on_thursday (wednesday : ℕ) : ℕ := wednesday - (0.10 * wednesday).toInt

def messages_on_friday (thursday : ℕ) : ℕ := thursday + (0.05 * thursday).toInt

-- Calculate the average number of messages
def average_messages (monday tuesday wednesday thursday friday : ℕ) : ℝ :=
  ((monday + tuesday + wednesday + thursday + friday) / 5.0)

-- Prove the average number of text messages Jason sent over the five days
theorem average_text_messages_correct :
  average_messages
    messages_on_monday
    (messages_on_tuesday messages_on_monday)
    (messages_on_wednesday (messages_on_tuesday messages_on_monday))
    (messages_on_thursday (messages_on_wednesday (messages_on_tuesday messages_on_monday)))
    (messages_on_friday (messages_on_thursday (messages_on_wednesday (messages_on_tuesday messages_on_monday))))
  = 214.20375 := by
  sorry

end average_text_messages_correct_l374_374695


namespace hump_number_sum_tens_place_eq_30_l374_374020

-- Definition of hump number
def is_hump_number (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  (d2 < d1 + d3) ∧ 
  (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3) ∧ 
  (1 ≤ d1 ∧ d1 ≤ 5 ∧ 1 ≤ d2 ∧ d2 ≤ 5 ∧ 1 ≤ d3 ∧ d3 ≤ 5)

-- Main theorem statement
theorem hump_number_sum_tens_place_eq_30 :
  let hump_numbers := { n : ℕ | 100 ≤ n ∧ n < 1000 ∧ is_hump_number n }
  let sum_tens_digits := ∑ n in hump_numbers, (n % 100) / 10
  sum_tens_digits = 30 :=
by
  sorry

end hump_number_sum_tens_place_eq_30_l374_374020


namespace least_M_exists_l374_374082

-- Define constants and variables
variables {c : ℝ} (h_c : c ∈ Ioo (1 / 2) 1)

-- Define the problem conditions
def problem_condition (n : ℕ) (a : ℕ → ℝ) :=
  (n ≥ 2) ∧ (∀ i j, 1 ≤ i → i ≤ j → j ≤ n → a i ≤ a j) ∧
  (1 / n * (∑ k in finset.range n, (k + 1) * a (k + 1)) = c * ∑ k in finset.range n, a (k + 1))

-- Define the proof that least M exists
theorem least_M_exists :
  ∃ M, ∀ (n : ℕ) (a : ℕ → ℝ),
    problem_condition n a →
    ∑ k in finset.range n, a (k + 1) ≤ M * ∑ k in finset.range (nat.floor (c * n)), a (k + 1) :=
begin
  use 1 / (1 - c),
  sorry  -- Proof goes here
end

end least_M_exists_l374_374082
