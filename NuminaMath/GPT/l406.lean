import Mathlib
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Order.Nonneg
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.QuadraticEquation
import Mathlib.Algebra.Ring.Basic
import Mathlib.Algebra.Vector
import Mathlib.Analysis.Calculus.ContDiff
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.MeasureTheory.IntegrableFunction
import Mathlib.MeasureTheory.ProbabilityDistribution
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.Notation
import Mathlib.Tactic
import Mathlib.Tactic.FinCases
import Mathlib.Topology.Instances.Real
import analysis.special_functions.pow

namespace average_length_of_two_strings_l406_406108

theorem average_length_of_two_strings :
  (∀ (string1 string2 : ℝ), string1 = 2 ∧ string2 = 5 → (string1 + string2) / 2 = 3.5) :=
by
  intros string1 string2 h
  cases h with h1 h2
  rw [h1, h2]
  norm_num

end average_length_of_two_strings_l406_406108


namespace rhombus_angles_l406_406955

-- We define the points and given conditions
variables (A B C D E F : Type)
variables [is_rhombus A B C D] -- Rhombus condition
variables (E_on_AB : E ∈ seg A B) (F_on_BC : F ∈ seg B C) -- Points E and F on respective segments
variables (ratio_AE_BE : AE / BE = 5) (ratio_BF_CF : BF / CF = 5) -- Ratios given
variables (is_equilateral_DEF : is_equilateral_triangle D E F) -- Triangle DEF is equilateral

-- Question and final goal
theorem rhombus_angles (h : is_rhombus A B C D) : 
  angles A B C D = [60, 120] :=
sorry

end rhombus_angles_l406_406955


namespace am_gm_inequality_positive_real_l406_406468

theorem am_gm_inequality_positive_real (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  x^2 + y^4 + z^6 ≥ x * y^2 + y^2 * z^3 + x * z^3 :=
begin
  sorry
end

end am_gm_inequality_positive_real_l406_406468


namespace calculate_expression_l406_406971

theorem calculate_expression : (π - 2)^0 - (-2)⁻¹ + |real.sqrt 3 - 2| = (7 / 2) - real.sqrt 3 :=
by sorry

end calculate_expression_l406_406971


namespace survey_min_people_l406_406757

theorem survey_min_people (p : ℕ) : 
  (∃ p, ∀ k ∈ [18, 10, 5, 9], k ∣ p) → p = 90 :=
by sorry

end survey_min_people_l406_406757


namespace sqrt_expression_l406_406378

theorem sqrt_expression (x y : ℝ) (h : sqrt (x - 2) + abs (2 * y + 1) = 0) : sqrt (x + 2 * y) = 1 ∨ sqrt (x + 2 * y) = -1 := by
  sorry

end sqrt_expression_l406_406378


namespace find_vector_norm_l406_406361

variables (a b : ℝ^3) (angle_ab : ℝ)
noncomputable def vector_norm_equiv : ℝ :=
  let la := 4 in
  let lb := 2 in
  let angle := 120 in
  if |a| = la ∧ |b| = lb ∧ angle_ab = angle then
    |3 • a - 4 • b|
  else 
    0

theorem find_vector_norm : 
  ∃ la lb angle_ab : ℝ, 
  la = 4 ∧ lb = 2 ∧ angle_ab = 120 ∧ 
  |3 • a - 4 • b| = 4 * sqrt 19 :=
begin
  -- Given conditions
  assume H1 : |a| = 4, 
  assume H2 : |b| = 2,
  assume H3 : angle_ab = 120,
  -- Proof placeholder
  sorry
end

end find_vector_norm_l406_406361


namespace correct_calculation_l406_406902

theorem correct_calculation : (sqrt 3 + 1) * (sqrt 3 - 1) = 2 := 
by sorry

end correct_calculation_l406_406902


namespace ratio_value_l406_406726

theorem ratio_value (x y : ℝ) (h : x / y = 2 / 3) : x / (x + y) = 2 / 5 := 
by
  sorry

end ratio_value_l406_406726


namespace inequality_proof_l406_406349

variable (x1 x2 y1 y2 z1 z2 : ℝ)
variable (h0 : 0 < x1)
variable (h1 : 0 < x2)
variable (h2 : x1 * y1 > z1^2)
variable (h3 : x2 * y2 > z2^2)

theorem inequality_proof :
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2)^2) ≤ 1 / (x1 * y1 - z1^2) + 1 / (x2 * y2 - z2^2) :=
sorry

end inequality_proof_l406_406349


namespace beth_wins_config_B_l406_406963

-- Define the nim-values for walls of size n
def nim_value (n : ℕ) : ℕ :=
match n with
| 0 => 0
| 1 => 1
| 2 => 2
| 3 => 3
| 4 => 1
| 5 => 4
| 6 => 3
| _ => 0 -- Only considering up to size 6 as given in the problem
end

-- Define a function to compute the combined nim-value of a list of walls
def combined_nim_value (walls : List ℕ) : ℕ :=
walls.foldr (λ w acc => wxor acc (nim_value w)) 0

-- Define the specific initial state configurations given in the problem
def config_A : List ℕ := [6,1,1]
def config_B : List ℕ := [6,2,1]
def config_C : List ℕ := [6,2,2]
def config_D : List ℕ := [6,3,1]
def config_E : List ℕ := [6,3,2]

-- Statement: Proving that the nim-value of configuration B is zero 
theorem beth_wins_config_B : combined_nim_value config_B = 0 := by {
  -- Define nim-values for individual walls in config_B: [6, 2, 1]
  have h1 : nim_value 6 = 3 := rfl,
  have h2 : nim_value 2 = 2 := rfl,
  have h3 : nim_value 1 = 1 := rfl,
  -- Calculate combined nim-value for config_B using XOR
  show combined_nim_value config_B = 0,
  rw [←h1, ←h2, ←h3],
  simp only [config_B, combined_nim_value, List.foldr, List.foldr_cons],
  calc
    (3 wxor 2 wxor 1) = (3 wxor (2 wxor 1)) : by simp [Nat.wxor_assoc]
                  ... = (3 wxor 3) : by rw [Nat.wxor_comm 2 1, Nat.wxor_self_add_eq, Nat.wxor_self]
                  ... = 0 : by rw [Nat.wxor_self],
  exact rfl,
}

end beth_wins_config_B_l406_406963


namespace simplify_expression_l406_406561

variable {a b : ℝ}

theorem simplify_expression : (4 * a^3 * b - 2 * a * b) / (2 * a * b) = 2 * a^2 - 1 := by
  sorry

end simplify_expression_l406_406561


namespace sum_radii_incircle_circumcircle_l406_406020

theorem sum_radii_incircle_circumcircle 
  (α : ℝ) (hα : 0 < α ∧ α < π / 2) (a b : ℝ)
  (h_a : a = sin α) (h_b : b = cos α) 
  (R r : ℝ) 
  (hR : R = 1 / 2)
  (hr : r = (a + b - 1) / 2)
  : R + r = (sin α + cos α) / 2 :=
by {
  sorry
}

end sum_radii_incircle_circumcircle_l406_406020


namespace polynomial_coeff_sum_sq_diff_l406_406723

theorem polynomial_coeff_sum_sq_diff {a : ℕ → ℝ} (h : (√2 - x)^10 = (finset.range 11).sum (λ i, a i * x^i)) :
  let s_even := (finset.range 6).sum (λ k, a (2 * k))
  let s_odd := (finset.range 5).sum (λ k, a (2 * k + 1))
  (s_even * s_even) - (s_odd * s_odd) = 1 :=
by
  sorry

end polynomial_coeff_sum_sq_diff_l406_406723


namespace find_angle_A_l406_406775

theorem find_angle_A (ABC : Triangle) (O : Point) (BC : Line)
  (hO : is_incenter O ABC)
  (h_sym : symmetric_point_lies_on_circumcircle O BC ABC) :
  ∠ABC.ABC.A = 60 :=
  sorry

end find_angle_A_l406_406775


namespace not_all_same_probability_l406_406528

-- Definition of the total number of outcomes when rolling 5 8-sided dice
def total_outcomes : ℕ := 8^5

-- Definition of the number of outcomes where all five dice show the same number
def same_number_outcomes : ℕ := 8

-- Definition to find the probability that not all 5 dice show the same number
def probability_not_all_same : ℚ := 1 - (same_number_outcomes / total_outcomes)

-- Statement of the main theorem
theorem not_all_same_probability : probability_not_all_same = (4095 : ℚ) / 4096 :=
by
  rw [probability_not_all_same, same_number_outcomes, total_outcomes]
  -- Simplification steps would go here, but we use sorry to skip the proof
  sorry

end not_all_same_probability_l406_406528


namespace arithmetic_computation_l406_406291

theorem arithmetic_computation : 65 * 1515 - 25 * 1515 = 60600 := by
  sorry

end arithmetic_computation_l406_406291


namespace smallest_n_for_roots_of_polynomial_l406_406538

noncomputable def smallest_positive_integer_root_unity (z : ℂ) : ℕ :=
if h : z^4 + z^3 + 1 = 0 then
  let roots := {z : ℂ | z^4 + z^3 + 1 = 0} in
  let n := Inf {n : ℕ | ∀ z ∈ roots, z^n = 1} in
  n
else 0

theorem smallest_n_for_roots_of_polynomial :
  smallest_positive_integer_root_unity 1 = 5 :=
sorry

end smallest_n_for_roots_of_polynomial_l406_406538


namespace cost_of_insulation_l406_406213

theorem cost_of_insulation 
  (length : ℝ) (width : ℝ) (height : ℝ) (cost_per_sqft : ℝ) 
  (h1 : length = 5) (h2 : width = 3) (h3 : height = 2) (h4 : cost_per_sqft = 20) : 
  let surface_area := 2 * (length * width + length * height + width * height) in
  let total_cost := surface_area * cost_per_sqft in
  total_cost = 1240 :=
by
  sorry

end cost_of_insulation_l406_406213


namespace per_minute_charge_plan_B_l406_406235

theorem per_minute_charge_plan_B (x : ℝ) :
  (∀ minutes : ℝ, minutes = 6 → 
    (if minutes <= 8 then 0.60 else 0.60 + 0.06 * (minutes - 8)) = minutes * x) → 
  x = 0.10 :=
by
  intro h
  specialize h 6 rfl
  linarith
  sorry

end per_minute_charge_plan_B_l406_406235


namespace arccos_gt_arctan_on_interval_l406_406644

noncomputable def c : ℝ := sorry -- placeholder for the numerical solution of arccos x = arctan x

theorem arccos_gt_arctan_on_interval (x : ℝ) (hx : -1 ≤ x ∧ x < c) :
  Real.arccos x > Real.arctan x := 
sorry

end arccos_gt_arctan_on_interval_l406_406644


namespace find_line_l_l406_406677

variables (A B : Matrix (Fin 2) (Fin 2) ℝ)
variables (l l' : ℝ → ℝ → Prop)

-- Given matrices A and B
def matrix_A : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 0], ![0, 2]]
def matrix_B : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 2], ![0, 1]]
-- Given condition on line l'
def line_l' : ℝ → ℝ → Prop := λ x' y', x' + y' - 2 = 0

-- The transformation given by AB^{-1}
noncomputable def transformation (x y : ℝ) : ℝ × ℝ :=
  let B_inv := B⁻¹ in 
  let AB_inv := A ⬝ B_inv in
  (AB_inv 0 0 * x + AB_inv 0 1 * y, AB_inv 1 0 * x + AB_inv 1 1 * y)

-- Statement to prove
def equation_of_line_l : Prop :=
  ∀ (x y : ℝ), transformation A B x y → l' (x - 2 * y) (2 * y) → (x = 2)

theorem find_line_l :
  equation_of_line_l matrix_A matrix_B line_l' :=
sorry

end find_line_l_l406_406677


namespace acute_triangle_sec_csc_inequality_l406_406398

theorem acute_triangle_sec_csc_inequality (A B C : ℝ) (h : A + B + C = π) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hA90 : A < π / 2) (hB90 : B < π / 2) (hC90 : C < π / 2) :
  (1 / Real.cos A) + (1 / Real.cos B) + (1 / Real.cos C) ≥
  (1 / Real.sin (A / 2)) + (1 / Real.sin (B / 2)) + (1 / Real.sin (C / 2)) :=
by sorry

end acute_triangle_sec_csc_inequality_l406_406398


namespace find_natural_numbers_l406_406312

-- Problem statement: Find all natural numbers x, y, z such that 3^x + 4^y = 5^z
theorem find_natural_numbers (x y z : ℕ) (h : 3^x + 4^y = 5^z) : x = 2 ∧ y = 2 ∧ z = 2 :=
sorry

end find_natural_numbers_l406_406312


namespace movement_coordinates_l406_406405

theorem movement_coordinates (a b : ℝ) 
  (hA : a - 5 = 0)
  (hB : b + 3 = 0) :
  let C := (a, b) in let C_new := (C.1 + 2, C.2 - 3) in
  C_new = (7, -6) :=
by
  sorry

end movement_coordinates_l406_406405


namespace probability_of_event_l406_406942

-- Define the setup of the problem
variables {A B C P : Point} (ABC : Triangle)
variable (isosceles : ABC.isosceles AB AC)

-- Define the event of interest
def event (P : Point) : Prop :=
  let area_ABP := area (triangle A B P)
  let area_ACP := area (triangle A C P)
  let area_BCP := area (triangle B C P)
  area_ABP > area_ACP ∧ area_ABP > area_BCP

-- The probability we aim to prove
theorem probability_of_event :
  probability (interior_point_event ABC isosceles event) = 1 / 6 := 
sorry

end probability_of_event_l406_406942


namespace tan_sum_formula_l406_406118

noncomputable def tan_sum (θ : ℝ) (n : ℕ) :=
  ∑ k in Finset.range n, Real.tan (θ + k * (Real.pi / n))

theorem tan_sum_formula (θ : ℝ) (n : ℕ) (hn : 0 < n) :
  tan_sum θ n = 
  if odd n then 
    n * Real.tan (n * θ)
  else 
    -n * Real.cot (n * θ) := 
sorry

end tan_sum_formula_l406_406118


namespace area_APBD_l406_406588

open Real

/-- Define the geometry of the square and point properties -/
structure Square (A B C D P F : Point) : Prop :=
(side_length : ∀ (A B C D : Point), dist A B = 8 ∧ dist B C = 8 ∧ dist C D = 8 ∧ dist D A = 8)
(equidistant : ∀ (A B C D P : Point), dist P A = dist P B ∧ dist P B = dist P C ∧ dist P C = dist P D)
(perpendicular : ∀ (P C F D : Point), ∃ F, P = midpoint C F ∧ angle P C F = π / 2 ∧ dist F D = 8)

/-- Prove that the area of quadrilateral APBD is 32 square inches -/
theorem area_APBD (A B C D P F : Point) (h : Square A B C D P F) : 
  let APBD : Quadrilateral := {A, P, B, D} in 
  area APBD = 32 := sorry

end area_APBD_l406_406588


namespace slope_range_of_line_l_l406_406011

-- Define the points
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, 0)

-- Define the line equation
def line_l (a x y : ℝ) : Prop := a * x + y - 2 * a + 1 = 0

-- Define the intersection condition and slopes
def slope_PA : ℝ := -4
def slope_PB : ℝ := 1 / 2
def valid_slope_range (k : ℝ) : Prop := k ≤ slope_PA ∨ k ≥ slope_PB

-- Main theorem statement
theorem slope_range_of_line_l (a x y k : ℝ) (hx : line_l a x y) :
  valid_slope_range k :=
sorry

end slope_range_of_line_l_l406_406011


namespace tangent_circles_of_XYZ_and_BES_l406_406675

noncomputable theory
open_locale classical

variables {A B C D E X Y Q P S Z : Point}

def is_inscribed (pentagon : List Point) (circle : Circle) : Prop :=
  ∀ (p : Point), p ∈ pentagon → on_circle p circle

def is_tangent (line : Line) (circle : Circle) : Prop :=
  // Define tangent property: line touches circle at exactly one point
  ∃ p : Point, on_circle p circle ∧ on_line p line ∧ ∀ q : Point, q ≠ p → ¬on_line_circle q line

def between_ABC (A B C : Point) : Prop :=
  collinear A B C ∧ dist A B + dist B C = dist A C

def intersects_at (circle1 circle2 : Circle) (segment : Segment) (intersection_point : Point) : Prop :=
  on_segment intersection_point segment ∧ on_circle intersection_point circle1 ∧ on_circle intersection_point circle2

def prove_tangent_circles (circXYZ circBES : Circle) : Prop :=
  ∃ (tangent_point : Point), on_circle tangent_point circXYZ ∧ on_circle tangent_point circBES

theorem tangent_circles_of_XYZ_and_BES
  (pentagon : List Point)
  (circ : Circle)
  (line : Line)
  (circXYZ circBES : Circle)
  (H1 : is_inscribed [A, B, C, D, E] circ)
  (H2 : is_tangent line circ)
  (H3 : X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X)  
  (H4 : between_ABC X A Y)
  (H5 : intersects_at (circumcircle X E D) circ AD Q)
  (H6 : intersects_at (circumcircle Y B C) circ AC P)
  (H7 : intersects_at (Line.intersection (segment_line (X,E)) (segment_line (Y,B))) (segment_line S))
  (H8 : intersects_at (Line.intersection (segment_line (X,Q)) (segment_line (Y,P))) (segment_line Z)) :
  prove_tangent_circles circXYZ circBES := sorry

end tangent_circles_of_XYZ_and_BES_l406_406675


namespace div_identity_l406_406897

theorem div_identity :
  let a := 6 / 2
  let b := a * 3
  120 / b = 120 / 9 :=
by
  sorry

end div_identity_l406_406897


namespace trig_expr_evaluation_l406_406981

theorem trig_expr_evaluation :
    (1 - (1 / cos (Real.pi / 6))) * 
    (1 + (1 / sin (Real.pi / 3))) * 
    (1 - (1 / sin (Real.pi / 6))) * 
    (1 + (1 / cos (Real.pi / 3))) = 1 := 
by
  let cos_30 := cos (Real.pi / 6)
  let sin_60 := sin (Real.pi / 3)
  let sin_30 := sin (Real.pi / 6)
  let cos_60 := cos (Real.pi / 3)
  have h_cos_30 : cos_30 = (Real.sqrt 3) / 2 := sorry
  have h_sin_60 : sin_60 = (Real.sqrt 3) / 2 := sorry
  have h_sin_30 : sin_30 = 1 / 2 := sorry
  have h_cos_60 : cos_60 = 1 / 2 := sorry
  sorry

end trig_expr_evaluation_l406_406981


namespace sum_of_areas_of_circles_l406_406168

-- Definitions of conditions
def r : ℝ := by sorry  -- radius of the circle at vertex A
def s : ℝ := by sorry  -- radius of the circle at vertex B
def t : ℝ := by sorry  -- radius of the circle at vertex C

axiom sum_radii_r_s : r + s = 6
axiom sum_radii_r_t : r + t = 8
axiom sum_radii_s_t : s + t = 10

-- The statement we want to prove
theorem sum_of_areas_of_circles : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by
  -- Use given axioms and properties of the triangle and circles
  sorry

end sum_of_areas_of_circles_l406_406168


namespace max_value_of_expression_l406_406842

open Real

theorem max_value_of_expression
  (x y : ℝ)
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : x^2 - 2 * x * y + 3 * y^2 = 10) 
  : x^2 + 2 * x * y + 3 * y^2 ≤ 10 * (45 + 42 * sqrt 3) := 
sorry

end max_value_of_expression_l406_406842


namespace paco_cookie_paradox_l406_406454

theorem paco_cookie_paradox : 
  ∀ (initial_eaten given_found : ℕ),
    initial_eaten = 18 →
    given_found = 21 →
    (initial_eaten - given_found = 0 ∨ initial_eaten - given_found < 0) :=
begin
  introv heaten hgiven hfound,
  service := heaten - hfound,
  cases lt | eq : service,
  { right, exact gypsum  heateal },
  { left, exact heate ->,
  sorry,
end.

end paco_cookie_paradox_l406_406454


namespace find_angle_BED_l406_406026

open Classical

noncomputable def angle_sum (A B C : ℕ) : ℕ := 180 - A - B

-- Definitions based on the problem conditions
def triangle_data : Type := ℕ × ℕ × (ℕ × ℕ) × (ℕ × ℕ)
def input_data : triangle_data := (45, 85, (0,0), (0,0)) -- angles A and C, points D and E are on sides AB and BC

-- Statements for the proof
theorem find_angle_BED (A C DB BE : ℕ) (hA : A = 45) (hC : C = 85) (hDB : DB = 2 * BE) :
  ∃ BED : ℕ, BED = 43 :=
by
  -- Angle B calculation
  let B := angle_sum A C
  have hB : B = 50 := by
    unfold angle_sum
    rw [hA, hC]
    norm_num
  -- Proof given in the problem's solution
  use 43
  sorry

end find_angle_BED_l406_406026


namespace log_101600_div_3_l406_406023

/-- 
Given some specific logarithm values, we want to prove the value of a composite logarithm.
Given:
  log102 = 0.3010
  log3 = 0.4771
Prove:
  log(101600/3) = 0.1249
-/

def log102 := 0.3010
def log3 := 0.4771

theorem log_101600_div_3 : log (101600 / 3) = 0.1249 :=
by
  -- Apply logarithmic rules and simplifications
  sorry

end log_101600_div_3_l406_406023


namespace complex_triangle_product_l406_406972

theorem complex_triangle_product (z1 z2 z3 : ℂ) (h1 : abs (z1 + z2 + z3) = 48) 
(h2 : abs (z1 - z2) = 24) (h3 : abs (z2 - z3) = 24) (h4 : abs (z3 - z1) = 24) : 
abs (z1 * z2 + z2 * z3 + z3 * z1) = 768 := 
sorry

end complex_triangle_product_l406_406972


namespace gcf_90_108_l406_406896

-- Given two integers 90 and 108
def a : ℕ := 90
def b : ℕ := 108

-- Question: What is the greatest common factor (GCF) of 90 and 108?
theorem gcf_90_108 : Nat.gcd a b = 18 :=
by {
  sorry
}

end gcf_90_108_l406_406896


namespace number_of_correct_props_is_zero_l406_406679

/-- Proposition 1: If point P is not on plane α, and points A, B, and C are all on plane α, then points P, A, B, and C are not in the same plane -/
def prop1 (P A B C : Point) (α : Plane) : Prop :=
  (¬ (P ∈ α) ∧ A ∈ α ∧ B ∈ α ∧ C ∈ α) → ¬ coplanar P A B C

/-- Proposition 2: Three lines that intersect each other pairwise are in the same plane -/
def prop2 (l1 l2 l3 : Line) : Prop :=
  (∃ P1 P2 P3, l1 ∩ l2 = {P1} ∧ l2 ∩ l3 = {P2} ∧ l3 ∩ l1 = {P3}) → coplanar_lines l1 l2 l3

/-- Proposition 3: A quadrilateral whose opposite sides are equal is a parallelogram -/
def prop3 (abcd : Quadrilateral) : Prop :=
  (opposite_sides_equal abcd) → is_parallelogram abcd

/-- Main theorem: The number of correct propositions is 0 -/
theorem number_of_correct_props_is_zero :
  (∀ (P A B C : Point) (α : Plane), ¬ prop1 P A B C α) ∧
  (∀ (l1 l2 l3 : Line), ¬ prop2 l1 l2 l3) ∧
  (∀ (abcd : Quadrilateral), ¬ prop3 abcd) →
  0 = 0 :=
by
  sorry

end number_of_correct_props_is_zero_l406_406679


namespace sum_floor_sqrt_1_to_25_l406_406995

theorem sum_floor_sqrt_1_to_25 :
  (∑ n in finset.range 25, ⌊real.sqrt (n + 1)⌋) = 75 :=
sorry

end sum_floor_sqrt_1_to_25_l406_406995


namespace Jillian_had_200_friends_l406_406055

def oranges : ℕ := 80
def pieces_per_orange : ℕ := 10
def pieces_per_friend : ℕ := 4
def number_of_friends : ℕ := oranges * pieces_per_orange / pieces_per_friend

theorem Jillian_had_200_friends :
  number_of_friends = 200 :=
sorry

end Jillian_had_200_friends_l406_406055


namespace hypotenuse_right_triangle_l406_406196

theorem hypotenuse_right_triangle (a b : ℕ) (h₁ : a = 80) (h₂ : b = 150) : 
  ∃ c, c = 170 ∧ c^2 = a^2 + b^2 :=
by
  use 170
  simp [h₁, h₂]
  norm_num
  sorry

end hypotenuse_right_triangle_l406_406196


namespace percent_full_time_more_than_three_years_l406_406755

variable (total_associates : ℕ)
variable (second_year_percentage : ℕ)
variable (third_year_percentage : ℕ)
variable (non_first_year_percentage : ℕ)
variable (part_time_percentage : ℕ)
variable (part_time_more_than_two_years_percentage : ℕ)
variable (full_time_more_than_three_years_percentage : ℕ)

axiom condition_1 : second_year_percentage = 30
axiom condition_2 : third_year_percentage = 20
axiom condition_3 : non_first_year_percentage = 60
axiom condition_4 : part_time_percentage = 10
axiom condition_5 : part_time_more_than_two_years_percentage = 5

theorem percent_full_time_more_than_three_years : 
  full_time_more_than_three_years_percentage = 10 := 
sorry

end percent_full_time_more_than_three_years_l406_406755


namespace probability_of_green_marbles_l406_406776

variable (totalMarbles greenMarbles purpleMarbles draws greenDraws : ℕ)
variable (probGreen : ℝ) (probPurple : ℝ)

-- Conditions
def initialBag : Prop :=
  totalMarbles = 10 ∧ greenMarbles = 6 ∧ purpleMarbles = 4

def drawProcess : Prop :=
  draws = 8

-- The theorem to prove
theorem probability_of_green_marbles :
  initialBag totalMarbles greenMarbles purpleMarbles ∧ drawProcess draws ∧ 
  probGreen = (choose 8 5 * (6/10)^5 * (4/10)^3) →
  round (probGreen * 1000) / 1000 = 0.028 :=
by
  sorry

end probability_of_green_marbles_l406_406776


namespace dice_probability_not_all_same_l406_406532

theorem dice_probability_not_all_same : 
  let total_outcomes := (8 : ℕ)^5 in
  let same_number_outcomes := 8 in
  let probability_all_same := (same_number_outcomes : ℚ) / total_outcomes in
  let probability_not_all_same := 1 - probability_all_same in
  probability_not_all_same = 4095 / 4096 :=
by
  sorry

end dice_probability_not_all_same_l406_406532


namespace pyramid_volume_l406_406224

theorem pyramid_volume (s : ℝ) (V_cube : ℝ) 
  (h1 : V_cube = s^3) 
  (h2 : s = 4) : 
  1 / 3 * (1 / 2 * s * s) * s = 32 / 3 :=
by
  -- We state the volume of the pyramid PQSU calculation here
  -- but we use an imported library fact here.
  calc
    1 / 3 * (1 / 2 * s * s) * s = 1 / 3 * 8 * s : by rw [h2, pow_succ']
                           ... = 1 / 3 * 8 * 4 : by rw h2
                           ... = 32 / 3 : by ring

end pyramid_volume_l406_406224


namespace most_probable_germinated_seeds_l406_406857

noncomputable theory
open_locale big_operators

def binom_prob {n : ℕ} (p : ℝ) (k : ℕ) : ℝ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem most_probable_germinated_seeds (n : ℕ) (p : ℝ) (h_n : n = 9) (h_p : p = 0.8) :
  let k₀ := (range (n + 1)).max_on (binom_prob p) in k₀ = 7 ∨ k₀ = 8 :=
sorry

end most_probable_germinated_seeds_l406_406857


namespace other_acute_angle_in_right_triangle_l406_406734

theorem other_acute_angle_in_right_triangle (a : ℝ) (h_a : a = 40) :
  ∃ b : ℝ, a + b = 90 ∧ b = 50 :=
by
  use 50
  split
  case left => sorry
  case right => sorry

end other_acute_angle_in_right_triangle_l406_406734


namespace required_bricks_l406_406932

-- Define the dimensions of the brick
def brick_length_cm : ℝ := 20
def brick_width_cm : ℝ := 10
def brick_height_cm : ℝ := 7.5

-- Define the dimensions of the wall
def wall_length_m : ℝ := 29
def wall_width_m : ℝ := 2
def wall_height_m : ℝ := 0.75

-- Define the conversion factor from cm³ to m³
def cm_to_m : ℝ := 1 / 100

-- Define the volume of the brick in cubic meters
noncomputable def volume_brick_m³ : ℝ :=
  (brick_length_cm * cm_to_m) * (brick_width_cm * cm_to_m) * (brick_height_cm * cm_to_m)

-- Define the volume of the wall in cubic meters
def volume_wall_m³ : ℝ := wall_length_m * wall_width_m * wall_height_m

-- Define the number of bricks required
noncomputable def number_of_bricks : ℝ := volume_wall_m³ / volume_brick_m³

-- The theorem that needs to be proven
theorem required_bricks : number_of_bricks = 2900 :=
by sorry

end required_bricks_l406_406932


namespace bicycle_helmet_savings_l406_406498

theorem bicycle_helmet_savings :
  let bicycle_regular_price := 320
  let bicycle_discount := 0.2
  let helmet_regular_price := 80
  let helmet_discount := 0.1
  let bicycle_savings := bicycle_regular_price * bicycle_discount
  let helmet_savings := helmet_regular_price * helmet_discount
  let total_savings := bicycle_savings + helmet_savings
  let total_regular_price := bicycle_regular_price + helmet_regular_price
  let percentage_savings := (total_savings / total_regular_price) * 100
  percentage_savings = 18 := 
by sorry

end bicycle_helmet_savings_l406_406498


namespace log_expression_equality_l406_406309

-- Define the problem conditions
variables (a b c d x y z w : ℝ)
-- Assume all conditions are positive and real
variables (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
          (h_pos_d : 0 < d) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
          (h_pos_z : 0 < z) (h_pos_w : 0 < w)

-- The statement to be proven
theorem log_expression_equality :
  log (a / b) + log (b / c) + log (c / d) - log (a * y * z / (d * x * w)) = log (x * w / (y * z)) :=
  sorry

end log_expression_equality_l406_406309


namespace sum_of_areas_of_circles_l406_406177

theorem sum_of_areas_of_circles 
  (r s t : ℝ) 
  (h1 : r + s = 6) 
  (h2 : r + t = 8) 
  (h3 : s + t = 10) 
  : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by 
  sorry

end sum_of_areas_of_circles_l406_406177


namespace find_n_times_s_l406_406081

def f : ℕ → ℕ :=
  sorry

axiom functional_equation : ∀ a b : ℕ, f(a^2 + 2*b^2) = (f(a))^2 + 2*(f(b))^2

noncomputable def n : ℕ :=
  if h : f 34 = 0 then 1 else if h' : f 34 = 34 then 2 else 0

noncomputable def s : ℕ :=
  if h : f 34 = 0 then 0 else if h' : f 34 = 34 then 0 + 34 else 0

theorem find_n_times_s : n * s = 68 :=
  by sorry

end find_n_times_s_l406_406081


namespace sum_of_positive_real_solutions_l406_406899

noncomputable def sum_of_solutions := (1080 : ℝ) * Real.pi

theorem sum_of_positive_real_solutions (x : ℝ) :
  (0 < x) →
  (2 * Real.cos(2 * x) * (Real.cos(2 * x) - Real.cos((2014 * Real.pi^2) / x)) = Real.cos(4 * x) - 1) →
  (∑ y in {x : ℝ | 0 < x ∧ 2 * Real.cos(2 * x) * (Real.cos(2 * x) - Real.cos((2014 * Real.pi^2) / x)) = Real.cos(4 * x) - 1}.toFinset, y) = sum_of_solutions :=
sorry

end sum_of_positive_real_solutions_l406_406899


namespace angle_alpha_eq_expr_value_l406_406687

theorem angle_alpha_eq :
  (∀ α : ℝ, (α > π / 2 ∧ α < 3 * π / 2) →
  let A := (3, 0)
      B := (0, 3)
      C := (Real.cos α, Real.sin α)
      AC := (3 - Real.cos α, 0 - Real.sin α)
      BC := (0 - Real.cos α, 3 - Real.sin α)
  in (Real.sqrt ((3 - Real.cos α) ^ 2 + (Real.sin α) ^ 2) = Real.sqrt ((Real.cos α) ^ 2 + (3 - Real.sin α) ^ 2)) → 
  (Real.tan α = 1) → α = 5 * π / 4) :=
begin
  intros α hcond hdist htan,
  sorry
end

theorem expr_value :
  (∀ α : ℝ, (α > π / 2 ∧ α < 3 * π / 2) →
  let A := (3, 0)
      B := (0, 3)
      C := (Real.cos α, Real.sin α)
      AC := (Real.cos α - 3, Real.sin α)
      BC := (Real.cos α, Real.sin α - 3)
  in (AC.1 * BC.1 + AC.2 * BC.2 = -1) → 
  (Real.sin α + Real.cos α = 2 / 3) → 
  (2 * (Real.sin α) ^ 2 + 2 * Real.sin α * Real.cos α) / (1 + Real.tan α) = -(5 / 9)) :=
begin
  intros α hcond hdots hsum,
  sorry
end

end angle_alpha_eq_expr_value_l406_406687


namespace right_triangle_area_and_perimeter_l406_406486

theorem right_triangle_area_and_perimeter (a c : ℕ) (h₁ : c = 13) (h₂ : a = 5) :
  ∃ (b : ℕ), b^2 = c^2 - a^2 ∧
             (1/2 : ℝ) * (a : ℝ) * (b : ℝ) = 30 ∧
             (a + b + c : ℕ) = 30 :=
by
  sorry

end right_triangle_area_and_perimeter_l406_406486


namespace sum_of_floor_sqrt_1_to_25_l406_406990

theorem sum_of_floor_sqrt_1_to_25 : 
  ( ∑ i in Finset.range 26, ∥Real.sqrt i∥ ) = 75 :=
by
  sorry

end sum_of_floor_sqrt_1_to_25_l406_406990


namespace hula_hoop_ratio_l406_406446

variable (Nancy Casey Morgan : ℕ)
variable (hula_hoop_time_Nancy : Nancy = 10)
variable (hula_hoop_time_Casey : Casey = Nancy - 3)
variable (hula_hoop_time_Morgan : Morgan = 21)

theorem hula_hoop_ratio (hula_hoop_time_Nancy : Nancy = 10) (hula_hoop_time_Casey : Casey = Nancy - 3) (hula_hoop_time_Morgan : Morgan = 21) :
  Morgan / Casey = 3 := by
  sorry

end hula_hoop_ratio_l406_406446


namespace compute_expression_l406_406290

theorem compute_expression :
  120 * 2400 - 20 * 2400 - 100 * 2400 = 0 :=
sorry

end compute_expression_l406_406290


namespace cost_of_snake_toy_l406_406297

-- Given conditions
def cost_of_cage : ℝ := 14.54
def dollar_bill_found : ℝ := 1.00
def total_cost : ℝ := 26.30

-- Theorem to find the cost of the snake toy
theorem cost_of_snake_toy : 
  (total_cost + dollar_bill_found - cost_of_cage) = 12.76 := 
  by sorry

end cost_of_snake_toy_l406_406297


namespace beth_wins_with_starting_configuration_l406_406965

-- Define the nim-values of walls with lengths 1 to 6
def nim_value : ℕ → ℕ
| 1 := 1
| 2 := 2
| 3 := 3
| 4 := 1
| 5 := 4
| 6 := 3
| _ := sorry -- values for walls greater than 6 are not needed for this proof

-- Define the nim-value of a configuration as the xor of nim-values of individual walls
def configuration_nim_value (walls : list ℕ) : ℕ :=
walls.foldl (λ acc wall_length, acc xor nim_value wall_length) 0

-- The main proof statement
theorem beth_wins_with_starting_configuration :
  configuration_nim_value [6, 2, 1] = 0 :=
by
  -- Prove that the nim-value of the configuration (6,2,1) is 0
  sorry

end beth_wins_with_starting_configuration_l406_406965


namespace pennies_on_friday_l406_406833

-- Define the initial number of pennies and the function for doubling
def initial_pennies : Nat := 3
def double (n : Nat) : Nat := 2 * n

-- Prove the number of pennies on Friday
theorem pennies_on_friday : double (double (double (double initial_pennies))) = 48 := by
  sorry

end pennies_on_friday_l406_406833


namespace commodity_house_value_l406_406746

noncomputable def P : ℝ := 600000
noncomputable def r : ℝ := 0.1
noncomputable def t : ℕ := 4

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t

theorem commodity_house_value :
  abs (compound_interest P r t - 878000) < 1000 :=
by
  sorry

end commodity_house_value_l406_406746


namespace afternoon_registration_l406_406452

variable (m a t morning_absent : ℕ)

theorem afternoon_registration (m a t morning_absent afternoon : ℕ) (h1 : m = 25) (h2 : a = 4) (h3 : t = 42) (h4 : morning_absent = 3) : 
  afternoon = t - (m - morning_absent + morning_absent + a) :=
by sorry

end afternoon_registration_l406_406452


namespace sam_watermelons_with_spots_l406_406461

noncomputable def initial_watermelons := 4

noncomputable def doubled_watermelons := initial_watermelons * 2

noncomputable def remaining_watermelons := doubled_watermelons - 2

noncomputable def total_watermelons := remaining_watermelons + 3

noncomputable def watermelons_with_spots := (total_watermelons / 2).toNat

theorem sam_watermelons_with_spots : 
  watermelons_with_spots = 4 := by
  sorry

end sam_watermelons_with_spots_l406_406461


namespace perpendicular_midpoints_l406_406450

-- Given Definitions and Conditions
variables (A B C M I1 I2 J1 J2 : Type) 
variables [add_comm_group A] [add_comm_group B] [add_comm_group C]
variables [add_comm_group M] [add_comm_group I1] [add_comm_group I2]
variables [add_comm_group J1] [add_comm_group J2]

-- Assumptions
variables (hM : M ∈ (segment A B))
variables (hI1 : I1 = incenter (triangle A C M))
variables (hJ1 : J1 = excenter (triangle A C M) C)
variables (hI2 : I2 = incenter (triangle B C M))
variables (hJ2 : J2 = excenter (triangle B C M) C)

-- Midpoints of segments I1I2 and J1J2
noncomputable def midpoint_I1_I2 : Type := midpoint I1 I2
noncomputable def midpoint_J1_J2 : Type := midpoint J1 J2

-- Prove the target statement
theorem perpendicular_midpoints (A B C M I1 I2 J1 J2 : Type) 
  [add_comm_group A] [add_comm_group B] [add_comm_group C] 
  [add_comm_group M] [add_comm_group I1] [add_comm_group I2] 
  [add_comm_group J1] [add_comm_group J2]
  (hM : M ∈ (segment A B)) 
  (hI1 : I1 = incenter (triangle A C M)) 
  (hJ1 : J1 = excenter (triangle A C M) C) 
  (hI2 : I2 = incenter (triangle B C M)) 
  (hJ2 : J2 = excenter (triangle B C M) C) : 
  (line_through (midpoint I1 I2) (midpoint J1 J2)).is_perpendicular (segment A B) :=
  sorry

end perpendicular_midpoints_l406_406450


namespace rate_of_glass_bowls_l406_406259

theorem rate_of_glass_bowls (R : ℝ) :
    (∃ (R : ℝ),
        (118 * R + 0.08050847457627118 * 118 * R = 1530) ∧ R = 12) :=
by
    existsi 12
    split
    {
        norm_num
    }
    {
        refl
    }

end rate_of_glass_bowls_l406_406259


namespace true_propositions_l406_406429

-- Definitions for lines and plane as given in conditions
variable (a b c : Type)
variable (y : Type)

-- Proposition ①
def prop_one (a b c : Type) := (a ∥ b) ∧ (b ∥ c) → (a ∥ c)

-- Proposition ④
def prop_four (a b : Type) (y : Type) := (a ⊥ y) ∧ (b ⊥ y) → (a ∥ b)

-- Statement of theorem
theorem true_propositions (a b c : Type) (y : Type) : 
  (prop_one a b c) ∧ (prop_four a b y) :=
  sorry

end true_propositions_l406_406429


namespace triangle_obtuse_of_sin_inequality_l406_406027

variables (A B C: ℝ)
variables (a b c : ℝ) [IsTriangle a b c]

theorem triangle_obtuse_of_sin_inequality
  (h1 : - √ 3 * Real.sin A * Real.sin B < Real.sin A ^ 2 + Real.sin B ^ 2 - Real.sin C ^ 2)
  (h2 : Real.sin A ^ 2 + Real.sin B ^ 2 - Real.sin C ^ 2 < - Real.sin A * Real.sin B) :
  ∃ (A B C : ℝ), ∠ C > 90 := sorry

end triangle_obtuse_of_sin_inequality_l406_406027


namespace area_triangle_par_asymp_perp_OP_OQ_const_dist_MN_l406_406046

-- (1) Area of triangle problem
theorem area_triangle_par_asymp (x y : ℝ) (hx : 2 * x ^ 2 - y ^ 2 = 1) 
  : ∃ (area : ℝ), area = sqrt 2 / 8 :=
sorry

-- (2) Perpendicular OP and OQ problem
theorem perp_OP_OQ (x y : ℝ) (hx : 2 * x ^ 2 - y ^ 2 = 1) 
  (slope : ℝ) (hs : slope = 1) (tangent : x ^ 2 + y ^ 2 = 1)
  : ∃ (P Q : ℝ × ℝ), (P.1 * Q.1 + P.2 * Q.2 = 0) :=
sorry

-- (3) Distance from origin to line MN problem
theorem const_dist_MN (M N : ℝ × ℝ) 
  (hM : 2 * M.1 ^ 2 - M.2 ^ 2 = 1) 
  (hN : 4 * N.1 ^ 2 + N.2 ^ 2 = 1) 
  (perp : M.1 * N.1 + M.2 * N.2 = 0)
  : ∃ (d : ℝ), d = sqrt 3 / 3 :=
sorry

end area_triangle_par_asymp_perp_OP_OQ_const_dist_MN_l406_406046


namespace perimeter_triangle_CEF_l406_406115

open Real
open EuclideanGeometry

-- Definitions based on the conditions of the problem
def square_side_length := 1

def angle_EAF := 45

/--
  Theorem: Given points E and F on the sides BC and CD of the square ABCD with side length 1,
  and the angle EAF is 45 degrees, the perimeter of the triangle CEF is 2.
-/
theorem perimeter_triangle_CEF
  (A B C D E F : Point)
  (square_length : A\(B) = square_side_length ∧ B(C) = square_side_length ∧ C(D) = square_side_length ∧ D(A) = square_side_length)
  (E_on_BC : E ∈ line_segment B C)
  (F_on_CD : F ∈ line_segment C D)
  (angle_EAF : ∠ E A F = 45) :
  perimeter (triangle C E F) = 2 :=
sorry

end perimeter_triangle_CEF_l406_406115


namespace not_all_same_probability_l406_406527

-- Definition of the total number of outcomes when rolling 5 8-sided dice
def total_outcomes : ℕ := 8^5

-- Definition of the number of outcomes where all five dice show the same number
def same_number_outcomes : ℕ := 8

-- Definition to find the probability that not all 5 dice show the same number
def probability_not_all_same : ℚ := 1 - (same_number_outcomes / total_outcomes)

-- Statement of the main theorem
theorem not_all_same_probability : probability_not_all_same = (4095 : ℚ) / 4096 :=
by
  rw [probability_not_all_same, same_number_outcomes, total_outcomes]
  -- Simplification steps would go here, but we use sorry to skip the proof
  sorry

end not_all_same_probability_l406_406527


namespace arccos_gt_arctan_on_interval_l406_406645

noncomputable def c : ℝ := sorry -- placeholder for the numerical solution of arccos x = arctan x

theorem arccos_gt_arctan_on_interval (x : ℝ) (hx : -1 ≤ x ∧ x < c) :
  Real.arccos x > Real.arctan x := 
sorry

end arccos_gt_arctan_on_interval_l406_406645


namespace combined_perimeter_l406_406587

theorem combined_perimeter (side_square : ℝ) (a b c : ℝ) (diameter : ℝ) 
  (h_square : side_square = 7) 
  (h_triangle : a = 5 ∧ b = 6 ∧ c = 7) 
  (h_diameter : diameter = 4) : 
  4 * side_square + (a + b + c) + (2 * Real.pi * (diameter / 2) + diameter) = 50 + 2 * Real.pi := 
by 
  sorry

end combined_perimeter_l406_406587


namespace solve_complex_eq_l406_406356

theorem solve_complex_eq (z : ℂ) (a b : ℝ) (h_z : z = a + b * complex.I)
  (h1 : -2 * a = 2)
  (h2 : a^2 + b^2 + 2 * b = 1) :
  z = -1 ∨ z = -1 - 2 * complex.I := by
  sorry

end solve_complex_eq_l406_406356


namespace petya_could_not_spend_5000_l406_406821

-- Define the problem conditions and proof functions
def petya_spent_at_least_5000 (spent : ℕ) : Prop :=
  spent >= 5000

-- Main theorem to prove
theorem petya_could_not_spend_5000 :
  ∀ (bill_count : ℕ),
  let initial_money := 100 * bill_count in
  let total_spent := initial_money / 2 in
  total_spent < 5000 ->
  ∀ (books : list ℕ),
  (∀ b ∈ books, b > 0) -> -- The cost of each book is a positive integer.
  ∃ n : ℕ, 
  n = bill_count ->
  ¬ petya_spent_at_least_5000 total_spent := 
by
  intros bill_count books Hbook_zero b_in_books H_spent conditions
  sorry -- proof to be filled in

end petya_could_not_spend_5000_l406_406821


namespace number_of_liars_l406_406033

/-- Definition of conditions -/
def total_islands : Nat := 17
def population_per_island : Nat := 119

-- Conditions based on the problem description
def islands_yes_first_question : Nat := 7
def islands_no_first_question : Nat := total_islands - islands_yes_first_question

def islands_no_second_question : Nat := 7
def islands_yes_second_question : Nat := total_islands - islands_no_second_question

def minimum_knights_for_no_second_question : Nat := 60  -- At least 60 knights

/-- Main theorem -/
theorem number_of_liars : 
  ∃ x y: Nat, 
    (x + (islands_no_first_question - y) = islands_yes_first_question ∧ 
     y - x = 3 ∧ 
     60 * x + 59 * y + 119 * (islands_no_first_question - y) = 1010 ∧
     (total_islands * population_per_island - 1010 = 1013)) := by
  sorry

end number_of_liars_l406_406033


namespace ganesh_speed_x_to_y_l406_406558

-- Define the conditions
variables (D : ℝ) (V : ℝ)

-- Theorem statement: Prove that Ganesh's average speed from x to y is 44 km/hr
theorem ganesh_speed_x_to_y
  (H1 : 39.6 = 2 * D / (D / V + D / 36))
  (H2 : V = 44) :
  true :=
sorry

end ganesh_speed_x_to_y_l406_406558


namespace total_guests_l406_406191

-- Define the conditions.
def number_of_tables := 252.0
def guests_per_table := 4.0

-- Define the statement to prove.
theorem total_guests : number_of_tables * guests_per_table = 1008.0 := by
  sorry

end total_guests_l406_406191


namespace symmetry_axis_of_given_function_l406_406484

-- Definition and conditions
def given_function (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 6)

-- Target statement to prove
theorem symmetry_axis_of_given_function : ∃ k : ℤ, (λ (x : ℝ), x = k * Real.pi / 2 + Real.pi / 6) ∧ given_function = (λ (x : ℝ), given_function (-(k * Real.pi / 2 + Real.pi / 6) - x)) :=
begin
  sorry
end

end symmetry_axis_of_given_function_l406_406484


namespace find_positive_q_l406_406008

noncomputable theory

def f (k : ℤ) (x : ℝ) : ℝ := x ^ (-k^2 + k + 2)

theorem find_positive_q :
  (∀ k ∈ {0, 1} : Set ℤ, f k 2 < f k 3) →
  ∃ q > 0, 
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, 
    1 - q * (x : ℝ) ^ 2 + (2 * q - 1) * x ∈ Set.Icc (-4 : ℝ) (17 / 8)) := sorry

end find_positive_q_l406_406008


namespace tiling_lateral_parallelepiped_even_count_l406_406488

open Nat

universe u

-- Given the problem statement conditions
variables {a b c : ℕ}

-- This function will represent the problem statement for Lean to prove
def even_lateral_tiling_count (a b c : ℕ) [Hc : Odd c] : Prop := sorry

-- The main statement that will be proved
theorem tiling_lateral_parallelepiped_even_count (a b c : ℕ) (H : Odd c) :
  even_lateral_tiling_count a b c :=
sorry

end tiling_lateral_parallelepiped_even_count_l406_406488


namespace cookie_division_l406_406565

theorem cookie_division (C : ℝ) (blue_fraction : ℝ := 1/4) (green_fraction_of_remaining : ℝ := 5/9)
  (remaining_fraction : ℝ := 3/4) (green_fraction : ℝ := 5/12) :
  blue_fraction + green_fraction = 2/3 := by
  sorry

end cookie_division_l406_406565


namespace expression_equals_one_l406_406977

noncomputable def compute_expression : ℝ :=
  (1 - 1 / Real.cos (30 * Real.pi / 180)) *
  (1 + 1 / Real.sin (60 * Real.pi / 180)) *
  (1 - 1 / Real.sin (30 * Real.pi / 180)) *
  (1 + 1 / Real.cos (60 * Real.pi / 180))

theorem expression_equals_one : compute_expression = 1 :=
by
  sorry

end expression_equals_one_l406_406977


namespace total_cost_of_sandwiches_and_sodas_l406_406201

theorem total_cost_of_sandwiches_and_sodas :
  let price_sandwich := 2.49
  let num_sandwiches := 2
  let price_soda := 1.87
  let num_sodas := 4
  let total_cost := 12.46
  (num_sandwiches * price_sandwich + num_sodas * price_soda = total_cost) :=
begin
  let price_sandwich := 2.49
  let num_sandwiches := 2
  let price_soda := 1.87
  let num_sodas := 4
  let total_cost := 12.46
  sorry -- proof not required
end

end total_cost_of_sandwiches_and_sodas_l406_406201


namespace distance_center_to_tangency_eq_radius_l406_406236

noncomputable def distance (a b : ℝ) : ℝ := abs (a - b)

def circle_center (h : ℝ) : ℝ × ℝ := (h, 0)

def point_of_tangency (x : ℝ) : ℝ × ℝ := (x, 0)

theorem distance_center_to_tangency_eq_radius (h : ℝ) (r : ℝ) :
  distance h 3 = r ↔ distance (circle_center h).1 (point_of_tangency 3).1 = r :=
by
  intro h r
  rw [circle_center, point_of_tangency, distance]
  sorry

end distance_center_to_tangency_eq_radius_l406_406236


namespace geometric_progression_terms_l406_406478

theorem geometric_progression_terms 
  (q b4 S_n : ℚ) 
  (hq : q = 1/3) 
  (hb4 : b4 = 1/54) 
  (hS : S_n = 121/162) 
  (b1 : ℚ) 
  (hb1 : b1 = b4 * q^3)
  (Sn : ℚ) 
  (hSn : Sn = b1 * (1 - q^5) / (1 - q)) : 
  ∀ (n : ℕ), S_n = Sn → n = 5 :=
by
  intro n hn
  sorry

end geometric_progression_terms_l406_406478


namespace karthik_weight_average_l406_406747

noncomputable def average_probable_weight_of_karthik (weight : ℝ) : Prop :=
  (55 < weight ∧ weight < 62) ∧
  (50 < weight ∧ weight < 60) ∧
  (weight ≤ 58) →
  weight = 56.5

theorem karthik_weight_average :
  ∀ weight : ℝ, average_probable_weight_of_karthik weight :=
by
  sorry

end karthik_weight_average_l406_406747


namespace division_result_l406_406085

def n : ℕ := 16^1024

theorem division_result : n / 8 = 2^4093 :=
by sorry

end division_result_l406_406085


namespace problem_statement_l406_406619

def sqrt_300 := real.sqrt 300
def mul_5_8 := 5 * 8
def pow_2_3 := 2^3
def factorial_4 := nat.factorial 4
def step_5_result := (sqrt_300 + mul_5_8) / pow_2_3
def final_result := step_5_result + factorial_4
def rounded_result := real.round (final_result * 10000) / 10000

theorem problem_statement :
  rounded_result = 31.1651 := sorry

end problem_statement_l406_406619


namespace congruent_triangle_with_colored_points_l406_406582

theorem congruent_triangle_with_colored_points (colors : set ℕ) (T : Triangle) (h1 : colors.card = 1992) (h2 : T.is_on_plane) :
  ∃ T' : Triangle, T'.is_congruent_to T ∧ ∀ (s1 s2 : Triangle.side), s1 ≠ s2 → ∃ p1 ∈ s1.interior, ∃ p2 ∈ s2.interior, p1.color = p2.color :=
by
  sorry

end congruent_triangle_with_colored_points_l406_406582


namespace product_nonreal_roots_eq_2009_l406_406653

noncomputable def poly (x : ℂ) : ℂ :=
  x^5 - 5 * x^4 + 10 * x^3 - 10 * x^2 + 5 * x - 1

theorem product_nonreal_roots_eq_2009 :
  let roots := {x : ℂ | poly x = 2006}
  let nonreal_roots := {x ∈ roots | ¬ x.re=0 ∧ ¬ x.im=0}
  nonreal_roots.prod id = 2009 :=
by
  let roots := {x : ℂ | poly x = 2006}
  let nonreal_roots := {x ∈ roots | ¬ x.re=0 ∧ ¬ x.im=0}
  exact nonreal_roots.prod id = 2009

end product_nonreal_roots_eq_2009_l406_406653


namespace quinary_to_octal_correct_evaluate_polynomial_correct_l406_406924

noncomputable def quinary_to_octal : nat :=
  let decimal := 1 * 5^3 + 2 * 5^2 + 3 * 5^1 + 4 * 5^0 in
  let octal := 2 + 0 * 8 + 3 * 8^2 in
  octal

noncomputable def evaluate_polynomial (x : nat) : nat :=
  let f := (7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x) in
  f

theorem quinary_to_octal_correct : quinary_to_octal = 302 := 
  by sorry

theorem evaluate_polynomial_correct : evaluate_polynomial 3 = 21324 := 
  by sorry

end quinary_to_octal_correct_evaluate_polynomial_correct_l406_406924


namespace milk_chocolate_bars_count_l406_406443

theorem milk_chocolate_bars_count (dark_bars : ℕ) (almond_bars : ℕ) (white_bars : ℕ) 
    (percent : ℕ) (h1 : dark_bars = 25) (h2 : almond_bars = 25) (h3 : white_bars = 25) (h4 : percent = 25) :
    ∃ milk_bars : ℕ, milk_bars = 25 :=
by
  use 25
  intros
  sorry

end milk_chocolate_bars_count_l406_406443


namespace perpendicularity_condition_l406_406089

variables {α : Type*} [Field α]

-- Geometric definitions and assumptions
variables (A B C D E F G H T I O A' M : α) 
variables (triangle_ABC : Triangle α) (incircle : Circle α)
variables (circumcircle_ABC : Circle α) (circumcircle_AEF : Circle α)

/-- Given an acute triangle ABC with incenter I and circumcenter O, 
    the incircle touches sides BC, CA, and AB at points D, E, and F, respectively. 
    A' is the reflection of A over O. If the circumcircles of ABC and A'EF 
    intersect at G, and the circumcircles of AMG and A'EF meet at H ≠ G, 
    where M is the midpoint of EF, prove that if GH and EF meet at T, 
    then DT is perpendicular to EF. -/
theorem perpendicularity_condition 
    (triangle_ABC : acute_triangle A B C)
    (incenter_I : is_incenter I triangle_ABC)
    (circumcenter_O : is_circumcenter O triangle_ABC)
    (reflect_A' : reflect_over A O = A')
    (incircle_touches_DE : incircle.touches D E at_triangle_sides A B C)
    (incircle_touches_EF : incircle.touches E F at_triangle_sides B C A)
    (incircle_touches_FD : incircle.touches F D at_triangle_sides C A B)
    (M_midpoint : is_midpoint M E F)
    (circumcircles_intersect_G : circumcircle_ABC ∩ circumcircle_AEF = {G})
    (circumcircles_meet_H : (circumcircle (triangle AMG)).meet (circumcircle (A'EF)) = {H, G})
    (GH_meet_EF_AT_T : meet GH EF = {T})
    (H_not_eq_G : H ≠ G) :
    perpendicular D T E F := 
sorry

end perpendicularity_condition_l406_406089


namespace parallelogram_area_l406_406135

theorem parallelogram_area (b : ℕ) (h : ℕ) (A : ℕ)
  (h_eq : h = 2 * b)
  (b_eq : b = 12)
  (A_eq : A = b * h) :
  A = 288 :=
by
  rw [b_eq, h_eq, Nat.mul_comm, Nat.mul_assoc]
  exact rfl

end parallelogram_area_l406_406135


namespace sum_is_24_l406_406205

-- Define the conditions
def A := 3
def B := 7 * A

-- Define the theorem to prove that the sum is 24
theorem sum_is_24 : A + B = 24 :=
by
  -- Adding sorry here since we're not required to provide the proof
  sorry

end sum_is_24_l406_406205


namespace find_k_and_direction_l406_406711

noncomputable def a : ℝ × ℝ := (1, 0)
noncomputable def b : ℝ × ℝ := (0, 1)
def c (k : ℝ) : ℝ × ℝ := (k * a.1 + b.1, k * a.2 + b.2)
def d : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def parallel (u v : ℝ × ℝ) : Prop := ∃ λ : ℝ, u = (λ * v.1, λ * v.2)

theorem find_k_and_direction (k : ℝ) (h : parallel (c k) d) : k = -1 ∧ ∃ λ : ℝ, λ < 0 ∧ c k = (λ * d.1, λ * d.2) :=
by 
    sorry

end find_k_and_direction_l406_406711


namespace common_chord_eq_l406_406142

-- Conditions: Definitions of the equations of the circles.
def circle1 := (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 8 = 0
def circle2 := (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

-- Statement: The equation of the line where the common chord lies
theorem common_chord_eq (x y : ℝ) (h1 : circle1 x y) (h2 : circle2 x y) : x - y + 1 = 0 := 
sorry

end common_chord_eq_l406_406142


namespace evaluate_expression_l406_406637

theorem evaluate_expression : (164^2 - 148^2) / 16 = 312 := 
by 
  sorry

end evaluate_expression_l406_406637


namespace part_a_part_b_l406_406193

-- Define the property (S)
def has_property_S (A : set (ℝ × ℝ)) : Prop :=
  (|A| >= 3) ∧ (∀ ⦃u⦄, u ∈ A → ∃ v w, v ∈ A ∧ w ∈ A ∧ v ≠ w ∧ u = v + w)

-- Theorem 1: For all n ≥ 6, there exists a set A of n vectors with property (S)
theorem part_a (n : ℕ) (hn : n ≥ 6) : ∃ A : set (ℝ × ℝ), |A| = n ∧ has_property_S A := sorry

-- Theorem 2: If a set A of non-zero vectors has property (S), then it has at least 6 elements
theorem part_b (A : set (ℝ × ℝ)) (hA : has_property_S A) : |A| ≥ 6 := sorry

end part_a_part_b_l406_406193


namespace range_of_k_l406_406855

noncomputable def f (x : ℝ) : ℝ := 2^x
noncomputable def g (x k : ℝ) : ℝ := x^2 - 2 * k * x + 5 / 2

theorem range_of_k (k : ℝ) :
  (∀ s ∈ set.Icc (-1 : ℝ) 2, ∃ t ∈ set.Icc k (2 * k + 1), f s = g t k) ↔ 
  (k ∈ set.Ici (real.sqrt 2)) := 
sorry

end range_of_k_l406_406855


namespace complex_imaginary_a_value_l406_406736

theorem complex_imaginary_a_value (a : ℝ) :
  (∃ (a : ℝ), let z := (a + 2 * complex.I) / (1 + complex.I) in z.re = 0) → a = -2 :=
by
  sorry

end complex_imaginary_a_value_l406_406736


namespace certain_number_equals_l406_406384

theorem certain_number_equals (p q : ℚ) (h1 : 3 / p = 8) (h2 : 3 / q = 18) (h3 : p - q = 0.20833333333333334) : q = 1/6 := sorry

end certain_number_equals_l406_406384


namespace Peter_can_complete_work_in_20_days_l406_406445

-- Define the premises
constants (W : ℝ) (M P : ℝ)
axiom h1 : M + P = 1 / 20
axiom h2 : W / 2 / 10 = P

-- Proven statement
theorem Peter_can_complete_work_in_20_days : W / P = 20 := 
by
  sorry

end Peter_can_complete_work_in_20_days_l406_406445


namespace probability_perfect_square_l406_406265

theorem probability_perfect_square (rolls : ℕ → ℕ) (cond : ∀ i, 1 ≤ rolls i ∧ rolls i ≤ 8) 
  (at_least_one_4_5_6 : ∃ i, rolls i = 4 ∨ rolls i = 5 ∨ rolls i = 6) : 
  (1 / 256 : ℚ) = 57 / 256 :=
by
  sorry

end probability_perfect_square_l406_406265


namespace prove_a_plus_b_l406_406680

noncomputable def A : set ℝ := { x | x^3 + 3x^2 + 2x > 0 }
noncomputable def B (a b : ℝ) : set ℝ := { x | x^2 + a * x + b ≤ 0 }

theorem prove_a_plus_b (a b : ℝ) :
  (A ∩ B a b = {x : ℝ | 0 < x ∧ x ≤ 2}) →
  (A ∪ B a b = {x : ℝ | x > -2}) →
  a + b = -3 :=
by
  sorry

end prove_a_plus_b_l406_406680


namespace windingNumberOdd_l406_406239

variable {P : Type} [polygon_chain P] (O : Point) -- P is a closed polygonal chain, O is a point

-- Definitions for conditions
def isClosed (P : Type) [polygon_chain P] : Prop := true -- Placeholder for actual definition
def symmetricToPoint (P : Type) [polygon_chain P] (O : Point) : Prop := true -- Placeholder
def evenVertices (P : Type) [polygon_chain P] : Prop := true -- Placeholder

-- Definition of winding number, using placeholder types and functions
noncomputable def windingNumber (P : polygon_chain) (O : Point) : ℤ := sorry -- Placeholder

-- Main theorem statement
theorem windingNumberOdd (h1 : isClosed P) (h2 : symmetricToPoint P O) (h3 : evenVertices P) :
  windingNumber P O % 2 = 1 := sorry

end windingNumberOdd_l406_406239


namespace complex_expression_simplification_l406_406226

-- Given: i is the imaginary unit
def i := Complex.I

-- Prove that the expression simplifies to -1
theorem complex_expression_simplification : (i^3 * (i + 1)) / (i - 1) = -1 := by
  -- We are skipping the proof and adding sorry for now
  sorry

end complex_expression_simplification_l406_406226


namespace find_phi_l406_406144

theorem find_phi :
  ∃ (ϕ : ℝ), 
  (0 < ϕ ∧ ϕ < π / 2) ∧
  (∀ x : ℝ, 3 * sin(2 * (x + ϕ) + π / 4) = -3 * sin(2 * (-x + ϕ) + π / 4)) ∧
  ϕ = 3 * π / 8 :=
by
  sorry

end find_phi_l406_406144


namespace sum_floor_sqrt_1_to_25_l406_406997

theorem sum_floor_sqrt_1_to_25 :
  (∑ n in finset.range 25, ⌊real.sqrt (n + 1)⌋) = 75 :=
sorry

end sum_floor_sqrt_1_to_25_l406_406997


namespace sum_modulo_nine_l406_406654

theorem sum_modulo_nine :
  (88135 + 88136 + 88137 + 88138 + 88139 + 88140) % 9 = 3 := 
by
  sorry

end sum_modulo_nine_l406_406654


namespace fraction_playing_in_field_l406_406031

def class_size : ℕ := 50
def students_painting : ℚ := 3/5
def students_left_in_classroom : ℕ := 10

theorem fraction_playing_in_field :
  (class_size - students_left_in_classroom - students_painting * class_size) / class_size = 1/5 :=
by
  sorry

end fraction_playing_in_field_l406_406031


namespace rice_grains_difference_l406_406593

theorem rice_grains_difference :
  let grains_on_square (k : ℕ) : ℕ := 3^k in
  let grains_12th_square := grains_on_square 12 in
  let grains_first_9_squares_sum := 
    3 * ((3^9 - 1) / (3 - 1)) in
  grains_12th_square - grains_first_9_squares_sum = 501693 :=
by
  let grains_on_square (k : ℕ) : ℕ := 3^k
  have h1 : grains_on_square 12 = 531441 := by rfl
  have h2 : grains_first_9_squares_sum = 29748 := by
    calc
      grains_first_9_squares_sum
      = 3 * ((3^9 - 1) / (3 - 1)) : rfl
      ... = 3 * (19683 - 1) / 2 : by { sorry } -- Detailed steps for simplification are omitted for clarity
      ... = 29748 : by norm_num
  calc
    grains_12th_square - grains_first_9_squares_sum
    = 531441 - 29748 : by rw [h1, h2]
    ... = 501693 : by norm_num

end rice_grains_difference_l406_406593


namespace find_k_l406_406082

noncomputable def f : ℝ → ℝ
| x := if x < 0 then -x else 3 * x - 12

theorem find_k (k : ℝ) (h_neg_k : k < 0) : f (f (f 4)) = f (f (f k)) ↔ k = -20 / 3 := 
by
  sorry

end find_k_l406_406082


namespace incorrect_reasoning_form_l406_406151

-- Define what it means to be a rational number
def is_rational (x : ℚ) : Prop := true

-- Define what it means to be a fraction
def is_fraction (x : ℚ) : Prop := true

-- Define what it means to be an integer
def is_integer (x : ℤ) : Prop := true

-- State the premises as hypotheses
theorem incorrect_reasoning_form (h1 : ∃ x : ℚ, is_rational x ∧ is_fraction x)
                                 (h2 : ∀ z : ℤ, is_rational z) :
  ¬ (∀ z : ℤ, is_fraction z) :=
by
  -- We are stating the conclusion as a hypothesis that needs to be proven incorrect
  sorry

end incorrect_reasoning_form_l406_406151


namespace trig_expression_equality_l406_406974

theorem trig_expression_equality :
  (1 - 1 / (Real.cos (Real.pi / 6))) *
  (1 + 1 / (Real.sin (Real.pi / 3))) *
  (1 - 1 / (Real.sin (Real.pi / 6))) *
  (1 + 1 / (Real.cos (Real.pi / 3))) = 1 :=
by
  have hcos30 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2 := sorry
  have hsin60 : Real.sin (Real.pi / 3) = (Real.sqrt 3) / 2 := sorry
  have hsin30 : Real.sin (Real.pi / 6) = 1 / 2 := sorry
  have hcos60 : Real.cos (Real.pi / 3) = 1 / 2 := sorry
  sorry

end trig_expression_equality_l406_406974


namespace tom_drives_distance_before_karen_wins_l406_406063

def karen_late_minutes := 4
def karen_speed_mph := 60
def tom_speed_mph := 45

theorem tom_drives_distance_before_karen_wins : 
  ∃ d : ℝ, d = 21 := by
  sorry

end tom_drives_distance_before_karen_wins_l406_406063


namespace yan_ratio_l406_406907

theorem yan_ratio (w x y : ℝ) (h₁ : ∀ t : ℝ, t = y / w -> t = x / w + (x + y) / (6 * w)) :
  x / y = 5 / 7 :=
by 
  -- Given h₁, rewrite in terms of y and x
  have h : y / w = (6 * x + y) / (6 * w),
  {
    apply h₁,
    sorry,
  },
  -- Simplify the equation
  sorry

end yan_ratio_l406_406907


namespace savings_in_a_year_l406_406254

-- Definitions of the conditions
def expenditure_first_3_months := 1700 * 3
def expenditure_next_4_months := 1550 * 4
def expenditure_last_5_months := 1800 * 5
def total_expenditure := expenditure_first_3_months + expenditure_next_4_months + expenditure_last_5_months
def monthly_income := 2125
def yearly_income := monthly_income * 12

-- Theorem statement
theorem savings_in_a_year : 25500 - total_expenditure = 5200 :=
by
  have h1 : total_expenditure = 1700 * 3 + 1550 * 4 + 1800 * 5 := rfl
  have h2 : total_expenditure = 5100 + 6200 + 9000 := by rw [h1]
  have h3 : total_expenditure = 20300 := by norm_num [h2]
  have h4 : yearly_income = 2125 * 12 := rfl
  have h5 : yearly_income = 25500 := by norm_num [h4]
  rw [h3, h5]
  norm_num

sorry -- Proof steps here

end savings_in_a_year_l406_406254


namespace probability_log_condition_l406_406190

-- Define conditions
def is_valid_die_value (n : ℕ) : Prop :=
  n ∈ {1, 2, 3, 4, 5, 6}

-- Define the probability calculation
def count_favorable_outcomes : ℕ :=
  { (x, y) | is_valid_die_value x ∧ is_valid_die_value y ∧ 2 * x = y }.to_finset.card

def total_possible_outcomes : ℕ := 36

-- Define the final probability
def probability : ℚ :=
  count_favorable_outcomes.to_rat / total_possible_outcomes.to_rat

-- Statement we want to prove
theorem probability_log_condition :
  probability = 1 / 12 :=
by
  sorry

end probability_log_condition_l406_406190


namespace even_function_value_l406_406921

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 3 * x - 1 else 0  -- We are not interested in the actual value for x < 0 other than at -1 which is defined by even property.

theorem even_function_value : ∀ (f : ℝ → ℝ), 
  (∀ x : ℝ, f (-x) = f x) →
  (∀ x : ℝ, 0 ≤ x → f x = 3 * x - 1) →
  f (-1) = 2 :=
by
  intro f h_even h_nonneg
  have h1 : f (-1) = f 1 := h_even 1
  have h2 : f 1 = 3 * 1 - 1 := h_nonneg 1 (by norm_num)
  rw [h1, h2]
  norm_num
   
  sorry -- placeholder for remaining proof part

end even_function_value_l406_406921


namespace carter_second_pick_green_probability_is_24_59_percent_l406_406288

noncomputable def probability_green_second_pick 
  (greens : ℕ) (reds : ℕ) (blues : ℕ) (oranges : ℕ) (carter_eats_green : ℕ)
  (carter_eats_red : ℕ) (sister_adds_yellow : ℕ) (alex_adds_purple : ℕ) 
  (alex_eats_orange : ℕ) (alex_eats_yellow : ℕ) (cousin_adds_brown : ℕ) 
  (cousin_eats_blue_multiplier : ℕ) (initial_red_pick : ℕ)
  (sister_adds_pink_if_red : ℕ)
  : ℚ :=
  let total_initial := greens + reds + blues + oranges in
  let remaining_green := greens - carter_eats_green in
  let remaining_red := reds - carter_eats_red in
  let remaining_red_after_sister := remaining_red / 2 in
  let remaining_yellow := sister_adds_yellow - alex_eats_yellow in
  let remaining_orange := oranges - alex_eats_orange in
  let remaining_purple := alex_adds_purple in
  let remaining_blue := blues - (cousin_eats_blue_multiplier * blues) in
  let remaining_brown := cousin_adds_brown in
  let remaining_pink := sister_adds_pink_if_red in
  let total := remaining_green + remaining_red_after_sister +
               remaining_yellow + remaining_orange + remaining_purple +
               remaining_brown + remaining_pink in
  let final_red := remaining_red_after_sister - initial_red_pick in
  let final_total := total - initial_red_pick in
  if final_total = 0 then 0 else (remaining_green : ℚ) / final_total

theorem carter_second_pick_green_probability_is_24_59_percent :
  probability_green_second_pick
    35 25 10 15 20 8 14 8 15 3 10 2 1 10 ≈ 0.2459 := 
sorry

end carter_second_pick_green_probability_is_24_59_percent_l406_406288


namespace more_birds_joined_l406_406927

theorem more_birds_joined (x : ℕ) (initial_birds total_birds : ℕ) (h_initial : initial_birds = 6) (h_total : total_birds = 10) (h_eq : initial_birds + x = total_birds) : x = 4 :=
by {
  rw [h_initial, h_total] at h_eq,
  linarith,
  sorry
}

end more_birds_joined_l406_406927


namespace triangle_DEF_angles_l406_406754

-- Definition of the triangle and its properties
variables {A B C D E F : Type*}
variables {angle_A angle_B angle_C : ℝ}
variables {triangle_ABC : triangle A B C}
variables (is_right_angle : angle_B = 90)
variables (tangent_points : incircle_tangency_points triangle_ABC D E F)

-- Lean 4 statement of the problem
theorem triangle_DEF_angles 
  (h1 : angle_B = 90) 
  (h2 : ∀ {P Q R : Type*}, is_tangent P Q R → point_of_tangency P Q R ∈ {D, E, F}) :
  ∃ (angle_DFE angle_FED angle_EFD : ℝ),
  angle_DFE = 135 - angle_A / 2 ∧
  angle_FED = 135 - angle_C / 2 ∧
  angle_EFD = 90 :=
begin
  -- Proof statemets goes here
  sorry
end

end triangle_DEF_angles_l406_406754


namespace seq_equal_S₆_l406_406672

noncomputable def seq (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 3
  else 2^(n-1)

def S₆ : ℕ := seq 1 + seq 2 + seq 3 + seq 4 + seq 5 + seq 6

theorem seq_equal_S₆ : S₆ = 64 := by
  sorry

end seq_equal_S₆_l406_406672


namespace segment_intersection_l406_406501

variables {Point : Type*} [metric_space Point]

-- Definitions of right angle, distance, and angle operations in your synthetic geometry context
def right_angle_AB_CD (A B C D : Point) : Prop := sorry
def dist (x y : Point) : ℝ := sorry
def angle (A B C : Point) : ℝ := sorry

theorem segment_intersection (A B C D : Point) 
  (h1 : right_angle_AB_CD A B C D) 
  (h2 : dist A C = dist A D) : 
  dist B C = dist B D ∧ angle A C B = angle A D B :=
sorry

end segment_intersection_l406_406501


namespace portion_of_circle_illuminated_by_light_source_l406_406448

variables (P : Type) [plane P]
variables (r h H l R : ℝ) (S : P) (α β : ℝ)

-- Assuming the right circular cone and light source are defined with given conditions
def cone_base := circle r
def right_circular_cone (S : P) := { cone_base with height := h }
def light_source := { plane_distance := H, lateral_distance := l }

-- Define the angular measures of the unilluminated arc based on height H relative to h
def angular_measure_unilluminated_arc :=
  if H > h then β - α
  else if H = h then π / 2 - α
  else π - (α + β)

theorem portion_of_circle_illuminated_by_light_source
  : angular_measure_unilluminated_arc H h α β = 
    if H > h then β - α
    else if H = h then π / 2 - α
    else π - (α + β) := 
sorry

end portion_of_circle_illuminated_by_light_source_l406_406448


namespace count_primes_with_valid_remainder_l406_406375

-- Define the upper and lower bounds
def lower_bound := 30
def upper_bound := 90

-- Define the list of prime numbers in the given range
def primes_in_range : List ℕ := [31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89]

-- Define the function to check the remainder when divided by 12
def is_valid_remainder (n : ℕ) : Prop :=
  let remainder := n % 12
  remainder = 1 ∨ remainder = 5 ∨ remainder = 7 ∨ remainder = 11

-- Define the prime numbers that also have a valid prime remainder when divided by 12
def primes_with_valid_remainders : List ℕ :=
  primes_in_range.filter is_valid_remainder

-- The proof problem statement
theorem count_primes_with_valid_remainder :
  primes_with_valid_remainders.length = 14 :=
by
  sorry

end count_primes_with_valid_remainder_l406_406375


namespace leanna_cds_purchase_l406_406789

theorem leanna_cds_purchase :
  ∀ (C : ℝ), 2 * 14 + C = 37 → (14 * (1:ℝ) + 2 * C + 5 = 37) :=
by
  intros C h1
  have hC : C = 9 :=
    calc
      C = 37 - 2 * 14 : by linarith [h1]
      ... = 37 - 28 : by rw [mul_two X, maka]
      ... = 9 : by norm_num
  rw [h1, hC]
  norm_num
  done

end leanna_cds_purchase_l406_406789


namespace bottles_needed_exceed_initial_l406_406462

-- Define the initial conditions and their relationships
def initial_bottles : ℕ := 4 * 12 -- four dozen bottles

def bottles_first_break (players : ℕ) (bottles_per_player : ℕ) : ℕ :=
  players * bottles_per_player

def bottles_second_break (total_players : ℕ) (bottles_per_player : ℕ) (exhausted_players : ℕ) (extra_bottles : ℕ) : ℕ :=
  total_players * bottles_per_player + exhausted_players * extra_bottles

def bottles_third_break (remaining_players : ℕ) (bottles_per_player : ℕ) : ℕ :=
  remaining_players * bottles_per_player

-- Prove that the bottles needed exceed the initial amount by 4
theorem bottles_needed_exceed_initial : 
  bottles_first_break 11 2 + bottles_second_break 14 1 4 1 + bottles_third_break 12 1 = initial_bottles + 4 :=
by
  -- Proof will be completed here
  sorry

end bottles_needed_exceed_initial_l406_406462


namespace laser_beam_path_distance_l406_406251

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem laser_beam_path_distance : 
  let A := (4, 7)
  let B := (-4, 7)
  let C := (0, 0)
  let D := (8, 7)
  distance A B + distance B C + distance C D = 8 + 8 * real.sqrt 2 + real.sqrt 113 :=
by
  sorry

end laser_beam_path_distance_l406_406251


namespace non_intersecting_segments_exists_l406_406449

theorem non_intersecting_segments_exists 
  (n : ℕ) 
  (red_points blue_points : fin n → EuclideanSpace ℝ (fin 2)) 
  (h_no_three_collinear : ∀ (i j k : fin n), 
    i ≠ j → j ≠ k → i ≠ k → ¬ collinear ℝ {red_points i, blue_points j, red_points k}) :
  ∃ (segments : fin n → (EuclideanSpace ℝ (fin 2) × EuclideanSpace ℝ (fin 2))), 
  (∀ i, segments i.1 = red_points i ∧ segments i.2 ∈ blue_points) ∧ 
  pairwise (λ s1 s2, ¬ intersect s1 s2) (set.range (λ i, segment (segments i.1) (segments i.2))) :=
sorry

end non_intersecting_segments_exists_l406_406449


namespace area_of_border_l406_406261

theorem area_of_border
  (h_photo : Nat := 9)
  (w_photo : Nat := 12)
  (border_width : Nat := 3) :
  (let area_photo := h_photo * w_photo
    let h_frame := h_photo + 2 * border_width
    let w_frame := w_photo + 2 * border_width
    let area_frame := h_frame * w_frame
    let area_border := area_frame - area_photo
    area_border = 162) := 
  sorry

end area_of_border_l406_406261


namespace number_cards_problem_l406_406315

theorem number_cards_problem : 
  let cards := {2, 5, 8}
  ∧ let largest := 852
  ∧ let smallest := 258
  ∧ let second_largest := 825
  in (largest - smallest) * second_largest = 490050 :=
by
  sorry

end number_cards_problem_l406_406315


namespace prove_correct_option_l406_406338

variable {f : ℝ → ℝ}

def condition_1 : Prop :=
  ∀ x : ℝ, f(x+2) = -f(x)

def condition_2 : Prop :=
  ∀ x1 x2 : ℝ, 0 ≤ x1 → x1 < x2 → x2 ≤ 2 → f(x1) < f(x2)

def condition_3 : Prop :=
  ∀ x : ℝ, f(x+2) = f(2-x)

theorem prove_correct_option (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) : 
  f 4.5 < f 7 ∧ f 7 < f 6.5 :=
by
  sorry

end prove_correct_option_l406_406338


namespace radius_of_circle_l406_406738

-- Define the condition: the diameter (longest chord) of the circle is 10
def diameter (d : ℝ) := d = 10

-- Define the statement: the radius of the circle is half the diameter
theorem radius_of_circle : ∀ d : ℝ, diameter d → ∃ r : ℝ, r = d / 2 ∧ r = 5 :=
by
  intro d hd
  use d / 2
  rw hd
  exact ⟨rfl, by norm_num⟩

end radius_of_circle_l406_406738


namespace new_car_distance_l406_406939

theorem new_car_distance (old_car_distance : ℝ) (speed_increase_percent : ℝ) (new_car_distance : ℝ) : 
  old_car_distance = 120 → speed_increase_percent = 0.15 → new_car_distance = old_car_distance * (1 + speed_increase_percent) → new_car_distance = 138 :=
begin
  -- Proof goes here
  intros h_old_car_distance h_speed_increase_percent h_new_car_distance,
  rw [h_old_car_distance, h_speed_increase_percent] at h_new_car_distance,
  simp at h_new_car_distance,
  exact h_new_car_distance,
end

end new_car_distance_l406_406939


namespace g_inv_g_inv_of_12_l406_406131

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 7

-- Define the inverse function of g
def g_inv (x : ℝ) : ℝ := (x - 7) / 3

-- State the problem as a theorem
theorem g_inv_g_inv_of_12 : g_inv (g_inv 12) = -16 / 9 :=
by
  -- The proof is omitted
  sorry

end g_inv_g_inv_of_12_l406_406131


namespace trains_cross_in_9_seconds_l406_406928

noncomputable def time_to_cross (length1 length2 : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / (speed1 + speed2)

theorem trains_cross_in_9_seconds :
  time_to_cross 240 260.04 (120 * (5 / 18)) (80 * (5 / 18)) = 9 := 
by
  sorry

end trains_cross_in_9_seconds_l406_406928


namespace arithmetic_sequence_sin_cos_l406_406681

theorem arithmetic_sequence_sin_cos 
  (θ α β : ℝ) 
  (h1 : ∃ (a b c : ℝ), 2*a = b + c ∧ a = sin θ ∧ b = sin α ∧ c = cos θ)
  (h2 : sin β ^ 2 = sin θ * cos θ) : 
  2 * cos (2 * α) = cos (2 * β) :=
by
  sorry

end arithmetic_sequence_sin_cos_l406_406681


namespace solve_for_b_l406_406379

theorem solve_for_b (b : ℝ) (h : b + b / 4 = 10 / 4) : b = 2 :=
sorry

end solve_for_b_l406_406379


namespace circles_area_sum_l406_406162

noncomputable def sum_of_areas_of_circles (r s t : ℝ) : ℝ :=
  π * (r^2 + s^2 + t^2)

theorem circles_area_sum (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  sum_of_areas_of_circles r s t = 56 * π :=
begin
  sorry
end

end circles_area_sum_l406_406162


namespace sum_areas_of_circles_l406_406166

theorem sum_areas_of_circles (r s t : ℝ) 
  (h1 : r + s = 6)
  (h2 : r + t = 8)
  (h3 : s + t = 10) : 
  r^2 * Real.pi + s^2 * Real.pi + t^2 * Real.pi = 56 * Real.pi := 
by 
  sorry

end sum_areas_of_circles_l406_406166


namespace prob_not_all_same_correct_l406_406530

-- Define the number of dice and the number of sides per die
def num_dice : ℕ := 5
def sides_per_die : ℕ := 8

-- Total possible outcomes when rolling num_dice dice with sides_per_die sides each
def total_outcomes : ℕ := sides_per_die ^ num_dice

-- Outcomes where all dice show the same number
def same_number_outcomes : ℕ := sides_per_die

-- Probability that all five dice show the same number
def prob_same : ℚ := same_number_outcomes / total_outcomes

-- Probability that not all five dice show the same number
def prob_not_same : ℚ := 1 - prob_same

-- The expected probability that not all dice show the same number
def expected_prob_not_same : ℚ := 4095 / 4096

-- The main theorem to prove
theorem prob_not_all_same_correct : prob_not_same = expected_prob_not_same := by
  sorry  -- Proof will be filled in by the user

end prob_not_all_same_correct_l406_406530


namespace focus_parabola_l406_406851

theorem focus_parabola (x : ℝ) (y : ℝ): (y = 8 * x^2) → (0, 1 / 32) = (0, 1 / 32) :=
by
  intro h
  sorry

end focus_parabola_l406_406851


namespace quadratic_range_l406_406370

open Real

def quadratic (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 5

theorem quadratic_range :
  ∀ x, -2 ≤ x ∧ x ≤ 2 → -8 ≤ quadratic x ∧ quadratic x ≤ 19 :=
by
  intro x h
  sorry

end quadratic_range_l406_406370


namespace determine_m_value_l406_406700

theorem determine_m_value
  (m : ℝ)
  (h : ∀ x : ℝ, -7 < x ∧ x < -1 ↔ mx^2 + 8 * m * x + 28 < 0) :
  m = 4 := by
  sorry

end determine_m_value_l406_406700


namespace at_most_one_x_intersection_l406_406360

-- Assume f is a function from ℝ to ℝ that is strictly increasing.
variables {f : ℝ → ℝ}

-- The main theorem to prove
theorem at_most_one_x_intersection (h : ∀ a b : ℝ, a < b → f(a) < f(b)) 
  (h1 : ∃ x1 : ℝ, f x1 = 0) (h2 : ∃ x2 : ℝ, f x2 = 0) : 
  (∃ x1 x2 : ℝ, f x1 = 0 ∧ f x2 = 0 → x1 = x2) :=
sorry

end at_most_one_x_intersection_l406_406360


namespace sum_of_first_n_terms_seq_l406_406575

variable {a_n : ℕ → ℝ}

def is_geometric_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ (a₁ q : ℝ), (∀ n : ℕ, a_n = a₁ * q^n)

def is_increasing (a_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a_n < a_n.succ

theorem sum_of_first_n_terms_seq (a_n : ℕ → ℝ)
  (h_geom : is_geometric_sequence a_n)
  (h_increase : is_increasing a_n)
  (h1 : a_n 1 + a_n 4 = 9)
  (h2 : a_n 2 * a_n 3 = 8) :
  ∑ i in Finset.range n, a_n i = 2^n - 1 := by
sorry

end sum_of_first_n_terms_seq_l406_406575


namespace letters_arrangement_count_l406_406272

theorem letters_arrangement_count : 
  let letters := ['A', 'B', 'C', 'D', 'E']
  let valid_positions := (permutations letters).filter (λ l, l.head ≠ 'A' ∧ l.head ≠ 'E' ∧ l.last ≠ 'A' ∧ l.last ≠ 'E')
  valid_positions.length = 36 := 
  sorry

end letters_arrangement_count_l406_406272


namespace sum_of_floor_sqrt_1_to_25_l406_406994

theorem sum_of_floor_sqrt_1_to_25 : 
  ( ∑ i in Finset.range 26, ∥Real.sqrt i∥ ) = 75 :=
by
  sorry

end sum_of_floor_sqrt_1_to_25_l406_406994


namespace john_spending_l406_406060

variable (initial_cost : ℕ) (sale_price : ℕ) (new_card_cost : ℕ)

theorem john_spending (h1 : initial_cost = 1200) (h2 : sale_price = 300) (h3 : new_card_cost = 500) :
  initial_cost - sale_price + new_card_cost = 1400 := 
by
  sorry

end john_spending_l406_406060


namespace brokerage_calculation_l406_406554

theorem brokerage_calculation (cash_realized : ℝ) (brokerage_rate : ℝ) 
  (h1 : cash_realized = 104.25) 
  (h2 : brokerage_rate = 0.0025) : 
  let brokerage_fee := cash_realized * brokerage_rate in
  (Float.round (brokerage_fee * 100) / 100) = 0.26 :=
by
  -- the proof would go here
  sorry

end brokerage_calculation_l406_406554


namespace trig_expression_equality_l406_406976

theorem trig_expression_equality :
  (1 - 1 / (Real.cos (Real.pi / 6))) *
  (1 + 1 / (Real.sin (Real.pi / 3))) *
  (1 - 1 / (Real.sin (Real.pi / 6))) *
  (1 + 1 / (Real.cos (Real.pi / 3))) = 1 :=
by
  have hcos30 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2 := sorry
  have hsin60 : Real.sin (Real.pi / 3) = (Real.sqrt 3) / 2 := sorry
  have hsin30 : Real.sin (Real.pi / 6) = 1 / 2 := sorry
  have hcos60 : Real.cos (Real.pi / 3) = 1 / 2 := sorry
  sorry

end trig_expression_equality_l406_406976


namespace find_m_find_tangent_lines_l406_406228

section problem1
variables (a m : ℝ)
def f (x : ℝ) : ℝ := (1 / 2) * x^2 - (a + m) * x + a * real.log x
def f' (x : ℝ) : ℝ := deriv f x

theorem find_m (h : f' 1 = 0) : m = 1 :=
sorry
end problem1

section problem2
def f2 (x : ℝ) : ℝ := x^3 + 3*x^2 - 9*x - 3
def f2' (x : ℝ) : ℝ := deriv f2 x
def line_perpendicular_slope := -9
def tangent_lines (x0 : ℝ) (y : ℝ) : Prop :=
(9 * x0 + y + 3 = 0) ∨ (9 * x0 + y - 18 = 0)

theorem find_tangent_lines (x0 : ℝ) (h : f2' x0 = line_perpendicular_slope) : tangent_lines x0 (f2 x0) :=
sorry
end problem2

end find_m_find_tangent_lines_l406_406228


namespace base_addition_is_10_l406_406628

-- The problem states that adding two numbers in a particular base results in a third number in the same base.
def valid_base_10_addition (n m k b : ℕ) : Prop :=
  let n_b := n / b^2 * b^2 + (n / b % b) * b + n % b
  let m_b := m / b^2 * b^2 + (m / b % b) * b + m % b
  let k_b := k / b^2 * b^2 + (k / b % b) * b + k % b
  n_b + m_b = k_b

theorem base_addition_is_10 : valid_base_10_addition 172 156 340 10 :=
  sorry

end base_addition_is_10_l406_406628


namespace algebraic_expression_value_l406_406334

variable x : Real

theorem algebraic_expression_value (h : x = 2 - Real.sqrt 3) :
  (7 + 4 * Real.sqrt 3) * x^2 - (2 + Real.sqrt 3) * x + Real.sqrt 3 = 2 + Real.sqrt 3 := 
 by
  sorry

end algebraic_expression_value_l406_406334


namespace cone_radius_l406_406885

-- Step 1: Define the radii and distances
def r : ℝ := Real.sqrt 24
def height_of_cone (R : ℝ) : ℝ := R

-- Step 2: State the radius of the base of the cone given all conditions
def radius_base_cone (R : ℝ) : Prop :=
  R = 7 + 4 * Real.sqrt 3 + 2 * Real.sqrt 6

-- Step 3: The proof statement
theorem cone_radius (R : ℝ) (h_eq : height_of_cone R = R) :
  radius_base_cone R := by
  sorry

end cone_radius_l406_406885


namespace smallest_positive_integer_exists_l406_406324

theorem smallest_positive_integer_exists
  (m : ℕ)
  (y : Fin m → ℝ)
  (hm_pos : 0 < m)
  (h_sum_y : (∑ i, y i) = 500)
  (h_sum_y_fourth : (∑ i, (y i)^4) = 62500) :
  m = 10 :=
sorry

end smallest_positive_integer_exists_l406_406324


namespace circles_area_sum_l406_406159

noncomputable def sum_of_areas_of_circles (r s t : ℝ) : ℝ :=
  π * (r^2 + s^2 + t^2)

theorem circles_area_sum (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  sum_of_areas_of_circles r s t = 56 * π :=
begin
  sorry
end

end circles_area_sum_l406_406159


namespace frog_arrangements_l406_406184

-- Define the total number of frogs and their specific colors
def total_frogs : Nat := 8
def green_frogs : Nat := 2
def red_frogs : Nat := 3
def blue_frogs : Nat := 3

-- Define the constraints
def green_not_next_to_red (arrangement : Array Nat) : Prop :=
  ∀ i, i < arrangement.size - 1 →
    (arrangement[i] == 1 ∧ arrangement[i + 1] ≠ 2) ∧ (arrangement[i] == 2 ∧ arrangement[i + 1] ≠ 1)

def blue_not_next_to_blue (arrangement : Array Nat) : Prop :=
  ∀ i, i < arrangement.size - 1 →
    arrangement[i] ≠ 3 ∨ arrangement[i + 1] ≠ 3

-- Main statement to be proved
theorem frog_arrangements : 
  ∃ arrangements : Finset (Array Nat), 
    (arrangements.card = 96) ∧ 
    ∀ a ∈ arrangements, a.size = total_frogs ∧
      green_not_next_to_red a ∧ blue_not_next_to_blue a := 
sorry

end frog_arrangements_l406_406184


namespace total_students_l406_406901

theorem total_students (students_between : ℕ) (right_of_hoseok : ℕ) (left_of_yoongi : ℕ) :
  students_between = 5 → right_of_hoseok = 9 → left_of_yoongi = 6 → students_between + right_of_hoseok + left_of_yoongi + 2 = 22 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end total_students_l406_406901


namespace solutions_to_quadratic_l406_406141

noncomputable def a : ℝ := (6 + Real.sqrt 92) / 2
noncomputable def b : ℝ := (6 - Real.sqrt 92) / 2

theorem solutions_to_quadratic :
  a ≥ b ∧ ((∀ x : ℝ, x^2 - 6 * x + 11 = 25 → x = a ∨ x = b) → 3 * a + 2 * b = 15 + Real.sqrt 92 / 2) := by
  sorry

end solutions_to_quadratic_l406_406141


namespace vector_equality_l406_406707

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

theorem vector_equality {a x : V} (h : 2 • x - 3 • (x - 2 • a) = 0) : x = 6 • a :=
by
  sorry

end vector_equality_l406_406707


namespace smallest_divisor_l406_406537

theorem smallest_divisor (N D : ℕ) (hN : N = D * 7) (hD : D > 0) (hsq : (N / D) = 7) :
  D = 7 :=
by 
  sorry

end smallest_divisor_l406_406537


namespace amy_math_problems_l406_406660

-- Definitions based on conditions
def num_spelling_problems : ℕ := 6
def problems_per_hour : ℕ := 4
def total_hours : ℕ := 6

-- Calculation based on the given conditions
def total_problems : ℕ := problems_per_hour * total_hours
def num_math_problems (total_problems : ℕ) : ℕ := total_problems - num_spelling_problems

-- Statement that we need to prove
theorem amy_math_problems : num_math_problems total_problems = 18 :=
by
  unfold num_math_problems
  unfold total_problems
  norm_num

end amy_math_problems_l406_406660


namespace number_of_divisors_l406_406499

theorem number_of_divisors (n : ℕ) : 
  (∃ k : ℕ, 111 = k * n + 6) → (finset.filter (> 6) (finset.divisors 105)).card = 5 :=
by
  intro h
  sorry

end number_of_divisors_l406_406499


namespace zoo_tickets_total_cost_l406_406576

-- Define the given conditions
def num_children := 6
def num_adults := 10
def cost_child_ticket := 10
def cost_adult_ticket := 16

-- Calculate the expected total cost
def total_cost := 220

-- State the theorem
theorem zoo_tickets_total_cost :
  num_children * cost_child_ticket + num_adults * cost_adult_ticket = total_cost :=
by
  sorry

end zoo_tickets_total_cost_l406_406576


namespace paper_cups_count_l406_406276

variables (P C : ℝ) (x : ℕ)

theorem paper_cups_count :
  100 * P + x * C = 7.50 ∧ 20 * P + 40 * C = 1.50 → x = 200 :=
sorry

end paper_cups_count_l406_406276


namespace problem_statement_l406_406689

variable {f : ℝ → ℝ} 

theorem problem_statement (hf : ∀ x y, x < y → f x < f y) (a b : ℝ) (h : a + b > 0) : 
  f(a) + f(b) > f(-a) + f(-b) :=
sorry

end problem_statement_l406_406689


namespace hyperbola_foci_distance_l406_406620

theorem hyperbola_foci_distance : 
  ∀ (x y : ℝ), x * y = 4 → (∃ d : ℝ, d = 4 * real.sqrt 2) :=
begin
  intros x y h,
  use 4 * real.sqrt 2,
  sorry
end

end hyperbola_foci_distance_l406_406620


namespace john_large_bottles_count_l406_406062

theorem john_large_bottles_count :
  ∃ L : ℕ, 
    (1.73 = (1.89 * L + 1.42 * 720) / (L + 720)) ∧ 
    L = 1395 :=
begin
  use 1395,
  split,
  { calc 1.73  = (1.89 * 1395 + 1.42 * 720) / (1395 + 720) : by sorry },
  { refl }
end

end john_large_bottles_count_l406_406062


namespace bone_meal_percentage_growth_l406_406414

-- Definitions for the problem conditions
def control_height : ℝ := 36
def cow_manure_height : ℝ := 90
def bone_meal_to_cow_manure_ratio : ℝ := 0.5 -- since cow manure plant is 200% the height of bone meal plant

noncomputable def bone_meal_height : ℝ := cow_manure_height * bone_meal_to_cow_manure_ratio

-- The main theorem to prove
theorem bone_meal_percentage_growth : 
  ( (bone_meal_height - control_height) / control_height ) * 100 = 25 := 
by
  sorry

end bone_meal_percentage_growth_l406_406414


namespace second_car_rental_rate_l406_406126

theorem second_car_rental_rate :
  ∃ x : ℝ, (x + 0.16 * 48 = 17.99 + 0.18 * 48) ∧ x = 18.95 :=
by
  use 18.95
  split
  sorry

end second_car_rental_rate_l406_406126


namespace solve_eq_for_log_base_l406_406836

theorem solve_eq_for_log_base :
  (∃ x, (9 : ℝ) ^ (x + 9) = (12 : ℝ) ^ x) → (x = real.logb (4/3) (9^9)) :=
begin
  sorry
end

end solve_eq_for_log_base_l406_406836


namespace pyramid_properties_l406_406947

def pyramid_base_side : ℝ := 20
def pyramid_slant_height : ℝ := 27
def pyramid_height := pyramid_base_side + 3

def pyramid_surface_area : ℝ := 
  pyramid_base_side^2 + 
  pyramid_base_side * sqrt (4 * pyramid_slant_height^2 - pyramid_base_side^2)

def pyramid_volume : ℝ :=
  (pyramid_base_side^2 / 3) * pyramid_height

theorem pyramid_properties :
  (pyramid_surface_area ≈ 1403.19) ∧
  (pyramid_volume ≈ 3066.67) :=
by
  -- Here we would prove the theorem.
  sorry

end pyramid_properties_l406_406947


namespace percent_greater_than_l406_406147

theorem percent_greater_than (M N : ℝ) (hN : N ≠ 0) : (M - N) / N * 100 = 100 * (M - N) / N :=
by sorry

end percent_greater_than_l406_406147


namespace pattern_repeats_per_necklace_l406_406476

-- Definitions for the conditions
def green_beads : ℕ := 3
def purple_beads : ℕ := 5
def red_beads : ℕ := 2 * green_beads
def beads_per_pattern : ℕ := green_beads + purple_beads + red_beads
def repeats_per_bracelet : ℕ := 3
def beads_per_bracelet : ℕ := repeats_per_bracelet * beads_per_pattern
def total_beads_needed : ℕ := 742

-- Translate the question into Lean 4 statement with the assumptions about conditions
theorem pattern_repeats_per_necklace (N : ℕ) 
    (h : beads_per_bracelet + 10 * N * beads_per_pattern = total_beads_needed) :
    N = 5 :=
sory

end pattern_repeats_per_necklace_l406_406476


namespace arccos_gt_arctan_l406_406647

open Real

theorem arccos_gt_arctan (x : ℝ) (hx : x ∈ set.Ico (-1 : ℝ) 1) : arccos x > arctan x :=
sorry

end arccos_gt_arctan_l406_406647


namespace num_newborn_members_l406_406030

noncomputable def animal_survival_problem : ℕ :=
  let p := 1 / 10 in
  let survive_prob := (1 - p) ^ 3 in
  let expected_survivors := 364.5 in
  let N := expected_survivors / survive_prob in
  N

theorem num_newborn_members (N : ℕ) (p : ℝ) (expected_survivors : ℝ) :
  p = 1 / 10 →
  expected_survivors = 364.5 →
  let survive_prob := (1 - p) ^ 3 in
  N = expected_survivors / survive_prob →
  N ≈ 500 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  calc
    N = 364.5 / (9 / 10) ^ 3 : by sorry
    ... = 500 : by sorry

#eval animal_survival_problem  -- This should evaluate to the approximate number of newborns

end num_newborn_members_l406_406030


namespace trigonometric_identity_solution_l406_406838

theorem trigonometric_identity_solution {k : ℤ} :
  (∃ n : ℕ, k = n) →
  (sin x + sin (2 * x) + sin (3 * x) = 1 + cos x + cos (2 * x)) →
  (x = 90 + k * 180 ∨ x = 120 + k * 360 ∨ x = 240 + k * 360 ∨
   x = 30 + k * 360 ∨ x = 150 + k * 360) :=
begin
  sorry -- proof to be provided
end

end trigonometric_identity_solution_l406_406838


namespace sum_areas_of_circles_l406_406163

theorem sum_areas_of_circles (r s t : ℝ) 
  (h1 : r + s = 6)
  (h2 : r + t = 8)
  (h3 : s + t = 10) : 
  r^2 * Real.pi + s^2 * Real.pi + t^2 * Real.pi = 56 * Real.pi := 
by 
  sorry

end sum_areas_of_circles_l406_406163


namespace fraction_of_girls_participated_l406_406232

theorem fraction_of_girls_participated
  (total_students : ℕ)
  (participating_students : ℕ)
  (participating_girls : ℕ)
  (total_boys total_girls : ℕ)
  (fraction_of_boys_participating : ℚ)
  (total_students_eq : total_students = 800)
  (participating_students_eq : participating_students = 550)
  (participating_girls_eq : participating_girls = 150)
  (fraction_of_boys_participating_eq : fraction_of_boys_participating = 2/3)
  (B_plus_G_eq : total_boys + total_girls = total_students)
  (boys_participating : ℚ)
  (boys_participating_eq : boys_participating = (fraction_of_boys_participating * total_boys))
  (boys_participating_value : total_boys = 600)
  (girls_participating : ℚ)
  (fraction_calculation_eq : girls_participating = 150 / 200)
  :
  girls_participating = 3/4 := by
  sorry -- Proof goes here

end fraction_of_girls_participated_l406_406232


namespace double_line_chart_comparison_l406_406574

/-- 
A double line chart is a type of line chart that is used to compare data trends.
-/
def double_line_chart_convenient (A B : Type) (data_A : A → ℝ) (data_B : B → ℝ) : Prop :=
  ∀ x, (data_A x ≤ data_B x) ∨ (data_A x ≥ data_B x)

/--
Prove that a double line chart is convenient for comparing the changes in quantities between two sets of data given the characteristics of a double line chart.
-/
theorem double_line_chart_comparison (A B : Type) (data_A : A → ℝ) (data_B : B → ℝ)
  (h : double_line_chart_convenient A B data_A data_B) :
  ∀ x, (data_A x ≤ data_B x) ∨ (data_A x ≥ data_B x) :=
by
  sorry

end double_line_chart_comparison_l406_406574


namespace guides_tourists_grouping_l406_406519

theorem guides_tourists_grouping (tourists : ℕ) (guides : ℕ) (h_t : tourists = 6) (h_g : guides = 2) :
  (∑ k in finset.Ico 1 tourists, nat.choose tourists k) = 62 :=
by
  rw [h_t, h_g]
  -- usual steps to prove the sum, currently omitted
  sorry

end guides_tourists_grouping_l406_406519


namespace prove_p_q_s_l406_406381

noncomputable def polynomial_equiv_divisibility (p q s : ℚ) : Prop :=
  (x^4 + 3 * x^3 + 5 * p * x^2 + 2 * q * x + s) ∣ (x^3 + 2 * x^2 + 4 * x + 1)

theorem prove_p_q_s (p q s : ℚ) (h : polynomial_equiv_divisibility p q s) : (p + q) * s = -1.3 := by
  sorry

end prove_p_q_s_l406_406381


namespace total_swordfish_caught_l406_406466

theorem total_swordfish_caught (fishing_trips : ℕ) (shelly_each_trip : ℕ) (sam_each_trip : ℕ) : 
  shelly_each_trip = 3 → 
  sam_each_trip = 2 → 
  fishing_trips = 5 → 
  (shelly_each_trip + sam_each_trip) * fishing_trips = 25 :=
by
  sorry

end total_swordfish_caught_l406_406466


namespace find_angle_BKN_l406_406342

noncomputable def angle_BKN (AB AD : ℝ) (angle_BAD : ℝ) (K_ratio : ℝ) (KL_eq_AM : ℝ → ℝ) : ℝ :=
  let AK := (AK_ratio / (K_ratio + 1)) * AB
  let KB := AB - AK
  -- other intermediate steps to determine coordinates of N and calc angle BKN can be scripted within the proof (sorry omitted here)
  75

theorem find_angle_BKN (AB AD : ℝ) (angle_BAD : ℝ) (K_ratio : ℝ) (KL_eq_AM : ℝ → ℝ) : 
  AB = 5 → AD = 2 * Math.sqrt 3 + 2 → angle_BAD = 30 → K_ratio = 4 / 1 → 
  KL_eq_AM = λ x, (4 / 5) * (2 * Math.sqrt 3 + 2) →
  angle_BKN AB AD angle_BAD K_ratio KL_eq_AM = 75 :=
begin
  sorry
end

end find_angle_BKN_l406_406342


namespace isosceles_triangle_perimeter_l406_406350

theorem isosceles_triangle_perimeter (x y : ℝ) (h : |x - 4| + (y - 8)^2 = 0) :
  4 + 8 + 8 = 20 :=
by
  sorry

end isosceles_triangle_perimeter_l406_406350


namespace Darcy_walking_speed_proof_l406_406622

noncomputable def Darcy_walking_speed : ℝ := 3

theorem Darcy_walking_speed_proof
  (distance : ℝ)
  (train_speed : ℝ)
  (additional_time_minutes : ℝ)
  (time_difference_minutes : ℝ)
  (v : ℝ) :
  distance = 1.5 →
  train_speed = 20 →
  additional_time_minutes = 23.5 →
  time_difference_minutes = 2 →
  (distance / v) = (distance / train_speed) + (additional_time_minutes / 60) + (time_difference_minutes / 60)
  → v = Darcy_walking_speed :=
by {
  intros,
  sorry
}

end Darcy_walking_speed_proof_l406_406622


namespace problem_1_problem_2_l406_406722

theorem problem_1 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) (x : ℝ)
  (h : (2 * x - 1) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) :
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 1 :=
sorry

theorem problem_2 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) (x : ℝ)
  (h : (2 * x - 1) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) :
  a_0 + a_2 + a_4 + a_6 = 365 :=
sorry

end problem_1_problem_2_l406_406722


namespace circles_area_sum_l406_406158

noncomputable def sum_of_areas_of_circles (r s t : ℝ) : ℝ :=
  π * (r^2 + s^2 + t^2)

theorem circles_area_sum (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  sum_of_areas_of_circles r s t = 56 * π :=
begin
  sorry
end

end circles_area_sum_l406_406158


namespace talias_fathers_age_l406_406761

-- Definitions based on the conditions
variable (T M F : ℕ)

-- The conditions
axiom h1 : T + 7 = 20
axiom h2 : M = 3 * T
axiom h3 : F + 3 = M

-- Goal: Prove that Talia's father (F) is currently 36 years old
theorem talias_fathers_age : F = 36 :=
by
  sorry

end talias_fathers_age_l406_406761


namespace largest_sin_angle_PQS_l406_406187

-- Definitions of the conditions in the problem
def triangle_PQR (P Q R : Type) := ∃ P Q R : Type, -- Type parameters for vertices
  ∃ (angle_R : ℝ) (QR : ℝ),                -- Angle at R and length QR
  angle_R = real.pi / 4 ∧ QR = 6          -- ∠R = 45°, QR = 6 

def point_S (S : Type) (Q R : Type) := ∃ S Q R : Type, -- Type parameters for points
  ∃ (QR : ℝ) (S_is_midpoint : bool),      -- Length QR and midpoint condition
  QR = 6 ∧ S_is_midpoint = true           -- QR = 6, S is midpoint of QR

noncomputable def sin_angle_PQS (sin_angle_value : ℝ) := ∃ (angle_PQS : ℝ), 
  real.sin angle_PQS = sin_angle_value

-- Rewriting the proof problem
theorem largest_sin_angle_PQS : ∀ (P Q R S : Type), 
  triangle_PQR P Q R → 
  point_S S Q R → 
  sin_angle_PQS (real.sqrt 2) :=
by
  intros,
  exact sorry

end largest_sin_angle_PQS_l406_406187


namespace find_x_l406_406014

theorem find_x 
  (x y : ℤ) 
  (h1 : 2 * x - y = 5) 
  (h2 : x + 2 * y = 5) : 
  x = 3 := 
sorry

end find_x_l406_406014


namespace problem1_problem2_l406_406914

-- Proof problem for the first condition
theorem problem1 {p : ℕ} (hp : Nat.Prime p) 
  (h : ∃ n : ℕ, (7^(p-1) - 1) = p * n^2) : p = 3 :=
sorry

-- Proof problem for the second condition
theorem problem2 {p : ℕ} (hp : Nat.Prime p)
  (h : ∃ n : ℕ, (11^(p-1) - 1) = p * n^2) : false :=
sorry

end problem1_problem2_l406_406914


namespace sum_of_areas_of_circles_l406_406182

theorem sum_of_areas_of_circles (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  ∃ (π : ℝ), π * (r^2 + s^2 + t^2) = 56 * π :=
by {
  sorry
}

end sum_of_areas_of_circles_l406_406182


namespace tourists_speed_l406_406250

theorem tourists_speed
  (arrival_time_difference : ℝ)
  (bus_early_arrival_time : ℝ)
  (bus_speed : ℝ)
  (expected_speed : ℝ) :
  arrival_time_difference = 1.75 →
  bus_early_arrival_time = 0.25 →
  bus_speed = 60 →
  expected_speed = 5 →
  (7.5 / (arrival_time_difference - bus_early_arrival_time) = expected_speed) :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end tourists_speed_l406_406250


namespace degree_polynomial_horizontal_asymptote_l406_406627

theorem degree_polynomial_horizontal_asymptote (p : Polynomial ℝ) :
  (3 * (X : Polynomial ℝ)^6 - 2 * X^3 + X - 5 ≠ 0) →
  (degree (p) < 6) ↔ has_horizontal_asymptote_at_y_0 (p / (3 * X^6 - 2 * X^3 + X - 5)) :=
sorry

end degree_polynomial_horizontal_asymptote_l406_406627


namespace area_of_triangle_XPQ_l406_406410

noncomputable def area_triangle_XPQ (XY YZ XZ XP XQ : ℝ) (hXY : XY = 12) (hYZ : YZ = 13) (hXZ : XZ = 15) (hXP : XP = 5) (hXQ : XQ = 9) : ℝ :=
  let s := (XY + YZ + XZ) / 2
  let area_XYZ := Real.sqrt (s * (s - XY) * (s - YZ) * (s - XZ))
  let cosX := (XY^2 + YZ^2 - XZ^2) / (2 * XY * YZ)
  let sinX := Real.sqrt (1 - cosX^2)
  (1 / 2) * XP * XQ * sinX

theorem area_of_triangle_XPQ :
  area_triangle_XPQ 12 13 15 5 9 (by rfl) (by rfl) (by rfl) (by rfl) (by rfl) = 45 * Real.sqrt 1400 / 78 :=
by
  sorry

end area_of_triangle_XPQ_l406_406410


namespace roads_probability_l406_406555

theorem roads_probability :
  ∀ (P_A P_B : ℝ), P_A = 2 / 3 → P_B = 3 / 4 →
    (1 - (1 - P_A) * (1 - P_B) = 11 / 12) :=
by
  intros P_A P_B hA hB
  rw [hA, hB]
  -- rest will be the proof, which we skip with sorry
  sorry

end roads_probability_l406_406555


namespace rational_terms_in_expansion_a_rational_terms_in_expansion_b_l406_406721

-- Define the conditions for the first problem
def is_rational_term_a (k : ℕ) : Prop :=
  (100 - k) % 2 = 0 ∧ k % 4 = 0

-- Define the conditions for the second problem
def is_rational_term_b (k : ℕ) : Prop :=
  (300 - k) % 2 = 0 ∧ k % 3 = 0 ∧ (k / 3) % 2 = 0

-- Problem a: Prove there are 26 rational terms in the expansion of (sqrt(2) + sqrt(3)^1/4)^100
theorem rational_terms_in_expansion_a :
  (Finset.filter is_rational_term_a (Finset.range 101)).card = 26 :=
by
  sorry

-- Problem b: Prove there are 51 rational terms in the expansion of (sqrt(2) + sqrt(3)^1/3)^300
theorem rational_terms_in_expansion_b :
  (Finset.filter is_rational_term_b (Finset.range 301)).card = 51 :=
by
  sorry

end rational_terms_in_expansion_a_rational_terms_in_expansion_b_l406_406721


namespace area_of_10th_square_l406_406631

theorem area_of_10th_square (a₁ : ℝ) (r : ℝ) (n : ℕ) (h₀ : a₁ = 2) (h₁ : r = real.sqrt 2) (h₂ : n = 10) :
  let aₙ := a₁ * r^(n - 1)
  let area := aₙ ^ 2
  area = 2048 := 
by
  sorry

end area_of_10th_square_l406_406631


namespace sphere_volume_correct_l406_406372

noncomputable def sphere_center_to_plane_distance := 1
def edge_length := 2 * Real.sqrt 3
def sphere_radius := Real.sqrt 5
def sphere_volume := (4 / 3) * Real.pi * sphere_radius ^ 3
def expected_sphere_volume := 20 * Real.sqrt 5 * Real.pi / 3

-- Statement to prove that the computed volume is as expected
theorem sphere_volume_correct 
      (h1 : AB = edge_length) 
      (h2 : AC = edge_length) 
      (h3 : BC = edge_length) 
      (h4 : distance_to_plane = sphere_center_to_plane_distance) : 
  sphere_volume = expected_sphere_volume := 
by
  sorry

end sphere_volume_correct_l406_406372


namespace find_inverse_zero_l406_406093

def f (x : ℝ) : ℝ := 2 * log (2 * x - 1)

theorem find_inverse_zero :
  (∃ x : ℝ, f(x) = 0) ∧ 
  (∀ x : ℝ, f(x) = 0 → x = 1) :=
sorry

end find_inverse_zero_l406_406093


namespace minimal_factors_erased_l406_406194

def minimal_factors_erased_has_no_real_solutions :
    Nat :=
  2016

theorem minimal_factors_erased {x : ℝ} :
  ∀ (lhs rhs : ℝ → ℝ),
  (lhs x = rhs x) →
  (∃ m n : Finset ℕ,
    m.card = 2016 ∧
    n.card = 2016 ∧
    lhs = (λ x, Finset.prod ((Finset.range 2016) \ m) (λ i, x - (i+1))) ∧
    rhs = (λ x, Finset.prod ((Finset.range 2016) \ n) (λ i, x - (i+1))) ∧
    ∀ x : ℝ, (Finset.prod ((Finset.range 2016) \ m) (λ i, x - (i+1))) ≠
             (Finset.prod ((Finset.range 2016) \ n) (λ i, x - (i+1)))) :=
sorry

end minimal_factors_erased_l406_406194


namespace negation_of_exists_l406_406862

theorem negation_of_exists (x : ℕ) : (¬ ∃ x : ℕ, x^2 ≤ x) := 
by 
  sorry

end negation_of_exists_l406_406862


namespace find_integer_l406_406203

theorem find_integer:
  ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ (m : ℕ), n = m * Nat.lcm [2, 3, 4, 5, 6] + 1) :=
by
  sorry

end find_integer_l406_406203


namespace crate_stack_probability_l406_406642

theorem crate_stack_probability : 
  ∃ (a b c : ℕ), 3 * a + 4 * b + 5 * c = 50 ∧ a + b + c = 15 ∧ 
    ( ( (nat.factorial 15) / (nat.factorial a * nat.factorial b * nat.factorial c)) /
      (3 ^ 15) = 660283 / 14348907)
  :=
by {
  use [10, 5, 0],
  split,
  {
    -- (3 * 10 + 4 * 5 + 5 * 0 = 50)
    norm_num,
  },
  split,
  {
    -- (10 + 5 + 0 = 15)
    norm_num,
  },
  {
    -- Calculating the probability
    calc
      ((nat.factorial 15) / (nat.factorial 10 * nat.factorial 5 * nat.factorial 0)) / (3 ^ 15)
        = 660283 / 14348907 : sorry
  }
}

end crate_stack_probability_l406_406642


namespace model_tower_height_l406_406238

theorem model_tower_height (h_real : ℝ) (vol_real : ℝ) (vol_model : ℝ) 
  (h_real_eq : h_real = 60) (vol_real_eq : vol_real = 150000) (vol_model_eq : vol_model = 0.15) :
  (h_real * (vol_model / vol_real)^(1/3) = 0.6) :=
by
  sorry

end model_tower_height_l406_406238


namespace solve_abc_l406_406472

theorem solve_abc (a b c : ℝ) (h1 : a ≤ b ∧ b ≤ c) (h2 : a + b + c = -1) (h3 : a * b + b * c + a * c = -4) (h4 : a * b * c = -2) :
  a = -1 - Real.sqrt 3 ∧ b = -1 + Real.sqrt 3 ∧ c = 1 :=
by
  -- Proof goes here
  sorry

end solve_abc_l406_406472


namespace area_of_inscribed_circle_l406_406570

-- Defining the problem
variables (d : ℝ) (s : ℝ) (r : ℝ) (A : ℝ)

-- Given conditions
def conditions : Prop :=
  (d = 10) ∧ (d = s * Real.sqrt 2) ∧ (r = s / 2)

-- Target statement
theorem area_of_inscribed_circle (h : conditions d s r A) : A = 12.5 * Real.pi :=
by {
  sorry
}

end area_of_inscribed_circle_l406_406570


namespace find_g_of_2_l406_406856

open Real

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_of_2
  (H: ∀ x : ℝ, g (2 ^ x) + x * g (2 ^ (-x)) + x = 1) : g 2 = -1 :=
by
  sorry

end find_g_of_2_l406_406856


namespace how_long_it_lasts_l406_406096

-- Define a structure to hold the conditions
structure MoneySpending where
  mowing_income : ℕ
  weeding_income : ℕ
  weekly_expense : ℕ

-- Example conditions given in the problem
def lukesEarnings : MoneySpending :=
{ mowing_income := 9,
  weeding_income := 18,
  weekly_expense := 3 }

-- Main theorem proving the number of weeks he can sustain his spending
theorem how_long_it_lasts (data : MoneySpending) : 
  (data.mowing_income + data.weeding_income) / data.weekly_expense = 9 := by
  sorry

end how_long_it_lasts_l406_406096


namespace count_center_int_points_in_triangle_l406_406764

theorem count_center_int_points_in_triangle :
  let A := (0, 0)
  let B := (200, 100)
  let C := (30, 330)
  let area := 31500
  ∃ (I_c : ℕ), 
    let B_c := 40
    I_c = 31480 :=
begin
  sorry
end

end count_center_int_points_in_triangle_l406_406764


namespace residual_analysis_is_effectiveness_judging_l406_406207

-- Define the residuals
def residuals (n : ℕ) : Type := vector ℝ n

-- Define the condition for judging the effectiveness of model fitting using residuals
def uses_residuals_to_judge_effectiveness (n : ℕ) (e : residuals n) : Prop := 
  sorry -- This would be a placeholder for the actual definition

-- The statement we want to prove
theorem residual_analysis_is_effectiveness_judging (n : ℕ) (e : residuals n) :
  uses_residuals_to_judge_effectiveness n e → ∃ analysis_type : string, analysis_type = "residual" :=
by
  sorry

end residual_analysis_is_effectiveness_judging_l406_406207


namespace nth_number_in_sequence_l406_406395

theorem nth_number_in_sequence (n : ℕ) (hn : n = 40) : 
  let sequence := λ (k : ℕ), (2 * (2 + (k - 1)), ⌊2 + ((2 * k - 1) / 2)⌋) in
  nth_number sequence n = 18 :=
begin
  -- All necessary conditions and definitions are declared above
  sorry,
end

end nth_number_in_sequence_l406_406395


namespace pairs_count_l406_406494

def log_base (b x : ℝ) : ℝ := log x / log b

theorem pairs_count : 
  (3 : ℝ) ^ (867 : ℝ) > (4 : ℝ) ^ (1352 : ℝ) ∧ (3 : ℝ) ^ (867 : ℝ) < (4 : ℝ) ^ (1353 : ℝ) → 
  ∃ n m : ℕ, 1 ≤ m ∧ m ≤ 1351 ∧ nat.floor(log_base (4 : ℝ) (3 : ℝ ^ n)) < m ∧ m < nat.floor(log_base (4 : ℝ) (3 : ℝ ^ (n + 1))) ∧ 
  count_pairs = 687 := sorry
  
noncomputable def count_pairs : ℕ := 
nat.card {p : ℕ × ℕ // 1 ≤ p.1 ∧ p.1 ≤ 1351 ∧ (3 : ℝ) ^ (p.2 : ℝ) < (4 : ℝ) ^ (p.1 : ℝ) ∧ 
  ((4 : ℝ) ^ (p.1 + 1 : ℝ)) < (3 : ℝ) ^ (p.2 + 1 : ℝ)}

end pairs_count_l406_406494


namespace euler_theorem_l406_406829

open Nat

theorem euler_theorem (a n : ℕ) (h_coprime : gcd a n = 1) : a ^ φ n ≡ 1 [MOD n] := 
sorry

end euler_theorem_l406_406829


namespace tan_alpha_eq_neg_four_thirds_l406_406728

theorem tan_alpha_eq_neg_four_thirds (α : ℝ) (h1 : α ∈ set.Ioo (π / 2) π) (h2 : cos (π - α) = 3 / 5) : 
  tan α = -4 / 3 :=
by
  sorry

end tan_alpha_eq_neg_four_thirds_l406_406728


namespace base8_to_decimal_l406_406894

theorem base8_to_decimal (n : ℕ) (h : n = 54321) : 
  (5 * 8^4 + 4 * 8^3 + 3 * 8^2 + 2 * 8^1 + 1 * 8^0) = 22737 := 
by
  sorry

end base8_to_decimal_l406_406894


namespace no_solution_when_k_equals_7_l406_406662

noncomputable def no_solution_eq (k x : ℝ) : Prop :=
  (x - 3) / (x - 4) = (x - k) / (x - 8)
  
theorem no_solution_when_k_equals_7 :
  ∀ x : ℝ, x ≠ 4 → x ≠ 8 → ¬ no_solution_eq 7 x :=
by
  sorry

end no_solution_when_k_equals_7_l406_406662


namespace David_pushups_l406_406298

-- Definitions and setup conditions
def Zachary_pushups : ℕ := 7
def additional_pushups : ℕ := 30

-- Theorem statement to be proved
theorem David_pushups 
  (zachary_pushups : ℕ) 
  (additional_pushups : ℕ) 
  (Zachary_pushups_val : zachary_pushups = Zachary_pushups) 
  (additional_pushups_val : additional_pushups = additional_pushups) :
  zachary_pushups + additional_pushups = 37 :=
sorry

end David_pushups_l406_406298


namespace exists_monotonic_perfect_square_l406_406584

def digit_count (n : ℕ) : ℕ → Prop
| x := x.to_digits.length = n

def monotonic (x : ℕ) : Prop :=
  let ds := x.to_digits in ds = ds.sorted

def perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem exists_monotonic_perfect_square (n : ℕ) (hn : 0 < n) :
  ∃ x : ℕ, digit_count n x ∧ monotonic x ∧ perfect_square x :=
sorry

end exists_monotonic_perfect_square_l406_406584


namespace max_students_in_auditorium_l406_406960

def increment (i : ℕ) : ℕ :=
  (i * (i + 1)) / 2

def seats_in_row (i : ℕ) : ℕ :=
  10 + increment i

def max_students_in_row (n : ℕ) : ℕ :=
  (n + 1) / 2

def total_max_students_up_to_row (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => max_students_in_row (seats_in_row (i + 1)))

theorem max_students_in_auditorium : total_max_students_up_to_row 20 = 335 := 
sorry

end max_students_in_auditorium_l406_406960


namespace sum_floor_sqrt_1_to_25_l406_406996

theorem sum_floor_sqrt_1_to_25 :
  (∑ n in finset.range 25, ⌊real.sqrt (n + 1)⌋) = 75 :=
sorry

end sum_floor_sqrt_1_to_25_l406_406996


namespace evaluate_f_at_215_l406_406623

-- Define the piecewise function f
noncomputable def f : ℝ → ℝ :=
λ x, if x < 0 then Real.log (1 - x) / Real.log 2 else f (x) - 6

-- Theorem to prove f(215) = 1 given the function definition
theorem evaluate_f_at_215 : f 215 = 1 :=
sorry

end evaluate_f_at_215_l406_406623


namespace sum_of_floor_sqrt_1_to_25_l406_406989

theorem sum_of_floor_sqrt_1_to_25 : 
  ( ∑ i in Finset.range 26, ∥Real.sqrt i∥ ) = 75 :=
by
  sorry

end sum_of_floor_sqrt_1_to_25_l406_406989


namespace ext_9_implication_l406_406367

theorem ext_9_implication (a b : ℝ) (h1 : 3 + 2 * a + b = 0) (h2 : 1 + a + b + a^2 = 10) : (2 : ℝ)^3 + a * (2 : ℝ)^2 + b * (2 : ℝ) + a^2 - 1 = 17 := by
  sorry

end ext_9_implication_l406_406367


namespace final_discount_is_50_percent_off_l406_406950

-- Define the conditions
def original_price : ℝ := 100  -- For simplicity, assume the original price to be 100%

def promotional_price (p : ℝ) : ℝ := (2 / 3) * p

def price_after_coupon (p : ℝ) : ℝ := 0.75 * p

-- State the proof problem
theorem final_discount_is_50_percent_off (original_price : ℝ) :
  price_after_coupon (promotional_price original_price) = 0.5 * original_price :=
by
  sorry

end final_discount_is_50_percent_off_l406_406950


namespace Q_as_sum_of_squares_Q_sum_of_squares_zero_l406_406496

variables {R : Type*} [CommRing R] (x₁ x₂ x₃ x₄ : R)

def Q (x₁ x₂ x₃ x₄ : R) : R :=
  4 * (x₁^2 + x₂^2 + x₃^2 + x₄^2) - (x₁ + x₂ + x₃ + x₄)^2

theorem Q_as_sum_of_squares :
  Q x₁ x₂ x₃ x₄ =
    (x₁ + x₂ - x₃ - x₄)^2 +
    (x₁ - x₂ + x₃ - x₄)^2 +
    (x₁ - x₂ - x₃ + x₄)^2 :=
by sorry

theorem Q_sum_of_squares_zero (P₁ P₂ P₃ P₄ : R → R → R → R → R) :
  (∀ x₁ x₂ x₃ x₄, Q x₁ x₂ x₃ x₄ = P₁ x₁ x₂ x₃ x₄^2 + P₂ x₁ x₂ x₃ x₄^2 + P₃ x₁ x₂ x₃ x₄^2 + P₄ x₁ x₂ x₃ x₄^2) →
  (∃ i, ∀ x₁ x₂ x₃ x₄, P₁ x₁ x₂ x₃ x₄ = 0 ∧ P₂ x₁ x₂ x₃ x₄ = 0 ∧ P₃ x₁ x₂ x₃ x₄ = 0 ∧ P₄ x₁ x₂ x₃ x₄ = 0) :=
by sorry

end Q_as_sum_of_squares_Q_sum_of_squares_zero_l406_406496


namespace monotonic_g_range_of_a_l406_406431

open Real

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := exp x - a * x
def g (x : ℝ) : ℝ := (cos x - sin x) / exp x

-- Theorem 1: Discussing the monotonicity of g(x) on the interval (0, π)
theorem monotonic_g : (∀ x : ℝ, 0 < x ∧ x < π → 
  if x < π / 2 then deriv g x < 0 else deriv g x > 0) :=
sorry

-- Theorem 2: Finding the range of values for a given f(2x) >= g(x) for all x in [0, +∞)
theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x → f a (2 * x) ≥ g x) ↔ a ≤ 2 :=
sorry

end monotonic_g_range_of_a_l406_406431


namespace values_of_a_and_b_l406_406354

noncomputable def i : ℂ := complex.I

theorem values_of_a_and_b : 
  let z := (1 + i)^2 + 3 * (1 - i) / (2 + i),
      a := -3,
      b := 4
  in z^2 + a * z + b = 1 + i :=
by {
  let z := (1 + i)^2 + 3 * (1 - i) / (2 + i),
  have : z = 1 - i, sorry,
  have : (z^2 + -3 * z + 4) = 1 + i, sorry,
  exact this,
  sorry
}

end values_of_a_and_b_l406_406354


namespace intersection_is_01_l406_406704

def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℝ := {x | x^2 - x - 2 < 0}

theorem intersection_is_01 : (A ∩ B) = ({0, 1} : Set ℝ) :=
by
  sorry

end intersection_is_01_l406_406704


namespace find_length_of_train_l406_406266

def speed_kmh : Real := 60
def time_to_cross_bridge : Real := 26.997840172786177
def length_of_bridge : Real := 340

noncomputable def speed_ms : Real := speed_kmh * (1000 / 3600)
noncomputable def total_distance : Real := speed_ms * time_to_cross_bridge
noncomputable def length_of_train : Real := total_distance - length_of_bridge

theorem find_length_of_train :
  length_of_train = 109.9640028797695 := 
sorry

end find_length_of_train_l406_406266


namespace relationship_abc_l406_406685

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.exp (-Real.pi)
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relationship_abc : b < a ∧ a < c :=
by
  -- proofs would be added here
  sorry

end relationship_abc_l406_406685


namespace tangent_line_eq_l406_406481

open Real

noncomputable def f (x : ℝ) : ℝ := exp x * log x

theorem tangent_line_eq (x y : ℝ) (h : x = 1 ∧ y = 0) :
  ∃ m b, (∀ t, y = m * (t - 1) + b) ∧ (f x = y) ∧ (m = exp 1) ∧ (b = -exp 1) :=
by
  sorry

end tangent_line_eq_l406_406481


namespace puzzle_not_solvable_5_puzzle_solvable_2014_l406_406882

-- Define the initial state and the conditions
def initial_state (n : ℕ) : list ℕ := list.range' 1 n

-- Function to determine if the puzzle can be solved
def can_solve_puzzle (n : ℕ) : Prop := sorry -- To be defined with actual rules

-- Proof that the puzzle cannot be solved for n = 5
theorem puzzle_not_solvable_5 : ¬ can_solve_puzzle 5 :=
by sorry

-- Proof that the puzzle can be solved for n = 2014
theorem puzzle_solvable_2014 : can_solve_puzzle 2014 :=
by sorry

end puzzle_not_solvable_5_puzzle_solvable_2014_l406_406882


namespace tiling_remainder_is_888_l406_406564

noncomputable def boardTilingWithThreeColors (n : ℕ) : ℕ :=
  if n = 8 then
    4 * (21 * (3^3 - 3*2^3 + 3) +
         35 * (3^4 - 4*2^4 + 6) +
         35 * (3^5 - 5*2^5 + 10) +
         21 * (3^6 - 6*2^6 + 15) +
         7 * (3^7 - 7*2^7 + 21) +
         1 * (3^8 - 8*2^8 + 28))
  else
    0

theorem tiling_remainder_is_888 :
  boardTilingWithThreeColors 8 % 1000 = 888 :=
by
  sorry

end tiling_remainder_is_888_l406_406564


namespace discount_percentage_l406_406600

-- Definitions for the conditions in the problem
def cost_price : ℝ := 100
def markup_percentage : ℝ := 50
def actual_profit_percentage : ℝ := 27.5

-- Calculated values based on the conditions
def marked_price : ℝ := cost_price + (cost_price * (markup_percentage / 100))
def actual_selling_price : ℝ := cost_price + (cost_price * (actual_profit_percentage / 100))

-- Proof problem: Calculate the percentage discount offered
theorem discount_percentage :
  let discount_amount := marked_price - actual_selling_price in
  let discount_percentage := (discount_amount / marked_price) * 100 in
  discount_percentage = 15 :=
by
  sorry

end discount_percentage_l406_406600


namespace areas_of_quadrilaterals_are_equal_l406_406439

-- Definitions based on the conditions
def is_on_hyperbola (x y : ℝ) : Prop := y = 1 / x

def are_parallel_lines (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) : Prop :=
  ∀ i, i ∈ {1, 2, 3, 4} → - (1 / (match i with
    | 1 => a1 * b1
    | 2 => a2 * b2
    | 3 => a3 * b3
    | 4 => a4 * b4
  end)) = - (1 / (a1 * b1))

def area_of_quadrilateral (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) : ℝ :=
  (1 / 2) * abs (x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1 - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1))

-- Main theorem
theorem areas_of_quadrilaterals_are_equal
  (l1 l2 l3 l4 : ℝ → Prop)
  (h_1 : ∀ x, l1 x → is_on_hyperbola x (1 / x))
  (h_2 : ∀ x, l2 x → is_on_hyperbola x (1 / x))
  (h_3 : ∀ x, l3 x → is_on_hyperbola x (1 / x))
  (h_4 : ∀ x, l4 x → is_on_hyperbola x (1 / x))
  (A1 A2 A3 A4 B1 B2 B3 B4 : ℝ × ℝ)
  (hA1 : A1.2 = 1 / A1.1) (hA2 : A2.2 = 1 / A2.1) 
  (hA3 : A3.2 = 1 / A3.1) (hA4 : A4.2 = 1 / A4.1)
  (hB1 : B1.2 = 1 / B1.1) (hB2 : B2.2 = 1 / B2.1)
  (hB3 : B3.2 = 1 / B3.1) (hB4 : B4.2 = 1 / B4.1)
  (h_parallel : are_parallel_lines A1.1 A2.1 A3.1 A4.1 B1.1 B2.1 B3.1 B4.1) :
  area_of_quadrilateral A1.1 A1.2 A2.1 A2.2 A3.1 A3.2 A4.1 A4.2 = 
  area_of_quadrilateral B1.1 B1.2 B2.1 B2.2 B3.1 B3.2 B4.1 B4.2 :=
sorry

end areas_of_quadrilaterals_are_equal_l406_406439


namespace find_line_equation_l406_406258

theorem find_line_equation 
  (p : ℝ → ℝ) (l : ℝ → ℝ) (M : ℝ × ℝ)
  (h1 : ∀ x, p x = - (x^2) / 2)
  (h2 : M = (0, -1))
  (intersects : ∃ x1 x2 : ℝ, p x1 = l x1 ∧ p x2 = l x2)
  (sum_slopes : (λ x y : ℝ, y / x) ⟨x1, l x1⟩ + (λ x y : ℝ, y / x) ⟨x2, l x2⟩ = 1) :
  l = λ x, x - 1 :=
by
  sorry

end find_line_equation_l406_406258


namespace circles_area_sum_l406_406160

noncomputable def sum_of_areas_of_circles (r s t : ℝ) : ℝ :=
  π * (r^2 + s^2 + t^2)

theorem circles_area_sum (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  sum_of_areas_of_circles r s t = 56 * π :=
begin
  sorry
end

end circles_area_sum_l406_406160


namespace find_x_l406_406841

theorem find_x (a b x : ℝ) (h1 : ∀ a b, a * b = 2 * a - b) (h2 : 2 * (6 * x) = 2) : x = 10 := 
sorry

end find_x_l406_406841


namespace num_of_possible_values_l406_406029

def balls := {1, 2, 3, 4, 5}

def possible_sums := {x + y | x ∈ balls, y ∈ balls}

theorem num_of_possible_values : possible_sums.card = 9 :=
by sorry

end num_of_possible_values_l406_406029


namespace student_second_subject_percentage_l406_406951

theorem student_second_subject_percentage (x : ℝ) (h : (50 + x + 90) / 3 = 70) : x = 70 :=
by { sorry }

end student_second_subject_percentage_l406_406951


namespace tom_distance_before_karen_wins_l406_406065

theorem tom_distance_before_karen_wins :
  let speed_Karen := 60
  let speed_Tom := 45
  let delay_Karen := (4 : ℝ) / 60
  let distance_advantage := 4
  let time_to_catch_up := (distance_advantage + speed_Tom * delay_Karen) / (speed_Karen - speed_Tom)
  let distance_Tom := speed_Tom * time_to_catch_up
  distance_Tom = 21 :=
by
  sorry

end tom_distance_before_karen_wins_l406_406065


namespace number_of_divisors_l406_406221

def num_divisors (a b c l : ℕ) (α β γ λ : ℕ) : ℕ :=
  (α + 1) * (β + 1) * (γ + 1) * (λ + 1)

theorem number_of_divisors (a b c l : ℕ) (α β γ λ : ℕ) :
  prime a → prime b → prime c → prime l →
  α ≥ 0 → β ≥ 0 → γ ≥ 0 → λ ≥ 0 →
  ∀ N = a ^ α * b ^ β * c ^ γ * l ^ λ, 
  num_divisors a b c l α β γ λ = (α + 1) * (β + 1) * (γ + 1) * (λ + 1) :=
by
  intros Ha Hb Hc Hl Hα Hβ Hγ Hλ HN
  -- Proof omitted.
  sorry

end number_of_divisors_l406_406221


namespace ratio_area_hexagon_l406_406518

-- Assumptions from the conditions:
variables {s : ℝ} (A B C D F E G H : Point)

-- Define that A, B, C, D form a square and E, F, G, H form another square:
def square_A (A B C D : Point) : Prop := 
  distance A B = s ∧ distance B C = s ∧ distance C D = s ∧ distance D A = s ∧
  angle A B C = π/2 ∧ angle B C D = π/2 ∧ angle C D A = π/2 ∧ angle D A B = π/2

def square_E (E F G H : Point) : Prop := 
  distance E F = s ∧ distance F G = s ∧ distance G H = s ∧ distance H E = s ∧
  angle E F G = π/2 ∧ angle F G H = π/2 ∧ angle G H E = π/2 ∧ angle H E F = π/2

-- Points B and C are midpoints of EF and FG respectively:
def midpoints (B C F G : Point) : Prop := 
  distance E B = (s / 2) ∧ distance B F = (s / 2) ∧ distance F C = (s / 2) ∧ distance C G = (s / 2)

-- Coordinates or geometric relations of points can be omitted for brevity but are implicitly defined
-- in further calculations of areas which is a dependency on the above axioms.

-- The problem statement:
theorem ratio_area_hexagon :
  square_A A B C D → square_E E F G H → midpoints B C F G →
  (area_hexagon A F E D C B / (2 * s^2) = 3 / 4) :=
sorry

end ratio_area_hexagon_l406_406518


namespace eval_arith_expression_l406_406614

theorem eval_arith_expression : 2 + 3^2 * 4 - 5 + 6 / 2 = 36 := 
by sorry

end eval_arith_expression_l406_406614


namespace arccos_gt_arctan_l406_406646

open Real

theorem arccos_gt_arctan (x : ℝ) (hx : x ∈ set.Ico (-1 : ℝ) 1) : arccos x > arctan x :=
sorry

end arccos_gt_arctan_l406_406646


namespace value_of_a_purely_imaginary_l406_406735

-- Define the conditions under which a given complex number is purely imaginary
def is_purely_imaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = Complex.im z * Complex.I ∧ b ≠ 0

-- Define the complex number based on the variable a
def given_complex_number (a : ℝ) : ℂ :=
  ⟨a^2 - 3*a + 2, a - 1⟩

-- The proof statement
theorem value_of_a_purely_imaginary :
  is_purely_imaginary (given_complex_number 2) := sorry

end value_of_a_purely_imaginary_l406_406735


namespace equilateral_triangle_on_parallel_lines_l406_406596

-- Definitions for parallel lines and points on them
noncomputable def l1 := sorry -- define line l1
noncomputable def l2 := sorry -- define line l2 (parallel to l1)
noncomputable def l3 := sorry -- define line l3 (parallel to l1 and l2)

-- Proof that we can construct an equilateral triangle with vertices on given lines
theorem equilateral_triangle_on_parallel_lines :
  ∃ (A B C : Point), 
    A ∈ l1 ∧ B ∈ l2 ∧ C ∈ l3 ∧ 
    (AB = BC ∧ BC = CA ∧ CA = AB) :=
sorry

end equilateral_triangle_on_parallel_lines_l406_406596


namespace triangle_probability_l406_406508

theorem triangle_probability 
  (sticks : Finset ℕ)
  (H : sticks = {1, 3, 5, 7, 9})
  (valid_triangles : {({3, 5, 7}, {3, 7, 9}, {5, 7, 9})}) :
  (3/10) :=
by sorry

end triangle_probability_l406_406508


namespace symmetric_point_proof_l406_406138

def symmetric_point (p q : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  let midpoint := (p.1 + q.1) / 2, (p.2 + q.2) / 2 
  l (midpoint.1, midpoint.2)

theorem symmetric_point_proof :
  symmetric_point (-3, 4) (-2, 5) (fun x => x.1 + x.2 - 2 = 0) :=
by
  sorry

end symmetric_point_proof_l406_406138


namespace seats_needed_l406_406917

theorem seats_needed (children seats_per_seat : ℕ) (h1 : children = 58) (h2 : seats_per_seat = 2) : children / seats_per_seat = 29 :=
by sorry

end seats_needed_l406_406917


namespace expected_value_l406_406039

noncomputable def p : ℝ := 0.25
noncomputable def P_xi_1 : ℝ := 0.24
noncomputable def P_black_bag_b : ℝ := 0.8
noncomputable def P_xi_0 : ℝ := (1 - p) * (1 - P_black_bag_b) * (1 - P_black_bag_b)
noncomputable def P_xi_2 : ℝ := p * (1 - P_black_bag_b) * (1 - P_black_bag_b) + (1 - p) * P_black_bag_b * P_black_bag_b
noncomputable def P_xi_3 : ℝ := p * P_black_bag_b + p * (1 - P_black_bag_b) * P_black_bag_b
noncomputable def E_xi : ℝ := 0 * P_xi_0 + 1 * P_xi_1 + 2 * P_xi_2 + 3 * P_xi_3

theorem expected_value : E_xi = 1.94 := by
  sorry

end expected_value_l406_406039


namespace Q_subset_P_l406_406092

def P : Set ℝ := {x | x < 2}
def Q : Set ℝ := {y | y < 1}

theorem Q_subset_P : Q ⊆ P := by
  sorry

end Q_subset_P_l406_406092


namespace equation1_equation2_equation3_equation4_l406_406839

theorem equation1 (x : ℝ) : (x - 1) ^ 2 - 5 = 0 ↔ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 := by
  sorry

theorem equation2 (x : ℝ) : x * (x + 4) = -3 * (x + 4) ↔ x = -4 ∨ x = -3 := by
  sorry

theorem equation3 (y : ℝ) : 2 * y ^ 2 - 5 * y + 2 = 0 ↔ y = 1 / 2 ∨ y = 2 := by
  sorry

theorem equation4 (m : ℝ) : 2 * m ^ 2 - 7 * m - 3 = 0 ↔ m = (7 + Real.sqrt 73) / 4 ∨ m = (7 - Real.sqrt 73) / 4 := by
  sorry

end equation1_equation2_equation3_equation4_l406_406839


namespace rhombus_sufficient_but_not_necessary_l406_406343

-- Define a quadrilateral ABCD
structure Quadrilateral (A B C D : Type) :=
  (A B C D : A)

-- Define the rhombus as a special case of a quadrilateral
def is_rhombus {A B C D : Type} (quad : Quadrilateral A B C D) : Prop :=
  true -- This is a placeholder. Define suitably as per geometrical properties of a rhombus

-- Define the perpendicular diagonals property
def diagonals_perpendicular {A B C D : Type} (quad : Quadrilateral A B C D) : Prop :=
  true -- This is a placeholder. Define suitably as per the property of perpendicular diagonals

-- Theorem statement proving the sufficient but not necessary condition
theorem rhombus_sufficient_but_not_necessary {A B C D : Type} (quad : Quadrilateral A B C D) :
  (is_rhombus quad → diagonals_perpendicular quad) ∧ ¬(diagonals_perpendicular quad → is_rhombus quad) :=
by
  admit -- This skips the actual proof
  

end rhombus_sufficient_but_not_necessary_l406_406343


namespace value_range_of_function_l406_406507

def f (x : ℝ) : ℝ := x^2 + 2 * x - 1

theorem value_range_of_function : set.range (λ x, f x) ∩ [{ x | -3 ≤ x ∧ x ≤ 2 }] = { y | -2 ≤ y ∧ y ≤ 7 } :=
by {
  sorry
}

end value_range_of_function_l406_406507


namespace calculate_sum_of_powers_l406_406515

theorem calculate_sum_of_powers :
  (6^2 - 3^2)^4 + (7^2 - 2^2)^4 = 4632066 :=
by
  sorry

end calculate_sum_of_powers_l406_406515


namespace find_k_l406_406724

theorem find_k (k : ℝ) (h : (3:ℝ)^4 + k * (3:ℝ)^2 - 26 = 0) : k = -55 / 9 := 
by sorry

end find_k_l406_406724


namespace sequence_properties_l406_406703

theorem sequence_properties :
  (∀ n: ℕ, n > 0 → a (n+1) - a n = 2) ∧ (a 3 = 3) →
  (a 1 = -1) ∧ (∀ n: ℕ, S n = n^2 - 2*n) :=
by
  intro h
  sorry

end sequence_properties_l406_406703


namespace set_of_x_satisfying_2f_less_than_x_plus_1_l406_406799

theorem set_of_x_satisfying_2f_less_than_x_plus_1 (f : ℝ → ℝ) 
  (h1 : f 1 = 1) 
  (h2 : ∀ x : ℝ, deriv f x > 1 / 2) :
  { x : ℝ | 2 * f x < x + 1 } = { x : ℝ | x < 1 } :=
by
  sorry

end set_of_x_satisfying_2f_less_than_x_plus_1_l406_406799


namespace problem_solution_l406_406202

theorem problem_solution (a b d : ℤ) (ha : a = 2500) (hb : b = 2409) (hd : d = 81) :
  (a - b) ^ 2 / d = 102 := by
  sorry

end problem_solution_l406_406202


namespace number_of_valid_lines_l406_406273

/-
Proof Problem: In the cube ABCD A1B1C1D1, prove that there are exactly 3 lines l
passing through vertex A_1, such that the angle between l and line AC is 60 degrees,
and the angle between l and line BC1 is 60 degrees.
-/

structure Cube :=
  (A B C D A1 B1 C1 D1 : Point)

def angle_between (p1 p2 p3 : Point) : ℝ := sorry

def valid_line (l : Line) (cube : Cube) : Prop :=
  let A1 := cube.A1 in
  let AC := Line.mk cube.A cube.C in
  let BC1 := Line.mk cube.B cube.C1 in
  l.through A1 ∧ 
  angle_between A1 (l.projection A1) AC = 60 ∧ 
  angle_between A1 (l.projection A1) BC1 = 60

theorem number_of_valid_lines (cube : Cube) : 
  ∃! (l : List Line), (∀ x ∈ l, valid_line x cube) ∧ l.length = 3 := sorry

end number_of_valid_lines_l406_406273


namespace angle_BED_is_15_l406_406088

theorem angle_BED_is_15 (A B C D E : Type*)
  [equilateral_triangle A B C] 
  [is_midpoint B A D]
  (h1 : distance D E = distance A B) : 
  angle B E D = 15 :=
by
  sorry

end angle_BED_is_15_l406_406088


namespace sqrt_expression_nonneg_l406_406871

theorem sqrt_expression_nonneg {b : ℝ} : b - 3 ≥ 0 ↔ b ≥ 3 := by
  sorry

end sqrt_expression_nonneg_l406_406871


namespace sum_of_areas_of_circles_l406_406179

theorem sum_of_areas_of_circles (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  ∃ (π : ℝ), π * (r^2 + s^2 + t^2) = 56 * π :=
by {
  sorry
}

end sum_of_areas_of_circles_l406_406179


namespace distance_correct_l406_406263

-- Define geometry entities and properties
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Sphere where
  center : Point
  radius : ℝ

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define conditions
def sphere_center : Point := { x := 0, y := 0, z := 0 }
def sphere : Sphere := { center := sphere_center, radius := 5 }
def triangle : Triangle := { a := 13, b := 13, c := 10 }

-- Define the distance calculation
noncomputable def distance_from_sphere_center_to_plane (O : Point) (T : Triangle) : ℝ :=
  let h := 12  -- height calculation based on given triangle sides
  let A := 60  -- area of the triangle
  let s := 18  -- semiperimeter
  let r := 10 / 3  -- inradius calculation
  let x := 5 * (Real.sqrt 5) / 3  -- final distance calculation
  x

-- Prove the obtained distance matches expected value
theorem distance_correct :
  distance_from_sphere_center_to_plane sphere_center triangle = 5 * (Real.sqrt 5) / 3 :=
by
  sorry

end distance_correct_l406_406263


namespace repeating_decimal_as_fraction_l406_406640

-- Define the repeating decimal x as .overline{37}
def x : ℚ := 37 / 99

-- The theorem we need to prove
theorem repeating_decimal_as_fraction : x = 37 / 99 := by
  sorry

end repeating_decimal_as_fraction_l406_406640


namespace girls_fraction_l406_406604

/-- Maple Grove Middle School has 300 students with a boys to girls ratio of 3:2.
    Pine Ridge Middle School has 240 students with a boys to girls ratio of 1:3.
    Prove that the fraction of the attendees at the social event who are girls is 5/9. -/
theorem girls_fraction (mg_students : ℕ) (pr_students : ℕ)
                       (mg_boy_ratio : ℕ) (mg_girl_ratio : ℕ)
                       (pr_boy_ratio : ℕ) (pr_girl_ratio : ℕ) :
                       mg_students = 300 → pr_students = 240 →
                       mg_boy_ratio = 3 → mg_girl_ratio = 2 →
                       pr_boy_ratio = 1 → pr_girl_ratio = 3 →
                       (let mg_total_girls := (mg_students * mg_girl_ratio) / (mg_boy_ratio + mg_girl_ratio) in
                        let pr_total_girls := (pr_students * pr_girl_ratio) / (pr_boy_ratio + pr_girl_ratio) in
                        let total_students := mg_students + pr_students in
                        let total_girls := mg_total_girls + pr_total_girls in
                        (total_girls : ℚ) / total_students = 5 / 9) :=
by 
  intros mg_students_eq pr_students_eq mg_boy_ratio_eq mg_girl_ratio_eq pr_boy_ratio_eq pr_girl_ratio_eq
  sorry

end girls_fraction_l406_406604


namespace triangle_area_l406_406744

noncomputable def area_triangle (b c angle_C : ℝ) : ℝ :=
  (1 / 2) * b * c * Real.sin angle_C

theorem triangle_area :
  let b := 1
  let c := Real.sqrt 3
  let angle_C := 2 * Real.pi / 3
  area_triangle b c (Real.sin angle_C) = Real.sqrt 3 / 4 :=
by
  sorry

end triangle_area_l406_406744


namespace sum_cis_sequence_l406_406970

def cis (θ : ℝ) : Complex := Complex.exp (θ * Complex.I)

theorem sum_cis_sequence (r : ℝ) (h₁ : r > 0) :
    let θ := 90 * (Real.pi / 180)
    let angles := List.range' 40 11
    (∑ i in angles, cis (i * (Real.pi / 180))) = r * cis θ →
    θ = 90 * (Real.pi / 180) := by
  sorry

end sum_cis_sequence_l406_406970


namespace car_speed_in_kmh_l406_406591

theorem car_speed_in_kmh (rev_per_min : ℕ) (circumference : ℕ) (speed : ℕ) 
  (h1 : rev_per_min = 400) (h2 : circumference = 4) : speed = 96 :=
  sorry

end car_speed_in_kmh_l406_406591


namespace incorrect_conclusion_l406_406304

theorem incorrect_conclusion (p q : ℝ) (h1 : p < 0) (h2 : q < 0) : ¬ ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ (x1 * |x1| + p * x1 + q = 0) ∧ (x2 * |x2| + p * x2 + q = 0) ∧ (x3 * |x3| + p * x3 + q = 0) :=
by
  sorry

end incorrect_conclusion_l406_406304


namespace total_spent_on_computer_l406_406059

def initial_cost_of_pc : ℕ := 1200
def sale_price_old_card : ℕ := 300
def cost_new_card : ℕ := 500

theorem total_spent_on_computer : 
  (initial_cost_of_pc + (cost_new_card - sale_price_old_card)) = 1400 :=
by
  sorry

end total_spent_on_computer_l406_406059


namespace total_cost_correct_l406_406852

variables (x1 x2 x3 x4 x5 x6 x7 : ℝ)

def cat_preparation_cost : ℝ := 50
def adult_dog_preparation_cost : ℝ := 100
def puppy_preparation_cost : ℝ := 150

def total_preparation_cost (x1 x2 x3 x4 x5 x6 x7 : ℝ) : ℝ :=
  2 * cat_preparation_cost + 3 * adult_dog_preparation_cost + 2 * puppy_preparation_cost +
  (x1 + x2) + (x3 + x4 + x5) + (x6 + x7)

theorem total_cost_correct (x1 x2 x3 x4 x5 x6 x7 : ℝ) :
  total_preparation_cost x1 x2 x3 x4 x5 x6 x7 = 700 + (x1 + x2 + x3 + x4 + x5 + x6 + x7) :=
by {
  sorry, -- Proof to be completed
}

end total_cost_correct_l406_406852


namespace parabola_focus_directrix_l406_406387

theorem parabola_focus_directrix (a : ℝ) (p : ℝ) (h1 : y^2 = ax) (h2 : 2 * p = 2) (h3 : a = 4 * p) : a = 4 :=
by
  sorry

end parabola_focus_directrix_l406_406387


namespace find_eccentricity_of_hyperbola_l406_406340

noncomputable def hyperbola_eccentricity (e : ℝ) : Prop :=
  ∃ (a1 a2 c : ℝ), 
    (a1^2 / c^2) + (3 * a2^2 / c^2) = 4 ∧
    1 / ( (√2 / 2) ^ 2 ) + 3 / e^2 = 4

theorem find_eccentricity_of_hyperbola :
  ∀ e : ℝ, 
    hyperbola_eccentricity e → 
      e = (√6 / 2) :=
by
  sorry

end find_eccentricity_of_hyperbola_l406_406340


namespace petya_could_not_spent_atleast_5000_l406_406823

noncomputable def petya_spent_less_than_5000 (k : ℕ) : Prop :=
  let M := 100 * k
  in ∃ (books : list ℕ), 
    (∀ (x ∈ books), (x < 100 ∨ x >= 100)) ∧ -- books prices conditions
    ((∃ (num_100s : ℕ), num_100s = k) ∧ -- initially k number of 100-ruble bills
    (sum books = M / 2) ∧ -- spend exactly half of total money on books 
    (sum books < 5000)) -- verify if the total amount spent on books is less than 5000

theorem petya_could_not_spent_atleast_5000 (k : ℕ) (h : k > 0) : petya_spent_less_than_5000 k :=
sorry

end petya_could_not_spent_atleast_5000_l406_406823


namespace sqrt8000_minus_50_eq_form_l406_406475

theorem sqrt8000_minus_50_eq_form (a b : ℤ) (h1 : 0 < a ∧ 0 < b) 
  (h2 : (sqrt 8000) - 50 = (sqrt a - b)^3) : 
  a + b = 16 := 
by
  sorry

end sqrt8000_minus_50_eq_form_l406_406475


namespace proof_problem_l406_406404

noncomputable def cartesian_equation_C (α : ℝ) (hα : α ≠ (λ k : ℤ, k * π + π / 2)) : Prop :=
  let x := 1 / Real.cos α;
  let y := (Real.sqrt 3 * Real.sin α) / Real.cos α;
  x^2 - y^2 / 3 = 1

noncomputable def rectangular_equation_l : Prop :=
  ∀ ρ θ, ρ * Real.cos (θ + π / 3) = 1 ↔ ρ * (1 - (Real.sqrt 3) * (Real.sin θ / Real.cos θ)) = 2

noncomputable def value_PA_PB (A B P : ℝ × ℝ) : ℝ :=
  | 1 / (Real.sqrt ((fst A - fst P)^2 + (snd A - snd P)^2)) - 
    1 / (Real.sqrt ((fst B - fst P)^2 + (snd B - snd P)^2)) |

theorem proof_problem (α : ℝ) (hα : α ≠ (λ k : ℤ, k * π + π / 2)) (A B : ℝ × ℝ) (P : ℝ × ℝ) :
  cartesian_equation_C α hα ∧ rectangular_equation_l ∧ P = (2, 0) → value_PA_PB A B P = 2 / 3 :=
sorry

end proof_problem_l406_406404


namespace sequence_general_term_l406_406048

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 2^n) :
  ∀ n : ℕ, a n = 2^n :=
sorry

end sequence_general_term_l406_406048


namespace arith_seq_formula_geom_seq_formula_sum_c_formula_range_t_l406_406362

-- Given conditions
def arith_seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = a 1 + d ∧ a 5 = a 1 + 4 * d

def geom_seq (b : ℕ → ℤ) (q : ℤ) : Prop :=
  q ≠ 1 ∧ b 1 = 1 ∧ b 2 = q ∧ b 3 = q^2

def common_diff (d : ℤ) : Prop := d = 2

def common_ratio (q : ℤ) : Prop := q = 3

-- Sequences definitions
def a (n : ℕ) : ℤ := 2 * n - 1
def b (n : ℕ) : ℤ := 3^(n - 1)

-- Proving the general formula for arithmetic sequence
theorem arith_seq_formula : ∀ n, arith_seq a := by
  sorry

-- Proving the general formula for the geometric sequence
theorem geom_seq_formula : ∀ n, ∃ q, geom_seq b q := by
  sorry

-- Sum of the first n terms of sequence c_n
def c (n : ℕ) : ℤ := b n + 1 / (a n * a (n + 1))

def T (n : ℕ) : ℤ := (3^n - 1 / (2 * n + 1)) / 2
def sum_c (n : ℕ) : ℤ := ∑ i in finset.range n, c i

theorem sum_c_formula : ∀ n, sum_c n = T n := by
  sorry

-- Proving the inequality condition on t
theorem range_t : ∀ t, (∀ (n : ℕ), 2 * T n > (4 * n - 3) * t - 1 / (2 * n + 1)) ↔ t < 9 / 5 := by
  sorry

end arith_seq_formula_geom_seq_formula_sum_c_formula_range_t_l406_406362


namespace complement_of_union_l406_406706

open Set

variable (U : Set ℕ) [DecidableEq ℕ] (A B : Set ℕ)

theorem complement_of_union {U : Set ℕ} [Finset U] (hU : U = {1, 2, 3, 4, 5}) 
                              {A : Set ℕ} (hA : A = {1, 2, 3}) 
                              {B : Set ℕ} (hB : B = {3, 4}) : 
  compl (A ∪ B) = {5} :=
by
  intro x
  sorry  -- proof omitted

end complement_of_union_l406_406706


namespace proof_statements_correct_l406_406749

variable (candidates : Nat) (sample_size : Nat)

def is_sampling_survey (survey_type : String) : Prop :=
  survey_type = "sampling"

def is_population (pop_size sample_size : Nat) : Prop :=
  (pop_size = 60000) ∧ (sample_size = 1000)

def is_sample (sample_size pop_size : Nat) : Prop :=
  sample_size < pop_size

def sample_size_correct (sample_size : Nat) : Prop :=
  sample_size = 1000

theorem proof_statements_correct :
  ∀ (survey_type : String) (pop_size sample_size : Nat),
  is_sampling_survey survey_type →
  is_population pop_size sample_size →
  is_sample sample_size pop_size →
  sample_size_correct sample_size →
  survey_type = "sampling" ∧
  pop_size = 60000 ∧
  sample_size = 1000 :=
by
  intros survey_type pop_size sample_size hs hp hsamp hsiz
  sorry

end proof_statements_correct_l406_406749


namespace range_of_omega_l406_406005

noncomputable theory

def function_has_no_zeros (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x ∈ I, f x ≠ 0

def f (ω x : ℝ) : ℝ :=
  sin^2 (ω * x / 2) + 0.5 * sin (ω * x) - 0.5

theorem range_of_omega (ω : ℝ) (h : ω > 0) :
  (function_has_no_zeros (f ω) {x | π < x ∧ x < 2 * π}) ↔
  (ω ∈ (set.Ioo 0 (1/8) ∪ set.Icc (1/4) (5/8))) :=
sorry

end range_of_omega_l406_406005


namespace luna_total_monthly_budget_l406_406097

theorem luna_total_monthly_budget
  (H F phone_bill : ℝ)
  (h1 : F = 0.60 * H)
  (h2 : H + F = 240)
  (h3 : phone_bill = 0.10 * F) :
  H + F + phone_bill = 249 :=
by sorry

end luna_total_monthly_budget_l406_406097


namespace solution_count_l406_406374

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem solution_count (a : ℝ) : 
  (∃ x : ℝ, f x = a) ↔ 
  ((a > 2 ∨ a < -2 ∧ ∃! x₁, f x₁ = a) ∨ 
   ((a = 2 ∨ a = -2) ∧ ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = a ∧ f x₂ = a) ∨ 
   (-2 < a ∧ a < 2 ∧ ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ = a ∧ f x₂ = a ∧ f x₃ = a)) := 
by sorry

end solution_count_l406_406374


namespace Beto_can_determine_xy_l406_406790

theorem Beto_can_determine_xy (m n : ℤ) :
  (∃ k t : ℤ, 0 < t ∧ m = 2 * k + 1 ∧ n = 2 * t * (2 * k + 1)) ↔ 
  (∀ x y : ℝ, (∃ a b : ℝ, a ≠ b ∧ x = a ∧ y = b) →
    ∃ xy_val : ℝ, (x^m + y^m = xy_val) ∧ (x^n + y^n = xy_val)) := 
sorry

end Beto_can_determine_xy_l406_406790


namespace right_triangle_geo_seq_ratio_l406_406156

theorem right_triangle_geo_seq_ratio (l r : ℝ) (ht : 0 < l)
  (hr : 1 < r) (hgeo : l^2 + (l * r)^2 = (l * r^2)^2) :
  (l * r^2) / l = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end right_triangle_geo_seq_ratio_l406_406156


namespace sum_of_areas_of_circles_l406_406178

theorem sum_of_areas_of_circles (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  ∃ (π : ℝ), π * (r^2 + s^2 + t^2) = 56 * π :=
by {
  sorry
}

end sum_of_areas_of_circles_l406_406178


namespace dice_probability_not_all_same_l406_406534

theorem dice_probability_not_all_same : 
  let total_outcomes := (8 : ℕ)^5 in
  let same_number_outcomes := 8 in
  let probability_all_same := (same_number_outcomes : ℚ) / total_outcomes in
  let probability_not_all_same := 1 - probability_all_same in
  probability_not_all_same = 4095 / 4096 :=
by
  sorry

end dice_probability_not_all_same_l406_406534


namespace Ivan_pays_1_point_5_times_more_l406_406632

theorem Ivan_pays_1_point_5_times_more (x y : ℝ) (h : x = 2 * y) : 1.5 * (0.6 * x + 0.8 * y) = x + y :=
by
  sorry

end Ivan_pays_1_point_5_times_more_l406_406632


namespace primeFactors_of_3_pow_6_minus_1_l406_406610

def calcPrimeFactorsSumAndSumOfSquares (n : ℕ) : ℕ × ℕ :=
  let factors := [2, 7, 13]  -- Given directly
  let sum_factors := 2 + 7 + 13
  let sum_squares := 2^2 + 7^2 + 13^2
  (sum_factors, sum_squares)

theorem primeFactors_of_3_pow_6_minus_1 :
  calcPrimeFactorsSumAndSumOfSquares (3^6 - 1) = (22, 222) :=
by
  sorry

end primeFactors_of_3_pow_6_minus_1_l406_406610


namespace parallel_mq_np_l406_406401

theorem parallel_mq_np 
  {A B C D E F G H M N P Q O : Point}
  (h_rhombus: is_rhombus A B C D)
  (h_incircle: incircle O A B C D E F G H)
  (h_tangent1: tangent O E F M N)
  (h_tangent2: tangent O G H P Q)
  (h_touch_AB: touches O A B E)
  (h_touch_BC: touches O B C F)
  (h_touch_CD: touches O C D G)
  (h_touch_DA: touches O D A H):
  parallel M Q N P := sorry

end parallel_mq_np_l406_406401


namespace petya_could_not_spend_5000_l406_406820

-- Define the problem conditions and proof functions
def petya_spent_at_least_5000 (spent : ℕ) : Prop :=
  spent >= 5000

-- Main theorem to prove
theorem petya_could_not_spend_5000 :
  ∀ (bill_count : ℕ),
  let initial_money := 100 * bill_count in
  let total_spent := initial_money / 2 in
  total_spent < 5000 ->
  ∀ (books : list ℕ),
  (∀ b ∈ books, b > 0) -> -- The cost of each book is a positive integer.
  ∃ n : ℕ, 
  n = bill_count ->
  ¬ petya_spent_at_least_5000 total_spent := 
by
  intros bill_count books Hbook_zero b_in_books H_spent conditions
  sorry -- proof to be filled in

end petya_could_not_spend_5000_l406_406820


namespace value_of_f_at_5π_over_3_l406_406247

-- Define the function f(x) with given properties
def f (x : ℝ) : ℝ := if 0 ≤ x ∧ x ≤ π/2 then π/2 - x else sorry

-- f is even, f(x) = f(-x) 
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- f has period π, f(x) = f(x + π) 
def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f x = f (x + p)

-- Main statement
theorem value_of_f_at_5π_over_3 :
  even_function f ∧ periodic_function f π ∧ (∀ x, 0 ≤ x ∧ x ≤ π/2 → f x = π/2 - x) →
  f (5 * π / 3) = π / 6 :=
begin
  sorry
end

end value_of_f_at_5π_over_3_l406_406247


namespace cycle_not_divisible_by_3_l406_406853

theorem cycle_not_divisible_by_3 (G : Type) [Graph G] (h1: ∀ v : G, 3 ≤ degree v) : 
  ∃ c : Cycle G, ¬ 3 ∣ length c :=
sorry

end cycle_not_divisible_by_3_l406_406853


namespace problem1_problem2_l406_406759

noncomputable def probability_one_from_mixed_cards : ℚ :=
  let red_cards := [1, 2, 3, 4]
  let blue_cards := [1, 2, 3]
  let total_cards := red_cards ++ blue_cards
  let number_of_ones := total_cards.count (λ x => x = 1)
  number_of_ones / total_cards.length

theorem problem1 :
  probability_one_from_mixed_cards = 2 / 7 :=
by
  sorry

noncomputable def probability_two_digit_greater_than_22 : ℚ :=
  let red_cards := [1, 2, 3, 4]
  let blue_cards := [1, 2, 3]
  let two_digit_numbers := (red_cards.product blue_cards).map (λ (x, y) => 10 * x + y)
  let numbers_greater_than_22 := two_digit_numbers.filter (λ n => n > 22)
  numbers_greater_than_22.length / two_digit_numbers.length

theorem problem2 :
  probability_two_digit_greater_than_22 = 7 / 12 :=
by
  sorry

end problem1_problem2_l406_406759


namespace probability_of_picking_two_red_balls_l406_406547

noncomputable def pick_two_red_probability : ℚ :=
  let red_balls := 5
  let blue_balls := 4
  let green_balls := 3
  let total_balls := red_balls + blue_balls + green_balls
  let total_ways := (total_balls.choose 2 : ℕ) 
  let favorable_ways := (red_balls.choose 2 : ℕ)
  (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_picking_two_red_balls :
  pick_two_red_probability = 5 / 33 :=
by {
  sorry,
}

end probability_of_picking_two_red_balls_l406_406547


namespace library_books_new_iff_conditions_l406_406739

-- Definitions of conditions
def all_books_new (L : Type) (B : L → Prop) : Prop :=
  ∀ x, B x

def all_books_old (L : Type) (B : L → Prop) (O : L → Prop) : Prop :=
  ∀ x, O x

def exists_not_new (L : Type) (B : L → Prop) : Prop :=
  ∃ x, ¬ B x

def no_books_new (L : Type) (B : L → Prop) : Prop :=
  ∀ x, ¬ B x

def not_all_books_new (L : Type) (B : L → Prop) : Prop :=
  ∃ x, ¬ B x

-- Theorem statement to be proven
theorem library_books_new_iff_conditions (L : Type) (B : L → Prop) :
  ¬(all_books_new L B) → (exists_not_new L B ∧ not_all_books_new L B) := 
begin
  sorry
end

end library_books_new_iff_conditions_l406_406739


namespace find_T_268_l406_406130

def T (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

axiom linearity (a b : ℝ) (v w : ℝ × ℝ × ℝ) : 
  T (a • v + b • w) = a • T v + b • T w

axiom preserves_cross_product (v w : ℝ × ℝ × ℝ) : 
  T (v × w) = T v × T w

axiom T_442 : T (4, 4, 2) = (3, -2, 5)
axiom T_-424 : T (-4, 2, 4) = (3, 5, -2)

theorem find_T_268 : 
  T (2, 6, 8) = (16/3, 6, 3) :=
sorry

end find_T_268_l406_406130


namespace imaginary_part_of_z_l406_406858

variable (z : ℂ) (h : z = 1 - complex.i)

theorem imaginary_part_of_z : complex.im z = -1 := by
  rw [h]
  simp
  exact rfl

end imaginary_part_of_z_l406_406858


namespace range_of_t_l406_406369

theorem range_of_t 
  (k t : ℝ)
  (tangent_condition : (t + 1)^2 = 1 + k^2)
  (intersect_condition : ∃ x y, y = k * x + t ∧ y = x^2 / 4) : 
  t > 0 ∨ t < -3 :=
sorry

end range_of_t_l406_406369


namespace convex_irregular_pentagon_impossible_l406_406285

theorem convex_irregular_pentagon_impossible
: ¬ ∃ (P : Fin 5 → ℝ²),
  -- Define the convex pentagon P
  (convex_hull (P '' (Finset.univ : Finset (Fin 5))).points) ∧
  -- Define the irregularity
  (¬∃ r, ∀ i j, i ≠ j → P i = r) ∧
  -- Define exactly four sides of equal length
  (∃ l, (Finset.card (Finset.filter (λ (s, t), (P s = l ∨ P t = l) ∧ P (s + 1) = l) ((Finset.product (Finset.univ : Finset (Fin 5)) (Finset.univ : Finset (Fin 5))))) = 4)) ∧
  -- Define exactly four diagonals of equal length
  (∃ d, (Finset.card (Finset.filter (λ (s, t), (P s ≠ d ∧ P t ≠ d) ∧ P (s + 2) = d) ((Finset.product (Finset.univ : Finset (Fin 5)) (Finset.univ : Finset (Fin 5))))) = 4)) ∧
  -- Define a fifth side intersecting with a fifth diagonal
  (∃ (s : Fin 5), ∃ (d : Fin 5), (s ≠ d) ∧ 
    ∃ (P s : ℝ), (P s ≠ P (s + 1)) ∧ (P s ≠ P (s + 3))) 
 := sorry

end convex_irregular_pentagon_impossible_l406_406285


namespace part_a_part_b_l406_406192

def max_rectangles (n m : ℕ) : ℕ :=
  -- function to calculate the maximum number of non-overlapping rectangles
  sorry

def T (table : ℕ × ℕ) : set (set (ℕ × ℕ)) :=
  -- definition of the set of rectangles such that no rectangle is a subrectangle of another
  sorry

theorem part_a : 
  let t := max_rectangles 10 10 in
  t > 300 := 
by 
  sorry

theorem part_b : 
  let t := max_rectangles 10 10 in
  t < 600 := 
by 
  sorry

end part_a_part_b_l406_406192


namespace perpendicular_lines_k_value_l406_406629

theorem perpendicular_lines_k_value (k : ℝ) : 
  (∃ (m₁ m₂ : ℝ), (m₁ = k/3) ∧ (m₂ = 3) ∧ (m₁ * m₂ = -1)) → k = -1 :=
by
  sorry

end perpendicular_lines_k_value_l406_406629


namespace matrix_power_50_l406_406419

def P : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![3, 2],
  ![-4, -3]
]

theorem matrix_power_50 :
  P ^ 50 = ![
    ![1, 0],
    ![0, 1]
  ] :=
sorry

end matrix_power_50_l406_406419


namespace crayons_left_l406_406455

-- Define the initial number of crayons
def initial_crayons : ℕ := 440

-- Define the crayons given away
def crayons_given : ℕ := 111

-- Define the crayons lost
def crayons_lost : ℕ := 106

-- Prove the final number of crayons left
theorem crayons_left : (initial_crayons - crayons_given - crayons_lost) = 223 :=
by
  sorry

end crayons_left_l406_406455


namespace eval_closest_value_l406_406636

theorem eval_closest_value :
  abs ((3.76 * real.sqrt 16.81 * (8.13 + 1.87)) - 156) ≤ abs ((3.76 * real.sqrt 16.81 * (8.13 + 1.87)) - 150) ∧
  abs ((3.76 * real.sqrt 16.81 * (8.13 + 1.87)) - 156) ≤ abs ((3.76 * real.sqrt 16.81 * (8.13 + 1.87)) - 160) ∧
  abs ((3.76 * real.sqrt 16.81 * (8.13 + 1.87)) - 156) ≤ abs ((3.76 * real.sqrt 16.81 * (8.13 + 1.87)) - 170) ∧
  abs ((3.76 * real.sqrt 16.81 * (8.13 + 1.87)) - 156) ≤ abs ((3.76 * real.sqrt 16.81 * (8.13 + 1.87)) - 180) :=
by
  sorry

end eval_closest_value_l406_406636


namespace proportion_solution_l406_406383

theorem proportion_solution (x : ℝ) (h : 0.6 / x = 5 / 8) : x = 0.96 :=
by 
  -- The proof will go here
  sorry

end proportion_solution_l406_406383


namespace frances_towel_weight_in_ounces_l406_406099

theorem frances_towel_weight_in_ounces :
  (∀ Mary_towels Frances_towels : ℕ,
    Mary_towels = 4 * Frances_towels →
    Mary_towels = 24 →
    (Mary_towels + Frances_towels) * 2 = 60 →
    Frances_towels * 2 * 16 = 192) :=
by
  intros Mary_towels Frances_towels h1 h2 h3
  sorry

end frances_towel_weight_in_ounces_l406_406099


namespace f_neg_2008_value_l406_406000

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x^7 + b * x - 2

theorem f_neg_2008_value (h : f a b 2008 = 10) : f a b (-2008) = -12 := by
  sorry

end f_neg_2008_value_l406_406000


namespace coloring_count_correct_l406_406293

def proper_divisors (n : ℕ) : List ℕ :=
  (List.range (n - 1)).tail.filter (λ d, n % d = 0)

def different_color_constraint (coloring : ℕ → Bool) (n : ℕ) : Prop :=
  ∀ d ∈ proper_divisors n, coloring n ≠ coloring d

def valid_coloring_count : ℕ :=
  let numbers := [2, 3, 4, 5, 8, 10, 12]
  let valid_colorings := {coloring : ℕ → Bool // ∀ n ∈ numbers, different_color_constraint coloring n}
  valid_colorings.card

theorem coloring_count_correct :
  valid_coloring_count = 128 := 
sorry

end coloring_count_correct_l406_406293


namespace coin_toss_probability_l406_406906

noncomputable def fair_coin := (1 / 2 : ℝ)

theorem coin_toss_probability (h : fair_coin = 1/2) : 
  ∀ n : ℕ, n > 0 → (coin_tosses : fin n → bool) (count_head : ℕ), (count_head = 0) ∨ (count_head = 1) :
  P(coin_tosses(n+1) = heads) = 1/2 :=
sorry

end coin_toss_probability_l406_406906


namespace linked_pairs_includes_one_l406_406941

def is_linked (m n : ℕ) : Prop :=
  m ∣ (3 * n + 1) ∧ n ∣ (3 * m + 1)

theorem linked_pairs_includes_one
  (a b c : ℕ)
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : is_linked a b) (h5 : is_linked b c) :
  1 ∈ {a, b, c} :=
sorry

end linked_pairs_includes_one_l406_406941


namespace a21_units_digit_is_6_l406_406009

noncomputable def a_seq : ℕ → ℕ
| 1 := 1
| 2 := 4
| 3 := 10
| (n+1) := if n < 3 then a_seq (n+1) else
  let a_np1 := a_seq (n + 1),
      a_n := a_seq n,
      a_nm1 := a_seq (n - 1),
      a_np2 := a_seq (n + 2) in
  /- This equation corresponds to the condition given in the problem statement -/
  if h : 2 ≤ n then
    let rel := a_np1^2 - 2 * a_n^2 = a_n * a_np2 - 2 * a_nm1 * a_np1 in
    a_np2
  else
    sorry

theorem a21_units_digit_is_6 : (a_seq 21) % 10 = 6 := sorry

end a21_units_digit_is_6_l406_406009


namespace solve_quadratic_substitution_l406_406119

theorem solve_quadratic_substitution (x : ℝ) : 
  (x^2 + x)^2 - 4*(x^2 + x) - 12 = 0 ↔ x = -3 ∨ x = 2 := 
by sorry

end solve_quadratic_substitution_l406_406119


namespace coplanar_of_linear_combination_l406_406903

variables {A B C D : Type*}
variables {P Q R S : A → A → A} -- Considering vectors AB, BC, BD as variables
variables {λ μ : ℝ}

def collinear (P Q R : A) : Prop :=
∃ k : ℝ, ∃ l : ℝ, Q = k • P + l • R

theorem coplanar_of_linear_combination (P Q R S : A) (λ μ : ℝ) (h: R = λ • Q + μ • S) :
  ∃ k_A B C D : ℝ, collinear P Q R ∧ collinear Q R S ∧ collinear R S P :=
sorry

end coplanar_of_linear_combination_l406_406903


namespace product_less_than_e_l406_406363

open Real

-- Define the function f(x)
def f (x : ℝ) : ℝ := log x - x

-- Define the sequence a_n
def a (n : ℕ) : ℝ := 1 + 1 / 2^(n + 1)

-- State the proof problem
theorem product_less_than_e (n : ℕ) : (∏ k in Finset.range(n + 1), a k) < exp 1 := by
  sorry

end product_less_than_e_l406_406363


namespace pages_revised_twice_theorem_l406_406872

noncomputable def pages_revised_twice (total_pages : ℕ) (cost_per_page : ℕ) (revision_cost_per_page : ℕ) 
                                      (pages_revised_once : ℕ) (total_cost : ℕ) : ℕ :=
  let pages_revised_twice := (total_cost - (total_pages * cost_per_page + pages_revised_once * revision_cost_per_page)) 
                             / (revision_cost_per_page * 2)
  pages_revised_twice

theorem pages_revised_twice_theorem : 
  pages_revised_twice 100 10 5 30 1350 = 20 :=
by
  unfold pages_revised_twice
  norm_num

end pages_revised_twice_theorem_l406_406872


namespace final_temperature_correct_l406_406968

-- Define the initial conditions
def initial_temperature : ℝ := 12
def decrease_per_hour : ℝ := 5
def time_duration : ℕ := 4

-- Define the expected final temperature
def expected_final_temperature : ℝ := -8

-- The theorem to prove that the final temperature after a given time is as expected
theorem final_temperature_correct :
  initial_temperature + (-decrease_per_hour * time_duration) = expected_final_temperature :=
by
  sorry

end final_temperature_correct_l406_406968


namespace find_x_given_conditions_l406_406087

variables {x y z w c k : ℝ}

theorem find_x_given_conditions
  (prop1 : x = k * y^3)
  (prop2 : y = c * (real.sqrt z) * w)
  (initial_condition : x = 5)
  (z_initial : z = 8)
  (w_initial : w = 1)
  (z_new : z = 36)
  (w_new : w = 2)
  : x = 540 * real.sqrt 3 :=
  sorry

end find_x_given_conditions_l406_406087


namespace capacity_of_new_bucket_l406_406563

def number_of_old_buckets : ℕ := 26
def capacity_of_old_bucket : ℝ := 13.5
def total_volume : ℝ := number_of_old_buckets * capacity_of_old_bucket
def number_of_new_buckets : ℕ := 39

theorem capacity_of_new_bucket :
  total_volume / number_of_new_buckets = 9 :=
sorry

end capacity_of_new_bucket_l406_406563


namespace integer_satisfying_condition_l406_406943

-- Define the condition for a polynomial p(x) to satisfy condition n
def satisfies_condition (p : ℝ[X]) (n : ℤ) : Prop :=
  (degree p = 2000) ∧
  (p.coeff 0 ≠ 0) ∧
  (∃ perm : List ℝ → List ℝ, 
    permutation perm ∧ 
    (perm p.coeffs = p.coeffs ∨ (∃ i j, i ≠ j ∧ perm (swap i j p.coeffs) = p.coeffs)))

-- Define the theorem to prove that n can only be 0 or 1
theorem integer_satisfying_condition (n : ℤ) :
  (∃ p : ℝ[X], satisfies_condition p n) ↔ n = 0 ∨ n = 1 :=
sorry

end integer_satisfying_condition_l406_406943


namespace vec_dot_product_l406_406710

variables (u v w : ℝ^3)
variables (h1 : ∥u∥ = 2) (h2 : ∥v∥ = 2) (h3 : ∥u + v∥ = 2 * real.sqrt 2)
variables (h4 : w - 2 * u - v = 2 * (u × v))

theorem vec_dot_product : v ∘ w = 4 :=
sorry

end vec_dot_product_l406_406710


namespace gradeA_frequency_estimated_gradeA_in_population_l406_406035

-- Definitions based on conditions
def isGradeA (m : ℕ) : Prop := m ≥ 10

def sample_data : List ℕ := [11, 10, 6, 15, 9, 16, 13, 12, 0, 8, 2, 8, 10, 17, 6, 13, 7, 5, 7, 3, 12, 10, 7, 11, 3, 6, 8, 14, 15, 12]

def total_sample_size : ℕ := 30
def total_population_size : ℕ := 1000

-- Prove the frequency of Grade A
theorem gradeA_frequency :
  (sample_data.filter isGradeA).length = 15 ∧
  (sample_data.filter isGradeA).length / total_sample_size = 1 / 2 := by
  sorry

-- Prove the estimation based on the frequency
theorem estimated_gradeA_in_population :
  (sample_data.filter isGradeA).length / total_sample_size = 1 / 2 →
  1000 * ((sample_data.filter isGradeA).length / total_sample_size) = 500 := by
  sorry

end gradeA_frequency_estimated_gradeA_in_population_l406_406035


namespace sin_alpha_value_l406_406357

theorem sin_alpha_value (α : ℝ) (x : ℝ) (h1 : α > π / 2 ∧ α < π) 
  (h2 : cos α = ((1 / 5) * x))
  (h3 : sin α = 4 / 5) : sin α = 4 / 5 :=
  by
    sorry

end sin_alpha_value_l406_406357


namespace rise_in_water_level_l406_406548

theorem rise_in_water_level (edge : ℝ) (length : ℝ) (width : ℝ)
  (h_edge : edge = 12) (h_length : length = 20) (h_width : width = 15) :
  let V_cube := edge ^ 3
  let A_base := length * width
  let h_increase := V_cube / A_base
  h_increase = 5.76 :=
by
  -- Definitions to use:
  let V_cube := edge ^ 3
  let A_base := length * width
  let h_increase := V_cube / A_base
  -- Conditions
  have hV_cube : V_cube = 12 ^ 3 := by rw [h_edge]; refl
  have hA_base : A_base = 20 * 15 := by rw [h_length, h_width]; refl
  show h_increase = 5.76
  sorry -- proof steps are omitted

end rise_in_water_level_l406_406548


namespace min_value_complex_l406_406803

open complex

/-- Let z be a complex number such that |z - 3 - 2i| = 7.
    Find the minimum value of |z + 1 + i|^2 + |z - 7 - 3i|^2. -/
theorem min_value_complex (z : ℂ) (h : abs (z - (3 + 2*I)) = 7) :
  ∃ z, abs (z - (3 + 2*I)) = 7 ∧ (abs (z + 1 + I)^2 + abs (z - (7 + 3*I))^2) = 174 :=
sorry

end min_value_complex_l406_406803


namespace arccos_sin_3_l406_406289

theorem arccos_sin_3 : Real.arccos (Real.sin 3) = (Real.pi / 2) + 3 := 
by
  sorry

end arccos_sin_3_l406_406289


namespace least_number_of_gumballs_to_ensure_five_of_same_color_l406_406940

theorem least_number_of_gumballs_to_ensure_five_of_same_color
  (red white blue green : ℕ)
  (h_red : red = 12)
  (h_white : white = 10)
  (h_blue : blue = 9)
  (h_green : green = 8) :
  ∃ n, n = 17 ∧ ∀ picks : Fin n → Fin (red + white + blue + green), 
    ∃ color, (picks.filter (λ pick, pick ∈ color)).count ≥ 5 := 
sorry

end least_number_of_gumballs_to_ensure_five_of_same_color_l406_406940


namespace expression_equals_one_l406_406978

noncomputable def compute_expression : ℝ :=
  (1 - 1 / Real.cos (30 * Real.pi / 180)) *
  (1 + 1 / Real.sin (60 * Real.pi / 180)) *
  (1 - 1 / Real.sin (30 * Real.pi / 180)) *
  (1 + 1 / Real.cos (60 * Real.pi / 180))

theorem expression_equals_one : compute_expression = 1 :=
by
  sorry

end expression_equals_one_l406_406978


namespace cutlery_total_l406_406183

theorem cutlery_total (forks knives spoons teaspoons : ℕ) 
  (h1 : forks = 6)
  (h2 : knives = forks + 9)
  (h3 : spoons = 2 * knives)
  (h4 : teaspoons = forks / 2) :
  let total_pieces := forks + knives + spoons + teaspoons + 8 in
  total_pieces = 62 :=
by
  sorry

end cutlery_total_l406_406183


namespace parabola_coeff_a_l406_406143

theorem parabola_coeff_a 
  (a b c : ℤ) 
  (vertex : (-2, 3))
  (point : (1, 6))
  (h_vertex : ∀ x, a * (x + 2) ^ 2 + 3 = a * x ^ 2 + b * x + c) 
  (h_point : ∀ x, y, y = a * (x + 2) ^ 2 + 3 → y = 6 → x = 1) :
  a = 1 / 3 := 
sorry

end parabola_coeff_a_l406_406143


namespace final_weight_of_statue_is_correct_l406_406589

noncomputable def final_statue_weight (initial_weight : ℝ) : ℝ :=
let weight_after_first_week := initial_weight - 0.35 * initial_weight in
let weight_after_second_week := weight_after_first_week - 0.2 * weight_after_first_week in
let weight_after_third_week := 
  let day1 := weight_after_second_week - 0.05 * weight_after_second_week in
  let day2 := day1 - 0.05 * day1 in
  let day3 := day2 - 0.05 * day2 in
  let day4 := day3 - 0.05 * day3 in
  let day5 := day4 - 0.05 * day4 in
  day5 in
let weight_after_erosion := weight_after_third_week - 0.02 * weight_after_third_week in
let weight_after_fourth_week := weight_after_erosion - 0.08 * weight_after_erosion in
let weight_of_final_statue := weight_after_fourth_week - 0.25 * weight_after_fourth_week in
weight_of_final_statue

theorem final_weight_of_statue_is_correct : 
  final_statue_weight 500 ≈ 136.04 :=
by
  sorry

end final_weight_of_statue_is_correct_l406_406589


namespace birds_twigs_and_trips_l406_406566

-- Define the conditions for the birds and the twigs they need
variables (first_twigs: ℕ) (first_weave: ℕ) (first_carry: ℕ) (first_tree_ratio: ℚ)
          (second_twigs: ℕ) (second_weave: ℕ)
          (third_twigs: ℕ) (third_weave: ℕ)

def twigs_needed_by_first_bird := first_twigs * first_weave
def first_bird_twigs_left := (twigs_needed_by_first_bird : ℚ) * (1 - first_tree_ratio)

-- Ensure integer calculation for trips
noncomputable def trips_first_bird := nat_ceil (first_bird_twigs_left / first_carry)

#check (trips_first_bird : ℕ)
          
def twigs_needed_by_second_bird := second_twigs * second_weave
def twigs_needed_by_third_bird := third_twigs * third_weave

def total_twigs := twigs_needed_by_first_bird + twigs_needed_by_second_bird + twigs_needed_by_third_bird

theorem birds_twigs_and_trips : 
  total_twigs first_twigs first_weave second_twigs second_weave third_twigs third_weave = 232 ∧ 
  trips_first_bird first_twigs first_weave first_carry first_tree_ratio = 16 :=
by {
  -- Use the conditions provided to prove both equalities
  sorry
}

end birds_twigs_and_trips_l406_406566


namespace ratio_of_ages_l406_406111

def rupert_candles : ℕ := 35
def peter_candles : ℕ := 10
def gcd (a b : ℕ) : ℕ := Nat.gcd a b  -- Define a gcd function to compute greatest common divisor

-- Statement to prove the ratio:
theorem ratio_of_ages (h_same_birthday : True): 
  rupert_candles / gcd rupert_candles peter_candles = 7 ∧ 
  peter_candles / gcd rupert_candles peter_candles = 2 := by
  sorry

end ratio_of_ages_l406_406111


namespace decreasing_log_func_range_of_a_l406_406483

theorem decreasing_log_func_range_of_a :
  ∀ (a : ℝ), (∀ x, 0 ≤ x ∧ x ≤ 2 → 0 < a ∧ 1 < a ∧ ∀ b, f b = log a (6 - a * b)) → 1 < a ∧ a < 3 := by
  sorry

end decreasing_log_func_range_of_a_l406_406483


namespace shaded_area_is_one_third_l406_406948

noncomputable def fractional_shaded_area : ℕ → ℚ
| 0 => 1 / 4
| n + 1 => (1 / 4) * fractional_shaded_area n

theorem shaded_area_is_one_third : (∑' n, fractional_shaded_area n) = 1 / 3 := 
sorry

end shaded_area_is_one_third_l406_406948


namespace find_angle_BMC_l406_406274

theorem find_angle_BMC 
  (A B C M : Type)
  [InnerProductSpace ℝ (triangle A B C)]
  [AngleType ℝ A B C]
  (BAC BCA : ℝ) (MCA MAC : ℝ)
  (h1 : BAC = 44)
  (h2 : BCA = 44)
  (h3 : MAC = 16)
  (h4 : MCA = 30) :
  ∠ BMC = 150 :=
by
  sorry

end find_angle_BMC_l406_406274


namespace neg_exists_eq_forall_ne_l406_406863

variable (x : ℝ)

theorem neg_exists_eq_forall_ne : (¬ ∃ x : ℝ, x^2 - 2 * x = 0) ↔ ∀ x : ℝ, x^2 - 2 * x ≠ 0 := by
  sorry

end neg_exists_eq_forall_ne_l406_406863


namespace distance_PQ_l406_406423

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def distance (p q : Point3D) : ℝ :=
  real.sqrt ((p.x - q.x) ^ 2 + (p.y - q.y) ^ 2 + (p.z - q.z) ^ 2)

noncomputable def midpoint (p q : Point3D) : Point3D :=
  ⟨ (p.x + q.x) / 2, (p.y + q.y) / 2, (p.z + q.z) / 2 ⟩

def A : Point3D := ⟨0, 0, 0⟩
def B : Point3D := ⟨1, 0, 0⟩
def C : Point3D := ⟨1, 1, 0⟩
def D : Point3D := ⟨0, 1, 0⟩

def A' : Point3D := ⟨0, 0, 12⟩
def B' : Point3D := ⟨1, 0, 20⟩
def C' : Point3D := ⟨1, 1, 12⟩
def D' : Point3D := ⟨0, 1, 20⟩

def P : Point3D := midpoint A' C'
def Q : Point3D := midpoint B' D'

theorem distance_PQ : distance P Q = 8 := 
by
  sorry

end distance_PQ_l406_406423


namespace find_number_of_folders_l406_406057

theorem find_number_of_folders :
  let price_pen := 1
  let price_notebook := 3
  let price_folder := 5
  let pens_bought := 3
  let notebooks_bought := 4
  let bill := 50
  let change := 25
  let total_cost_pens_notebooks := pens_bought * price_pen + notebooks_bought * price_notebook
  let amount_spent := bill - change
  let amount_spent_on_folders := amount_spent - total_cost_pens_notebooks
  let number_of_folders := amount_spent_on_folders / price_folder
  number_of_folders = 2 :=
by
  sorry

end find_number_of_folders_l406_406057


namespace chameleons_cannot_be_one_color_l406_406817

theorem chameleons_cannot_be_one_color 
  (B W R : ℕ) 
  (hB : B = 800) 
  (hW : W = 220) 
  (hR : R = 1003) 
  (hSum : B + W + R = 2023) : 
  ¬ ∃ c, (c = B + W + R) ∈ {B, W, R} :=
by 
  sorry

end chameleons_cannot_be_one_color_l406_406817


namespace range_of_b_for_non_monotonicity_l406_406696

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := - (1 / 2) * x^2 + b * Real.log x
def f_prime (b : ℝ) (x : ℝ) : ℝ := -x + b / x

theorem range_of_b_for_non_monotonicity :
  ∀ (b : ℝ), (∃ x ∈ Set.Icc (1:ℝ) (2:ℝ), ¬Monotone (λ y, f b y)) ↔ b ∈ Set.Ioo (-1:ℝ) 4 :=
sorry

end range_of_b_for_non_monotonicity_l406_406696


namespace number_of_triangles_formed_l406_406242

theorem number_of_triangles_formed (dots_on_AB dots_on_BC dots_on_CA : ℕ) 
(h_AB : dots_on_AB = 2) 
(h_BC : dots_on_BC = 3) 
(h_CA : dots_on_CA = 7) 
: 
  ( let total_dots := 3 + dots_on_AB + dots_on_BC + dots_on_CA in
    let total_combinations := nat.choose total_dots 3 in
    let collinear_AB := nat.choose (dots_on_AB + 2) 3 in
    let collinear_BC := nat.choose (dots_on_BC + 2) 3 in
    let collinear_CA := nat.choose (dots_on_CA + 2) 3 in
    let total_collinear := collinear_AB + collinear_BC + collinear_CA in
    total_combinations - total_collinear ) = 357 :=
by
  sorry

end number_of_triangles_formed_l406_406242


namespace selection_schemes_count_l406_406664

theorem selection_schemes_count :
  let P := {p1, p2, p3, p4, p5, p6 : Type} in
  let S := {bifeeng_gorge, mengding_mountain, laba_river, longcang_gorge : Type} in
  (∀ p ∈ P, ∃! s ∈ S, ∃ s' ∈ S, s ≠ s') ∧
  (∀ s ∈ S, ∃! p ∈ P, ∃ p' ∈ P, p ≠ p') ∧
  ¬(p1 = longcang_gorge ∨ p2 = longcang_gorge) →
  num_selection_schemes P S = 240 :=
sorry

end selection_schemes_count_l406_406664


namespace find_equation_of_line_l406_406883

theorem find_equation_of_line
  (m b : ℝ) 
  (h1 : ∃ k : ℝ, (k^2 - 2*k + 3 = k*m + b ∧ ∃ d : ℝ, d = 4) 
        ∧ (4*m - k^2 + 2*m*k - 3 + b = 0)) 
  (h2 : 8 = 2*m + b)
  (h3 : b ≠ 0) 
  : y = 8 :=
by 
  sorry

end find_equation_of_line_l406_406883


namespace parametric_to_standard_l406_406296

theorem parametric_to_standard (t a b x y : ℝ)
(h1 : x = (a / 2) * (t + 1 / t))
(h2 : y = (b / 2) * (t - 1 / t)) :
  (x^2 / a^2) - (y^2 / b^2) = 1 :=
by
  sorry

end parametric_to_standard_l406_406296


namespace ellipse_and_line_eq_l406_406345

noncomputable def ellipse_eq (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (a > b ∧ b > 0 ∧ (x^2 / a^2) + (y^2 / b^2) = 1) → (x^2 / 3) + (y^2 / 2) = 1

noncomputable def line_eq (k x y : ℝ) : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ), 
  (x1 + x2 = 6 * k^2 / (2 + 3 * k^2) ∧ y1 + y2 = -4 * k / (2 + 3 * k^2)) → 
  (y = k * (x - 1)) →
  (P : ℝ × ℝ) (x0 y0 : ℝ), 
  P = (x0, y0) →
  x0 = x1 + x2 →
  y0 = y1 + y2 →
  ((x0^2 / 3) + (y0^2 / 2) = 1) →
  k = ±sqrt(2)

theorem ellipse_and_line_eq (a b k x y : ℝ) :
  (a > b ∧ b > 0 ∧ abs((sqrt 3) / 3) = ((sqrt 3) / 3) ∧ a = sqrt(3) * 1 ∧ 2 * a + 2 * 1 = 4 * sqrt(3) ∧ b^2 = a^2 - 1^2) →
  ellipse_eq a b ∧ line_eq k x y :=
by
  split
  · sorry
  · sorry

end ellipse_and_line_eq_l406_406345


namespace vertex_of_f_C_l406_406543

def f_A (x : ℝ) : ℝ := (x + 4) ^ 2 - 3
def f_B (x : ℝ) : ℝ := (x + 4) ^ 2 + 3
def f_C (x : ℝ) : ℝ := (x - 4) ^ 2 - 3
def f_D (x : ℝ) : ℝ := (x - 4) ^ 2 + 3

theorem vertex_of_f_C : ∃ (h k : ℝ), h = 4 ∧ k = -3 ∧ ∀ x, f_C x = (x - h) ^ 2 + k :=
by
  sorry

end vertex_of_f_C_l406_406543


namespace sequence_arithmetic_sum_b_n_l406_406812

-- Definitions based on given conditions
variable (a_n : ℕ → ℝ)
variable (T : ℕ → ℝ) (b : ℕ → ℝ)

-- Condition: T_n = 2 - 2a_n
axiom T_def : ∀ n, T n = 2 - 2 * a_n n

-- Condition: b_n = sqrt(2) / (sqrt(1/T_n) + sqrt(1/T_(n+1)))
axiom b_def : ∀ n, b n = √2 / (√(1 / T n) + √(1 / T (n + 1)))

-- Question 1: Prove sequence {1/T_n} is arithmetic with first term 3/2 and common difference 1/2
theorem sequence_arithmetic : ∃ a d, ∀ n, (1 / T n) = a + n * d :=
sorry

-- Question 2: Prove sum of first n terms of sequence {b_n} is 2(√(n+3) - √3)
theorem sum_b_n (S : ℕ → ℝ) : ∀ n, (S n) = ∑ i in finset.range n, b i → S n = 2 * (√(n + 3) - √3) :=
sorry

end sequence_arithmetic_sum_b_n_l406_406812


namespace cover_superset_cover_idempotent_cover_superset_implication_cond_from_goals_l406_406844

variables {X Y : set ℝ}
variable (cover : set ℝ → set ℝ)
-- Conditions
axiom cover_cond : ∀ (X Y : set ℝ), cover (X ∪ Y) ⊇ cover (cover X) ∪ cover Y ∪ Y

-- Proof Goals
theorem cover_superset : ∀ (X : set ℝ), cover X ⊇ X :=
sorry

theorem cover_idempotent : ∀ (X : set ℝ), cover (cover X) = cover X :=
sorry

theorem cover_superset_implication : ∀ (X Y : set ℝ), X ⊇ Y → cover X ⊇ cover Y :=
sorry

theorem cond_from_goals : (∀ (X : set ℝ), cover X ⊇ X) →
                           (∀ (X : set ℝ), cover (cover X) = cover X) →
                           (∀ (X Y : set ℝ), X ⊇ Y → cover X ⊇ cover Y) →
                           (∀ (X Y : set ℝ), cover (X ∪ Y) ⊇ cover (cover X) ∪ cover Y ∪ Y) :=
sorry

end cover_superset_cover_idempotent_cover_superset_implication_cond_from_goals_l406_406844


namespace area_of_rectangle_l406_406945

theorem area_of_rectangle (w d : ℝ) (h : 3 * w = length_of_rectangle) :
  w^2 + (3 * w)^2 = d^2 → area_of_rectangle = 3 / 10 * d^2 :=
begin
  sorry,
end

end area_of_rectangle_l406_406945


namespace emily_entire_order_cost_l406_406307

def curtain_price : Float := 30.00
def curtain_quantity : Int := 2
def wall_print_price : Float := 15.00
def wall_print_quantity : Int := 9
def discount_rate : Float := 0.10
def sales_tax_rate : Float := 0.08
def installation_service_fee : Float := 50.00

theorem emily_entire_order_cost : 
  let total_purchase_pre_discount := (curtain_quantity * curtain_price) + (wall_print_quantity * wall_print_price)
  let discount_amount := total_purchase_pre_discount * discount_rate
  let discounted_price := total_purchase_pre_discount - discount_amount
  let sales_tax := discounted_price * sales_tax_rate
  let total_cost_before_installation := discounted_price + sales_tax
  let entire_order_cost := total_cost_before_installation + installation_service_fee
  entire_order_cost = 239.54 :=
by
  sorry

end emily_entire_order_cost_l406_406307


namespace number_of_triangles_l406_406244

theorem number_of_triangles (dots_on_AB dots_on_BC dots_on_CA: ℕ) (h1: dots_on_AB = 2) (h2: dots_on_BC = 3) (h3: dots_on_CA = 7) :
  let total_dots := 3 + dots_on_AB + dots_on_BC + dots_on_CA in
  let total_combinations := Nat.choose total_dots 3 in
  let collinear_combinations := Nat.choose (dots_on_AB + 2) 3 + Nat.choose (dots_on_BC + 2) 3 + Nat.choose (dots_on_CA + 2) 3 in
  total_combinations - collinear_combinations = 357 :=
by
  rw [h1, h2, h3]
  simp only [Nat.choose_eq_factorization, h1, h2, h3]
  -- Total dots
  have ht : total_dots = 15 := by norm_num
  -- Total combinations of choosing 3 dots from 15
  have ht_comb : Nat.choose total_dots 3 = 455 := by norm_num
  -- Collinear combinations
  have hc_AB : Nat.choose (dots_on_AB + 2) 3 = 4 := by norm_num
  have hc_BC : Nat.choose (dots_on_BC + 2) 3 = 10 := by norm_num
  have hc_CA : Nat.choose (dots_on_CA + 2) 3 = 84 := by norm_num
  have hc_total : hc_AB + hc_BC + hc_CA = 98 := by norm_num
  -- Final result
  have result : 455 - 98 = 357 := by norm_num
  exact result

end number_of_triangles_l406_406244


namespace parallel_conditions_l406_406071

-- Given: Definitions of points and circumferences, and their properties
variables {C1 C2 : Circle} {A B C P Q R S X Y Z : Point}

-- Given: Circumferences intersecting at points A and B
axiom h1 : intersects C1 C2 = {A, B}

-- Given: C on line AB with B between A and C
axiom h2 : collinear A B C ∧ between B A C

-- Given: CP and CQ are tangent to C1 and C2 respectively
axiom h3 : tangent C C1 P ∧ tangent C C2 Q

-- Given: P not inside C2 and Q not inside C1
axiom h4 : ¬inside C2 P ∧ ¬inside C1 Q

-- Given: Line PQ cuts C1 at R and C2 at S, both points different from P, Q, and B
axiom h5 : on_circle R C1 ∧ on_circle S C2 ∧ R ≠ P ∧ R ≠ Q ∧ R ≠ B ∧ S ≠ P ∧ S ≠ Q ∧ S ≠ B

-- Given: CR cuts C1 again at X and CS cuts C2 again at Y
axiom h6 : second_intersection CR C1 X ∧ second_intersection CS C2 Y

-- Given: Z is a point on line XY
axiom h7 : on_line Z X Y

-- Goal: Prove SZ is parallel to QX iff PZ is parallel to RX
theorem parallel_conditions : 
  parallel S Z Q X ↔ parallel P Z R X :=
sorry

end parallel_conditions_l406_406071


namespace least_multiple_greater_than_500_l406_406524

theorem least_multiple_greater_than_500 : ∃ n : ℕ, n > 0 ∧ 35 * n > 500 ∧ 35 * n = 525 :=
by
  sorry

end least_multiple_greater_than_500_l406_406524


namespace max_value_geometric_sequence_product_l406_406692

theorem max_value_geometric_sequence_product
  (a : ℕ → ℝ) -- sequence definition
  (q : ℝ) -- common ratio of the sequence
  (h_decreasing : ∀ n, a (n + 1) = a n * q) -- decreasing geometric sequence definition
  (h_q_lt_one : q < 1) -- q is less than 1
  (h_q_gt_zero : 0 < q) -- q is greater than 0
  (h_a2a7 : a 2 * a 7 = 1/2) -- condition a_2 * a_7 = 1/2
  (h_a3a6 : a 3 + a 6 = 9/4) -- condition a_3 + a_6 = 9/4)
  (n : ℕ) -- general term for the sequence
  : (∀ k, 1 ≤ k → k ≤ 2*n → a k) -- for each n, all terms up to a_{2n}
  → a 1 * a 2 * ... * a (2*n) ≤ 64 := -- product of the terms up to a_{2n} is max 64 
sorry -- proof omitted

end max_value_geometric_sequence_product_l406_406692


namespace price_decrease_is_9_5_percent_l406_406070

def price_last_month : ℝ := 9 / 6
def price_before_tax : ℝ := 8 / 7
def tax_per_notebook : ℝ := 0.50
def price_this_month : ℝ := price_before_tax + tax_per_notebook

def percent_decrease (old_price new_price : ℝ) : ℝ :=
  ((old_price - new_price) / old_price) * 100

theorem price_decrease_is_9_5_percent : percent_decrease price_last_month price_this_month = -9.5 :=
by sorry

end price_decrease_is_9_5_percent_l406_406070


namespace find_a_l406_406695

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (√3 * sin x * cos x + cos x ^ 2 + a)

theorem find_a (a : ℝ) :
  (∀ x : ℝ, x ∈ set.Icc (-π / 6) (π / 3) → f a x ≤ 1 + a + 1/2) ∧
  (∀ x : ℝ, x ∈ set.Icc (-π / 6) (π / 3) → f a x ≥ -1/2 + a + 1/2) →
  a = 0 :=
begin
  assume h,
  sorry -- Proof omitted
end

end find_a_l406_406695


namespace projection_correct_l406_406319

def vector3 := ℝ × ℝ × ℝ

def dot_product (u v : vector3) : ℝ :=
u.1 * v.1 + u.2 * v.2 + u.3 * v.3  -- Calculates the dot product of two 3D vectors

def projection (u v : vector3) : vector3 :=  -- Calculates the projection of u onto v
  let uv := dot_product u v in
  let vv := dot_product v v in
  (uv / vv * v.1, uv / vv * v.2, uv / vv * v.3)

def u : vector3 := (3, -2, 4)
def v : vector3 := (1, 1, 1)
def proj_u_v := (5/3, 5/3, 5/3)

theorem projection_correct :
  projection u v = proj_u_v :=
by
  sorry

end projection_correct_l406_406319


namespace range_of_m_l406_406094

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_m {
  h1 : ∀ x : ℝ, differentiable_at ℝ f x,
  h2 : ∀ x : ℝ, f x = 6 * x^2 - f (-x),
  h3 : ∀ x : ℝ, x < 0 → 2 * (f' x) + 1 < 12 * x,
  h4 : ∀ m : ℝ, f (m + 2) ≤ f (-2 * m) + 12 * m + 12 - 9 * m^2
} : set.Ici (-2/3) = {m : ℝ | f (m + 2) ≤ f (-2 * m) + 12 * m + 12 - 9 * m^2} :=
begin
  sorry
end

end range_of_m_l406_406094


namespace perimeter_triangle_COK_l406_406116

variables {A B C M K O : Point}
variables {AK CM : Line}

-- Given conditions
variables (angle_BMC_eq_angle_BKA : ∠ B M C = ∠ B K A)
variables (BM_eq_BK : BM = BK)
variables (AB_len : AB = 15)
variables (BK_len : BK = 8)
variables (CM_len : CM = 9)
variables (O_intersect_AK_CM : O ∈ AK ∧ O ∈ CM)

-- To prove: the perimeter of triangle COK is 16
theorem perimeter_triangle_COK (angle_BMC_eq_angle_BKA BM_eq_BK AB_len BK_len CM_len O_intersect_AK_CM : Prop) :
  perimeter (triangle C O K) = 16 :=
sorry

end perimeter_triangle_COK_l406_406116


namespace largest_among_given_l406_406626

def largest_number among (a b c d e : ℝ) : ℝ :=
  if a ≥ b ∧ a ≥ c ∧ a ≥ d ∧ a ≥ e then a
  else if b ≥ a ∧ b ≥ c ∧ b ≥ d ∧ b ≥ e then b
  else if c ≥ a ∧ c ≥ b ∧ c ≥ d ∧ c ≥ e then c
  else if d ≥ a ∧ d ≥ b ∧ d ≥ c ∧ d ≥ e then d
  else e

theorem largest_among_given : largest_number 0.995 0.9995 0.99 0.999 0.9959 = 0.9995 := by
  sorry

end largest_among_given_l406_406626


namespace max_shui_value_l406_406049

open BigOperators

/-- Each Chinese character represents a unique digit between 1 and 8. 
Given the following:
  1. 2 * jx + x + li = 19
  2. li + ke + ba + shan = 19
  3. shan + qiong + shui + jx = 19
  4. jx > shan > li
Prove that the digit represented by shui is at most 7. -/
theorem max_shui_value (jx x li ke ba shan qiong shui : ℕ)
  (Hunique : (Finset.univ : Finset ℕ).filter (λ n, n ∈ {jx, x, li, ke, ba, shan, qiong, shui}).card = 8)
  (Hrange : ∀ n ∈ {jx, x, li, ke, ba, shan, qiong, shui}, n ≥ 1 ∧ n ≤ 8)
  (Hsum1 : 2 * jx + x + li = 19)
  (Hsum2 : li + ke + ba + shan = 19)
  (Hsum3 : shan + qiong + shui + jx = 19)
  (Horder : jx > shan ∧ shan > li) : 
  shui ≤ 7 := sorry

end max_shui_value_l406_406049


namespace gloopers_perimeter_l406_406392

-- Definitions based on conditions
def circle_radius : ℝ := 2
def mouth_angle : ℝ := 90 * (Real.pi / 180) -- converting 90 degrees to radians
def full_circle_angle : ℝ := 2 * Real.pi

-- Definition of the sector perimeter problem
theorem gloopers_perimeter : 
  let arc_length := (full_circle_angle - mouth_angle) / full_circle_angle * (2 * Real.pi * circle_radius)
  let perimeter := arc_length + 2 * circle_radius
  perimeter = 3 * Real.pi + 4 :=
by sorry

end gloopers_perimeter_l406_406392


namespace first_place_prize_is_200_l406_406944

-- Define the conditions from the problem
def total_prize_money : ℤ := 800
def num_winners : ℤ := 18
def second_place_prize : ℤ := 150
def third_place_prize : ℤ := 120
def fourth_to_eighteenth_prize : ℤ := 22
def fourth_to_eighteenth_winners : ℤ := num_winners - 3

-- Define the amount awarded to fourth to eighteenth place winners
def total_fourth_to_eighteenth_prize : ℤ := fourth_to_eighteenth_winners * fourth_to_eighteenth_prize

-- Define the total amount awarded to second and third place winners
def total_second_and_third_prize : ℤ := second_place_prize + third_place_prize

-- Define the total amount awarded to second to eighteenth place winners
def total_second_to_eighteenth_prize : ℤ := total_fourth_to_eighteenth_prize + total_second_and_third_prize

-- Define the amount awarded to first place
def first_place_prize : ℤ := total_prize_money - total_second_to_eighteenth_prize

-- Statement for proof required
theorem first_place_prize_is_200 : first_place_prize = 200 :=
by
  -- Assuming the conditions are correct
  sorry

end first_place_prize_is_200_l406_406944


namespace last_two_digits_sum_of_factorials_1_to_100_l406_406893

-- Define a function to compute factorial
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define a function to get last two digits of a number
def last_two_digits (n : ℕ) : ℕ := n % 100

-- Define the sum of factorials from 1! to 9!
def sum_factorials_upto_9 : ℕ := (List.sum $ List.map factorial [1, 2, 3, 4, 5, 6, 7, 8, 9])

-- State the theorem
theorem last_two_digits_sum_of_factorials_1_to_100 : 
  last_two_digits (List.sum (List.map factorial (List.range 100).succ)) = 13 :=
by
  /-
    Here, we'd have the proof steps which would assert:
    - The sum contribution of factorials of numbers from 10 to 99 is zero to the last two digits.
    - Calculate sum of the first 9 factorials and take modulo 100.
  -/
  sorry

end last_two_digits_sum_of_factorials_1_to_100_l406_406893


namespace solve_a_l406_406698

noncomputable def f (a : ℝ) (f'_2 : ℝ) (x : ℝ) : ℝ := a*x^3 + f'_2*x^2 + 3
noncomputable def f' (a : ℝ) (f'_2 : ℝ) (x : ℝ) : ℝ := 3*a*x^2 + 2*f'_2*x

theorem solve_a (a f'_2 : ℝ) (h : f' a f'_2 1 = -5) : a = 1 :=
by
  -- using the given condition f'(2) = -4a
  have h_f'_2 : f'_2 = -4*a := sorry
  -- substituting f'_2 in the equation f'(1) = -5
  rw [f'] at h
  rw h_f'_2 at h
  simp at h
  -- solving the equation 3a + 2(-4a) = -5
  exact sorry

end solve_a_l406_406698


namespace QT_squared_l406_406762

-- Definitions and conditions
def X : ℝ × ℝ := (0, 0)
def Y : ℝ × ℝ := (2 * Real.sqrt 3, 0)
def Z : ℝ × ℝ := (2 * Real.sqrt 3, 2 * Real.sqrt 3)
def W : ℝ × ℝ := (0, 2 * Real.sqrt 3)
def P : ℝ × ℝ := (Real.sqrt 3, 0)
def S : ℝ × ℝ := (0, Real.sqrt 3)
def PS_line (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 - x, x)
def Q : ℝ × ℝ := (2 * Real.sqrt 3, Real.sqrt 3 / 2)
def R : ℝ × ℝ := (Real.sqrt 3 / 2, 2 * Real.sqrt 3)
def QT : Prop := Q.1 = PS_line (Q.2)

-- Statement
theorem QT_squared :
  (∃ T : ℝ × ℝ, QT) →
  P.1 = Real.sqrt 3 ∧ P.2 = 0 ∧ 
  S.1 = 0 ∧ S.2 = Real.sqrt 3 ∧
  -- Each region has an area of 1.5
  0.5 * (Real.sqrt 3) * (Real.sqrt 3) = 1.5 ∧
  (∃ A1 : ℝ, (2 * Real.sqrt 3 * A1 / 2) * T.2 = 1.5) ∧
  (∃ A2 : ℝ, (0.5) * (2 * Real.sqrt 3 * A2) = 1.5) ∧
  (∃ A3 : ℝ, (2 * (2 * Real.sqrt 3) * A3 * 0.5) = 1.5) ∧
  -- Conclusion
  (∃ QT_length : ℝ, (QT_length ^ 2 = 3)): sorry

end QT_squared_l406_406762


namespace smallest_whole_number_l406_406898

theorem smallest_whole_number (m : ℕ) :
  m % 2 = 1 ∧
  m % 3 = 1 ∧
  m % 4 = 1 ∧
  m % 5 = 1 ∧
  m % 6 = 1 ∧
  m % 8 = 1 ∧
  m % 11 = 0 → 
  m = 1801 :=
by
  intros h
  sorry

end smallest_whole_number_l406_406898


namespace range_of_omega_l406_406006

noncomputable theory

def function_has_no_zeros (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x ∈ I, f x ≠ 0

def f (ω x : ℝ) : ℝ :=
  sin^2 (ω * x / 2) + 0.5 * sin (ω * x) - 0.5

theorem range_of_omega (ω : ℝ) (h : ω > 0) :
  (function_has_no_zeros (f ω) {x | π < x ∧ x < 2 * π}) ↔
  (ω ∈ (set.Ioo 0 (1/8) ∪ set.Icc (1/4) (5/8))) :=
sorry

end range_of_omega_l406_406006


namespace PQ_eq_QR_iff_angle_bisectors_meet_AC_l406_406073

variable {α : Type} [LinearOrderedField α]

structure Point (α : Type) : Type := (x y : α)

def is_cyclic (A B C D : Point α) : Prop := sorry

def projection (D : Point α) (l : Point α × Point α) : Point α := sorry

def is_on (p : Point α) (l : Point α × Point α) : Prop := sorry

def is_angle_bisector (A B C : Point α) (l : Point α × Point α) : Prop := sorry

theorem PQ_eq_QR_iff_angle_bisectors_meet_AC :
  ∀ (A B C D P Q R : Point α),
  is_cyclic A B C D →
  (P = projection D (B, C)) →
  (Q = projection D (C, A)) →
  (R = projection D (A, B)) →
  (dist P Q = dist Q R ↔ ∃ E, is_on E (A, C) ∧ is_angle_bisector A B C (B, E) ∧ is_angle_bisector A D C (D, E)) :=
by
  sorry

end PQ_eq_QR_iff_angle_bisectors_meet_AC_l406_406073


namespace triangle_ratio_problem_l406_406772

theorem triangle_ratio_problem
  (A B C D E F P Q : Type*)
  [is_line_segment D B C]
  [is_line_segment E A C]
  [is_line_segment F A B]
  [intersects AD CF P]
  [intersects BE CF Q]
  (AP_PD : ratio AP PD 3 2)
  (FQ_QC : ratio FQ QC 3 4) :
  AF_FB = 2 / 3 :=
by
  sorry

end triangle_ratio_problem_l406_406772


namespace part1_latest_time_not_late_part2_probability_late_exactly_one_day_part3_average_journey_time_l406_406211

section
variable (XiaoLi_walk_home_to_work : Type)
variable (arrive_no_later_than : XiaoLi_walk_home_to_work → Prop)
variable (intersections : List XiaoLi_walk_home_to_work)
variable (probability_red_light : XiaoLi_walk_home_to_work → ℝ)
variable (average_waiting_time_per_red_light : XiaoLi_walk_home_to_work → ℝ)
variable (time_without_red_lights : ℝ)

axiom red_light_independent : ∀ {x y : XiaoLi_walk_home_to_work}, x ≠ y → probability_red_light x = 1/2 → probability_red_light y = 1/2
axiom waiting_time_per_red_light : ∀ x, average_waiting_time_per_red_light x = 1
axiom no_red_light_time : time_without_red_lights = 10

def latest_time_not_late (x : XiaoLi_walk_home_to_work) : Prop :=
  ∃ t, arrive_no_later_than t ∧ probability_red_light t > 0.90

theorem part1_latest_time_not_late (x : XiaoLi_walk_home_to_work) :
  latest_time_not_late x → t = 7:47 := sorry

def probability_late_exactly_one_day (x : XiaoLi_walk_home_to_work) (time : ℝ) : ℝ :=
(C 2 1) * (5/16) * (11/16)

theorem part2_probability_late_exactly_one_day (x : XiaoLi_walk_home_to_work) :
  probability_late_exactly_one_day x 7:48 = 55/128 := sorry

def average_journey_time (x : XiaoLi_walk_home_to_work) : ℝ :=
10 * (1/16) + 11 * (1/4) + 12 * (3/8) + 13 * (1/4) + 14 * (1/16)

theorem part3_average_journey_time (x : XiaoLi_walk_home_to_work) :
  average_journey_time x = 12 := sorry

end

end part1_latest_time_not_late_part2_probability_late_exactly_one_day_part3_average_journey_time_l406_406211


namespace find_pair_min_rounds_l406_406825

theorem find_pair_min_rounds :
  (∃ N : ℕ, (∀ x y : ℕ, x ≤ 20 ∧ y ≤ 23 →
    (∃ (strategy : Π (round : ℕ), {a : ℕ × ℕ // a.1 ≤ 20 ∧ a.2 ≤ 23}),
      (∀ x y : ℕ, x ≤ 20 ∧ y ≤ 23 →
        (∃ (n : ℕ), n ≤ N ∧
          ∀ (m : ℕ), m < n →
            ∃ (response : bool), 
              (strategy m).val.1 ≤ x ∧ (strategy m).val.2 ≤ y ↔ response = tt))) ∧ N = 9))

end find_pair_min_rounds_l406_406825


namespace area_triangle_ABD_l406_406034

-- Define relevant points and configurations
variables (A B C D E S: Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited S]

-- Define the segments and conditions
variables (AC BD AE EC: ℝ)
variables (AB CD AD BC : ℝ)
variables (A1 B1 C1 D1: Type) [Inhabited A1] [Inhabited B1] [Inhabited C1] [Inhabited D1]

-- Conditions
axiom h1 : AC = 9
axiom h2 : AE < EC
axiom h3 : (2: ℝ) = 2 -- Regular hexagon with side length 2 formed by the plane intersection

-- Objective: Prove the area of triangle ABD is 4
theorem area_triangle_ABD : 
  let area := (1 / 2) * (AC / 2) * (BD / 2) in -- Example computation of area using symmetry and midsegment
  area = 4 :=
by
  sorry

end area_triangle_ABD_l406_406034


namespace time_after_3108_hours_l406_406139

/-- The current time is 3 o'clock. On a 12-hour clock, 
 what time will it be 3108 hours from now? -/
theorem time_after_3108_hours : (3 + 3108) % 12 = 3 := 
by
  sorry

end time_after_3108_hours_l406_406139


namespace simplify_and_evaluate_l406_406835

theorem simplify_and_evaluate (m : ℝ) (h : m = 5) :
  (m + 2 - (5 / (m - 2))) / ((3 * m - m^2) / (m - 2)) = - (8 / 5) :=
by
  sorry

end simplify_and_evaluate_l406_406835


namespace inequality_holds_l406_406828

theorem inequality_holds (a b : ℝ) (ha : 0 ≤ a) (ha' : a ≤ 1) (hb : 0 ≤ b) (hb' : b ≤ 1) : 
  a^5 + b^3 + (a - b)^2 ≤ 2 :=
sorry

end inequality_holds_l406_406828


namespace least_fraction_to_unity_l406_406900

theorem least_fraction_to_unity :
  (∑ n in Finset.range 20, (1 / ((n + 2) * (n + 3)))) + (sin^2 (x) / (22 * 23)) + (153 / 506) = 1 
  ∧ (0 ≤ x) ∧ (x ≤ π / 2) :=
sorry

end least_fraction_to_unity_l406_406900


namespace right_triangle_hypotenuse_l406_406188

noncomputable def hypotenuse_length (PQ PR QN MR : ℝ) :=
  let a := PQ in
  let b := PR in
  let PM := a / 4 in
  let MQ := 3 * a / 4 in
  let PN := b / 4 in
  let NR := 3 * b / 4 in
  let QN := 20 in
  let MR := 36 in
  sqrt (a^2 + b^2)

theorem right_triangle_hypotenuse :
  ∀ (PQ PR QN MR : ℝ),
    PQ = PQ → PR = PR → QN = 20 → MR = 36 →
    PM:MQ = 1:3 → PN:NR = 1:3 →
    hypotenuse_length PQ PR QN MR = 2 * sqrt 399 :=
by
  intros PQ PR QN MR hPQ hPR hQN hMR hRatio1 hRatio2
  sorry

end right_triangle_hypotenuse_l406_406188


namespace always_odd_l406_406427

theorem always_odd (p m : ℕ) (hp : p % 2 = 1) : (p^3 + 3*p*m^2 + 2*m) % 2 = 1 := 
by sorry

end always_odd_l406_406427


namespace candies_in_box_more_than_pockets_l406_406824

theorem candies_in_box_more_than_pockets (x : ℕ) : 
  let initial_pockets := 2 * x
  let pockets_after_return := 2 * (x - 6)
  let candies_returned_to_box := 12
  let total_candies_after_return := initial_pockets + candies_returned_to_box
  (total_candies_after_return - pockets_after_return) = 24 :=
by
  sorry

end candies_in_box_more_than_pockets_l406_406824


namespace final_price_lower_l406_406597

theorem final_price_lower (x : ℝ) (h₁ : x > 0) :
  (x * 1.02 * 0.98) < x :=
by {
  have h₂ : 1.02 * 0.98 = 1.0196, by norm_num,
  rw [mul_assoc, h₂],
  have h₃ : 1.0196 < 1, by norm_num,
  linarith,
}

end final_price_lower_l406_406597


namespace magnitude_sum_l406_406668

variables {R : Type*} [inner_product_space ℝ R] (a b c : R)

noncomputable
def magnitude (v : R) : ℝ := Real.sqrt(⟪v, v⟫)

axiom a_perp_b : ⟪a, b⟫ = 0
axiom a_dot_c_zero : ⟪a, c⟫ = 0
axiom b_dot_c_zero : ⟪b, c⟫ = 0
axiom mag_a : magnitude a = 1
axiom mag_b : magnitude b = 2
axiom mag_c : magnitude c = 3

theorem magnitude_sum : magnitude (a + b + c) = Real.sqrt 14 :=
by
  sorry

end magnitude_sum_l406_406668


namespace five_digit_palindromes_count_l406_406717

theorem five_digit_palindromes_count : 
  ∃ (a b c : Fin 10), (a ≠ 0) ∧ (∃ (count : Nat), count = 9 * 10 * 10 ∧ count = 900) :=
by
  sorry

end five_digit_palindromes_count_l406_406717


namespace sum_of_areas_of_circles_l406_406170

-- Definitions of conditions
def r : ℝ := by sorry  -- radius of the circle at vertex A
def s : ℝ := by sorry  -- radius of the circle at vertex B
def t : ℝ := by sorry  -- radius of the circle at vertex C

axiom sum_radii_r_s : r + s = 6
axiom sum_radii_r_t : r + t = 8
axiom sum_radii_s_t : s + t = 10

-- The statement we want to prove
theorem sum_of_areas_of_circles : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by
  -- Use given axioms and properties of the triangle and circles
  sorry

end sum_of_areas_of_circles_l406_406170


namespace beth_wins_with_starting_configuration_l406_406966

-- Define the nim-values of walls with lengths 1 to 6
def nim_value : ℕ → ℕ
| 1 := 1
| 2 := 2
| 3 := 3
| 4 := 1
| 5 := 4
| 6 := 3
| _ := sorry -- values for walls greater than 6 are not needed for this proof

-- Define the nim-value of a configuration as the xor of nim-values of individual walls
def configuration_nim_value (walls : list ℕ) : ℕ :=
walls.foldl (λ acc wall_length, acc xor nim_value wall_length) 0

-- The main proof statement
theorem beth_wins_with_starting_configuration :
  configuration_nim_value [6, 2, 1] = 0 :=
by
  -- Prove that the nim-value of the configuration (6,2,1) is 0
  sorry

end beth_wins_with_starting_configuration_l406_406966


namespace xy_product_l406_406377

theorem xy_product (x y : ℝ) (h : sqrt (x - 1) + (y - 2)^2 = 0) : x * y = 2 := 
sorry

end xy_product_l406_406377


namespace average_weight_of_children_l406_406137

theorem average_weight_of_children
  (n_boys n_girls : ℕ)
  (avg_weight_boys avg_weight_girls : ℝ)
  (hb : n_boys = 8)
  (hg : n_girls = 6)
  (avg_b : avg_weight_boys = 160)
  (avg_g : avg_weight_girls = 110) : 
  let total_children := n_boys + n_girls,
      total_weight := n_boys * avg_weight_boys + n_girls * avg_weight_girls,
      avg_weight_children := total_weight / total_children
  in avg_weight_children = 139 := 
sorry

end average_weight_of_children_l406_406137


namespace fourth_person_height_l406_406881

variables (H1 H2 H3 H4 : ℝ)

theorem fourth_person_height :
  H2 = H1 + 2 →
  H3 = H2 + 3 →
  H4 = H3 + 6 →
  H1 + H2 + H3 + H4 = 288 →
  H4 = 78.5 :=
by
  intros h2_def h3_def h4_def total_height
  -- Proof steps would follow here
  sorry

end fourth_person_height_l406_406881


namespace percent_twelve_equals_eighty_four_l406_406886

theorem percent_twelve_equals_eighty_four (x : ℝ) (h : (12 / 100) * x = 84) : x = 700 :=
by
  sorry

end percent_twelve_equals_eighty_four_l406_406886


namespace VishalInvestedMoreThanTrishulBy10Percent_l406_406892

variables (R T V : ℝ)

-- Given conditions
def RaghuInvests (R : ℝ) : Prop := R = 2500
def TrishulInvests (R T : ℝ) : Prop := T = 0.9 * R
def TotalInvestment (R T V : ℝ) : Prop := V + T + R = 7225
def PercentageInvestedMore (T V : ℝ) (P : ℝ) : Prop := P * T = V - T

-- Main theorem to prove
theorem VishalInvestedMoreThanTrishulBy10Percent (R T V : ℝ) (P : ℝ) :
  RaghuInvests R ∧ TrishulInvests R T ∧ TotalInvestment R T V → PercentageInvestedMore T V P → P = 0.1 :=
by
  intros
  sorry

end VishalInvestedMoreThanTrishulBy10Percent_l406_406892


namespace g_of_25_l406_406090

def g (x : ℝ) : ℝ :=
  if x < 10 then x^2 + 6 * x + 9 else x - 20

theorem g_of_25 : g (g (g 25)) = 44 := by
  sorry

end g_of_25_l406_406090


namespace one_point_inside_circle_l406_406618

theorem one_point_inside_circle {A B C D : Point} [plane Point]
  (h_no_three_collinear : ¬ collinear A B C ∧ ¬ collinear A B D ∧ ¬ collinear A C D ∧ ¬ collinear B C D)
  (h_circle_cond : ∀ P Q R S : Point, S ∉ circle_through P Q R) :
  ∃ P Q R S : Point, S ∈ inside (circle_through P Q R) :=
by
  sorry

end one_point_inside_circle_l406_406618


namespace bandit_conflicts_l406_406510

theorem bandit_conflicts :
  (∃ (bandits : Finset ℕ), bandits.card = 50 ∧
    (∀ {x y : ℕ}, x ≠ y → x ∈ bandits → y ∈ bandits → encountered x y) ∧
    (∀ x y : ℕ, x ≠ y → encountered x y → encountered y x) ∧
    (∀ x : ℕ, x ∈ bandits → (number_of_conflicts x bandits) ≥ 8)) :=
sorry

end bandit_conflicts_l406_406510


namespace gasoline_reduction_l406_406912

theorem gasoline_reduction
  (P Q : ℝ)
  (h1 : 0 < P)
  (h2 : 0 < Q)
  (price_increase_percent : ℝ := 0.25)
  (spending_increase_percent : ℝ := 0.05)
  (new_price : ℝ := P * (1 + price_increase_percent))
  (new_total_cost : ℝ := (P * Q) * (1 + spending_increase_percent)) :
  100 - (100 * (new_total_cost / new_price) / Q) = 16 :=
by
  sorry

end gasoline_reduction_l406_406912


namespace h_monotonic_intervals_l406_406346

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (2 * x - 1)
noncomputable def g (x : ℝ) : ℝ := x - 1
noncomputable def h (x : ℝ) : ℝ := f x / g x

theorem h_monotonic_intervals :
  (∀ x, x ∈ Ioo (-∞) 0 → Monotone.intervalOn Ioo (-∞) 0 h) ∧
  (∀ x, x ∈ Ioo 0 1 → MonotoneDec.intervalOn Ioo 0 1 h) ∧
  (∀ x, x ∈ Ioo 1 (3/2) → MonotoneDec.intervalOn Ioo 1 (3/2) h) ∧
  (∀ x, x ∈ Ioo (3/2) ∞ → Monotone.intervalOn Ioo (3/2) ∞ h) := sorry

end h_monotonic_intervals_l406_406346


namespace sum_of_floor_sqrt_1_to_25_l406_406991

theorem sum_of_floor_sqrt_1_to_25 : 
  ( ∑ i in Finset.range 26, ∥Real.sqrt i∥ ) = 75 :=
by
  sorry

end sum_of_floor_sqrt_1_to_25_l406_406991


namespace prob_2_lt_X_lt_4_l406_406497

noncomputable def normal_dist_p (μ σ : ℝ) (x : ℝ) : ℝ := sorry -- Assume this computes the CDF at x for a normal distribution

variable {X : ℝ → ℝ}
variable {σ : ℝ}

-- Condition: X follows a normal distribution with mean 3 and variance σ^2
axiom normal_distribution_X : ∀ x, X x = normal_dist_p 3 σ x

-- Condition: P(X ≤ 4) = 0.84
axiom prob_X_leq_4 : normal_dist_p 3 σ 4 = 0.84

-- Goal: Prove P(2 < X < 4) = 0.68
theorem prob_2_lt_X_lt_4 : normal_dist_p 3 σ 4 - normal_dist_p 3 σ 2 = 0.68 := by
  sorry

end prob_2_lt_X_lt_4_l406_406497


namespace unique_sums_count_l406_406277

def chips_bag_A : Set ℕ := {2, 3, 4}
def chips_bag_B : Set ℕ := {3, 4, 5}

theorem unique_sums_count :
  (Set.image (λ x y, x + y) chips_bag_A chips_bag_B).finite.to_finset.card = 5 :=
sorry

end unique_sums_count_l406_406277


namespace find_a_l406_406850

theorem find_a
  (a : ℝ)
  (h : ∃ (x : ℝ) (c : ℕ), ∑ k in finset.range 9, (nat.choose 8 k) * (a^k) * ((-1)^(8-k)) * (x^(k-(8-k)/2)) = 70 * x^2) :
  a = 1 ∨ a = -1 :=
by
  sorry

end find_a_l406_406850


namespace necessary_and_sufficient_condition_l406_406959

variable (U : Type) (A B : Set U)

-- Prove that A ∩ B = A is a necessary and sufficient condition for
-- A ⊆ U ∧ B ⊆ U ∧ complement_U B ⊆ complement_U A

theorem necessary_and_sufficient_condition :
  (A ∩ B = A) ↔ (A ⊆ U ∧ B ⊆ U ∧ (Set.compl B ⊆ Set.compl A)) := 
by
  sorry

end necessary_and_sufficient_condition_l406_406959


namespace sum_of_areas_of_circles_l406_406180

theorem sum_of_areas_of_circles (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  ∃ (π : ℝ), π * (r^2 + s^2 + t^2) = 56 * π :=
by {
  sorry
}

end sum_of_areas_of_circles_l406_406180


namespace cos_alpha_minus_pi_over_3_l406_406666

theorem cos_alpha_minus_pi_over_3 (α : ℝ) (h1 : cos α = 3/5) (h2 : α ∈ Ioo (3 * π / 2) (2 * π)) : 
  cos (α - π / 3) = (3 - 4 * sqrt 3) / 10 :=
sorry

end cos_alpha_minus_pi_over_3_l406_406666


namespace boys_play_both_l406_406390

-- Given conditions
variable (T B F N : ℕ)
variable (hT : T = 30) (hB : B = 18) (hF : F = 21) (hN : N = 4)

-- Define the proof problem
theorem boys_play_both :
  ∃ BF : ℕ, BF = B + F - (T - N) ∧ BF = 13 :=
by
  use B + F - (T - N)
  rw [hT, hB, hF, hN]
  norm_num
  sorry

end boys_play_both_l406_406390


namespace rectangle_perimeter_eq_l406_406832

noncomputable def rectangle_perimeter (x y : ℝ) := 2 * (x + y)

theorem rectangle_perimeter_eq (x y a b : ℝ)
  (h_area_rect : x * y = 2450)
  (h_area_ellipse : a * b = 2450)
  (h_foci_distance : x + y = 2 * a)
  (h_diag : x^2 + y^2 = 4 * (a^2 - b^2))
  (h_b : b = Real.sqrt (a^2 - 1225))
  : rectangle_perimeter x y = 120 * Real.sqrt 17 := by
  sorry

end rectangle_perimeter_eq_l406_406832


namespace proof_f_prime_at_3_l406_406333

def f (x : ℝ) : ℝ := x^2 * f' 2 - 3 * x

theorem proof_f_prime_at_3 (f' : ℝ → ℝ) (h : ∀ x, deriv (λ y, y^2 * f' 2 - 3 * y) x = 2 * x * f' 2 - 3) : f' 2 = 1 → (∀ x, deriv (f x) x = 2 * x - 3) → f' 3 = 3 := 
sorry

end proof_f_prime_at_3_l406_406333


namespace n_sided_polygon_angle_l406_406617

theorem n_sided_polygon_angle (n : ℕ) (h : 4 ∣ n) :
  let θ_n := (n - 2) * 180 / n in
  let θ_n_div_4 := (3 * n / 4 - 2) * 180 / (3 * n / 4) in
  θ_n = θ_n_div_4 - 12 → n = 10 := 
by 
  sorry

end n_sided_polygon_angle_l406_406617


namespace cost_price_of_table_l406_406217

theorem cost_price_of_table (CP : ℝ) (SP : ℝ) (h1 : SP = CP * 1.10) (h2 : SP = 8800) : CP = 8000 :=
by
  sorry

end cost_price_of_table_l406_406217


namespace trig_expr_evaluation_l406_406983

theorem trig_expr_evaluation :
    (1 - (1 / cos (Real.pi / 6))) * 
    (1 + (1 / sin (Real.pi / 3))) * 
    (1 - (1 / sin (Real.pi / 6))) * 
    (1 + (1 / cos (Real.pi / 3))) = 1 := 
by
  let cos_30 := cos (Real.pi / 6)
  let sin_60 := sin (Real.pi / 3)
  let sin_30 := sin (Real.pi / 6)
  let cos_60 := cos (Real.pi / 3)
  have h_cos_30 : cos_30 = (Real.sqrt 3) / 2 := sorry
  have h_sin_60 : sin_60 = (Real.sqrt 3) / 2 := sorry
  have h_sin_30 : sin_30 = 1 / 2 := sorry
  have h_cos_60 : cos_60 = 1 / 2 := sorry
  sorry

end trig_expr_evaluation_l406_406983


namespace jillian_oranges_l406_406053

theorem jillian_oranges:
  let oranges := 80 in
  let pieces_per_orange := 10 in
  let pieces_per_friend := 4 in
  (oranges * (pieces_per_orange / pieces_per_friend) = 200) :=
by sorry

end jillian_oranges_l406_406053


namespace minimum_value_of_quadratic_function_l406_406007

def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 8 * x + 15

theorem minimum_value_of_quadratic_function :
  ∃ x : ℝ, quadratic_function x = -1 ∧ ∀ y : ℝ, quadratic_function y ≥ -1 :=
by
  sorry

end minimum_value_of_quadratic_function_l406_406007


namespace original_cards_l406_406413

-- Define the number of cards Jason gave away
def cards_given_away : ℕ := 9

-- Define the number of cards Jason now has
def cards_now : ℕ := 4

-- Prove the original number of Pokemon cards Jason had
theorem original_cards (x : ℕ) : x = cards_given_away + cards_now → x = 13 :=
by {
    sorry
}

end original_cards_l406_406413


namespace correct_comparison_l406_406208

theorem correct_comparison :
  ( 
    (-1 > -0.1) = false ∧ 
    (-4 / 3 < -5 / 4) = true ∧ 
    (-1 / 2 > -(-1 / 3)) = false ∧ 
    (Real.pi = 3.14) = false 
  ) :=
by
  sorry

end correct_comparison_l406_406208


namespace lines_parallel_if_perpendicular_to_plane_l406_406084

variables (m n l : Line) (α β γ : Plane)

def perpendicular (m : Line) (α : Plane) : Prop := sorry
def parallel (m n : Line) : Prop := sorry

theorem lines_parallel_if_perpendicular_to_plane
  (h1 : perpendicular m α) (h2 : perpendicular n α) : parallel m n :=
sorry

end lines_parallel_if_perpendicular_to_plane_l406_406084


namespace find_c_l406_406868

theorem find_c 
  (c : ℝ)
  (h : (vector.dot_product ![-5, c] ![3, -1]) / (∥![3, -1]∥ * ∥![3, -1]∥) * ![3, -1] = (1 / 10) * ![3, -1]) :
  c = -16 := by
  sorry

end find_c_l406_406868


namespace equation_of_circle_l406_406480

-- Conditions
def center_of_circle : ℝ × ℝ := (2, -1)
def chord_length : ℝ := 2 * Real.sqrt 2
def chord_line (x y : ℝ) : Prop := x - y - 1 = 0

-- Theorem statement
theorem equation_of_circle 
  (center := center_of_circle)
  (length := chord_length)
  (line := chord_line) :
  ∃ r : ℝ, r = 2 ∧ 
    (by ∀ x y : ℝ, chord_line x y → (x - center.1)^2 + (y - center.2)^2 = r^2) :=
sorry

end equation_of_circle_l406_406480


namespace probability_area_circle_between_l406_406578

theorem probability_area_circle_between (AB_length : ℝ) (h1 : AB_length = 10) :
  (let G := Classical.choose (Classical.arbitrary (Subtype (λ x : ℝ, 0 ≤ x ∧ x ≤ AB_length))),
       radius := λ G, G,
       area := λ r, Real.pi * r^2 in
   (λ P, P = (2 / AB_length)) (Set.probability (Set.Icc 6 8) G)) :=
by
  rw h1
  simp
  exact (by norm_num) = (by norm_num : (6 * 6 * Real.pi ≤ area (radius G) ∧ area (radius G) ≤ 8 * 8 * Real.pi))

end probability_area_circle_between_l406_406578


namespace exists_city_available_for_all_l406_406040

variable {City : Type}
variable (flights : City → City → Prop)

-- Definition of "available"
def available (A B : City) : Prop :=
  ∃ (path : List City), path.head = B ∧ path.last = some A ∧ 
  path.chain' (λ x y, flights x y)

-- Condition: For every pair of cities P and Q, there exists a city R such that both P and Q are available from R.
variable (exists_r : ∀ P Q : City, ∃ R : City, available flights P R ∧ available flights Q R)

-- Theorem: There exists a city A such that every city is available for A.
theorem exists_city_available_for_all : ∃ A : City, ∀ B : City, available flights B A :=
sorry

end exists_city_available_for_all_l406_406040


namespace probability_B_before_A_and_C_l406_406306

theorem probability_B_before_A_and_C :
  let squadrons : List (List Char) := 
    [
      ['A', 'B', 'C'],
      ['A', 'C', 'B'],
      ['B', 'A', 'C'],
      ['B', 'C', 'A'],
      ['C', 'A', 'B'],
      ['C', 'B', 'A']
    ] in
  let favorable_orders := 
    [
      ['B', 'A', 'C'],
      ['B', 'C', 'A']
    ] in
  (favorable_orders.length : ℚ) / (squadrons.length : ℚ) = 1 / 3 :=
  sorry

end probability_B_before_A_and_C_l406_406306


namespace smallest_a_for_polynomial_roots_l406_406150

theorem smallest_a_for_polynomial_roots :
  ∃ (a b c : ℕ), 
         (∃ (r s t u : ℕ), r > 0 ∧ s > 0 ∧ t > 0 ∧ u > 0 ∧ r * s * t * u = 5160 ∧ a = r + s + t + u) 
    ∧  (∀ (r' s' t' u' : ℕ), r' > 0 ∧ s' > 0 ∧ t' > 0 ∧ u' > 0 ∧ r' * s' * t' * u' = 5160 ∧ r' + s' + t' + u' < a → false) 
    := sorry

end smallest_a_for_polynomial_roots_l406_406150


namespace projection_circle_cover_l406_406753

theorem projection_circle_cover (N : ℕ) (hN : N ≥ 2) 
  (lines : Fin N → AffineLine ℝ) (non_parallel : ∀ i j, i ≠ j → ¬parallel (lines i) (lines j)) 
  (P : AffinePoint ℝ) : 
  ∃ (O : AffinePoint ℝ) (R : ℝ), ∀ Q, Q ∈ (projections P lines) → dist Q O ≤ R := 
sorry

end projection_circle_cover_l406_406753


namespace least_number_to_subtract_l406_406317

theorem least_number_to_subtract :
  ∃ k : ℕ, k = 45 ∧ (568219 - k) % 89 = 0 :=
by
  sorry

end least_number_to_subtract_l406_406317


namespace digit_sum_of_large_product_l406_406322

theorem digit_sum_of_large_product : 
  let A := (10 ^ 2012 - 1) in
  let B := (4 * ((10 ^ 2011 - 1) / 9)) in
  let product := A * B in
  digit_sum product = 18108 :=
by sorry

end digit_sum_of_large_product_l406_406322


namespace john_spending_l406_406061

variable (initial_cost : ℕ) (sale_price : ℕ) (new_card_cost : ℕ)

theorem john_spending (h1 : initial_cost = 1200) (h2 : sale_price = 300) (h3 : new_card_cost = 500) :
  initial_cost - sale_price + new_card_cost = 1400 := 
by
  sorry

end john_spending_l406_406061


namespace area_of_region_l406_406635

-- Definitions drawn from conditions
def circle_radius := 36
def num_small_circles := 8

-- Main statement to be proven
theorem area_of_region :
  ∃ K : ℝ, 
    K = π * (circle_radius ^ 2) - num_small_circles * π * ((circle_radius * (Real.sqrt 2 - 1)) ^ 2) ∧
    ⌊ K ⌋ = ⌊ π * (circle_radius ^ 2) - num_small_circles * π * ((circle_radius * (Real.sqrt 2 - 1)) ^ 2) ⌋ :=
  sorry

end area_of_region_l406_406635


namespace sum_first_10_terms_seq_c_l406_406076

def geom_seq (a_1 q : ℕ → ℕ) (n : ℕ) : ℕ := a_1 * q^(n-1)
def arith_seq (b_1 d : ℕ → ℤ) (n : ℕ) : ℤ := b_1 + (n-1) * d

def seq_c (a b : ℕ → ℤ) (n : ℕ) : ℤ := a n + b n

theorem sum_first_10_terms_seq_c (a_1 q : ℕ) (b_1 d : ℕ) :
  ∀ {n : ℕ}, n = 10 →
    let a := λ n, geom_seq a_1 q n,
        b := λ n, arith_seq b_1 d n,
        c := λ n, seq_c a b n in
    a_1 = 1 ∧ q = 2 ∧ b_1 = 0 ∧ d = -1 →
    (∑ i in Finset.range n, c (i + 1)) = 978 :=
by
  intros n h_eq hx
  conv at hx {
    rw [Finset.sum_range' 10 h_eq]
  }
  sorry

end sum_first_10_terms_seq_c_l406_406076


namespace distinguishable_arrangements_l406_406718

theorem distinguishable_arrangements :
  let n := 9
  let n1 := 3
  let n2 := 2
  let n3 := 4
  (Nat.factorial n) / ((Nat.factorial n1) * (Nat.factorial n2) * (Nat.factorial n3)) = 1260 :=
by sorry

end distinguishable_arrangements_l406_406718


namespace isosceles_triangle_base_angles_l406_406400

theorem isosceles_triangle_base_angles (A B C : Type) [triangle ABC]
  (H1 : is_isosceles_triangle ABC)
  (H2 : ∃ θ : ℝ, θ = 80 ∧ (interior_angle A = θ ∨ interior_angle B = θ ∨ interior_angle C = θ)) :
  (base_angle B = 50 ∨ base_angle B = 80) :=
by
  sorry

end isosceles_triangle_base_angles_l406_406400


namespace no_prime_1111_in_base_l406_406326

theorem no_prime_1111_in_base (n : ℕ) (h : n ≥ 2) : 
  ¬ prime (n^3 + n^2 + n + 1) := by
  sorry

end no_prime_1111_in_base_l406_406326


namespace find_sum_of_a_and_b_l406_406729

theorem find_sum_of_a_and_b (a b : ℝ) (h1 : 0.005 * a = 0.65) (h2 : 0.0125 * b = 1.04) : a + b = 213.2 :=
  sorry

end find_sum_of_a_and_b_l406_406729


namespace faster_train_length_is_correct_l406_406890

-- Definitions based on conditions
def speed_train_slow : ℝ := 36 -- in kmph
def speed_train_fast : ℝ := 45 -- in kmph
def time_seconds : ℝ := 4     -- in seconds

-- Convert from kmph to m/s
def kmph_to_mps (v : ℝ) : ℝ := v * (5 / 18)

-- Define the relative speed in m/s
def relative_speed_mps : ℝ := kmph_to_mps (speed_train_slow + speed_train_fast)

-- The length of the faster train
def length_faster_train : ℝ := relative_speed_mps * time_seconds

-- Proof statement
theorem faster_train_length_is_correct : length_faster_train = 90 :=
by
  -- skip the proof
  sorry 

end faster_train_length_is_correct_l406_406890


namespace curve_C_fixed_point_l406_406792

-- Definition of circle O where P is any point on the circle
def is_on_circle (P : ℝ × ℝ) : Prop := P.1 ^ 2 + P.2 ^ 2 = 1

-- Definition of point Q satisfying DQ = 2PQ
def satisfies_DQ_2PQ (P Q D : ℝ × ℝ) : Prop :=
  Q.1 = 2 * P.1 ∧ Q.2 = P.2 ∧ D.1 = P.1 ∧ D.2 = foot_y_axis P ∧ 2 * (Q.1 - D.1, Q.2 - D.2) = (Q.1 - P.1, Q.2 - P.2)

-- Equation of the locus of Q to be a curve C
theorem curve_C (Q : ℝ × ℝ) (h : ∃ P : ℝ × ℝ, ∃ D : ℝ × ℝ, is_on_circle P ∧ satisfies_DQ_2PQ P Q D) : 
  Q.1 ^ 2 / 4 + Q.2 ^ 2 = 1 := 
  sorry

-- Definition of A and condition for line l
def intersection_point_A (A : ℝ × ℝ) : Prop := A = (0, 1)

-- Definition of line l passing through A and intersecting C at M and N such that AM ⋅ AN = 0
def line_l_through_A_intersects_C (A M N : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  A = (0, 1) ∧ is_on_curve_C M ∧ is_on_curve_C N ∧ l A.1 = A.2 ∧ l M.1 = M.2 ∧ l N.1 = N.2 ∧ (M.1 * N.1 + (M.2 - 1) * (N.2 - 1)) = 0

-- Prove that the line l passes through a fixed point (0, -3/5)
theorem fixed_point (A : ℝ × ℝ) (hA : intersection_point_A A) 
  (M N : ℝ × ℝ) (l : ℝ → ℝ) (h : line_l_through_A_intersects_C A M N l) : 
  ∃ F : ℝ × ℝ, F = (0, -3/5) :=
  sorry

end curve_C_fixed_point_l406_406792


namespace percentage_students_absent_l406_406509

theorem percentage_students_absent (total_students : ℕ) (students_present : ℕ) (h1 : total_students = 50) (h2 : students_present = 44) : 
  ((total_students - students_present) / total_students.to_rat) * 100 = 12 :=
by
  sorry

end percentage_students_absent_l406_406509


namespace find_missing_figure_l406_406551

theorem find_missing_figure (x : ℝ) (h : 0.003 * x = 0.15) : x = 50 :=
sorry

end find_missing_figure_l406_406551


namespace fastest_growth_rate_l406_406958

theorem fastest_growth_rate :
  ∀ x : ℝ, (20^x) > (x^20) ∧ (20^x) > (log 20 x) ∧ (20^x) > (20 * x) :=
by
  sorry

end fastest_growth_rate_l406_406958


namespace exists_sequence_satisfying_conditions_l406_406630

def F : ℕ → ℕ := sorry

theorem exists_sequence_satisfying_conditions :
  (∀ n, ∃ k, F k = n) ∧ 
  (∀ n, ∃ m > n, F m = n) ∧ 
  (∀ n ≥ 2, F (F (n ^ 163)) = F (F n) + F (F 361)) :=
sorry

end exists_sequence_satisfying_conditions_l406_406630


namespace locus_of_point_l406_406579

variable {α x y : ℝ}

def P (α : ℝ) : ℝ × ℝ := (Real.sin α, Real.cos α)

def distance_to_point (x y α : ℝ) : ℝ :=
  Real.sqrt ((x - Real.sin α) ^ 2 + (y - Real.cos α) ^ 2)

def distance_to_line (x y α : ℝ) : ℝ :=
  Real.abs (x * Real.sin α + y * Real.cos α - 1)

theorem locus_of_point (α : ℝ) :
  ∀ (x y : ℝ), 
  (distance_to_point x y α = distance_to_line x y α) → 
  -- locus of M(x, y) is the straight line passing through P(α) and perpendicular to the line l
  ∃ k b : ℝ, ∀ (x y : ℝ), y = k * x + b ∧ 
                  (k = - Real.cos α / Real.sin α) ∧ 
                  (y = (- (Real.cos α / Real.sin α) * x + 1 / Real.sin α)) :=
sorry

end locus_of_point_l406_406579


namespace sum_difference_of_odd_and_even_integers_l406_406323

noncomputable def sum_of_first_n_odds (n : ℕ) : ℕ :=
  n * n

noncomputable def sum_of_first_n_evens (n : ℕ) : ℕ :=
  n * (n + 1)

theorem sum_difference_of_odd_and_even_integers :
  sum_of_first_n_evens 50 - sum_of_first_n_odds 50 = 50 := 
by
  sorry

end sum_difference_of_odd_and_even_integers_l406_406323


namespace tournament_committee_count_l406_406768

theorem tournament_committee_count :
  let teams := 4
  let members_per_team := 8
  let host_team_choices := choose 8 4
  let non_host_team_choices := choose 8 2
  (teams * host_team_choices * (non_host_team_choices ^ (teams - 1))) = 6146560 :=
by
  let teams := 4
  let members_per_team := 8
  let host_team_choices := choose 8 4 sorry
  let non_host_team_choices := choose 8 2 sorry
  calc
    _ = 4 * 70 * (28 ^ 3) : sorry
    ... = 6146560 : sorry
  sorry

end tournament_committee_count_l406_406768


namespace triangle_inequality_satisfies_l406_406592

theorem triangle_inequality_satisfies (x : ℕ) : (x > 0 ∧ x < 16) ↔ x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15} :=
by
  sorry

end triangle_inequality_satisfies_l406_406592


namespace collinear_equiv_l406_406967

variables (A B C D E F M P : Point)
variables (O : Circle)
variables (hTangent : Tangent PA A O) (hSecant : Secant PBC O)
variables (hMid : Midpoint M PA) (hIntersect : Intersects AD BC E)
variables (hAngle : ∠ FBD = ∠ FED)

theorem collinear_equiv :
  Collinear {P, F, D} ↔ Collinear {M, B, D} :=
sorry

end collinear_equiv_l406_406967


namespace exponent_is_two_and_power_expression_l406_406731

theorem exponent_is_two_and_power_expression (x : ℝ) (h₁ : 2^x = 4) : x = 2 ∧ 2^(2*x + 2) = 64 :=
by
  sorry

end exponent_is_two_and_power_expression_l406_406731


namespace number_of_good_weeks_l406_406716

-- Definitions from conditions
def tough_week_sales : ℕ := 800
def good_week_sales : ℕ := 2 * tough_week_sales
def tough_weeks : ℕ := 3
def total_money_made : ℕ := 10400
def total_tough_week_sales : ℕ := tough_weeks * tough_week_sales
def total_good_week_sales : ℕ := total_money_made - total_tough_week_sales

-- Question to be proven
theorem number_of_good_weeks (G : ℕ) : 
  (total_good_week_sales = G * good_week_sales) → G = 5 := by
  sorry

end number_of_good_weeks_l406_406716


namespace trig_expression_equality_l406_406975

theorem trig_expression_equality :
  (1 - 1 / (Real.cos (Real.pi / 6))) *
  (1 + 1 / (Real.sin (Real.pi / 3))) *
  (1 - 1 / (Real.sin (Real.pi / 6))) *
  (1 + 1 / (Real.cos (Real.pi / 3))) = 1 :=
by
  have hcos30 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2 := sorry
  have hsin60 : Real.sin (Real.pi / 3) = (Real.sqrt 3) / 2 := sorry
  have hsin30 : Real.sin (Real.pi / 6) = 1 / 2 := sorry
  have hcos60 : Real.cos (Real.pi / 3) = 1 / 2 := sorry
  sorry

end trig_expression_equality_l406_406975


namespace cos_half_alpha_l406_406351

open Real -- open the Real namespace for convenience

theorem cos_half_alpha {α : ℝ} (h1 : cos α = 1 / 5) (h2 : 0 < α ∧ α < π) :
  cos (α / 2) = sqrt (15) / 5 :=
by
  sorry -- Proof is omitted

end cos_half_alpha_l406_406351


namespace find_a_l406_406702

-- Definitions for the conditions
def parabola_eq (x y p : ℝ) := y^2 = 2 * p * x
def on_parabola (p : ℝ) (M : ℝ × ℝ) := parabola_eq M.1 M.2 p
def parabola_focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def distance (P Q : ℝ × ℝ) : ℝ := real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

-- Problem statement as a Lean 4 theorem
theorem find_a (p m a b : ℝ) (h_cond : p > 0)
  (h_on_parabola : on_parabola p (1, m))
  (h_distance : distance (1, m) (parabola_focus p) = 5)
  (h_hyperbola : a ≠ 0 ∧ b ≠ 0 ∧ (∀ x y, x^2 / a^2 - y^2 / b^2 = 1))
  (h_perpendicular_asymptote : (m / (1 + a)) * (-b / a) = -1) :
  a = 1 / 4 :=
by
  sorry

end find_a_l406_406702


namespace maximum_garden_area_l406_406417

theorem maximum_garden_area (l w : ℝ) (h_perimeter : 2 * l + 2 * w = 400) : 
  l * w ≤ 10000 :=
by {
  -- proving the theorem
  sorry
}

end maximum_garden_area_l406_406417


namespace eccentricity_of_ellipse_l406_406690

noncomputable def ellipse_eccentricity (a b : ℝ) (h : a = 2 * b) : ℝ :=
  let c := Real.sqrt (a^2 - b^2) in
  c / a

theorem eccentricity_of_ellipse (a b : ℝ) (h : a = 2 * b) : ellipse_eccentricity a b h = Real.sqrt 3 / 2 :=
by
  sorry

end eccentricity_of_ellipse_l406_406690


namespace sqrt_sum_is_integer_or_irrational_l406_406437

noncomputable def is_integer_or_irrational (x : ℝ) : Prop :=
  int.cast ℤ → ℝ x ∨ irrational x

theorem sqrt_sum_is_integer_or_irrational (n1 n2 : ℕ) : 
  is_integer_or_irrational (real.sqrt n1 + real.cbrt n2) :=
sorry

end sqrt_sum_is_integer_or_irrational_l406_406437


namespace triangle_pentagon_side_ratio_l406_406599

theorem triangle_pentagon_side_ratio (triangle_perimeter : ℕ) (pentagon_perimeter : ℕ) 
  (h1 : triangle_perimeter = 60) (h2 : pentagon_perimeter = 60) :
  (triangle_perimeter / 3 : ℚ) / (pentagon_perimeter / 5 : ℚ) = 5 / 3 :=
by {
  sorry
}

end triangle_pentagon_side_ratio_l406_406599


namespace red_balloon_probability_l406_406782

-- Define the conditions
def initial_red_balloons := 2
def initial_blue_balloons := 4
def inflated_red_balloons := 2
def inflated_blue_balloons := 2

-- Define the total number of balloons after inflation
def total_red_balloons := initial_red_balloons + inflated_red_balloons
def total_blue_balloons := initial_blue_balloons + inflated_blue_balloons
def total_balloons := total_red_balloons + total_blue_balloons

-- Define the probability calculation
def red_probability := (total_red_balloons : ℚ) / total_balloons * 100

-- The theorem to prove
theorem red_balloon_probability : red_probability = 40 := by
  sorry -- Skipping the proof itself

end red_balloon_probability_l406_406782


namespace spring_compression_l406_406514

theorem spring_compression (s F : ℝ) (h : F = 16 * s^2) (hF : F = 4) : s = 0.5 :=
by
  sorry

end spring_compression_l406_406514


namespace variance_D_X_correct_l406_406889

namespace ShooterStatistics

noncomputable def probability_A := 2 / 3
noncomputable def probability_B := 4 / 5

-- Define the random variable X as the number of shooters hitting the target
inductive X
| zero_hit : X
| one_hit  : X
| two_hit  : X

-- Probabilities of each case
def P_zero_hit : ℚ := (1 - probability_A) * (1 - probability_B)
def P_one_hit  : ℚ := (1 - probability_A) * probability_B + (1 - probability_B) * probability_A
def P_two_hit  : ℚ := probability_A * probability_B

-- Calculating the variance D(X)
def expected_X : ℚ := 0 * P_zero_hit + 1 * P_one_hit + 2 * P_two_hit
def variance_X : ℚ := (0 - expected_X)^2 * P_zero_hit + (1 - expected_X)^2 * P_one_hit + (2 - expected_X)^2 * P_two_hit

-- The goal is to prove that variance_X equals 86/225
theorem variance_D_X_correct :
  variance_X = 86 / 225 :=
sorry

end ShooterStatistics

end variance_D_X_correct_l406_406889


namespace graph_shift_cosine_to_sine_l406_406513

theorem graph_shift_cosine_to_sine :
    ∀ x, (cos (2*x - (Real.pi / 3)) = sin ((Real.pi / 2) + 2*(x - (Real.pi / 6)))) :=
by
  intro x
  sorry

end graph_shift_cosine_to_sine_l406_406513


namespace harriet_trip_time_l406_406904

theorem harriet_trip_time :
  ∀ (t1 : ℝ) (s1 s2 t2 d : ℝ), 
  t1 = 2.8 ∧ 
  s1 = 110 ∧ 
  s2 = 140 ∧ 
  d = s1 * t1 ∧ 
  t2 = d / s2 → 
  t1 + t2 = 5 :=
by intros t1 s1 s2 t2 d
   sorry

end harriet_trip_time_l406_406904


namespace original_price_of_coffee_l406_406240

/-- 
  Define the prices of the cups of coffee as per the conditions.
  Let x be the original price of one cup of coffee.
  Assert the conditions and find the original price.
-/
theorem original_price_of_coffee (x : ℝ) 
  (h1 : x + x / 2 + 3 = 57) 
  (h2 : (x + x / 2 + 3)/3 = 19) : 
  x = 36 := 
by
  sorry

end original_price_of_coffee_l406_406240


namespace one_incorrect_proposition_l406_406708

-- Defining the data types for planes and lines.
variable (Plane Line : Type) [Nonempty Plane] [Nonempty Line]

-- Defining relations between lines and planes.
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (subset_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (intersection_line_plane : Plane → Plane → Line)

-- Propositions.
def proposition_1 (m n : Line) (α : Plane) : Prop :=
  parallel_lines m n ∧ perpendicular_line_plane m α → perpendicular_line_plane n α

def proposition_2 (m : Line) (α β : Plane) : Prop :=
  perpendicular_line_plane m α ∧ perpendicular_line_plane m β → parallel_planes α β

def proposition_3 (m n : Line) (α β : Plane) : Prop :=
  perpendicular_line_plane m α ∧ parallel_lines m n ∧ subset_line_plane n β → perpendicular_line_plane n α ∧ perpendicular_line_plane m β

def proposition_4 (m n : Line) (α β : Plane) : Prop :=
  parallel_lines m α ∧ intersection_line_plane α β = n → parallel_lines m n

-- Translate the question to a theorem in Lean.
theorem one_incorrect_proposition :
  ∃ p1 p2 p3 p4 : Prop, 
    (p1 ↔ proposition_1 m n α) ∧ 
    (p2 ↔ proposition_2 m α β) ∧ 
    (p3 ↔ proposition_3 m n α β) ∧ 
    (p4 ↔ proposition_4 m n α β) ∧ 
    [(false ↔ p1), (false ↔ p2), (false ↔ p3), (false ↔ p4)].count false = 1 :=
by {
  sorry
}

end one_incorrect_proposition_l406_406708


namespace area_ratio_of_squares_l406_406109

theorem area_ratio_of_squares (a b : ℕ) (h : 4 * a = 4 * (4 * b)) : (a * a) = 16 * (b * b) :=
by
  sorry

end area_ratio_of_squares_l406_406109


namespace circles_area_sum_l406_406161

noncomputable def sum_of_areas_of_circles (r s t : ℝ) : ℝ :=
  π * (r^2 + s^2 + t^2)

theorem circles_area_sum (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  sum_of_areas_of_circles r s t = 56 * π :=
begin
  sorry
end

end circles_area_sum_l406_406161


namespace trigonometric_identity_l406_406015

theorem trigonometric_identity (θ : ℝ) (h : cos θ = 3 / 5) : 
  cos (3 * θ) = -117 / 125 ∧ sin (3 * θ) = 44 / 125 :=
by
  sorry

end trigonometric_identity_l406_406015


namespace Steve_cup_stacks_l406_406125

theorem Steve_cup_stacks :
  ∀ (cups : ℕ → ℕ),
  cups 1 = 17 →
  cups 2 = 21 →
  cups 3 = 25 →
  cups 5 = 33 →
  (∀ n : ℕ, cups (n + 1) - cups n = 4) →
  cups 4 = 29 :=
begin
  intros cups h1 h2 h3 h5 hp,
  sorry
end

end Steve_cup_stacks_l406_406125


namespace no_real_solution_to_system_l406_406840

theorem no_real_solution_to_system :
  ∀ (x y z : ℝ), (x + y - 2 - 4 * x * y = 0) ∧
                 (y + z - 2 - 4 * y * z = 0) ∧
                 (z + x - 2 - 4 * z * x = 0) → false := 
by 
    intros x y z h
    rcases h with ⟨h1, h2, h3⟩
    -- Here would be the proof steps, which are omitted.
    sorry

end no_real_solution_to_system_l406_406840


namespace point_in_circle_l406_406300

-- Define the coordinates of the point P and the center of the circle C
def P : ℝ × ℝ := (1, 1)
def C_center : ℝ × ℝ := (2, -3)

-- Define the radius of the circle
def r : ℝ := 3 * Real.sqrt 2

-- Define the distance function between two points
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- State the problem as a theorem
theorem point_in_circle : dist P C_center < r :=
  sorry

end point_in_circle_l406_406300


namespace eq_one_half_eq_one_eq_log_five_l406_406124

theorem eq_one_half (x : ℝ) (hx : 2^x = real.sqrt 2) : x = 1 / 2 :=
sorry

theorem eq_one (x : ℝ) (hx : real.log (3 * x) / real.log 2 = real.log (2 * x + 1) / real.log 2) : x = 1 :=
sorry

theorem eq_log_five (x : ℝ) (hx : 2 * 5^(x + 1) - 3 = 0) : x = real.log (3 / 2) / real.log 5 - 1 :=
sorry

end eq_one_half_eq_one_eq_log_five_l406_406124


namespace range_of_m_for_locally_odd_function_l406_406742

noncomputable def f (x m : ℝ) := 4^x - m * 2^(x+1) + m^2 - 3

def locally_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

theorem range_of_m_for_locally_odd_function (m : ℝ) :
  locally_odd_function (λ x, 4^x - m * 2^(x + 1) + m^2 - 3) ↔ 1 - real.sqrt 3 ≤ m ∧ m ≤ 2 * real.sqrt 2 :=
by sorry

end range_of_m_for_locally_odd_function_l406_406742


namespace sum_of_numbers_l406_406873

theorem sum_of_numbers (x : ℕ) (first_num second_num third_num sum : ℕ) 
  (h1 : 5 * x = first_num) 
  (h2 : 3 * x = second_num)
  (h3 : 4 * x = third_num) 
  (h4 : second_num = 27)
  : first_num + second_num + third_num = 108 :=
by {
  sorry
}

end sum_of_numbers_l406_406873


namespace inequality_solution_solution_set_l406_406667

noncomputable def f (x a : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + 6

theorem inequality_solution (a : ℝ) : 
  f 1 a > 0 ↔ 3 - 2 * Real.sqrt 3 < a ∧ a < 3 + 2 * Real.sqrt 3 :=
by sorry

theorem solution_set (a b : ℝ) : 
  (∀ x, -1 < x ∧ x < 3 → f x a > b) ∧ (∃ x, -1 ≤ x ∧ x ≤ 3 ∧ f x a = b) ↔ 
  (a = 3 + Real.sqrt 3 ∨ a = 3 - Real.sqrt 3) ∧ b = -3 :=
by sorry

end inequality_solution_solution_set_l406_406667


namespace sum_areas_of_circles_l406_406165

theorem sum_areas_of_circles (r s t : ℝ) 
  (h1 : r + s = 6)
  (h2 : r + t = 8)
  (h3 : s + t = 10) : 
  r^2 * Real.pi + s^2 * Real.pi + t^2 * Real.pi = 56 * Real.pi := 
by 
  sorry

end sum_areas_of_circles_l406_406165


namespace problem_D_l406_406800

variable (f : ℕ → ℝ)

-- Function condition: If f(k) ≥ k^2, then f(k+1) ≥ (k+1)^2
axiom f_property (k : ℕ) (hk : f k ≥ k^2) : f (k + 1) ≥ (k + 1)^2

theorem problem_D (hf4 : f 4 ≥ 25) : ∀ k ≥ 4, f k ≥ k^2 :=
by
  sorry

end problem_D_l406_406800


namespace max_pi_n_l406_406859

def initial_term : ℕ := 1536
def common_ratio : ℝ := -1/2
def geometric_sequence (n : ℕ) := initial_term * common_ratio^(n - 1)

def pi_n (n : ℕ) : ℝ := ∏ i in range n, geometric_sequence (i + 1)

theorem max_pi_n : (pi_n 12) = max (pi_n 9) (max (pi_n 11) (max (pi_n 12) (pi_n 13))) :=
  sorry

end max_pi_n_l406_406859


namespace split_eq_parts_l406_406106

-- Defining the centers of symmetry for the rectangle and circle
variables (L W R a b : ℝ)

-- Define the center of the rectangular cake
def center_rect := (L / 2, W / 2)

-- Define the center of the circular chocolate
def center_circle := (a, b)

-- Define a line passing through the centers of both the rectangle and the circle
def cut_line (L W a b : ℝ) : (ℝ × ℝ) := 
  (y - W / 2 = (b - W / 2) / (a - L / 2) * (x - L / 2))

-- Prove that this line splits both the rectangle and the chocolate piece in half
theorem split_eq_parts : 
  (∀ L W R a b : ℝ, 
  cut_line L W a b 
  := sorry

end split_eq_parts_l406_406106


namespace sum_of_three_eq_six_l406_406474

theorem sum_of_three_eq_six
  (a b c : ℕ) (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h : (a + b + c)^3 - a^3 - b^3 - c^3 = 150) :
  a + b + c = 6 :=
sorry

end sum_of_three_eq_six_l406_406474


namespace equilateral_triangle_angle_EDF_120_l406_406770

theorem equilateral_triangle_angle_EDF_120
  (A B C D E F : Point)
  (equilateral : equilateral_triangle A B C)
  (D_midpoint : midpoint D B C)
  (E_one_third : one_third A E C)
  (F_one_third : one_third A F B)
  (DE_eq_DF : DE = DF) :
  angle D E F = 120 :=
sorry

end equilateral_triangle_angle_EDF_120_l406_406770


namespace parametric_equation_of_curve_C_l406_406923

open Real

theorem parametric_equation_of_curve_C (θ : ℝ) :
  let x := cos θ,
  let y := sin θ,
  ∀ x' y', (x' = 3*x) ∧ (y' = y) → (x'^2 + 9*y'^2 = 9) → (x^2 + y^2 = 1) :=
by
  sorry

end parametric_equation_of_curve_C_l406_406923


namespace bowls_remaining_after_rewards_l406_406608

noncomputable theory

def initial_bowls : ℕ := 150
def customers_15_bowls : ℕ := 10
def customers_25_bowls : ℕ := 12
def customers_35_bowls : ℕ := 4
def customers_0_bowls : ℕ := 4

def free_bowls_15_bowls : ℕ := 2
def free_bowls_25_bowls : ℕ := 5
def free_bowls_35_bowls : ℕ := 8

def total_free_bowls := (customers_15_bowls * free_bowls_15_bowls) +
                        (customers_25_bowls * free_bowls_25_bowls) +
                        (customers_35_bowls * free_bowls_35_bowls)

def remaining_bowls (initial : ℕ) (free : ℕ) : ℕ :=
  initial - free

theorem bowls_remaining_after_rewards :
  remaining_bowls initial_bowls total_free_bowls = 38 :=
by
  sorry

end bowls_remaining_after_rewards_l406_406608


namespace driving_wheel_diameter_l406_406854

noncomputable def diameter_of_driving_wheel 
  (revolutions_per_minute : ℝ) (speed_kmph : ℝ) : ℝ :=
  let distance_per_minute := (speed_kmph * 1000 * 100) / 60
  let distance_per_revolution := distance_per_minute / revolutions_per_minute
  let π := Real.pi
  distance_per_revolution / π

theorem driving_wheel_diameter :
  diameter_of_driving_wheel 75.75757575757576 20 ≈ 140 :=
sorry

end driving_wheel_diameter_l406_406854


namespace slope_tangent_is_3_l406_406148

theorem slope_tangent_is_3 :
  ∀ x y : ℝ, (y = x ^ 3) ∧ (derivative (λ x, x ^ 3) x = 3) ↔ ((x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1)) :=
begin
  sorry
end

end slope_tangent_is_3_l406_406148


namespace mark_wait_between_vaccines_l406_406444

theorem mark_wait_between_vaccines (X : ℕ) :
  (4 + X + 14 = 38) → X = 20 :=
by
  assume h : 4 + X + 14 = 38
  sorry

end mark_wait_between_vaccines_l406_406444


namespace new_necklaces_bought_l406_406279

-- Given constants
def n1 : ℕ := 50
def b : ℕ := 3
def g : ℕ := 15
def n2 : ℕ := 37

-- Calculate the number of new necklaces bought, k
def k : ℕ := n2 - (n1 - b - g)

-- The key theorem to prove
theorem new_necklaces_bought : k = 5 :=
by calc
  k = n2 - (n1 - b - g) : by rfl
  ... = 37 - (50 - 3 - 15) : by rfl
  ... = 37 - 32 : by rfl
  ... = 5 : by rfl

end new_necklaces_bought_l406_406279


namespace find_c_d_l406_406432

noncomputable def g (c d x : ℝ) : ℝ := c * x^3 + 5 * x^2 + d * x + 7

theorem find_c_d : ∃ (c d : ℝ), 
  (g c d 2 = 11) ∧ (g c d (-3) = 134) ∧ c = -35 / 13 ∧ d = 16 / 13 :=
  by
  sorry

end find_c_d_l406_406432


namespace sum_of_areas_of_circles_l406_406176

theorem sum_of_areas_of_circles 
  (r s t : ℝ) 
  (h1 : r + s = 6) 
  (h2 : r + t = 8) 
  (h3 : s + t = 10) 
  : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by 
  sorry

end sum_of_areas_of_circles_l406_406176


namespace exchange_rmb_ways_l406_406639

theorem exchange_rmb_ways : 
  {n : ℕ // ∃ (x y z : ℕ), x + 2 * y + 5 * z = 10 ∧ n = 10} :=
sorry

end exchange_rmb_ways_l406_406639


namespace correctness_of_propositions_l406_406083

-- Define basic geometric concepts
open_locale classical

variables {Point Line Plane : Type}
variable (Incidence : Point → Line → Prop)
variable (Containment : Point → Plane → Prop)
variable (Perpendicular : Line → Plane → Prop)
variable (ParallelLine : Line → Line → Prop)
variable (ParallelPlane : Plane → Plane → Prop)

-- Given Conditions
variables (l m : Line) (α β : Plane)
variable [hlm : l ≠ m]
variable [hαβ : α ≠ β]
variable [h₁ : Perpendicular l α]
variable [h₂ : Perpendicular m β]

-- Propositions
theorem correctness_of_propositions : 
  (ParallelPlane α β → ParallelLine l m) ∧ 
  (Perpendicular α β → Perpendicular l m) := 
by {
  sorry
}

end correctness_of_propositions_l406_406083


namespace starting_player_wins_optimally_l406_406880

theorem starting_player_wins_optimally (n : ℕ) (m : ℕ) (h1 : n = 100) (h2 : m = 7) :
  ∃ (winning_strategy : (fin (m+1) → fin (m+1))), true :=
by
  exists sorry

end starting_player_wins_optimally_l406_406880


namespace color_white_cells_l406_406107

theorem color_white_cells
  (black_cells : Finset (ℤ × ℤ))
  (even_white_neighbors : ∀ (b ∈ black_cells), (Nat.gcd (card (neighbors b ∩ white_cells)) 2 = 0)) : 
  ∃ (coloring : (ℤ × ℤ) → Prop), 
    (∀ (b ∈ black_cells), 
      card {w ∈ neighbors b | coloring w} = card {w ∈ neighbors b | ¬coloring w}) :=
by
  sorry

end color_white_cells_l406_406107


namespace max_crosses_l406_406402

-- Define the condition on the grid for rows and columns
def unique_cross_configuration (grid : matrix (fin 10) (fin 10) bool) : Prop :=
  ∀ (i : fin 10), (∃! j : fin 10, grid i j = tt) ∨ (∃! j : fin 10, grid j i = tt)

-- Define the predicate that the total number of crosses is ≤ n
def cross_count_le (grid : matrix (fin 10) (fin 10) bool) (n : ℕ) : Prop :=
  (∑ i j, if grid i j then 1 else 0) ≤ n

-- Statement of the problem in Lean 4
theorem max_crosses (grid : matrix (fin 10) (fin 10) bool) (h : unique_cross_configuration grid) :
  cross_count_le grid 18 :=
sorry -- This needs to be proved.

end max_crosses_l406_406402


namespace dot_product_of_vectors_in_regular_triangle_l406_406393

-- Definitions
def side_length := 2
def angle_ABC := 120 -- degrees
def cos_120 := -1 / 2

-- Theorem statement
theorem dot_product_of_vectors_in_regular_triangle :
  let AB := (2 : ℝ)
  let BC := (2 : ℝ)
  AB * BC * real.cos (angle_ABC * real.pi / 180) = -2 :=
by {
  have h1 : real.cos (120 * real.pi / 180) = -1 / 2,
  from real.cos_pi_div_two_add_pi_div_three,
  rw h1,
  rw [mul_assoc, mul_assoc],
  norm_num,
  sorry
}

end dot_product_of_vectors_in_regular_triangle_l406_406393


namespace least_value_of_N_l406_406146

theorem least_value_of_N : 
  ∃ N : ℕ, (∀ (S : finset ℕ), (∀ a d, S ⊆ (finset.range 100).filter (λ x, a + d * x ∈ S) → S.card > 10) → N = 11) :=
by
  sorry

end least_value_of_N_l406_406146


namespace max_sequences_in_S_l406_406436

def max_sequences : ℕ := 2048

theorem max_sequences_in_S (S : set (list bool)) (h_len : ∀ s ∈ S, s.length = 15)
  (h_diff : ∀ s t ∈ S, s ≠ t → (list.zip_with (≠) s t).count true ≥ 3) : 
  ∃ M, M = max_sequences ∧ ∀ T : set (list bool), (∀ t ∈ T, t.length = 15) → 
  (∀ s t ∈ T, s ≠ t → (list.zip_with (≠) s t).count true ≥ 3) → T.card ≤ M :=
sorry

end max_sequences_in_S_l406_406436


namespace evaluate_f_l406_406671

-- Define the polynomial function f
noncomputable def f (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - m) + 10 * x

-- Establish the roots given the conditions
lemma roots_of_f (k : ℝ) :
  (F (1) = 0 → k = 10) ∧
  (F (2) = 0 → k = 20) ∧
  (F (3) = 0 → k = 30) := sorry

-- Define the problem to prove
theorem evaluate_f :
  f(10) + f(-6) = 8104 := sorry

end evaluate_f_l406_406671


namespace correct_operation_l406_406542

theorem correct_operation : -5 * 3 = -15 :=
by sorry

end correct_operation_l406_406542


namespace find_constant_and_general_term_l406_406874

theorem find_constant_and_general_term :
  ∃ c : ℝ, 
    (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + c * n) ∧
    (a 1 = 2) ∧
    (is_geometric_sequence {a 1, a 2, a 3} ∧ ratio ≠ 1) →
    c = 2 ∧
    (∀ n : ℕ, n ≥ 1 → a n = n^2 - n + 2) := sorry

end find_constant_and_general_term_l406_406874


namespace identify_linear_inequality_l406_406541

def is_linear_inequality (P : Prop) : Prop :=
  ∃ a b c : ℝ, P = (a * x + b * y + c > 0)

theorem identify_linear_inequality :
  is_linear_inequality (frac x 2 > 0) ∧ 
  ¬ is_linear_inequality (x + y > 0) ∧
  ¬ is_linear_inequality (x^2 ≠ 3) ∧
  ¬ is_linear_inequality (2 / x ≠ 3) :=
by
  sorry

end identify_linear_inequality_l406_406541


namespace lines_perpendicular_l406_406477

open EuclideanGeometry -- Assuming a module for Euclidean geometry primitives
open Circle

-- Definitions of initial conditions
section

variables (A B C D P Q : Point) 
variables (circle : Circle)

-- Assumptions about the points and properties
axiom intersecting_chords : OnChord circle A C ∧ OnChord circle B D ∧ IntersectAt P A C B D
axiom perpendicular_perpendiculars : PerpendicularAt Q C A ∧ PerpendicularAt Q D B

-- Theorem statement to prove
theorem lines_perpendicular (h1 : intersecting_chords) (h2 : perpendicular_perpendiculars) : Perpendicular AB PQ :=
sorry

end

end lines_perpendicular_l406_406477


namespace number_to_remove_l406_406540

def mean (s : List ℕ) : ℚ := (s.sum : ℚ) / s.length

theorem number_to_remove (s : List ℕ) (x : ℕ) 
  (h₀ : s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
  (h₁ : x ∈ s)
  (h₂ : mean (List.erase s x) = 6.1) : x = 5 := sorry

end number_to_remove_l406_406540


namespace solution_set_inequality_l406_406624

noncomputable def f (x : ℝ) : ℝ := sorry

axiom h₀ : ∀ x : ℝ, f(x) + deriv (deriv f) x < Real.exp 1
axiom h₁ : f 0 = Real.exp 1 + 2

theorem solution_set_inequality :
  { x : ℝ | Real.exp x * f x > Real.exp (x + 1) + 2 } = set.Iio 0 :=
by
  sorry

end solution_set_inequality_l406_406624


namespace find_g_f_neg3_l406_406430

def f (x : ℝ) : ℝ := 5 * x^2 - 9

def g (y : ℝ) : ℝ := if y = 36 then 12 else 0

theorem find_g_f_neg3 : g (f (-3)) = 12 :=
by
  -- Here we can just assume g(36) = 12 for the proof sake, even though it's formally included in the definition.
  sorry

end find_g_f_neg3_l406_406430


namespace friendly_number_pair_a_equals_negative_three_fourths_l406_406523

theorem friendly_number_pair_a_equals_negative_three_fourths (a : ℚ) (h : (a / 2) + (3 / 4) = (a + 3) / 6) : 
  a = -3 / 4 :=
sorry

end friendly_number_pair_a_equals_negative_three_fourths_l406_406523


namespace shorts_and_jersey_different_colors_l406_406786

variable {shorts_colors : Finset String} (jersey_colors : Finset String)
variable [DecidableEq String]

def shorts_colors := {"black", "gold", "blue"}
def jersey_colors := {"white", "gold"}

noncomputable def probability_different_colors : ℚ :=
  let total_combinations := shorts_colors.card * jersey_colors.card
  let different_combinations := 
    (if "black" ∈ shorts_colors then jersey_colors.card else 0) +
    (if "gold" ∈ shorts_colors then jersey_colors.card - 1 else 0) +
    (if "blue" ∈ shorts_colors then jersey_colors.card else 0)
  (different_combinations : ℚ) / (total_combinations : ℚ)

theorem shorts_and_jersey_different_colors :
  probability_different_colors = 5 / 6 := sorry

end shorts_and_jersey_different_colors_l406_406786


namespace number_of_students_who_selected_water_l406_406396

-- Definitions for the given conditions
def percentageFruitJuice : ℝ := 0.7
def percentageWater : ℝ := 0.3
def studentsFruitJuice : ℝ := 140

-- Calculating the total number of students
def totalStudents : ℝ := studentsFruitJuice / percentageFruitJuice

-- Calculating the number of students who selected water
def studentsWater : ℝ := totalStudents * percentageWater

-- The final theorem statement
theorem number_of_students_who_selected_water :
  studentsWater = 60 :=
by
  sorry

end number_of_students_who_selected_water_l406_406396


namespace symmetric_points_y_axis_l406_406406

theorem symmetric_points_y_axis (a b : ℝ) (h₁ : b = 3) (h₂ : a = -5) : a - b = -8 :=
by {
  rw [h₁, h₂],
  norm_num
}

end symmetric_points_y_axis_l406_406406


namespace area_triple_sides_l406_406733

theorem area_triple_sides (a b : ℝ) (θ : ℝ) : 
  let A := (a * b * Real.sin θ) / 2
  let A' := (3 * a) * (3 * b) * Real.sin θ / 2
  A' = 9 * A := by
  let A := (a * b * Real.sin θ) / 2
  let A' := (3 * a) * (3 * b) * Real.sin θ / 2
  have hA' : A' = (3 * a) * (3 * b) * Real.sin θ / 2 := by
    sorry
  have hA : A = (a * b * Real.sin θ) / 2 := by
    sorry
  rw [hA, hA'],
  sorry

end area_triple_sides_l406_406733


namespace hypotenuse_of_right_triangle_l406_406199

theorem hypotenuse_of_right_triangle (a b : ℕ) (h₁ : a = 80) (h₂ : b = 150) : 
  (real.sqrt (a^2 + b^2) = 170) :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end hypotenuse_of_right_triangle_l406_406199


namespace golden_section_BC_length_l406_406826

-- Definition of a golden section point
def is_golden_section_point (A B C : ℝ) : Prop :=
  ∃ (φ : ℝ), φ = (1 + Real.sqrt 5) / 2 ∧ B = φ * C

-- The given problem translated to Lean
theorem golden_section_BC_length (A B C : ℝ) (h1 : is_golden_section_point A B C) (h2 : B - A = 6) : 
  C - B = 3 * Real.sqrt 5 - 3 ∨ C - B = 9 - 3 * Real.sqrt 5 :=
by
  sorry

end golden_section_BC_length_l406_406826


namespace numBills_is_9_l406_406102

-- Define the conditions: Mike has 45 dollars in 5-dollar bills
def totalDollars : ℕ := 45
def billValue : ℕ := 5
def numBills : ℕ := 9

-- Prove that the number of 5-dollar bills Mike has is 9
theorem numBills_is_9 : (totalDollars = billValue * numBills) → (numBills = 9) :=
by
  intro h
  sorry

end numBills_is_9_l406_406102


namespace union_M_N_l406_406791

def M : Set ℤ := { x | x^2 - x - 12 = 0 }
def N : Set ℤ := { x | x^2 + 3x = 0 }

theorem union_M_N : M ∪ N = {0, -3, 4} := by
  sorry

end union_M_N_l406_406791


namespace composite_for_positive_integers_l406_406327

def is_composite (n : ℤ) : Prop :=
  ∃ a b : ℤ, 1 < a ∧ 1 < b ∧ n = a * b

theorem composite_for_positive_integers (n : ℕ) (h_pos : 1 < n) :
  is_composite (3^(2*n+1) - 2^(2*n+1) - 6*n) := 
sorry

end composite_for_positive_integers_l406_406327


namespace suitcase_combinations_count_l406_406069

theorem suitcase_combinations_count :
  let seq := (1 : ℕ) ≤ 40 in
  let odd_count := (40 / 2) in
  let multiple_of_4_count := (40 / 4) in
  let multiple_of_5_count := (40 / 5) in
  (odd_count * multiple_of_4_count * multiple_of_5_count) = 1600 :=
by
  sorry

end suitcase_combinations_count_l406_406069


namespace find_number_l406_406019

def number_condition (N : ℝ) : Prop := 
  0.20 * 0.15 * 0.40 * 0.30 * 0.50 * N = 180

theorem find_number (N : ℝ) (h : number_condition N) : N = 1000000 :=
sorry

end find_number_l406_406019


namespace continuous_on_integrable_l406_406459

theorem continuous_on_integrable {a b : ℝ} (f : ℝ → ℝ) (h_cont : ContinuousOn f (Set.Icc a b)) :
  IntervalIntegrable f measure_theory.measure_space.volume a b := 
sorry

end continuous_on_integrable_l406_406459


namespace laboratory_painting_area_laboratory_paint_needed_l406_406937

section
variable (l w h excluded_area : ℝ)
variable (paint_per_sqm : ℝ)

def painting_area (l w h excluded_area : ℝ) : ℝ :=
  let total_area := (l * w + w * h + h * l) * 2 - (l * w)
  total_area - excluded_area

def paint_needed (painting_area paint_per_sqm : ℝ) : ℝ :=
  painting_area * paint_per_sqm

theorem laboratory_painting_area :
  painting_area 12 8 6 28.4 = 307.6 :=
by
  simp [painting_area, *]
  norm_num

theorem laboratory_paint_needed :
  paint_needed 307.6 0.2 = 61.52 :=
by
  simp [paint_needed, *]
  norm_num

end

end laboratory_painting_area_laboratory_paint_needed_l406_406937


namespace range_magnitude_b_plus_c_min_g_value_k_l406_406712

open Real

def vec_a (x : ℝ) : ℝ × ℝ := (sin x, cos x)
def vec_b (x k : ℝ) : ℝ × ℝ := (sin x, k)
def vec_c (x k : ℝ) : ℝ × ℝ := (-2 * cos x, sin x - k)
def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1^2 + v.2^2)
def g (x k : ℝ) : ℝ := 
  let ⟨a1, a2⟩ := vec_a x
  let ⟨b1, b2⟩ := vec_b x k
  let ⟨c1, c2⟩ := vec_c x k
  (a1 + b1) * c1 + (a2 + b2) * c2

theorem range_magnitude_b_plus_c (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 4) (k : ℝ) :
  let b_plus_c := (vec_b x k).fst + (vec_c x k).fst, (vec_b x k).snd + (vec_c x k).snd
  2 ≤ magnitude b_plus_c ∧ magnitude b_plus_c ≤ sqrt 8 :=
sorry

theorem min_g_value_k (k : ℝ) (hmin : ∃ x: ℝ, g x k = -3/2) : 
  k = 0 :=
sorry

end range_magnitude_b_plus_c_min_g_value_k_l406_406712


namespace compare_neg_fractions_and_neg_values_l406_406612

theorem compare_neg_fractions_and_neg_values :
  (- (3 : ℚ) / 4 > - (4 : ℚ) / 5) ∧ (-(-3 : ℤ) > -|(3 : ℤ)|) :=
by
  apply And.intro
  sorry
  sorry

end compare_neg_fractions_and_neg_values_l406_406612


namespace find_blue_balls_l406_406602

theorem find_blue_balls (N : ℕ) :
  let green_urn1 := 6
  let blue_urn1 := 4
  let green_urn2 := 10
  (green_urn1 / (green_urn1 + blue_urn1) * green_urn2 / (green_urn2 + N) +
  blue_urn1 / (green_urn1 + blue_urn1) * N / (green_urn2 + N) = 0.50) → N = 10 :=
by
  intros
  sorry

end find_blue_balls_l406_406602


namespace expression_equals_one_l406_406979

noncomputable def compute_expression : ℝ :=
  (1 - 1 / Real.cos (30 * Real.pi / 180)) *
  (1 + 1 / Real.sin (60 * Real.pi / 180)) *
  (1 - 1 / Real.sin (30 * Real.pi / 180)) *
  (1 + 1 / Real.cos (60 * Real.pi / 180))

theorem expression_equals_one : compute_expression = 1 :=
by
  sorry

end expression_equals_one_l406_406979


namespace calc_expression_l406_406969

theorem calc_expression : (Complex (1 - Complex.i) / Real.sqrt 2) ^ 32 = 1 := by sorry

end calc_expression_l406_406969


namespace surface_area_of_figure_l406_406067

theorem surface_area_of_figure 
  (block_surface_area : ℕ) 
  (loss_per_block : ℕ) 
  (number_of_blocks : ℕ) 
  (effective_surface_area : ℕ)
  (total_surface_area : ℕ) 
  (h_block : block_surface_area = 18) 
  (h_loss : loss_per_block = 2) 
  (h_blocks : number_of_blocks = 4) 
  (h_effective : effective_surface_area = block_surface_area - loss_per_block) 
  (h_total : total_surface_area = number_of_blocks * effective_surface_area) : 
  total_surface_area = 64 :=
by
  sorry

end surface_area_of_figure_l406_406067


namespace range_of_a_l406_406699

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ≥ 1 → (x^2 + a * x + 9) ≥ 0) : a ≥ -6 := 
sorry

end range_of_a_l406_406699


namespace verify_cubes_in_sexagesimal_l406_406223

open Int

-- Define a function to convert sexagesimal to decimal
def sexagesimal_to_decimal (a b : ℕ) : ℕ :=
  a * 60 + b

-- The main theorem to be proven
theorem verify_cubes_in_sexagesimal :
  ∀ (n : ℕ) (a b : ℕ), (1 ≤ n ∧ n ≤ 32) → 
  (sexagesimal_to_decimal a b = n^3 ∧ n = nat.sqrt (nat.sqrt (sexagesimal_to_decimal a b))) → 
  sexagesimal_to_decimal a b = n^3 :=
by 
  intros n a b h1 h2
  exact h2.left

end verify_cubes_in_sexagesimal_l406_406223


namespace reflect_P_y_axis_l406_406763

def P : ℝ × ℝ := (2, 1)

def reflect_y_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, point.2)

theorem reflect_P_y_axis :
  reflect_y_axis P = (-2, 1) :=
by
  sorry

end reflect_P_y_axis_l406_406763


namespace omega_range_l406_406004

noncomputable def f (ω x : ℝ) : ℝ :=
  (sin(ω * x / 2))^2 + 1/2 * sin(ω * x) - 1/2

theorem omega_range (ω : ℝ) (hω : 0 < ω):
  (∀ x, x ∈ (real.pi, 2 * real.pi) → f ω x ≠ 0) ↔
  ω ∈ (Icc 0 (1/8) ∪ Icc (1/4) (5/8)) :=
sorry

end omega_range_l406_406004


namespace correct_propositions_l406_406355

def lines_and_planes : Prop :=
  let l m n : Type -- Representing the lines as types
  let α β γ : Type -- Representing the planes as types
  let distinct_lines : Prop := l ≠ m ∧ m ≠ n ∧ l ≠ n -- Distinct lines
  let non_coincident_planes : Prop := α ≠ β ∧ β ≠ γ ∧ α ≠ γ -- Non-coincident planes

  distinct_lines →
  non_coincident_planes →
  (∀ m α β, m ⊥ α ∧ m ⊥ β → α ∥ β) ∧ -- Proposition ①
  (∀ α β γ, α ⊥ γ ∧ β ⊥ γ → ¬ (α ∥ β)) ∧ -- Proposition ② 
  (∀ m α β, m ∥ α ∧ m ∥ β → ¬ (α ∥ β)) ∧ -- Proposition ③ 
  (∀ l m α, l ∥ α ∧ m ⊆ α → ¬ (l ∥ m)) → -- Proposition ④ 
  ∃ x : ℕ, x = 1 -- The number of correct propositions is 1

theorem correct_propositions :
  lines_and_planes := by sorry

end correct_propositions_l406_406355


namespace harry_ron_difference_l406_406373

-- Define the amounts each individual paid
def harry_paid : ℕ := 150
def ron_paid : ℕ := 180
def hermione_paid : ℕ := 210

-- Define the total amount
def total_paid : ℕ := harry_paid + ron_paid + hermione_paid

-- Define the amount each should have paid
def equal_share : ℕ := total_paid / 3

-- Define the amount Harry owes to Hermione
def harry_owes : ℕ := equal_share - harry_paid

-- Define the amount Ron owes to Hermione
def ron_owes : ℕ := equal_share - ron_paid

-- Define the difference between what Harry and Ron owe Hermione
def difference : ℕ := harry_owes - ron_owes

-- Prove that the difference is 30
theorem harry_ron_difference : difference = 30 := by
  sorry

end harry_ron_difference_l406_406373


namespace number_of_exponential_functions_l406_406271

def is_exponential_function (f : ℝ → ℝ) : Prop := 
  ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ ∀ x, f x = a^x

def f1 (x : ℝ) : ℝ := 2 * 3^x
def f2 (x : ℝ) : ℝ := 3^(x+1)
def f3 (x : ℝ) : ℝ := 3^x
def f4 (x : ℝ) : ℝ := x^3

theorem number_of_exponential_functions : 
  (if is_exponential_function f1 then 1 else 0) +
  (if is_exponential_function f2 then 1 else 0) +
  (if is_exponential_function f3 then 1 else 0) +
  (if is_exponential_function f4 then 1 else 0) = 1 := 
sorry

end number_of_exponential_functions_l406_406271


namespace relayRaceOrders_l406_406788

def countRelayOrders (s1 s2 s3 s4 : String) : Nat :=
  if s1 = "Laura" then
    (if s2 ≠ "Laura" ∧ s3 ≠ "Laura" ∧ s4 ≠ "Laura" then
      if (s2 = "Alice" ∨ s2 = "Bob" ∨ s2 = "Cindy") ∧ 
         (s3 = "Alice" ∨ s3 = "Bob" ∨ s3 = "Cindy") ∧ 
         (s4 = "Alice" ∨ s4 = "Bob" ∨ s4 = "Cindy") then
        if s2 ≠ s3 ∧ s3 ≠ s4 ∧ s2 ≠ s4 then 6 else 0
      else 0
    else 0)
  else 0

theorem relayRaceOrders : countRelayOrders "Laura" "Alice" "Bob" "Cindy" = 6 := 
by sorry

end relayRaceOrders_l406_406788


namespace m_range_l406_406424

theorem m_range (x m : ℝ) (G : ℝ) (hG : G = (9 * x^2 + 27 * x + 4 * m) / 9)
  (h_square : ∃ c d : ℝ, G = (c * x + d) ^ 2) :
  5 < m ∧ m < 6 :=
begin
  sorry
end

end m_range_l406_406424


namespace most_suitable_sampling_plan_l406_406569

-- Definitions based on the conditions
def production_lines := 5
def boxes_per_line := 20

-- Sampling methods as options
inductive SamplingPlan
| A   -- Randomly select 1 box from the 100 boxes of products
| B   -- Select the last box of products from each production line
| C   -- Randomly select 1 box from the products of each production line
| D   -- Select 20 boxes of products from one of the production lines

-- Statement of the problem to be proved
theorem most_suitable_sampling_plan :
  ∀ (production_lines : ℕ) (boxes_per_line : ℕ),
    production_lines = 5 →
    boxes_per_line = 20 →
    (∃ (plan : SamplingPlan), plan = SamplingPlan.C) :=
by intros production_lines boxes_per_line h1 h2
   use SamplingPlan.C
   sorry

end most_suitable_sampling_plan_l406_406569


namespace probability_four_or_more_students_same_month_l406_406849

theorem probability_four_or_more_students_same_month (students : ℕ) (months : ℕ) (h1 : students = 37) (h2 : months = 12) :
  (∃ m, students / months ≥ 4) :=
by
  -- Definitions corresponding to the conditions
  have h_total_students : students = 37 := h1
  have h_total_months : months = 12 := h2
  -- Provided that there are 37 students and 12 months,
  -- prove that there exists a month with 4 or more students.
  sorry

end probability_four_or_more_students_same_month_l406_406849


namespace f_decreasing_on_interval_f_increasing_on_interval_g_eq_three_sol_a4_g_range_a_le_sqrt2_g_range_a_gt_sqrt2_l406_406920

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1 / x

-- The two intervals for monotonicity
def interval_decreasing : set ℝ := {x : ℝ | 0 < x ∧ x < real.sqrt 2 / 2}
def interval_increasing : set ℝ := {x : ℝ | real.sqrt 2 / 2 < x}

-- Define the function g
def g (a x : ℝ) : ℝ := a ^ (real.abs x) + 2 * a ^ x

-- The required theorem statements
theorem f_decreasing_on_interval : strict_anti_on f interval_decreasing := sorry

theorem f_increasing_on_interval : strict_mono_on f interval_increasing := sorry

theorem g_eq_three_sol_a4 (x : ℝ) : g 4 x = 3 ↔ x = 0 ∨ x = -1/2 := sorry

-- The range of g on [-1, ∞)
theorem g_range_a_le_sqrt2 (a : ℝ) (h1 : 1 < a) (h2 : a ≤ real.sqrt 2) : 
  set.range (g a) = set.Ici (a + 2 / a) := sorry

theorem g_range_a_gt_sqrt2 (a : ℝ) (h : a > real.sqrt 2) : 
  set.range (g a) = set.Ici (2 * real.sqrt 2) := sorry

end f_decreasing_on_interval_f_increasing_on_interval_g_eq_three_sol_a4_g_range_a_le_sqrt2_g_range_a_gt_sqrt2_l406_406920


namespace units_digit_6_l406_406359

theorem units_digit_6 (p : ℤ) (hp : 0 < p % 10) (h1 : (p^3 % 10) = (p^2 % 10)) (h2 : (p + 2) % 10 = 8) : p % 10 = 6 :=
by
  sorry

end units_digit_6_l406_406359


namespace notebook_pen_ratio_l406_406256

theorem notebook_pen_ratio (pen_cost notebook_total_cost : ℝ) (num_notebooks : ℕ)
  (h1 : pen_cost = 1.50) (h2 : notebook_total_cost = 18) (h3 : num_notebooks = 4) :
  (notebook_total_cost / num_notebooks) / pen_cost = 3 :=
by
  -- The steps to prove this would go here
  sorry

end notebook_pen_ratio_l406_406256


namespace perfect_square_solutions_l406_406808

theorem perfect_square_solutions (a b : ℕ) (ha : a > b) (ha_pos : 0 < a) (hb_pos : 0 < b) (hA : ∃ k : ℕ, a^2 + 4 * b + 1 = k^2) (hB : ∃ l : ℕ, b^2 + 4 * a + 1 = l^2) :
  a = 8 ∧ b = 4 ∧ (a^2 + 4 * b + 1 = (a+1)^2) ∧ (b^2 + 4 * a + 1 = (b + 3)^2) :=
by
  sorry

end perfect_square_solutions_l406_406808


namespace students_in_zack_classroom_l406_406512

theorem students_in_zack_classroom 
(T M Z : ℕ)
(h1 : T = M)
(h2 : Z = (T + M) / 2)
(h3 : T + M + Z = 69) :
Z = 23 :=
by
  sorry

end students_in_zack_classroom_l406_406512


namespace points_C_exists_l406_406752

noncomputable def distance (A B : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((A.1 - B.1) * (A.1 - B.1) + (A.2 - B.2) * (A.2 - B.2))

def area (A B C : (ℝ × ℝ)) : ℝ :=
  (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

def perimeter (A B C : (ℝ × ℝ)) : ℝ :=
  distance A B + distance A C + distance B C

theorem points_C_exists :
  let A := (0 : ℝ, 0 : ℝ)
  let B := (12 : ℝ, 0 : ℝ)
  ∃ C1 C2 : ℝ × ℝ,
  (distance A B = 12) ∧
  (area A B C1 = 144) ∧
  (perimeter A B C1 = 60) ∧
  (area A B C2 = 144) ∧
  (perimeter A B C2 = 60) ∧
  (C1 ≠ C2) ∧
  ∀ C : ℝ × ℝ,
  (area A B C = 144) ∧
  (perimeter A B C = 60) →
  (C = C1 ∨ C = C2) :=
by
  sorry

end points_C_exists_l406_406752


namespace all_elements_same_color_l406_406801

-- Assume n and k are natural numbers that are coprime, and 1 ≤ k ≤ n-1
variables (n k : ℕ)
variables (h_coprime : Nat.coprime n k) (h_k_range : 1 ≤ k ∧ k ≤ n - 1)

-- M is the set {1, 2, ..., n-1}
def M : set ℕ := { i | 1 ≤ i ∧ i < n }

-- Assume we have a coloring function coloring elements of M with color1 and color2.
inductive Color | color1 | color2

variables (color : ℕ → Color)
variables (h_color : ∀ i ∈ M n, color i = color (n - i))
variables (h_color_diff_k : ∀ i ∈ M n, i ≠ k → color i = color (|i - k|))

-- Prove that all elements in M are of the same color.
theorem all_elements_same_color : ∀ i j ∈ M n, color i = color j :=
begin
  sorry
end

end all_elements_same_color_l406_406801


namespace four_digit_numbers_no_seven_eight_nine_l406_406719

theorem four_digit_numbers_no_seven_eight_nine : 
  ∃ count : ℕ, count = 2058 ∧ 
  (∀ n : ℕ, 1000 ≤ n ∧ n < 10000 →
    (∀ d ∈ [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10], d ∈ [0, 1, 2, 3, 4, 5, 6])) :=
begin
  sorry
end

end four_digit_numbers_no_seven_eight_nine_l406_406719


namespace avg_growth_rate_l406_406568

theorem avg_growth_rate {a p q x : ℝ} (h_eq : (1 + p) * (1 + q) = (1 + x) ^ 2) : 
  x ≤ (p + q) / 2 := 
by
  sorry

end avg_growth_rate_l406_406568


namespace solve_for_L_l406_406416

noncomputable def numberOfLargeBottles : ℕ :=
  let L := (242.55 / 0.1866).round
  L

theorem solve_for_L :
  ∃ L : ℕ, (L : ℝ) ≈ 1300 ∧ 
  ( (1.89 * L + 750 * 1.38) / (L + 750) ≈ 1.7034 ) :=
by
  existsi (numberOfLargeBottles)
  split
  · assumption
  · sorry

end solve_for_L_l406_406416


namespace number_of_routes_600_l406_406305

-- Define the problem conditions
def number_of_routes (total_cities : Nat) (pick_cities : Nat) (selected_cities : List Nat) : Nat := sorry

-- The number of ways to pick and order 3 cities from remaining 5
def num_ways_pick_three (total_cities : Nat) (pick_cities : Nat) : Nat :=
  Nat.factorial total_cities / Nat.factorial (total_cities - pick_cities)

-- The number of ways to choose positions for M and N
def num_ways_positions (total_positions : Nat) (pick_positions : Nat) : Nat :=
  Nat.choose total_positions pick_positions

-- The main theorem to prove
theorem number_of_routes_600 :
  number_of_routes 7 5 [M, N] = num_ways_pick_three 5 3 * num_ways_positions 4 2 :=
  by sorry

end number_of_routes_600_l406_406305


namespace identify_fake_coins_l406_406104

-- Defining the problem with relevant conditions
def Coin := ℕ

-- Assigning properties to the coins 
structure CoinsCondition :=
  (all_coins : list Coin)
  (genuine_coins : list Coin)
  (fake_coins : list Coin)
  (fake_heavy : Coin)
  (fake_light : Coin)
  (weighings : list (Coin × Coin))
  (weighing_results : list (ordering))

-- Problem statement in Lean
theorem identify_fake_coins (conds : CoinsCondition) :
  (length conds.all_coins = 5) →
  (length conds.genuine_coins = 3) →
  (length conds.fake_coins = 2) →
  (conds.fake_coins = [conds.fake_heavy, conds.fake_light]) →
  (∀ (w ∈ conds.weighings), w.fst ∈ conds.all_coins ∧ w.snd ∈ conds.all_coins) →
  (length conds.weighings = 3) →
  -- Ensuring fake coins are identified from weighings
  (∃ (heavy light : Coin), heavy ∈ conds.fake_coins ∧ light ∈ conds.fake_coins ∧ 
    ∀ wr ∈ conds.weighing_results, 
      ((wr = ordering.lt ∧ heavy = conds.fake_heavy ∧ light = conds.fake_light) ∨
       (wr = ordering.gt ∧ heavy = conds.fake_light ∧ light = conds.fake_heavy))):
  True := sorry

end identify_fake_coins_l406_406104


namespace Sasha_added_digit_l406_406068

noncomputable def Kolya_number : Nat := 45 -- Sum of all digits 0 to 9

theorem Sasha_added_digit (d x : Nat) (h : 0 ≤ d ∧ d ≤ 9) (h1 : 0 ≤ x ∧ x ≤ 9) (condition : Kolya_number - d + x ≡ 0 [MOD 9]) : x = 0 ∨ x = 9 := 
sorry

end Sasha_added_digit_l406_406068


namespace circumcircle_radius_l406_406399

-- Define the conditions
structure IsoscelesTriangle (A B C : Type) where
  base : C
  condition : is_isosceles B C A

def divides_in_ratio (D B C : Type) (ratio : ℕ) : Prop := sorry -- Assume this is the correct definition.

def midpoint (E A D : Type) : Prop := sorry -- Assume this is the correct definition.

-- Define the problem
theorem circumcircle_radius (A B C D E : Type) 
  [is_isosceles B C A]
  [divides_in_ratio D B C 3]
  [midpoint E A D]
  (BE CE : ℝ)
  (h1 : BE = real.sqrt 7)
  (h2 : CE = 3) :
  ∃ R : ℝ, R = 8 / 3 := sorry

end circumcircle_radius_l406_406399


namespace trig_expr_evaluation_l406_406984

theorem trig_expr_evaluation :
    (1 - (1 / cos (Real.pi / 6))) * 
    (1 + (1 / sin (Real.pi / 3))) * 
    (1 - (1 / sin (Real.pi / 6))) * 
    (1 + (1 / cos (Real.pi / 3))) = 1 := 
by
  let cos_30 := cos (Real.pi / 6)
  let sin_60 := sin (Real.pi / 3)
  let sin_30 := sin (Real.pi / 6)
  let cos_60 := cos (Real.pi / 3)
  have h_cos_30 : cos_30 = (Real.sqrt 3) / 2 := sorry
  have h_sin_60 : sin_60 = (Real.sqrt 3) / 2 := sorry
  have h_sin_30 : sin_30 = 1 / 2 := sorry
  have h_cos_60 : cos_60 = 1 / 2 := sorry
  sorry

end trig_expr_evaluation_l406_406984


namespace portia_high_school_students_l406_406458

theorem portia_high_school_students
  (L P M : ℕ)
  (h1 : P = 4 * L)
  (h2 : M = 2 * L)
  (h3 : P + L + M = 4200) :
  P = 2400 :=
sorry

end portia_high_school_students_l406_406458


namespace minimum_trucks_on_lot_l406_406819

variable (total_trucks : ℕ) (rented_percentage : ℝ) (max_rented_trucks : ℕ)
variable (initial_trucks : total_trucks = 24) (returned_percentage : rented_percentage = 0.5) (max_trucks_rented : max_rented_trucks = 24)

theorem minimum_trucks_on_lot (total_trucks : ℕ) (rented_percentage : ℝ) (max_rented_trucks : ℕ)
  (initial_trucks : total_trucks = 24) (returned_percentage : rented_percentage = 0.5) (max_trucks_rented : max_rented_trucks = 24) :
  max_rented_trucks / 2 = 12 :=
by sorry

end minimum_trucks_on_lot_l406_406819


namespace problem_l406_406460

theorem problem (a : ℕ → ℝ) (A1 : a 1 = 1)
  (An : ∀ n, 2 ≤ n → n ≤ 10 → a n = (1 / 2) * (a (n - 1) + 2 / a (n - 1))) :
  0 < a 10 - Real.sqrt 2 ∧ a 10 - Real.sqrt 2 < 10 ^ (-370) := by
  sorry

end problem_l406_406460


namespace total_spent_on_computer_l406_406058

def initial_cost_of_pc : ℕ := 1200
def sale_price_old_card : ℕ := 300
def cost_new_card : ℕ := 500

theorem total_spent_on_computer : 
  (initial_cost_of_pc + (cost_new_card - sale_price_old_card)) = 1400 :=
by
  sorry

end total_spent_on_computer_l406_406058


namespace eggs_purchased_l406_406834

theorem eggs_purchased (initial final : ℕ) (h1 : initial = 98) (h2 : final = 106) : final - initial = 8 := by
  rw [h1, h2]
  exact rfl

end eggs_purchased_l406_406834


namespace total_students_in_class_l406_406816

def current_students : ℕ := 6 * 3
def students_bathroom : ℕ := 5
def students_canteen : ℕ := 5 * 5
def group1_students : ℕ := 4
def group2_students : ℕ := 5
def group3_students : ℕ := 6
def group4_students : ℕ := 3
def new_group_students : ℕ := group1_students + group2_students + group3_students + group4_students
def germany_students : ℕ := 3
def france_students : ℕ := 4
def norway_students : ℕ := 3
def italy_students : ℕ := 2
def spain_students : ℕ := 2
def australia_students : ℕ := 1
def foreign_exchange_students : ℕ :=
  germany_students + france_students + norway_students + italy_students + spain_students + australia_students

def total_students : ℕ :=
  current_students + students_bathroom + students_canteen + new_group_students + foreign_exchange_students

theorem total_students_in_class : total_students = 81 := by
  rfl  -- Reflective equality since total_students already sums to 81 based on the definitions

end total_students_in_class_l406_406816


namespace overall_gain_is_10_percent_l406_406132

noncomputable def total_cost_price : ℝ := 700 + 500 + 300
noncomputable def total_gain : ℝ := 70 + 50 + 30
noncomputable def overall_gain_percentage : ℝ := (total_gain / total_cost_price) * 100

theorem overall_gain_is_10_percent :
  overall_gain_percentage = 10 :=
by
  sorry

end overall_gain_is_10_percent_l406_406132


namespace new_ratio_of_milk_to_water_l406_406751

theorem new_ratio_of_milk_to_water
  (total_volume : ℕ) (initial_ratio_milk : ℕ) (initial_ratio_water : ℕ) (added_water : ℕ)
  (h_total_volume : total_volume = 45)
  (h_initial_ratio : initial_ratio_milk = 4 ∧ initial_ratio_water = 1)
  (h_added_water : added_water = 11) :
  let initial_milk := (initial_ratio_milk * total_volume) / (initial_ratio_milk + initial_ratio_water)
  let initial_water := (initial_ratio_water * total_volume) / (initial_ratio_milk + initial_ratio_water)
  let new_water := initial_water + added_water
  let gcd := Nat.gcd initial_milk new_water
  (initial_milk / gcd : ℕ) = 9 ∧ (new_water / gcd : ℕ) = 5 :=
by
  sorry

end new_ratio_of_milk_to_water_l406_406751


namespace negation_of_proposition_true_l406_406869

theorem negation_of_proposition_true :
  (¬ (∀ x: ℝ, x^2 < 1 → -1 < x ∧ x < 1)) ↔ (∃ x: ℝ, x^2 ≥ 1 ∧ (x ≤ -1 ∨ x ≥ 1)) :=
by
  sorry

end negation_of_proposition_true_l406_406869


namespace range_of_a_l406_406697

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 - 2 * a

theorem range_of_a (a : ℝ) :
  (∃ (x₀ : ℝ), x₀ ≤ a ∧ f x₀ a ≥ 0) ↔ (a ∈ Set.Icc (-1 : ℝ) 0 ∪ Set.Ici 2) := by
  sorry

end range_of_a_l406_406697


namespace sum_of_areas_of_circles_l406_406172

-- Definitions of conditions
def r : ℝ := by sorry  -- radius of the circle at vertex A
def s : ℝ := by sorry  -- radius of the circle at vertex B
def t : ℝ := by sorry  -- radius of the circle at vertex C

axiom sum_radii_r_s : r + s = 6
axiom sum_radii_r_t : r + t = 8
axiom sum_radii_s_t : s + t = 10

-- The statement we want to prove
theorem sum_of_areas_of_circles : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by
  -- Use given axioms and properties of the triangle and circles
  sorry

end sum_of_areas_of_circles_l406_406172


namespace ads_ratio_l406_406891

noncomputable def ads_on_pages (a1 a2 a3 a4 : ℕ) : Prop :=
  a1 = 12 ∧
  a2 = 2 * a1 ∧
  a3 = a2 + 24 ∧
  2 / 3 * (a1 + a2 + a3 + a4) = 68

theorem ads_ratio (a1 a2 a3 a4 : ℕ) (h : ads_on_pages a1 a2 a3 a4) : a4 / a2 = 3 / 4 :=
by
  -- Assuming (a1 = 12, a2 = 24, a3 = 48, a4 = 18)
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest,
  cases h_rest with h3 h4,
  have h_tot := calc
    2 / 3 * (a1 + a2 + a3 + a4) = 68 : by exact h4
    ... = 102 := by norm_num,
  suffices : 84 + a4 = 102,
  suffices : a4 = 18,
  show a4 / a2 = 3 / 4,
  sorry

end ads_ratio_l406_406891


namespace find_a_l406_406336

-- Definitions given the conditions in the problem
def M (a : ℝ) : set (ℝ × ℝ) := { p | (p.1 - a)^2 + (p.2 - 2)^2 = 4 }
def l : set (ℝ × ℝ) := { p | p.1 - p.2 + 3 = 0 }
def chord_length (l : set (ℝ × ℝ)) (circle : set (ℝ × ℝ)) : ℝ := 
  sorry -- Placeholder definition for chord length

-- Problem statement:
theorem find_a (a : ℝ) :
  let center := (a, 2)
  let radius := 2
  (distance l center = 0) ∧ (chord_length l (M a) = 4) → a = -1 :=
by
  sorry

end find_a_l406_406336


namespace stock_value_sale_l406_406910

theorem stock_value_sale
  (X : ℝ)
  (h1 : 0.20 * X * 0.10 - 0.80 * X * 0.05 = -350) :
  X = 17500 := by
  -- Proof goes here
  sorry

end stock_value_sale_l406_406910


namespace probability_of_both_selected_l406_406557

-- Define the probability of Carol getting selected
def pCarol : ℝ := 4/5

-- Define the probability of Bernie getting selected
def pBernie : ℝ := 3/5

-- Define the statement proving the probability that both Carol and Bernie get selected
theorem probability_of_both_selected : (pCarol * pBernie) = 12/25 :=
by
  simp [pCarol, pBernie]
  sorry

end probability_of_both_selected_l406_406557


namespace total_swordfish_caught_correct_l406_406464

-- Define the parameters and the catch per trip
def shelly_catch_per_trip : ℕ := 5 - 2
def sam_catch_per_trip (shelly_catch : ℕ) : ℕ := shelly_catch - 1
def total_catch_per_trip (shelly_catch sam_catch : ℕ) : ℕ := shelly_catch + sam_catch
def total_trips : ℕ := 5

-- Define the total number of swordfish caught over the trips
def total_swordfish_caught : ℕ :=
  let shelly_catch := shelly_catch_per_trip
  let sam_catch := sam_catch_per_trip shelly_catch
  let total_catch := total_catch_per_trip shelly_catch sam_catch
  total_catch * total_trips

-- The theorem states that the total swordfish caught is 25
theorem total_swordfish_caught_correct : total_swordfish_caught = 25 := by
  sorry

end total_swordfish_caught_correct_l406_406464


namespace omega_range_l406_406003

noncomputable def f (ω x : ℝ) : ℝ :=
  (sin(ω * x / 2))^2 + 1/2 * sin(ω * x) - 1/2

theorem omega_range (ω : ℝ) (hω : 0 < ω):
  (∀ x, x ∈ (real.pi, 2 * real.pi) → f ω x ≠ 0) ↔
  ω ∈ (Icc 0 (1/8) ∪ Icc (1/4) (5/8)) :=
sorry

end omega_range_l406_406003


namespace value_of_f_g_l406_406683

def f (x : ℝ) : ℝ := x^2 - 3*x + 7
def g (x : ℝ) : ℝ := x + 4

theorem value_of_f_g (h₁ : f (g 3) = 35) (h₂ : g (f 3) = 11) : f (g 3) - g (f 3) = 24 :=
by
  calc
    f (g 3) - g (f 3) = 35 - 11 := by rw [h₁, h₂]
                      _         = 24 := by norm_num

end value_of_f_g_l406_406683


namespace find_k_l406_406435

noncomputable def volume_of_tetrahedron (P1 P2 P3 P4 : ℝ × ℝ × ℝ) : ℝ :=
  abs (1/6 * (matrix.det ![
    [P1.1, P1.2, P1.3, 1],
    [P2.1, P2.2, P2.3, 1],
    [P3.1, P3.2, P3.3, 1],
    [P4.1, P4.2, P4.3, 1]
  ]))

theorem find_k (k : ℝ) :
  let P1 := (1, 2, 3)
      P2 := (2, 4, 1)
      P3 := (1, k, 5)
      P4 := (4, k+1, 3) in
  volume_of_tetrahedron P1 P2 P3 P4 = 1 →
  (k = 1 ∨ k = -2) :=
sorry

end find_k_l406_406435


namespace vaccine_II_more_effective_l406_406134

noncomputable theory

structure VaccineResult where
  n : ℕ   -- number of people vaccinated
  infected : ℕ   -- number of people infected

def infection_rate : ℝ := 0.2

def vaccine_I_result : VaccineResult := { n := 8, infected := 0 }
def vaccine_II_result : VaccineResult := { n := 25, infected := 1 }

theorem vaccine_II_more_effective : 
  (vaccine_I_result.infected = 0 → P(vaccine_I_result) > P(vaccine_II_result)) → (vaccine_II_result.infected = 1 → P(vaccine_I_result) < P(vaccine_II_result)) → 
  (1 / (vaccine_I_result.n ^ infection_rate) < 1 / (vaccine_II_result.n ^ infection_rate)) :=
by
sorry

end vaccine_II_more_effective_l406_406134


namespace AreaOfEquilateralTriangle_l406_406611

open Real

variable (ω1 ω2 ω3 : Circle) (r : ℝ)
variable (P1 P2 P3 : Point) 
variable (tangent : ∀ (i j : Circle), externally_tangent i j)

-- Conditions
def Circles (ω1 ω2 ω3 : Circle) : Prop :=
  ω1.radius = 5 ∧ ω2.radius = 5 ∧ ω3.radius = 5 ∧
  tangent ω1 ω2 ∧ tangent ω2 ω3 ∧ tangent ω3 ω1
  
def PointsOnCircles (P1 P2 P3 : Point) (ω1 ω2 ω3 : Circle) : Prop :=
  P1 ∈ ω1 ∧ P2 ∈ ω2 ∧ P3 ∈ ω3

def EquilateralTriangle (P1 P2 P3 : Point) (ω1 ω2 ω3 : Circle) : Prop :=
  dist P1 P2 = dist P2 P3 ∧ dist P2 P3 = dist P3 P1 ∧ dist P3 P1 = dist P1 P2 ∧
  tangent_line_through P1 P2 ω1 ∧ tangent_line_through P2 P3 ω2 ∧ tangent_line_through P3 P1 ω3

-- Statement
theorem AreaOfEquilateralTriangle (P1 P2 P3 : Point) (ω1 ω2 ω3 : Circle) (r : ℝ) :
  Circles ω1 ω2 ω3 ∧ PointsOnCircles P1 P2 P3 ω1 ω2 ω3 ∧ EquilateralTriangle P1 P2 P3 ω1 ω2 ω3 →
  ∃ a b : ℕ, (Area P1 P2 P3 = sqrt a + sqrt b) ∧ (a + b = 675) := 
by 
  sorry

end AreaOfEquilateralTriangle_l406_406611


namespace smallest_n_for_product_sequence_l406_406302

theorem smallest_n_for_product_sequence (n : ℕ) 
  (h₀ : n > 0)
  (h₁ : (∏ k in finset.range n.succ, 100^(k/13)) > 10^6) : 
  n = 9 :=
by 
  sorry

end smallest_n_for_product_sequence_l406_406302


namespace remainder_when_112222333_divided_by_37_l406_406200

theorem remainder_when_112222333_divided_by_37 : 112222333 % 37 = 0 :=
by
  sorry

end remainder_when_112222333_divided_by_37_l406_406200


namespace radius_of_circle_with_chords_l406_406331

theorem radius_of_circle_with_chords 
  (chord1_length : ℝ) (chord2_length : ℝ) (distance_between_midpoints : ℝ) 
  (h1 : chord1_length = 9) (h2 : chord2_length = 17) (h3 : distance_between_midpoints = 5) : 
  ∃ r : ℝ, r = 85 / 8 :=
by
  sorry

end radius_of_circle_with_chords_l406_406331


namespace correct_statements_l406_406661

variable {r : ℝ}

theorem correct_statements (r : ℝ) : 
  (|r| → (1), (4)) → 
  ((|r| → (1))) :=
by
  sorry

end correct_statements_l406_406661


namespace sum_of_areas_of_circles_l406_406181

theorem sum_of_areas_of_circles (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  ∃ (π : ℝ), π * (r^2 + s^2 + t^2) = 56 * π :=
by {
  sorry
}

end sum_of_areas_of_circles_l406_406181


namespace find_constant_term_l406_406155

-- Definitions based on conditions:
def sum_of_coeffs (n : ℕ) : ℕ := 4 ^ n
def sum_of_binom_coeffs (n : ℕ) : ℕ := 2 ^ n
def P_plus_Q_equals (n : ℕ) : Prop := sum_of_coeffs n + sum_of_binom_coeffs n = 272

-- Constant term in the binomial expansion:
def constant_term (n r : ℕ) : ℕ := Nat.choose n r * (3 ^ (n - r))

-- The proof statement
theorem find_constant_term : 
  ∃ n r : ℕ, P_plus_Q_equals n ∧ n = 4 ∧ r = 1 ∧ constant_term n r = 108 :=
by {
  sorry
}

end find_constant_term_l406_406155


namespace frances_towel_weight_in_ounces_l406_406098

theorem frances_towel_weight_in_ounces :
  (∀ Mary_towels Frances_towels : ℕ,
    Mary_towels = 4 * Frances_towels →
    Mary_towels = 24 →
    (Mary_towels + Frances_towels) * 2 = 60 →
    Frances_towels * 2 * 16 = 192) :=
by
  intros Mary_towels Frances_towels h1 h2 h3
  sorry

end frances_towel_weight_in_ounces_l406_406098


namespace find_article_cost_l406_406590

theorem find_article_cost (total_cost : ℝ) (num_articles : ℕ) (discount : ℝ) (tax_rate : ℝ) (cost_with_tax : ℝ)
  (h_cost_with_tax : cost_with_tax = 1649.43) 
  (h_num_articles : num_articles = 3) 
  (h_discount : discount = 0.24) 
  (h_tax_rate : tax_rate = 0.08) : 
    let P := cost_with_tax / (3 * 0.76 * 1.08) 
    in P = 669.99 :=
by
  let P := 1649.43 / (3 * 0.76 * 1.08); 
  have hP : P = 669.99, by sorry;
  exact hP

end find_article_cost_l406_406590


namespace solve_equation_l406_406655

theorem solve_equation (x : ℝ) :
  (16 * x - x^2) / (x + 2) * (x + (16 - x) / (x + 2)) = 60 → x = 4 := by
  sorry

end solve_equation_l406_406655


namespace perfect_cubes_between_l406_406720

open Nat

theorem perfect_cubes_between :
  let lower_bound := 2^10 + 1
  let upper_bound := 2^16 + 1
  ∃ n : ℕ, n = 30 ∧ 
  (∀ k, (∃ m₁, m₁^3 = k ∧ lower_bound ≤ k ∧ k ≤ upper_bound) ↔ (11 ≤ m₁ ∧ m₁ ≤ 40)) :=
by
  -- defining lower and upper bounds
  let lower_bound := 2^10 + 1
  let upper_bound := 2^16 + 1
  
  -- defining the expected answer
  let answer := 30

  -- creating existential statement
  exists answer 

  -- defining condition on cubes within bounds
  constructor
  · rfl
  · intro k
    split
    · rintro ⟨m₁, ⟨h₁, h₂, h₃⟩⟩
      exact ⟨h₂.symm ▸ h₃, h₂.symm ▸ h₂⟩
    · rintro ⟨h₁, h₂⟩
      exact ⟨h₁ ^ 3, ⟨rfl, h₂, h₁⟩⟩
  sorry

end perfect_cubes_between_l406_406720


namespace BY_eq_CX_l406_406117

open Point Segment Triangle

theorem BY_eq_CX 
  (A B C M N X Y : Point)
  (H1 : is_midpoint M A B)
  (H2 : is_midpoint N B C)
  (H3 : on_extension X N M)
  (H4 : between Y N X)
  (H5 : length (segment M N) = length (segment X Y)) : 
  length (segment B Y) = length (segment C X) := 
by
  sorry


end BY_eq_CX_l406_406117


namespace sphere_surface_area_l406_406344

noncomputable def surface_area (R : ℝ) : ℝ := 4 * π * R^2

theorem sphere_surface_area
    (R : ℝ) -- radius of the sphere
    (A B C : EuclideanGeometry.Point ℝ) -- points on the surface of the sphere
    (hRpos : R > 0) -- the radius is positive
    (AB_eq : EuclideanGeometry.distance A B = 2)
    (AC_eq : EuclideanGeometry.distance A C = 2)
    (angle_BAC : EuclideanGeometry.angle B A C = 120 * π / 180)
    (d_eq : EuclideanGeometry.distance (EuclideanGeometry.ProjPlane 0 (EuclideanGeometry.Plane.mk A B C)) 0 = R / 2) :
    surface_area R = (64 / 3) * π := sorry

end sphere_surface_area_l406_406344


namespace no_such_function_exists_l406_406311

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x - f y) = 1 + x - y := by
  sorry

end no_such_function_exists_l406_406311


namespace vector_identity_l406_406658

variables {Point : Type*} [add_comm_group Point]
variables {A B C D : Point}

-- Definition of vector operation (difference between points)
def vector (P Q : Point) : Point := P - Q

theorem vector_identity (A B C D : Point) : 
  vector D A + vector C D - vector C B = vector B A :=
by sorry

end vector_identity_l406_406658


namespace domain_of_function_y_l406_406299

def domain (f : ℝ → ℝ) : Set ℝ :=
  {x | ∃ y, f x = y}

noncomputable def function_y (x : ℝ) : ℝ :=
  (sqrt (x + 1)) + Real.log (x - 2)

theorem domain_of_function_y :
  domain function_y = {x | 2 < x} := by
  sorry -- proof placeholder

end domain_of_function_y_l406_406299


namespace sum_is_zero_l406_406214

variable (a : ℕ → ℝ) (n : ℕ)
variable (n_eq : n = 14)
variable (circle_cond : ∀ i, a i = a ((i - 1) % 14) + a ((i + 1) % 14))

theorem sum_is_zero (n_eq : n = 14) (circle_cond : ∀ i, a i = a ((i - 1) % 14) + a ((i + 1) % 14)) : 
    ∑ i in Finset.range 14, a i = 0 := 
sorry

end sum_is_zero_l406_406214


namespace question_1_question_2_question_3_l406_406044

noncomputable def P (α : ℝ) : ℝ × ℝ :=
  (2 + Real.cos α, Real.sin α)

def trajectory_C (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 1

def polar_line_eq (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ + Real.pi / 4) = 2 * Real.sqrt 2

def cartesian_line_eq (x y : ℝ) : Prop :=
  x + y - 4 = 0

def max_distance_to_line (x y : ℝ) : ℝ :=
  (Real.abs (x + y - 4)) / Real.sqrt 2 + 1

theorem question_1 (α : ℝ) :
  ∃ x y, P α = (x, y) ∧ trajectory_C x y := by
  sorry

theorem question_2 (x y : ℝ) :
  polar_line_eq (Real.sqrt ((x - 2)^2 + y^2)) (Real.atan2 y (x - 2)) → cartesian_line_eq x y := by
  sorry

theorem question_3 (α : ℝ) :
  ∃ x y, P α = (x, y) ∧ max_distance_to_line x y = Real.sqrt 2 + 1 := by
  sorry

end question_1_question_2_question_3_l406_406044


namespace radius_circumsphere_exists_smaller_radius_l406_406140

structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

structure Tetrahedron :=
  (S A B C : Point3D)
  (length_SA : ℝ)
  (length_SB : ℝ)
  (length_SC : ℝ)
  (perpendicular_SA_SB : SA.z = S.z ∧ SA.y = S.y ∧ SB.x = S.x ∧ SB.z = S.z)
  (perpendicular_SA_SC : SA.z = S.z ∧ SA.x = S.x ∧ SC.x = S.x ∧ SC.y = S.y)
  (perpendicular_SB_SC : SB.x = S.x ∧ SB.y = S.y ∧ SC.y = S.y ∧ SC.z = S.z)
  (length_SA_cond : length_SA = 2)
  (length_SB_cond : length_SB = 3)
  (length_SC_cond : length_SC = 6)

def circumsphere_radius (T : Tetrahedron) : ℝ := 
  let d := (isa : T.SA - T.S.x, isa_s : T.SA - T.S.y, isa_z : T.SA - T.S.z)
  let db := (isb : T.SA - T.S.x, isb_s : T.SA - T.S.y, isb_z : T.SA - T.S.z)
  let dc := (isc : T.SA - T.S.x, isc_s : T.SA - T.S.y, isc_z : T.SA - T.S.z)

  sorry  

theorem radius_circumsphere (T : Tetrahedron) : circumsphere_radius T = 7 / 2 := sorry

theorem exists_smaller_radius (T : Tetrahedron) : ∃ (r : ℝ), r < 7 / 2 ∧ ∀ (p : Point3D), 
  ((p ≠ T.S ∧ p ≠ T.A ∧ p ≠ T.B ∧ p ≠ T.C) → 
  dist p {circumsphere_radius := 7 / 2} < r ∧ r > 0) := 
  sorry

end radius_circumsphere_exists_smaller_radius_l406_406140


namespace prob_not_all_same_correct_l406_406529

-- Define the number of dice and the number of sides per die
def num_dice : ℕ := 5
def sides_per_die : ℕ := 8

-- Total possible outcomes when rolling num_dice dice with sides_per_die sides each
def total_outcomes : ℕ := sides_per_die ^ num_dice

-- Outcomes where all dice show the same number
def same_number_outcomes : ℕ := sides_per_die

-- Probability that all five dice show the same number
def prob_same : ℚ := same_number_outcomes / total_outcomes

-- Probability that not all five dice show the same number
def prob_not_same : ℚ := 1 - prob_same

-- The expected probability that not all dice show the same number
def expected_prob_not_same : ℚ := 4095 / 4096

-- The main theorem to prove
theorem prob_not_all_same_correct : prob_not_same = expected_prob_not_same := by
  sorry  -- Proof will be filled in by the user

end prob_not_all_same_correct_l406_406529


namespace has_real_root_neg_one_l406_406229

theorem has_real_root_neg_one : 
  (-1)^2 - (-1) - 2 = 0 :=
by 
  sorry

end has_real_root_neg_one_l406_406229


namespace nonagon_side_length_l406_406884

theorem nonagon_side_length (x : ℝ) :
  let A_small := 64,
      A_large := 3136,
      A_x := x^2,
      A_middle := A_x - A_small,
      A_outer := A_large - A_x in
  (A_middle / A_outer) = 1 / 7 → x = 8 * Real.sqrt 7 :=
by
  sorry

end nonagon_side_length_l406_406884


namespace median_of_sequence_l406_406292

theorem median_of_sequence : ∃ median, median = 71 ∧
(∀ S : list ℕ, 
 (∀ n, 1 ≤ n ∧ n ≤ 100 → count n S = n + 1) → 
 (length S = 5150) →
 (∃ m1 m2 : ℕ, 1 ≤ m1 ∧ m1 ≤ 5150 ∧ 1 ≤ m2 ∧ m2 ≤ 5150 ∧ 
 (nth S (m1 - 1) = some 71) ∧ 
 (nth S (m2 - 1) = some 71) ∧ 
 m1 = 2575 ∧ m2 = 2576) → 
 (median = 71)) :=
by
  sorry

end median_of_sequence_l406_406292


namespace proof_l406_406441

noncomputable def M : set ℝ := {x | (x + 2) / (x - 1) ≤ 0}
def N : set ℕ := {n | true}

theorem proof : M ∩ (N : set ℝ) = {0} := sorry

end proof_l406_406441


namespace max_n_Sn_gt_zero_l406_406428

def a (n : ℕ) : ℝ := sorry -- define your Arithmetic sequence here
def S (n : ℕ) : ℝ := (n : ℝ) * (a 1 + a n) / 2

theorem max_n_Sn_gt_zero :
  ∀ (a : ℕ → ℝ), (a 1 > 0) →
  (a 8)^2 + (a 8) - 2023 = 0 →
  (a 9)^2 + (a 9) - 2023 = 0 →
  ∀ m, S m > 0 → m ≤ 15 :=
begin
  -- problem's conditions with proof steps are encapsulated here
  sorry
end

end max_n_Sn_gt_zero_l406_406428


namespace percentage_increase_l406_406233

theorem percentage_increase (initial_number final_number : ℕ) (h1 : initial_number = 800) (h2 : final_number = 1680) :
  ((final_number - initial_number : ℤ).toRat / initial_number : ℚ) * 100 = 110 := 
by 
  sorry

end percentage_increase_l406_406233


namespace number_of_selections_l406_406022

theorem number_of_selections : 
  let S := { i ∈ (Finset.range 15) | 1 ≤ i }
  in (∃ 
    (a1 a2 a3 ∈ S), 
    a1 < a2 ∧ 
    a2 < a3 ∧
    a2 - a1 ≥ 3 ∧ 
    a3 - a2 ≥ 3
  ) → 
  (S.card = 120) :=
by sorry

end number_of_selections_l406_406022


namespace log_sum_implies_product_eq_10_l406_406919

theorem log_sum_implies_product_eq_10 (a b : ℝ) (h : log a + log b = 1) : a * b = 10 :=
sorry

end log_sum_implies_product_eq_10_l406_406919


namespace greatest_monthly_average_price_drop_is_March_l406_406861

variable minPriceJan : ℕ := 30
variable maxPriceJan : ℕ := 34
variable minPriceFeb : ℕ := 37
variable maxPriceFeb : ℕ := 42
variable minPriceMar : ℕ := 32
variable maxPriceMar : ℕ := 34
variable minPriceApr : ℕ := 35
variable maxPriceApr : ℕ := 39
variable minPriceMay : ℕ := 33
variable maxPriceMay : ℕ := 33
variable minPriceJun : ℕ := 29
variable maxPriceJun : ℕ := 31

def average (min max : ℕ) : ℝ := (min + max) / 2

-- Calculate averages
def avgJan := average minPriceJan maxPriceJan
def avgFeb := average minPriceFeb maxPriceFeb
def avgMar := average minPriceMar maxPriceMar
def avgApr := average minPriceApr maxPriceApr
def avgMay := average minPriceMay maxPriceMay
def avgJun := average minPriceJun maxPriceJun

-- Calculate differences
def deltaJanFeb := avgFeb - avgJan
def deltaFebMar := avgMar - avgFeb
def deltaMarApr := avgApr - avgMar
def deltaAprMay := avgMay - avgApr
def deltaMayJun := avgJun - avgMay

theorem greatest_monthly_average_price_drop_is_March : deltaFebMar = -6.5 := sorry

end greatest_monthly_average_price_drop_is_March_l406_406861


namespace factorize_expression_l406_406641

variable {a b : ℕ}

theorem factorize_expression (a b : ℕ) : 9 * a - 6 * b = 3 * (3 * a - 2 * b) :=
by
  sorry

end factorize_expression_l406_406641


namespace smallest_positive_period_maximum_value_tan_a_value_l406_406694

noncomputable def f (x : ℝ) := sqrt 3 * sin (2 * x) + cos (2 * x) + 4

theorem smallest_positive_period (x : ℝ) : ∀ x, f (x + π) = f x := 
by 
  sorry

theorem maximum_value : ∃ x : ℝ, f x = 6 := 
by 
  sorry

theorem tan_a_value (a : ℝ) (h : f a = 5) : tan a = 0 ∨ tan a = sqrt 3 :=
by 
  sorry

end smallest_positive_period_maximum_value_tan_a_value_l406_406694


namespace necessary_but_not_sufficient_l406_406918

theorem necessary_but_not_sufficient (x : ℝ) :
  (x - 1) * (x + 2) = 0 → (x = 1 ∨ x = -2) ∧ (x = 1 → (x - 1) * (x + 2) = 0) ∧ ¬((x - 1) * (x + 2) = 0 ↔ x = 1) :=
by
  sorry

end necessary_but_not_sufficient_l406_406918


namespace problem_equiv_conditions_l406_406314

-- Define an instance of ℝ and all real values
variable (x : ℝ)

-- Define the condition
def condition := (2 / (x + 2)) + (4 / (x + 8)) >= (4 / 3)

-- Define the set of correct answers
def solution := x ∈ set.Ioc (-2 : ℝ) 1 ∨ x = 1

-- State the equivalence
theorem problem_equiv_conditions : 
  ∃ x : ℝ, condition x ↔ solution x :=
by
  sorry

end problem_equiv_conditions_l406_406314


namespace expected_worth_flip_l406_406931

/-- A biased coin lands on heads with probability 2/3 and on tails with probability 1/3.
Each heads flip gains $5, and each tails flip loses $9.
If three consecutive flips all result in tails, then an additional loss of $10 is applied.
Prove that the expected worth of a single coin flip is -1/27. -/
theorem expected_worth_flip :
  let P_heads := 2 / 3
  let P_tails := 1 / 3
  (P_heads * 5 + P_tails * -9) - (P_tails ^ 3 * 10) = -1 / 27 :=
by
  sorry

end expected_worth_flip_l406_406931


namespace tenth_term_series_l406_406673

theorem tenth_term_series :
  let left_term (n : ℕ) := 1 + (n - 1) * 2 in
  let right_term (n : ℕ) := 3 + (n - 1) * 2 in
  let term (n : ℕ) := left_term n * right_term n in
  term 10 = 399 :=
by
  sorry

end tenth_term_series_l406_406673


namespace percentage_change_area_right_triangle_l406_406021

theorem percentage_change_area_right_triangle
  (b h : ℝ)
  (hb : b = 0.5 * h)
  (A_original A_new : ℝ)
  (H_original : A_original = (1 / 2) * b * h)
  (H_new : A_new = (1 / 2) * (1.10 * b) * (1.10 * h)) :
  ((A_new - A_original) / A_original) * 100 = 21 := by
  sorry

end percentage_change_area_right_triangle_l406_406021


namespace parallelogram_angles_and_area_l406_406456

open Real

variables (A B C D M : Point) (AD_base : ℝ) (angle_CM_AD : ℝ) (M_midpoint_AD : M = midpoint A D) (B_eq_dist : dist B (line CM) = dist B A)

-- Conditions
noncomputable def AD_length : ℝ := 4
def angle_CM_AD := 75

-- Translate main problem to Lean problem statement
theorem parallelogram_angles_and_area (H1 : M_midpoint_AD)
                                       (H2 : angle_CM_AD = 75)
                                       (H3 : B_eq_dist) :
  ∃ (theta : ℝ), (theta = 75 ∧ (180 - theta) = 105) ∧ (area_parallelogram A B C D AD_length = 4 * (sqrt 3 + 2)) :=
by
  sorry

end parallelogram_angles_and_area_l406_406456


namespace final_result_l406_406438

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then x - y
  else if x < 0 ∧ y < 0 then x + 2 y
  else 2 * x - y

theorem final_result : p (p 2 (-3)) (p (-3) 1) = 21 := by
  sorry

end final_result_l406_406438


namespace common_tangent_exists_l406_406560

theorem common_tangent_exists :
  let y1 : ℝ → ℝ := λ x => 1 + x - x^2,
      y2 : ℝ → ℝ := λ x => 0.5 * (x^2 + 3),
      is_tangent (a b : ℝ) :=
        let y := λ x => a * x + b
        ∃ x : ℝ, y1 x = y x ∧ (∀ z : ℝ, y1 z = y z → z = x) ∧
                 ∃ x' : ℝ, y2 x' = y x' ∧ (∀ z : ℝ, y2 z = y2 x' → z = x')
  in (is_tangent 1 1 ∨ is_tangent (-1/3) (13/9)) :=
by
  sorry

end common_tangent_exists_l406_406560


namespace cylinder_height_l406_406875

theorem cylinder_height (C d : ℝ) (h : ℝ) (π : ℝ := Real.pi) 
  (C_eq : C = 2 * π * (3 / π)) (d_eq : d = sqrt (C^2 + h^2)) : 
  h = 8 :=
by
  have r := 3 / π
  -- from the perimeter of the circle
  have C_calc : C = 2 * π * r := by rw [C_eq, ←mul_assoc, mul_inv_cancel, mul_one]; exact ne_of_gt Real.pi_pos
  -- use the Pythagorean theorem on the rectangle
  have h_calc := calc
    d^2 = 100 : by rw d_eq; exact rfl -- diagonal squared
    C^2 + h^2 = d^2 : by rw d_eq
    36 + h^2 = 100 : by ring
    h^2 = 100 - 36 : by linarith
    h^2 = 64 : by norm_num
    h = sqrt 64 : by rw Real.sqrt_eq_rfl -- h squared
  exact rfl -- height

end cylinder_height_l406_406875


namespace tangent_circles_m_values_l406_406385

noncomputable def is_tangent (m : ℝ) : Prop :=
  let o1_center := (m, 0)
  let o2_center := (-1, 2 * m)
  let distance := Real.sqrt ((m + 1)^2 + (2 * m)^2)
  (distance = 5 ∨ distance = 1)

theorem tangent_circles_m_values :
  {m : ℝ | is_tangent m} = {-12 / 5, -2 / 5, 0, 2} := by
  sorry

end tangent_circles_m_values_l406_406385


namespace range_of_positive_integers_in_list_l406_406442

theorem range_of_positive_integers_in_list :
  ∃ L G : ℤ, (K = list.range' L 20) ∧ (L = -3 * x + 7) ∧ (x > -2) ∧ (range_of_positive_integers = G - L) ∧ (range_of_positive_integers = 19) :=
by
  let L := -3 * x + 7
  let G := L + 20 - 1
  let range_of_positive_integers := G - L
  sorry

end range_of_positive_integers_in_list_l406_406442


namespace positive_integer_sequence_product_minus_one_is_perfect_square_l406_406153

def sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = (7 * a n + (45 * (a n) ^ 2 - 36).sqrt) / 2

theorem positive_integer_sequence (a : ℕ → ℝ) (h0 : a 0 = 1) (h_seq : sequence a) :
∀ n : ℕ, ∃ k : ℤ, a n = k ∧ k > 0 :=
sorry

theorem product_minus_one_is_perfect_square (a : ℕ → ℝ) (h0 : a 0 = 1) (h_seq : sequence a) :
∀ n : ℕ, ∃ k : ℤ, a n * a (n + 1) - 1 = k ^ 2 :=
sorry

end positive_integer_sequence_product_minus_one_is_perfect_square_l406_406153


namespace min_a_b_l406_406807

theorem min_a_b (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 45 * a + b = 2021) : a + b = 85 :=
sorry

end min_a_b_l406_406807


namespace find_b_eq_five_l406_406403

/--
Given points A(4, 2) and B(0, b) in the Cartesian coordinate system,
and the condition that the distances from O (the origin) to B and from B to A are equal,
prove that b = 5.
-/
theorem find_b_eq_five : ∃ b : ℝ, (dist (0, 0) (0, b) = dist (0, b) (4, 2)) ∧ b = 5 :=
by
  sorry

end find_b_eq_five_l406_406403


namespace sum_of_areas_of_circles_l406_406173

theorem sum_of_areas_of_circles 
  (r s t : ℝ) 
  (h1 : r + s = 6) 
  (h2 : r + t = 8) 
  (h3 : s + t = 10) 
  : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by 
  sorry

end sum_of_areas_of_circles_l406_406173


namespace min_value_inequality_l406_406796

open Real

theorem min_value_inequality
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  (bc / a) + (ac / b) + (ab / c) ≥ 1 := 
by 
  -- Proof goes here
  sorry

end min_value_inequality_l406_406796


namespace min_path_length_l406_406341

-- Define the entities: points, line, reflection and distance
variables {Point : Type} [MetricSpace Point]
variables l : Set Point 
variables A B : Point

-- Define the reflection of the point A across the line l
noncomputable def reflection (P : Point) (l : Set Point) : Point := sorry

-- Define the point where the given question reaches the minimal distance
noncomputable def min_point (l : Set Point) (A B : Point) : Point := 
  let A' := reflection A l in
  sorry -- This construction will be defined in the actual proof

theorem min_path_length (X : Point) (A B : Point) 
  (l : Set Point) (hX_on_l : X ∈ l) :
  let A' := reflection A l in
  X = min_point l A B → 
  dist A X + dist X B ≤ dist A' X + dist X B :=
begin
  sorry
end

end min_path_length_l406_406341


namespace simplify_and_evaluate_fraction_l406_406123

theorem simplify_and_evaluate_fraction (x : ℤ) (hx : x = 5) :
  ((2 * x + 1) / (x - 1) - 1) / ((x + 2) / (x^2 - 2 * x + 1)) = 4 :=
by
  rw [hx]
  sorry

end simplify_and_evaluate_fraction_l406_406123


namespace minimum_sum_of_real_and_imag_value_l406_406440

noncomputable def minimum_sum_of_real_and_imag (z : ℂ) : ℝ :=
  re z + im z

theorem minimum_sum_of_real_and_imag_value (z : ℂ) (cond : z * conj(z) + (1 - 2*complex.i)*z + (1 + 2*complex.i)*conj(z) = 3) :
  ∃ x y : ℝ, (z = x + y * complex.i) ∧ minimum_sum_of_real_and_imag z = -7 :=
by
  sorry

end minimum_sum_of_real_and_imag_value_l406_406440


namespace tom_drives_distance_before_karen_wins_l406_406064

def karen_late_minutes := 4
def karen_speed_mph := 60
def tom_speed_mph := 45

theorem tom_drives_distance_before_karen_wins : 
  ∃ d : ℝ, d = 21 := by
  sorry

end tom_drives_distance_before_karen_wins_l406_406064


namespace cube_volume_l406_406572

theorem cube_volume {side V : ℝ} (h1 : 6 * side^2 = 486) : V = 729 :=
by {
  have h2 : side^2 = 81, by linarith,
  have h3 : side = 9, from (real.sqrt_eq_iff_eq_square _ _).1 (by norm_num[]),
  rw [h3] at *,
  change V = 729,
  simp [*],
  linarith,
} sorry

end cube_volume_l406_406572


namespace smallest_number_of_eggs_l406_406545

/-- The problem statement: determine the smallest number of eggs you could possibly have given certain conditions. -/
theorem smallest_number_of_eggs 
  (k : ℕ) 
  (c1 : 15 * k - 6 > 150) 
  (c2 : k ∈ {n : ℕ | 15 * n - 6 > 150}) 
  : 15 * 11 - 6 = 159 :=
by library_search

end smallest_number_of_eggs_l406_406545


namespace union_of_sets_l406_406091

def A : set ℝ := {x | -1 ≤ x ∧ x < 5 }
def B : set ℝ := {x | x < -1 ∨ x > 4 }

theorem union_of_sets : (A ∪ B) = set.univ :=
by
  sorry

end union_of_sets_l406_406091


namespace minimum_value_of_a_l406_406024

theorem minimum_value_of_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y ≥ 9) : ∃ a > 0, a ≥ 4 :=
sorry

end minimum_value_of_a_l406_406024


namespace person_speed_l406_406580

-- Define the given conditions
def distance_meters : ℝ := 1440
def time_minutes : ℝ := 12

-- Define the conversions needed
def distance_kilometers := distance_meters / 1000
def time_hours := time_minutes / 60

-- Define the speed calculation
def speed_kmh := distance_kilometers / time_hours

-- The statement we need to prove
theorem person_speed : speed_kmh = 7.2 := 
sorry

end person_speed_l406_406580


namespace find_f_value_l406_406002

noncomputable def f : ℝ → ℝ
| x := if x > 2 then f (x - 4)
       else if -2 <= x ∧ x <= 2 then Real.exp x
       else f (-x)

theorem find_f_value : f (-2019) = 1 / Real.exp 1 :=
by
  sorry

end find_f_value_l406_406002


namespace value_of_b_l406_406248

theorem value_of_b (f : ℝ → ℝ) (a b : ℝ) (h1 : ∀ x ≠ 0, f x = -1 / x) (h2 : f a = -1 / 3) (h3 : f (a * b) = 1 / 6) : b = -2 :=
sorry

end value_of_b_l406_406248


namespace solve_for_x_l406_406473

theorem solve_for_x (x : ℝ) 
  (h : 6 * x + 12 * x = 558 - 9 * (x - 4)) : 
  x = 22 := 
sorry

end solve_for_x_l406_406473


namespace expected_male_teachers_in_sample_l406_406663

theorem expected_male_teachers_in_sample 
  (total_male total_female sample_size : ℕ) 
  (h1 : total_male = 56) 
  (h2 : total_female = 42) 
  (h3 : sample_size = 14) :
  (total_male * sample_size) / (total_male + total_female) = 8 :=
by
  sorry

end expected_male_teachers_in_sample_l406_406663


namespace number_of_triangles_l406_406245

theorem number_of_triangles (dots_on_AB dots_on_BC dots_on_CA: ℕ) (h1: dots_on_AB = 2) (h2: dots_on_BC = 3) (h3: dots_on_CA = 7) :
  let total_dots := 3 + dots_on_AB + dots_on_BC + dots_on_CA in
  let total_combinations := Nat.choose total_dots 3 in
  let collinear_combinations := Nat.choose (dots_on_AB + 2) 3 + Nat.choose (dots_on_BC + 2) 3 + Nat.choose (dots_on_CA + 2) 3 in
  total_combinations - collinear_combinations = 357 :=
by
  rw [h1, h2, h3]
  simp only [Nat.choose_eq_factorization, h1, h2, h3]
  -- Total dots
  have ht : total_dots = 15 := by norm_num
  -- Total combinations of choosing 3 dots from 15
  have ht_comb : Nat.choose total_dots 3 = 455 := by norm_num
  -- Collinear combinations
  have hc_AB : Nat.choose (dots_on_AB + 2) 3 = 4 := by norm_num
  have hc_BC : Nat.choose (dots_on_BC + 2) 3 = 10 := by norm_num
  have hc_CA : Nat.choose (dots_on_CA + 2) 3 = 84 := by norm_num
  have hc_total : hc_AB + hc_BC + hc_CA = 98 := by norm_num
  -- Final result
  have result : 455 - 98 = 357 := by norm_num
  exact result

end number_of_triangles_l406_406245


namespace intersection_A_B_l406_406371

-- Define set A based on the given condition.
def setA : Set ℝ := {x | x^2 - 4 < 0}

-- Define set B based on the given condition.
def setB : Set ℝ := {x | x < 0}

-- Prove that the intersection of sets A and B is the given set.
theorem intersection_A_B : setA ∩ setB = {x | -2 < x ∧ x < 0} := by
  sorry

end intersection_A_B_l406_406371


namespace original_price_l406_406866

-- Define the initial conditions and variables
variables (p : ℝ) (x : ℝ)

-- Define the hypotheses
def price_increase (x : ℝ) (p : ℝ) := x * (1 + p / 100)
def price_decrease (x : ℝ) (p : ℝ) := x * (1 - p / 100)
def final_price (x : ℝ) (p : ℝ) := (price_increase x p) * (1 - p / 100)

-- State the theorem to be proven
theorem original_price (p : ℝ) (h : final_price x p = 1) : x = 10000 / (10000 - p^2) :=
by sorry

end original_price_l406_406866


namespace hypotenuse_of_right_triangle_l406_406198

theorem hypotenuse_of_right_triangle (a b : ℕ) (h₁ : a = 80) (h₂ : b = 150) : 
  (real.sqrt (a^2 + b^2) = 170) :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end hypotenuse_of_right_triangle_l406_406198


namespace polynomial_factors_integers_l406_406650

theorem polynomial_factors_integers (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 500)
  (h₃ : ∃ a : ℤ, n = a * (a + 1)) :
  n ≤ 21 :=
sorry

end polynomial_factors_integers_l406_406650


namespace prob_not_all_same_correct_l406_406531

-- Define the number of dice and the number of sides per die
def num_dice : ℕ := 5
def sides_per_die : ℕ := 8

-- Total possible outcomes when rolling num_dice dice with sides_per_die sides each
def total_outcomes : ℕ := sides_per_die ^ num_dice

-- Outcomes where all dice show the same number
def same_number_outcomes : ℕ := sides_per_die

-- Probability that all five dice show the same number
def prob_same : ℚ := same_number_outcomes / total_outcomes

-- Probability that not all five dice show the same number
def prob_not_same : ℚ := 1 - prob_same

-- The expected probability that not all dice show the same number
def expected_prob_not_same : ℚ := 4095 / 4096

-- The main theorem to prove
theorem prob_not_all_same_correct : prob_not_same = expected_prob_not_same := by
  sorry  -- Proof will be filled in by the user

end prob_not_all_same_correct_l406_406531


namespace angle_bisector_midline_tangency_intersection_l406_406830

theorem angle_bisector_midline_tangency_intersection
  (A B C : Point)
  (incircle : Circle)
  (A_bisector : Line)
  (midline_parallel_AC : Line)
  (tangent_CB_CA : Line)
  (h1 : is_triangle A B C)
  (h2 : tangency_point incircle CB CA)
  (h3 : is_angle_bisector A_bisector ∠BAC)
  (h4 : is_midline_parallel midline_parallel_AC AC)
  (h5 : connect_tangency_points tangent_CB_CA incircle CB CA) :
  intersects_at_single_point A_bisector midline_parallel_AC tangent_CB_CA :=
sorry

end angle_bisector_midline_tangency_intersection_l406_406830


namespace sum_of_floor_sqrt_1_to_25_l406_406988

theorem sum_of_floor_sqrt_1_to_25 : 
  ( ∑ i in Finset.range 26, ∥Real.sqrt i∥ ) = 75 :=
by
  sorry

end sum_of_floor_sqrt_1_to_25_l406_406988


namespace least_number_to_subtract_l406_406220

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (r : ℕ) (h : n = 9679) (d = 15) (r = 4): 
  (n - r) % d = 0 :=
by {
  rw [h, d_],
  norm_num,
  sorry
}

end least_number_to_subtract_l406_406220


namespace find_f11_l406_406231

-- Definition of the functional equation
def functional_eq (f : ℝ → ℝ) :=
  ∀ x : ℝ, f(x) + 2 * f(27 - x) = x

-- The main theorem to prove
theorem find_f11 : 
  ∃ f : ℝ → ℝ, functional_eq f ∧ f(11) = 7 :=
by
  existsi (λ x, -x + 18)  -- Placeholder for the function, to make the statement complete
  sorry

end find_f11_l406_406231


namespace equal_angles_l406_406516

variables {P Q A A' B B' C C' M N R : Point} 
variables {circle1 circle2 : Circle}
variables (AA' BB' CC' : Line)

-- Given conditions
axiom circles_intersect : circle1 ∩ circle2 = {P, Q}
axiom lines_through_P (AA' BB' CC' : Line) : P ∈ AA' ∧ P ∈ BB' ∧ P ∈ CC'
axiom intersection_points : 
  ∃ A A' B B' C C', (A ∈ circle1 ∧ A' ∈ circle2 ∧ B ∈ circle1 ∧ B' ∈ circle2 ∧ C ∈ circle1 ∧ C' ∈ circle2)
axiom line_intersections :
  M = (line_through ⟨A, B⟩) ∩ (line_through ⟨A', B'⟩) ∧
  N = (line_through ⟨A, C⟩) ∩ (line_through ⟨A', C'⟩) ∧
  R = (line_through ⟨B, C⟩) ∩ (line_through ⟨B', C'⟩)

theorem equal_angles (h1 : circles_intersect)
  (h2 : lines_through_P AA' BB' CC') 
  (h3 : intersection_points)
  (h4 : line_intersections) : 
  ∠ M = ∠ N ∧ ∠ N = ∠ R := 
sorry

end equal_angles_l406_406516


namespace area_of_triangle_min_value_c2_ab_l406_406050

-- Definitions for conditions in the problem
def sides_of_triangle (a b c : ℝ) : Prop :=
  ∃ A B C : ℝ, -- angles opposite to sides a, b, c
  2 * sin^2 A + sin^2 B = sin^2 C ∧
  ∀ θ, 0 < θ < π → sin θ > 0

/-- Problem Part 1: given b = 2a = 4, find the area of triangle ABC given specific trigonometric relation -/
theorem area_of_triangle {a b c : ℝ} (A B C : ℝ) 
  (h₁ : sides_of_triangle a b c)
  (h₂ : b = 2 * a) 
  (h₃ : a = 2) :
  area_of_triangle a b c = sqrt 15 :=
by
  sorry

/-- Problem Part 2: find the minimum value of c^2 / (ab) and determine the value of c / a at this minimum -/
theorem min_value_c2_ab {a b c : ℝ}
  (h₁ : 2*a^2 + b^2 = c^2) :
  (∃ (min_val : ℝ), min_val = 2 * sqrt 2 ∧ (c / a = 2)) :=
by
  sorry

end area_of_triangle_min_value_c2_ab_l406_406050


namespace minimum_visible_pairs_l406_406936

-- Definitions of the problem conditions
def num_birds : ℕ := 155
def max_arc : ℝ := 10
def circle_degree : ℝ := 360
def positions : ℕ := 35
def min_pairs(visible_pairs: ℕ) : Prop :=
  ∃ x : Fin 36 → Finset ℕ,  -- x denotes the distribution of birds
  (x.Sum = 155) ∧
  (Σ p in (Finset.range positions), (x p).card.choose 2) = visible_pairs

-- Problem Statement: Prove that the smallest number of mutually visible pairs is 270
theorem minimum_visible_pairs : min_pairs 270 := by
  sorry

end minimum_visible_pairs_l406_406936


namespace circle_area_l406_406114

theorem circle_area (C D : ℝ × ℝ) (hC : C = (-2, 3)) (hD : D = (4, -1)) : 
  let d := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2),
      r := d / 2,
      A := Real.pi * (r ^ 2)
  in A = 13 * Real.pi :=
by
  sorry

end circle_area_l406_406114


namespace parallel_vectors_implies_lambda_l406_406709

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (2, 1)

-- Define the scalar λ
variable (λ : ℝ)

-- Define the vectors a + 2b and 3a + λb
def vec1 : ℝ × ℝ := (a.1 + 2 * b.1, a.2 + 2 * b.2)
def vec2 : ℝ × ℝ := (3 * a.1 + λ * b.1, 3 * a.2 + λ * b.2)

-- State that vec1 and vec2 are parallel
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem parallel_vectors_implies_lambda (h: are_parallel vec1 vec2) : λ = 6 := 
sorry

end parallel_vectors_implies_lambda_l406_406709


namespace sum_of_floor_sqrt_1_to_25_l406_406993

theorem sum_of_floor_sqrt_1_to_25 : 
  ( ∑ i in Finset.range 26, ∥Real.sqrt i∥ ) = 75 :=
by
  sorry

end sum_of_floor_sqrt_1_to_25_l406_406993


namespace kevin_eggs_l406_406633

theorem kevin_eggs : 
  ∀ (bonnie george cheryl kevin : ℕ),
  bonnie = 13 → 
  george = 9 → 
  cheryl = 56 → 
  cheryl = bonnie + george + kevin + 29 →
  kevin = 5 :=
by
  intros bonnie george cheryl kevin h_bonnie h_george h_cheryl h_eqn
  subst h_bonnie
  subst h_george
  subst h_cheryl
  simp at h_eqn
  sorry

end kevin_eggs_l406_406633


namespace sum_of_areas_of_circles_l406_406169

-- Definitions of conditions
def r : ℝ := by sorry  -- radius of the circle at vertex A
def s : ℝ := by sorry  -- radius of the circle at vertex B
def t : ℝ := by sorry  -- radius of the circle at vertex C

axiom sum_radii_r_s : r + s = 6
axiom sum_radii_r_t : r + t = 8
axiom sum_radii_s_t : s + t = 10

-- The statement we want to prove
theorem sum_of_areas_of_circles : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by
  -- Use given axioms and properties of the triangle and circles
  sorry

end sum_of_areas_of_circles_l406_406169


namespace angle_C_eq_pi_div_six_l406_406028

theorem angle_C_eq_pi_div_six (A B C a b c : ℝ) (h : a^2 = 3*b^2 + 3*c^2 - 2*sqrt 3 * b * c * sin A) : 
  C = Real.pi / 6 :=
sorry

end angle_C_eq_pi_div_six_l406_406028


namespace minimum_distance_of_AB_l406_406860

noncomputable def f (x : ℝ) := Real.exp x + 1
noncomputable def g (x : ℝ) := 2 * x - 1
def y (x : ℝ) := f x - g x

theorem minimum_distance_of_AB : Inf (Set.range y) = 4 - 2 * Real.log 2 := by
  sorry

end minimum_distance_of_AB_l406_406860


namespace jillian_oranges_l406_406054

theorem jillian_oranges:
  let oranges := 80 in
  let pieces_per_orange := 10 in
  let pieces_per_friend := 4 in
  (oranges * (pieces_per_orange / pieces_per_friend) = 200) :=
by sorry

end jillian_oranges_l406_406054


namespace number_one_number_two_number_three_number_four_number_five_number_six_number_seven_number_eight_number_nine_number_ten_number_eleven_number_twelve_number_thirteen_number_fourteen_number_fifteen_number_sixteen_number_seventeen_l406_406521

section FiveFives

def five : ℕ := 5

-- Definitions for each number 1 to 17 using five fives.
def one : ℕ := (five / five) * (five / five)
def two : ℕ := (five / five) + (five / five)
def three : ℕ := (five * five - five) / five
def four : ℕ := (five - five / five) * (five / five)
def five_num : ℕ := five + (five - five) * (five / five)
def six : ℕ := five + (five + five) / (five + five)
def seven : ℕ := five + (five * five - five^2) / five
def eight : ℕ := (five + five + five) / five + five
def nine : ℕ := five + (five - five / five)
def ten : ℕ := five + five
def eleven : ℕ := (55 - 55 / five) / five
def twelve : ℕ := five * (five - five / five) / five
def thirteen : ℕ := (five * five - five - five) / five + five
def fourteen : ℕ := five + five + five - (five / five)
def fifteen : ℕ := five + five + five
def sixteen : ℕ := five + five + five + (five / five)
def seventeen : ℕ := five + five + five + ((five / five) + (five / five))

-- Proof statements to be provided
theorem number_one : one = 1 := sorry
theorem number_two : two = 2 := sorry
theorem number_three : three = 3 := sorry
theorem number_four : four = 4 := sorry
theorem number_five : five_num = 5 := sorry
theorem number_six : six = 6 := sorry
theorem number_seven : seven = 7 := sorry
theorem number_eight : eight = 8 := sorry
theorem number_nine : nine = 9 := sorry
theorem number_ten : ten = 10 := sorry
theorem number_eleven : eleven = 11 := sorry
theorem number_twelve : twelve = 12 := sorry
theorem number_thirteen : thirteen = 13 := sorry
theorem number_fourteen : fourteen = 14 := sorry
theorem number_fifteen : fifteen = 15 := sorry
theorem number_sixteen : sixteen = 16 := sorry
theorem number_seventeen : seventeen = 17 := sorry

end FiveFives

end number_one_number_two_number_three_number_four_number_five_number_six_number_seven_number_eight_number_nine_number_ten_number_eleven_number_twelve_number_thirteen_number_fourteen_number_fifteen_number_sixteen_number_seventeen_l406_406521


namespace width_of_wall_l406_406237

noncomputable def radius_pool : ℝ := 20
noncomputable def area_pool := π * radius_pool^2
noncomputable def area_wall := (11 / 25) * area_pool
noncomputable def area_total := area_pool + area_wall
theorem width_of_wall :
  ∃ w : ℝ, area_total = π * (radius_pool + w)^2 ∧ w = 4 :=
  sorry

end width_of_wall_l406_406237


namespace frances_towels_weight_in_ounces_l406_406100

theorem frances_towels_weight_in_ounces (Mary_towels Frances_towels : ℕ) (Mary_weight Frances_weight : ℝ) (total_weight : ℝ) :
  Mary_towels = 24 ∧ Mary_towels = 4 * Frances_towels ∧ total_weight = Mary_weight + Frances_weight →
  Frances_weight * 16 = 240 :=
by
  sorry

end frances_towels_weight_in_ounces_l406_406100


namespace garden_fraction_flower_beds_l406_406938

/-- Define the problem conditions and statement -/
theorem garden_fraction_flower_beds:
  ∀ (length side1 side2 : ℝ) (n_triangles : ℕ),
  side1 = 30 → side2 = 46 → 
  (n_triangles = 2) → (length = 46) → 
    (∀ (a : ℝ), (a = (side2 - side1) / 2) →
    (∀ (garden_width : ℝ), (garden_width = a) →
      (let triangle_area := (1 / 2) * a^2 in 
       let total_flower_area := n_triangles * triangle_area in
       let garden_area := length * garden_width in
       total_flower_area / garden_area = 4 / 23))) :=
begin
  intros length side1 side2 n_triangles h_side1 h_side2 h_ntriangles h_length a h_a garden_width h_width,
  let triangle_area := (1 / 2) * a^2,
  let total_flower_area := n_triangles * triangle_area,
  let garden_area := length * garden_width,
  show total_flower_area / garden_area = 4 / 23, from sorry,
end

end garden_fraction_flower_beds_l406_406938


namespace sum_first_13_l406_406765

variable {a_n : ℕ → ℝ}

-- Given condition
def condition (a_n : ℕ → ℝ) : Prop :=
  2 * (a_n 1 + a_n 4 + a_n 7) + 3 * (a_n 9 + a_n 11) = 24

-- Arithmetic sequence definition
def arithmetic_sequence (a_n : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n, a_n n = a1 + (n - 1) * d

-- Sum of the first 13 terms
def sum_first_13_terms (a_n : ℕ → ℝ) : ℝ :=
  ∑ i in Finset.range 13, a_n (i + 1)

-- Main theorem to prove
theorem sum_first_13 (a_n : ℕ → ℝ) (a1 d : ℝ) (h_seq : arithmetic_sequence a_n a1 d)
  (h_cond : condition a_n) :
  sum_first_13_terms a_n = 26 := by
  sorry

end sum_first_13_l406_406765


namespace find_m_l406_406010

theorem find_m (x y m : ℝ)
  (h1 : 2 * x + y = 6 * m)
  (h2 : 3 * x - 2 * y = 2 * m)
  (h3 : x / 3 - y / 5 = 4) :
  m = 15 :=
by
  sorry

end find_m_l406_406010


namespace transformed_curve_C_l406_406701

variable (x y : ℝ)

def matrix_A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, -2], ![0, 1]]

def curve_C (x y : ℝ) : Prop := (x - y)^2 + y^2 = 1

theorem transformed_curve_C' (x y : ℝ) :
  (∃ x₀ y₀ : ℝ, (matrix_A ⬝ ![x₀, y₀] = ![x, y]) ∧ curve_C x₀ y₀) →
  (x^2 / 4 + y^2 = 1) :=
sorry

end transformed_curve_C_l406_406701


namespace sqrt_sum_inequality_l406_406798

theorem sqrt_sum_inequality 
  (n : ℕ) 
  (a : ℕ → ℝ)
  (h : ∀ k, k ≤ n → a k ≥ a (k + 1))
  (h_n_plus_1 : a (n + 1) = 0) :
  sqrt (∑ k in finset.range (n + 1), a k) 
  ≤ ∑ k in finset.range (n + 1), (sqrt k) * (sqrt (a k) - sqrt (a (k + 1))) := 
sorry

end sqrt_sum_inequality_l406_406798


namespace find_y_l406_406382

def is_divisible_by (x y : ℕ) : Prop := x % y = 0

def ends_with_digit (x : ℕ) (d : ℕ) : Prop :=
  x % 10 = d

theorem find_y (y : ℕ) :
  (y > 0) ∧
  is_divisible_by y 4 ∧
  is_divisible_by y 5 ∧
  is_divisible_by y 7 ∧
  is_divisible_by y 13 ∧
  ¬ is_divisible_by y 8 ∧
  ¬ is_divisible_by y 15 ∧
  ¬ is_divisible_by y 50 ∧
  ends_with_digit y 0
  → y = 1820 :=
sorry

end find_y_l406_406382


namespace midpoint_FG_on_MN_l406_406845

theorem midpoint_FG_on_MN
  (A B C D E M N F G : Point)
  (O O1 : Circle)
  (H1 : excircle A B C O)
  (H2 : incircle A D E O1)
  (H3 : touches O BC M)
  (H4 : touches O1 DE N)
  (H5 : DE ∥ BC)
  (H6 : Line B G ∩ Line D O = Some F)
  (H7 : Line C G ∩ Line E O = Some G) :
  midpoint (F, G) ∈ line (M, N) :=
sorry

end midpoint_FG_on_MN_l406_406845


namespace city_rentals_cost_per_mile_l406_406127

theorem city_rentals_cost_per_mile :
  ∀ (x : ℝ),
    (∀ (miles : ℝ), miles = 48.0 →
      (17.99 + 0.18 * miles = 18.95 + x * miles)) →
    x = 0.16 :=
by
  intros x H
  specialize H 48.0 rfl
  have : 17.99 + 0.18 * 48.0 = 26.63 := by norm_num
  rw this at H
  linarith

end city_rentals_cost_per_mile_l406_406127


namespace ice_cream_depth_l406_406264

theorem ice_cream_depth 
  (r_sphere : ℝ) 
  (r_cylinder : ℝ) 
  (h_cylinder : ℝ) 
  (V_sphere : ℝ) 
  (V_cylinder : ℝ) 
  (constant_density : V_sphere = V_cylinder)
  (r_sphere_eq : r_sphere = 2) 
  (r_cylinder_eq : r_cylinder = 8) 
  (V_sphere_def : V_sphere = (4 / 3) * Real.pi * r_sphere^3) 
  (V_cylinder_def : V_cylinder = Real.pi * r_cylinder^2 * h_cylinder) 
  : h_cylinder = 1 / 6 := 
by 
  sorry

end ice_cream_depth_l406_406264


namespace value_at_x_minus_5_l406_406206

theorem value_at_x_minus_5 :
  ∀ (x : ℝ), x = -5 → -2 * x^2 + 5 / x = -51 :=
by
  intro x
  intro h
  rw [h]
  have h1 := (by norm_num : (-5 : ℝ)^2 = 25)
  have h2 := (by norm_num : -2 * 25 = -50)
  have h3 := (by norm_num : 5 / (-5) = -1)
  rw [h1, h2, h3]
  norm_num
  sorry

end value_at_x_minus_5_l406_406206


namespace domain_linear_domain_rational_domain_sqrt_domain_sqrt_denominator_domain_rational_complex_domain_arcsin_l406_406648

-- 1. Domain of z = 4 - x - 2y
theorem domain_linear (x y : ℝ) : true := 
by sorry

-- 2. Domain of p = 3 / (x^2 + y^2)
theorem domain_rational (x y : ℝ) : x^2 + y^2 ≠ 0 → true := 
by sorry

-- 3. Domain of z = sqrt(1 - x^2 - y^2)
theorem domain_sqrt (x y : ℝ) : 1 - x^2 - y^2 ≥ 0 → true := 
by sorry

-- 4. Domain of q = 1 / sqrt(xy)
theorem domain_sqrt_denominator (x y : ℝ) : xy > 0 → true := 
by sorry

-- 5. Domain of u = x^2 y / (2x + 1 - y)
theorem domain_rational_complex (x y : ℝ) : 2x + 1 - y ≠ 0 → true := 
by sorry

-- 6. Domain of v = arcsin(x + y)
theorem domain_arcsin (x y : ℝ) : -1 ≤ x + y ∧ x + y ≤ 1 → true := 
by sorry

end domain_linear_domain_rational_domain_sqrt_domain_sqrt_denominator_domain_rational_complex_domain_arcsin_l406_406648


namespace evaluate_function_l406_406693

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then Real.logb 3 x else 3 ^ x

theorem evaluate_function :
  f (f (1 / 9)) = 1 / 9 :=
sorry

end evaluate_function_l406_406693


namespace find_c_l406_406847

/-- Define the conditions given in the problem --/
def parabola_equation (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def vertex_condition (a b c : ℝ) : Prop := 
  ∀ x, parabola_equation a b c x = a * (x - 3)^2 - 1

def passes_through_point (a b c : ℝ) : Prop := 
  parabola_equation a b c 1 = 5

/-- The main statement -/
theorem find_c (a b c : ℝ) 
  (h_vertex : vertex_condition a b c) 
  (h_point : passes_through_point a b c) :
  c = 12.5 :=
sorry

end find_c_l406_406847


namespace instantaneous_velocity_at_2_l406_406688

-- Define the displacement equation
def displacement (t : ℝ) : ℝ := t^2 * real.exp(t - 2)

-- Define the derivative of the displacement equation
def velocity (t : ℝ) : ℝ := deriv displacement t

-- The theorem to prove: The instantaneous velocity of the particle at t=2 is 8
theorem instantaneous_velocity_at_2 : velocity 2 = 8 :=
sorry

end instantaneous_velocity_at_2_l406_406688


namespace problem_l406_406934

noncomputable def rotated_quadrilateral (A B C D O : ℂ) : Prop :=
  ∃ A' B' C' D' P Q R S : ℂ,
    A' = A * complex.I ∧
    B' = B * complex.I ∧
    C' = C * complex.I ∧
    D' = D * complex.I ∧
    P = (A' + B) / 2 ∧
    Q = (B' + C) / 2 ∧
    R = (C' + D) / 2 ∧
    S = (D' + A) / 2 ∧
    ‖P - R‖ = ‖Q - S‖ ∧
    (P - R) * complex.conj (Q - S) = 0

-- To prove the two goals in one statement
theorem problem (A B C D O : ℂ) (h : rotated_quadrilateral A B C D O) : ∃ (P Q R S : ℂ),
  ((P - R) * complex.conj (Q - S) = 0) ∧ (‖P - R‖ = ‖Q - S‖) :=
by
  obtain ⟨A', B', C', D', P, Q, R, S, hA', hB', hC', hD', hP, hQ, hR, hS, h1, h2⟩ := h
  use [P, Q, R, S]
  split
  · exact h1
  · exact h2

end problem_l406_406934


namespace probability_circles_intersect_l406_406887

-- Definitions for the problem
noncomputable def Circle_C_center_uniform (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 3) : ℝ × ℝ :=
  (x, 0)

noncomputable def Circle_D_center_uniform (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 3) : ℝ × ℝ :=
  (x, 2)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Math.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Main statement for the problem
theorem probability_circles_intersect :
  (∃ C_X D_X : ℝ, (0 ≤ C_X ∧ C_X ≤ 3) ∧ (0 ≤ D_X ∧ D_X ≤ 3) ∧ 
  distance (Circle_C_center_uniform C_X ⟨0, le_refl 3⟩) (Circle_D_center_uniform D_X ⟨0, le_refl 3⟩) ≤ 3) →
  (filter.at_top (probability_circles_intersect (Circle_C_center_uniform C_X ⟨0, le_refl 3⟩), Circle_D_center_uniform D_X).probability = real.to_nnreal (sqrt 5 / 3)) :=
by
  sorry

end probability_circles_intersect_l406_406887


namespace min_area_section_ABD_l406_406408

-- Define the given conditions and variables
def PA_perp_plane_ABC (P A B C : Point) : Prop := -- define perpendicularity of PA to plane ABC
sorry

def AB_perp_AC (A B C : Point) : Prop := -- define perpendicularity of AB to AC
sorry

def length_PA (P A : Point) (a : ℝ) : Prop := (dist P A) = 3 * a
def length_AC (A C : Point) (a : ℝ) : Prop := (dist A C) = a
def length_AB (A B : Point) (a : ℝ) : Prop := (dist A B) = 2 * a

-- Main theorem stating the minimum area condition
theorem min_area_section_ABD 
  (P A B C D : Point)
  (a : ℝ)
  (h1 : PA_perp_plane_ABC P A B C) 
  (h2 : AB_perp_AC A B C)
  (h3 : length_AC A C a)
  (h4 : length_AB A B a)
  (h5 : length_PA P A a) 
  (h6 : D ∈ (line_through AB ∩ line_through PC)) -- D is intersection of AB and PC
: (area_triangle A B D) = (3 / sqrt(10)) * a^2 :=
sorry

end min_area_section_ABD_l406_406408


namespace average_correct_percentile_75_correct_l406_406391

-- Definitions based on the conditions
def data_set : List ℕ := [2, 6, 7, 5, 9, 17, 10]

-- Statement that the average of the data set is 8
theorem average_correct : (List.sum data_set.toList) / data_set.length = 8 := 
by
  sorry

-- Statement that the 75th percentile of the data set is 10
theorem percentile_75_correct : 
  let n := data_set.length
  let sorted_data_set := data_set.toList.qsort (· ≤ ·)
  let p := 75 / 100 * n
  sorted_data_set.nthd (Nat.ceil p - 1) = some 10 :=
by
  sorry

end average_correct_percentile_75_correct_l406_406391


namespace rohan_monthly_salary_l406_406913

theorem rohan_monthly_salary (s : ℝ) 
  (h_food : s * 0.40 = f)
  (h_rent : s * 0.20 = hr) 
  (h_entertainment : s * 0.10 = e)
  (h_conveyance : s * 0.10 = c)
  (h_savings : s * 0.20 = 1000) : 
  s = 5000 := 
sorry

end rohan_monthly_salary_l406_406913


namespace find_f_and_decreasing_intervals_l406_406366

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem find_f_and_decreasing_intervals (φ : ℝ) 
  (h₁ : ∀ x : ℝ, f x φ ≤ |f (Real.pi / 6) φ|) 
  (h₂ : f (Real.pi / 2) φ > f Real.pi φ) :
  (f x (2 * n * Real.pi + 7 * Real.pi / 6) = Real.sin (2 * x + 7 * Real.pi / 6) ∧
   (∀ x ∈ set.Icc 0 Real.pi, monotonic_on (f x (7 * Real.pi / 6)) ({0, Real.pi / 6} ∪ {Real.pi * 2 / 3, Real.pi}))) :=
by
  sorry

end find_f_and_decreasing_intervals_l406_406366


namespace P_finishes_job_in_correct_time_l406_406453

theorem P_finishes_job_in_correct_time :
  ∃ P : ℝ, (P > 0) ∧ 
           (∀ Q : ℝ, Q = 9 → 
                     let work_together := 2 * (1 / P + 1 / Q),
                         remaining_work := 1 - work_together,
                         remaining_time := 1 / 3 in
                     remaining_work = remaining_time * (1 / P)) ∧
           P = 4.5 :=
by
  sorry

end P_finishes_job_in_correct_time_l406_406453


namespace tom_distance_before_karen_wins_l406_406066

theorem tom_distance_before_karen_wins :
  let speed_Karen := 60
  let speed_Tom := 45
  let delay_Karen := (4 : ℝ) / 60
  let distance_advantage := 4
  let time_to_catch_up := (distance_advantage + speed_Tom * delay_Karen) / (speed_Karen - speed_Tom)
  let distance_Tom := speed_Tom * time_to_catch_up
  distance_Tom = 21 :=
by
  sorry

end tom_distance_before_karen_wins_l406_406066


namespace second_wrongly_copied_number_l406_406136

theorem second_wrongly_copied_number 
  (avg_err : ℝ) 
  (total_nums : ℕ) 
  (sum_err : ℝ) 
  (first_err_corr : ℝ) 
  (correct_avg : ℝ) 
  (correct_num : ℝ) 
  (second_num_wrong : ℝ) :
  (avg_err = 40.2) → 
  (total_nums = 10) → 
  (sum_err = total_nums * avg_err) → 
  (first_err_corr = 16) → 
  (correct_avg = 40) → 
  (correct_num = 31) → 
  sum_err - first_err_corr + (correct_num - second_num_wrong) = total_nums * correct_avg → 
  second_num_wrong = 17 := 
by 
  intros h_avg h_total h_sum_err h_first_corr h_correct_avg h_correct_num h_corrected_sum 
  sorry

end second_wrongly_copied_number_l406_406136


namespace distance_from_apex_l406_406189

theorem distance_from_apex (A B : ℝ)
  (h_A : A = 216 * Real.sqrt 3)
  (h_B : B = 486 * Real.sqrt 3)
  (distance_planes : ℝ)
  (h_distance_planes : distance_planes = 8) :
  ∃ h : ℝ, h = 24 :=
by
  sorry

end distance_from_apex_l406_406189


namespace conversion_correct_l406_406295

def binary_to_decimal (b : List ℕ) : ℕ :=
  b.reverse.enum.foldl (λ acc ⟨i, digit⟩ => acc + digit * 2^i) 0

def n : List ℕ := [1, 0, 1, 1, 1, 1, 0, 1, 1]

theorem conversion_correct :
  binary_to_decimal n = 379 :=
by 
  sorry

end conversion_correct_l406_406295


namespace part1_part2_l406_406714

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b : ℝ × ℝ := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)
noncomputable def m (t : ℝ) : ℝ × ℝ := (1 + t * (Real.sqrt 2 / 2), 2 + t * (Real.sqrt 2 / 2))

theorem part1 : ∃ t : ℝ, t = -3 * Real.sqrt 2 / 2 ∧ 
  (|m t| = (∀ u, |m u| ≥ |m t|)) := sorry

theorem part2 (ha_perp : a.1 * b.1 + a.2 * b.2 = 0) : 
  ∃ t : ℝ, ∀ u : ℝ, 
  (Real.cos (π / 4) = ((m u - b) • (m u + t * b))
  / (|m u - b| * |m u + t * b|)) :=
  sorry

end part1_part2_l406_406714


namespace carolyn_sum_removes_n_5_l406_406128

def sum_carolyn_removes (n : ℕ) : ℕ :=
  let list := {i | i ∈ Finset.range (n + 1)} in
  let even_numbers := list.filter (λ x, x % 2 = 0) in
  let removes := {x ∈ even_numbers | x = 4} in -- as per problem, Carolyn removes 4
  removes.sum id

theorem carolyn_sum_removes_n_5 : sum_carolyn_removes 5 = 4 :=
by
  sorry

end carolyn_sum_removes_n_5_l406_406128


namespace stratified_sample_business_personnel_l406_406571

def total_employees : ℕ := 160
def business_personnel : ℕ := 120
def management_personnel : ℕ := 16
def logistics_personnel : ℕ := 24
def sample_size : ℕ := 20

theorem stratified_sample_business_personnel :
  (120 * 20 / 160 = 15) :=
by
  have total_employees = 160
  have business_personnel = 120
  have sample_size = 20
  have prop := (120 * 20 / 160 = 15)
  sorry -- proof goes here

end stratified_sample_business_personnel_l406_406571


namespace maximum_log_sum_l406_406352

theorem maximum_log_sum (a b : ℝ) (ha : a > 1) (hb : b > 1) (h_sum : a + b = 4 * real.sqrt 2) : 
  log 2 a + log 2 b ≤ 3 :=
sorry

end maximum_log_sum_l406_406352


namespace train_speed_l406_406267

theorem train_speed (length_train length_bridge time_pass : ℝ)
  (h_train : length_train = 320) (h_bridge : length_bridge = 140) (h_time : time_pass = 36.8) :
  let distance := length_train + length_bridge in
  let speed_mps := distance / time_pass in
  let speed_kmph := speed_mps * 3.6 in
  speed_kmph = 45 :=
by
  intros
  sorry

end train_speed_l406_406267


namespace coin_game_outcome_l406_406278

-- Define the possible moves and determine the winner
def winning_position (n : ℕ) : Prop :=
  (n % 8 = 2 ∨ n % 8 = 3 ∨ n % 8 = 4 ∨ n % 8 = 5) ∧ n > 0

-- Theorem stating the outcome based on the initial number of coins
theorem coin_game_outcome (n : ℕ) : 
  n = 1001 → ¬ winning_position n ∧ n = 1002 → winning_position n :=
by
  intro h1001 h1002
  split
  { -- For 1001 coins, the starting player loses
    have h1 : 1001 % 8 = 1 :=
    by norm_num
    simp at h1
    rw [h1, or_self]
    norm_num
    sorry
  }
  { -- For 1002 coins, the starting player wins
    have h2 : 1002 % 8 = 2 :=
    by norm_num
    simp at h2
    rw [h2, or_self]
    norm_num
    sorry
  }

end coin_game_outcome_l406_406278


namespace jed_initial_cards_l406_406777

def jed_gets_cards_per_week : ℕ := 6
def jed_gives_cards_every_two_weeks : ℕ := 2
def jed_final_cards : ℕ := 40
def weeks : ℕ := 4

theorem jed_initial_cards :
  ∃ (X : ℕ), jed_final_cards = X + jed_gets_cards_per_week * weeks - jed_gives_cards_every_two_weeks * (weeks / 2) :=
begin
  use 20,
  sorry,
end

end jed_initial_cards_l406_406777


namespace sum_floor_sqrt_1_to_25_l406_406999

theorem sum_floor_sqrt_1_to_25 :
  (∑ n in finset.range 25, ⌊real.sqrt (n + 1)⌋) = 75 :=
sorry

end sum_floor_sqrt_1_to_25_l406_406999


namespace road_length_l406_406815

theorem road_length (width length n_truckloads truckload_area truckload_cost cost_with_tax sales_tax pre_tax_cost : ℕ)
  (h1 : width = 20)
  (h2 : truckload_area = 800)
  (h3 : truckload_cost = 75)
  (h4 : sales_tax = 20)
  (h5 : cost_with_tax = 4500)
  (h6 : pre_tax_cost = cost_with_tax * 100 / (100 + sales_tax)) 
  (h7 : n_truckloads = pre_tax_cost / truckload_cost)
  (h8 : n_truckloads * truckload_area = width * length) :
  length = 2000 :=
begin
  sorry
end

end road_length_l406_406815


namespace compare_abc_l406_406797

def a : ℝ := 3^(0.5)
def b : ℝ := log 3 2
def c : ℝ := Real.cos 2

theorem compare_abc : c < b ∧ b < a :=
by
  sorry

end compare_abc_l406_406797


namespace cos_double_alpha_two_alpha_minus_beta_l406_406922

variable (α β : ℝ)
variable (α_pos : 0 < α)
variable (α_lt_pi : α < π)
variable (tan_α : Real.tan α = 2)

variable (β_pos : 0 < β)
variable (β_lt_pi : β < π)
variable (cos_β : Real.cos β = -((7 * Real.sqrt 2) / 10))

theorem cos_double_alpha (hα : 0 < α ∧ α < π) (htan : Real.tan α = 2) : 
  Real.cos (2 * α) = -3 / 5 := by
  sorry

theorem two_alpha_minus_beta (hα : 0 < α ∧ α < π) (htan : Real.tan α = 2)
  (hβ : 0 < β ∧ β < π) (hcosβ : Real.cos β = -((7 * Real.sqrt 2) / 10)) : 
  2 * α - β = -π / 4 := by
  sorry

end cos_double_alpha_two_alpha_minus_beta_l406_406922


namespace problem_a_lt_b_lt_0_implies_ab_gt_b_sq_l406_406016

theorem problem_a_lt_b_lt_0_implies_ab_gt_b_sq (a b : ℝ) (h : a < b ∧ b < 0) : ab > b^2 := by
  sorry

end problem_a_lt_b_lt_0_implies_ab_gt_b_sq_l406_406016


namespace trajectory_and_constant_k_l406_406451

open Classical

theorem trajectory_and_constant_k (x y: ℝ) : 
    (∀ P: ℝ × ℝ, P ∈ set_of (λ P, P.fst^2 + P.snd^2 = 4) → 
    (∃ Q: ℝ × ℝ, Q = (P.fst, 0) ∧ ∃ M: ℝ × ℝ, M = (x, y) ∧ (P.fst - x = 2(x - P.fst) ∧ P.snd - y = 2(y - 0)))) →
    (∀ A: ℝ × ℝ, A = (2, 0) →
    (∀ B: ℝ × ℝ, B ∈ set_of (λ B, (B.fst^2 / 4) + B.snd^2 = 1) →
    (∀ D: ℝ × ℝ, D ∈ set_of (λ D, (D.fst^2 / 4) + D.snd^2 = 1) →
    (∀ k_1 k_2: ℝ, k_1 = (B.snd - A.snd) / (B.fst - A.fst) ∧ k_2 = (D.snd - A.snd) / (D.fst - A.fst) →
    k_1 * k_2 = -3 / 4)))) sorry

end trajectory_and_constant_k_l406_406451


namespace problem_l406_406725

noncomputable def f (x : ℝ) : ℝ := real.log x / x

theorem problem 
  (a b : ℝ) 
  (ha : a > 3) 
  (hb : b > a) : 
  f a > f (real.sqrt (a * b)) 
  ∧ f (real.sqrt (a * b)) > f ((a + b) / 2) 
  ∧ f ((a + b) / 2) > f b := 
by 
  -- Proof omitted
  sorry

end problem_l406_406725


namespace vectors_relations_l406_406012

def a : ℝ × ℝ × ℝ := (-2, -3, 1)
def b : ℝ × ℝ × ℝ := (2, 0, 4)
def c : ℝ × ℝ × ℝ := (-4, -6, 2)

theorem vectors_relations :
  (∃ k : ℝ, c = (k • a)) ∧ (a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0) :=
by
  sorry

end vectors_relations_l406_406012


namespace index_of_100th_term_lt_0_l406_406625

noncomputable def a (n : ℕ) : ℝ := (∑ k in (finset.range n).map finset.succ, real.sin k)

theorem index_of_100th_term_lt_0 : ∃ n : ℕ, a n < 0 ∧ n = 628 :=
begin
  sorry
end

end index_of_100th_term_lt_0_l406_406625


namespace find_f_at_7_l406_406358

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^7 + b * x^3 + c * x - 5

theorem find_f_at_7 (a b c : ℝ) (h : f a b c (-7) = 7) : f a b c 7 = -17 :=
by
  sorry

end find_f_at_7_l406_406358


namespace problem_s2_equal_8r2_l406_406562

-- Define the problem conditions
variables (A B C : ℝ) (r' : ℝ)
hypothesis (h1 : A ≠ C) (h2 : B ≠ C) -- Points A and B don't coincide with C
hypothesis (h3 : C ≠ A) (h4 : C ≠ B) -- Points C doesn't coincide with A or B
hypothesis (h5 : ∠ACB = real.pi/4) -- θ = 45 degrees

-- Prove that s^2 = 8r'^2 given the conditions
theorem problem_s2_equal_8r2 :
  let AB := 2 * r' in
  let AC := r' * real.sqrt 2 in
  let BC := r' * real.sqrt 2 in
  (AC + BC) ^ 2 = 8 * r' ^ 2 :=
by
  sorry

end problem_s2_equal_8r2_l406_406562


namespace find_n_modulo_10_l406_406316

theorem find_n_modulo_10 :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -1345 [MOD 10] ∧ n = 5 :=
sorry

end find_n_modulo_10_l406_406316


namespace rectangle_perimeter_l406_406818

theorem rectangle_perimeter (w : ℝ) (P : ℝ) (l : ℝ) (A : ℝ) 
  (h1 : l = 18)
  (h2 : A = l * w)
  (h3 : P = 2 * l + 2 * w) 
  (h4 : A + P = 2016) : 
  P = 234 :=
by
  sorry

end rectangle_perimeter_l406_406818


namespace female_employees_literate_l406_406215

def total_employees : ℕ := 1600
def percentage_female : ℝ := 0.60
def percentage_male_literate : ℝ := 0.50
def total_percentage_literate : ℝ := 0.62

theorem female_employees_literate (E : ℕ) (F_perc : ℝ) (M_lit_perc : ℝ) (T_lit_perc : ℝ) :
  E = 1600 →
  F_perc = 0.60 →
  M_lit_perc = 0.50 →
  T_lit_perc = 0.62 →
  let F := F_perc * E,
      M := (1 - F_perc) * E,
      CL_M := M_lit_perc * M,
      CL := T_lit_perc * E,
      CL_F := CL - CL_M in
  CL_F = 672 :=
by
  intros hE hF_perc hM_lit_perc hT_lit_perc
  have F := F_perc * E
  have M := (1 - F_perc) * E
  have CL_M := M_lit_perc * M
  have CL := T_lit_perc * E
  have CL_F := CL - CL_M
  sorry

end female_employees_literate_l406_406215


namespace sum_of_midpoint_coordinates_eq_one_l406_406457

open Function

-- Define the points P and R
def P : ℝ × ℝ := (3, 2)
def R : ℝ × ℝ := (13, 16)

-- Define the reflection function over the y-axis
def reflect_y (point : ℝ × ℝ) : ℝ × ℝ := (-point.1, point.2)

-- Apply reflection to points P and R
def P' := reflect_y P
def R' := reflect_y R

-- Define the midpoint calculation function
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Calculate the midpoint of reflected points P' and R'
def M' := midpoint P' R'

-- Define the sum of coordinates of a given point
def sum_of_coordinates (point : ℝ × ℝ) : ℝ := point.1 + point.2

-- The theorem to prove
theorem sum_of_midpoint_coordinates_eq_one : sum_of_coordinates M' = 1 := by
  sorry

end sum_of_midpoint_coordinates_eq_one_l406_406457


namespace cost_of_rice_l406_406522

theorem cost_of_rice (x : ℝ) 
  (h : 5 * x + 3 * 5 = 25) : x = 2 :=
by {
  sorry
}

end cost_of_rice_l406_406522


namespace reentrant_number_count_l406_406740

def is_reentrant (H T U : ℕ) : Prop :=
  1 ≤ H ∧ H ≤ 9 ∧ 0 ≤ T ∧ T ≤ 9 ∧ 0 ≤ U ∧ U ≤ 9 ∧ T < H ∧ T < U

noncomputable def count_reentrant_numbers : ℕ :=
  Nat.card {n : ℕ // ∃ H T U, n = 100 * H + 10 * T + U ∧ is_reentrant H T U}

theorem reentrant_number_count : count_reentrant_numbers = 285 :=
  sorry

end reentrant_number_count_l406_406740


namespace max_value_7x_10y_z_l406_406649

theorem max_value_7x_10y_z (x y z : ℝ) 
  (h : x^2 + 2 * x + (1 / 5) * y^2 + 7 * z^2 = 6) : 
  7 * x + 10 * y + z ≤ 55 := 
sorry

end max_value_7x_10y_z_l406_406649


namespace solve_for_x_l406_406503

theorem solve_for_x (x : ℝ) : 3x + 12 = (1 / 3) * (7x + 42) → x = 3 :=
by
  intro h
  sorry

end solve_for_x_l406_406503


namespace total_triangles_in_figure_l406_406615

theorem total_triangles_in_figure :
  let row1 := 3
  let row2 := 2
  let row3 := 1
  let small_triangles := row1 + row2 + row3
  let two_small_comb := 3
  let three_small_comb := 1
  let all_small_comb := 1
  small_triangles + two_small_comb + three_small_comb + all_small_comb = 11 :=
by
  let row1 := 3
  let row2 := 2
  let row3 := 1
  let small_triangles := row1 + row2 + row3
  let two_small_comb := 3
  let three_small_comb := 1
  let all_small_comb := 1
  show small_triangles + two_small_comb + three_small_comb + all_small_comb = 11
  sorry

end total_triangles_in_figure_l406_406615


namespace sum_of_possible_w_l406_406280

theorem sum_of_possible_w :
  ∃ w x y z : ℤ, w > x ∧ x > y ∧ y > z ∧ w + x + y + z = 44 ∧ 
  {w - x, w - y, w - z, x - y, x - z, y - z} = {1, 3, 4, 5, 6, 9} ∧
  (w = 16 ∨ w = 15) ∧ w + 15 + 16 = 31 :=
by
  -- Here we state the theorem and outline our conditions.
  sorry

end sum_of_possible_w_l406_406280


namespace number_of_distinct_values_l406_406253

theorem number_of_distinct_values (n : ℕ) (mode_count : ℕ) (second_count : ℕ) (total_count : ℕ) 
    (h1 : n = 3000) (h2 : mode_count = 15) (h3 : second_count = 14) : 
    (n - mode_count - second_count) / 13 + 2 ≥ 232 :=
by 
  sorry

end number_of_distinct_values_l406_406253


namespace curve_equations_range_OA_OB_l406_406045

-- Define C1 and C2 curves and their equations
def C1_param_eq (α : ℝ) : ℝ × ℝ := (1 + cos α, sin α)
def C1_eq (x y : ℝ) := (x - 1)^2 + y^2 = 1

def C2_polar_eq (ρ θ : ℝ) := ρ * cos θ * cos θ = sin θ
def C2_eq (x y : ℝ) := x^2 = y

-- Prove the general and Cartesian equations
theorem curve_equations :
  (∀ α, C1_eq (1 + cos α) (sin α)) ∧
  (∀ (ρ θ : ℝ), x = ρ * cos θ → y = ρ * sin θ → C2_eq x y) :=
by {
  sorry
}

-- Define ray l and intersection points A and B
def ray_eq (x k : ℝ) : ℝ := k * x

-- Prove the range of |OA| * |OB|
theorem range_OA_OB (k : ℝ) (hk : 1 < k ∧ k ≤ sqrt 3) :
  ∃ (OA OB : ℝ), OA * OB ∈ set.Ioo 2 (2 * sqrt 3) :=
by {
  sorry
}

end curve_equations_range_OA_OB_l406_406045


namespace multiply_by_ten_of_specific_x_l406_406204

theorem multiply_by_ten_of_specific_x :
  (∃ (x : ℝ), x * (1/1000) = 0.735) → (∃ (y : ℝ), y = 10 * 735) :=
by {
  intro h,
  cases h with x hx,
  use 10 * x,
  have hy: x = 735,
  {
    field_simp,
    rw ← hx,
    norm_num,
    simp,
    ring,
  },
  rw hy,
  simp,
  ring,
}

end multiply_by_ten_of_specific_x_l406_406204


namespace initial_candies_equal_twenty_l406_406110

-- Definitions based on conditions
def friends : ℕ := 6
def candies_per_friend : ℕ := 4
def total_needed_candies : ℕ := friends * candies_per_friend
def additional_candies : ℕ := 4

-- Main statement
theorem initial_candies_equal_twenty :
  (total_needed_candies - additional_candies) = 20 := by
  sorry

end initial_candies_equal_twenty_l406_406110


namespace train_B_speed_l406_406219

-- Given conditions
def speed_train_A := 70 -- km/h
def time_after_meet_A := 9 -- hours
def time_after_meet_B := 4 -- hours

-- Proof statement
theorem train_B_speed : 
  ∃ (V_b : ℕ),
    V_b * time_after_meet_B + V_b * s = speed_train_A * time_after_meet_A + speed_train_A * s ∧
    V_b = speed_train_A := 
sorry

end train_B_speed_l406_406219


namespace determine_ordered_pair_l406_406078

open Complex

theorem determine_ordered_pair
  (a b : ℝ)
  (h1 : (⟨a, 5⟩ + ⟨b, 6⟩ = ⟨12, 11⟩))
  (h2 : (⟨a, 5⟩ * ⟨b, 6⟩ = ⟨9, 61⟩)) :
  (a, b) = (9, 3) :=
sorry

end determine_ordered_pair_l406_406078


namespace rearrangements_count_l406_406013

def vowels : List Char := ['E', 'E', 'E']
def consonants : List Char := ['R', 'P', 'R', 'S', 'N', 'T']

theorem rearrangements_count : 
  (number_of_distinguishable_rearrangements vowels consonants 
     where_all_vowels_come_first) = 360 := 
  sorry

end rearrangements_count_l406_406013


namespace right_triangle_tan_A_l406_406394

open Real

theorem right_triangle_tan_A (hypotenuse : ℝ) (leg1 : ℝ) (leg2 : ℝ) (A : ℝ)
  (h1 : hypotenuse = 13) (h2 : leg1 = 5) (right_triangle : leg1^2 + leg2^2 = hypotenuse^2)
  (A_opposite_leg1 : tan A = leg1 / leg2) : tan A = 5 / 12 :=
by
  rw [h1, h2] at right_triangle
  have leg2_eq : leg2 = sqrt ((13 : ℝ)^2 - (5 : ℝ)^2),
  { sorry },
  rw leg2_eq at A_opposite_leg1
  have sqrt_eq_12 : sqrt (169 - 25) = 12,
  { sorry },
  rw sqrt_eq_12 at A_opposite_leg1
  exact A_opposite_leg1

end right_triangle_tan_A_l406_406394


namespace sum_of_floor_sqrt_1_to_25_l406_406986

theorem sum_of_floor_sqrt_1_to_25 : 
  ( ∑ i in Finset.range 26, ∥Real.sqrt i∥ ) = 75 :=
by
  sorry

end sum_of_floor_sqrt_1_to_25_l406_406986


namespace vector_problem_l406_406425

variable {V : Type*} [AddCommGroup V]

/-- Let P be a point in the plane of triangle ABC -/
variable (A B C P : V)

/-- Given condition: \overrightarrow{BC} + \overrightarrow{BA} = 2 \overrightarrow{BP} -/
variable (h : (C - B) + (A - B) = 2 * (P - B))

/-- Prove that \overrightarrow{PC} + \overrightarrow{PA} = \overrightarrow{0} -/
theorem vector_problem (C A P : V) (h : (C - B) + (A - B) = 2 * (P - B)) : (C - P) + (A - P) = 0 :=
sorry

end vector_problem_l406_406425


namespace first_term_exceeding_10000_l406_406482

def sequence (a : Nat → Nat) : Prop :=
  a 1 = 3 ∧ ∀ n > 1, a n = 3 * (Finset.sum (Finset.range (n - 1)) (λ k, a (k + 1)))

theorem first_term_exceeding_10000 (a : Nat → Nat) (h : sequence a) : ∃ n, a n > 10000 ∧ a (n - 1) ≤ 10000 ∧ a n = 36864 :=
  sorry

end first_term_exceeding_10000_l406_406482


namespace book_price_decrease_l406_406865

noncomputable def initial_percentage_decrease (P : ℝ) (decrease_amount : ℝ := 10.000000000000014) : ℝ :=
  let x := (50 / 3)
  in x

theorem book_price_decrease (P : ℝ) (decrease_amount : ℝ := 10.000000000000014) :
  (P - (P * (initial_percentage_decrease P / 100))) * 1.20 = P :=
sorry

end book_price_decrease_l406_406865


namespace equal_area_segments_l406_406676

-- Define the lengths of the sides of the isosceles triangle
def AB := 10 -- base of the triangle
def AC := 20 -- legs of the triangle
def BC := 20

-- Define the midpoint of AB
def S_midpoint : Prop := AB / 2 = 5

-- Define segments of the sides of the triangle at equal area division
def AX := 8
def XY := 8
def YC := 4

-- Main theorem to be proven
theorem equal_area_segments :
  ∀ (A B C S : ℝ), 
  (A = 0 ∧ B = AB ∧ C = S ∧ C = AB / 2) →
  (AC = AX + XY + YC) ∧ (AX = XY) ∧ (YC + YC + YC + YC + YC = AC) →
  (YC = 4 ∧ AX = 8) :=
begin
  sorry
end

end equal_area_segments_l406_406676


namespace simplify_log_expression_equals_neg_log2_l406_406471

variable (a b c d : ℝ)

-- Using the properties of logarithms and exponents
axiom log_self_eq_one (base : ℝ) (h : base > 0 ∧ base ≠ 1) : log base base = 1
axiom exp_log_eq_value (x base : ℝ) (hx : x > 0) (hb : base > 0 ∧ base ≠ 1) : base ^ (log base x) = x
axiom log_mul (x y base : ℝ) (hx : x > 0) (hy : y > 0) (hb : base > 0 ∧ base ≠ 1) : log base (x * y) = log base x + log base y
axiom log_pow (x : ℝ) (n : ℕ) (base : ℝ) (hx : x > 0) (hb : base > 0 ∧ base ≠ 1) : log base (x ^ n) = n * log base x

noncomputable def simplify_log_expression : ℝ :=
  (log 6 2) ^ 2 + log 6 2 * log 6 3 + 2 * log 6 3 - 6 ^ (log 6 2)

theorem simplify_log_expression_equals_neg_log2 (h₁ : 6 > 0 ∧ 6 ≠ 1) (h₂ : log 6 2 > 0) (h₃ : log 6 3 > 0) :
  simplify_log_expression = - log 6 2 := by
  sorry

end simplify_log_expression_equals_neg_log2_l406_406471


namespace sum_of_numbers_excluding_special_cube_l406_406935

noncomputable def cube : ℕ := 20
noncomputable def total_unit_cubes : ℕ := cube ^ 3
noncomputable def columns_sum_to_one (n : ℕ) := (n = 20) → (Σ i : fin cube, number_in_column i = 1)
noncomputable def special_cube_position : ℕ × ℕ × ℕ := (1, 1, 1)
noncomputable def number_of_special_cube : ℕ := 10
noncomputable def layers_through_special_cube := [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

theorem sum_of_numbers_excluding_special_cube (cube = 20) 
  (columns_sum: ∀ (i : ℕ), columns_sum_to_one i) 
  (special_pos := special_cube_position) 
  (num_special_cube := number_of_special_cube) : 
  (Σ i : fin total_unit_cubes, number_on_cube i) - 9 = 333 := 
sorry

end sum_of_numbers_excluding_special_cube_l406_406935


namespace compute_f_1986_l406_406017

-- Definition of the function f
def f : ℕ → ℕ

-- Conditions
axiom f_defined_for_all_x_ge_0 : ∀ x : ℕ, x ≥ 0 → f x = f x
axiom f_at_1 : f 1 = 2
axiom functional_eq : ∀ a b : ℕ, a ≥ 0 → b ≥ 0 → f (a + b) = f a + f b - 2 * f (a * b)

-- The main theorem to be proved
theorem compute_f_1986 : f 1986 = 2 :=
by 
  sorry

end compute_f_1986_l406_406017


namespace calculate_exponent_product_l406_406282

theorem calculate_exponent_product : (9/10)^4 * (9/10)^(-4) = 1 := by
  sorry

end calculate_exponent_product_l406_406282


namespace xiao_li_first_three_l406_406769

def q1_proba_correct (p1 p2 p3 : ℚ) : ℚ :=
  p1 * p2 * p3 + 
  (1 - p1) * p2 * p3 + 
  p1 * (1 - p2) * p3 + 
  p1 * p2 * (1 - p3)

theorem xiao_li_first_three (p1 p2 p3 : ℚ) (h1 : p1 = 3/4) (h2 : p2 = 1/2) (h3 : p3 = 5/6) :
  q1_proba_correct p1 p2 p3 = 11 / 24 := by
  rw [h1, h2, h3]
  sorry

end xiao_li_first_three_l406_406769


namespace smallest_n_501_l406_406328

theorem smallest_n_501 (m : ℕ) (n : ℕ) (h : n > 250 ∧ 0.501 ≤ m / n ∧ m / n < 0.502) : n = 251 := 
sorry

end smallest_n_501_l406_406328


namespace simplify_cos_sum_l406_406469

theorem simplify_cos_sum :
  cos (π / 11) + cos (3 * π / 11) + cos (7 * π / 11) + cos (9 * π / 11) = -1 / 2 :=
sorry

end simplify_cos_sum_l406_406469


namespace find_b_l406_406743

open Real

noncomputable def triangle_b (a b c : ℝ) (A B C : ℝ) (sin_A sin_B : ℝ) (area : ℝ) : Prop :=
  B < π / 2 ∧
  sin_B = sqrt 7 / 4 ∧
  area = 5 * sqrt 7 / 4 ∧
  sin_A / sin_B = 5 * c / (2 * b) ∧
  a = 5 / 2 * c ∧
  area = 1 / 2 * a * c * sin_B

theorem find_b (a b c : ℝ) (A B C : ℝ) (sin_A sin_B : ℝ) (area : ℝ) :
  triangle_b a b c A B C sin_A sin_B area → b = sqrt 14 := by
  sorry

end find_b_l406_406743


namespace number_of_students_scoring_above_120_l406_406389

open Real

/-- Given a normally distributed random variable 𝛏 with mean 110 and variance 102,
and given that the probability of the variable being between 100 and 110 is 0.35,
prove that the estimated number of students scoring above 120 out of 60 students is 9. -/
theorem number_of_students_scoring_above_120 :
  ∀ (ξ : ℝ → ℝ) (μ σ : ℝ) (n : ℕ),
  (μ = 110) →
  (σ = sqrt 102) →
  (ξ ~ (Normal μ σ)) →
  (prob (100 ≤ ξ ≤ 110) = 0.35) →
  n = 60 →
  estimated_number ξ (120, ∞) n = 9 :=
by
  intro ξ μ σ n hμ hσ hξ hprob hn
  -- proof goes here
  sorry

end number_of_students_scoring_above_120_l406_406389


namespace distance_traveled_l406_406933

-- Define the velocity function
def v (t : ℝ) : ℝ := 7 - 3 * t + 25 / (1 + t)

-- Define the statement to be proved
theorem distance_traveled : 
  (∫ t in 0..4, v t) = 4 + 25 * Real.log 5 :=
by
  sorry

end distance_traveled_l406_406933


namespace tournament_full_list_exists_l406_406036

-- Conditions of the problem: Tournament setup
variables (Player : Type) [Fintype Player] [DecidableEq Player]

-- Assumptions: No draws and each match has a winner
variable (wins : Player → Player → Prop)
variable [DecidableRel wins]

-- Assumption for contradiction: No player has a list containing all other players
def has_full_list (p : Player) : Prop :=
∀ q : Player, wins p q ∨ ∃ r : Player, wins p r ∧ wins r q

theorem tournament_full_list_exists :
  ∀ p : Player, has_full_list wins p →
  ∃ p : Player, ∀ q : Player, wins p q ∨ ∃ r : Player, wins p r ∧ wins r q :=
sorry

end tournament_full_list_exists_l406_406036


namespace curve_intersects_median_l406_406916

noncomputable def complex_intersection (a b c : ℝ) (h : a + c ≠ 2 * b) : ℂ :=
  let z0 := complex.I * a
  let z1 := 1 / 2 + complex.I * b
  let z2 := 1 + complex.I * c
  let z (t : ℝ) := (z0 * (real.cos t)^4 + 2 * z1 * (real.cos t)^2 * (real.sin t)^2 + z2 * (real.sin t)^4)
  let intersection_point := (1 / 2) + complex.I * ((a + c + 2 * b) / 4)
  intersection_point

theorem curve_intersects_median (a b c : ℝ) (h : a + c ≠ 2 * b) :
  ∃ t : ℝ, complex_intersection a b c h = ((1 / 2 : ℝ) + complex.I * ((a + c + 2 * b) / 4)) :=
sorry

end curve_intersects_median_l406_406916


namespace poods_of_sugar_problem_l406_406926

noncomputable def solve_poods_of_sugar : ℕ :=
  let x := nat.sqrt 2025 - 5 in -- basic computation to find 20
  x

theorem poods_of_sugar_problem (x p : ℕ) 
  (h1 : x * p = 500) 
  (h2 : 500 / (x + 5) = p - 5) 
  : x = 20 := by
  sorry

end poods_of_sugar_problem_l406_406926


namespace new_cost_percentage_l406_406550

variable (t b : ℝ)

-- Define the original cost
def original_cost : ℝ := t * b ^ 4

-- Define the new cost when b is doubled
def new_cost : ℝ := t * (2 * b) ^ 4

-- The theorem statement
theorem new_cost_percentage (t b : ℝ) : new_cost t b = 16 * original_cost t b := 
by
  -- Proof steps are skipped
  sorry

end new_cost_percentage_l406_406550


namespace triangle_equilateral_of_angle_and_side_sequences_l406_406741

theorem triangle_equilateral_of_angle_and_side_sequences 
  (A B C : ℝ) (a b c : ℝ) 
  (h_angles_arith_seq: B = (A + C) / 2)
  (h_sides_geom_seq : b^2 = a * c) 
  (h_sum_angles : A + B + C = 180) 
  (h_pos_angles : 0 < A ∧ 0 < B ∧ 0 < C) 
  (h_pos_sides : 0 < a ∧ 0 < b ∧ 0 < c) :
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c :=
by
  sorry

end triangle_equilateral_of_angle_and_side_sequences_l406_406741


namespace greatest_distance_l406_406310

noncomputable def distance (t : ℝ) : ℝ :=
  real.sqrt (25 + (real.sin t - real.cos (t - 5))^2)

theorem greatest_distance :
  ∃ t : ℝ, distance t = 3 * real.sqrt 3 := sorry

end greatest_distance_l406_406310


namespace license_group_count_l406_406241

-- Defining the conditions
def letters := { 'B', 'N', 'T' }
def odd_digits := { 1, 3, 5, 7, 9 }

-- In this context, no need to define the generic set nature of letters and odd_digits in Lean explicitly, as this setup is straightforward enough.

theorem license_group_count : 
  ∃ (count : ℕ), count = 3 * 5^5 ∧ count = 9375 :=
by {
  use 9375,
  split,
  {
    norm_num,
  },
  {
    refl,
  },
}

end license_group_count_l406_406241


namespace difference_between_cost_and_money_l406_406303

theorem difference_between_cost_and_money :
  let cost_of_cookies := 65
  let dianes_money := 27
  cost_of_cookies - dianes_money = 38 := by
  let cost_of_cookies := 65
  let dianes_money := 27
  show 65 - 27 = 38 from sorry

end difference_between_cost_and_money_l406_406303


namespace max_value_f_on_0_4_l406_406491

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem max_value_f_on_0_4 : ∃ (x : ℝ) (hx : x ∈ Set.Icc (0 : ℝ) (4 : ℝ)), ∀ (y : ℝ), y ∈ Set.Icc (0 : ℝ) (4 : ℝ) → f y ≤ f x ∧ f x = 1 / Real.exp 1 :=
by
  sorry

end max_value_f_on_0_4_l406_406491


namespace gamma_seq_converges_l406_406705

noncomputable def gamma_seq (alpha : ℝ) : ℕ → ℝ
| 0     := some_initial_value -- This needs initial value which is implied from the sequence
| (n+1) := (pi - alpha - gamma_seq alpha n) / 2

theorem gamma_seq_converges (alpha : ℝ) (hα : 0 < alpha ∧ alpha < pi) :
  ∃ (β : ℝ), (∀ ε > 0, ∃ N, ∀ n ≥ N, |gamma_seq alpha n - β| < ε) ∧
             β = (pi - alpha) / 3 :=
by
  sorry

end gamma_seq_converges_l406_406705


namespace quadrant_606_quadrant_minus_950_same_terminal_side_quadrant_minus_97_l406_406227

theorem quadrant_606 (θ : ℝ) : θ = 606 → (180 < (θ % 360) ∧ (θ % 360) < 270) := by
  sorry

theorem quadrant_minus_950 (θ : ℝ) : θ = -950 → (90 < (θ % 360) ∧ (θ % 360) < 180) := by
  sorry

theorem same_terminal_side (α k : ℤ) : (α = -457 + k * 360) ↔ (∃ n : ℤ, α = -457 + n * 360) := by
  sorry

theorem quadrant_minus_97 (θ : ℝ) : θ = -97 → (180 < (θ % 360) ∧ (θ % 360) < 270) := by
  sorry

end quadrant_606_quadrant_minus_950_same_terminal_side_quadrant_minus_97_l406_406227


namespace min_n_for_prime_in_S_l406_406086

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def pairwise_coprime (s : set ℕ) : Prop :=
  ∀ x y : ℕ, x ∈ s → y ∈ s → x ≠ y → Nat.coprime x y

theorem min_n_for_prime_in_S (S : set ℕ) (hS : S = { n | 1 ≤ n ∧ n ≤ 2005 }) :
  ∀ A : set ℕ, (A ⊆ S) → (pairwise_coprime A) → (16 ≤ A.card) → (∃ p : ℕ, p ∈ A ∧ is_prime p) :=
sorry

end min_n_for_prime_in_S_l406_406086


namespace range_a_l406_406422

def set_a (a : ℝ) : set ℝ := { x | -2 ≤ x ∧ x ≤ a }
def set_b (a : ℝ) : set ℝ := { y | ∃ x, -2 ≤ x ∧ x ≤ a ∧ y = 2 * x + 3 }
def set_c (a : ℝ) : set ℝ := { z | ∃ x, -2 ≤ x ∧ x ≤ a ∧ z = x^2 }

theorem range_a (a : ℝ) :
  (∃ x, -2 ≤ x ∧ x ≤ a ∧ 2 * x + 3 = x^2) → 
  (a < -2 ∨ (1 / 2 ≤ a ∧ a ≤ 3)) :=
sorry

end range_a_l406_406422


namespace number_of_elements_in_set_l406_406463

def is_median (s : List ℤ) (m : ℤ) : Prop :=
  let sorted_s := s.sort (· ≤ ·)
  if h : sorted_s.length % 2 = 1 then
    sorted_s.get ⟨sorted_s.length / 2, (Nat.div_lt_iff_lt_mul two_pos).mpr (Nat.add_lt_add_right ((Nat.succ_lt_succ (zero_lt_iff_mk zero_lt_one)).mpr) 1)⟩ = m
  else
    false

def is_range (s : List ℤ) (r : ℤ) : Prop :=
  (s.maximumD 0 - s.minimumD 0) = r

def has_max_element (s : List ℤ) (max_elem : ℤ) : Prop :=
  s.maximumD 0 = max_elem

theorem number_of_elements_in_set :
  ∃ (s : List ℤ), is_median s 10 ∧ is_range s 10 ∧ has_max_element s 20 ∧ s.length ≥ 4 :=
by
  sorry

end number_of_elements_in_set_l406_406463


namespace alpha_gamma_shopping_ways_l406_406962

theorem alpha_gamma_shopping_ways :
  let oreos := 5
  let milks := 3
  let cookies := 2
  let total_items := oreos + milks + cookies

  let alpha_ways := binomial total_items 2
  let gamma_ways_2_items := binomial (oreos + cookies) 2 + (oreos + cookies)
  let case1 := alpha_ways * gamma_ways_2_items

  let alpha_ways_1 := total_items
  let gamma_ways_3_items :=
    binomial (oreos + cookies) 3 +
    (oreos + cookies) * (oreos + cookies - 1) +
    (oreos + cookies)
  let case2 := alpha_ways_1 * gamma_ways_3_items
  
  case1 + case2 = 2100 := by
    sorry

end alpha_gamma_shopping_ways_l406_406962


namespace selection_methods_l406_406330

theorem selection_methods (boys girls : ℕ) (total_selection : ℕ) 
  (at_least_one_boy_and_one_girl : ℕ) (h_boys : boys = 4) (h_girls : girls = 3) 
  (h_total_selection : total_selection = 4) (h_at_least_one_boy_and_one_girl : at_least_one_boy_and_one_girl = 34) : 
  (∃ s : finset (set (fin boys ⊕ fin girls)), s.card = total_selection ∧ s.count (λ x, ∃ (b : fin boys) (g : fin girls), x = b ⊕ g) = at_least_one_boy_and_one_girl) :=
sorry

end selection_methods_l406_406330


namespace intersection_polygon_area_q_l406_406949

def midpoint (A B : Point3D) : Point3D := 
⟨(A.x + B.x) / 2, (A.y + B.y) / 2, (A.z + B.z) / 2⟩

def pyramid_base : list Point3D := 
[⟨0, 0, 0⟩, ⟨5, 0, 0⟩, ⟨5, 5, 0⟩, ⟨0, 5, 0⟩]

def E : Point3D := 
⟨2.5, 2.5, 5/Real.sqrt 2⟩

def P := midpoint ⟨0, 0, 0⟩ E
def Q := midpoint ⟨0, 0, 0⟩ ⟨5, 0, 0⟩
def R := midpoint ⟨5, 5, 0⟩ ⟨0, 5, 0⟩

def vector (A B : Point3D) : Vector3D := 
⟨B.x - A.x, B.y - A.y, B.z - A.z⟩

def cross_product (u v : Vector3D) : Vector3D :=
⟨u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x⟩

noncomputable def intersection_area : ℝ :=
let PQ := vector P Q in
let PR := vector P R in
let normal := cross_product PQ PR in
if normal = ⟨0, 0, 12.5⟩ then 25 else 0 -- area is 25 if correct plane

theorem intersection_polygon_area_q : 
  intersection_area = 25 → ∃ q, q = 25 :=
by {
  intro h,
  use 25,
  exact h,
  sorry -- proof continuation
}

end intersection_polygon_area_q_l406_406949


namespace messages_tuesday_l406_406929

theorem messages_tuesday (T : ℕ) (h1 : 300 + T + (T + 300) + 2 * (T + 300) = 2000) : 
  T = 200 := by
  sorry

end messages_tuesday_l406_406929


namespace area_percentage_less_l406_406152

-- Define the variables and conditions
variable (r1 r2 : ℝ) (A1 A2 : ℝ)

-- Ratio of the radii and area formulas
axiom h1 : r1 / r2 = 3 / 10
axiom h2 : A1 = π * r1^2
axiom h3 : A2 = π * r2^2

-- Lean theorem statement to prove the question
theorem area_percentage_less : ((1 - (A1 / A2)) * 100) = 91 :=
by
  -- Proof will go here
  sorry

end area_percentage_less_l406_406152


namespace no_such_n_exists_l406_406659

def P (n : ℕ) : ℕ := if is_prime_power, then n else 0

theorem no_such_n_exists :
  ∀ (n : ℕ), n > 1 ∧ (P n = Int.sqrt n) ∧ (P (n + 36) = Int.sqrt (n + 36)) → false :=
by
  sorry

end no_such_n_exists_l406_406659


namespace sum_of_areas_of_circles_l406_406175

theorem sum_of_areas_of_circles 
  (r s t : ℝ) 
  (h1 : r + s = 6) 
  (h2 : r + t = 8) 
  (h3 : s + t = 10) 
  : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by 
  sorry

end sum_of_areas_of_circles_l406_406175


namespace math_proof_problem_l406_406308

noncomputable def evaluate_expression : ℕ := 
  let a : ℚ := 15 / 8
  ⌈Real.sqrt a⌉ + ⌈a⌉ + ⌈a^2⌉ + ⌈a^3⌉

theorem math_proof_problem : evaluate_expression = 15 :=
by
  let a : ℚ := 15 / 8
  have h1: ⌈Real.sqrt a⌉ = 2 := sorry
  have h2: ⌈a⌉ = 2 := sorry
  have h3: ⌈a^2⌉ = 4 := sorry
  have h4: ⌈a^3⌉ = 7 := sorry
  calc
    evaluate_expression = ⌈Real.sqrt a⌉ + ⌈a⌉ + ⌈a^2⌉ + ⌈a^3⌉ := by rfl
    ... = 2 + 2 + 4 + 7 := by rw [h1, h2, h3, h4]
    ... = 15 := by norm_num

end math_proof_problem_l406_406308


namespace vector_at_t_neg1_is_correct_l406_406252

-- Define vectors at specific parameters
def vec_t0 : ℝ × ℝ × ℝ := (2, 5, 9)
def vec_t1 : ℝ × ℝ × ℝ := (3, 3, 5)

-- Define the unknown vector at t = -1
def vec_t_neg1 : ℝ × ℝ × ℝ := (1, 7, 13)

theorem vector_at_t_neg1_is_correct : vec_t_neg1 =
  let d := (vec_t1.1 - vec_t0.1, vec_t1.2 - vec_t0.2, vec_t1.3 - vec_t0.3) in
  (vec_t0.1 + (-1) * d.1, vec_t0.2 + (-1) * d.2, vec_t0.3 + (-1) * d.3) :=
by
  sorry

end vector_at_t_neg1_is_correct_l406_406252


namespace area_triangle_AKD_eq_three_halves_P_l406_406037

   variables (A B C D K : Point)
   variables (P : ℝ)
   variables [real_plane]

   open_locale real

   -- Assumptions and definitions based on the problem's conditions
   -- Definitions of points, trapezoid, perpendicular diagonals, angles, etc.
   def trapezoid_ABCD (A B C D : Point) := is_trapezoid A B C D
   def diagonals_perpendicular (A B C D : Point) := is_perpendicular (diagonal A C) (diagonal B D)
   def equal_angles (A B C D : Point) := angle A B C = angle C D B
   def angle_AKD_30 (A K D : Point) := angle A K D = 30

   -- Main theorem statement
   theorem area_triangle_AKD_eq_three_halves_P
     (trapezoid : trapezoid_ABCD A B C D)
     (perpendiculars : diagonals_perpendicular A B C D)
     (angles_eq : equal_angles A B C D)
     (intersect_K : extensions_intersect A B D C = K)
     (angle_AKD : angle_AKD_30 A K D) :
     area_triangle A K D = (3 / 2) * P := sorry
   
end area_triangle_AKD_eq_three_halves_P_l406_406037


namespace calculation_correct_l406_406283

theorem calculation_correct :
  (∑ k in (Finset.range 1013).filter (λ k, odd (2 * k + 1)), (2 * k + 1)) -
  (∑ k in (Finset.range 1012).filter (λ k, even (2 * (k + 1))), (2 * (k + 1))) + 50 = 963 :=
sorry

end calculation_correct_l406_406283


namespace probability_more_than_five_draws_is_20_over_63_l406_406567

open Probability

-- Define the conditions as types
def shiny_pennies : ℕ := 4
def dull_pennies : ℕ := 5
def total_pennies : ℕ := shiny_pennies + dull_pennies

def total_ways : ℕ := nat.choose total_pennies shiny_pennies -- Calculated as 126

-- Define the events and their probabilities
def event_more_than_five_draws : ℕ := (nat.choose 5 3) * (nat.choose 4 1) -- Calculated as 40

def probability_event : ℚ := event_more_than_five_draws / total_ways -- Simplified to 20/63

-- Prove the probability condition
theorem probability_more_than_five_draws_is_20_over_63 :
  probability_event = 20 / 63 ∧ 20 + 63 = 83 :=
by
  have h1 : total_ways = 126 := by sorry
  have h2 : event_more_than_five_draws = 40 := by sorry
  have h3 : probability_event = 20 / 63 := by
    rw [event_more_than_five_draws, total_ways]
    sorry
  exact ⟨h3, rfl⟩

end probability_more_than_five_draws_is_20_over_63_l406_406567


namespace find_unknown_rate_of_two_blankets_l406_406549

-- Definitions of conditions based on the problem statement
def purchased_blankets_at_100 : Nat := 3
def price_per_blanket_at_100 : Nat := 100
def total_cost_at_100 := purchased_blankets_at_100 * price_per_blanket_at_100

def purchased_blankets_at_150 : Nat := 3
def price_per_blanket_at_150 : Nat := 150
def total_cost_at_150 := purchased_blankets_at_150 * price_per_blanket_at_150

def purchased_blankets_at_x : Nat := 2
def blankets_total : Nat := 8
def average_price : Nat := 150
def total_cost := blankets_total * average_price

-- The proof statement
theorem find_unknown_rate_of_two_blankets (x : Nat) 
  (h : purchased_blankets_at_100 * price_per_blanket_at_100 + 
       purchased_blankets_at_150 * price_per_blanket_at_150 + 
       purchased_blankets_at_x * x = total_cost) : x = 225 :=
by sorry

end find_unknown_rate_of_two_blankets_l406_406549


namespace zero_vector_incorrect_statement_l406_406544

noncomputable def is_incorrect (s : String) : Prop :=
  s = "The zero vector has no direction"

noncomputable def zero_vector_conditions : Prop :=
  ∀ v : ℝ^3, (v = 0 ∧ v.direction = arbitrary) ∨ (v = 0 ∧ ∀ w : ℝ^3, ∃ t : ℝ, w = t • v) ∨ (v = 0 → v = 0)

-- Formulate the Lean theorem statement
theorem zero_vector_incorrect_statement : zero_vector_conditions → is_incorrect "The zero vector has no direction" :=
by
  sorry

end zero_vector_incorrect_statement_l406_406544


namespace root_next_interval_bisect_l406_406539

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x - 1

theorem root_next_interval_bisect :
  ∃ c : ℝ, 1 < c ∧ c < 2 ∧ f 1 < 0 ∧ f 2 > 0 ∧ f (3 / 2) < 0 ∧ f (3 / 2) * f 2 < 0 → c ∈ (3 / 2, 2) := 
by
  sorry

end root_next_interval_bisect_l406_406539


namespace initial_contribution_amount_l406_406905

variable (x : ℕ)
variable (workers : ℕ := 1200)
variable (total_with_extra_contribution: ℕ := 360000)
variable (extra_contribution_each: ℕ := 50)

theorem initial_contribution_amount :
  (workers * x = total_with_extra_contribution - workers * extra_contribution_each) →
  workers * x = 300000 :=
by
  intro h
  sorry

end initial_contribution_amount_l406_406905


namespace distance_between_parallel_lines_l406_406479

theorem distance_between_parallel_lines :
  let line1 := λ p : ℝ × ℝ, p.1 + 2 * p.2 - 1 = 0 in
  let line2 := λ p : ℝ × ℝ, 2 * p.1 + 4 * p.2 + 3 = 0 in
  ∃ d : ℝ, 
    (∀ p, line1 p → p = (1, 0)) →
    (∀ p, line2 p → abs (2 * p.1 + 4 * p.2 + 3) / sqrt (2^2 + 4^2) = d) →
    d = sqrt 5 / 2 :=
by
  sorry

end distance_between_parallel_lines_l406_406479


namespace fraction_of_inhabitable_surface_l406_406275

theorem fraction_of_inhabitable_surface {s : Type*} [real_space s] (earth : s) : 
  (1 / 3 * 1 / 3) = 1 / 9 :=
by
  sorry

end fraction_of_inhabitable_surface_l406_406275


namespace remaining_miles_l406_406415

theorem remaining_miles (total_miles : ℕ) (driven_miles : ℕ) (h1: total_miles = 1200) (h2: driven_miles = 642) :
  total_miles - driven_miles = 558 :=
by
  sorry

end remaining_miles_l406_406415


namespace gcd_a_b_l406_406077

noncomputable def a : ℕ := 3333333
noncomputable def b : ℕ := 666666666

theorem gcd_a_b : Nat.gcd a b = 3 := by
  sorry

end gcd_a_b_l406_406077


namespace positive_difference_of_mean_and_median_l406_406605

def GalacticDrop : ℝ := 210
def QuantumLeap : ℝ := 95
def TheSkydiver : ℝ := 155
def FreefallExtreme : ℝ := 275
def InverseReaction : ℝ := 125
def TwisterMax : ℝ := 250

def heights : List ℝ := [GalacticDrop, QuantumLeap, TheSkydiver, FreefallExtreme, InverseReaction, TwisterMax]

def mean (l : List ℝ) : ℝ :=
  (l.sum) / (l.length)

def median (l : List ℝ) : ℝ :=
  let sorted_l := l.qsort (≤)
  if l.length % 2 = 0 then (sorted_l.nth! (l.length / 2 - 1) + sorted_l.nth! (l.length / 2)) / 2
  else sorted_l.nth! (l.length / 2)

def positive_difference (a b : ℝ) : ℝ :=
  abs (a - b)

theorem positive_difference_of_mean_and_median : 
  positive_difference (mean heights) (median heights) = 2.5 :=
sorry

end positive_difference_of_mean_and_median_l406_406605


namespace valid_sum_range_l406_406230

-- Define the set of numbers and the range of sums.
def nums := {n | 1 ≤ n ∧ n ≤ 20}

-- Define the function that calculates the sum of five elements from the set.
def sum_five (s : Finset ℕ) : ℕ := s.sum id

theorem valid_sum_range (s : Finset ℕ) (h1 : s ⊆ nums) (h2 : s.card = 5) :
  15 ≤ sum_five s ∧ sum_five s ≤ 90 :=
by
  sorry

end valid_sum_range_l406_406230


namespace red_balloon_probability_l406_406783

-- Define the conditions
def initial_red_balloons := 2
def initial_blue_balloons := 4
def inflated_red_balloons := 2
def inflated_blue_balloons := 2

-- Define the total number of balloons after inflation
def total_red_balloons := initial_red_balloons + inflated_red_balloons
def total_blue_balloons := initial_blue_balloons + inflated_blue_balloons
def total_balloons := total_red_balloons + total_blue_balloons

-- Define the probability calculation
def red_probability := (total_red_balloons : ℚ) / total_balloons * 100

-- The theorem to prove
theorem red_balloon_probability : red_probability = 40 := by
  sorry -- Skipping the proof itself

end red_balloon_probability_l406_406783


namespace simplify_expression_l406_406909

theorem simplify_expression (b : ℝ) (h : b > 2) :
  ( (b^2 - 3*b - (b-1)*√(b^2 - 4) + 2) / (b^2 + 3*b - (b+1)*√(b^2 - 4) + 2) ) * √( (b + 2) / (b - 2) )
  = (1 - b) / (1 + b) :=
by
  sorry

end simplify_expression_l406_406909


namespace no_real_solution_log_eq_l406_406732

theorem no_real_solution_log_eq :
  ∀ x : ℝ, x + 5 > 0 → x - 3 > 0 → x^2 - 8x + 7 > 0 → ¬(log (x + 5) + log (x - 3) = log (x^2 - 8x + 7)) :=
by
  sorry

end no_real_solution_log_eq_l406_406732


namespace degree_to_radian_radian_to_degree_l406_406294

theorem degree_to_radian (d : ℝ) (rad : ℝ) (deg : ℝ) :
  (180 * rad = π) → (d = 210) → rad = (π / 180) → d * rad = 7 * π / 6 :=
by sorry 

theorem radian_to_degree (r : ℝ) (rad : ℝ) (deg : ℝ) :
  (180 * rad = π) → (r = -5 * π / 2) → deg = (180 / π) → r * deg = -450 :=
by sorry

end degree_to_radian_radian_to_degree_l406_406294


namespace number_of_female_students_in_sample_l406_406038

theorem number_of_female_students_in_sample (male_students female_students sample_size : ℕ)
  (h1 : male_students = 560)
  (h2 : female_students = 420)
  (h3 : sample_size = 280) :
  (female_students * sample_size) / (male_students + female_students) = 120 := 
sorry

end number_of_female_students_in_sample_l406_406038


namespace sections_created_by_5_lines_l406_406376

-- Define the basic setup: a rectangle and proper line segments drawn inside it.
variables (rectangle : Type) [linear_ordered_field rectangle]
noncomputable def num_sections (line_segments : ℕ) : ℕ :=
  match line_segments with
  | 0     => 1
  | n + 1 => num_sections n + n + 1
  end

-- Statement of the problem: Prove that 5 line segments create 16 sections in a rectangle.
theorem sections_created_by_5_lines : num_sections 5 = 16 := by
  sorry

end sections_created_by_5_lines_l406_406376


namespace length_faster_train_in_meters_l406_406556

-- Define the conditions
def speed_faster_train_kmph : ℕ := 72
def speed_slower_train_kmph : ℕ := 36
def time_crossing_seconds : ℕ := 7

-- Convert speeds from kmph to m/s
def kmph_to_mps (kmph : ℕ) : ℝ :=
  (kmph * 1000) / 3600

def speed_faster_train_mps : ℝ := kmph_to_mps speed_faster_train_kmph
def speed_slower_train_mps : ℝ := kmph_to_mps speed_slower_train_kmph

-- Calculate the relative speed in m/s
def relative_speed_mps : ℝ := speed_faster_train_mps - speed_slower_train_mps

-- Define the proof statement: the length of the faster train
theorem length_faster_train_in_meters : relative_speed_mps * (time_crossing_seconds : ℝ) = 70 :=
by
  sorry

end length_faster_train_in_meters_l406_406556


namespace triangle_smallest_side_l406_406502

theorem triangle_smallest_side (a b c : ℝ) (h : b^2 + c^2 ≥ 5 * a^2) : 
    (a ≤ b ∧ a ≤ c) := 
sorry

end triangle_smallest_side_l406_406502


namespace card_of_A_l406_406420

variables {α : Type*} -- Type of elements in the sets
variables (n k : ℕ) -- Natural numbers n and k
variables (A : set α) -- Set A
variables (A_i : ℕ → set α) -- Family of sets A_i

/-- Given the conditions of the problem, we prove the cardinality of A --/
theorem card_of_A (h₀ : ∀ i, i ≥ 1 ∧ i ≤ n+1 → (A_i i).finite)
    (h₁ : ∀ i, i ≥ 1 ∧ i ≤ n+1 → (A_i i).card = n)
    (h₂ : ∀ i j, i ≠ j → i ≥ 1 ∧ i ≤ n+1 → j ≥ 1 ∧ j ≤ n+1 → ((A_i i) ∩ (A_i j)).finite)
    (h₃ : ∀ i j, i ≠ j → i ≥ 1 ∧ i ≤ n+1 → j ≥ 1 ∧ j ≤ n+1 → ((A_i i) ∩ (A_i j)).card ≤ k)
    (h₄ : A = ⋃ i in finset.range (n+1), A_i i)
    (hA : A.finite ∧ A.card ≤ n * (n + 1) / (k + 1)) :
  A.card = n * (n + 1) / (k + 1) := sorry

end card_of_A_l406_406420


namespace radius_of_passing_circle_l406_406517

-- Given two intersecting circles with radius 2 units
-- intersecting at points A and B
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the two circles
def circle₁ : Circle := { center := (0, 0), radius := 2 }
def circle₂ : Circle := { center := (d, 0), radius := 2 }  -- intersect distance d

-- Assume the points of intersection
def A : ℝ × ℝ := sorry  -- Point of intersection
def B : ℝ × ℝ := sorry  -- Second point of intersection

-- Tangents touching at points E and F
def E : ℝ × ℝ := sorry
def F : ℝ × ℝ := sorry

-- The new circle passing through E, F, and A
def circle₃ : Circle := { center := (e, f), radius := 1 }

-- The target theorem
theorem radius_of_passing_circle (h₁ : circle₁.radius = 2)
                                  (h₂ : circle₂.radius = 2)
                                  (e : (ℝ × ℝ)) (f : (ℝ × ℝ)) (h₃ : circle₃.center = (e, f))
                                  (h₄ : A ≠ B) (h₅ : E ≠ F) :
  circle₃.radius = 1 :=
sorry

end radius_of_passing_circle_l406_406517


namespace test_scores_order_l406_406781

def kaleana_score : ℕ := 75

variable (M Q S : ℕ)

-- Assuming conditions from the problem
axiom h1 : Q = kaleana_score
axiom h2 : M < max Q S
axiom h3 : S > min Q M
axiom h4 : M ≠ Q ∧ Q ≠ S ∧ M ≠ S

-- Theorem statement
theorem test_scores_order (M Q S : ℕ) (h1 : Q = kaleana_score) (h2 : M < max Q S) (h3 : S > min Q M) (h4 : M ≠ Q ∧ Q ≠ S ∧ M ≠ S) :
  M < Q ∧ Q < S :=
sorry

end test_scores_order_l406_406781


namespace triangles_similarity_and_brocard_point_l406_406113

noncomputable def common_angle {A B C A1 B1 C1 : Type} [HasAngle A B C]
  (hA1 : A1 ∈ line CA) (hB1 : B1 ∈ line AB) (hC1 : C1 ∈ line BC)
  (h_angles : angle (entity AB1 A1) (entity BC1 B1) = angle (entity CA1 C1) :=
    angle := ...) : 
  Type := sorry

theorem triangles_similarity_and_brocard_point
  {A B C A1 B1 C1 O : Type} [HasAngle A B C] [HasBrocardPoint A B C] [HasBrocardPoint A1 B1 C1]
  (h1 : A1 ∈ line CA) (h2 : B1 ∈ line AB) (h3 : C1 ∈ line BC)
  (h4 : common_angle h1 h2 h3) :
  similar (triangle A1 B1 C1) (triangle A B C) ∧
  brocard_center (triangle A B C) = brocard_center (triangle A1 B1 C1) :=
sorry

end triangles_similarity_and_brocard_point_l406_406113


namespace cos_arcsin_plus_sin_arccos_eq_one_l406_406609

theorem cos_arcsin_plus_sin_arccos_eq_one :
  cos (arcsin (8 / 17)) + sin (arccos (15 / 17)) = 1 :=
by
  sorry

end cos_arcsin_plus_sin_arccos_eq_one_l406_406609


namespace total_unique_plants_l406_406511

open Finset

def bed_A : Finset ℕ := {1,2, ..., 500}.to_finset  -- Placeholders for example
def bed_B : Finset ℕ := {1001, 1002, ..., 1450}.to_finset
def bed_C : Finset ℕ := {2001, 2002, ..., 2350}.to_finset

theorem total_unique_plants :
  |bed_A| + |bed_B| + |bed_C| - |bed_A ∩ bed_B| - |bed_A ∩ bed_C| - |bed_B ∩ bed_C| + |bed_A ∩ bed_B ∩ bed_C| = 1150 :=
by
  -- Given conditions
  have h1 : |bed_A| = 500 := by simp only [card_to_finset, nat.card_fin_eq, card_range, finset.card_fin_eq]
  have h2 : |bed_B| = 450 := by simp only [card_to_finset, nat.card_fin_eq, card_range, finset.card_fin_eq]
  have h3 : |bed_C| = 350 := by simp only [card_to_finset, nat.card_fin_eq, card_range, finset.card_fin_eq]
  have h4 : |bed_A ∩ bed_B| = 50 := by sorry
  have h5 : |bed_A ∩ bed_C| = 100 := by sorry
  have h6 : |bed_B ∩ bed_C| = 0 := by sorry
  have h7 : |bed_A ∩ bed_B ∩ bed_C| = 0 := by sorry

  -- Apply Inclusion-Exclusion Principle
  calc
    |bed_A| + |bed_B| + |bed_C| - |bed_A ∩ bed_B| - |bed_A ∩ bed_C| - |bed_B ∩ bed_C| + |bed_A ∩ bed_B ∩ bed_C| = 500 + 450 + 350 - 50 - 100 - 0 + 0 : by congr; assumption
    ... = 1150 : by norm_num

end total_unique_plants_l406_406511


namespace probability_A_eq_B_l406_406957

open Real

noncomputable def problem_statement : Prop :=
  let interval := Set.Icc (-5 * π / 2) (5 * π / 2)
  ∃ (a b : ℝ), a ∈ interval ∧ b ∈ interval ∧ cos (cos a) = cos (cos b) → a = b

theorem probability_A_eq_B : problem_statement → (A = B) :=
  begin
    -- Add formal proof here 
    sorry
  end

end probability_A_eq_B_l406_406957


namespace integral_equality_l406_406559

noncomputable def integral_expression (α : ℝ) (hα : 0 < α ∧ α < 1) : ℝ :=
  ∫ (ϕ : ℝ) in 0..2*π, 1 / (1 + α^2 - 2 * α * real.cos ϕ)

theorem integral_equality (α : ℝ) (hα : 0 < α ∧ α < 1) :
  integral_expression α hα = 2 * π / (1 - α^2) :=
by
  sorry

end integral_equality_l406_406559


namespace addition_of_fractions_l406_406281

theorem addition_of_fractions : (6/7 : ℚ) + (7/9 : ℚ) = 103/63 := by
  sorry

end addition_of_fractions_l406_406281


namespace simplify_and_evaluate_l406_406470

def expr (a b : ℤ) := -a^2 * b + (3 * a * b^2 - a^2 * b) - 2 * (2 * a * b^2 - a^2 * b)

theorem simplify_and_evaluate : expr (-1) (-2) = -4 := by
  sorry

end simplify_and_evaluate_l406_406470


namespace sum_floor_sqrt_1_to_25_l406_406998

theorem sum_floor_sqrt_1_to_25 :
  (∑ n in finset.range 25, ⌊real.sqrt (n + 1)⌋) = 75 :=
sorry

end sum_floor_sqrt_1_to_25_l406_406998


namespace farmer_full_price_l406_406246

theorem farmer_full_price : 
  let total_spent := 120 in
  let chicken_feed := 0.30 * total_spent in
  let goat_feed := 0.20 * total_spent in
  let cow_horse_feed := 0.50 * total_spent in
  let chicken_full_price := chicken_feed / 0.60 in
  let goat_full_price := goat_feed / 0.90 in
  chicken_full_price + goat_full_price + cow_horse_feed = 146.67 :=
by
  -- Proof will be provided here
  sorry

end farmer_full_price_l406_406246


namespace min_PM_PN_min_PM_squared_PN_squared_l406_406332

noncomputable def min_value_PM_PN := 3 * Real.sqrt 5

noncomputable def min_value_PM_squared_PN_squared := 229 / 10

structure Point :=
  (x : ℝ)
  (y : ℝ)

def M : Point := ⟨2, 5⟩
def N : Point := ⟨-2, 4⟩

def on_line (P : Point) : Prop :=
  P.x - 2 * P.y + 3 = 0

theorem min_PM_PN {P : Point} (h : on_line P) :
  dist (P.x, P.y) (M.x, M.y) + dist (P.x, P.y) (N.x, N.y) = min_value_PM_PN := sorry

theorem min_PM_squared_PN_squared {P : Point} (h : on_line P) :
  (dist (P.x, P.y) (M.x, M.y))^2 + (dist (P.x, P.y) (N.x, N.y))^2 = min_value_PM_squared_PN_squared := sorry

end min_PM_PN_min_PM_squared_PN_squared_l406_406332


namespace solveForT_l406_406485

def height (t : ℝ) : ℝ := 60 - 8 * t - 5 * t^2

theorem solveForT : ∃ t : ℝ, height t = 40 ∧ t = 3.308 := 
by
  use 3.308
  constructor
  { show height 3.308 = 40, sorry }
  { show 3.308 = 3.308, refl }

end solveForT_l406_406485


namespace sophie_germain_identity_factorization_math_problem_solution_l406_406613

theorem sophie_germain_identity_factorization (x : ℕ) : x^4 + 324 = (x^2 - 6*x + 18) * (x^2 + 6*x + 18) := 
by sorry

theorem math_problem_solution :
    (\frac{(12^4 + 324) * (24^4 + 324) * (36^4 + 324) * (48^4 + 324) * (60^4 + 324)}{(6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324)} = 221) 
:= by 
    have h1 := sophie_germain_identity_factorization 12,
    have h2 := sophie_germain_identity_factorization 24,
    have h3 := sophie_germain_identity_factorization 36,
    have h4 := sophie_germain_identity_factorization 48,
    have h5 := sophie_germain_identity_factorization 60,
    have h6 := sophie_germain_identity_factorization 6,
    have h7 := sophie_germain_identity_factorization 18,
    have h8 := sophie_germain_identity_factorization 30,
    have h9 := sophie_germain_identity_factorization 42,
    have h10 := sophie_germain_identity_factorization 54,
    sorry

end sophie_germain_identity_factorization_math_problem_solution_l406_406613


namespace find_increasing_intervals_l406_406715

variable {X : Type*} [has_le X] [has_lt X] [linear_order X]

def is_even (f : X → X) : Prop :=
  ∀ x, f x = f (-x)

def is_decreasing_on (f : X → X) (S : set X) : Prop :=
  ∀ ⦃x y : X⦄, x ∈ S → y ∈ S → x ≤ y → f y ≤ f x

def is_increasing_on (f : X → X) (S : set X) : Prop :=
  ∀ ⦃x y : X⦄, x ∈ S → y ∈ S → x ≤ y → f x ≤ f y

noncomputable def intervals_increasing (f : X → X) : set X :=
  {x | x ≤ -1} ∪ {x | 0 < x ∧ x ≤ 1}

theorem find_increasing_intervals (f : X → X) :
  is_even f →
  is_decreasing_on f {x | 0 ≤ x} →
  is_increasing_on (λ x, f (1 - x^2)) (intervals_increasing f) :=
by
  sorry

end find_increasing_intervals_l406_406715


namespace number_of_triangles_formed_l406_406243

theorem number_of_triangles_formed (dots_on_AB dots_on_BC dots_on_CA : ℕ) 
(h_AB : dots_on_AB = 2) 
(h_BC : dots_on_BC = 3) 
(h_CA : dots_on_CA = 7) 
: 
  ( let total_dots := 3 + dots_on_AB + dots_on_BC + dots_on_CA in
    let total_combinations := nat.choose total_dots 3 in
    let collinear_AB := nat.choose (dots_on_AB + 2) 3 in
    let collinear_BC := nat.choose (dots_on_BC + 2) 3 in
    let collinear_CA := nat.choose (dots_on_CA + 2) 3 in
    let total_collinear := collinear_AB + collinear_BC + collinear_CA in
    total_combinations - total_collinear ) = 357 :=
by
  sorry

end number_of_triangles_formed_l406_406243


namespace right_triangles_count_l406_406041

def squared_distance (x0 y0 x1 y1 : ℤ) : ℤ :=
  (x1 - x0)^2 + (y1 - y0)^2

def forms_right_triangle (a b c : ℤ) : Prop :=
  let AB2 := squared_distance (-1) a 0 b in
  let BC2 := squared_distance 0 b 1 c in
  let AC2 := squared_distance (-1) a 1 c in
  AC2 = AB2 + BC2 ∨ AB2 = AC2 + BC2 ∨ BC2 = AC2 + AB2

theorem right_triangles_count : 
  (∑ a b c in Finset.range 101, if 1 ≤ a ∧ a ≤ 100 ∧
                                    1 ≤ b ∧ b ≤ 100 ∧ 
                                    1 ≤ c ∧ c ≤ 100 ∧
                                    forms_right_triangle a b c
                                then 1 else 0) = 974 := 
by sorry

end right_triangles_count_l406_406041


namespace sum_areas_of_circles_l406_406164

theorem sum_areas_of_circles (r s t : ℝ) 
  (h1 : r + s = 6)
  (h2 : r + t = 8)
  (h3 : s + t = 10) : 
  r^2 * Real.pi + s^2 * Real.pi + t^2 * Real.pi = 56 * Real.pi := 
by 
  sorry

end sum_areas_of_circles_l406_406164


namespace distance_traveled_by_P_l406_406212

-- Definitions for triangle sides and the circle's radius
def side_a : ℝ := 8
def side_b : ℝ := 10
def side_c : ℝ := 12.5
def radius : ℝ := 1

-- The aim is to prove that the distance traveled by P is 15.25
theorem distance_traveled_by_P :
  let x := 1 in
  4 * x + 5 * x + 6.25 * x = 15.25 :=
by
  let x := 1
  exact rfl

end distance_traveled_by_P_l406_406212


namespace hypotenuse_right_triangle_l406_406197

theorem hypotenuse_right_triangle (a b : ℕ) (h₁ : a = 80) (h₂ : b = 150) : 
  ∃ c, c = 170 ∧ c^2 = a^2 + b^2 :=
by
  use 170
  simp [h₁, h₂]
  norm_num
  sorry

end hypotenuse_right_triangle_l406_406197


namespace cotangent_distribution_equivalence_cosine_squared_distribution_equivalence_l406_406075

noncomputable section

open MeasureTheory

variables {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

def uniformDistribution : ProbabilityDistribution ℝ := sorry
def cauchyDistribution : ProbabilityDistribution ℝ := sorry

variable {θ : Ω → ℝ}
variable {C : Ω → ℝ}

axiom θ_uniform : (μ.with_density (λ _, (uniformDistribution.density (/2π)))).pdf θ = (μ.with_density (λ _, 1)).pdf θ
axiom C_cauchy : μ.with_density (cauchyDistribution.density C) = μ.with_density (λ _, 1 / (π * (1 + C^2)))

theorem cotangent_distribution_equivalence :
  (∀ ω, MeasureTheory.PDF (λ ω, Real.cot (θ ω))) =
  (∀ ω, MeasureTheory.PDF (λ ω, Real.cot ((θ ω) / 2))) ∧
  (∀ ω, MeasureTheory.PDF (λ ω, Real.cot ((θ ω) / 2))) =
  (∀ ω, MeasureTheory.PDF (λ ω, C ω)) :=
sorry

theorem cosine_squared_distribution_equivalence :
  (∀ ω, MeasureTheory.PDF (λ ω, Real.cos (θ ω) ^ 2)) =
  (∀ ω, MeasureTheory.PDF (λ ω, Real.cos ((θ ω) / 2) ^ 2)) ∧
  (∀ ω, MeasureTheory.PDF (λ ω, Real.cos ((θ ω) / 2) ^ 2)) =
  (∀ ω, MeasureTheory.PDF (λ ω, (C ω ^ 2) / (1 + C ω ^ 2))) :=
sorry

end cotangent_distribution_equivalence_cosine_squared_distribution_equivalence_l406_406075


namespace sum_first_10_terms_l406_406504

variable {a : ℕ → ℕ}
variable S : ℕ → ℕ

-- Define arithmetic sequence property
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n_terms (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a 1 + a n)) / 2

-- Given conditions
axiom a5_a6_sum : a 5 + a 6 = 18

-- Definition of S₁₀ based on sum of first n terms
def S_10 := sum_first_n_terms a 10

theorem sum_first_10_terms :
  is_arithmetic_sequence a →
  (S 10 = S_10) →
  a5_a6_sum →
  S 10 = 90 :=
by
  intros h_arith h_S10 h_sum
  sorry

end sum_first_10_terms_l406_406504


namespace smallest_square_board_for_battleships_l406_406774

-- Definition to represent the problem
def battleship_ships : Nat := 10  -- Standard set of battleships
def ship_dims := [(1, 4), (1, 3), (1, 3), (1, 2), (1, 2), (1, 2), (1, 1), (1, 1), (1, 1), (1, 1)] -- Sizes of ships
def grid_dims (n : Nat) := n * n -- Dimensions of the square board

-- Calculating if all ships can fit on a given grid size without touching each other
def can_place_ships (n : Nat) : Bool :=
  let total_occupied_nodes := 10 + 2 * 8 + 3 * 6 + 4 * 4 
  let grid_nodes := (n + 1) * (n + 1)
  grid_nodes ≥ total_occupied_nodes

-- The main theorem
theorem smallest_square_board_for_battleships : ∃ n, can_place_ships 7 ∧ (∀ m < 7, ¬ can_place_ships m) :=
  by
    sorry

end smallest_square_board_for_battleships_l406_406774


namespace jack_lifetime_l406_406216

-- Defining Jack's lifetime as a variable
variable (L : ℕ)

-- Given conditions
def condition_1 := L = 6 * adolescent_life
def condition_2 := facial_hair_start := adolescent_life + 1 / 12 * L
def condition_3 := marriage := facial_hair_start + 1 / 7 * L
def condition_4 := son_birth := marriage + 5
def condition_5 := son_lifetime := 1 / 2 * L
def condition_6 := jack_death := son_lifetime + 4

-- Calculating Jack's lifetime based on the conditions
theorem jack_lifetime : 
  (1 / 6) * L + (1 / 12) * L + (1 / 7) * L + 5 + (1 / 2) * L + 4 = L
:=
by
  sorry

end jack_lifetime_l406_406216


namespace income_calculation_l406_406487

theorem income_calculation (x : ℕ) (h1 : ∃ x : ℕ, income = 8*x ∧ expenditure = 7*x)
  (h2 : savings = 5000)
  (h3 : income = expenditure + savings) : income = 40000 :=
by {
  sorry
}

end income_calculation_l406_406487


namespace Jillian_had_200_friends_l406_406056

def oranges : ℕ := 80
def pieces_per_orange : ℕ := 10
def pieces_per_friend : ℕ := 4
def number_of_friends : ℕ := oranges * pieces_per_orange / pieces_per_friend

theorem Jillian_had_200_friends :
  number_of_friends = 200 :=
sorry

end Jillian_had_200_friends_l406_406056


namespace count_no_carry_pairs_l406_406325

def is_no_carry_pair (x y : ℕ) : Prop :=
  ∀ i, (x % (10^(i+1)) - x % 10^i + y % (10^(i+1)) - y % 10^i) / 10^i < 10

def consecutive_pairs_no_carry (n m : ℕ) : Prop :=
  (n + 1 = m) ∧ (is_no_carry_pair n m)

theorem count_no_carry_pairs : 
  let pairs := (λ n, consecutive_pairs_no_carry n (n+1)) in
    (list.range' 1100 (2201 - 1100)).filter pairs |>.length = 1100 :=
by
  sorry

end count_no_carry_pairs_l406_406325


namespace orange_juice_serving_size_l406_406956

theorem orange_juice_serving_size (n_servings : ℕ) (c_concentrate : ℕ) (v_concentrate : ℕ) (c_water_per_concentrate : ℕ)
    (v_cans : ℕ) (expected_serving_size : ℕ) 
    (h1 : n_servings = 200)
    (h2 : c_concentrate = 60)
    (h3 : v_concentrate = 5)
    (h4 : c_water_per_concentrate = 3)
    (h5 : v_cans = 5)
    (h6 : expected_serving_size = 6) : 
   (c_concentrate * v_concentrate + c_concentrate * c_water_per_concentrate * v_cans) / n_servings = expected_serving_size := 
by 
  sorry

end orange_juice_serving_size_l406_406956


namespace distance_between_stores_l406_406412

-- Define the variables and constants
variables (x : ℝ)

-- Define the conditions
def condition1 : Prop := ∃ x, x > 0
def condition2 : x = x + (2/3) * x
def condition3 : 4
def condition4 : 4
def condition5 : 4 + x + (x + (2/3) * x) + 4 = 24

-- State the theorem
theorem distance_between_stores : ∀ (x : ℝ), condition1 → condition2 → (x = (9.6 : ℝ)) :=
by
  assume x,
  sorry

end distance_between_stores_l406_406412


namespace tetra_lines_concur_l406_406804

noncomputable def are_lines_concurrent : Prop :=
  ∀ (A B C : Type) [RegularTetrahedron A] [RegularTetrahedron B] [RegularTetrahedron C]
    (A1 A2 A3 A4 B1 B2 B3 B4 C1 C2 C3 C4 : A)
    (hA: distinct A1 A2 A3 A4) (hB: distinct B1 B2 B3 B4) (hC: distinct C1 C2 C3 C4)
    (h_no_cong1: ¬ congruent A B) (h_no_cong2: ¬ congruent A C) (h_no_cong3: ¬ congruent B C),
      (∀ i: fin 4, midpoint (A i) (B i) (C i)) →
      concurrent (A1, B1) (A2, B2) (A3, B3) (A4, B4)

theorem tetra_lines_concur : are_lines_concurrent :=
sorry

end tetra_lines_concur_l406_406804


namespace polygon_sides_sum_l406_406505

theorem polygon_sides_sum (triangle_hexagon_sum : ℕ) (triangle_sides : ℕ) (hexagon_sides : ℕ) 
  (h1 : triangle_hexagon_sum = 1260) 
  (h2 : triangle_sides = 3) 
  (h3 : hexagon_sides = 6) 
  (convex : ∀ n, 3 <= n) : 
  triangle_sides + hexagon_sides + 4 = 13 :=
by 
  sorry

end polygon_sides_sum_l406_406505


namespace harmony_statements_correct_l406_406426

def is_harmonious_set (S : set ℝ) : Prop :=
  S.nonempty ∧ (∀ a b ∈ S, a + b ∈ S ∧ a - b ∈ S)

def harmonious_set_statement_A : Prop :=
  let S := {0} in is_harmonious_set S ∧ S.finite

def harmonious_set_statement_B : Prop :=
  let S := {x | ∃ k : ℤ, x = real.sqrt 3 * k} in is_harmonious_set S

def harmonious_set_statement_C : Prop :=
  ∀ S₁ S₂ : set ℝ, is_harmonious_set S₁ → is_harmonious_set S₂ → (S₁ ∩ S₂).nonempty

def harmonious_set_statement_D : Prop :=
  ∀ S₁ S₂ : set ℝ, is_harmonious_set S₁ → is_harmonious_set S₂ → S₁ = S₂ ∨ (S₁ ∪ S₂ = @set.univ ℝ)

theorem harmony_statements_correct :
  harmonious_set_statement_A ∧ harmonious_set_statement_B ∧ harmonious_set_statement_C ∧ ¬ harmonious_set_statement_D :=
by
  split
  { sorry }
  split
  { sorry }
  split
  { sorry }
  { by_contradiction H,
    sorry }

end harmony_statements_correct_l406_406426


namespace rational_roots_of_equation_l406_406320

theorem rational_roots_of_equation :
  (∀ x : ℝ, x > -2 ∧ x ≠ 0 →
     (∃ r : ℚ, (r : ℝ) = x ∧ 
       (sqrt (x + 2) / |x| + |x| / sqrt (x + 2) = (4/3) * sqrt 3) →
         x = 1 ∨ x = -2/3)) :=
by
  sorry

end rational_roots_of_equation_l406_406320


namespace sum_of_reciprocal_squares_lt_formula_l406_406447

-- Define the main theorem based on identified conjecture
theorem sum_of_reciprocal_squares_lt_formula (n : ℕ) (h : n ≥ 2) :
  1 + ∑ i in finset.range (n - 1), (1 / (i + 2)^2 : ℝ) < (2 * n - 1 : ℝ) / n :=
sorry

end sum_of_reciprocal_squares_lt_formula_l406_406447


namespace find_b_find_sin_2C_l406_406771

-- Define the conditions of the triangle.
def triangle_conditions (a b c : ℝ) (cos_B : ℝ) :=
a = 2 ∧ c = 3 ∧ cos_B = 1 / 4

-- Prove that b = sqrt(10) given the conditions.
theorem find_b (a b c cos_B : ℝ) (h : triangle_conditions a b c cos_B) : b = Real.sqrt 10 :=
by
  rcases h with ⟨ha, hc, hcos_B⟩
  sorry

-- Prove that sin 2C = 3 * sqrt(15) / 16 given the conditions and b = sqrt(10).
theorem find_sin_2C (a b c cos_B : ℝ) (h : triangle_conditions a b c cos_B) (hb : b = Real.sqrt 10) : 
  sin (2 * acos ((a^2 + b^2 - c^2) / (2 * a * b))) = 3 * Real.sqrt 15 / 16 :=
by
  rcases h with ⟨ha, hc, hcos_B⟩
  sorry

end find_b_find_sin_2C_l406_406771


namespace no_such_abc_exists_l406_406767

-- Define the conditions for the leading coefficients and constant terms
def leading_coeff_conditions (a b c : ℝ) : Prop :=
  ((a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ c > 0 ∧ b < 0) ∨ (b > 0 ∧ c > 0 ∧ a < 0))

def constant_term_conditions (a b c : ℝ) : Prop :=
  ((c > 0 ∧ a < 0 ∧ b < 0) ∨ (a > 0 ∧ b < 0 ∧ c < 0) ∨ (b > 0 ∧ c < 0 ∧ a < 0))

-- The final statement that encapsulates the contradiction
theorem no_such_abc_exists : ¬ ∃ a b c : ℝ, leading_coeff_conditions a b c ∧ constant_term_conditions a b c :=
by
  sorry

end no_such_abc_exists_l406_406767


namespace total_swordfish_caught_l406_406467

theorem total_swordfish_caught (fishing_trips : ℕ) (shelly_each_trip : ℕ) (sam_each_trip : ℕ) : 
  shelly_each_trip = 3 → 
  sam_each_trip = 2 → 
  fishing_trips = 5 → 
  (shelly_each_trip + sam_each_trip) * fishing_trips = 25 :=
by
  sorry

end total_swordfish_caught_l406_406467


namespace problem_statement_l406_406794

def S := {x : ℝ // 0 < x}

def g (x : S) : ℝ :=
sorry  -- The function g is defined but not specified here as the solution steps are not included.

theorem problem_statement :
  (let m := 1 in let t := (6007 / 2 : ℝ) in m * t = 6007 / 2) :=
by
  sorry

end problem_statement_l406_406794


namespace f_2017_eq_2_l406_406670

noncomputable def f : ℝ → ℝ
| x := if x < 0 then x^3 - 1
       else if -1 ≤ x ∧ x ≤ 1 then f (-x)
       else if x > 1/2 then f (x - 1)

theorem f_2017_eq_2 : f 2017 = 2 := 
sorry

end f_2017_eq_2_l406_406670


namespace A_and_C_amount_l406_406594

variables (A B C : ℝ)

def amounts_satisfy_conditions : Prop :=
  (A + B + C = 500) ∧ (B + C = 320) ∧ (C = 20)

theorem A_and_C_amount (h : amounts_satisfy_conditions A B C) : A + C = 200 :=
by {
  sorry
}

end A_and_C_amount_l406_406594


namespace frances_towels_weight_in_ounces_l406_406101

theorem frances_towels_weight_in_ounces (Mary_towels Frances_towels : ℕ) (Mary_weight Frances_weight : ℝ) (total_weight : ℝ) :
  Mary_towels = 24 ∧ Mary_towels = 4 * Frances_towels ∧ total_weight = Mary_weight + Frances_weight →
  Frances_weight * 16 = 240 :=
by
  sorry

end frances_towels_weight_in_ounces_l406_406101


namespace acid_base_ratio_new_l406_406577

variable (acid_initial : ℚ) (base_initial : ℚ)
variable (acid_removed : ℚ) (base_removed : ℚ)
variable (base_added : ℚ)

-- Given conditions
axiom R1 : acid_initial = 16
axiom R2 : base_initial = 4
axiom R3 : acid_removed = (4 / 5) * 10
axiom R4 : base_removed = (1 / 5) * 10
axiom R5 : base_added = 10

-- Derived quantities
def acid_remaining := acid_initial - acid_removed
def base_remaining := base_initial - base_removed + base_added

-- New ratio of acid to base
def new_ratio := acid_remaining / base_remaining

-- Goal: Prove the new ratio is 2:3
theorem acid_base_ratio_new : 
  (acid_remaining : ℚ) / base_remaining = 2 / 3 :=
  by
  sorry

end acid_base_ratio_new_l406_406577


namespace triangle_side_lengths_relationship_l406_406953

variable {a b c : ℝ}

def is_quadratic_mean (a b c : ℝ) : Prop :=
  (2 * b^2 = a^2 + c^2)

def is_geometric_mean (a b c : ℝ) : Prop :=
  (b * a = c^2)

theorem triangle_side_lengths_relationship (a b c : ℝ) :
  (is_quadratic_mean a b c ∧ is_geometric_mean a b c) → 
  ∃ a b c, (2 * b^2 = a^2 + c^2) ∧ (b * a = c^2) :=
sorry

end triangle_side_lengths_relationship_l406_406953


namespace x_y_sum_l406_406018

theorem x_y_sum (x y : ℝ) (h1 : |x| - 2 * x + y = 1) (h2 : x - |y| + y = 8) :
  x + y = 17 ∨ x + y = 1 :=
by
  sorry

end x_y_sum_l406_406018


namespace solution_set_l406_406810

def min_func (p q : ℝ) : ℝ := if p ≤ q then p else q

noncomputable def f (x : ℝ) : ℝ :=
  min_func (3 + Real.log x / Real.log (1 / 4)) (Real.log x / Real.log 2)

theorem solution_set:
  {x : ℝ | f x < 2} = {x | 0 < x ∧ x < Real.sqrt 2} ∪ {x | 4 < x} :=
by
  sorry

end solution_set_l406_406810


namespace Zhu_Zaiyu_problem_l406_406546

theorem Zhu_Zaiyu_problem
  (f : ℕ → ℝ) 
  (q : ℝ)
  (h_geom_seq : ∀ n, f (n+1) = q * f n)
  (h_octave : f 13 = 2 * f 1) :
  (f 7) / (f 3) = 2^(1/3) :=
by
  sorry

end Zhu_Zaiyu_problem_l406_406546


namespace proof_problem_l406_406047

noncomputable def lineCartesianEquation (ρ : ℝ) (θ : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  ρ * ((Real.sqrt 3 / 2) * Real.cos θ - (1 / 2) * Real.sin θ) = 2 * Real.sqrt 3 →
  Real.sqrt 3 * x - y - 4 * Real.sqrt 3 = 0

noncomputable def curveStandardEquation (α : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  x = 2 * Real.cos α ∧ y = Real.sqrt 3 * Real.sin α →
  (x^2) / 4 + (y^2) / 3 = 1

noncomputable def maxDistanceToLine (P : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∀ α : ℝ, let P := (2 * Real.cos α, Real.sqrt 3 * Real.sin α) in 
  l (P.fst) (P.snd) →
  ∃ d : ℝ, d = (|Real.sqrt 15 * Real.cos (α + Real.arctan (1/2)) - 4 * Real.sqrt 3|) / 2 ∧
  d = (Real.sqrt 15 + 4 * Real.sqrt 3) / 2 

theorem proof_problem (ρ : ℝ) (θ : ℝ) (α : ℝ) (P : ℝ × ℝ) :
  lineCartesianEquation ρ θ P.fst P.snd ∧ 
  curveStandardEquation α P.fst P.snd ∧
  maxDistanceToLine P lineCartesianEquation :=
sorry

end proof_problem_l406_406047


namespace sharona_bought_5_more_pencils_l406_406051

variable (p : ℝ) (nj ns : ℕ)
variable h_p_positive : p > 0.01
variable h_jamar_payment : 3.25 = nj * p
variable h_sharona_payment : 4.25 = ns * p
variable h_jamar_pencils : nj ≥ 15

-- Prove that Sharona bought 5 more pencils than Jamar
theorem sharona_bought_5_more_pencils :
  ns - nj = 5 :=
  sorry

end sharona_bought_5_more_pencils_l406_406051


namespace blue_balls_to_be_removed_l406_406601

/-- An urn contains 150 balls, of which 40% are red and the rest are blue. 
    How many blue balls must be removed so that the percentage of red balls in the urn will be 60%?
    No red balls are to be removed. 
-/
theorem blue_balls_to_be_removed
  (initial_total : ℕ := 150)
  (initial_red_percentage : ℝ := 0.40)
  (desired_red_percentage : ℝ := 0.60) :
  ∃ (y : ℕ), 
    let initial_red := initial_red_percentage * initial_total,
        initial_blue := initial_total - initial_red,
        remaining_total := initial_total - y
    in 60 / remaining_total = desired_red_percentage ∧ initial_red = 60 ∧ y = 50 :=
begin
  sorry
end

end blue_balls_to_be_removed_l406_406601


namespace periodic_function_l406_406072

noncomputable def is_periodic {α : Type*} [has_add α] (f : α → α) (p : α) : Prop :=
∀ x, f (x + p) = f x

theorem periodic_function
  (f : ℝ → ℝ)
  (hf_bounded : ∀ x, |f x| ≤ 1)
  (hf_eq : ∀ x, f (x + 13 / 42) + f x = f (x + 1 / 7) + f (x + 1 / 6)) :
  ∃ T, is_periodic f T :=
sorry

end periodic_function_l406_406072


namespace max_rooks_max_rooks_4x4_max_rooks_8x8_l406_406195

theorem max_rooks (n : ℕ) : ℕ :=
  2 * (2 * n / 3)

theorem max_rooks_4x4 :
  max_rooks 4 = 4 :=
  sorry

theorem max_rooks_8x8 :
  max_rooks 8 = 10 :=
  sorry

end max_rooks_max_rooks_4x4_max_rooks_8x8_l406_406195


namespace expression_equals_one_l406_406980

noncomputable def compute_expression : ℝ :=
  (1 - 1 / Real.cos (30 * Real.pi / 180)) *
  (1 + 1 / Real.sin (60 * Real.pi / 180)) *
  (1 - 1 / Real.sin (30 * Real.pi / 180)) *
  (1 + 1 / Real.cos (60 * Real.pi / 180))

theorem expression_equals_one : compute_expression = 1 :=
by
  sorry

end expression_equals_one_l406_406980


namespace grims_groks_zeets_l406_406032

variable {T : Type}
variable (Groks Zeets Grims Snarks : Set T)

-- Given conditions as definitions in Lean 4
variable (h1 : Groks ⊆ Zeets)
variable (h2 : Grims ⊆ Zeets)
variable (h3 : Snarks ⊆ Groks)
variable (h4 : Grims ⊆ Snarks)

-- The statement to be proved
theorem grims_groks_zeets : Grims ⊆ Groks ∧ Grims ⊆ Zeets := by
  sorry

end grims_groks_zeets_l406_406032


namespace regular_polygon_perimeter_l406_406946

def exterior_angle (n : ℕ) := 360 / n

theorem regular_polygon_perimeter
  (side_length : ℕ)
  (exterior_angle_deg : ℕ)
  (polygon_perimeter : ℕ)
  (h1 : side_length = 8)
  (h2 : exterior_angle_deg = 72)
  (h3 : ∃ n : ℕ, exterior_angle n = exterior_angle_deg)
  (h4 : ∀ n : ℕ, exterior_angle n = exterior_angle_deg → polygon_perimeter = n * side_length) :
  polygon_perimeter = 40 :=
sorry

end regular_polygon_perimeter_l406_406946


namespace two_digit_numbers_l406_406643

def is_digit (n : ℕ) : Prop := n ≤ 9

theorem two_digit_numbers (a b : ℕ) (h1 : is_digit a) (h2 : is_digit b) 
  (h3 : a ≠ b) (h4 : (a + b) = 11) : 
  (∃ n m : ℕ, (n = 10 * a + b) ∧ (m = 10 * b + a) ∧ (∃ k : ℕ, (10 * a + b)^2 - (10 * b + a)^2 = k^2)) := 
sorry

end two_digit_numbers_l406_406643


namespace find_q_l406_406149

def Q (x : ℝ) (p q r : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem find_q (p q r : ℝ) (h1 : -p = 2 * (-r)) (h2 : -p = 1 + p + q + r) (hy_intercept : r = 5) : q = -24 :=
by
  sorry

end find_q_l406_406149


namespace monotonicity_f_when_a_is_1_range_of_a_for_g_l406_406811

noncomputable def f (a x : ℝ) : ℝ := a * x * Real.log x + 1 / x
noncomputable def g (a x : ℝ) : ℝ := f a x - a * x

-- Problem (I): The monotonicity of f(x) when a = 1
theorem monotonicity_f_when_a_is_1 :
  ∀ x > 0, (f 1 x) = x * Real.log x + 1 / x ∧ 
            ((∀ y, 0 < y < 1 → (Real.log y + 1 - 1 / y^2) < 0) ∧ 
             (∀ y, y > 1 → (Real.log y + 1 - 1 / y^2) > 0)) :=
sorry

-- Problem (II): Range of values for a such that g(x) ≥ 0 always holds
theorem range_of_a_for_g :
  ∀ a > 0, (∀ x > 0, g a x ≥ 0) → a ≤ 2 / Real.exp 1 :=
sorry

end monotonicity_f_when_a_is_1_range_of_a_for_g_l406_406811


namespace seven_distinct_integers_exist_pair_l406_406827

theorem seven_distinct_integers_exist_pair (a : Fin 7 → ℕ) (h_distinct : Function.Injective a)
  (h_bound : ∀ i, 1 ≤ a i ∧ a i ≤ 126) :
  ∃ i j : Fin 7, i ≠ j ∧ (1 / 2 : ℚ) ≤ (a i : ℚ) / a j ∧ (a i : ℚ) / a j ≤ 2 := sorry

end seven_distinct_integers_exist_pair_l406_406827


namespace high_school_elite_games_l406_406133

theorem high_school_elite_games :
  ∃ G : ℕ, G = 124 ∧
    (let teams := 8 in
     let intra_league_games := (teams * (teams - 1) / 2) * 3 in
     let non_conference_games := teams * 5 in
     G = intra_league_games + non_conference_games) :=
by
  -- This is where the proof would be placed
  sorry

end high_school_elite_games_l406_406133


namespace total_swordfish_caught_correct_l406_406465

-- Define the parameters and the catch per trip
def shelly_catch_per_trip : ℕ := 5 - 2
def sam_catch_per_trip (shelly_catch : ℕ) : ℕ := shelly_catch - 1
def total_catch_per_trip (shelly_catch sam_catch : ℕ) : ℕ := shelly_catch + sam_catch
def total_trips : ℕ := 5

-- Define the total number of swordfish caught over the trips
def total_swordfish_caught : ℕ :=
  let shelly_catch := shelly_catch_per_trip
  let sam_catch := sam_catch_per_trip shelly_catch
  let total_catch := total_catch_per_trip shelly_catch sam_catch
  total_catch * total_trips

-- The theorem states that the total swordfish caught is 25
theorem total_swordfish_caught_correct : total_swordfish_caught = 25 := by
  sorry

end total_swordfish_caught_correct_l406_406465


namespace part1_part2_l406_406805

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + x - a

-- Part 1
theorem part1 (a x : ℝ) (h1 : |a| ≤ 1) (h2 : |x| ≤ 1) : |f a x| ≤ 5/4 :=
by
  sorry

-- Part 2
theorem part2 (a : ℝ) (h : ∃ x ∈ Set.Icc (-1:ℝ) (1:ℝ), f a x = 17/8) : a = -2 :=
by
  sorry

end part1_part2_l406_406805


namespace eccentricity_of_hyperbola_l406_406665

noncomputable def hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (h : (4:ℝ) * a^2 = c^2) : ℝ :=
  c / a

theorem eccentricity_of_hyperbola (a b c : ℝ) (ha : a > 0) (hb : b > 0) (h : (4:ℝ) * a^2 = c^2) :
  hyperbola_eccentricity a b c ha hb h = 2 :=
by
  sorry


end eccentricity_of_hyperbola_l406_406665


namespace fraction_of_raisins_l406_406911

-- Define the cost of a single pound of raisins
variables (R : ℝ) -- R represents the cost of one pound of raisins

-- Conditions
def mixed_raisins := 5 -- Chris mixed 5 pounds of raisins
def mixed_nuts := 4 -- with 4 pounds of nuts
def nuts_cost_ratio := 3 -- A pound of nuts costs 3 times as much as a pound of raisins

-- Statement to prove
theorem fraction_of_raisins
  (R_pos : R > 0) : (5 * R) / ((5 * R) + (4 * (3 * R))) = 5 / 17 :=
by
  -- The proof is omitted here.
  sorry

end fraction_of_raisins_l406_406911


namespace not_all_same_probability_l406_406526

-- Definition of the total number of outcomes when rolling 5 8-sided dice
def total_outcomes : ℕ := 8^5

-- Definition of the number of outcomes where all five dice show the same number
def same_number_outcomes : ℕ := 8

-- Definition to find the probability that not all 5 dice show the same number
def probability_not_all_same : ℚ := 1 - (same_number_outcomes / total_outcomes)

-- Statement of the main theorem
theorem not_all_same_probability : probability_not_all_same = (4095 : ℚ) / 4096 :=
by
  rw [probability_not_all_same, same_number_outcomes, total_outcomes]
  -- Simplification steps would go here, but we use sorry to skip the proof
  sorry

end not_all_same_probability_l406_406526


namespace ratio_of_areas_of_squares_l406_406262

theorem ratio_of_areas_of_squares (r : ℝ) (h : 0 < r) :
  let A_inscribed := (r * Real.sqrt 2) ^ 2,
      A_circumscribed := (2 * r) ^ 2
  in A_circumscribed / A_inscribed = 2 := by
    let A_inscribed := (r * Real.sqrt 2) ^ 2
    let A_circumscribed := (2 * r) ^ 2
    calc
      A_circumscribed / A_inscribed
          = (2 * r) ^ 2 / (r * Real.sqrt 2) ^ 2 : by rfl
      ... = (4 * r^2) / (2 * r^2) : by sorry
      ... = 2 : by sorry

end ratio_of_areas_of_squares_l406_406262


namespace sequence_correctness_l406_406795

def sequence_a (n : ℕ) : ℤ :=
  if n = 1 then -2
  else -(2^(n - 1))

def partial_sum_S (n : ℕ) : ℤ := -2^n

theorem sequence_correctness (n : ℕ) (h : n ≥ 1) :
  (sequence_a 1 = -2) ∧ (∀ n ≥ 2, sequence_a (n + 1) = partial_sum_S n) ∧
  (sequence_a n = -(2^(n - 1))) ∧ (partial_sum_S n = -2^n) :=
by
  sorry

end sequence_correctness_l406_406795


namespace leap_day_2024_is_sunday_l406_406418

-- Define the concept of a day of the week as an enumeration
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

-- Define the number of days in a regular year and a leap year
def days_in_year (leap : Bool) : ℕ := if leap then 366 else 365

-- Calculate total days from the year 2000 to the year 2024, inclusive
def total_days_between (start end : ℕ) : ℕ :=
  (List.range' (start + 1) (end - start)).sum (λ y, days_in_year ((y % 4 = 0) && (y % 100 ≠ 0 || y % 400 = 0)))

-- Given condition: February 29, 2000, was a Sunday
def day_of_week_2000_02_29 : DayOfWeek := DayOfWeek.Sunday

-- Define the function to determine the day of the week after a number 
-- of days starting from a given day.
def day_of_week_after (start_day : DayOfWeek) (days : ℕ) : DayOfWeek :=
  let days_mod_week := days % 7
  match (start_day, days_mod_week) with
  | (DayOfWeek.Sunday, 0) => DayOfWeek.Sunday
  | (DayOfWeek.Sunday, 1) => DayOfWeek.Monday
  | (DayOfWeek.Sunday, 2) => DayOfWeek.Tuesday
  | (DayOfWeek.Sunday, 3) => DayOfWeek.Wednesday
  | (DayOfWeek.Sunday, 4) => DayOfWeek.Thursday
  | (DayOfWeek.Sunday, 5) => DayOfWeek.Friday
  | (DayOfWeek.Sunday, 6) => DayOfWeek.Saturday
  | (DayOfWeek.Monday, 0) => DayOfWeek.Monday
  | (DayOfWeek.Monday, 1) => DayOfWeek.Tuesday
  | (DayOfWeek.Monday, 2) => DayOfWeek.Wednesday
  | (DayOfWeek.Monday, 3) => DayOfWeek.Thursday
  | (DayOfWeek.Monday, 4) => DayOfWeek.Friday
  | (DayOfWeek.Monday, 5) => DayOfWeek.Saturday
  | (DayOfWeek.Monday, 6) => DayOfWeek.Sunday
  | (DayOfWeek.Tuesday, 0) => DayOfWeek.Tuesday
  | (DayOfWeek.Tuesday, 1) => DayOfWeek.Wednesday
  | (DayOfWeek.Tuesday, 2) => DayOfWeek.Thursday
  | (DayOfWeek.Tuesday, 3) => DayOfWeek.Friday
  | (DayOfWeek.Tuesday, 4) => DayOfWeek.Saturday
  | (DayOfWeek.Tuesday, 5) => DayOfWeek.Sunday
  | (DayOfWeek.Tuesday, 6) => DayOfWeek.Monday
  | (DayOfWeek.Wednesday, 0) => DayOfWeek.Wednesday
  | (DayOfWeek.Wednesday, 1) => DayOfWeek.Thursday
  | (DayOfWeek.Wednesday, 2) => DayOfWeek.Friday
  | (DayOfWeek.Wednesday, 3) => DayOfWeek.Saturday
  | (DayOfWeek.Wednesday, 4) => DayOfWeek.Sunday
  | (DayOfWeek.Wednesday, 5) => DayOfWeek.Monday
  | (DayOfWeek.Wednesday, 6) => DayOfWeek.Tuesday
  | (DayOfWeek.Thursday, 0) => DayOfWeek.Thursday
  | (DayOfWeek.Thursday, 1) => DayOfWeek.Friday
  | (DayOfWeek.Thursday, 2) => DayOfWeek.Saturday
  | (DayOfWeek.Thursday, 3) => DayOfWeek.Sunday
  | (DayOfWeek.Thursday, 4) => DayOfWeek.Monday
  | (DayOfWeek.Thursday, 5) => DayOfWeek.Tuesday
  | (DayOfWeek.Thursday, 6) => DayOfWeek.Wednesday
  | (DayOfWeek.Friday, 0) => DayOfWeek.Friday
  | (DayOfWeek.Friday, 1) => DayOfWeek.Saturday
  | (DayOfWeek.Friday, 2) => DayOfWeek.Sunday
  | (DayOfWeek.Friday, 3) => DayOfWeek.Monday
  | (DayOfWeek.Friday, 4) => DayOfWeek.Tuesday
  | (DayOfWeek.Friday, 5) => DayOfWeek.Wednesday
  | (DayOfWeek.Friday, 6) => DayOfWeek.Thursday
  | (DayOfWeek.Saturday, 0) => DayOfWeek.Saturday
  | (DayOfWeek.Saturday, 1) => DayOfWeek.Sunday
  | (DayOfWeek.Saturday, 2) => DayOfWeek.Monday
  | (DayOfWeek.Saturday, 3) => DayOfWeek.Tuesday
  | (DayOfWeek.Saturday, 4) => DayOfWeek.Wednesday
  | (DayOfWeek.Saturday, 5) => DayOfWeek.Thursday
  | (DayOfWeek.Saturday, 6) => DayOfWeek.Friday
  | _ => sorry

-- Assert the final proof statement
theorem leap_day_2024_is_sunday :
  let total_days := total_days_between 2000 2024
  day_of_week_after day_of_week_2000_02_29 total_days = DayOfWeek.Sunday := sorry

end leap_day_2024_is_sunday_l406_406418


namespace train_cross_pole_time_l406_406553

-- Define the length of the train in meters
def train_length : ℕ := 175

-- Define the speed of the train in km/hr
def train_speed_kmhr : ℕ := 180

-- Conversion factor from kilometers per hour to meters per second 
def kmhr_to_mps (speed_kmhr : ℕ) : ℝ :=
  (speed_kmhr : ℝ) * 1000 / 3600

-- Define the speed of the train in m/s
def train_speed_mps : ℝ := kmhr_to_mps train_speed_kmhr

-- Define the time it takes for the train to cross an electric pole
def time_to_cross (length : ℕ) (speed : ℝ) : ℝ :=
  length / speed

-- Prove that the time is 3.5 seconds
theorem train_cross_pole_time :
  time_to_cross train_length train_speed_mps = 3.5 := by
  -- the proof is not required, adding sorry for now
  sorry

end train_cross_pole_time_l406_406553


namespace seating_arrangements_l406_406103
open Nat

theorem seating_arrangements (Mr_Smith Mrs_Smith : Type) (children : Set Type) (h : children.card = 3) :
  {arr | (arr \ {Mr_Smith, Mrs_Smith}).card = 3 ∧ 
          Mr_Smith ∈ arr ∧ 
          Mrs_Smith ∈ arr ∧ 
          (arr ∩ {Mr_Smith, Mrs_Smith}).card = 2}.card = 48 := 
sorry

end seating_arrangements_l406_406103


namespace range_of_a_l406_406368

variable {a : ℝ} (h : MonotoneOn (λ x, log 2 (a * x - 1)) (Set.Ioo 1 2))

theorem range_of_a : ∀ a, MonotoneOn (λ x, log 2 (a * x - 1)) (Set.Ioo 1 2) → a ∈ Set.Ici 1 :=
by
  sorry

end range_of_a_l406_406368


namespace problem1_problem2_l406_406284

noncomputable def expr1 : ℚ := (((9 : ℚ) / 4)^((1 : ℚ) / 2) - (-9.6 : ℝ)^0 - ((27 : ℚ) / 8)^(-(2 : ℚ) / 3) + ((3 : ℚ) / 2)^(-2))

noncomputable def expr2 : ℚ := (-(Real.log 2 / Real.log 25) * (Real.log 5 / Real.log 4) - (Real.log 3 / Real.log (1 / 3)) - 2 + (5 ^ (Real.log 2 / Real.log 5)))

theorem problem1 : expr1 = 1 / 2 := by sorry

theorem problem2 : expr2 = 3 / 4 := by sorry

#eval expr1 -- This should evaluate to 1 / 2
#eval expr2 -- This should evaluate to 3 / 4

end problem1_problem2_l406_406284


namespace first_half_speed_l406_406954

noncomputable def speed_first_half : ℝ := 21

theorem first_half_speed (total_distance first_half_distance second_half_distance second_half_speed total_time : ℝ)
  (h1 : total_distance = 224)
  (h2 : first_half_distance = total_distance / 2)
  (h3 : second_half_distance = total_distance / 2)
  (h4 : second_half_speed = 24)
  (h5 : total_time = 10)
  (h6 : total_time = first_half_distance / speed_first_half + second_half_distance / second_half_speed) :
  speed_first_half = 21 :=
sorry

end first_half_speed_l406_406954


namespace smallest_period_and_monotonic_interval_cos_2x_value_l406_406713

noncomputable def vector_dot_prod (a b : ℝ × ℝ) : ℝ := (a.1 * b.1 + a.2 * b.2 : ℝ)

def a (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin x, cos x)
def b (x : ℝ) : ℝ × ℝ := (cos x, -cos x)
def f (x : ℝ) : ℝ := vector_dot_prod (a x) (b x)

theorem smallest_period_and_monotonic_interval :
  (∀ x, f x = sin (2 * x - π / 6) - 1 / 2) ∧
  (∃ (k : ℤ), ∃ (x_min x_max : ℝ), 
    ∀ x ∈ set.Icc (k * π - π / 6) (k * π + π / 3), 
      f x ≥ f (x_min) ∧ f x ≤ f (x_max)) ∧
  (∀ x, f (x + π) = f x) :=
begin
  sorry
end

theorem cos_2x_value (x : ℝ) (h1 : x ∈ set.Ioc (7 * π / 12) (5 * π / 6)) 
  (h2 : f x = -5 / 4) : 
  cos (2 * x) = (3 - sqrt 21) / 8 :=
begin
  sorry
end

end smallest_period_and_monotonic_interval_cos_2x_value_l406_406713


namespace simplify_expression_zero_l406_406080

noncomputable def simplify_expression (a b c d : ℝ) : ℝ :=
  1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2)

theorem simplify_expression_zero (a b c d : ℝ) (h : a + b + c = d)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  simplify_expression a b c d = 0 :=
by
  sorry

end simplify_expression_zero_l406_406080


namespace kelly_initial_sony_games_l406_406787

def nintendo_games : ℕ := 46
def sony_games_given_away : ℕ := 101
def sony_games_left : ℕ := 31

theorem kelly_initial_sony_games :
  sony_games_given_away + sony_games_left = 132 :=
by
  sorry

end kelly_initial_sony_games_l406_406787


namespace jens_son_age_l406_406778

theorem jens_son_age
  (J : ℕ)
  (S : ℕ)
  (h1 : J = 41)
  (h2 : J = 3 * S - 7) :
  S = 16 :=
by
  sorry

end jens_son_age_l406_406778


namespace opposite_of_neg_abs_is_positive_two_l406_406864

theorem opposite_of_neg_abs_is_positive_two : -(abs (-2)) = -2 :=
by sorry

end opposite_of_neg_abs_is_positive_two_l406_406864


namespace optionD_is_not_linear_system_l406_406209

-- Define the equations for each option
def eqA1 (x y : ℝ) : Prop := 3 * x + 2 * y = 10
def eqA2 (x y : ℝ) : Prop := 2 * x - 3 * y = 5

def eqB1 (x y : ℝ) : Prop := 3 * x + 5 * y = 1
def eqB2 (x y : ℝ) : Prop := 2 * x - y = 4

def eqC1 (x y : ℝ) : Prop := x + 5 * y = 1
def eqC2 (x y : ℝ) : Prop := x - 5 * y = 2

def eqD1 (x y : ℝ) : Prop := x - y = 1
def eqD2 (x y : ℝ) : Prop := y + 1 / x = 3

-- Define the property of a linear equation
def is_linear (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, eq x y → a * x + b * y = c

-- State the theorem
theorem optionD_is_not_linear_system : ¬ (is_linear eqD1 ∧ is_linear eqD2) :=
by
  sorry

end optionD_is_not_linear_system_l406_406209


namespace draw_cards_to_ensure_even_product_l406_406634

noncomputable def isEven (n : ℕ) : Prop :=
  ∃ k, n = 2 * k

theorem draw_cards_to_ensure_even_product :
  ∀ drawn_cards : Finset ℕ,
    (∀ x ∈ drawn_cards, 1 ≤ x ∧ x ≤ 18) →
    (Finset.card drawn_cards = 10) →
    ∃ x ∈ drawn_cards, isEven x := 
by
  intros drawn_cards range h_card
  sorry

end draw_cards_to_ensure_even_product_l406_406634


namespace minimize_expression_l406_406525

theorem minimize_expression (x y : ℝ) (k : ℝ) (h : k = -1) : (xy + k)^2 + (x - y)^2 ≥ 0 ∧ (∀ x y : ℝ, (xy + k)^2 + (x - y)^2 = 0 ↔ k = -1) := 
by {
  sorry
}

end minimize_expression_l406_406525


namespace find_a3_plus_a9_l406_406691

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (n : ℕ)

-- Conditions stating sequence is arithmetic and a₁ + a₆ + a₁₁ = 3
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def a_1_6_11_sum (a : ℕ → ℝ) : Prop :=
  a 1 + a 6 + a 11 = 3

theorem find_a3_plus_a9 
  (h_arith : is_arithmetic_sequence a d)
  (h_sum : a_1_6_11_sum a) : 
  a 3 + a 9 = 2 := 
sorry

end find_a3_plus_a9_l406_406691


namespace circles_are_tangent_l406_406269

variable {A B C : Type} [Point A] [Point B] [Point C]

-- The lengths of the sides opposite to the points A, B, and C
def a : ℝ := distance B C
def b : ℝ := distance A C
def c : ℝ := distance A B

-- Semi-perimeter of the triangle
def s : ℝ := (a + b + c) / 2

-- Variables for the radii of circles centered at A, B, and C respectively
variables (r1 r2 r3 : ℝ)

-- The conditions for pairwise tangency
def tangency_conditions (r1 r2 r3 s : ℝ) : Prop :=
  (r1 + r2 + r3 = s) ∧
  (r1 = s - a ∧ r2 = s - b ∧ r3 = s - c) ∨
  (r1 = s ∧ r2 = s - c ∧ r3 = s - b) ∨
  (r1 = s - c ∧ r2 = s ∧ r3 = s - a) ∨
  (r1 = s - b ∧ r2 = s - a ∧ r3 = s)

theorem circles_are_tangent :
  ∃ (r1 r2 r3 : ℝ), tangency_conditions r1 r2 r3 s :=
begin
  sorry
end

end circles_are_tangent_l406_406269


namespace shooter_probability_l406_406867

theorem shooter_probability (P_hit_10_ring : ℝ) (h : P_hit_10_ring = 0.22) : 
  ∃ P_less_than_10, P_less_than_10 = 1 - P_hit_10_ring ∧ P_less_than_10 = 0.78 :=
by {
  use 1 - P_hit_10_ring,
  split,
  { refl, },
  { rw h, norm_num, }
}

end shooter_probability_l406_406867


namespace complex_transformation_l406_406603

open Complex

theorem complex_transformation :
  let z := -1 + (7 : ℂ) * I
  let rotation := (1 / 2 + (Real.sqrt 3) / 2 * I)
  let dilation := 2
  (z * rotation * dilation = -22 - ((Real.sqrt 3) - 7) * I) :=
by
  sorry

end complex_transformation_l406_406603


namespace dice_probability_not_all_same_l406_406533

theorem dice_probability_not_all_same : 
  let total_outcomes := (8 : ℕ)^5 in
  let same_number_outcomes := 8 in
  let probability_all_same := (same_number_outcomes : ℚ) / total_outcomes in
  let probability_not_all_same := 1 - probability_all_same in
  probability_not_all_same = 4095 / 4096 :=
by
  sorry

end dice_probability_not_all_same_l406_406533


namespace three_digit_number_is_657_l406_406878

theorem three_digit_number_is_657 :
  ∃ (a b c : ℕ), (100 * a + 10 * b + c = 657) ∧ (a + b + c = 18) ∧ (a = b + 1) ∧ (c = b + 2) :=
by
  sorry

end three_digit_number_is_657_l406_406878


namespace relationship_among_abc_l406_406353

noncomputable def a : ℝ := 4^0.3
noncomputable def b : ℝ := (1/2)^(-0.9)
noncomputable def c : ℝ := 2 * Real.log 2 / Real.log 6

theorem relationship_among_abc : c < a ∧ a < b := by
  sorry

end relationship_among_abc_l406_406353


namespace fraction_of_income_from_tips_l406_406268

variable (S T I : ℝ)

-- Conditions
def tips_are_fraction_of_salary : Prop := T = (3/4) * S
def total_income_is_sum_of_salary_and_tips : Prop := I = S + T

-- Statement to prove
theorem fraction_of_income_from_tips (h1 : tips_are_fraction_of_salary S T) (h2 : total_income_is_sum_of_salary_and_tips S T I) :
  T / I = 3 / 7 := by
  sorry

end fraction_of_income_from_tips_l406_406268


namespace fewer_people_third_bus_l406_406500

noncomputable def people_first_bus : Nat := 12
noncomputable def people_second_bus : Nat := 2 * people_first_bus
noncomputable def people_fourth_bus : Nat := people_first_bus + 9
noncomputable def total_people : Nat := 75
noncomputable def people_other_buses : Nat := people_first_bus + people_second_bus + people_fourth_bus
noncomputable def people_third_bus : Nat := total_people - people_other_buses

theorem fewer_people_third_bus :
  people_second_bus - people_third_bus = 6 :=
by
  sorry

end fewer_people_third_bus_l406_406500


namespace exists_equal_degree_common_neighbour_l406_406339

noncomputable def graph_problem (G : Type) [Fintype G] :=
  ∃ (vertices : Finset G) (deg : G → ℕ),
    (vertices.card = 99) ∧
    (∀ v ∈ vertices, deg v ∈ {81, 82, ..., 90}) ∧
    (∃ S : Finset G, S.card = 10 ∧ ∃ n ∈ vertices, ∀ v ∈ S, deg v = deg n ∧ ∃ u ∈ vertices, u ≠ n ∧ ∀ v ∈ S, v ≠ u ∧ (v, u) ∈ edges)

-- Prove the statement using Lean proofs
theorem exists_equal_degree_common_neighbour {G : Type} [Fintype G]
  (vertices : Finset G) (deg : G → ℕ) (edges : Finset (G × G)):
  (vertices.card = 99) ∧
  (∀ v ∈ vertices, deg v ∈ {81, 82, ..., 90}) →
  (∃ S : Finset G, S.card = 10 ∧ ∃ n ∈ vertices, ∀ v ∈ S, deg v = deg n ∧ ∃ u ∈ vertices, u ≠ n ∧ ∀ v ∈ S, v ≠ u ∧ (v, u) ∈ edges) := sorry

end exists_equal_degree_common_neighbour_l406_406339


namespace computation_of_fraction_l406_406802

theorem computation_of_fraction (x y z : ℝ) (h : x + y + z = 3) :
    (xy + yz + zx) / (x^2 + y^2 + z^2) = (xy + yz + zx) / (9 - 2 (xy + yz + zx)) :=
by
  sorry

end computation_of_fraction_l406_406802


namespace future_age_relation_l406_406876

-- Conditions
def son_present_age : ℕ := 8
def father_present_age : ℕ := 4 * son_present_age

-- Theorem statement
theorem future_age_relation : ∃ x : ℕ, 32 + x = 3 * (8 + x) ↔ x = 4 :=
by {
  sorry
}

end future_age_relation_l406_406876


namespace sin_cos_value_sin_minus_cos_value_tan_value_l406_406669

variable (x : ℝ)

theorem sin_cos_value 
  (h1 : - (Real.pi / 2) < x) 
  (h2 : x < 0) 
  (h3 : Real.sin x + Real.cos x = 1 / 5) : 
  Real.sin x * Real.cos x = - 12 / 25 := 
sorry

theorem sin_minus_cos_value 
  (h1 : - (Real.pi / 2) < x) 
  (h2 : x < 0) 
  (h3 : Real.sin x + Real.cos x = 1 / 5) : 
  Real.sin x - Real.cos x = - 7 / 5 := 
sorry

theorem tan_value 
  (h1 : - (Real.pi / 2) < x) 
  (h2 : x < 0) 
  (h3 : Real.sin x + Real.cos x = 1 / 5) : 
  Real.tan x = - 3 / 4 := 
sorry

end sin_cos_value_sin_minus_cos_value_tan_value_l406_406669


namespace percent_red_prob_l406_406784

-- Define the conditions
def initial_red := 2
def initial_blue := 4
def additional_red := 2
def additional_blue := 2
def total_balloons := initial_red + initial_blue + additional_red + additional_blue
def total_red := initial_red + additional_red

-- State the theorem
theorem percent_red_prob : (total_red.toFloat / total_balloons.toFloat) * 100 = 40 :=
by
  sorry

end percent_red_prob_l406_406784


namespace angle_sum_in_triangle_l406_406745

theorem angle_sum_in_triangle (A B C : ℝ) (h₁ : A + B = 90) (h₂ : A + B + C = 180) : C = 90 := by
  sorry

end angle_sum_in_triangle_l406_406745


namespace complex_problem_l406_406079

noncomputable def complex_norm (z : ℂ) : ℝ := complex.abs z

theorem complex_problem (a b c : ℂ) 
  (ha : complex_norm a = 2) 
  (hb : complex_norm b = 2) 
  (hc : complex_norm c = 2) 
  (h : (a^3 / (b * c)) + (b^3 / (a * c)) + (c^3 / (a * b)) = 0) :
  complex_norm (a + b + c) = 6 + 2 * real.sqrt 6 ∨ complex_norm (a + b + c) = 6 - 2 * real.sqrt 6 := 
sorry

end complex_problem_l406_406079


namespace closest_fraction_to_medals_won_l406_406607

theorem closest_fraction_to_medals_won :
  let gamma_fraction := (13:ℚ) / 80
  let fraction_1_4 := (1:ℚ) / 4
  let fraction_1_5 := (1:ℚ) / 5
  let fraction_1_6 := (1:ℚ) / 6
  let fraction_1_7 := (1:ℚ) / 7
  let fraction_1_8 := (1:ℚ) / 8
  abs (gamma_fraction - fraction_1_6) <
    abs (gamma_fraction - fraction_1_4) ∧
  abs (gamma_fraction - fraction_1_6) <
    abs (gamma_fraction - fraction_1_5) ∧
  abs (gamma_fraction - fraction_1_6) <
    abs (gamma_fraction - fraction_1_7) ∧
  abs (gamma_fraction - fraction_1_6) <
    abs (gamma_fraction - fraction_1_8) := by
  sorry

end closest_fraction_to_medals_won_l406_406607


namespace program_arrangement_count_l406_406756

theorem program_arrangement_count :
  let num_singing_programs := 6
  let num_dance_programs := 4
  let num_singing_permutations := Nat.factorial num_singing_programs
  let num_positions := num_singing_programs + 1
  let num_dance_choices := Nat.perm num_positions num_dance_programs
  let total_arrangements := num_singing_permutations * num_dance_choices
  total_arrangements = 604800 :=
by
  begin
    sorry
  end

end program_arrangement_count_l406_406756


namespace rainfall_in_may_l406_406877

-- Define the rainfalls for the months
def march_rain : ℝ := 3.79
def april_rain : ℝ := 4.5
def june_rain : ℝ := 3.09
def july_rain : ℝ := 4.67

-- Define the average rainfall over five months
def avg_rain : ℝ := 4

-- Define total rainfall calculation
def calc_total_rain (may_rain : ℝ) : ℝ :=
  march_rain + april_rain + may_rain + june_rain + july_rain

-- Problem statement: proving the rainfall in May
theorem rainfall_in_may : ∃ (may_rain : ℝ), calc_total_rain may_rain = avg_rain * 5 ∧ may_rain = 3.95 :=
sorry

end rainfall_in_may_l406_406877


namespace jaron_chocolate_bunnies_sold_l406_406052

example (points_needed : ℕ) (chocolate_bunny_points : ℕ) (snickers_points : ℕ) (snickers_count : ℕ) : ℕ :=
  let total_snickers_points := snickers_count * snickers_points in
  let points_from_bunnies := points_needed - total_snickers_points in
  let num_chocolate_bunnies := points_from_bunnies / chocolate_bunny_points in
  num_chocolate_bunnies

theorem jaron_chocolate_bunnies_sold (points_needed chocolate_bunny_points snickers_points snickers_count : ℕ) 
  (h1 : points_needed = 2000) 
  (h2 : chocolate_bunny_points = 100) 
  (h3 : snickers_points = 25) 
  (h4 : snickers_count = 48) : points_needed = 2000 → chocolate_bunny_points = 100 → snickers_points = 25 → snickers_count = 48 → example points_needed chocolate_bunny_points snickers_points snickers_count = 8 :=
by
  intros
  unfold example
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end jaron_chocolate_bunnies_sold_l406_406052


namespace not_polynomial_option_B_l406_406270

-- Definitions
def is_polynomial (expr : String) : Prop :=
  -- Assuming we have a function that determines if a given string expression is a polynomial.
  sorry

def option_A : String := "m+n"
def option_B : String := "x=1"
def option_C : String := "xy"
def option_D : String := "0"

-- Problem Statement
theorem not_polynomial_option_B : ¬ is_polynomial option_B := 
sorry

end not_polynomial_option_B_l406_406270


namespace yard_length_l406_406397

theorem yard_length
  (trees : ℕ) (gaps : ℕ) (distance_between_trees : ℕ) :
  trees = 26 → 
  gaps = trees - 1 → 
  distance_between_trees = 14 → 
  length_of_yard = gaps * distance_between_trees → 
  length_of_yard = 350 :=
by
  intros h_trees h_gaps h_distance h_length
  sorry

end yard_length_l406_406397


namespace euler_totient_prime_factors_l406_406806

theorem euler_totient_prime_factors (n : ℕ) (s : ℕ) (p : ℕ → ℕ) (α : ℕ → ℕ)
  (h₀ : n = ∏ i in finset.range s, (p i) ^ (α i))
  (h₁ : ∀ m k, coprime m k → ∀ φ : ℕ → ℕ, (φ (m * k) = φ m * φ k))
  (h₂ : ∀ i, ∀ φ : ℕ → ℕ, (φ ((p i) ^ (α i)) = (p i) ^ (α i) * (1 - (1 / (p i))))) :
  ∃ φ : ℕ → ℕ, (φ n = n * (∏ i in finset.range s, (1 - (1 / (p i))))) :=
by
  sorry

end euler_totient_prime_factors_l406_406806


namespace unoccupied_volume_proof_l406_406780

-- Definitions based on conditions
def tank_length : ℕ := 12
def tank_width : ℕ := 10
def tank_height : ℕ := 8
def tank_volume : ℕ := tank_length * tank_width * tank_height

def oil_fill_ratio : ℚ := 2 / 3
def ice_cube_volume : ℕ := 1
def number_of_ice_cubes : ℕ := 15

-- Volume calculations
def oil_volume : ℚ := oil_fill_ratio * tank_volume
def total_ice_volume : ℚ := number_of_ice_cubes * ice_cube_volume
def occupied_volume : ℚ := oil_volume + total_ice_volume

-- The final question to be proved
theorem unoccupied_volume_proof : tank_volume - occupied_volume = 305 := by
  sorry

end unoccupied_volume_proof_l406_406780


namespace math_lovers_l406_406961

/-- The proof problem: 
Given 1256 students in total and the difference of 408 between students who like math and others,
prove that the number of students who like math is 424, given that students who like math are fewer than 500.
--/
theorem math_lovers (M O : ℕ) (h1 : M + O = 1256) (h2: O - M = 408) (h3 : M < 500) : M = 424 :=
by
  sorry

end math_lovers_l406_406961


namespace false_proposition_l406_406388

variable (a : ℝ) (h_a : a ≠ -1)
variable (ω : ℝ) (h_ω : ω > 0)

def line1 (x : ℝ) : ℝ := (1 / a) * x + a
def line2 (x : ℝ) : ℝ := -(x : ℝ)
def are_parallel (f g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = g x

def p : Prop := ¬are_parallel line1 line2
def q : Prop := ω > 4

theorem false_proposition : ¬ (p ∧ q) := 
  by {
    sorry
  }

end false_proposition_l406_406388


namespace sum_of_areas_of_circles_l406_406174

theorem sum_of_areas_of_circles 
  (r s t : ℝ) 
  (h1 : r + s = 6) 
  (h2 : r + t = 8) 
  (h3 : s + t = 10) 
  : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by 
  sorry

end sum_of_areas_of_circles_l406_406174


namespace nth_equation_l406_406105

theorem nth_equation (n : ℕ) : 2 * n * (2 * n + 2) + 1 = (2 * n + 1)^2 :=
by
  sorry

end nth_equation_l406_406105


namespace solution_positive_then_opposite_signs_l406_406025

theorem solution_positive_then_opposite_signs
  (a b : ℝ) (h : a ≠ 0) (x : ℝ) (hx : ax + b = 0) (x_pos : x > 0) :
  (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0) :=
by
  sorry

end solution_positive_then_opposite_signs_l406_406025


namespace Democrats_in_House_l406_406506

-- Let D be the number of Democrats.
-- Let R be the number of Republicans.
-- Given conditions.

def Democrats (D R : ℕ) : Prop := 
  D + R = 434 ∧ R = D + 30

theorem Democrats_in_House : ∃ D, ∃ R, Democrats D R ∧ D = 202 :=
by
  -- skip the proof
  sorry

end Democrats_in_House_l406_406506


namespace min_abs_value_of_f_l406_406335

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sin x) + (Real.cos x) + (Real.tan x) 
  + (Real.cot x) + (Real.sec x) + (Real.csc x)

theorem min_abs_value_of_f : ∀ x : ℝ, abs (f x) ≥  2 * Real.sqrt 2 - 1 := 
by 
  sorry

end min_abs_value_of_f_l406_406335


namespace initial_orange_balloons_l406_406120

-- Definitions
variable (x : ℕ)
variable (h1 : x - 2 = 7)

-- Theorem to prove
theorem initial_orange_balloons (h1 : x - 2 = 7) : x = 9 :=
sorry

end initial_orange_balloons_l406_406120


namespace find_correct_answer_l406_406249

noncomputable def correct_answer (x : ℕ) : ℕ :=
  if h : 987 * x == 559981 then
    let a := 559989 in a
  else
    sorry

theorem find_correct_answer (x : ℕ) (hx : 987 * x = 559981) : 987 * x = 559989 :=
  by
    have correct_digits : ∀ d, (String.toList (559981.repr)).filter (λ c, c = '9') = ['9', '9'] := sorry
    have correct_digits_replacement : ∀ a b, (String.toList (559989.repr)).filter (λ c, c = '9') = ['9', '9'] := sorry
    rw hx at correct_digits
    rw correct_digits_replacement
    refl

end find_correct_answer_l406_406249


namespace ratio_of_football_to_hockey_l406_406930

variables (B F H s : ℕ)

-- Definitions from conditions
def condition1 : Prop := B = F - 50
def condition2 : Prop := F = s * H
def condition3 : Prop := H = 200
def condition4 : Prop := B + F + H = 1750

-- Proof statement
theorem ratio_of_football_to_hockey (B F H s : ℕ) 
  (h1 : condition1 B F)
  (h2 : condition2 F s H)
  (h3 : condition3 H)
  (h4 : condition4 B F H) : F / H = 4 :=
sorry

end ratio_of_football_to_hockey_l406_406930


namespace plane_equivalent_l406_406616

def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 2*s - 3*t, 1 + s, 4 - 3*s + t)

def plane_equation (x y z : ℝ) : Prop :=
  x - 7*y + 3*z - 8 = 0

theorem plane_equivalent :
  ∃ (s t : ℝ), parametric_plane s t = (x, y, z) ↔ plane_equation x y z :=
by
  sorry

end plane_equivalent_l406_406616


namespace rectangular_parallelepiped_surface_area_of_sphere_l406_406489

theorem rectangular_parallelepiped_surface_area_of_sphere (l w h : ℝ) 
  (hl : l = 3) (hw : w = 2) (hh : h = 1) : 
  let d := real.sqrt (l^2 + w^2 + h^2) in
  let r := d / 2 in
  4 * real.pi * r^2 = 14 * real.pi :=
by
  sorry

end rectangular_parallelepiped_surface_area_of_sphere_l406_406489


namespace probability_good_or_excellent_l406_406750

noncomputable def P_H1 : ℚ := 5 / 21
noncomputable def P_H2 : ℚ := 10 / 21
noncomputable def P_H3 : ℚ := 6 / 21

noncomputable def P_A_given_H1 : ℚ := 1
noncomputable def P_A_given_H2 : ℚ := 1
noncomputable def P_A_given_H3 : ℚ := 1 / 3

noncomputable def P_A : ℚ := 
  P_H1 * P_A_given_H1 + 
  P_H2 * P_A_given_H2 + 
  P_H3 * P_A_given_H3

theorem probability_good_or_excellent : P_A = 17 / 21 :=
by
  sorry

end probability_good_or_excellent_l406_406750


namespace smallest_positive_period_is_pi_strictly_increasing_intervals_range_of_f_in_interval_l406_406365

noncomputable def f (x : ℝ) := 2 * cos x * cos (π / 6 - x) - sqrt 3 * (sin x)^2 + sin x * cos x

theorem smallest_positive_period_is_pi : (∀ x : ℝ, f (x + π) = f x) ∧ (∀ ε > 0, ε < π → ∃ x : ℝ, f (x + ε) ≠ f x) :=
by
  sorry

theorem strictly_increasing_intervals (k : ℤ) : 
  ∀ x y : ℝ, (frac (-5 * π / 12) + k * π) < x ∧ x < y ∧ y < (frac π / 12 + k * π) → f x < f y :=
by
  sorry

theorem range_of_f_in_interval : 
  ∃ m M : ℝ, (∀ x : ℝ, - π / 3 ≤ x ∧ x ≤ π / 2 → m ≤ f x ∧ f x ≤ M) ∧ m = - sqrt 3 ∧ M = 2 :=
by
  sorry

end smallest_positive_period_is_pi_strictly_increasing_intervals_range_of_f_in_interval_l406_406365


namespace monotonic_decreasing_interval_log_function_l406_406145

noncomputable def log_function (x : ℝ) : ℝ := log (x^2 - 2 * x + 4)

theorem monotonic_decreasing_interval_log_function :
  ∀ x : ℝ, x < 1 → monotone_decreasing_on log_function (Set.Iio 1) :=
sorry

end monotonic_decreasing_interval_log_function_l406_406145


namespace inverse_proportion_quadrants_l406_406737

theorem inverse_proportion_quadrants (k : ℝ) : (∀ x, x ≠ 0 → ((x < 0 → (2 - k) / x > 0) ∧ (x > 0 → (2 - k) / x < 0))) → k > 2 :=
by sorry

end inverse_proportion_quadrants_l406_406737


namespace original_prices_sum_l406_406552

theorem original_prices_sum
  (new_price_candy_box : ℝ)
  (new_price_soda_can : ℝ)
  (increase_candy_box : ℝ)
  (increase_soda_can : ℝ)
  (h1 : new_price_candy_box = 10)
  (h2 : new_price_soda_can = 9)
  (h3 : increase_candy_box = 0.25)
  (h4 : increase_soda_can = 0.50) :
  let original_price_candy_box := new_price_candy_box / (1 + increase_candy_box)
  let original_price_soda_can := new_price_soda_can / (1 + increase_soda_can)
  original_price_candy_box + original_price_soda_can = 19 :=
by
  sorry

end original_prices_sum_l406_406552


namespace rhombus_diagonal_length_l406_406585

theorem rhombus_diagonal_length
  (A : ℝ) (a b : ℝ) (h1 : A = 108) (h2 : a = 3 * b) :
  3 * (sqrt (A * 2 / (3 * 2))) = 18 :=
by
  sorry

end rhombus_diagonal_length_l406_406585


namespace beth_wins_config_B_l406_406964

-- Define the nim-values for walls of size n
def nim_value (n : ℕ) : ℕ :=
match n with
| 0 => 0
| 1 => 1
| 2 => 2
| 3 => 3
| 4 => 1
| 5 => 4
| 6 => 3
| _ => 0 -- Only considering up to size 6 as given in the problem
end

-- Define a function to compute the combined nim-value of a list of walls
def combined_nim_value (walls : List ℕ) : ℕ :=
walls.foldr (λ w acc => wxor acc (nim_value w)) 0

-- Define the specific initial state configurations given in the problem
def config_A : List ℕ := [6,1,1]
def config_B : List ℕ := [6,2,1]
def config_C : List ℕ := [6,2,2]
def config_D : List ℕ := [6,3,1]
def config_E : List ℕ := [6,3,2]

-- Statement: Proving that the nim-value of configuration B is zero 
theorem beth_wins_config_B : combined_nim_value config_B = 0 := by {
  -- Define nim-values for individual walls in config_B: [6, 2, 1]
  have h1 : nim_value 6 = 3 := rfl,
  have h2 : nim_value 2 = 2 := rfl,
  have h3 : nim_value 1 = 1 := rfl,
  -- Calculate combined nim-value for config_B using XOR
  show combined_nim_value config_B = 0,
  rw [←h1, ←h2, ←h3],
  simp only [config_B, combined_nim_value, List.foldr, List.foldr_cons],
  calc
    (3 wxor 2 wxor 1) = (3 wxor (2 wxor 1)) : by simp [Nat.wxor_assoc]
                  ... = (3 wxor 3) : by rw [Nat.wxor_comm 2 1, Nat.wxor_self_add_eq, Nat.wxor_self]
                  ... = 0 : by rw [Nat.wxor_self],
  exact rfl,
}

end beth_wins_config_B_l406_406964


namespace not_possible_partitions_l406_406286

def sum_of_elements (A : Set ℕ) : ℕ :=
  A.to_finset.sum id

theorem not_possible_partitions :
  ¬(∃ (A : ℕ → Set ℕ), (∀ i j : ℕ, i ≠ j → (A i ∩ A j = ∅)) ∧ ((⋃ i, A i) = Set.univ)
  ∧ (∀ k : ℕ, sum_of_elements (A k) = k + 2020)
  ∧ (∀ k : ℕ, sum_of_elements (A k) = k^2 + 2020)) := sorry

end not_possible_partitions_l406_406286


namespace verify_min_n_for_coprime_subset_l406_406321

def is_pairwise_coprime (s : Finset ℕ) : Prop :=
  ∀ (a b : ℕ) (ha : a ∈ s) (hb : b ∈ s), a ≠ b → Nat.gcd a b = 1

def contains_4_pairwise_coprime (s : Finset ℕ) : Prop :=
  ∃ t : Finset ℕ, t ⊆ s ∧ t.card = 4 ∧ is_pairwise_coprime t

def min_n_for_coprime_subset : ℕ :=
  111

theorem verify_min_n_for_coprime_subset (S : Finset ℕ) (hS : S = Finset.range 151) :
  ∀ (n : ℕ), (∀ s : Finset ℕ, s ⊆ S ∧ s.card = n → contains_4_pairwise_coprime s) ↔ (n ≥ min_n_for_coprime_subset) :=
sorry

end verify_min_n_for_coprime_subset_l406_406321


namespace area_of_region_Z_l406_406674

theorem area_of_region_Z
  (PQRS : set (ℝ × ℝ))
  (P Q R S T : ℝ × ℝ)
  (h_unit_square : PQRS = {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1})
  (h_equilateral_triangle : equilateral (T::[P, Q]))
  (h_T_inside : T ∈ PQRS)
  (h_distance : ∀ p ∈ PQRS, (0.25 ≤ p.2 ∧ p.2 ≤ 0.5) ↔ p ∈ Z)
  : area Z = (4 - real.sqrt 3) / 16 := 
sorry

end area_of_region_Z_l406_406674


namespace maximize_expression_l406_406434

open Real

theorem maximize_expression :
  ∃ (x y z v w : ℝ),
    (x > 0 ∧ y > 0 ∧ z > 0 ∧ v > 0 ∧ w > 0) ∧ 
    (x^2 + y^2 + z^2 + v^2 + w^2 = 2024) ∧
    let M := ∀ u v w x y : ℝ, 
      (u > 0 ∧ v > 0 ∧ w > 0 ∧ x > 0 ∧ y > 0) ∧ (u^2 + v^2 + w^2 + x^2 + y^2 = 2024) → 
      u*w + 3*v*w + 2*w*x + 4*w*y in
    M + x + y + z + v + w = 20*(sqrt 15) + sqrt 1012 + 3036*(sqrt 30) :=
by {
  use [2*(sqrt 15), 6*(sqrt 15), sqrt 1012, 4*(sqrt 15), 8*(sqrt 15)],
  split,
  {
    repeat {split; norm_num; apply sqrt_pos.},
  },
  split,
  {
    norm_num,
    rw [pow_two, pow_two, pow_two, pow_two, pow_two,
        mul_self_sqrt, mul_self_sqrt, mul_self_sqrt, Real.mul_self_sqrt, Real.mul_self_sqrt],
  },
  {
    sorry
  }
}

end maximize_expression_l406_406434


namespace sum_binary_correct_l406_406595

-- Define the given binary numbers
def b1 := 110
def b2 := 101
def b3 := 1011
def b4 := 10011

-- Define a function to convert binary to natural numbers
noncomputable def binary_to_nat (b : Nat) : Nat :=
Nat.recOn b 0 (λ b natVal, 
    if b % 10 = 1 then 2 * natVal + 1 else 2 * natVal + 0)

-- Sum the binary numbers after converting them to decimals
def sum_b := binary_to_nat b1 + binary_to_nat b2 + binary_to_nat b3 + binary_to_nat b4

-- Define the expected result in binary
def result_in_binary := 101001

-- Define a function to convert natural numbers to binary (provided method for completeness)
noncomputable def nat_to_binary (n : Nat) : Nat :=
Nat.recOn n 0 (λ n binaryVal, binaryVal * 10 + (n % 2))

-- The theorem statement
theorem sum_binary_correct : nat_to_binary sum_b = result_in_binary :=
by intros; sorry

end sum_binary_correct_l406_406595


namespace integer_solution_pairs_l406_406495

theorem integer_solution_pairs (x y : ℕ) (Hx : ∃ a b : ℕ, x = a^2 ∧ y = b^2) : (sqrt (x) + sqrt (y) = sqrt (336)) → ∃ (pairs : list (ℕ × ℕ)), pairs.length = 5 := 
by
  sorry

end integer_solution_pairs_l406_406495


namespace max_distance_and_edge_length_l406_406583

-- Definition of the configuration in the problem
structure Tetrahedron :=
(A B C D : ℝ^3)
(is_regular : ∀ e1 e2 : (ℝ^3), e1 ∈ ({A, B, C, D} : set (ℝ^3)) → e2 ∈ ({A, B, C, D} : set (ℝ^3)) → e1 ≠ e2 → dist e1 e2 = dist A B)

-- Given problem conditions
def point_P : ℝ^3 := sorry
def distance_PA : ℝ := 2
def distance_PB : ℝ := 3

-- Tetrahedron formation and conditions
noncomputable def regular_tetrahedron : Tetrahedron := {
  A := sorry,
  B := sorry,
  C := sorry,
  D := sorry,
  is_regular := sorry
}

-- The distance function
noncomputable def dist (x y : ℝ^3) : ℝ := sorry

-- The line segment CD defined by points C and D
def line_CD (C D : ℝ^3) (x : ℝ) : ℝ^3 := C + x • (D - C)

-- Maximum distance computation function
noncomputable def max_distance_from_P_to_line_CD (P C D : ℝ^3) : ℝ := 
  sorry 

-- The target theorem
theorem max_distance_and_edge_length (hT : regular_tetrahedron) (hPA : dist point_P hT.A = distance_PA) (hPB : dist point_P hT.B = distance_PB) :
  max_distance_from_P_to_line_CD point_P hT.C hT.D = dist point_P (line_CD hT.C hT.D (1 / 2 : ℝ)) → 
  dist hT.A hT.B = (sqrt 153) / 3 :=
sorry

end max_distance_and_edge_length_l406_406583


namespace find_prime_p_l406_406652

theorem find_prime_p :
  ∃ p : ℕ, Prime p ∧ (∃ a b : ℤ, p = 5 ∧ 1 < p ∧ p ≤ 11 ∧ (a^2 + p * a - 720 * p = 0) ∧ (b^2 - p * b + 720 * p = 0)) :=
sorry

end find_prime_p_l406_406652


namespace ordered_pair_solution_l406_406651

theorem ordered_pair_solution :
  ∃ (x y : ℚ), (7 * x - 3 * y = 5) ∧ (y - 3 * x = 8) ∧ 
                (x = -29/2 ∧ y = -71/2) :=
by
  existsi (-29/2 : ℚ)
  existsi (-71/2 : ℚ)
  split
  · -- prove 7x - 3y = 5
    sorry
  split
  · -- prove y - 3x = 8
    sorry
  · -- prove x = -29/2 ∧ y = -71/2
    sorry

end ordered_pair_solution_l406_406651


namespace find_green_weights_l406_406606

theorem find_green_weights (b_w g_w b_c bar_w total_w : Nat) (h_b_w : b_w = 2) (h_g_w : g_w = 3) (h_b_c : b_c = 4) (h_bar_w : bar_w = 2) (h_total_w : total_w = 25) : 
  ∃ g_c : Nat, b_c * b_w + g_c * g_w + bar_w = total_w ∧ g_c = 5 :=
by 
  use 5
  simp [h_b_w, h_g_w, h_b_c, h_bar_w, h_total_w]
  norm_num
  sorry

end find_green_weights_l406_406606


namespace increasing_iff_range_a_l406_406001

def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + 2 * a * x else (2 * a - 1) * x - 3 * a + 6

theorem increasing_iff_range_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ a ≤ f x₂ a) ↔ 1 ≤ a ∧ a ≤ 2 :=
sorry

end increasing_iff_range_a_l406_406001


namespace sid_initial_money_l406_406122

variable (M : ℝ)
variable (spent_on_accessories : ℝ := 12)
variable (spent_on_snacks : ℝ := 8)
variable (remaining_money_condition : ℝ := (M / 2) + 4)

theorem sid_initial_money : (M = 48) → (remaining_money_condition = M - (spent_on_accessories + spent_on_snacks)) :=
by
  sorry

end sid_initial_money_l406_406122


namespace r_plus_s_eq_l406_406490

noncomputable def area_triangle (P Q : ℝ × ℝ) : ℝ :=
  0.5 * (P.1 * Q.2 - Q.1 * P.2).abs

lemma point_on_line_segment (x y : ℝ) : y = (-5 / 3) * x + 15 :=
  sorry

lemma intersects_x_axis : (9, 0 : ℝ) := 
  sorry

lemma intersects_y_axis : (0, 15 : ℝ) :=
  sorry

theorem r_plus_s_eq :
  let P := (9, 0 : ℝ)
  let Q := (0, 15 : ℝ)
  let area_PQO := area_triangle P Q
  -- Given area of ΔPOQ == 4 * area of ΔTOP
  assume (r s : ℝ) 
  (h1 : (area_triangle P Q = 4 * (0.5 * 9 * s)))  -- 0.5 * 9 * s representing the area ΔTOP
  (h2 : (s = (-5 / 3) * r + 15))  -- Point T(r, s) on the line
  -- Prove that r + s = 10.5
  (h3 : (area_triangle (9, 0) (0, 15) = 67.5)),
  r + s = 10.5 := 
begin
  sorry
end

end r_plus_s_eq_l406_406490


namespace corner_sum_eq_l406_406748

variable (a : Fin 4 → Fin 4 → ℝ) (k : ℝ)

-- Conditions
def row_sum (i : Fin 4) : Prop := (∑ j, a i j) = k
def col_sum (j : Fin 4) : Prop := (∑ i, a i j) = k
def main_diag_sum : Prop := (a 0 0 + a 1 1 + a 2 2 + a 3 3) = k
def anti_diag_sum : Prop := (a 0 3 + a 1 2 + a 2 1 + a 3 0) = k

-- Theorem to prove
theorem corner_sum_eq:
  (∀ i, row_sum a k i) →
  (∀ j, col_sum a k j) →
  main_diag_sum a k →
  anti_diag_sum a k →
  (a 0 0 + a 0 3 + a 3 0 + a 3 3) = k :=
sorry

end corner_sum_eq_l406_406748


namespace largest_sum_of_watch_digits_l406_406573

theorem largest_sum_of_watch_digits (h: ℕ) (m: ℕ) (hh: 0 ≤ h ∧ h < 24) (mm: 0 ≤ m ∧ m < 60) : 
  ∃ t, t = largest_sum_of_watch_digits ∧ t = 24 :=
by
  let max_hour_sum := 10
  let max_minute_sum := 14
  let largest_sum_of_watch_digits := max_hour_sum + max_minute_sum
  sorry

end largest_sum_of_watch_digits_l406_406573


namespace trig_expr_evaluation_l406_406982

theorem trig_expr_evaluation :
    (1 - (1 / cos (Real.pi / 6))) * 
    (1 + (1 / sin (Real.pi / 3))) * 
    (1 - (1 / sin (Real.pi / 6))) * 
    (1 + (1 / cos (Real.pi / 3))) = 1 := 
by
  let cos_30 := cos (Real.pi / 6)
  let sin_60 := sin (Real.pi / 3)
  let sin_30 := sin (Real.pi / 6)
  let cos_60 := cos (Real.pi / 3)
  have h_cos_30 : cos_30 = (Real.sqrt 3) / 2 := sorry
  have h_sin_60 : sin_60 = (Real.sqrt 3) / 2 := sorry
  have h_sin_30 : sin_30 = 1 / 2 := sorry
  have h_cos_60 : cos_60 = 1 / 2 := sorry
  sorry

end trig_expr_evaluation_l406_406982


namespace distinct_logarithmic_values_l406_406682

theorem distinct_logarithmic_values : ∀ (a b: ℕ), (a ∈ {2, 3, 4, 5, 6, 7, 8, 9}) ∧ (b ∈ {2, 3, 4, 5, 6, 7, 8, 9}) → 
  -- The set of distinct pairs (a, b) having distinct values of log_a b is 53
  card { x | ∃ a b, a ∈ {2, 3, 4, 5, 6, 7, 8, 9} ∧ b ∈ {2, 3, 4, 5, 6, 7, 8, 9} ∧ x = log a b } = 53 := by
  sorry

end distinct_logarithmic_values_l406_406682


namespace min_additional_games_l406_406846

-- Definitions of parameters
def initial_total_games : ℕ := 5
def initial_falcon_wins : ℕ := 2
def win_percentage_threshold : ℚ := 91 / 100

-- Theorem stating the minimum value for N
theorem min_additional_games (N : ℕ) : (initial_falcon_wins + N : ℚ) / (initial_total_games + N : ℚ) ≥ win_percentage_threshold → N ≥ 29 :=
by
  sorry

end min_additional_games_l406_406846


namespace cos_2alpha_plus_5pi_by_12_l406_406684

open Real

noncomputable def alpha : ℝ := sorry

axiom alpha_obtuse : π / 2 < alpha ∧ alpha < π

axiom sin_alpha_plus_pi_by_3 : sin (alpha + π / 3) = -4 / 5

theorem cos_2alpha_plus_5pi_by_12 : 
  cos (2 * alpha + 5 * π / 12) = 17 * sqrt 2 / 50 :=
by sorry

end cos_2alpha_plus_5pi_by_12_l406_406684


namespace remainder_sum_of_first_eight_primes_div_tenth_prime_l406_406535

theorem remainder_sum_of_first_eight_primes_div_tenth_prime :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) % 29 = 19 :=
by norm_num

end remainder_sum_of_first_eight_primes_div_tenth_prime_l406_406535


namespace find_numbers_l406_406952

theorem find_numbers (a b : ℕ) 
  (h1 : a / b * 6 = 10)
  (h2 : a - b + 4 = 10) :
  a = 15 ∧ b = 9 := by
  sorry

end find_numbers_l406_406952


namespace transformed_sequence_has_large_element_l406_406222

noncomputable def transformed_value (a : Fin 25 → ℤ) (i : Fin 25) : ℤ :=
  a i + a ((i + 1) % 25)

noncomputable def perform_transformation (a : Fin 25 → ℤ) (n : ℕ) : Fin 25 → ℤ :=
  if n = 0 then a
  else perform_transformation (fun i => transformed_value a i) (n - 1)

theorem transformed_sequence_has_large_element :
  ∀ a : Fin 25 → ℤ,
    (∀ i : Fin 13, a i = 1) →
    (∀ i : Fin 12, a (i + 13) = -1) →
    ∃ i : Fin 25, perform_transformation a 100 i > 10^20 :=
by
  sorry

end transformed_sequence_has_large_element_l406_406222


namespace locusOfP_CyclicQuadrilateral_l406_406848

-- Definitions of points and circle properties
def Point : Type := ℝ × ℝ

noncomputable def Circle (O: Point) (r: ℝ) : set Point :=
  { P | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2 }

noncomputable def isInterior (P : Point) (O : Point) (r: ℝ) : Prop :=
  (P.1 - O.1)^2 + (P.2 - O.2)^2 < r^2

noncomputable def isChord (A B P : Point) (O : Point) (r: ℝ) : Prop :=
  -- P is an interior point and A and B are endpoints of a diameter
  isInterior P O r ∧ Circle O r A ∧ Circle O r B

noncomputable def intersectsAt (O A B : Point) (X: Point) : Prop :=
  -- The point X is where a radius intersects a chord
  sorry  -- Replace with actual intersection logic

noncomputable def CyclicQuadrilateral (O X P Y : Point) : Prop :=
  -- The quadrilateral OXPY is cyclic
  ∑a (180 - 3 * (angle P A B + angle P B A)) = 180

theorem locusOfP_CyclicQuadrilateral (O A B P X Q R Y : Point) (r : ℝ)
  (hO_center: O = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hC_circle: Circle O r P)
  (hInter: isInterior P O r)
  (hChordA: isChord A P Q O r)
  (hChordB: isChord B P R O r)
  (hInterOQ: intersectsAt O B R X)
  (hInterOR: intersectsAt O A Q Y)
  (hCyclic: CyclicQuadrilateral O X P Y)
  : { P : Point | angle P A B + angle P B A = 60 } :=
sorry

end locusOfP_CyclicQuadrilateral_l406_406848


namespace infinite_subset_same_gcd_l406_406121

theorem infinite_subset_same_gcd 
  (A : Set ℕ) [Infinite A] 
  (hA : ∀ a ∈ A, ∃ (p : Finset ℕ) (h : p.length ≤ 2000), a = p.prod) :
  ∃ (B : Set ℕ), B ⊆ A ∧ Infinite B ∧ ∃ d, ∀ b1 b2 ∈ B, b1 ≠ b2 → gcd b1 b2 = d :=
sorry

end infinite_subset_same_gcd_l406_406121


namespace MP_equals_MQ_l406_406760

-- Definitions of the triangle, line, points, and midpoint
variable {A B C P Q M : Point} (ell : Line)
variable (triangle_ABC : Triangle A B C)
variable (ell_contains_C : C ∈ ell)
variable (BP_perpendicular : Perp (Line.mk B P) ell)
variable (AQ_perpendicular : Perp (Line.mk A Q) ell)
variable (M_midpoint : Midpoint M A B)

-- Statement to be proved
theorem MP_equals_MQ :
  MP = MQ :=
sorry

end MP_equals_MQ_l406_406760


namespace smallest_integer_y_l406_406301

theorem smallest_integer_y (y : ℤ) : (∃ (y : ℤ), (y / 4) + (3 / 7) > (4 / 7) ∧ ∀ (z : ℤ), z < y → (z / 4) + (3 / 7) ≤ (4 / 7)) := 
by
  sorry

end smallest_integer_y_l406_406301


namespace contradiction_proof_l406_406520

-- Definition of a triangle and its internal angles in Lean
structure Triangle :=
(angle1 angle2 angle3 : ℝ)
(angles_sum : angle1 + angle2 + angle3 = 180)

-- The proposition to be proved: At least one of the internal angles is not greater than 60°
def at_least_one_not_greater_than_60 (T : Triangle) : Prop :=
  T.angle1 ≤ 60 ∨ T.angle2 ≤ 60 ∨ T.angle3 ≤ 60

-- The negation of the proposition
def all_greater_than_60 (T : Triangle) : Prop :=
  T.angle1 > 60 ∧ T.angle2 > 60 ∧ T.angle3 > 60

-- The proof problem statement
theorem contradiction_proof (T : Triangle) : all_greater_than_60 T → ¬ at_least_one_not_greater_than_60 T :=
by
  intros h
  unfold at_least_one_not_greater_than_60 all_greater_than_60
  sorry

end contradiction_proof_l406_406520


namespace problem_statement_l406_406908

open Real

theorem problem_statement (t : ℝ) :
  cos (2 * t) ≠ 0 ∧ sin (2 * t) ≠ 0 →
  cos⁻¹ (2 * t) + sin⁻¹ (2 * t) + cos⁻¹ (2 * t) * sin⁻¹ (2 * t) = 5 →
  (∃ k : ℤ, t = arctan (1/2) + π * k) ∨ (∃ n : ℤ, t = arctan (1/3) + π * n) :=
by
  sorry

end problem_statement_l406_406908


namespace minimum_value_expression_l406_406318

noncomputable def minimum_expression : ℝ := 
  6 * sqrt 3

theorem minimum_value_expression (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2) :
  3 * sin θ + 2 / cos θ + 2 * sqrt 3 * cot θ ≥ minimum_expression :=
sorry

end minimum_value_expression_l406_406318


namespace area_of_ABCD_equals_formula_l406_406831

variables {A B C D : Type*} [metric_space A]

noncomputable def area_of_quadrilateral (AB BC CD AD : ℝ) : ℝ :=
  (AB * CD + BC * AD) / 2

theorem area_of_ABCD_equals_formula (AB BC CD AD : ℝ) :
  area_of_quadrilateral AB BC CD AD = (AB * CD + BC * AD) / 2 :=
by sorry

end area_of_ABCD_equals_formula_l406_406831


namespace total_votes_l406_406407

theorem total_votes (total_votes : ℕ) (brenda_votes : ℕ) (fraction : ℚ) (h : brenda_votes = fraction * total_votes) (h_fraction : fraction = 1 / 5) (h_brenda : brenda_votes = 15) : 
  total_votes = 75 := 
by
  sorry

end total_votes_l406_406407


namespace right_triangle_max_area_l406_406586

theorem right_triangle_max_area
  (a b : ℝ) (h_a_nonneg : 0 ≤ a) (h_b_nonneg : 0 ≤ b)
  (h_right_triangle : a^2 + b^2 = 20^2)
  (h_perimeter : a + b + 20 = 48) :
  (1 / 2) * a * b = 96 :=
by
  sorry

end right_triangle_max_area_l406_406586


namespace probability_all_in_between_l406_406421

noncomputable def probability_between {α : Type*} [MeasureSpace α] (p : α → ℝ) {n : ℕ} (h : ℕ := n) 
  (h_cont : Continuous p) (h_even : ∀ x, p x = p (-x))
  [i.i.d : IndependentlyIdenticallyDistributed (p : Measure ↥α)]
  (ξ : ℕ → α) (h_p : ∀ (i : ℕ), DensityFunction ξ i = p)
  (X : ℕ → ℝ) (X_def : ∀ k, X k = ∑ i in finset.range k, ξ i) : ℝ :=
  (measure_space.the_event (∃ (i : ℕ), 1 ≤ i ∧ i ≤ n - 1 ∧ X 1 ≤ X i ∧ X i ≤ X n)) = 1 / n

theorem probability_all_in_between (p : ℝ → ℝ) {n : ℕ} (h : ℕ := n) 
  (h_cont : Continuous p) (h_even : ∀ x, p x = p (-x))
  [i.i.d : IndependentlyIdenticallyDistributed (p : Measure ↥α : Type*)]
  (ξ : ℕ → ℝ) (h_p : ∀ (i : ℕ), DensityFunction ξ i = p)
  (X : ℕ → ℝ) (X_def : ∀ k, X k = ∑ i in finset.range k, ξ i) : Probability :=
  probability_between p h_cont h_even ξ h_p X X_def = 1 / n :=
sorry

end probability_all_in_between_l406_406421


namespace minimum_value_of_expression_l406_406678

noncomputable def min_expression_value (a b : ℝ) : ℝ :=
  1 / (1 + a) + 4 / (2 + b)

theorem minimum_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + 3 * b = 7) : 
  min_expression_value a b ≥ (13 + 4 * Real.sqrt 3) / 14 :=
by
  sorry

end minimum_value_of_expression_l406_406678


namespace find_k_l406_406347

theorem find_k (k : ℝ) :
  (∃ P : ℝ × ℝ, P.1^2 + P.2^2 - 4 * P.1 - 4 * P.2 + 7 = 0) ∧
  (∃ Q : ℝ × ℝ, Q.2 = k * Q.1) ∧
  (∀ P Q, P.1^2 + P.2^2 - 4 * P.1 - 4 * P.2 + 7 = 0 → Q.2 = k * Q.1 → 
   |dist P Q| = 2 * sqrt 2 - 1) → k = -1 :=
begin
  sorry
end

end find_k_l406_406347


namespace x_squared_minus_y_squared_l406_406380

open Real

theorem x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 4/9)
  (h2 : x - y = 2/9) :
  x^2 - y^2 = 8/81 :=
by
  sorry

end x_squared_minus_y_squared_l406_406380


namespace trig_expression_equality_l406_406973

theorem trig_expression_equality :
  (1 - 1 / (Real.cos (Real.pi / 6))) *
  (1 + 1 / (Real.sin (Real.pi / 3))) *
  (1 - 1 / (Real.sin (Real.pi / 6))) *
  (1 + 1 / (Real.cos (Real.pi / 3))) = 1 :=
by
  have hcos30 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2 := sorry
  have hsin60 : Real.sin (Real.pi / 3) = (Real.sqrt 3) / 2 := sorry
  have hsin30 : Real.sin (Real.pi / 6) = 1 / 2 := sorry
  have hcos60 : Real.cos (Real.pi / 3) = 1 / 2 := sorry
  sorry

end trig_expression_equality_l406_406973


namespace BD_proof_l406_406411

noncomputable def triangle_BD (α b : ℝ) : ℝ :=
let BD := b * sin α / sin (3 * α / 2) in
BD

theorem BD_proof (α b : ℝ) (h1 : α > 0) (h2 : b > 0) :
  let BD := triangle_BD α b in
  BD = b * sin α / sin (3 * α / 2) := 
by 
{
  sorry
}

end BD_proof_l406_406411


namespace first_land_cost_is_8000_l406_406287

noncomputable def cost_of_first_land
  (initial_land : ℕ) 
  (additional_land_cost : ℕ) 
  (cost_per_square_meter : ℕ) 
  (total_land : ℕ) : ℕ :=
let additional_land := total_land - initial_land in
let total_cost_for_additional_land := additional_land * cost_per_square_meter in
let first_land_cost := total_cost_for_additional_land - additional_land_cost in
first_land_cost

theorem first_land_cost_is_8000 :
  cost_of_first_land 300 4000 20 900 = 8000 :=
by
  sorry

end first_land_cost_is_8000_l406_406287


namespace stone_slabs_cover_67_5_square_meters_l406_406925

theorem stone_slabs_cover_67_5_square_meters 
  (num_slabs : ℕ) 
  (side_length_cm : ℕ)
  (side_square: side_length_cm^2 = 22500) 
  (num_slabs_def : num_slabs = 30) 
  (side_length_def : side_length_cm = 150) :
  num_slabs * (side_length_cm * side_length_cm) / 10000 = 67.5 :=
by
  sorry

end stone_slabs_cover_67_5_square_meters_l406_406925


namespace find_incorrect_observation_l406_406492

theorem find_incorrect_observation (n : ℕ) (initial_mean new_mean : ℝ) (correct_value incorrect_value : ℝ) (observations_count : ℕ)
  (h1 : observations_count = 50)
  (h2 : initial_mean = 36)
  (h3 : new_mean = 36.5)
  (h4 : correct_value = 44) :
  incorrect_value = 19 :=
by
  sorry

end find_incorrect_observation_l406_406492


namespace price_of_each_pastry_l406_406234

theorem price_of_each_pastry :
  (∃ (p : ℕ),  ∀ (n : ℕ), (n < 7) → 
    let pastries := 2 + n in
    let total_pastries := (finset.range 7).sum (λ n, 2 + n) in
    let avg_pastries := total_pastries / 7 in
    let total_revenue := avg_pastries * 7 in
    total_revenue = total_pastries * p → p = 1) :=
sorry

end price_of_each_pastry_l406_406234


namespace sum_powers_l406_406074

theorem sum_powers :
  ∃ (α β γ : ℂ), α + β + γ = 2 ∧ α^2 + β^2 + γ^2 = 5 ∧ α^3 + β^3 + γ^3 = 8 ∧ α^5 + β^5 + γ^5 = 46.5 :=
by
  sorry

end sum_powers_l406_406074


namespace correct_propositions_l406_406433

-- Definitions for sets of lines and planes
variables {Line Plane : Type}

-- Definitions for various geometric properties
variables 
  (m n : Line)
  (alpha beta gamma : Plane)

-- Predicate definitions
variable (isSubset : Line → Plane → Prop)
variable (isParallel : Plane → Plane → Prop)
variable (isPerpendicular : Plane → Plane → Prop)
variable (lineParallel : Line → Plane → Prop)
variable (linePerpendicular : Line → Plane → Prop)

-- Propositions
def prop1 := isSubset m beta ∧ isPerpendicular alpha beta → linePerpendicular m alpha
def prop2 := isParallel alpha beta ∧ isSubset m alpha → lineParallel m beta
def prop3 := linePerpendicular n alpha ∧ linePerpendicular n beta ∧ linePerpendicular m alpha → linePerpendicular m beta
def prop4 := isPerpendicular alpha gamma ∧ isPerpendicular beta gamma ∧ linePerpendicular m alpha → linePerpendicular m beta

-- The core theorem statement
theorem correct_propositions
  (h2 : prop2 m alpha beta isSubset isParallel lineParallel)
  (h3 : prop3 m n alpha beta linePerpendicular) : 
  h2 ∧ h3 :=
by
  sorry

end correct_propositions_l406_406433


namespace old_manufacturing_cost_l406_406621

theorem old_manufacturing_cost (P : ℝ) (h1 : 50 = 0.50 * P) : 0.60 * P = 60 :=
by
  sorry

end old_manufacturing_cost_l406_406621


namespace sum_areas_of_circles_l406_406167

theorem sum_areas_of_circles (r s t : ℝ) 
  (h1 : r + s = 6)
  (h2 : r + t = 8)
  (h3 : s + t = 10) : 
  r^2 * Real.pi + s^2 * Real.pi + t^2 * Real.pi = 56 * Real.pi := 
by 
  sorry

end sum_areas_of_circles_l406_406167


namespace no_intersection_line_segment_l406_406348

theorem no_intersection_line_segment (k : ℝ) :
  let A := (-2, 3)
  let B := (3, 2)
  let line := λ x, k * x - 2
  ¬(∃ x y : ℝ, (x, y) ∈ line ∧ (min (-2) 3 ≤ x ∧ x ≤ max (-2) 3 ∧ min 3 2 ≤ y ∧ y ≤ max 3 2)) →
  k ∈ Ioo (-5 / 2) (4 / 3) :=
by
  sorry

end no_intersection_line_segment_l406_406348


namespace circumcircle_radius_of_triangle_ABC_l406_406773

noncomputable def radius_of_circumcircle (BC : ℝ) (angle_A : ℝ) : ℝ :=
  BC / (2 * (Real.sin angle_A))

theorem circumcircle_radius_of_triangle_ABC (BC : ℝ) (angle_A : ℝ)
  (BC_eq : BC = 4) (angle_A_eq : angle_A = Real.pi / 3) : 
  radius_of_circumcircle BC angle_A = 4 * Real.sqrt 3 / 3 := 
by
  rw [BC_eq, angle_A_eq, radius_of_circumcircle]
  have sin_60 : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 := sorry
  rw sin_60
  norm_num
  field_simp [Real.sqrt_ne_zero, show (3 : ℝ) ≠ 0 by norm_num]


end circumcircle_radius_of_triangle_ABC_l406_406773


namespace sum_of_floor_sqrt_1_to_25_l406_406992

theorem sum_of_floor_sqrt_1_to_25 : 
  ( ∑ i in Finset.range 26, ∥Real.sqrt i∥ ) = 75 :=
by
  sorry

end sum_of_floor_sqrt_1_to_25_l406_406992


namespace curve_equation_exists_m_l406_406337

-- Define the curve C
def curve (P : ℝ × ℝ) := (P.1 > 0) ∧ (P.2^2 = 4 * P.1)

-- Define the focal point F
def F : ℝ × ℝ := (1, 0)

-- For a point P on curve C, the condition that the difference 
-- between its distance to F and its distance to the y-axis is 1
def distance_condition (P : ℝ × ℝ) :=
  (P.1 > 0) ∧ (real.sqrt ((P.1 - 1)^2 + P.2^2) - P.1 = 1)

-- Prove the distance condition implies the curve equation
theorem curve_equation (P : ℝ × ℝ) (h : distance_condition P) : curve P :=
sorry

-- Check if there exists an m satisfying the given conditions
def valid_m (m : ℝ) :=
  3 - 2 * real.sqrt 2 < m ∧ m < 3 + 2 * real.sqrt 2 ∧
  ∀ (t : ℝ) (A B : ℝ × ℝ), 
    (A ∈ {P | curve P}) →
    (B ∈ {P | curve P}) →
    ((A.1 = t * A.2 + m) ∧ (B.1 = t * B.2 + m)) →
    (let FA := (A.1 - 1, A.2) in
     let FB := (B.1 - 1, B.2) in
     (FA.1 * FB.1 + FA.2 * FB.2 < 0))

-- Prove that such an m exists and find the range
theorem exists_m : ∃ (m : ℝ), valid_m m :=
sorry

end curve_equation_exists_m_l406_406337


namespace triangle_inequality_l406_406758

-- Definitions and conditions
variable (ABC : Type) [NormedSpace ℚ ABC]
variable {a b c : ℚ}
variable {A B C : ABC}

-- Given conditions
theorem triangle_inequality (h_acute : ∀ {A B C}, A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
  (h_sides_angle : a * Real.cos B = b * (1 + Real.cos A))
  (h_area : 1/2 * a * b * Real.sin C = 2) :

  -- Required range
  8 * (Real.sqrt 2 - 1) < (c + a - b) * (c + b - a) < 8 :=
by sorry

end triangle_inequality_l406_406758


namespace k_values_l406_406313

/-- Definition of the problem conditions and the solution to prove --/
theorem k_values (k : ℕ) (x : ℕ → ℝ) 
  (h_sum : (∑ i in finset.range k, x i) = 9) 
  (h_reciprocal_sum : (∑ i in finset.range k, 1 / (x i)) = 1) 
  (h_pos : ∀ i ∈ finset.range k, x i > 0) : 
  k = 2 ∨ k = 3 :=
sorry

end k_values_l406_406313


namespace boys_under_six_feet_l406_406218

/-- There are 38 students in a certain geometry class.
    Two-thirds of the students are boys.
    Three-fourths of the boys are under 6 feet tall.
    Prove that there are 18 boys in the class who are under 6 feet tall.
-/
theorem boys_under_six_feet (total_students : ℕ) (p_boys : ℚ)
  (p_under_six : ℚ) (boys_under_six_feet : ℕ) :
  total_students = 38 → 
  p_boys = 2 / 3 → 
  p_under_six = 3 / 4 → 
  boys_under_six_feet = (⌊(p_under_six * (p_boys * total_students))⌋ : ℕ) →
  boys_under_six_feet = 18 := by
    intros h1 h2 h3 h4
    sorry

end boys_under_six_feet_l406_406218


namespace proportion_calculation_l406_406730

theorem proportion_calculation (x y : ℝ) (h1 : 0.75 / x = 5 / y) (h2 : x = 1.2) : y = 8 :=
by
  sorry

end proportion_calculation_l406_406730


namespace theatre_lost_revenue_l406_406255

def ticket_price (category : String) : Float :=
  match category with
  | "general" => 10.0
  | "children" => 6.0
  | "senior" => 8.0
  | "veteran" => 8.0  -- $10.00 - $2.00 discount
  | _ => 0.0

def vip_price (base_price : Float) : Float :=
  base_price + 5.0

def calculate_revenue_sold : Float :=
  let general_revenue := 12 * ticket_price "general" + 3 * (vip_price $ ticket_price "general") / 2
  let children_revenue := 3 * ticket_price "children" + vip_price (ticket_price "children")
  let senior_revenue := 4 * ticket_price "senior" + (vip_price (ticket_price "senior")) / 2
  let veteran_revenue := 2 * ticket_price "veteran" + vip_price (ticket_price "veteran")
  general_revenue + children_revenue + senior_revenue + veteran_revenue

def potential_total_revenue : Float :=
  40 * ticket_price "general" + 10 * vip_price (ticket_price "general")

def potential_revenue_lost : Float :=
  potential_total_revenue - calculate_revenue_sold

theorem theatre_lost_revenue : potential_revenue_lost = 224.0 :=
  sorry

end theatre_lost_revenue_l406_406255


namespace solve_equation1_solve_equation2_l406_406837

variable (x : ℚ) -- Assuming x is a rational number for simplicity as it involves fractions.

-- First problem definition and statement
def equation1 := x + 2 * (x - 3) = 3 * (1 - x)
def solution1 := x = 3 / 2

-- Second problem definition and statement
def equation2 := 1 - (2 * x - 1) / 3 = (3 + x) / 6
def solution2 := x = 1

theorem solve_equation1 (h : equation1) : solution1 := sorry
theorem solve_equation2 (h : equation2) : solution2 := sorry

end solve_equation1_solve_equation2_l406_406837


namespace radius_of_circle_l406_406870

-- Define the conditions
def max_inscribed_rectangle_area : ℝ := 50
def side_length_of_square := real.sqrt max_inscribed_rectangle_area

-- Define a helper lemma for the diagonal
lemma diagonal_of_square {s : ℝ} (h : s^2 = max_inscribed_rectangle_area) : real.sqrt (2 * s^2) = 10 :=
by 
  have hs : s = real.sqrt 50 := by
    rw h
    sorry
  calc
    real.sqrt (2 * s^2) = real.sqrt (2 * 50) : by rw [h]
    ... = real.sqrt 100 : by rw [mul_comm, <-mul_assoc, real.sqrt_eq_rpow, <-real.mul_rpow]
    ... = 10 : real.sqrt_eq_rpow 

-- Mathematical statement showing the radius
theorem radius_of_circle : real :=
begin
  let s := side_length_of_square,
  have h : s^2 = max_inscribed_rectangle_area := sorry, -- This follows directly from the definitions above
  let d := real.sqrt (2 * s^2),
  have h_diagonal := diagonal_of_square h,
  let r := d / 2,
  calc
    r = 5 : by rw [h_diagonal]
end

-- Final proof that radius is 5cm
example : radius_of_circle = 5 := by
  unfold radius_of_circle
  sorry

end radius_of_circle_l406_406870


namespace problem_1_problem_2_l406_406364

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ :=  
  (sqrt 3 * sin (ω * x) * cos (ω * x)) - (cos (ω * x))^2

theorem problem_1 (h1 : ∀ x, f x ω = sqrt 3 * (sin (ω * x)) * (cos (ω * x)) - (cos (ω * x)) ^ 2) (h2 : ω > 0) :
  ω = 2 ∧ ∀ k : ℤ, ∀ x, (1/2) * k * π - (π / 12) ≤ x ∧ x ≤ (1/2) * k * π + (π / 6) := 
sorry

theorem problem_2 (a b c : ℝ) (b_sq_eq_ac : b^2 = a * c) (hx : ∀ x, f x 2 = sin (4 * x - π / 6) - 1/2) (H : 0 < x ∧ x < π) : 
  -1 < sin (4 * x - π / 6) - 1/2 ∧ sin (4 * x - π / 6) - 1/2 ≤ 1/2 :=
sorry

end problem_1_problem_2_l406_406364


namespace petya_could_not_spent_atleast_5000_l406_406822

noncomputable def petya_spent_less_than_5000 (k : ℕ) : Prop :=
  let M := 100 * k
  in ∃ (books : list ℕ), 
    (∀ (x ∈ books), (x < 100 ∨ x >= 100)) ∧ -- books prices conditions
    ((∃ (num_100s : ℕ), num_100s = k) ∧ -- initially k number of 100-ruble bills
    (sum books = M / 2) ∧ -- spend exactly half of total money on books 
    (sum books < 5000)) -- verify if the total amount spent on books is less than 5000

theorem petya_could_not_spent_atleast_5000 (k : ℕ) (h : k > 0) : petya_spent_less_than_5000 k :=
sorry

end petya_could_not_spent_atleast_5000_l406_406822


namespace polynomial_root_l406_406210

namespace CubicEquation

-- Define the expression α
def α : ℝ := (1 / 2) * ((5 * Real.sqrt 2 + 7)^(1/3) - (5 * Real.sqrt 2 - 7)^(1/3))

-- State the theorem
theorem polynomial_root :
  let α := (1 / 2) * ((5 * Real.sqrt 2 + 7)^(1/3) - (5 * Real.sqrt 2 - 7)^(1/3))
  in α = 1 ∧ (α ^ 3 + (3 / 4) * α - 7 / 4 = 0) :=
by
  sorry

end CubicEquation

end polynomial_root_l406_406210


namespace reciprocals_sum_l406_406813

-- Definitions based on the conditions in the problem
def quadratic_sum_roots (a b c : ℚ) : ℚ := -b / a
def quadratic_product_roots (a b c : ℚ) : ℚ := c / a

theorem reciprocals_sum (r s : ℚ) (h : polynomial.eval 6 r * polynomial.eval 6 s = 6 * 7) :
  (1/r + 1/s) = 11/7 :=
by
  have rs_pos : c / a > 0 := 
    begin
      -- since c, a > 0, c / a > 0
      sorry
    end
  have r_plus_s_eq : -b / a = 11/6 := 
    begin
      sorry
    end
  have r_times_s_eq : c / a = 7/6 := 
    begin
      sorry
    end
  have hw_expr : ( -b / a) / (c / a) = 11/7 := 
    begin
      sorry
    end
  exact hw_expr

end reciprocals_sum_l406_406813


namespace total_time_elapse_l406_406888

-- Define the conditions
def meeting_point_time (dist : ℕ) (rate1 rate2 rate3 : ℕ) (delay : ℕ) : ℕ :=
  let t := ↑((dist + dist * delay - delay * delay * rate1 - delay * delay * rate2) / (rate1 + rate2 + rate3) : ℚ)
  t.toInt

-- Define the problem parameters
def dist : ℕ := 100
def rate1 : ℕ := 5
def rate2 : ℕ := 4
def rate3 : ℕ := 6
def delay : ℕ := 2

-- Prove that the total time elapsed t equals 112/15 hours
theorem total_time_elapse (t : ℚ) :
  meeting_point_time dist rate1 rate2 rate3 delay = 7.47 :=
sorry

end total_time_elapse_l406_406888


namespace find_number_satisfying_equation_l406_406656

theorem find_number_satisfying_equation :
  ∃ x : ℝ, (196 * x^3) / 568 = 43.13380281690141 ∧ x = 5 :=
by
  sorry

end find_number_satisfying_equation_l406_406656


namespace sequence_series_divergence_l406_406843

theorem sequence_series_divergence {a : ℕ → ℝ} 
    (h : ∀ n : ℕ, (0 < a n) ∧ (a n ≤ a (2*n) + a (2*n+1))) : 
    ¬ Σ' n, a n :=
by
  sorry

end sequence_series_divergence_l406_406843


namespace find_two_digit_number_l406_406793

def product_of_digits (n : ℕ) : ℕ := 
-- Implementation that calculates the product of the digits of n
sorry

def sum_of_digits (n : ℕ) : ℕ := 
-- Implementation that calculates the sum of the digits of n
sorry

theorem find_two_digit_number (M : ℕ) (h1 : 10 ≤ M ∧ M < 100) (h2 : M = product_of_digits M + sum_of_digits M + 1) : M = 18 :=
by
  sorry

end find_two_digit_number_l406_406793


namespace larrys_correct_substitution_l406_406095

noncomputable def lucky_larry_expression (a b c d e f : ℤ) : ℤ :=
  a + (b - (c + (d - (e + f))))

noncomputable def larrys_substitution (a b c d e f : ℤ) : ℤ :=
  a + b - c + d - e + f

theorem larrys_correct_substitution : 
  (lucky_larry_expression 2 4 6 8 e 5 = larrys_substitution 2 4 6 8 e 5) ↔ (e = 8) :=
by
  sorry

end larrys_correct_substitution_l406_406095


namespace reciprocal_gp_sum_eq_l406_406809

-- Conditions and definitions
variables (m : ℕ) (a r s : ℝ)
variable (hm : m > 0)
variable (ha : a ≠ 0)
variable (hr : r ≠ 1)
variable (hs : s = a * (1 - r^m) / (1 - r))

-- Proof goal
theorem reciprocal_gp_sum_eq (m : ℕ) (a r s : ℝ)
  (hm : m > 0) (ha : a ≠ 0) (hr : r ≠ 1) (hs : s = a * (1 - r^m) / (1 - r)) :
  let reciprocal_sum := (1 - (1 / r)^m) / (a * (r - 1)) in
  reciprocal_sum = s / (a * r^(m - 1)) :=
by
  sorry

end reciprocal_gp_sum_eq_l406_406809


namespace book_arrangement_l406_406042

theorem book_arrangement : 
  ∃ (n : ℕ), 
  let math_books := 4 in
  let history_books := 6 in
  let group_arrangements := (math_books + 2).choose(2) in
  let math_arrangements := math_books.factorial in
  let history_arrangements := (history_books.choose(3)) * (history_books - 3).choose(3) in
  n = group_arrangements * math_arrangements * history_arrangements ∧ n = 96000 :=
begin
  use 96000,
  let math_books := 4,
  let history_books := 6,
  let group_arrangements := nat.choose (math_books + 2) 2,
  let math_arrangements := nat.factorial math_books,
  let history_arrangements := (nat.choose history_books 3) * nat.choose (history_books - 3) 3,
  split,
  { rw [group_arrangements, math_arrangements, history_arrangements],
    norm_num },
  { refl }
end

end book_arrangement_l406_406042


namespace number_of_divisors_of_cube_l406_406257

theorem number_of_divisors_of_cube (n : ℕ) (h : ∃ (p : ℕ), p.prime ∧ n = p^2) : 
  (finset.divisors (n^3)).card = 7 :=
by 
  sorry

end number_of_divisors_of_cube_l406_406257


namespace triangle_area_l406_406157

theorem triangle_area (a b c : ℕ) (h1 : a = 32) (h2 : b = 68) (h3 : c = 60) : 
  let s := (a + b + c) / 2 in
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  area = 960 := 
by 
  sorry

end triangle_area_l406_406157


namespace expression_value_l406_406879

theorem expression_value : 2016 - 2017 + 2018 - 2019 + 2020 = 2018 := 
by 
  sorry

end expression_value_l406_406879


namespace area_of_rectangle_l406_406043

-- Definitions of the coordinates
def E : ℝ × ℝ := (-4, 3)
def F : ℝ × ℝ := (996, 43)
def H (y : ℝ) : ℝ × ℝ := (-2, y)

-- Proving the area of the rectangle
theorem area_of_rectangle (y : ℤ) (h1 : y - 3 = -50) : 
  let EF := real.sqrt ((996 + 4)^2 + (43 - 3)^2),
      EH := real.sqrt ((-2 + 4)^2 + (y - 3)^2) in
  EF * EH = 50050 :=
by
  -- some calculations would go here
  sorry

end area_of_rectangle_l406_406043


namespace sum_of_floor_sqrt_1_to_25_l406_406987

theorem sum_of_floor_sqrt_1_to_25 : 
  ( ∑ i in Finset.range 26, ∥Real.sqrt i∥ ) = 75 :=
by
  sorry

end sum_of_floor_sqrt_1_to_25_l406_406987


namespace students_olympiad_impossible_l406_406185

theorem students_olympiad_impossible (a : Fin 12 → ℕ) 
  (h : ∀ i : Fin 11, abs (a i.succ - a i) = 1) : 
  (∑ i, a ⟨i, sorry⟩ ≠ 245) :=
sorry

end students_olympiad_impossible_l406_406185


namespace QT_length_l406_406409

variable {P Q R S T : Type} [metric_space P] [metric_space Q] [metric_space R] [metric_space S] [metric_space T]
variable (dist_PQ : ℝ) (dist_PR : ℝ) (dist_QR : ℝ) (dist_ST : ℝ) (angle_R : ℝ)
variable (angle_PTS : ℝ)

-- Given conditions
def given_conditions : Prop :=
  ∃ P Q R S T,
    angle_R = 90 ∧
    dist_PR = 9 ∧
    dist_QR = 12 ∧
    S ∈ segment P Q ∧
    T ∈ segment Q R ∧
    angle_PTS = 90 ∧
    dist_ST = 6

-- The statement to be proved
def length_of_QT : ℝ := 8 

-- The theorem to be proved
theorem QT_length (h : given_conditions) : QT_distance = length_of_QT := sorry

end QT_length_l406_406409


namespace smaller_solution_of_quadratic_l406_406536

theorem smaller_solution_of_quadratic :
  ∀ x : ℝ, x^2 + 17 * x - 72 = 0 → x = -24 ∨ x = 3 :=
by sorry

end smaller_solution_of_quadratic_l406_406536


namespace area_under_curve_l406_406915

section
  open Real

  def integrand (x : ℝ) : ℝ := x / (1 + sqrt x)

  theorem area_under_curve : 
    ∫ x in 0..1, integrand x = (5 / 3) - 2 * ln 2 :=
  by sorry

end

end area_under_curve_l406_406915


namespace total_shaded_area_eq_9pi_l406_406766

-- Definitions of the conditions
def radius : ℝ := 1
def unshaded_angle : ℝ := 90
def total_circles : ℕ := 12

-- Proving the total shaded area
theorem total_shaded_area_eq_9pi :
  let total_area_of_circle := π * radius^2 in
  let shaded_area_fraction := 1 - (unshaded_angle / 360) in
  let shaded_area_per_circle := shaded_area_fraction * total_area_of_circle in
  let total_shaded_area := (total_circles : ℝ) * shaded_area_per_circle in
  total_shaded_area = 9 * π :=
by sorry

end total_shaded_area_eq_9pi_l406_406766


namespace evaluate_expression_l406_406638

theorem evaluate_expression :
  let x := (1/4 : ℚ)
  let y := (4/5 : ℚ)
  let z := (-2 : ℚ)
  x^3 * y^2 * z^2 = 1/25 :=
by
  sorry

end evaluate_expression_l406_406638


namespace ellipse_semi_major_minor_ratio_l406_406598

theorem ellipse_semi_major_minor_ratio (a b : ℝ) (h_condition : a / b = b / (sqrt (a^2 - b^2))) :
  a^2 / b^2 = 2 / (-1 + sqrt 5) :=
by
  -- Context and variables setup
  let c := sqrt (a^2 - b^2)
  have h1 : c^2 = a^2 - b^2 := by sorry
  have h2 : a / b = b / c := h_condition

  -- Final result
  have h3 : a^2 / b^2 = 2 / (-1 + sqrt 5) := by sorry
  
  exact h3

-- The proof is not included, following the provided instructions.

end ellipse_semi_major_minor_ratio_l406_406598


namespace sector_area_is_nine_l406_406686

-- Defining the given conditions
def arc_length (r θ : ℝ) : ℝ := r * θ
def sector_area (r θ : ℝ) : ℝ := 0.5 * r^2 * θ

-- Given conditions
variables (r : ℝ) (θ : ℝ)
variable (h1 : arc_length r θ = 6)
variable (h2 : θ = 2)

-- Goal: Prove that the area of the sector is 9
theorem sector_area_is_nine : sector_area r θ = 9 := by
  sorry

end sector_area_is_nine_l406_406686


namespace triangle_angles_l406_406154

noncomputable def angle_C_deg := real.arccos (1 - real.sqrt 5 / 2) * 180 / real.pi
noncomputable def angle_B_deg := real.arccos (1 - real.sqrt 5 / 2) * 180 / real.pi

def angle_A_deg := 180 - angle_C_deg - angle_B_deg

theorem triangle_angles (a b c : ℝ) (angle_C angle_B angle_A : ℝ)
  (h1 : a = 3) (h2 : b = real.sqrt 11) (h3 : c = 2 + real.sqrt 5)
  (h4 : angle_C = angle_C_deg)
  (h5 : angle_B = angle_B_deg)
  (h6 : angle_A = angle_A_deg) :
  angle_C + angle_B + angle_A = 180 ∧
  angle_C = angle_B ∧
  angle_B = 36 ∧
  angle_A = 108 :=
by
  sorry

end triangle_angles_l406_406154


namespace point_in_or_on_circle_l406_406386

theorem point_in_or_on_circle (θ : Real) :
  let P := (5 * Real.cos θ, 4 * Real.sin θ)
  let C_eq := ∀ (x y : Real), x^2 + y^2 = 25
  25 * Real.cos θ ^ 2 + 16 * Real.sin θ ^ 2 ≤ 25 := 
by 
  sorry

end point_in_or_on_circle_l406_406386


namespace Adam_smiley_count_l406_406329

theorem Adam_smiley_count :
  ∃ (adam mojmir petr pavel : ℕ), adam + mojmir + petr + pavel = 52 ∧
  petr + pavel = 33 ∧ adam >= 1 ∧ mojmir >= 1 ∧ petr >= 1 ∧ pavel >= 1 ∧
  mojmir > max petr pavel ∧ adam = 1 :=
by
  sorry

end Adam_smiley_count_l406_406329


namespace last_three_digits_x_squared_ends_001_l406_406493

theorem last_three_digits_x_squared_ends_001 (x : ℕ) :
  (x^2 % 1000 = 1) → (x % 1000 ∈ {1, 249, 251, 499, 501, 749, 751, 999}) :=
by
  sorry

end last_three_digits_x_squared_ends_001_l406_406493


namespace jensen_meetings_percentage_l406_406779

theorem jensen_meetings_percentage :
  ∃ (first second third total_work_day total_meeting_time : ℕ),
    total_work_day = 600 ∧
    first = 35 ∧
    second = 2 * first ∧
    third = first + second ∧
    total_meeting_time = first + second + third ∧
    (total_meeting_time * 100) / total_work_day = 35 := sorry

end jensen_meetings_percentage_l406_406779


namespace sum_lent_eq_400_l406_406581

theorem sum_lent_eq_400 :
  ∀ (P : ℝ), 4% per annum interest rate and a time period of 8 years
    → let SI := (P * 4 * 8) / 100 in
      SI = P - 272
    → P = 400 := sorry

end sum_lent_eq_400_l406_406581


namespace outfit_count_l406_406129

theorem outfit_count (shirts pants ties belts : ℕ) (h_shirts : shirts = 8) (h_pants : pants = 5) (h_ties : ties = 4) (h_belts : belts = 2) :
  shirts * pants * (ties + 1) * (belts + 1) = 600 := by
  sorry

end outfit_count_l406_406129


namespace intersection_A_B_l406_406814

def A := {x : ℝ | x > 3}
def B := {x : ℝ | (x - 1) * (x - 4) < 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | 3 < x ∧ x < 4} :=
by
  sorry

end intersection_A_B_l406_406814


namespace theta_pi_over_two_l406_406727

theorem theta_pi_over_two (θ : ℝ) (z: ℂ) (h: z = complex.cos θ - complex.i * complex.sin θ) :
  θ = real.pi / 2 -> z^2 = -1 :=
by 
-- Proof goes here 
sorry

end theta_pi_over_two_l406_406727


namespace runners_meet_again_l406_406657

theorem runners_meet_again 
  (v1 v2 v3 v4 v5 : ℕ)
  (h1 : v1 = 32) 
  (h2 : v2 = 40) 
  (h3 : v3 = 48) 
  (h4 : v4 = 56) 
  (h5 : v5 = 64) 
  (h6 : 400 % (v2 - v1) = 0)
  (h7 : 400 % (v3 - v2) = 0)
  (h8 : 400 % (v4 - v3) = 0)
  (h9 : 400 % (v5 - v4) = 0) :
  ∃ t : ℕ, t = 500 :=
by sorry

end runners_meet_again_l406_406657


namespace sampling_error_with_replacement_sampling_error_without_replacement_l406_406186

noncomputable def with_replacement_sampling_error 
  (sample_size : ℕ) (num_non_standard : ℕ) : ℝ :=
  let w := (num_non_standard : ℝ) / (sample_size : ℝ) in
  Real.sqrt (w * (1 - w) / (sample_size : ℝ))

theorem sampling_error_with_replacement 
  (sample_size : ℕ) (total_batch_size : ℕ) (num_non_standard : ℕ) 
  (h_sample_size : sample_size = 500) 
  (h_total_batch_size : total_batch_size = 10000) 
  (h_num_non_standard : num_non_standard = 10) :
  with_replacement_sampling_error sample_size num_non_standard = 0.00626 :=
by
  have h_w : (num_non_standard : ℝ) / (sample_size : ℝ) = 0.02 := by
    simp [h_sample_size, h_num_non_standard]
  have h_calc : 0.02 * 0.98 / 500 = 3.92e-5 := by
    simp [h_sample_size] 
  have h_std_dev : Real.sqrt 3.92e-5 = 0.00626 := by
    norm_num
  sorry

noncomputable def without_replacement_sampling_error 
  (sample_size : ℕ) (total_batch_size : ℕ) (num_non_standard : ℕ) : ℝ :=
  let w := (num_non_standard : ℝ) / (sample_size : ℝ) in
  let correction_factor := 1 - (sample_size : ℝ) / (total_batch_size : ℝ) in
  Real.sqrt ((w * (1 - w) / (sample_size : ℝ)) * correction_factor)

theorem sampling_error_without_replacement 
  (sample_size : ℕ) (total_batch_size : ℕ) (num_non_standard : ℕ) 
  (h_sample_size : sample_size = 500) 
  (h_total_batch_size : total_batch_size = 10000) 
  (h_num_non_standard : num_non_standard = 10) :
  without_replacement_sampling_error sample_size total_batch_size num_non_standard = 0.00610 :=
by
  have h_w : (num_non_standard : ℝ) / (sample_size : ℝ) = 0.02 := by
    simp [h_sample_size, h_num_non_standard]
  have h_correction : 1 - (sample_size : ℝ) / (total_batch_size : ℝ) = 0.95 := by
    simp [h_sample_size, h_total_batch_size]
  have h_calc : 0.02 * 0.98 / 500 * 0.95 = 3.724e-5 := by
    simp [h_sample_size, h_total_batch_size]
  have h_std_dev : Real.sqrt 3.724e-5 = 0.00610 := by
    norm_num
  sorry

end sampling_error_with_replacement_sampling_error_without_replacement_l406_406186


namespace dot_product_v_w_l406_406985

def v : ℝ × ℝ := (-5, 3)
def w : ℝ × ℝ := (7, -9)

theorem dot_product_v_w : v.1 * w.1 + v.2 * w.2 = -62 := 
  by sorry

end dot_product_v_w_l406_406985


namespace no_divisor_neighbors_l406_406112

def is_divisor (a b : ℕ) : Prop := b % a = 0

def circle_arrangement (arr : Fin 8 → ℕ) : Prop :=
  arr 0 = 7 ∧ arr 1 = 9 ∧ arr 2 = 4 ∧ arr 3 = 5 ∧ arr 4 = 3 ∧ arr 5 = 6 ∧ arr 6 = 8 ∧ arr 7 = 2

def valid_neighbors (arr : Fin 8 → ℕ) : Prop :=
  ¬ is_divisor (arr 0) (arr 1) ∧ ¬ is_divisor (arr 0) (arr 3) ∧
  ¬ is_divisor (arr 1) (arr 2) ∧ ¬ is_divisor (arr 1) (arr 3) ∧ ¬ is_divisor (arr 1) (arr 5) ∧
  ¬ is_divisor (arr 2) (arr 1) ∧ ¬ is_divisor (arr 2) (arr 6) ∧ ¬ is_divisor (arr 2) (arr 3) ∧
  ¬ is_divisor (arr 3) (arr 1) ∧ ¬ is_divisor (arr 3) (arr 4) ∧ ¬ is_divisor (arr 3) (arr 2) ∧ ¬ is_divisor (arr 3) (arr 0) ∧
  ¬ is_divisor (arr 4) (arr 3) ∧ ¬ is_divisor (arr 4) (arr 5) ∧
  ¬ is_divisor (arr 5) (arr 1) ∧ ¬ is_divisor (arr 5) (arr 4) ∧ ¬ is_divisor (arr 5) (arr 6) ∧
  ¬ is_divisor (arr 6) (arr 2) ∧ ¬ is_divisor (arr 6) (arr 5) ∧ ¬ is_divisor (arr 6) (arr 7) ∧
  ¬ is_divisor (arr 7) (arr 6)

theorem no_divisor_neighbors :
  ∀ (arr : Fin 8 → ℕ), circle_arrangement arr → valid_neighbors arr :=
by
  intros arr h
  sorry

end no_divisor_neighbors_l406_406112


namespace sum_of_areas_of_circles_l406_406171

-- Definitions of conditions
def r : ℝ := by sorry  -- radius of the circle at vertex A
def s : ℝ := by sorry  -- radius of the circle at vertex B
def t : ℝ := by sorry  -- radius of the circle at vertex C

axiom sum_radii_r_s : r + s = 6
axiom sum_radii_r_t : r + t = 8
axiom sum_radii_s_t : s + t = 10

-- The statement we want to prove
theorem sum_of_areas_of_circles : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by
  -- Use given axioms and properties of the triangle and circles
  sorry

end sum_of_areas_of_circles_l406_406171


namespace percent_red_prob_l406_406785

-- Define the conditions
def initial_red := 2
def initial_blue := 4
def additional_red := 2
def additional_blue := 2
def total_balloons := initial_red + initial_blue + additional_red + additional_blue
def total_red := initial_red + additional_red

-- State the theorem
theorem percent_red_prob : (total_red.toFloat / total_balloons.toFloat) * 100 = 40 :=
by
  sorry

end percent_red_prob_l406_406785


namespace base8_to_decimal_l406_406895

theorem base8_to_decimal (n : ℕ) (h : n = 54321) : 
  (5 * 8^4 + 4 * 8^3 + 3 * 8^2 + 2 * 8^1 + 1 * 8^0) = 22737 := 
by
  sorry

end base8_to_decimal_l406_406895


namespace box_volume_increase_l406_406260

theorem box_volume_increase (l w h : ℝ)
  (h1 : l * w * h = 5000)
  (h2 : l * w + w * h + h * l = 900)
  (h3 : l + w + h = 60) :
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
  sorry

end box_volume_increase_l406_406260


namespace evaluate_expression_l406_406225

theorem evaluate_expression : (2^1 + 2^0 + 2^(-1)) / (2^(-2) + 2^(-3) + 2^(-4)) = 8 := by
  sorry

end evaluate_expression_l406_406225
