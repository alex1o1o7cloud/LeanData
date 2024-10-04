import Data.Finset.Basic
import Data.Nat.Binomial
import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.EuclideanDomain.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.Geometry.GeometryBasics
import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.GraphColoring
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.ProbabilityMassFunction
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Linarith

namespace ms_warren_total_distance_l724_724486

-- Conditions as definitions
def running_speed : ℝ := 6 -- mph
def running_time : ℝ := 20 / 60 -- hours

def walking_speed : ℝ := 2 -- mph
def walking_time : ℝ := 30 / 60 -- hours

-- Total distance calculation
def distance_ran : ℝ := running_speed * running_time
def distance_walked : ℝ := walking_speed * walking_time
def total_distance : ℝ := distance_ran + distance_walked

-- Statement to be proved
theorem ms_warren_total_distance : total_distance = 3 := by
  sorry

end ms_warren_total_distance_l724_724486


namespace locus_of_medians_intersection_is_circle_l724_724731

-- Given two points A and B, and a circle with center O and radius R
variables (A B O : Point) (R : ℝ)

-- Define a point P on the circumference of the circle
def is_on_circumference (P : Point) : Prop := dist P O = R

-- The statement that needs to be proven
theorem locus_of_medians_intersection_is_circle :
  ∀ (P : Point), is_on_circumference P →
  (∃ (C : Point) (r : ℝ), is_circle (center := C) (radius := r) ∧ r = R / 3 ∧
    C = centroid_of_triangle A B O) :=
by { sorry }

end locus_of_medians_intersection_is_circle_l724_724731


namespace geometric_sequence_second_term_equals_12_l724_724342

variable (a : ℝ) (S : ℕ → ℝ)

-- The given geometric sequence's sum formula
def Sn (n : ℕ) : ℝ := a * 3 ^ n - 2

-- Definition of the first term in this sequence
def a1 : ℝ := Sn a S 1

-- Prove that the second term of the sequence equals 12
theorem geometric_sequence_second_term_equals_12
  (h1 : ∀ n, S n = Sn a n)
  (h2 : a1 = 3 * a - 2) :
  (2 * a * 3 ^ (2 - 1)) = 12 :=
by sorry

end geometric_sequence_second_term_equals_12_l724_724342


namespace inequality_solution_l724_724852

theorem inequality_solution (x : ℝ) : (x ≠ 4) → (frac x^2 - 16 x - 4) > 0 ↔ x ∈ Ioo (-4) 4 ∪ Ioi (4) :=
sorry

end inequality_solution_l724_724852


namespace three_point_sixty_eight_as_fraction_l724_724973

theorem three_point_sixty_eight_as_fraction : 3.68 = 92 / 25 := 
by 
  sorry

end three_point_sixty_eight_as_fraction_l724_724973


namespace count_terminating_decimals_with_nonzero_thousandths_digit_l724_724678

theorem count_terminating_decimals_with_nonzero_thousandths_digit :
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, ∀ p, prime p → p ∣ n → p = 2 ∨ p = 5) ∧
    (∀ n ∈ S, n ≤ 1000) ∧
    (∀ n ∈ S, (1 / (n:ℝ)).floor ≠ 0) ∧
    S.card = 17 :=
sorry

end count_terminating_decimals_with_nonzero_thousandths_digit_l724_724678


namespace circle_equation_range_l724_724869

theorem circle_equation_range (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + a + 1 = 0) → a < 4 := 
by 
  sorry

end circle_equation_range_l724_724869


namespace diameter_existence_l724_724519

-- Define a simple polygon 
structure SimplePolygon (V : Type) :=
(vertices : List V)
(non_intersecting : True) -- Placeholder for the non-intersecting property

-- Definition of a diameter in a simple polygon
def hasDiameterWithArcs (P : SimplePolygon V) : Prop :=
∃ (diameter : (V × V)),
  let n := P.vertices.length in
  -- Placeholder for the property that each arc has at least ⌊n/3⌋ vertices
  (∀ arc1 arc2 : List V, arc1.length ≥ n / 3 ∧ arc2.length ≥ n / 3) -- The arcs should follow the diameter 

-- The theorem statement
theorem diameter_existence (P : SimplePolygon V) : hasDiameterWithArcs P :=
sorry

end diameter_existence_l724_724519


namespace analytic_expression_of_f_range_of_m_l724_724720

noncomputable def f (x : ℝ) : ℝ := a * log x + b * x^2

theorem analytic_expression_of_f (a b : ℝ) (h1 : f 1 = -1) (h2 : f.derivative 1 = 2) :
    f = λ x, 4 * log x - x^2 :=
by
  sorry

noncomputable def g (x m : ℝ) : ℝ := (4 * log x - x^2) + m - log 4

theorem range_of_m (m : ℝ) (h3 : ∀ x ∈ Icc (1 / exp 1) 2, g x m = 0 → (x = (1 / exp 1) ∨ x = 2)) :
    2 < m ∧ m ≤ (4 - 2 * log 2) :=
by
  sorry

end analytic_expression_of_f_range_of_m_l724_724720


namespace torn_sheets_count_l724_724906

noncomputable def first_page_num : ℕ := 185
noncomputable def last_page_num : ℕ := 518
noncomputable def pages_per_sheet : ℕ := 2

theorem torn_sheets_count :
  last_page_num > first_page_num ∧
  last_page_num.digits = first_page_num.digits.rotate 1 ∧
  pages_per_sheet = 2 →
  (last_page_num - first_page_num + 1)/pages_per_sheet = 167 :=
by {
  sorry
}

end torn_sheets_count_l724_724906


namespace round_table_arrangements_l724_724006

theorem round_table_arrangements (n : ℕ) (h : n = 10) (two_specific_next_to_each_other : ∀ a b : ℕ, a = 1 ∧ b = 2) : 
  ∃ ways : ℕ, ways = 2 * (8!) ∧ ways = 80640 :=
  by {
    sorry
  }

end round_table_arrangements_l724_724006


namespace carl_gave_beth_35_coins_l724_724286

theorem carl_gave_beth_35_coins (x : ℕ) (h1 : ∃ n, n = 125) (h2 : ∃ m, m = (125 + x) / 2) (h3 : m = 80) : x = 35 :=
by
  sorry

end carl_gave_beth_35_coins_l724_724286


namespace find_number_l724_724235

theorem find_number : ∃ x : ℝ, 3550 - (1002 / x) = 3500 ∧ x = 20.04 :=
by
  sorry

end find_number_l724_724235


namespace simplify_expression_l724_724099

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
  ((x^2 + 1) / (x - 1) - 2 * x / (x - 1)) = x - 1 :=
by
  -- Proof goes here.
  sorry

end simplify_expression_l724_724099


namespace find_incorrect_statement_l724_724575

def is_opposite (a b : ℝ) := a = -b

theorem find_incorrect_statement :
  ¬∀ (a b : ℝ), (a * b < 0) → is_opposite a b := sorry

end find_incorrect_statement_l724_724575


namespace complement_cardinality_l724_724475

open Set

variable (A B U : Set ℕ)

theorem complement_cardinality :
  A = {4, 5, 7, 9} →
  B = {3, 4, 7, 8, 9} →
  U = A ∪ B →
  card (U \ (A ∩ B)) = 3 :=
by
  intros hA hB hU
  rw [hA, hB, hU]
  sorry

end complement_cardinality_l724_724475


namespace find_two_digit_numbers_l724_724167

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem find_two_digit_numbers :
  ∀ (A : ℕ), (10 ≤ A ∧ A ≤ 99) →
    (sum_of_digits A)^2 = sum_of_digits (A^2) →
    (A = 11 ∨ A = 12 ∨ A = 13 ∨ A = 20 ∨ A = 21 ∨ A = 22 ∨ A = 30 ∨ A = 31 ∨ A = 50) :=
by sorry

end find_two_digit_numbers_l724_724167


namespace tv_sets_in_shop_d_l724_724929

theorem tv_sets_in_shop_d :
  ∃ d : ℕ,
    let a := 20
    let b := 30
    let c := 60
    let e := 50
    let n := 5
    let avg := 48
    d = avg * n - (a + b + c + e) ∧ d = 80 :=
begin
  sorry -- Proof is not provided, but let the statement be correct
end

end tv_sets_in_shop_d_l724_724929


namespace sum_of_numbers_with_lcm_ratio_l724_724861

def lcm (a b : ℕ) : ℕ := sorry -- Assume lcm definition is available
def gcd (a b : ℕ) : ℕ := sorry -- Assume gcd definition is available

theorem sum_of_numbers_with_lcm_ratio (a b : ℕ) (h1 : lcm a b = 30) (h2 : a = 2 * (b / 3)) : a + b = 25 :=
  sorry

end sum_of_numbers_with_lcm_ratio_l724_724861


namespace perimeter_of_triangle_ADE_l724_724368

theorem perimeter_of_triangle_ADE
  (a b : ℝ) (F1 F2 A : ℝ × ℝ) (D E : ℝ × ℝ) 
  (h_ellipse : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1)
  (h_a_gt_b : a > b)
  (h_b_gt_0 : b > 0)
  (h_eccentricity : ∃ c, c / a = 1 / 2 ∧ a^2 - b^2 = c^2)
  (h_F1_F2 : ∀ F1 F2, distance F1 (0, 0) = distance F2 (0, 0) ∧ F1 ≠ F2 ∧ 
                       ∀ P : ℝ × ℝ, (distance P F1 + distance P F2 = 2 * a) ↔ (x : ℝ)(y : ℝ) (h_ellipse x y))
  (h_line_DE : ∃ k, ∃ c, ∀ x F1 A, (2 * a * x/(sqrt k^2 + 1)) = |DE|
  (h_length_DE : |DE| = 6)
  (h_A_vertex : A = (0, b))
  : ∃ perim : ℝ, perim = 13 :=
sorry

end perimeter_of_triangle_ADE_l724_724368


namespace cos_three_pi_over_four_l724_724307

theorem cos_three_pi_over_four :
  Real.cos (3 * Real.pi / 4) = -1 / Real.sqrt 2 :=
by
  sorry

end cos_three_pi_over_four_l724_724307


namespace min_quotient_of_group_products_l724_724135

theorem min_quotient_of_group_products :
  ∃ (S T : Finset ℕ), (S ∪ T = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) ∧
                      (S ∩ T = ∅) ∧
                      (∏ (s ∈ S), s) % (∏ (t ∈ T), t) = 0 ∧
                      (∏ (s ∈ S), s) / (∏ (t ∈ T), t) = 7 :=
sorry

end min_quotient_of_group_products_l724_724135


namespace quadrilateral_problem_l724_724009

theorem quadrilateral_problem
  (A B C D : Type)
  [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]
  (h1 : ∠BAD ≃ ∠ADC)
  (h2 : ∠ABD ≃ ∠BCD)
  (AB BD BC : ℚ)
  (h3 : AB = 10)
  (h4 : BD = 12)
  (h5 : BC = 7) :
  ∃ (p q : ℕ), (Nat.gcd p q = 1) ∧ (CD = p / q) ∧ (p + q = 101) := 
sorry

end quadrilateral_problem_l724_724009


namespace largest_B_div_by_4_l724_724017

-- Given conditions
def is_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9
def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- The seven-digit integer is 4B6792X
def number (B X : ℕ) : ℕ := 4000000 + B * 100000 + 60000 + 7000 + 900 + 20 + X

-- Problem statement: Prove that the largest digit B so that the seven-digit integer 4B6792X is divisible by 4
theorem largest_B_div_by_4 
(B X : ℕ) 
(hX : is_digit X)
(div_4 : divisible_by_4 (number B X)) : 
B = 9 := sorry

end largest_B_div_by_4_l724_724017


namespace circle_parabola_intersections_l724_724296

theorem circle_parabola_intersections : 
  ∃ (points : Finset (ℝ × ℝ)), 
  (∀ p ∈ points, (p.1 ^ 2 + p.2 ^ 2 = 16) ∧ (p.2 = p.1 ^ 2 - 4)) ∧
  points.card = 3 := 
sorry

end circle_parabola_intersections_l724_724296


namespace zero_of_g_function_l724_724389

theorem zero_of_g_function (f : ℝ → ℝ) (h_cont : Continuous f) 
  (h_pos : ∀x : ℝ, x ≠ 0 → f' x + (f x / x) > 0) : 
  ∀ x : ℝ, g x = f(x) + (1 / x) → g(x) ≠ 0 :=
by 
  sorry

end zero_of_g_function_l724_724389


namespace old_clock_duration_l724_724131

noncomputable def hour_hand_speed : ℝ := 0.5 -- degrees per minute
noncomputable def minute_hand_speed : ℝ := 6 -- degrees per minute
noncomputable def coincidence_period : ℝ := 66 -- minutes

theorem old_clock_duration:
  let T := 720 / 11 in 
  let standard_duration := 24 * 60 in 
  let old_clock_duration_in_standard := (standard_duration * coincidence_period) / T in
  old_clock_duration_in_standard = 1452 :=
sorry

end old_clock_duration_l724_724131


namespace triangle_ADE_perimeter_l724_724377

noncomputable def ellipse_perimeter (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) (e : ℝ) (h₃ : e = (1 / 2)) 
(F₁ F₂ : ℝ × ℝ) (h₄ : F₁ ≠ F₂) (D E : ℝ × ℝ) (h₅ : |D - E| = 6) : ℝ :=
  let c := (sqrt (a ^ 2 - b ^ 2)) in
  let A := (0, b) in
  let AD := sqrt ((fst D) ^ 2 + (snd D - b) ^ 2) in
  let AE := sqrt ((fst E) ^ 2 + (snd E - b) ^ 2) in
  AD + AE + |D - E|

theorem triangle_ADE_perimeter (a b : ℝ) (h₁ : a > b > 0) (e : ℝ) (h₂ : e = (1 / 2))
(F₁ F₂ : ℝ × ℝ) (h₃ : F₁ ≠ F₂)
(D E : ℝ × ℝ) (h₄ : |D - E| = 6) : 
  ellipse_perimeter a b (and.left h₁) (and.right h₁) e h₂ F₁ F₂ h₃ D E h₄ = 19 :=
sorry

end triangle_ADE_perimeter_l724_724377


namespace convex_quadrilateral_ineq_l724_724115

theorem convex_quadrilateral_ineq 
  (a b c d e f : ℝ)
  (hpos₁ : 0 < a) (hpos₂ : 0 < b) (hpos₃ : 0 < c) (hpos₄ : 0 < d)
  (he : e > 0) (hf : f > 0)
  : 2 * min a (min b (min c d)) ≤ real.sqrt (e^2 + f^2)
-- equality condition: the quadrilateral must be a rhombus for equality
:=
sorry

end convex_quadrilateral_ineq_l724_724115


namespace area_of_intersection_l724_724944

-- Define the circle centered at (3, 0) with radius 3
def circle1 (x y : ℝ) : Prop := (x - 3) ^ 2 + y ^ 2 = 9

-- Define the circle centered at (0, 3) with radius 3
def circle2 (x y : ℝ) : Prop := x ^ 2 + (y - 3) ^ 2 = 9

-- Defining the theorem to prove the area of intersection of these circles
theorem area_of_intersection : 
  let r := 3 in
  let a := (3, 0) in
  let b := (0, 3) in
  area_intersection (circle1) (circle2) = (9 * π - 18) / 2 := 
sorry

end area_of_intersection_l724_724944


namespace hyperbola_eccentricity_l724_724420

theorem hyperbola_eccentricity (a b : ℝ) (h : a = b) :
  (∃ (e : ℝ), e = (if (a = b ∧ a ≠ 0) then (real.sqrt 2) else 0)) :=
by sorry

end hyperbola_eccentricity_l724_724420


namespace min_value_of_expression_l724_724312

def E (θ : ℝ) : ℝ := 3 * cos θ + 2 / sin θ + 2 * sqrt 3 * cot θ

theorem min_value_of_expression :
  ∀ θ : ℝ, 0 < θ ∧ θ < π / 2 → E θ ≥ 6 * real.cbrt (2 * sqrt 3) := by
  sorry

end min_value_of_expression_l724_724312


namespace girl_receives_greater_than_17_l724_724859

noncomputable def sum_of_s (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  a ((n - 1) % 10) + a n + a ((n + 1) % 10)

theorem girl_receives_greater_than_17 :
  ∃ n : ℕ, n < 10 ∧ sum_of_s (λ i, i + 1) n > 17 :=
by
  sorry

end girl_receives_greater_than_17_l724_724859


namespace sum_of_local_values_l724_724210

def local_value (digit place_value : ℕ) : ℕ := digit * place_value

theorem sum_of_local_values :
  local_value 2 1000 + local_value 3 100 + local_value 4 10 + local_value 5 1 = 2345 :=
by
  sorry

end sum_of_local_values_l724_724210


namespace geometric_mean_l724_724326

-- Define the points on the circumcircle and the perpendicular feet
variables {P A1 A2 A3 P1 P2 P3 : Type}
variables [metric_space P] [metric_space A1] [metric_space A2] [metric_space A3] [metric_space P1] [metric_space P2] [metric_space P3]
variables [metric_space P] 

-- Define distances between points
def dist (x y : P) : ℝ := sorry 

-- Assume P is on the circumcircle of triangle A1 A2 A3
axiom on_circumcircle : (P: P) → P ∈ circumcircle A1 A2 A3

-- Assume P1, P2, and P3 are the feet of the perpendiculars from P to the respective lines and tangents
axiom perp_P1 : perpendicular_segment P A2 A3 P1
axiom perp_P2 : perpendicular_segment P (tangent_at_point A2 (circumcircle A1 A2 A3)) P2
axiom perp_P3 : perpendicular_segment P (tangent_at_point A3 (circumcircle A1 A2 A3)) P3

-- Show PP1^2 = PP2 * PP3
theorem geometric_mean : (dist P P1)^2 = (dist P P2) * (dist P P3) :=
sorry

end geometric_mean_l724_724326


namespace ratio_of_boys_to_girls_l724_724546

theorem ratio_of_boys_to_girls (G B : ℕ) (h_total : G + B = 100) (h_diff : B = G + 20) : B / G = 3 / 2 :=
by
  have hG : G = 40 := by {
    have : 2 * G + 20 = 100 := by {
      rw [← h_diff],
      linarith [h_total]
    },
    linarith
  },
  have hB : B = 60 := by {
    rw [hG, h_diff],
    linarith
  },
  rw [hB, hG],
  norm_num


end ratio_of_boys_to_girls_l724_724546


namespace lateral_area_cone_l724_724766

-- Define the cone problem with given conditions
def radius : ℝ := 5
def slant_height : ℝ := 10

-- Given these conditions, prove the lateral area is 50π
theorem lateral_area_cone (r : ℝ) (l : ℝ) (h_r : r = 5) (h_l : l = 10) : (1/2) * 2 * Real.pi * r * l = 50 * Real.pi :=
by 
  -- import useful mathematical tools
  sorry

end lateral_area_cone_l724_724766


namespace exists_disjoint_or_single_intersection_l724_724824

theorem exists_disjoint_or_single_intersection (n : ℕ) (h : n > 1)
  (S : Finset (Finset (Fin 2*n))) 
  (hS : S.card = (Nat.choose (2*n) n) / 2) :
  ∃ (A B : Finset (Fin 2*n)), A ∈ S ∧ B ∈ S ∧ (A = B ∨ (A ∩ B).card ≤ 1) :=
sorry

end exists_disjoint_or_single_intersection_l724_724824


namespace max_ab_correct_l724_724338

noncomputable def max_ab (k : ℝ) (a b: ℝ) : ℝ :=
if k = -3 then 9 else sorry

theorem max_ab_correct (k : ℝ) (a b: ℝ)
  (h1 : (-3 ≤ k ∧ k ≤ 1))
  (h2 : a + b = 2 * k)
  (h3 : a^2 + b^2 = k^2 - 2 * k + 3) :
  max_ab k a b = 9 :=
sorry

end max_ab_correct_l724_724338


namespace possible_values_l724_724494

def sequence := List.range 1598

def part_mean (n : ℕ) (part : List ℕ) : ℕ :=
  part.sum / part.length

noncomputable def valid_n : ℕ → Prop :=
  λ n, ∃ k : ℕ, k * n = 799

theorem possible_values (n : ℕ) : valid_n n ↔ n = 1 ∨ n = 17 ∨ n = 47 ∨ n = 799 :=
sorry

end possible_values_l724_724494


namespace washing_machine_capacity_l724_724489

def num_shirts : Nat := 19
def num_sweaters : Nat := 8
def num_loads : Nat := 3

theorem washing_machine_capacity :
  (num_shirts + num_sweaters) / num_loads = 9 := by
  sorry

end washing_machine_capacity_l724_724489


namespace gcd_102_238_is_34_l724_724198

noncomputable def gcd_102_238 : ℕ :=
  Nat.gcd 102 238

theorem gcd_102_238_is_34 : gcd_102_238 = 34 := by
  -- Conditions based on the Euclidean algorithm
  have h1 : 238 = 2 * 102 + 34 := by norm_num
  have h2 : 102 = 3 * 34 := by norm_num
  have h3 : Nat.gcd 102 34 = 34 := by
    rw [Nat.gcd, Nat.gcd_rec]
    exact Nat.gcd_eq_left h2

  -- Conclusion
  show gcd_102_238 = 34 from
    calc gcd_102_238 = Nat.gcd 102 238 : rfl
                  ... = Nat.gcd 34 102 : Nat.gcd_comm 102 34
                  ... = Nat.gcd 34 (102 % 34) : by rw [Nat.gcd_rec]
                  ... = Nat.gcd 34 34 : by rw [Nat.mod_eq_of_lt (by norm_num : 34 < 102)]
                  ... = 34 : Nat.gcd_self 34

end gcd_102_238_is_34_l724_724198


namespace factorial_divisibility_l724_724472

theorem factorial_divisibility (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : n! * (m!)^n ∣ (mn)! :=
  sorry

end factorial_divisibility_l724_724472


namespace equation_of_l2_l724_724010

-- Define the initial line equation
def l1 (x : ℝ) : ℝ := -2 * x - 2

-- Define the transformed line equation after translation
def l2 (x : ℝ) : ℝ := l1 (x + 1) + 2

-- Statement to prove
theorem equation_of_l2 : ∀ x, l2 x = -2 * x - 2 := by
  sorry

end equation_of_l2_l724_724010


namespace cube_fits_in_box_with_different_colors_l724_724248

-- Define the cube and the cubic box with painted faces
variable (cube : Fin 6 → ℕ) (box : Fin 6 → ℕ)

-- Define six distinct colors
variable (color_A color_B color_C color_D color_E color_F : ℕ)

-- Assumptions
variable (distinct_colors : 
  color_A ≠ color_B ∧ color_A ≠ color_C ∧ color_A ≠ color_D ∧
  color_A ≠ color_E ∧ color_A ≠ color_F ∧ color_B ≠ color_C ∧
  color_B ≠ color_D ∧ color_B ≠ color_E ∧ color_B ≠ color_F ∧
  color_C ≠ color_D ∧ color_C ≠ color_E ∧ color_C ≠ color_F ∧
  color_D ≠ color_E ∧ color_D ≠ color_F ∧ color_E ≠ color_F)

-- The assumption that each face of the cube and the box is painted uniquely using the six distinct colors
variable (cube_colors : ∀ i, cube i ∈ {color_A, color_B, color_C, color_D, color_E, color_F})
variable (box_colors : ∀ i, box i ∈ {color_A, color_B, color_C, color_D, color_E, color_F})

-- The goal is to place the cube in the box such that no adjacent faces of cube and box have the same color
theorem cube_fits_in_box_with_different_colors :
  ∃ placement : Fin 6 → Fin 6, ∀ i, cube (placement i) ≠ box i :=
sorry

end cube_fits_in_box_with_different_colors_l724_724248


namespace sixth_ingot_placement_l724_724588
noncomputable theory

/-- Define the initial condition of the warehouse with 89 storage chambers, first ingot placement at room 1, 
    and the second ingot placement at room 89. --/
def initial_setup (n : ℕ) : Prop :=
  n = 89 ∧ 
  first_ingot_placed 1 ∧ 
  second_ingot_placed 89

/-- Define the placement strategy: keeping the maximum possible distance 
    to the nearest ingot for each subsequent ingot. --/
def placement_strategy (chambers : ℕ → Prop) (k : ℕ) : Prop :=
  (∀ i ≤ k, ingots_placed = {
    1, 
    89, 
    45, 
    23 ∨ 67, 
    12 ∨ 34 ∨ 56 ∨ 78
  } k)

/-- Prove that the sixth ingot can be placed in rooms 12, 34, 56, or 78. --/
theorem sixth_ingot_placement (k : ℕ) : initial_setup 89 → placement_strategy 89 k → k = 6 → (k_placement = 12 ∨ k_placement = 34 ∨ k_placement = 56 ∨ k_placement = 78) :=
sorry

end sixth_ingot_placement_l724_724588


namespace reciprocal_of_neg_three_l724_724149

theorem reciprocal_of_neg_three : ∃ x : ℚ, (-3) * x = 1 ∧ x = (-1) / 3 := sorry

end reciprocal_of_neg_three_l724_724149


namespace sum_reciprocals_of_triangular_numbers_l724_724664

noncomputable def tn (n : ℕ) : ℚ := n * (n + 1) / 2

theorem sum_reciprocals_of_triangular_numbers :
  (∑ n in Finset.range 1000 | λ n, (1 / tn (n + 1)) ) = 2000 / 1001 := 
sorry

end sum_reciprocals_of_triangular_numbers_l724_724664


namespace cone_slant_height_angle_l724_724206

theorem cone_slant_height_angle (h R : ℝ) (h_pos : 0 < h) (R_pos : 0 < R)
  (V_eq : ∀ (k : ℝ), 0 < k ∧ k < h → (2 * ((h - k) ^ 3) / (h ^ 3)) = 1)
  (A_eq : ∀ (k : ℝ), 0 < k ∧ k < h → 
    (2 * ((R * (h - k) / h) * sqrt ((R * (h - k) / h) ^ 2 + (h - k) ^ 2)) / (R * sqrt (R ^ 2 + h ^ 2))) = 1) :
  ∀ (θ : ℝ), tan θ = R / h → θ = (π / 4) :=
by
  sorry

end cone_slant_height_angle_l724_724206


namespace similar_sizes_bound_l724_724059

theorem similar_sizes_bound (k : ℝ) (hk : k < 2) :
  ∃ (N_k : ℝ), ∀ (A : multiset ℝ), (∀ a ∈ A, a ≤ k * multiset.min A) → 
  A.card ≤ N_k := sorry

end similar_sizes_bound_l724_724059


namespace intersection_point_l724_724449

variables {A B C D E P : Type} [AddCommGroup D] [Module ℝ D]
variables (a b c : D) (bd_ratio : ℝ) (ae_ratio : ℝ)
variables (intersect : D → D → D → D)

def ratios := 
  bd_ratio = 2 ∧ bd_ratio + 1 = 3 ∧ ae_ratio = 3 ∧ ae_ratio + 2 = 5

noncomputable def vector_D := (2 / 3) • c + (1 / 3) • b
noncomputable def vector_E := (3 / 5) • a + (2 / 5) • c
noncomputable def vector_P := intersect b vector_E a vector_D

theorem intersection_point :
  ratios ∧ (vector_P = (6 / 13) • a + (2 / 13) • b + (4 / 13) • c) :=
begin
  sorry
end

end intersection_point_l724_724449


namespace polygon_encloses_250_square_units_l724_724660

def vertices : List (ℕ × ℕ) := [(0, 0), (20, 0), (20, 20), (10, 20), (10, 10), (0, 10)]

def polygon_area (vertices : List (ℕ × ℕ)) : ℕ :=
  -- Function to calculate the area of the given polygon
  sorry

theorem polygon_encloses_250_square_units : polygon_area vertices = 250 := by
  -- Proof that the area of the polygon is 250 square units
  sorry

end polygon_encloses_250_square_units_l724_724660


namespace altitudes_not_necessarily_intersect_at_single_point_l724_724297

-- Define a tetrahedron
structure Tetrahedron :=
  (vertices : Fin 4 → Point3d) -- Assume Point3d is suitably defined

-- Define the concept of altitude in a tetrahedron
def altitude (t : Tetrahedron) (v : Fin 4) : Line3d :=
  let opp_face := {i : Fin 4 | i ≠ v}
  let plane := plane_of_points (t.vertices opp_face.1) (t.vertices opp_face.2) (t.vertices opp_face.3)
  perpendicular_segment_to_plane (t.vertices v) plane -- Assume these functions are suitably defined

-- Define a property of intersection at a single point
def altitudes_intersect_at_single_point (t : Tetrahedron) : Prop :=
  ∃ p : Point3d, ∀ v : Fin 4, p ∈ altitude t v

theorem altitudes_not_necessarily_intersect_at_single_point :
  ¬∀ t : Tetrahedron, altitudes_intersect_at_single_point t :=
by sorry

end altitudes_not_necessarily_intersect_at_single_point_l724_724297


namespace cellphone_gifting_l724_724298

theorem cellphone_gifting (n m : ℕ) (h1 : n = 20) (h2 : m = 3) : 
    (Finset.range n).card * (Finset.range (n - 1)).card * (Finset.range (n - 2)).card = 6840 := by
  sorry

end cellphone_gifting_l724_724298


namespace distance_is_105km_l724_724939

noncomputable def distance_between_cities 
  (speed1 speed2 : ℝ) (time_difference : ℝ) 
  (h_speed1 : speed1 = 60) (h_speed2 : speed2 = 70)
  (h_time_diff : time_difference = 0.25) : ℝ := 
  let t := (time_difference * speed2) / (speed2 - speed1) in
  speed1 * t

theorem distance_is_105km 
  (speed1 speed2 : ℝ) (time_difference : ℝ)
  (h_speed1 : speed1 = 60) (h_speed2 : speed2 = 70)
  (h_time_diff : time_difference = 0.25) : 
  distance_between_cities speed1 speed2 time_difference h_speed1 h_speed2 h_time_diff = 105 :=
by
  sorry

end distance_is_105km_l724_724939


namespace age_ratio_in_4_years_l724_724300

variable {p k x : ℕ}

theorem age_ratio_in_4_years (h₁ : p - 8 = 2 * (k - 8)) (h₂ : p - 14 = 3 * (k - 14)) : x = 4 :=
by
  sorry

end age_ratio_in_4_years_l724_724300


namespace problem_one_problem_two_l724_724724

-- Definitions as per conditions
def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - 2 * a * x

def g (x : ℝ) (a : ℝ) : ℝ := f x a + (1/2 : ℝ) * x^2

-- Problem (1)
theorem problem_one (a : ℝ) (h : (1 : ℝ) - 2 * a = -1) : ∃ x, f x 1 = -1 - Real.log 2 := 
by 
  sorry

-- Problem (2)
theorem problem_two (x0 : ℝ) (h1 : ∃ x, g x 1 = g x0 1) (h2 : x0 > 0) : x0 * f x0 1 + 1 + 1 * x0^2 > 0 :=
by 
  sorry

end problem_one_problem_two_l724_724724


namespace cube_has_8_vertices_l724_724742

theorem cube_has_8_vertices : ∀ (c : Cube), c.vertices = 8 := sorry

end cube_has_8_vertices_l724_724742


namespace no_solution_if_and_only_if_l724_724753

theorem no_solution_if_and_only_if (n : ℝ) : 
  ¬ ∃ (x y z : ℝ), 
    (n * x + y = 1) ∧ 
    (n * y + z = 1) ∧ 
    (x + n * z = 1) ↔ n = -1 :=
by
  sorry

end no_solution_if_and_only_if_l724_724753


namespace equilateral_triangle_probability_l724_724260

noncomputable def probability_P_in_GHIJ (P D E F: Point) (DEF: Triangle D E F) (P_in_DEF: P ∈ DEF) : ℚ := 
  1 / 3

theorem equilateral_triangle_probability
  (D E F P: Point)
  (hD: D ≠ E) (hE: E ≠ F) (hF: F ≠ D)
  (is_equilateral_triangle: is_equilateral (Triangle.mk D E F))
  (P_in_DEF: P ∈ (Triangle.mk D E F)): 
  probability_P_in_GHIJ P D E F (Triangle.mk D E F) P_in_DEF = 1 / 3 :=
begin
  sorry,
end

end equilateral_triangle_probability_l724_724260


namespace exists_root_l724_724829

def f (x : ℝ) : ℝ := 2^x + x - 4

theorem exists_root (f : ℝ → ℝ) (hf : ∀ x y : ℝ, x < y → f x < f y) :
  ∃ c ∈ set.Ioo (1 : ℝ) 2, f c = 0 :=
by
  have hf1 : f 1 < 0 := by norm_num
  have hf2 : f 2 > 0 := by norm_num
  have hf_cont : continuous_on f (set.Icc 1 2) := sorry
  exact intermediate_value_Ioo 1 2 hf_cont hf1 hf2

end exists_root_l724_724829


namespace complex_conjugation_l724_724715

noncomputable def z : ℂ := 6 - 2i

theorem complex_conjugation :
  (∀ z : ℂ, z + I - 3 = 3 - I) → z.conj = 6 + 2 * I := by
  sorry

end complex_conjugation_l724_724715


namespace cube_painted_probability_l724_724264

theorem cube_painted_probability :
  let length := 20
  let width := 1
  let height := 7
  let total_cubes := length * width * height
  let corner_cubes := 8
  let edge_cubes := 4 * (length - 2) + 8 * (height - 2)
  let face_cubes := (length * height) - edge_cubes - corner_cubes
  let corner_prob := (corner_cubes / total_cubes : Rat) * (3 / 6 : Rat)
  let edge_prob := (edge_cubes / total_cubes : Rat) * (2 / 6 : Rat)
  let face_prob := (face_cubes / total_cubes : Rat) * (1 / 6 : Rat)
  corner_prob + edge_prob + face_prob = 9 / 35 := by
  sorry

end cube_painted_probability_l724_724264


namespace triangle_proportions_l724_724440

theorem triangle_proportions
  (A B C P₁ P₂ D E M : Type)
  [noncomputable A B C P₁ P₂ D E M]
  [isTriangle A B C]
  (isMidpoint : isMidpoint M B C)
  (angleBisectors : isSymmetricBisectors A P₁ P₂ (angle A B C))
  (equalDistances : AP₁ = AP₂)
  (MP₁_intersects_circumcircle : intersectsCircumcircle_of_ABP₁ MP₁ D)
  (MP₂_intersects_circumcircle : intersectsCircumcircle_of_ACP₂ MP₂ E)
  (given_ratio : (DP₁ / EM) = (sin(angle D M C) / sin(angle E M B)))
: (BP₁ / BC) = (1/2) * (DP₁ / EM) := 
sorry

end triangle_proportions_l724_724440


namespace reciprocal_of_neg_three_l724_724150

theorem reciprocal_of_neg_three : ∃ x : ℚ, (-3) * x = 1 ∧ x = (-1) / 3 := sorry

end reciprocal_of_neg_three_l724_724150


namespace correct_number_of_statements_l724_724536

def prop1 : Prop := ∀ x : ℚ, x^2 + 2 < 0
def prop2 : Prop := ∃ x : ℝ, x^2 + 4 * x + 4 ≤ 0
def prop3 : Prop := ∀ x : ℝ, x^2 + 4 * x + 4 ≤ 0

-- Define the three propositions based on conditions
def condition_1 := ¬ (∀ q : ℚ, False)
def condition_2 := prop1
def condition_3 := prop2 → prop3

-- Counting the number of correct conditions
def number_of_correct_statements : Nat :=
  [condition_1, condition_2, condition_3].count id

theorem correct_number_of_statements : number_of_correct_statements = 2 := by
  sorry

end correct_number_of_statements_l724_724536


namespace simplify_expression_l724_724098

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
  ((x^2 + 1) / (x - 1) - 2 * x / (x - 1)) = x - 1 :=
by
  -- Proof goes here.
  sorry

end simplify_expression_l724_724098


namespace complex_point_quadrant_l724_724526

theorem complex_point_quadrant :
  let z := (complex.mk 1 (-1)) / (complex.mk 2 (-1))
  (0 < z.re) ∧ (0 < z.im) :=
by
  sorry

end complex_point_quadrant_l724_724526


namespace reeya_third_subject_score_l724_724849

theorem reeya_third_subject_score (s1 s2 s3 s4 : ℝ) (average : ℝ) (num_subjects : ℝ) (total_score : ℝ) :
    s1 = 65 → s2 = 67 → s4 = 95 → average = 76.6 → num_subjects = 4 → total_score = 306.4 →
    (s1 + s2 + s3 + s4) / num_subjects = average → s3 = 79.4 :=
by
  intros h1 h2 h4 h_average h_num_subjects h_total_score h_avg_eq
  -- Proof steps can be added here
  sorry

end reeya_third_subject_score_l724_724849


namespace no_poly_map_unit_circle_to_polygon_l724_724847

theorem no_poly_map_unit_circle_to_polygon :
  ¬∃ (P : Polynomial ℂ), 
    ∃ (vertices : List (ℂ × ℂ)), 
    ∀ (z : ℂ), 
    |z| = 1 → z ∈ (convex_hull ℂ (set.range (λ v : (ℂ × ℂ), P v.1))) → 
    ∃ i, z ∈ Segment ℂ (vertices[i].1) (vertices[i].2) :=
by
  sorry

end no_poly_map_unit_circle_to_polygon_l724_724847


namespace find_m_l724_724531

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2^x + m

theorem find_m (m : ℝ) (h : Function.HasInverse (f x m)) (point_on_inverse : f (2 : ℝ) m = 5) : m = 1 := by
  sorry

end find_m_l724_724531


namespace coin_toss_5_times_same_side_l724_724984

noncomputable def probability_of_same_side (n : ℕ) : ℝ :=
  (1 / 2) ^ n

theorem coin_toss_5_times_same_side :
  probability_of_same_side 5 = 1 / 32 :=
by 
  -- The goal is to prove (1/2)^5 = 1/32
  sorry

end coin_toss_5_times_same_side_l724_724984


namespace range_of_s_l724_724759

def double_value_point (s t : ℝ) (ht : t ≠ -1) :
  Prop := 
  ∀ k : ℝ, (t + 1) * k^2 + t * k + s = 0 →
  (t^2 - 4 * s * (t + 1) > 0)

theorem range_of_s (s t : ℝ) (ht : t ≠ -1) :
  double_value_point s t ht ↔ -1 < s ∧ s < 0 :=
sorry

end range_of_s_l724_724759


namespace pentagon_sums_l724_724033

theorem pentagon_sums (ABCDE : Type) (a b c d e : ℝ) (angles : List ℝ) (sides : List ℝ) :
  (∀ x ∈ angles, x = 120) ∧ (∃ n : ℝ, sides = [n-2, n-1, n, n+1, n+2]) 
  → ∃ s ∈ {6, 9, 7}, s ∈ {a + b + c, a + b + d, b + c + d} :=
by
  sorry

end pentagon_sums_l724_724033


namespace max_area_triangle_l724_724019

theorem max_area_triangle (a b c : ℝ) (h1 : c = 2) (h2 : b = sqrt 2 * a) :
    ∃ S : ℝ, (∀ a b c, S ≤ (1 / 2) * a * b * sqrt (1 - ((c^2 - a^2 - b^2) / (2 * a * b))^2)) ∧ S = 2 * sqrt 2 :=
by
    sorry

end max_area_triangle_l724_724019


namespace perimeter_of_triangle_ADE_l724_724350

noncomputable def ellipse (x y a b : ℝ) : Prop :=
  (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1

def foci_distance (a : ℝ) : ℝ := a / 2

def line_through_f1_perpendicular_to_af2 (x y c : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * (x + c)

def distance_between_points (x1 x2 y1 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem perimeter_of_triangle_ADE
  (a b c : ℝ)
  (h_ellipse : ∀ (x y : ℝ), ellipse x y a b)
  (h_eccentricity : b = Real.sqrt 3 * c)
  (h_foci_distance : foci_distance a = c)
  (h_line : ∀ (x y : ℝ), line_through_f1_perpendicular_to_af2 x y c)
  (h_DE : ∀ (x1 y1 x2 y2 : ℝ), distance_between_points x1 x2 y1 y2 = 6) :
  perimeter_of_triangle_ADE = 13 := sorry

end perimeter_of_triangle_ADE_l724_724350


namespace daily_sales_volume_relationship_maximize_daily_sales_profit_l724_724179

variables (x : ℝ) (y : ℝ) (P : ℝ)

-- Conditions
def cost_per_box : ℝ := 40
def min_selling_price : ℝ := 45
def initial_selling_price : ℝ := 45
def initial_sales_volume : ℝ := 700
def decrease_in_sales_volume_per_dollar : ℝ := 20

-- The functional relationship between y and x
theorem daily_sales_volume_relationship (hx : min_selling_price ≤ x ∧ x < 80) : y = -20 * x + 1600 := by
  sorry

-- The profit function
def profit_function (x : ℝ) := (x - cost_per_box) * (initial_sales_volume - decrease_in_sales_volume_per_dollar * (x - initial_selling_price))

-- Maximizing the profit
theorem maximize_daily_sales_profit : ∃ x_max, x_max = 60 ∧ P = profit_function 60 ∧ P = 8000 := by
  sorry

end daily_sales_volume_relationship_maximize_daily_sales_profit_l724_724179


namespace diaz_age_twenty_years_later_l724_724598

theorem diaz_age_twenty_years_later (D S : ℕ) (h₁ : 10 * D - 40 = 10 * S + 20) (h₂ : S = 30) : D + 20 = 56 :=
sorry

end diaz_age_twenty_years_later_l724_724598


namespace ratio_of_areas_l724_724813

-- Define the points A, B, C, B', C'
variables {A B C C' B' : Type} [real_plane A B C C' B']
-- Given an isosceles right triangle ABC where angle ACB = 90 degrees and AB = AC
def isosceles_right_triangle (ABC : triangle A B C) : Prop :=
  right_angle (angle B C A) ∧ eq_length (segment A B) (segment A C)

-- Given side extensions such that C' is extended from C, CC' = 2 * AC (AC' = 3AC)
def extension_C' (AC CC' : segment) : Prop :=
  eq_length (CC' : segment) (2 * AC)

-- Given side extensions such that B' is extended from B, BB' = 2 * AB (AB' = 3AB)
def extension_B' (AB BB' : segment) : Prop :=
  eq_length (BB' : segment) (2 * AB)

-- Prove the ratio of the area of triangle AB'C' to triangle ABC is 9:1
theorem ratio_of_areas
  (h1 : isosceles_right_triangle ABC)
  (h2 : extension_C' AC CC')
  (h3 : extension_B' AB BB') :
  area (triangle A B' C') = 9 * area (triangle A B C) :=
sorry

end ratio_of_areas_l724_724813


namespace sum_of_segments_divided_9_equal_parts_l724_724282

theorem sum_of_segments_divided_9_equal_parts :
  ∀ (AB : ℝ), AB = 9 → (Σ i in finset.range 9, i + 1) * 2 = 165 :=
by
  assume AB,
  assume h : AB = 9,
  sorry

end sum_of_segments_divided_9_equal_parts_l724_724282


namespace abs_expression_eq_6500_l724_724828

def given_expression (x : ℝ) : ℝ := 
  abs (abs x - x - abs x + 500) - x

theorem abs_expression_eq_6500 (x : ℝ) (h : x = -3000) : given_expression x = 6500 := by
  sorry

end abs_expression_eq_6500_l724_724828


namespace true_propositions_l724_724172

-- Definitions for the problem conditions
def reciprocals (x y : ℝ) : Prop := x * y = 1

def triangle_congruence (T1 T2 : Type) (area1 area2 : ℝ) (congruent : Prop) : Prop :=
  congruent → area1 = area2

def quadratic_has_real_solutions (m : ℝ) : Prop :=
  (4 - 4 * m) ≥ 0

-- Propositions
def converse_reciprocal_proposition : Prop :=
  ∀ (x y : ℝ), reciprocals x y → x * y = 1

def contrapositive_quadratic_proposition (m : ℝ) : Prop :=
  ¬ quadratic_has_real_solutions m → m > 1

-- The overall statement to prove
theorem true_propositions :
  (converse_reciprocal_proposition) ∧ (contrapositive_quadratic_proposition) :=
by sorry

end true_propositions_l724_724172


namespace three_not_divide_thirtyone_l724_724512

theorem three_not_divide_thirtyone : ¬ ∃ q : ℤ, 31 = 3 * q := sorry

end three_not_divide_thirtyone_l724_724512


namespace ratio_male_to_total_first_class_l724_724496

theorem ratio_male_to_total_first_class
  (total_passengers : ℕ)
  (percent_females : ℝ)
  (percent_first_class : ℝ)
  (females_coach : ℕ)
  (h1 : total_passengers = 120)
  (h2 : percent_females = 0.55)
  (h3 : percent_first_class = 0.10)
  (h4 : females_coach = 58) :
  let total_females := total_passengers * percent_females
      first_class_passengers := total_passengers * percent_first_class
      total_coach_passengers := total_passengers - first_class_passengers
      females_first_class := total_females - females_coach
      males_first_class := first_class_passengers - females_first_class in
  males_first_class / first_class_passengers = 1 / 3 := 
by
  sorry

end ratio_male_to_total_first_class_l724_724496


namespace probability_of_two_digit_number_l724_724615

def total_elements_in_set : ℕ := 961
def two_digit_elements_in_set : ℕ := 60

theorem probability_of_two_digit_number :
  (two_digit_elements_in_set : ℚ) / total_elements_in_set = 60 / 961 := by
  sorry

end probability_of_two_digit_number_l724_724615


namespace PQE_or_PQF_angle_l724_724231

variables {A B C I D E F M Q P : Type} [incircle I] [triangle ABC]

noncomputable def midpoint (BC : line) : point := sorry

axiom incircle_touches (I : circle) (ABC : triangle) (D E F : point) : Prop
axiom midpoint_M (BC : line) : Prop
axiom point_on_incircle (Q : point) (I : circle) : Prop
axiom right_angle_AQD (A Q D : point) : Prop
axiom point_on_line (P : point) (AI : line) : Prop
axiom equal_distance (MD MP : length) : Prop

theorem PQE_or_PQF_angle (h1 : incircle_touches I ABC D E F)
    (h2 : midpoint_M BC)
    (h3 : point_on_incircle Q I)
    (h4 : right_angle_AQD A Q D)
    (h5 : point_on_line P (AI : line))
    (h6 : equal_distance (MP MD))
    : ∠PQE = 90° ∨ ∠PQF = 90° :=
sorry

end PQE_or_PQF_angle_l724_724231


namespace perimeter_of_triangle_ADE_l724_724361

noncomputable def ellipse_perimeter (a b : ℝ) (h : a > b) (e : ℝ) (he : e = 1/2) (h_ellipse : ∀ (x y : ℝ), 
                            x^2 / a^2 + y^2 / b^2 = 1) : ℝ :=
13 -- we assert that the perimeter is 13

theorem perimeter_of_triangle_ADE 
  (a b : ℝ) (h : a > b) (e : ℝ) (he : e = 1/2) 
  (C_eq : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1) 
  (upper_vertex_A : ℝ × ℝ)
  (focus_F1 : ℝ × ℝ)
  (focus_F2 : ℝ × ℝ)
  (line_through_F1_perpendicular_to_AF2 : ∀ x y, y = (√3 / 3) * (x + focus_F1.1))
  (points_D_E_on_ellipse : ∃ D E : ℝ × ℝ, line_through_F1_perpendicular_to_AF2 D.1 D.2 = true ∧
    line_through_F1_perpendicular_to_AF2 E.1 E.2 = true ∧ 
    (dist D E = 6)) :
  ∃ perimeter : ℝ, perimeter = ellipse_perimeter a b h e he C_eq :=
sorry

end perimeter_of_triangle_ADE_l724_724361


namespace number_of_cats_l724_724917

-- Define the variables and conditions
def ratio_cats_to_dogs : ℕ × ℕ := (5, 4)
def total_dogs : ℕ := 16

-- Define the total number of cats using the conditions
def total_cats : ℕ :=
  let (cats_ratio, dogs_ratio) := ratio_cats_to_dogs
  in cats_ratio * (total_dogs / dogs_ratio)

-- The theorem to be proven
theorem number_of_cats (h_ratio: ratio_cats_to_dogs = (5, 4)) (h_dogs: total_dogs = 16) : total_cats = 20 :=
by
  -- here goes the proof, which we omit
  sorry

end number_of_cats_l724_724917


namespace clock_angle_9_45_l724_724639

theorem clock_angle_9_45 :
  let h := 9
  let m := 45
  let angle := |(60 * h - 11 * m) / 2|
  angle = 22.5 :=
by
  let h := 9
  let m := 45
  let angle := |(60 * h - 11 * m) / 2|
  have : angle = 22.5 := sorry
  exact this

end clock_angle_9_45_l724_724639


namespace swimming_pool_time_l724_724267

theorem swimming_pool_time
  (A B C : ℝ)
  (h1 : A + B = 1 / 3)
  (h2 : A + C = 1 / 6)
  (h3 : B + C = 1 / 4.5) :
  1 / (A + B + C) = 2.25 :=
by
  sorry

end swimming_pool_time_l724_724267


namespace Petya_tore_out_sheets_l724_724890

theorem Petya_tore_out_sheets (n m : ℕ) (h1 : n = 185) (h2 : m = 518)
  (h3 : m.digits = n.digits) : (m - n + 1) / 2 = 167 :=
by
  sorry

end Petya_tore_out_sheets_l724_724890


namespace more_than_500_correct_at_least_999_correct_l724_724611

-- Define the conditions regarding the wizards and their strategies.
structure WizardTestConditions :=
  (hatNumbers : List ℕ)  -- The list of hat numbers each wizard sees.
  (strategy : List ℕ → ℕ)  -- The strategy function that maps the list of seen hat numbers to a spoken number between 1 and 1001.
  (numbersBetween1And1001 : ∀ (n : ℕ), n ∈ hatNumbers → (1 ≤ n ∧ n ≤ 1001))  -- Every number in hatNumbers is between 1 and 1001.
  (uniqueNumbers : List.Nodup hatNumbers)  -- Ensures that no wizard can say a number that has already been said.

-- Question (a): Can they guarantee more than 500 wizards will identify their hat number correctly?
theorem more_than_500_correct (conditions : WizardTestConditions) : 
  ∃ strategy, 501 ≤ List.countp (λ n, n = conditions.strategy conditions.hatNumbers) conditions.hatNumbers 
:=
  sorry

-- Question (b): Can they guarantee at least 999 wizards will identify their hat number correctly?
theorem at_least_999_correct (conditions : WizardTestConditions) : 
  ∃ strategy, 999 ≤ List.countp (λ n, n = conditions.strategy conditions.hatNumbers) conditions.hatNumbers 
:=
  sorry

end more_than_500_correct_at_least_999_correct_l724_724611


namespace polynomial_evaluation_l724_724318

noncomputable def g (x : ℝ) (p : ℝ) : ℝ := x^3 + p * x^2 + 2 * x + 15
noncomputable def f (x : ℝ) (p : ℝ) (q : ℝ) (d : ℝ) : ℝ := x^4 + 2 * x^3 + q * x^2 + 150 * x + d

theorem polynomial_evaluation :
  ∃ p q d : ℝ, 
    (∀ x : ℝ, (g(x, p) = 0 → f(x, p, q, d) = 0)) ∧ 
    (∀ a b c: ℝ, g(a, p) = 0 ∧ g(b, p) = 0 ∧ g(c, p) = 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c) →
    (f(2, -65.5, q, 1012.5) = -16342.5) :=
begin
  sorry
end

end polynomial_evaluation_l724_724318


namespace magic_8_ball_probability_l724_724797

def probability_positive (p_pos : ℚ) (questions : ℕ) (positive_responses : ℕ) : ℚ :=
  (Nat.choose questions positive_responses : ℚ) * (p_pos ^ positive_responses) * ((1 - p_pos) ^ (questions - positive_responses))

theorem magic_8_ball_probability :
  probability_positive (1/3) 7 3 = 560 / 2187 :=
by
  sorry

end magic_8_ball_probability_l724_724797


namespace volume_of_prism_region_l724_724674

noncomputable def volume_of_region : ℝ :=
  let A : Set (ℝ × ℝ × ℝ) := {p | let (x, y, z) := p in (|x + y + z| + |x + y - z| ≤ 8) ∧ (x ≥ 0) ∧ (y ≥ 0) ∧ (z ≥ 0)} in
  if isClosed A then 
    sorry -- Proof of volume calculation goes here
  else 0

theorem volume_of_prism_region : volume_of_region = 32 := 
by
  sorry

end volume_of_prism_region_l724_724674


namespace term_number_arithmetic_seq_l724_724347

def is_arithmetic_seq {α : Type*} [Add α] (a : ℕ → α) (a₀ d : α) : Prop :=
∀ n : ℕ, a n = a₀ + d * n

theorem term_number_arithmetic_seq:
  ∀ (a : ℕ → ℤ) (k : ℕ), is_arithmetic_seq a 1 2 → a k = 7 → k = 4 :=
by
  intros a k h1 h2
  sorry

end term_number_arithmetic_seq_l724_724347


namespace term_1004_of_sequence_l724_724676

theorem term_1004_of_sequence :
  (∀ (n : ℕ), 0 < n → (∑ i in finset.range n, a (i + 1)) / n = n + 1) →
  a 1004 = 2008 :=
by
  intro h
  sorry

end term_1004_of_sequence_l724_724676


namespace reciprocal_of_neg3_l724_724148

theorem reciprocal_of_neg3 : ∃ x : ℝ, (-3 : ℝ) * x = 1 :=  
begin
  use -1/3,
  norm_num,
end

end reciprocal_of_neg3_l724_724148


namespace leading_four_valid_l724_724851

variable (Person : Type)
variables (Arthur Burton Congreve Downs Ewald Flynn : Person)

variables (President VicePresident Secretary Treasurer : Person)

-- Conditions
axiom Arthur_with_Burton : ¬ (Arthur ≠ Burton → Arthur ≠ VicePresident)
axiom Burton_positions : ¬ (Burton = VicePresident ∨ Burton = Secretary)
axiom Congreve_with_Burton_unless_Flynn : ¬ (Congreve ≠ Burton ∧ Flynn ≠ President)
axiom Downs_constraints : ¬ (Downs = Ewald ∨ Downs = Flynn)
axiom Ewald_constraint : ¬ (Ewald = (Arthur ∧ Burton))
axiom Flynn_president_constraint : ¬ (Flynn = President ∧ Congreve = VicePresident)

theorem leading_four_valid : 
  President = Flynn ∧ 
  VicePresident = Ewald ∧ 
  Secretary = Congreve ∧ 
  Treasurer = Burton := 
by
  sorry

end leading_four_valid_l724_724851


namespace xiaoguang_advances_l724_724919

theorem xiaoguang_advances (x1 x2 x3 x4 : ℝ) (h1 : 96 ≤ (x1 + x2 + x3 + x4) / 4) (hx1 : x1 = 95) (hx2 : x2 = 97) (hx3 : x3 = 94) : 
  98 ≤ x4 := 
by 
  sorry

end xiaoguang_advances_l724_724919


namespace line_in_plane_neither_sufficient_nor_necessary_l724_724748

-- Define the mathematical entities in the problem
variables (α β : Type) 
variables (m : α) 

-- Define conditions
axiom is_plane (α : Type) : Prop
axiom is_line (m : α) : Prop
axiom parallel_planes (α β : Type) : Prop
axiom line_in_plane (m : α) (β : Type) : Prop

-- State the theorem
theorem line_in_plane_neither_sufficient_nor_necessary (h1 : is_plane α) (h2 : is_plane β) (h3 : is_line m) (h4 : parallel_planes α β) :
  ¬(line_in_plane m β → parallel_planes α β) ∧ ¬(parallel_planes α β → line_in_plane m β) :=
sorry

end line_in_plane_neither_sufficient_nor_necessary_l724_724748


namespace probability_min_difference_at_least_three_l724_724557

theorem probability_min_difference_at_least_three :
  let S : Set ℕ := {x | 1 ≤ x ∧ x ≤ 9} in
  let totalWays := Fintype.card (Finset.powersetLen 3 S.toFinset) in
  let validSets := {triplet | triplet ∈ Finset.powersetLen 3 S.toFinset ∧ (∀ {a b : ℕ}, a ∈ triplet → b ∈ triplet → a ≠ b → abs (a - b) ≥ 3)} in
  let numValidSets := Fintype.card validSets.toFinset in
  totalWays = 84 ∧ numValidSets = 1 →
  (numValidSets : ℚ) / totalWays = 1 / 84 :=
by
  sorry

end probability_min_difference_at_least_three_l724_724557


namespace total_cost_of_fruits_l724_724755

theorem total_cost_of_fruits (h_orange_weight : 12 * 2 = 24)
                             (h_apple_weight : 8 * 3.75 = 30)
                             (price_orange : ℝ := 1.5)
                             (price_apple : ℝ := 2.0) :
  (5 * 2 * price_orange + 4 * 3.75 * price_apple) = 45 :=
by
  sorry

end total_cost_of_fruits_l724_724755


namespace nonagon_side_length_l724_724224

theorem nonagon_side_length (perimeter : ℝ) (n : ℕ) (h_reg_nonagon : n = 9) (h_perimeter : perimeter = 171) :
  perimeter / n = 19 := by
  sorry

end nonagon_side_length_l724_724224


namespace marina_total_cost_l724_724070

theorem marina_total_cost (E P R X : ℕ) 
    (h1 : 15 + E + P = 47)
    (h2 : 15 + R + X = 58) :
    15 + E + P + R + X = 90 :=
by
  -- The proof will go here
  sorry

end marina_total_cost_l724_724070


namespace least_possible_length_XZ_is_zero_l724_724579

-- Definitions for given conditions
variable (P Q R X Y Z : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R]
variable (p q r x y z : P)

-- Conditions as definitions
def angle_Q_90_deg (p q r : P) : Prop := ∠ q = 90
def pq_len_4 (pq : ℝ) : Prop := pq = 4
def qr_len_8 (qr : ℝ) : Prop := qr = 8
def line_through_X_parallel_to_QR (x qr : ℝ) : Prop := x.parallel qr
def line_through_Y_parallel_to_PQ (y pq : ℝ) : Prop := y.parallel pq

-- Least possible length of XZ
def least_length_XZ (xz : ℝ) : Prop := xz = 0

-- The proof statement
theorem least_possible_length_XZ_is_zero 
  (h1 : angle_Q_90_deg p q r)
  (h2 : pq_len_4 4)
  (h3 : qr_len_8 8)
  (h4 : line_through_X_parallel_to_QR x 8)
  (h5 : line_through_Y_parallel_to_PQ y 4) :
  least_length_XZ 0 := 
sorry

end least_possible_length_XZ_is_zero_l724_724579


namespace borrowing_methods_l724_724800

theorem borrowing_methods (books : Finset Nat) 
  (h_books : books.card = 3) : 
  (∃ borrow_methods : Nat, borrow_methods = 7 ∧ borrow_methods = books.powerset.card - 1) :=
by
  use 7
  split
  . exact rfl
  . calc 7 = books.powerset.card - 1 : sorry

end borrowing_methods_l724_724800


namespace hexagon_diagonals_sum_l724_724255

theorem hexagon_diagonals_sum (r : ℝ) (A B C D E F : ℝ → ℝ) (d : ℝ)
  (h1 : d = 36) (h2 : (A B + B C + C D + D E + E F) = 5 * 90)
  (h3 : A F = 90):
  let x := EuclideanGeometry.distance A C
  let y := EuclideanGeometry.distance A D
  let z := EuclideanGeometry.distance A E in 
  x + y + z = 428.4 :=
by
  sorry

end hexagon_diagonals_sum_l724_724255


namespace point_B_number_l724_724843

theorem point_B_number (A B : ℤ) (hA : A = -2) (hB : abs (B - A) = 3) : B = 1 ∨ B = -5 :=
sorry

end point_B_number_l724_724843


namespace cube_surface_area_increase_l724_724215

theorem cube_surface_area_increase (s : ℝ) :
  let A_original := 6 * s^2
  let s' := 1.8 * s
  let A_new := 6 * s'^2
  (A_new - A_original) / A_original * 100 = 224 :=
by
  -- Definitions from the conditions
  let A_original := 6 * s^2
  let s' := 1.8 * s
  let A_new := 6 * s'^2
  -- Rest of the proof; replace sorry with the actual proof
  sorry

end cube_surface_area_increase_l724_724215


namespace perimeter_of_triangle_ADE_l724_724353

noncomputable def ellipse (x y a b : ℝ) : Prop :=
  (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1

def foci_distance (a : ℝ) : ℝ := a / 2

def line_through_f1_perpendicular_to_af2 (x y c : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * (x + c)

def distance_between_points (x1 x2 y1 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem perimeter_of_triangle_ADE
  (a b c : ℝ)
  (h_ellipse : ∀ (x y : ℝ), ellipse x y a b)
  (h_eccentricity : b = Real.sqrt 3 * c)
  (h_foci_distance : foci_distance a = c)
  (h_line : ∀ (x y : ℝ), line_through_f1_perpendicular_to_af2 x y c)
  (h_DE : ∀ (x1 y1 x2 y2 : ℝ), distance_between_points x1 x2 y1 y2 = 6) :
  perimeter_of_triangle_ADE = 13 := sorry

end perimeter_of_triangle_ADE_l724_724353


namespace reciprocal_of_neg3_l724_724144

theorem reciprocal_of_neg3 : ∃ x : ℝ, (-3 : ℝ) * x = 1 :=  
begin
  use -1/3,
  norm_num,
end

end reciprocal_of_neg3_l724_724144


namespace geom_seq_log_sum_value_l724_724386

noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
∀ n, a n > 0

noncomputable def condition (a : ℕ → ℝ) : Prop :=
(a 8 * a 10) + (a 7 * a 11) = 2 * real.exp 6

noncomputable def log_sum (a : ℕ → ℝ) : ℝ :=
(finset.range 17).sum (λ n, real.log (a (n+1)))

theorem geom_seq_log_sum_value {a : ℕ → ℝ} 
    (h1 : geom_seq a) (h2 : condition a) : 
    log_sum a = 51 :=
sorry

end geom_seq_log_sum_value_l724_724386


namespace isosceles_triangle_perimeter_l724_724695

-- Define the conditions
def isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ c = a) ∧ (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

-- Define the side lengths
def side1 := 2
def side2 := 2
def base := 5

-- Define the perimeter
def perimeter (a b c : ℝ) := a + b + c

-- State the theorem
theorem isosceles_triangle_perimeter : isosceles_triangle side1 side2 base → perimeter side1 side2 base = 9 :=
  by sorry

end isosceles_triangle_perimeter_l724_724695


namespace proposition_1_proposition_2_proposition_3_correct_proposition_numbers_l724_724508

/-
Definition of the hyperbola given by the equation y^2 / 2 - x^2 = 1 and its asymptotes
-/
def hyperbola (y x : ℝ) : Prop :=
  y^2 / 2 - x^2 = 1

/-
Definition of the function f(x) = log x - 1 / x and checking its zero point in the interval (1, 10)
-/
def f (x : ℝ) : ℝ := log x - 1 / x

/-
Definition of the linear regression equation and its property when x increases by 2 units
-/
def linear_regression (x : ℝ) : ℝ := 3 + 2 * x

/-
Proposition 1: Confirmation of asymptotic lines for the given hyperbola
-/
theorem proposition_1 (y x : ℝ) (hyp : hyperbola y x) : y = x * (sqrt 2) ∨ y = -x * (sqrt 2) := 
sorry

/-
Proposition 2: Zero point of function f(x) in the interval (1, 10)
-/
theorem proposition_2 : (∃ x ∈ set.Ioo 1 10, f x = 0) :=
sorry

/-
Proposition 3: Increase in predicted value when x increases by 2 units in the linear regression equation
-/
theorem proposition_3 (x : ℝ) : linear_regression (x + 2) - linear_regression x = 4 :=
sorry

/-
Proposition number correctness: The correct proposition numbers are 1, 2, and 3
-/
theorem correct_proposition_numbers : [1, 2, 3] =
  let prop1 := proposition_1 in
  let prop2 := proposition_2 in
  let prop3 := proposition_3 in
  [1, 2, 3] :=
  sorry

end proposition_1_proposition_2_proposition_3_correct_proposition_numbers_l724_724508


namespace abc_inequality_l724_724113

theorem abc_inequality (a b c : ℝ) (n : ℕ) (h1 : a ∈ set.Icc (-1 : ℝ) 1) (h2 : b ∈ set.Icc (-1 : ℝ) 1) (h3 : c ∈ set.Icc (-1 : ℝ) 1) (h4 : 1 + 2 * a * b * c ≥ a^2 + b^2 + c^2) :
  1 + 2 * (a * b * c)^n ≥ a^(2 * n) + b^(2 * n) + c^(2 * n) :=
by
  sorry

end abc_inequality_l724_724113


namespace median_of_mode_l724_724425

theorem median_of_mode {x : ℕ} (h : multiset.mode {2, 3, x, 5, 7} = 7) : 
  multiset.median {2, 3, x, 5, 7} = 5 :=
by
  sorry

end median_of_mode_l724_724425


namespace part_I_part_II_l724_724401

def f (x a : ℝ) : ℝ := x^2 - a * x + 3

/-- Part (I) -/
theorem part_I (a b : ℝ) (h1 : ∀ x, f x a ≤ -3 ↔ x ∈ set.Icc b 3) : a = 5 ∧ b = 2 :=
by sorry

/-- Part (II) -/
theorem part_II (a : ℝ) (h2 : ∀ x, (1/2 ≤ x ∧ x ≤ 2) → f x a ≤ 1 - x^2) : a ≥ 4 :=
by sorry

end part_I_part_II_l724_724401


namespace f_symmetric_solutions_l724_724125

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder for f definition

theorem f_symmetric_solutions :
  (∀ x : ℝ, x ≠ 0 → f(x) + 2 * f(1/x) = 3 * x) →
  (∃ x : ℝ, x ≠ 0 ∧ f(x) = f(-x) ↔ x = √2 ∨ x = -√2) :=
begin
  intros h,
  sorry -- Proof omitted
end

end f_symmetric_solutions_l724_724125


namespace simplify_expression_l724_724096

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
    ((x^2 + 1) / (x - 1)) - (2 * x / (x - 1)) = x - 1 := 
by
    sorry

end simplify_expression_l724_724096


namespace complex_in_fourth_quadrant_l724_724383

theorem complex_in_fourth_quadrant (a b : ℝ) :
  let z := (a^2 - 4 * a + 5) + (-b^2 + 2 * b - 6) * complex.I in
  0 < re(z) ∧ im(z) < 0 :=
by
  sorry

end complex_in_fourth_quadrant_l724_724383


namespace minimum_n_l724_724112

-- Define the set S
def S : Finset ℕ := (Finset.range 2005).image (λ n, n + 1)

-- Define a predicate for a set having all pairwise coprime elements
def pairwise_coprime (A : Finset ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ A → b ∈ A → a ≠ b → Nat.coprime a b

-- Define a predicate for a set containing at least one prime number
def contains_prime (A : Finset ℕ) : Prop :=
  ∃ a ∈ A, Nat.prime a

-- Define the main theorem to be proven
theorem minimum_n (n : ℕ) (hn : n = 16):
  (∀ A : Finset ℕ, A ⊆ S → A.card = n → pairwise_coprime A → contains_prime A) :=
sorry

end minimum_n_l724_724112


namespace torn_sheets_count_l724_724907

noncomputable def first_page_num : ℕ := 185
noncomputable def last_page_num : ℕ := 518
noncomputable def pages_per_sheet : ℕ := 2

theorem torn_sheets_count :
  last_page_num > first_page_num ∧
  last_page_num.digits = first_page_num.digits.rotate 1 ∧
  pages_per_sheet = 2 →
  (last_page_num - first_page_num + 1)/pages_per_sheet = 167 :=
by {
  sorry
}

end torn_sheets_count_l724_724907


namespace right_triangle_properties_l724_724781

open Real -- using real numbers
open Classical -- if using noncomputable definitions
noncomputable theory

def right_triangle (A B C : Type) :=
  let angle (P Q R : Type) := sorry -- assume a definition for angles
  (angle B A C + angle C A B + angle A B C = 180) ∧ 
  angle B A C = 45 ∧ 
  angle A B C = 90

def length_AB := 10 * (sqrt 2)
def length_BD (A B C D : Type) := sorry -- assume a definition for length

theorem right_triangle_properties :
  ∀ (A B C D : Type), 
  right_triangle A B C →
  length_AB = 10 * (sqrt 2) →
  -- Prove Area of triangle ABC is 100
  ∃ (area_ABC : ℝ), area_ABC = 100 ∧
  -- Prove Length of BD is 10
  ∃ (length_BD : ℝ), length_BD = 10 :=
begin
  intros A B C D h1 h2,
  existsi 100,
  split,
  { sorry },
  existsi 10,
  { sorry }
end

end right_triangle_properties_l724_724781


namespace offspring_selfcross_l724_724082

def gene : Type := ℕ
def naturalNumbers := (A: gene) (a: gene)

def is_dominant (gA : gene) (ga : gene) : Prop := gA > ga
def is_purebred_yellow (AA : gene × gene) : Prop := AA.fst = AA.snd ∧ AA.fst = 0
def is_purebred_green (aa : gene × gene) : Prop := aa.fst = aa.snd ∧ aa.fst = 1
def is_yellow (Aa : gene × gene) : Prop := (Aa.fst = 0 ∧ Aa.snd = 1) ∨ (Aa.fst = 1 ∧ Aa.snd = 0)

def combine_aa_selfcross_yellow (A a : gene) 
  (gene_combos : List (gene × gene)) : ℚ :=
  3 / 4

theorem offspring_selfcross (A a : gene) 
  (h1: is_dominant A a) 
  (h2: is_purebred_yellow (A, A)) 
  (h3: is_purebred_green (a, a)) 
  (h4: is_yellow (A, a))
  (h5: combine_aa_selfcross_yellow A a 
    [(A, A), (A, a), (a, A), (a, a)]) 
    = 3/4 := 
by {sorry}

end offspring_selfcross_l724_724082


namespace logarithmic_expression_range_l724_724657

theorem logarithmic_expression_range (a : ℝ) : 
  (a - 2 > 0) ∧ (5 - a > 0) ∧ (a - 2 ≠ 1) ↔ (2 < a ∧ a < 3) ∨ (3 < a ∧ a < 5) := 
by
  sorry

end logarithmic_expression_range_l724_724657


namespace inequality_C_l724_724751

variable (a b : ℝ)
variable (h : a > b)
variable (h' : b > 0)

theorem inequality_C : a + b > 2 * b := by
  sorry

end inequality_C_l724_724751


namespace decreasing_on_interval_inequality_range_l724_724711

noncomputable def f (x : ℝ) : ℝ :=
  real.logb 2 (abs x)

def g (x m : ℝ) : ℝ :=
  x^2 + m * abs x

theorem decreasing_on_interval (m : ℝ) (h : ∀ x, x < -2 → deriv (g x m) < 0) : m ≥ -4 :=
sorry

theorem inequality_range (m : ℝ) (h : m > 1 / 4) :
  ∀ x, 1 ≤ x ∧ x ≤ 2 → g x m > x / 4 + 1 / x :=
sorry

end decreasing_on_interval_inequality_range_l724_724711


namespace perimeter_difference_of_inscribed_quadrilateral_l724_724617

theorem perimeter_difference_of_inscribed_quadrilateral :
  ∃ d, d = 4 ∨ d = 8 :=
  let a := 3,
      b := 5,
      c := 7,
      d := 9 in
  -- Given quadrilateral sides a, b, c, d and an inscribed circle
  -- We need to prove there are two possible differences in the perimeter of triangles created by tangents drawn to the circle
  sorry

end perimeter_difference_of_inscribed_quadrilateral_l724_724617


namespace fraction_of_tea_in_cup2_l724_724027

theorem fraction_of_tea_in_cup2 :
  let cup1_initial_tea := 6
  let cup2_initial_milk := 6
  -- After transferring 3 ounces of tea from Cup 1 to Cup 2
  let cup1_after_first_transfer_tea := 3
  let cup2_after_first_transfer_tea := 3
  let cup2_after_first_transfer_mixture := 9
  -- After transferring 4 ounces from Cup 2 back to Cup 1
  let cup1_after_second_transfer_tea := 13 / 3
  let cup1_after_second_transfer_milk := 8 / 3
  let cup1_total_after_second_transfer := 7
  -- After transferring half of the content from Cup 1 back to Cup 2
  let cup1_after_third_transfer_tea_amount := 7 / 2 * (13 / 3) / 7
  let cup1_after_third_transfer_milk_amount := 7 / 2 * (8 / 3) / 7
  let cup2_after_third_transfer_tea_amount := 
    (3 + (7 / 2 * (13 / 3) / 7)) - (7 / 2 * (13 / 3) / 7)
  let cup2_total_after_third_transfer := 7
  fraction_of_tea_in_cup2 = (3.166 / 7) := by
  sorry

end fraction_of_tea_in_cup2_l724_724027


namespace miles_traveled_correct_l724_724839

def initial_odometer_reading := 212.3
def odometer_reading_at_lunch := 372.0
def miles_traveled := odometer_reading_at_lunch - initial_odometer_reading

theorem miles_traveled_correct : miles_traveled = 159.7 :=
by
  sorry

end miles_traveled_correct_l724_724839


namespace midpoint_equidistant_l724_724877

noncomputable def tetrahedron (S A B C H M : Type*) :=
(midpoint SA M)
(∀ x : Type*, dist x M = dist S M)

theorem midpoint_equidistant (S A B C H M : Type*) 
[pyramid : tetrahedron S A B C H M]
{SH_perpendicular : ∀ x ∈ plane ABC, x ⊥ SH}
: ∉ (triangle ABC) :=
begin
  sorry
end

end midpoint_equidistant_l724_724877


namespace locus_of_P_l724_724065

variable (α β γ : ℝ) (t : ℝ)
variable (A B C : ℝ × ℝ)
variable (P : ℝ × ℝ)

theorem locus_of_P 
  (hα : 0 < α ∧ α < π) 
  (hβ : 0 < β ∧ β < π) 
  (hγ : 0 < γ ∧ γ < π) 
  (hαβγ : α + β + γ = π) 
  (harea : t = 1/2 * dist A B * dist B C * sin α) :
  (dist P A ^ 2 * sin (2 * α) + dist P B ^ 2 * sin (2 * β) + dist P C ^ 2 * sin (2 * γ) = 4 * t) ↔ P ∈ set_of (λ P, is_circumcircle A B C P)
  :=
sorry

end locus_of_P_l724_724065


namespace angle_OA_BO_l724_724683

def vector_OA : ℝ × ℝ × ℝ := (1, 1, 1)
def vector_BO : ℝ × ℝ × ℝ := (-3, -3, -3)

theorem angle_OA_BO :
  let angle_oa_bo : ℝ := real.arccos ((vector_OA.1 * vector_BO.1 + vector_OA.2 * vector_BO.2 + vector_OA.3 * vector_BO.3) / 
                                        (real.sqrt (vector_OA.1 * vector_OA.1 + vector_OA.2 * vector_OA.2 + vector_OA.3 * vector_OA.3) * 
                                         real.sqrt (vector_BO.1 * vector_BO.1 + vector_BO.2 * vector_BO.2 + vector_BO.3 * vector_BO.3))) in
  angle_oa_bo = real.pi := 
sorry

end angle_OA_BO_l724_724683


namespace find_varphi_l724_724127

theorem find_varphi (varphi : ℝ) (h1 : 0 < varphi) (h2 : varphi < π / 2)
  (h3 : ∃ x : ℝ, (π/6) < x ∧ x < (π/3) ∧ x = k * (π / 2) + (π / 4) - (varphi / 2) ∧ k ∈ Int) :
  varphi = π / 12 :=
by
  sorry

end find_varphi_l724_724127


namespace total_amount_contributed_l724_724234

theorem total_amount_contributed (n : ℕ) (contributions : Fin n → ℝ) :
  n = 15 →
  (∀ i, 1 ≤ contributions i) →
  (∃ i, contributions i = 16) →
  ∑ i, contributions i = 30 :=
by
  intros h_n h_contributions h_max
  sorry

end total_amount_contributed_l724_724234


namespace three_hundred_percent_of_x_equals_seventy_five_percent_of_y_l724_724747

theorem three_hundred_percent_of_x_equals_seventy_five_percent_of_y
  (x y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 20) : y = 80 := by
  sorry

end three_hundred_percent_of_x_equals_seventy_five_percent_of_y_l724_724747


namespace find_real_x_l724_724309

theorem find_real_x (x : ℝ) : 
  (2 < x / (3 * x - 7) ∧ x / (3 * x - 7) ≤ 6) ↔ (7 / 3 < x ∧ x ≤ 14 / 5) :=
by sorry

end find_real_x_l724_724309


namespace reciprocal_of_neg3_l724_724145

theorem reciprocal_of_neg3 : ∃ x : ℝ, (-3 : ℝ) * x = 1 :=  
begin
  use -1/3,
  norm_num,
end

end reciprocal_of_neg3_l724_724145


namespace root_ratios_equal_l724_724616

theorem root_ratios_equal (a : ℝ) (ha : 0 < a)
  (hroots : ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁^3 + 1 = a * x₁ ∧ x₂^3 + 1 = a * x₂ ∧ x₂ / x₁ = 2018) :
  ∃ y₁ y₂ : ℝ, 0 < y₁ ∧ 0 < y₂ ∧ y₁^3 + 1 = a * y₁^2 ∧ y₂^3 + 1 = a * y₂^2 ∧ y₂ / y₁ = 2018 :=
sorry

end root_ratios_equal_l724_724616


namespace find_real_roots_of_PQ_l724_724474

noncomputable def P (x b : ℝ) : ℝ := x^2 + x / 2 + b
noncomputable def Q (x c d : ℝ) : ℝ := x^2 + c * x + d

theorem find_real_roots_of_PQ (b c d : ℝ)
  (h: ∀ x : ℝ, P x b * Q x c d = Q (P x b) c d)
  (h_d_zero: d = 0) :
  ∃ x : ℝ, P (Q x c d) b = 0 → x = (-c + Real.sqrt (c^2 + 2)) / 2 ∨ x = (-c - Real.sqrt (c^2 + 2)) / 2 :=
by
  sorry

end find_real_roots_of_PQ_l724_724474


namespace sin_cos_identity_l724_724336

theorem sin_cos_identity (a : ℝ) (h : a < 0) (α : ℝ) (P : ℝ × ℝ) 
  (hP : P = (3 * a, -4 * a)) 
  (h1 : let x := 3 * a in (3 * a  = x)) 
  (h2 : let y := -4 * a in (-4 * a = y)) 
  (h3 : let r := -5 * a in -5 * a = r) : 
  sin α + 2 * cos α = -2 / 5 := 
sorry

end sin_cos_identity_l724_724336


namespace reciprocal_of_neg_three_l724_724139

theorem reciprocal_of_neg_three : -3 * (-1 / 3) = 1 := 
by
  sorry

end reciprocal_of_neg_three_l724_724139


namespace power_of_i_2010_l724_724335

namespace ComplexNumbers

axiom i : ℂ
axiom i_power_cycle : ∀ n : ℕ, i^(4*n) = 1

def i1 := i
def i2 := -1
def i3 := -i
def i4 := 1
def i5 := i

theorem power_of_i_2010 : i ^ 2010 = -1 :=
by
  have cycle_repeat : i ^ 2010 = i ^ (502 * 4 + 2) := by
    sorry
  have i_squared : i ^ 2 = -1 := by
    sorry
  rw [cycle_repeat, i_squared]
  exact i_squared
end

end power_of_i_2010_l724_724335


namespace torn_sheets_count_l724_724903

noncomputable def first_page_num : ℕ := 185
noncomputable def last_page_num : ℕ := 518
noncomputable def pages_per_sheet : ℕ := 2

theorem torn_sheets_count :
  last_page_num > first_page_num ∧
  last_page_num.digits = first_page_num.digits.rotate 1 ∧
  pages_per_sheet = 2 →
  (last_page_num - first_page_num + 1)/pages_per_sheet = 167 :=
by {
  sorry
}

end torn_sheets_count_l724_724903


namespace findQuadraticFunctionAndVertex_l724_724726

noncomputable section

def quadraticFunction (x : ℝ) (b c : ℝ) : ℝ :=
  (1 / 2) * x^2 + b * x + c

theorem findQuadraticFunctionAndVertex :
  (∃ b c : ℝ, quadraticFunction 0 b c = -1 ∧ quadraticFunction 2 b c = -3) →
  (quadraticFunction x (-2) (-1) = (1 / 2) * x^2 - 2 * x - 1) ∧
  (∃ (vₓ vᵧ : ℝ), vₓ = 2 ∧ vᵧ = -3 ∧ quadraticFunction vₓ (-2) (-1) = vᵧ)  :=
by
  sorry

end findQuadraticFunctionAndVertex_l724_724726


namespace centroid_of_AXY_l724_724032

theorem centroid_of_AXY
  {A B C G X Y : Point}
  (hG : is_centroid G A B C)
  (hX : is_circumcircle_intersection X A G B B C)
  (hY : is_circumcircle_intersection Y A G C B C) :
  is_centroid G A X Y :=
sorry

end centroid_of_AXY_l724_724032


namespace prob_event_bounds_l724_724506

noncomputable def P (A : Set α) : ℝ := sorry -- Define the probability function P(A) but skip the actual implementation

axiom prob_axiom1 : P (∅ : Set α) = 0 -- Axiom: P(∅) = 0
axiom prob_axiom2 : P (Ω : Set α) = 1 -- Axiom: P(Ω) = 1
axiom prob_monotonicity (A B : Set α) (hAB : A ⊆ B) : P(A) ≤ P(B) -- Monotonicity: if A ⊆ B, then P(A) ≤ P(B)

theorem prob_event_bounds (A : Set α) (hA : ∅ ⊆ A ∧ A ⊆ Ω) : 0 ≤ P(A) ∧ P(A) ≤ 1 := by
  sorry -- Skipping the proof

end prob_event_bounds_l724_724506


namespace necessary_condition_l724_724332

variables (a b : ℝ)

theorem necessary_condition (h : a > b) : a > b - 1 :=
sorry

end necessary_condition_l724_724332


namespace gcd_102_238_l724_724197

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end gcd_102_238_l724_724197


namespace log_tan_ratio_l724_724381

theorem log_tan_ratio (α β : Real) (h1 : Real.sin (α + β) = 1 / 2) (h2 : Real.sin (α - β) = 1 / 3) :
  Real.log 5 (Real.tan α / Real.tan β) = 1 :=
sorry

end log_tan_ratio_l724_724381


namespace box_surface_area_is_276_l724_724266

-- Define the dimensions of the box
variables {l w h : ℝ}

-- Define the pricing function
def pricing (x y z : ℝ) : ℝ := 0.30 * x + 0.40 * y + 0.50 * z

-- Define the condition for the box fee
def box_fee (x y z : ℝ) (fee : ℝ) := pricing x y z = fee

-- Define the constraint that no faces are squares
def no_square_faces (l w h : ℝ) : Prop := 
  l ≠ w ∧ w ≠ h ∧ h ≠ l

-- Define the surface area calculation
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

-- The main theorem stating the problem
theorem box_surface_area_is_276 (l w h : ℝ) 
  (H1 : box_fee l w h 8.10 ∧ box_fee w h l 8.10)
  (H2 : box_fee l w h 8.70 ∧ box_fee w h l 8.70)
  (H3 : no_square_faces l w h) : 
  surface_area l w h = 276 := 
sorry

end box_surface_area_is_276_l724_724266


namespace log_cos_7pi_4_l724_724658

theorem log_cos_7pi_4 : Real.logBase 2 (Real.cos (7 * Real.pi / 4)) = -1 / 2 := by
  sorry

end log_cos_7pi_4_l724_724658


namespace large_triangle_construction_l724_724175

-- Definitions based on conditions
def colors : Finset ℕ := Finset.range 8  -- 8 different colors labeled as 0 to 7

noncomputable def distinguishable_large_triangles :ℕ := 2400

theorem large_triangle_construction (c : ℕ) (corner1 corner2 corner3 : ℕ) (center : ℕ) 
  (h1 : c ∈ colors) 
  (h2 : corner1 ∈ colors) 
  (h3 : corner2 ∈ colors) 
  (h4 : corner3 ∈ colors)
  (h5 : center ∈ colors) 
  (h6 : center ≠ 0)            -- center cannot be red (assuming red is labeled as 0)
  (h7 : center ≠ 1)            -- center cannot be blue (assuming blue is labeled as 1)
  (h8 : (corner1, corner2, corner3).Permutation (λ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a) ∨ 
        (corner1, corner2, corner3).Permutation (λ (a b c : ℕ), (a = b ∧ b = c) ∨ 
         (a = b ∧ b ≠ c) ∨ (a ≠ b ∧ b = c) ∨ (a = c ∧ b ≠ c))) : 

  ∃ t : ℕ, t = distinguishable_large_triangles := 
  sorry

end large_triangle_construction_l724_724175


namespace find_50th_term_index_l724_724653

noncomputable def sequence_b (n : ℕ) : ℝ := ∑ k in range(n+1), cos (2 * (k+1))

theorem find_50th_term_index : ∃ n : ℕ, (∀ k < 50, ∃ m, sequence_b m < 0) ∧ n = 314 :=
sorry

end find_50th_term_index_l724_724653


namespace floor_inequality_l724_724503

theorem floor_inequality (a : ℕ → ℕ) (n : ℕ) (h : 0 < n) :
  (∑ i in Finset.range n, ⌊(a i)^2 / a (i + 1)⌋) ≥ (∑ i in Finset.range n, a i) :=
by { sorry }

end floor_inequality_l724_724503


namespace initial_quarters_l724_724842

-- Define the conditions
def quartersAfterLoss (x : ℕ) : ℕ := (4 * x) / 3
def quartersAfterThirdYear (x : ℕ) : ℕ := x - 4
def quartersAfterSecondYear (x : ℕ) : ℕ := x - 36
def quartersAfterFirstYear (x : ℕ) : ℕ := x * 2

-- The main theorem
theorem initial_quarters (x : ℕ) (h1 : quartersAfterLoss x = 140)
    (h2 : quartersAfterThirdYear 140 = 136)
    (h3 : quartersAfterSecondYear 136 = 100)
    (h4 : quartersAfterFirstYear 50 = 100) :
  x = 50 := by
  simp [quartersAfterFirstYear, quartersAfterSecondYear,
        quartersAfterThirdYear, quartersAfterLoss] at *
  sorry

end initial_quarters_l724_724842


namespace coloring_ways_l724_724647

def color := fin 3 -- Define color as a finite type with 3 elements
def pos := fin 9 -- Define positions on the grid from 0 to 8

-- Define adjacency relationships
def adjacent : pos → pos → Prop
| 0 1 := true | 1 0 := true
| 0 3 := true | 3 0 := true
| 1 2 := true | 2 1 := true
| 1 4 := true | 4 1 := true
| 2 5 := true | 5 2 := true
| 3 4 := true | 4 3 := true
| 3 6 := true | 6 3 := true
| 4 5 := true | 5 4 := true
| 4 7 := true | 7 4 := true
| 5 8 := true | 8 5 := true
| 6 7 := true | 7 6 := true
| 7 8 := true | 8 7 := true
| _ _ := false

-- Define valid coloring as a function from positions to colors 
-- such that adjacent positions do not share the same color
def valid_coloring (f : pos → color) : Prop :=
  ∀ (i j : pos), adjacent i j → f i ≠ f j

-- The theorem states: there are exactly 3 valid colorings of the grid
theorem coloring_ways : ∃ f₁ f₂ f₃ : pos → color,
  valid_coloring f₁ ∧
  valid_coloring f₂ ∧
  valid_coloring f₃ ∧
  (∀ f, valid_coloring f → f = f₁ ∨ f = f₂ ∨ f = f₃) :=
sorry

end coloring_ways_l724_724647


namespace area_of_rectangle_l724_724443

-- Definitions for the conditions
def point := ℝ × ℝ

structure Rectangle :=
  (P Q R S : point)

def trisected_by (R T U : point) (PQ : point → point → ℝ) := 
  let angle_R := 90 -- Assuming the angle in PQRS at R is 90 degrees since it's a rectangle
  -- Formalizing the fact that RT and RU trisect the angle
  sorry

-- Given conditions
axiom PQRS : Rectangle
axiom point_R : PQRS.R = (a, b)
axiom point_T : PQRS.P = (0, c)
axiom point_U : PQRS.Q = (d, 0)
axiom QU : d = 8
axiom PT : c = 3

-- Question translated to Lean proof statement
theorem area_of_rectangle {PQRS : Rectangle}
  (trisected : trisected_by PQRS.R PQRS.T PQRS.U (λ p q : point, sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)))
  (QU_eq_8 : PQRS.Q.1 - PQRS.U.1 = 8)
  (PT_eq_3 : PQRS.P.2 - PQRS.T.2 = 3) :
  let RQ := √(8*8*3)
  let TS := √(8*8*3) - 3
  let RS := TS * √3
  let area := RQ * RS 
  area = 192 * √3 - 72 :=
sorry

end area_of_rectangle_l724_724443


namespace problem_midpoint_angle_bisector_l724_724638

open Real InnerProductGeometry

variables {G D E F H : Type} [InnerProductSpace ℝ G] 
variables (A B C D E F G H : G)
variables [Midpoint G B C G]
variables [PointsOnLines A B D A C E]
variables [IntersectionOfLines D E F B C]
variables [AngleBisector A G H F]

-- The proof problem statement in Lean 4
theorem problem_midpoint_angle_bisector (A B C D E F G H : G) 
  [Midpoint G B C G]
  [PointsOnLines A B D A C E]
  [IntersectionOfLines D E F B C]
  [AngleBisector A G H F]
  (h1 : Midpoint G B C)
  (h2 : on_side D A B ∧ BD = CE ∧ on_side E A C)
  (h3 : lines_intersect B E C D F)
  (h4 : angle_bisector A FG H)
  : FG = GH :=
sorry -- proof here

end problem_midpoint_angle_bisector_l724_724638


namespace walking_rate_ratio_l724_724602

theorem walking_rate_ratio :
  let T := 16
  let T' := 12
  (T : ℚ) / (T' : ℚ) = (4 : ℚ) / (3 : ℚ) := 
by
  sorry

end walking_rate_ratio_l724_724602


namespace common_divisor_of_power_less_numbers_l724_724204

-- Definition of power-less digits
def is_power_less_digit (d : ℕ) : Prop := 
  d ≠ 0 ∧ d ≠ 1 ∧ d ≠ 4 ∧ d ≠ 8 ∧ d ≠ 9

-- Definition of power-less number
def is_power_less_number (n : ℕ) : Prop :=
  let digits := [n / 10, n % 10] in
  ∀ d ∈ digits, is_power_less_digit d

-- The smallest power-less two-digit number
def smallest_power_less : ℕ := 22

-- The largest power-less two-digit number
def largest_power_less : ℕ := 77

-- Proof statement
theorem common_divisor_of_power_less_numbers :
  ∃ d, d ∣ smallest_power_less ∧ d ∣ largest_power_less ∧ d = 11 :=
sorry

end common_divisor_of_power_less_numbers_l724_724204


namespace pages_torn_l724_724910

theorem pages_torn (n : ℕ) (H1 : n = 185) (H2 : ∃ m, m = 518 ∧ (digits 10 m = digits 10 n) ∧ (m % 2 = 0)) : 
  ∃ k, k = ((518 - 185 + 1) / 2) ∧ k = 167 :=
by sorry

end pages_torn_l724_724910


namespace order_of_a_b_c_l724_724826

noncomputable def x_interval := {x : ℝ // 0 < x ∧ x < (π / 4) }

def a (x : x_interval) (r: ℝ) : ℝ := Real.cos ((x.1) ^ (Real.sin ((r) ^ (Real.sin (x.1)))))
def b (x : x_interval) : ℝ := Real.sin ((x.1) ^ (Real.cos ((x.1) ^ (Real.sin (x.1)))))
def c (x : x_interval) : ℝ := Real.cos ((x.1) ^ (Real.sin ((x.1) ^ (Real.sin (x.1)))))

theorem order_of_a_b_c (x : x_interval) (r: ℝ) : b x < a x r ∧ a x r < c x :=
sorry

end order_of_a_b_c_l724_724826


namespace reciprocal_of_neg_three_l724_724158

theorem reciprocal_of_neg_three : ∃ (x : ℚ), (-3 * x = 1) ∧ (x = -1 / 3) :=
by
  use (-1 / 3)
  split
  . rw [mul_comm]
    norm_num 
  . norm_num

end reciprocal_of_neg_three_l724_724158


namespace profit_from_first_purchase_functional_relationship_W_profit_comparison_l724_724239

section
variables (x y m : ℕ)

-- Define conditions for the first purchase
def first_purchase_conditions : Prop :=
  x + y = 120 ∧ 45 * x + 60 * y = 6000 ∧ x = 80 ∧ y = 40

-- Profit from the first purchase
def first_purchase_profit : ℕ :=
  (66 - 45) * 80 + (90 - 60) * 40

theorem profit_from_first_purchase
    (h : first_purchase_conditions x y) :
    first_purchase_profit = 2880 :=
  by sorry

-- Define conditions for the second purchase
def second_purchase_conditions : Prop :=
  50 ≤ m ∧ m ≤ 150 ∧ 150 - m ≤ 2 * m

-- Functional relationship between W and m
def W (m : ℕ) : ℕ :=
  -4 * m + 3000

theorem functional_relationship_W
    (h : second_purchase_conditions m) :
    W m = -4 * m + 3000 :=
  by sorry

-- Maximum profit from the second purchase
def max_profit_from_second_purchase : ℕ :=
  W 50

-- Comparison of the profits
theorem profit_comparison :
    max_profit_from_second_purchase ≤ first_purchase_profit :=
  by sorry
end

end profit_from_first_purchase_functional_relationship_W_profit_comparison_l724_724239


namespace radius_of_inscribed_circle_l724_724244

theorem radius_of_inscribed_circle (A1 A2 : ℝ) (r1 r2 : ℝ) 
  (h1 : r1 = 8)
  (h2 : A1 + A2 = π * r1^2)
  (h3 : A2 = (A1 + (A1 + A2)) / 2) :
  r2 = 8 * real.sqrt 3 / 3 :=
by
  sorry

end radius_of_inscribed_circle_l724_724244


namespace log_identity_l724_724991

theorem log_identity : 2 * log 5 10 + log 5 (1 / 4) = 2 :=
by
  sorry

end log_identity_l724_724991


namespace diameter_of_circle_given_radius_l724_724424

theorem diameter_of_circle_given_radius (radius: ℝ) (h: radius = 7): 
  2 * radius = 14 :=
by
  rw [h]
  sorry

end diameter_of_circle_given_radius_l724_724424


namespace tan_double_angle_l724_724385

theorem tan_double_angle 
  (x : ℝ) 
  (hx1 : x ∈ Ioo (-π/2) 0) 
  (hcos : Real.cos x = 3/5) : 
  Real.tan (2 * x) = 24 / 7 := 
by 
  sorry

end tan_double_angle_l724_724385


namespace problem_1_problem_2_l724_724411

open Real

noncomputable def vec_a (θ : ℝ) : ℝ × ℝ :=
( sin θ, cos θ - 2 * sin θ )

def vec_b : ℝ × ℝ :=
( 1, 2 )

theorem problem_1 (θ : ℝ) (h : (cos θ - 2 * sin θ) / sin θ = 2) : tan θ = 1 / 4 :=
by {
  sorry
}

theorem problem_2 (θ : ℝ) (h1 : sin θ ^ 2 + (cos θ - 2 * sin θ) ^ 2 = 5) (h2 : 0 < θ) (h3 : θ < π) : θ = π / 2 ∨ θ = 3 * π / 4 :=
by {
  sorry
}

end problem_1_problem_2_l724_724411


namespace broken_shells_leftover_l724_724108

-- Definitions for the conditions
def bags : List ℕ := [17, 20, 22, 24, 26, 36]

-- Question and given conditions reformulated as a Lean theorem
theorem broken_shells_leftover :
  ∃ (x : ℕ), (b : ℕ) (b ∈ bags) ∧ 
  (145 - b) % 4 = 0 ∧ 
  ∃ d_c_bags : List ℕ × List ℕ, 
    d_c_bags.1 ⊆ bags.erase b ∧
    d_c_bags.2 ⊆ bags.erase b ∧ 
    (d_c_bags.1.card = 2 ∧ d_c_bags.2.card = 3) ∧
    (d_c_bags.2.sum = 3 * d_c_bags.1.sum) ∧
    b = 17 :=
  sorry

end broken_shells_leftover_l724_724108


namespace sqrt_minimum_value_l724_724572

theorem sqrt_minimum_value (x : ℝ) (h : sqrt (x - 1) ≥ 0) : x = 1 ↔ sqrt (x - 1) = 0 :=
by
  sorry

end sqrt_minimum_value_l724_724572


namespace gcd_102_238_l724_724192

def gcd (a b : ℕ) : ℕ := if b = 0 then a else gcd b (a % b)

theorem gcd_102_238 : gcd 102 238 = 34 :=
by
  sorry

end gcd_102_238_l724_724192


namespace trigonometric_expression_value_l724_724703

theorem trigonometric_expression_value
  (α : ℝ)
  (h : ∃ x : ℝ, (sin (α) = x ∧ (5 * x^2 - 7 * x - 6 = 0))) :
  (sin (-α - 3 * π / 2) * sin (3 * π / 2 - α) * tan (2 * π - α)^2) /
  (cos (π / 2 - α) * cos (π / 2 + α) * cos (π - α)^2) = 25 / 16 := 
sorry

end trigonometric_expression_value_l724_724703


namespace perimeter_triangle_ADA_l724_724356

open Real

noncomputable def eccentricity : ℝ := 1 / 2

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

noncomputable def foci_distance (a b : ℝ) : ℝ :=
  (a^2 - b^2).sqrt

noncomputable def line_passing_through_focus_perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  sorry

noncomputable def distance_de (d e : ℝ) : ℝ := 6

theorem perimeter_triangle_ADA
  (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : foci_distance a b = c)
  (h4 : eccentricity * a = c) (h5 : distance_de 6 6) :
  4 * a = 13 :=
by sorry

end perimeter_triangle_ADA_l724_724356


namespace ryan_learning_hours_l724_724304

theorem ryan_learning_hours (total_hours : ℕ) (chinese_hours : ℕ) (english_hours : ℕ) 
  (h1 : total_hours = 3) (h2 : chinese_hours = 1) : 
  english_hours = 2 :=
by 
  sorry

end ryan_learning_hours_l724_724304


namespace dogwood_tree_count_l724_724547

def initial_dogwoods : ℕ := 34
def additional_dogwoods : ℕ := 49
def total_dogwoods : ℕ := initial_dogwoods + additional_dogwoods

theorem dogwood_tree_count :
  total_dogwoods = 83 :=
by
  -- omitted proof
  sorry

end dogwood_tree_count_l724_724547


namespace canIdentifyOriginalCode_totalNumberOfCodes_codesNotContainingFifteen_l724_724985

namespace AgriculturalCooperative

-- Definition for the sub-region, producer, and product type
abbreviation SubRegion := Fin 27
abbreviation Producer := Fin 40
abbreviation ProductType := Fin 28

-- Definition for a ProductCode, combining the above
structure ProductCode where
  subRegion : SubRegion
  producer : Producer
  productType : ProductType

-- Part (a) proof statement: Given a code, we can derive its parts if rearranged
theorem canIdentifyOriginalCode (code : Nat) (h₁ : code = 900950) :
  ∃ (sr : SubRegion) (pd : Producer) (pt : ProductType),
    code = (sr.val + 1) * 100000 + (pd.val + 31) * 100 + (pt.val + 71) := by
  sorry

-- Part (b) proof statement: Calculating the total number of different codes
theorem totalNumberOfCodes : 27 * 40 * 28 = 30240 := by
  norm_num

-- Part (c) proof statement: Counting codes with and without the sequence "15"
theorem codesNotContainingFifteen : 30240 - (1120 + 756) = 28320 := by
  norm_num

end AgriculturalCooperative

end canIdentifyOriginalCode_totalNumberOfCodes_codesNotContainingFifteen_l724_724985


namespace minimum_l_condition_l724_724476

def f (x : ℝ) : ℝ :=
if |x| ≤ 1 then 2 * Real.cos (π / 2 * x)
else x^2 - 1

def g (x l : ℝ) : ℝ :=
|f x + f (x + l) - 2| + |f x - f (x + l)|

theorem minimum_l_condition (l : ℝ) (h : l > 0) :
  (∀ x : ℝ, g x l ≥ 2) ↔ l ≥ 2 * Real.sqrt 3 :=
sorry

end minimum_l_condition_l724_724476


namespace eccentricity_of_ellipse_l724_724694

theorem eccentricity_of_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0)
    (h3 : ∀ x y : ℝ, x ≠ y ∧ x^2 a^2 + y^2 b^2 = 1)
    (h4 : (F1 F2 A : ℝ)  a = sqrt(24/7 c)): real.sqrt(3)/6 := 
sorry    

end eccentricity_of_ellipse_l724_724694


namespace carl_took_4_pink_hard_hats_l724_724776

-- Define the initial number of hard hats
def initial_pink : ℕ := 26
def initial_green : ℕ := 15
def initial_yellow : ℕ := 24

-- Define the number of hard hats John took
def john_pink : ℕ := 6
def john_green : ℕ := 2 * john_pink
def john_total : ℕ := john_pink + john_green

-- Define the total initial number of hard hats
def total_initial : ℕ := initial_pink + initial_green + initial_yellow

-- Define the number of hard hats remaining after John's removal
def remaining_after_john : ℕ := total_initial - john_total

-- Define the total number of hard hats that remained in the truck
def total_remaining : ℕ := 43

-- Define the number of pink hard hats Carl took away
def carl_pink : ℕ := remaining_after_john - total_remaining

-- State the proof problem
theorem carl_took_4_pink_hard_hats : carl_pink = 4 := by
  sorry

end carl_took_4_pink_hard_hats_l724_724776


namespace max_possible_value_b_l724_724648

theorem max_possible_value_b : 
  ∃ b : ℚ, 
  (∀ m : ℚ, (1/3 : ℚ) < m → m < b → ∀ x : ℤ, 1 ≤ x → x ≤ 150 → ∃ y : ℤ, y ≠ m * x + 3) ∧ 
  b = 50 / 149 :=
sorry

end max_possible_value_b_l724_724648


namespace lcm_product_is_perfect_square_l724_724035

theorem lcm_product_is_perfect_square
  (a b c : ℤ)
  (hpos_a : 0 < a)
  (hpos_b : 0 < b)
  (hpos_c : 0 < c)
  (hgcd_square : ∃ k : ℤ, k^2 = gcd a b * gcd b c * gcd c a) :
  ∃ m : ℤ, m^2 = lcm a b * lcm b c * lcm c a :=
begin
  sorry
end

end lcm_product_is_perfect_square_l724_724035


namespace at_least_one_not_less_than_one_l724_724690

theorem at_least_one_not_less_than_one (x : ℝ) (a b c : ℝ) 
  (ha : a = x^2 + 1/2) 
  (hb : b = 2 - x) 
  (hc : c = x^2 - x + 1) : 
  (1 ≤ a) ∨ (1 ≤ b) ∨ (1 ≤ c) := 
sorry

end at_least_one_not_less_than_one_l724_724690


namespace nails_remaining_l724_724252

theorem nails_remaining (nails_initial : ℕ) (kitchen_fraction : ℚ) (fence_fraction : ℚ) (nails_used_kitchen : ℕ) (nails_remaining_after_kitchen : ℕ) (nails_used_fence : ℕ) (nails_remaining_final : ℕ) 
  (h1 : nails_initial = 400) 
  (h2 : kitchen_fraction = 0.30) 
  (h3 : nails_used_kitchen = kitchen_fraction * nails_initial) 
  (h4 : nails_remaining_after_kitchen = nails_initial - nails_used_kitchen) 
  (h5 : fence_fraction = 0.70) 
  (h6 : nails_used_fence = fence_fraction * nails_remaining_after_kitchen) 
  (h7 : nails_remaining_final = nails_remaining_after_kitchen - nails_used_fence) :
  nails_remaining_final = 84 := by
sorry

end nails_remaining_l724_724252


namespace length_of_other_train_l724_724576

noncomputable def speed_in_mps (speed_kmph : ℕ) : ℝ := (speed_kmph.to_real * 1000) / 3600

theorem length_of_other_train
  (l1 : ℝ) (s1 : ℕ) (s2 : ℕ) (t : ℝ)
  (h1 : l1 = 150) (h2 : s1 = 120) (h3 : s2 = 80) (h4 : t = 9) :
  ∃ l2 : ℝ, l2 = 349.95 :=
by
  let s1_mps := speed_in_mps s1
  let s2_mps := speed_in_mps s2
  let relative_speed := s1_mps + s2_mps
  let total_distance := relative_speed * t
  let l2 := total_distance - l1
  have h_result : l2 = 349.95 := sorry
  exact ⟨349.95, h_result⟩

end length_of_other_train_l724_724576


namespace parking_cost_proof_l724_724867

variables 
  (C : ℝ) -- Cost for up to 2 hours of parking
  (avgCost : ℝ) -- Given average cost per hour for 9 hours
  (extraCost : ℝ) -- Cost for additional hours beyond the first 2 hours
  (hours : ℕ) -- Total hours parked

-- Given: extraCost for each additional hour beyond the first 2 hours
def extraCost := 1.75

-- Given: average cost per hour for 9 hours of parking
def avgCost := 2.6944444444444446

-- Given: total hours parked
def hours := 9

-- The proof statement
theorem parking_cost_proof : 
  (avgCost = (C + extraCost * (hours - 2)) / hours) →
  (C = 12) :=
by
  sorry

end parking_cost_proof_l724_724867


namespace average_value_s7_squared_l724_724233

-- Definition of sum of digits in base 7
def s7 (n : ℕ) : ℕ :=
  (n.digits 7).sum

-- Lean statement of the problem
theorem average_value_s7_squared :
  (∑ n in Finset.range (7^20), (s7 n)^2 : ℚ) / 7^20 = 3680 / 7^20 :=
by
  sorry

end average_value_s7_squared_l724_724233


namespace perimeter_of_triangle_ADE_l724_724349

noncomputable def ellipse (x y a b : ℝ) : Prop :=
  (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1

def foci_distance (a : ℝ) : ℝ := a / 2

def line_through_f1_perpendicular_to_af2 (x y c : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * (x + c)

def distance_between_points (x1 x2 y1 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem perimeter_of_triangle_ADE
  (a b c : ℝ)
  (h_ellipse : ∀ (x y : ℝ), ellipse x y a b)
  (h_eccentricity : b = Real.sqrt 3 * c)
  (h_foci_distance : foci_distance a = c)
  (h_line : ∀ (x y : ℝ), line_through_f1_perpendicular_to_af2 x y c)
  (h_DE : ∀ (x1 y1 x2 y2 : ℝ), distance_between_points x1 x2 y1 y2 = 6) :
  perimeter_of_triangle_ADE = 13 := sorry

end perimeter_of_triangle_ADE_l724_724349


namespace petals_per_ounce_l724_724023

-- Definitions of the given conditions
def petals_per_rose : ℕ := 8
def roses_per_bush : ℕ := 12
def bushes_harvested : ℕ := 800
def bottles_produced : ℕ := 20
def ounces_per_bottle : ℕ := 12

-- Calculation of petals per bush
def petals_per_bush : ℕ := roses_per_bush * petals_per_rose

-- Calculation of total petals harvested
def total_petals_harvested : ℕ := bushes_harvested * petals_per_bush

-- Calculation of total ounces of perfume
def total_ounces_produced : ℕ := bottles_produced * ounces_per_bottle

-- Main theorem statement
theorem petals_per_ounce : total_petals_harvested / total_ounces_produced = 320 :=
by
  sorry

end petals_per_ounce_l724_724023


namespace max_elements_union_l724_724587

open Set

theorem max_elements_union (A B : Set ℕ) (hA : ∀ n ∈ A, n ≤ 100)
  (hB : ∀ n ∈ B, n ≤ 100) (h_disjoint : A ∩ B = ∅) (h_size : A.card = B.card)
  (h_condition : ∀ n ∈ A, 2 * n + 2 ∈ B) :
  (A ∪ B).card = 66 := sorry

end max_elements_union_l724_724587


namespace kenny_books_l724_724807

def lawns_mowed := 35
def charge_per_lawn := 15
def video_game_price := 45
def book_price := 5
def desired_video_games := 5

theorem kenny_books :
  let total_earnings := lawns_mowed * charge_per_lawn in
  let total_video_game_cost := desired_video_games * video_game_price in
  let remaining_money := total_earnings - total_video_game_cost in
  let num_books := remaining_money / book_price in
  num_books = 60 :=
by
  sorry

end kenny_books_l724_724807


namespace parabola_circle_directrix_segment_length_l724_724532

theorem parabola_circle_directrix_segment_length (p : ℝ) (h : p > 0) 
  (circle_center : (-1 : ℝ, 0)) (radius : ℝ := 2)
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 + 2 * x - 3 = 0 ↔ (x + 1)^2 + y^2 = 4)
  (segment_length : ℝ := 4)
  (parabola_directrix : ℝ := -p / 2) :
  segment_length = 4 → circle_center.1 = parabola_directrix → p = 2 :=
by
  sorry

end parabola_circle_directrix_segment_length_l724_724532


namespace simplified_fraction_of_num_l724_724960

def num : ℚ := 368 / 100

theorem simplified_fraction_of_num : num = 92 / 25 := by
  sorry

end simplified_fraction_of_num_l724_724960


namespace num_new_students_l724_724864

-- Definitions based on the provided conditions
def original_class_strength : ℕ := 10
def original_average_age : ℕ := 40
def new_students_avg_age : ℕ := 32
def decrease_in_average_age : ℕ := 4
def new_average_age : ℕ := original_average_age - decrease_in_average_age
def new_class_strength (n : ℕ) : ℕ := original_class_strength + n

-- The proof statement
theorem num_new_students (n : ℕ) :
  (original_class_strength * original_average_age + n * new_students_avg_age) 
  = new_class_strength n * new_average_age → n = 10 :=
by
  sorry

end num_new_students_l724_724864


namespace arithmetic_sequence_a_100_l724_724448

theorem arithmetic_sequence_a_100 :
  ∀ (a : ℕ → ℕ), 
  (a 1 = 100) → 
  (∀ n : ℕ, a (n + 1) = a n + 2) → 
  a 100 = 298 :=
by
  intros a h1 hrec
  sorry

end arithmetic_sequence_a_100_l724_724448


namespace minimum_value_of_expression_l724_724584

-- Definitions based on the problem conditions
def expr (x y : ℝ) : ℝ :=
  (3 * real.sqrt (2 * (1 + real.cos 2 * x)) - real.sqrt (8 - 4 * real.sqrt 3) * real.sin x + 2) *
  (3 + 2 * real.sqrt (11 - real.sqrt 3) * real.cos y - real.cos (2 * y))

-- Statement of the proof problem
theorem minimum_value_of_expression :
  ∃ m : ℝ, ∀ x y : ℝ, expr x y ≥ -33 ∧ m = -33 :=
sorry

end minimum_value_of_expression_l724_724584


namespace tangent_line_eq_min_value_a_l724_724723

-- Question 1: Tangent line equation verification
theorem tangent_line_eq (f : ℝ → ℝ) (h : ∀ x, f x = 2 * log x - 3 * x^2 - 11 * x) :
  ∃ (m : ℝ) (c : ℝ), (f' : ℝ → ℝ) (h' : ∀ x, f' x = deriv f x) (m = f' 1) ∧ (f 1 = -14) ∧ 
  ∀ x, y = f x → (15 * x + y - 1 = 0) :=
sorry

-- Question 2: Minimum value of a
theorem min_value_a (f : ℝ → ℝ) (h : ∀ x ∈ Ioi 0, f x = 2 * log x - 3 * x^2 - 11 * x) :
  ∀ (a : ℤ), (∀ x ∈ Ioi 0, f x ≤ (a - 3) * x^2 + (2 * a - 13) * x + 1) ↔ a ≥ 1 :=
sorry

end tangent_line_eq_min_value_a_l724_724723


namespace bob_baked_more_cookies_l724_724630

theorem bob_baked_more_cookies (alice_init : Nat) (bob_init : Nat) (cookies_lost : Nat)
  (alice_add : Nat) (total_end : Nat) :
  alice_init = 74 → bob_init = 7 →
  cookies_lost = 29 → alice_add = 5 →
  total_end = 93 →
  bob_init + alice_add + 36 = total_end - cookies_lost :=
by
  intros h_alice_init h_bob_init h_cookies_lost h_alice_add h_total_end
  rw [h_alice_init, h_bob_init, h_cookies_lost, h_alice_add, h_total_end]
  sorry

end bob_baked_more_cookies_l724_724630


namespace yellow_not_greater_than_green_l724_724552

theorem yellow_not_greater_than_green
    (G Y S : ℕ)
    (h1 : G + Y + S = 100)
    (h2 : G + S / 2 = 50)
    (h3 : Y + S / 2 = 50) : ¬ Y > G :=
sorry

end yellow_not_greater_than_green_l724_724552


namespace reciprocal_of_neg3_l724_724146

theorem reciprocal_of_neg3 : ∃ x : ℝ, (-3 : ℝ) * x = 1 :=  
begin
  use -1/3,
  norm_num,
end

end reciprocal_of_neg3_l724_724146


namespace ratio_of_b_to_a_l724_724378

open Real

theorem ratio_of_b_to_a (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a * sin (π / 5) + b * cos (π / 5)) / (a * cos (π / 5) - b * sin (π / 5)) = tan (8 * π / 15) 
  → b / a = sqrt 3 :=
by
  intro h
  sorry

end ratio_of_b_to_a_l724_724378


namespace original_denominator_is_two_l724_724621

theorem original_denominator_is_two (d : ℕ) : 
  (∃ d : ℕ, 2 * (d + 4) = 6) → d = 2 :=
by sorry

end original_denominator_is_two_l724_724621


namespace pages_torn_l724_724911

theorem pages_torn (n : ℕ) (H1 : n = 185) (H2 : ∃ m, m = 518 ∧ (digits 10 m = digits 10 n) ∧ (m % 2 = 0)) : 
  ∃ k, k = ((518 - 185 + 1) / 2) ∧ k = 167 :=
by sorry

end pages_torn_l724_724911


namespace no_power_of_two_divides_3n_plus_1_l724_724507

theorem no_power_of_two_divides_3n_plus_1 (n : ℕ) (hn : n > 1) : ¬ (2^n ∣ 3^n + 1) := sorry

end no_power_of_two_divides_3n_plus_1_l724_724507


namespace pentagon_area_l724_724649

theorem pentagon_area {a b c d e : ℕ} (split: ℕ) (non_parallel1 non_parallel2 parallel1 parallel2 : ℕ)
  (h1 : a = 16) (h2 : b = 25) (h3 : c = 30) (h4 : d = 26) (h5 : e = 25)
  (split_condition : a + b + c + d + e = 5 * split)
  (np_condition1: non_parallel1 = c) (np_condition2: non_parallel2 = a)
  (p_condition1: parallel1 = d) (p_condition2: parallel2 = e)
  (area_triangle: 1 / 2 * b * a = 200)
  (area_trapezoid: 1 / 2 * (parallel1 + parallel2) * non_parallel1 = 765) :
  a + b + c + d + e = 965 := by
  sorry

end pentagon_area_l724_724649


namespace exists_constants_C1_C2_l724_724675

def sum_of_digits (m : ℕ) : ℕ := sorry
def f (n : ℕ) : ℕ := sorry

theorem exists_constants_C1_C2 : ∃ (C1 C2 : ℝ), 
  0 < C1 ∧ C1 < C2 ∧ C1 = 9 / 2 ∧ C2 = 45 ∧
  ∀ (n : ℕ), n ≥ 2 → C1 * Real.log10 n < f n ∧ f n < C2 * Real.log10 n := sorry

end exists_constants_C1_C2_l724_724675


namespace candy_left_l724_724183

theorem candy_left (total_candy : ℕ) (eaten_per_person : ℕ) (number_of_people : ℕ)
  (h_total_candy : total_candy = 68)
  (h_eaten_per_person : eaten_per_person = 4)
  (h_number_of_people : number_of_people = 2) :
  total_candy - (eaten_per_person * number_of_people) = 60 :=
by
  sorry

end candy_left_l724_724183


namespace tangents_intersect_or_parallel_l724_724262

variables {K : Type*} [Field K] [ProperSphere K]

-- We represent points in the 2D Euclidean plane
structure Point (K : Type*) :=
(x : K)
(y : K)

-- Circle definition
structure Circle (K : Type*) :=
(center : Point K)
(radius : K)

-- Tangent lines to a circle at a given point
noncomputable def tangent_line (c : Circle K) (p : Point K) : Line K := sorry


-- Quadrilateral inscribed in a circle
structure CyclicQuadrilateral (K : Type*) :=
(A B C D : Point K)
(circle : Circle K)

-- Definition for intersection of lines
noncomputable def lines_intersect (l1 l2 : Line K) : Prop := sorry

-- Definition for whether two lines are parallel
noncomputable def lines_parallel (l1 l2 : Line K) : Prop := sorry

-- Main theorem
theorem tangents_intersect_or_parallel
    {K : Type*} [Field K] [ProperSphere K]
    (quad : CyclicQuadrilateral K)
    (h_tangent_AC : lines_intersect (tangent_line quad.circle quad.A) (tangent_line quad.circle quad.C))
    (h_AC_intersect_BD : lines_intersect (tangent_line quad.circle quad.A) (quad.B, quad.D)) :
    lines_intersect (tangent_line quad.circle quad.B) (tangent_line quad.circle quad.D) ∨ 
    lines_parallel (tangent_line quad.circle quad.B) (tangent_line quad.circle quad.D) :=
sorry

end tangents_intersect_or_parallel_l724_724262


namespace probability_smallest_divides_l724_724558

open Finset

theorem probability_smallest_divides 
  (S : Finset ℕ := {1, 2, 3, 4, 5, 6}) 
  (choosing_fun : Finset ℕ → Finset (Finset ℕ) := λ s, filter (λ t, 3 = card t) (powerset s)) 
  (A : Finset (Finset ℕ) := filter (λ t, ∃ a b c, t = {a, b, c} ∧ a < b ∧ a < c ∧ b % a = 0 ∧ c % a = 0) (choosing_fun S)) :
  (A.card : ℚ) / (choosing_fun S).card = 11 / 20 := 
begin
  sorry
end

end probability_smallest_divides_l724_724558


namespace hyperbola_sum_l724_724770

noncomputable def hyperbola_properties
(center focus vertex : ℝ × ℝ) : ℝ :=
  let h := center.1
  let k := center.2
  let a := abs (vertex.2 - k)
  let c := abs (focus.2 - k)
  let b := real.sqrt (c^2 - a^2)
  h + k + a + b

theorem hyperbola_sum
  (center focus vertex : ℝ × ℝ)
  (h k a b : ℝ)
  (center_eq : center = (0, 2))
  (focus_eq : focus = (0, 8))
  (vertex_eq : vertex = (0, -1))
  (hx : h = center.1)
  (ky : k = center.2)
  (a_is : a = abs (vertex.2 - k))
  (c_is : c = abs (focus.2 - k))
  (b_is : b = real.sqrt (c^2 - a^2)) : 
  h + k + a + b = 3 * real.sqrt 3 + 5 :=
by 
  sorry

end hyperbola_sum_l724_724770


namespace gcd_102_238_is_34_l724_724200

noncomputable def gcd_102_238 : ℕ :=
  Nat.gcd 102 238

theorem gcd_102_238_is_34 : gcd_102_238 = 34 := by
  -- Conditions based on the Euclidean algorithm
  have h1 : 238 = 2 * 102 + 34 := by norm_num
  have h2 : 102 = 3 * 34 := by norm_num
  have h3 : Nat.gcd 102 34 = 34 := by
    rw [Nat.gcd, Nat.gcd_rec]
    exact Nat.gcd_eq_left h2

  -- Conclusion
  show gcd_102_238 = 34 from
    calc gcd_102_238 = Nat.gcd 102 238 : rfl
                  ... = Nat.gcd 34 102 : Nat.gcd_comm 102 34
                  ... = Nat.gcd 34 (102 % 34) : by rw [Nat.gcd_rec]
                  ... = Nat.gcd 34 34 : by rw [Nat.mod_eq_of_lt (by norm_num : 34 < 102)]
                  ... = 34 : Nat.gcd_self 34

end gcd_102_238_is_34_l724_724200


namespace hypotenuse_variance_incorrect_min_standard_deviation_legs_l724_724987

-- Part 1: Proving that the variance cannot be 2 if the hypotenuse is 3
theorem hypotenuse_variance_incorrect (a b : ℝ) 
    (hypotenuse_eq_3 : ∀ (a b : ℝ), a^2 + b^2 = 9)
    (variance_claimed : real) : variance_claimed ≠ 2 :=
by
    sorry

-- Part 2: Finding the minimum standard deviation of the side lengths given hypotenuse is 3
theorem min_standard_deviation_legs (a b : ℝ)
    (hypotenuse_eq_3 : ∀ (a b : ℝ), a^2 + b^2 = 9) : 
    let stddev_min := real.sqrt(2) - 1 in
    let leg_length := (3 * real.sqrt(2)) / 2 in
    standard_deviation = stddev_min ∧ a = leg_length ∧ b = leg_length :=
by
    sorry

end hypotenuse_variance_incorrect_min_standard_deviation_legs_l724_724987


namespace pages_sum_to_35_l724_724435

theorem pages_sum_to_35 (pages : List ℕ) (digits : List ℕ) :
  pages.sum = 35 ∧
  pages.length = 4 ∧
  ∃ evens odds, 
    (evens.length = 2 ∧ odds.length = 2 ∧ 
     evens.sum + odds.sum = 35 ∧ 
     evens.all (λ x => even x) ∧ odds.all (λ x => odd x)) ∧ 
  pages = evens ++ odds ∧ 
  digits = [0, 1, 1, 3, 3, 3, 4, 5, 6, 9] →
  pages = [59, 60, 133, 134] :=
by sorry

end pages_sum_to_35_l724_724435


namespace find_N_values_l724_724668

theorem find_N_values (N : ℕ) : 
  (∀ p : Nat.Prime, (Nat.Prime.divisors N).All (λ q, q = 2 ∨ q = 5)) →
  (∃ k : ℕ, N + 25 = k^2) → 
  (N = 200 ∨ N = 2000) := 
sorry

end find_N_values_l724_724668


namespace jenny_kenny_see_again_l724_724024

open Real

-- Defining the conditions
def jenny_path (t : ℝ) : ℝ × ℝ := (-75 + 2 * t, 150)
def kenny_path (t : ℝ) : ℝ × ℝ := (-75 + 4 * t, -150)
def circle_eq (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 75 ^ 2

-- Defining the tangency condition for the line-of-sight
def tangent_condition (t x y : ℝ) : Prop := x * t = 150 * y

-- Define the problem statement
theorem jenny_kenny_see_again : ∃ t : ℝ, t = 48 ∧ ∀x y, 
  (jenny_path t).fst - (kenny_path t).fst = 150 / t * (jenny_path t).snd - (kenny_path t).snd + 300 - 11250 / t / distance(jenny_path t, kenny_path t) = x ∧
  x = 11250 / sqrt (150 ^ 2 + t ^ 2) ∧ 
  height (circle_eq x y) ∧
  tangent_condition t x y := 
begin
  sorry
end

end jenny_kenny_see_again_l724_724024


namespace baskets_delivered_l724_724553

theorem baskets_delivered 
  (peaches_per_basket : ℕ := 25)
  (boxes : ℕ := 8)
  (peaches_per_box : ℕ := 15)
  (peaches_eaten : ℕ := 5)
  (peaches_in_boxes := boxes * peaches_per_box) 
  (total_peaches := peaches_in_boxes + peaches_eaten) : 
  total_peaches / peaches_per_basket = 5 :=
by
  sorry

end baskets_delivered_l724_724553


namespace perimeter_of_triangle_ADE_l724_724365

noncomputable def ellipse_perimeter (a b : ℝ) (h : a > b) (e : ℝ) (he : e = 1/2) (h_ellipse : ∀ (x y : ℝ), 
                            x^2 / a^2 + y^2 / b^2 = 1) : ℝ :=
13 -- we assert that the perimeter is 13

theorem perimeter_of_triangle_ADE 
  (a b : ℝ) (h : a > b) (e : ℝ) (he : e = 1/2) 
  (C_eq : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1) 
  (upper_vertex_A : ℝ × ℝ)
  (focus_F1 : ℝ × ℝ)
  (focus_F2 : ℝ × ℝ)
  (line_through_F1_perpendicular_to_AF2 : ∀ x y, y = (√3 / 3) * (x + focus_F1.1))
  (points_D_E_on_ellipse : ∃ D E : ℝ × ℝ, line_through_F1_perpendicular_to_AF2 D.1 D.2 = true ∧
    line_through_F1_perpendicular_to_AF2 E.1 E.2 = true ∧ 
    (dist D E = 6)) :
  ∃ perimeter : ℝ, perimeter = ellipse_perimeter a b h e he C_eq :=
sorry

end perimeter_of_triangle_ADE_l724_724365


namespace baseball_team_grouping_l724_724989

theorem baseball_team_grouping (new_players returning_players : ℕ) (group_size : ℕ) 
  (h_new : new_players = 4) (h_returning : returning_players = 6) (h_group : group_size = 5) : 
  (new_players + returning_players) / group_size = 2 := 
  by 
  sorry

end baseball_team_grouping_l724_724989


namespace lucy_cardinals_vs_blue_jays_l724_724477

noncomputable def day1_cardinals : ℕ := 3
noncomputable def day1_blue_jays : ℕ := 2
noncomputable def day2_cardinals : ℕ := 3
noncomputable def day2_blue_jays : ℕ := 3
noncomputable def day3_cardinals : ℕ := 4
noncomputable def day3_blue_jays : ℕ := 2

theorem lucy_cardinals_vs_blue_jays :
  (day1_cardinals + day2_cardinals + day3_cardinals) - (day1_blue_jays + day2_blue_jays + day3_blue_jays) = 3 :=
  by sorry

end lucy_cardinals_vs_blue_jays_l724_724477


namespace coefficient_of_x_sq_in_expansion_l724_724122

theorem coefficient_of_x_sq_in_expansion (x : ℝ) :
  (∃ c : ℝ, c * x^2 = ((x - 2)^4).coeff 2) :=
sorry

end coefficient_of_x_sq_in_expansion_l724_724122


namespace geometric_inequality_first_part_geometric_inequality_second_part_l724_724809

variables {R : ℝ}
variables {A B C : ℝ × ℝ}

-- Conditions that A, B, and C are points with integer coordinates
def integer_coordinates (P : ℝ × ℝ): Prop := 
  ∃ (x y : ℤ), P = (x, y)

-- Condition that a circle K with radius R passes through A, B, and C
def circle_passing_through (A B C : ℝ × ℝ) (R : ℝ) : Prop := 
  let dist := λ p q : ℝ × ℝ, (p.1 - q.1)^2 + (p.2 - q.2)^2 in
  dist A B = R^2 ∧ dist B C = R^2 ∧ dist C A = R^2

-- First part: Prove that AB * BC * CA ≥ 2R
theorem geometric_inequality_first_part (hA : integer_coordinates A) (hB : integer_coordinates B) 
(hC : integer_coordinates C) (hK : circle_passing_through A B C R) : 
  let dist := λ p q : ℝ × ℝ, (p.1 - q.1)^2 + (p.2 - q.2)^2 in
  dist A B * dist B C * dist C A ≥ 2 * R := 
sorry

-- Second part: Prove that AB * BC * CA ≥ 4R if the center is at the origin
def center_at_origin (A B C : ℝ × ℝ) (R : ℝ) : Prop := 
  let origin := (0 : ℝ, 0 : ℝ) in
  ∀ P ∈ {A, B, C}, (P.1^2 + P.2^2 = R^2)

theorem geometric_inequality_second_part (hA : integer_coordinates A) (hB : integer_coordinates B) 
(hC : integer_coordinates C) (hK : circle_passing_through A B C R) (hO : center_at_origin A B C R) : 
  let dist := λ p q : ℝ × ℝ, (p.1 - q.1)^2 + (p.2 - q.2)^2 in
  dist A B * dist B C * dist C A ≥ 4 * R := 
sorry

end geometric_inequality_first_part_geometric_inequality_second_part_l724_724809


namespace cubic_polynomial_a_value_l724_724426

theorem cubic_polynomial_a_value (a b c d y₁ y₂ : ℝ)
  (h₁ : y₁ = a + b + c + d)
  (h₂ : y₂ = -a + b - c + d)
  (h₃ : y₁ - y₂ = -8) : a = -4 :=
by
  sorry

end cubic_polynomial_a_value_l724_724426


namespace candy_left_l724_724184

theorem candy_left (total_candy : ℕ) (eaten_per_person : ℕ) (number_of_people : ℕ)
  (h_total_candy : total_candy = 68)
  (h_eaten_per_person : eaten_per_person = 4)
  (h_number_of_people : number_of_people = 2) :
  total_candy - (eaten_per_person * number_of_people) = 60 :=
by
  sorry

end candy_left_l724_724184


namespace square_division_possible_l724_724451

theorem square_division_possible :
  ∃ (S a b c : ℕ), 
    S^2 = a^2 + 3 * b^2 + 5 * c^2 ∧ 
    a = 3 ∧ 
    b = 2 ∧ 
    c = 1 :=
  by {
    sorry
  }

end square_division_possible_l724_724451


namespace solve_x1_x2_x3_x4_l724_724327

variable {x1 x2 x3 x4 : ℝ}

def conditions (x1 x2 x3 x4 : ℝ) :=
  ∀ i ∈ {0, 1, 2, 3}, 
    let xi := [x1, x2, x3, x4].get i in
    xi + x1 * x2 * x3 * x4 / xi = 2

theorem solve_x1_x2_x3_x4 (x1 x2 x3 x4 : ℝ) (h : conditions x1 x2 x3 x4) :
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨ (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = 3 ∧ x4 = -1) ∨ (x1 = -1 ∧ x2 = 3 ∧ x3 = -1 ∧ x4 = -1) ∨
  (x1 = 3 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = -1) :=
sorry

end solve_x1_x2_x3_x4_l724_724327


namespace candies_count_l724_724077

variable (m_and_m : Nat) (starbursts : Nat)
variable (ratio_m_and_m_to_starbursts : Nat → Nat → Prop)

-- Definition of the ratio condition
def ratio_condition : Prop :=
  ∃ (k : Nat), (m_and_m = 7 * k) ∧ (starbursts = 4 * k)

-- The main theorem to prove
theorem candies_count (h : m_and_m = 56) (r : ratio_condition m_and_m starbursts) : starbursts = 32 :=
  by
  sorry

end candies_count_l724_724077


namespace mr_brown_financial_outcome_l724_724484

theorem mr_brown_financial_outcome :
  ∃ (C₁ C₂ : ℝ), (2.40 = 1.25 * C₁) ∧ (2.40 = 0.75 * C₂) ∧ ((2.40 + 2.40) - (C₁ + C₂) = -0.32) :=
by
  sorry

end mr_brown_financial_outcome_l724_724484


namespace triangle_is_acute_l724_724863

def triangle_angles_acute (α β γ : ℝ) : Prop :=
  α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = π ∧ α < π / 2 ∧ β < π / 2 ∧ γ < π / 2

theorem triangle_is_acute (α β γ : ℝ) (h1 : sin α > cos β) (h2 : sin β > cos γ) (h3 : sin γ > cos α) :
  triangle_angles_acute α β γ :=
by 
  sorry

end triangle_is_acute_l724_724863


namespace total_money_spent_correct_l724_724481

def money_spent_at_mall : Int := 250

def cost_per_movie : Int := 24
def number_of_movies : Int := 3
def money_spent_at_movies := cost_per_movie * number_of_movies

def cost_per_bag_of_beans : Float := 1.25
def number_of_bags : Int := 20
def money_spent_at_market := cost_per_bag_of_beans * number_of_bags

def total_money_spent := money_spent_at_mall + money_spent_at_movies + money_spent_at_market

theorem total_money_spent_correct : total_money_spent = 347 := by
  sorry

end total_money_spent_correct_l724_724481


namespace necessary_but_not_sufficient_condition_log_arithmetic_l724_724990

theorem necessary_but_not_sufficient_condition_log_arithmetic
  (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) (hq : q ≠ 0) (geometric : ∀ n, a n = a1 * q ^ n) :
  (∀ n, ∃ c, (∑ i in finset.range n, log (a (i+1) + 1) - log (a i + 1)) = c) → ∃ n, ¬(∀ n, ∃ c, (∑ i in finset.range n, log (a (i+1) + 1) - log (a i + 1)) = c) :=
sorry

end necessary_but_not_sufficient_condition_log_arithmetic_l724_724990


namespace gamma_unique_45_degrees_l724_724031

theorem gamma_unique_45_degrees (α β γ : ℝ) 
  (h1 : 0 ≤ α ∧ α ≤ 90) 
  (h2 : 0 ≤ β ∧ β ≤ 90) 
  (h3 : 0 ≤ γ ∧ γ ≤ 90) 
  (h_sin_cos_eq_tan : Real.sin α - Real.cos β = Real.tan γ) 
  (h_sin_cos_eq_cot : Real.sin β - Real.cos α = Real.cot γ) : 
  γ = 45 :=
sorry

end gamma_unique_45_degrees_l724_724031


namespace perpendicular_to_plane_parallel_l724_724707

-- Define the context: lines, planes and their relationships
variables {m n : Type*} {α : Type*} (is_line : m -> Prop) (is_line : n -> Prop) (is_plane : α -> Prop)
variables (parallel : m -> α -> Prop) (perpendicular : m -> α -> Prop) 
variables (non_coinci : m -> n -> Prop) 

-- Assume initial conditions
axiom m_non_coinci : non_coinci m n
axiom α_non_coinci : non_coinci α β

-- Statement: If both m and n are perpendicular to plane α, then m and n are parallel lines.
theorem perpendicular_to_plane_parallel (hm_perp : perpendicular m α) (hn_perp : perpendicular n α) : parallel m n :=
sorry

end perpendicular_to_plane_parallel_l724_724707


namespace find_n_l724_724569

theorem find_n : ∃ n, 10^n = 10^(-7) * real.sqrt (10^92 / 0.0001) ∧ n = 41 := 
by
  sorry

end find_n_l724_724569


namespace arithmetic_mean_divisors_condition_l724_724493

theorem arithmetic_mean_divisors_condition (n : ℕ) : 
  (n ∣ 799) ↔ 
  (∃ (k m : ℕ), m = 1598 ∧ 
                k * n = m / 2 ∧ 
                ∀ i, i ∈ set.range k → 
                     ∑ j in finset.Ico (2 * i * n) ((2 * i + 1) * n), 
                     (j + 1) / n = i + 1) := 
sorry

end arithmetic_mean_divisors_condition_l724_724493


namespace find_f_neg9_l724_724652

noncomputable def f (x : ℝ) : ℝ :=
  if h : 0 ≤ x ∧ x ≤ 2 then 3^x else sorry

theorem find_f_neg9 (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_periodic : ∀ x : ℝ, f (x + 4) = f x)
  (h_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f x = 3^x) :
  f (-9) = 3 :=
begin
  sorry
end

end find_f_neg9_l724_724652


namespace cube_edge_length_l724_724393

theorem cube_edge_length (V : ℝ) (hV : V = (32 / 3) * Real.pi) : ∃ s : ℝ, s = (4 * Real.sqrt 3) / 3 :=
by
  sorry

end cube_edge_length_l724_724393


namespace reciprocal_of_neg3_l724_724147

theorem reciprocal_of_neg3 : ∃ x : ℝ, (-3 : ℝ) * x = 1 :=  
begin
  use -1/3,
  norm_num,
end

end reciprocal_of_neg3_l724_724147


namespace cone_volume_half_sector_l724_724608

-- Declare the given conditions as variables
variables (R : ℝ) (r b : ℝ)
-- Define the problem conditions
def is_half_sector_rolled_cone (R r b : ℝ) : Prop :=
  R = 6 ∧
  2 * π * r = π * R ∧
  R^2 - r^2 = b^2

-- Define the cone volume formula
def cone_volume (r h : ℝ) : ℝ := (1/3) * π * r^2 * h

-- Define the main proof statement
theorem cone_volume_half_sector :
  is_half_sector_rolled_cone R r b →
  cone_volume r √(R^2 - r^2) = 9 * π * √3 :=
by
  intro h
  sorry

end cone_volume_half_sector_l724_724608


namespace reciprocal_of_neg_three_l724_724141

theorem reciprocal_of_neg_three : -3 * (-1 / 3) = 1 := 
by
  sorry

end reciprocal_of_neg_three_l724_724141


namespace arithmetic_sequence_a9_l724_724461

theorem arithmetic_sequence_a9 (S : ℕ → ℤ) (a : ℕ → ℤ) :
  S 8 = 4 * a 3 → a 7 = -2 → a 9 = -6 := by
  sorry

end arithmetic_sequence_a9_l724_724461


namespace painting_house_cost_l724_724804

theorem painting_house_cost 
  (judson_contrib : ℕ := 500)
  (kenny_contrib : ℕ := judson_contrib + (judson_contrib * 20) / 100)
  (camilo_contrib : ℕ := kenny_contrib + 200) :
  judson_contrib + kenny_contrib + camilo_contrib = 1900 :=
by
  sorry

end painting_house_cost_l724_724804


namespace train_speed_l724_724273

theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 100) (h2 : time = 12) :
  let distance_km := length / 1000 in
  let time_hours := time / 3600 in
  let speed := distance_km / time_hours in
  speed = 30 :=
by
  sorry

end train_speed_l724_724273


namespace correct_conclusions_l724_724716

theorem correct_conclusions (P1 P2 P3 P4 : Prop) 
  (h1: P1 = (∀ x : ℝ, x^2 - x ≤ 0))
  (h2:  P2 = (∀ a b m : ℝ, am^2 < bm^2 → a < b))
  (h3: P3 = ∀ a b : ℝ, (ax + 2 * y - 1 = 0) ∧ (x + by + 2 = 0) → (a = -2 ↔ is_perpendicular))
  (h4: P4 = ∀ x : ℝ, (f(-x) = -f(x)) ∧ (g(-x) = g(x)) → (x > 0 → f'(x) > 0 ∧ g'(x) > 0) → (x < 0 → f'(x) > g'(x)))
: P1 ∧ P4 :=
by
  sorry

end correct_conclusions_l724_724716


namespace Petya_tore_out_sheets_l724_724888

theorem Petya_tore_out_sheets (n m : ℕ) (h1 : n = 185) (h2 : m = 518)
  (h3 : m.digits = n.digits) : (m - n + 1) / 2 = 167 :=
by
  sorry

end Petya_tore_out_sheets_l724_724888


namespace area_square_l724_724927

-- Given Conditions
def is_square (A B C D : Type) : Prop := 
  -- Assume necessary properties about the square ABCD 
  sorry
  
def right_angles_at (E F G H : Type) : Prop := 
  -- Assume necessary properties that all these points form right angles
  sorry

def segment_lengths (DE EF FG GH HB : ℝ) : Prop :=
  (DE = 6) ∧ (EF = 4) ∧ (FG = 4) ∧ (GH = 1) ∧ (HB = 2)

-- Main Statement to Prove
theorem area_square (A B C D E F G H : Type) (DE EF FG GH HB : ℝ) 
  (h₀ : is_square A B C D)
  (h₁ : right_angles_at E F G H) 
  (h₂ : segment_lengths DE EF FG GH HB) : 
  let DK := DE + FG + HB in
  let KB := EF + GH in
  let DB := Math.sqrt (DK^2 + KB^2) in
  let s := DB / Math.sqrt 2 in
  let S := s^2 in
  S = 84.5 := 
begin
  sorry
end

end area_square_l724_724927


namespace exist_line_with_chord_diff_l724_724560

-- Define the existence of a line with the required property

theorem exist_line_with_chord_diff (S1 S2 : Circle) (A : Point)
  (H1 : A ∈ S1 ∧ A ∈ S2) (a : ℝ) :
  ∃ l : Line, l.passes_through A ∧ 
  abs ((length_of_chord S1 l) - (length_of_chord S2 l)) = a :=
sorry

end exist_line_with_chord_diff_l724_724560


namespace sufficient_but_not_necessary_for_monotonic_l724_724388

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
∀ x y, x ≤ y → f x ≤ f y

noncomputable def is_sufficient_condition (P Q : Prop) : Prop :=
P → Q

noncomputable def is_not_necessary_condition (P Q : Prop) : Prop :=
¬ Q → ¬ P

noncomputable def is_sufficient_but_not_necessary (P Q : Prop) : Prop :=
is_sufficient_condition P Q ∧ is_not_necessary_condition P Q

theorem sufficient_but_not_necessary_for_monotonic (f : ℝ → ℝ) :
  (∀ x, 0 ≤ deriv f x) → is_monotonically_increasing f :=
sorry

end sufficient_but_not_necessary_for_monotonic_l724_724388


namespace circle_equation_of_diameter_l724_724698

theorem circle_equation_of_diameter (A B : ℝ × ℝ) (hA : A = (-4, -5)) (hB : B = (6, -1)) :
  ∃ h k r : ℝ, (x - h)^2 + (y - k)^2 = r ∧ h = 1 ∧ k = -3 ∧ r = 29 := 
by
  sorry

end circle_equation_of_diameter_l724_724698


namespace MN_passes_through_center_of_parallelogram_l724_724785

-- Define the geometrical entities of the problem.
variable {Point : Type} [Nonempty Point]
variable {Line : Type}
variable (parallelogram : Type)
variable (A B C D P Q O M N : Point)

-- Parallelogram specific definitions
variable (AB_CD_intersects : (parallelogram × parallelogram) = ∅)
variable (AB : Set Point)
variable (CD : Set Point)
variable {is_on_line_P : P ∈ AB}
variable {is_on_line_Q : Q ∈ CD}
variable (DP_AQ_intersect_M : ∃ M, ∃ DP AQ : Line, M ∈ DP ∩ AQ)
variable (CP_BQ_intersect_N : ∃ N, ∃ CP BQ : Line, N ∈ CP ∩ BQ)
variable (O_center : is_center O)
variable (proof_MO_NO_slopes : slope_line (O, M) = slope_line (O, N))

theorem MN_passes_through_center_of_parallelogram : 
  (MN_passes_through_center : Line) (center O) (M N O : Point) 
  (parallelogram : Type) : MN ⊃ center O :=
by
  sorry

end MN_passes_through_center_of_parallelogram_l724_724785


namespace part1_part2_l724_724498

-- Definition: There are 4 balls and 4 boxes
def num_balls := 4
def num_boxes := 4

-- Condition for Part 1: A and B must be in the same box
def part1_condition (A B C D : Type) (boxes : list (list Type)) : Prop :=
  ∃ box, box = [A, B] ∨ box = [B, A]

-- Question and answer for Part 1: Number of ways to place the entities given the condition
theorem part1 (A B C D : Type) (boxes : list (list Type)) (h : part1_condition A B C D boxes) : 
  ∃ n, n = 64 :=
sorry

-- Condition for Part 2: Each box can hold at most 2 balls
def part2_condition (A B C D : Type) (boxes : list (list Type)) : Prop :=
  ∀ box ∈ boxes, box.length ≤ 2

-- Question and answer for Part 2: Number of ways to place the balls given the condition
theorem part2 (A B C D : Type) (boxes : list (list Type)) (h : part2_condition A B C D boxes) : 
  ∃ n, n = 204 :=
sorry

end part1_part2_l724_724498


namespace speed_difference_l724_724276

-- Define the given conditions
def distance_to_library : ℝ := 8 -- miles
def nora_time_minutes : ℝ := 15 -- minutes
def mia_time_minutes : ℝ := 40 -- minutes

-- Define the conversion factors and calculated times in hours
def nora_time_hours : ℝ := nora_time_minutes / 60
def mia_time_hours : ℝ := mia_time_minutes / 60

-- Define the average speeds calculations
def nora_speed : ℝ := distance_to_library / nora_time_hours
def mia_speed : ℝ := distance_to_library / mia_time_hours

-- Prove the difference in speeds is 20 mph
theorem speed_difference : nora_speed - mia_speed = 20 := by 
  sorry

end speed_difference_l724_724276


namespace minimum_nine_points_distance_l724_724436

theorem minimum_nine_points_distance (n : ℕ) : 
  (∀ (p : Fin n → ℝ × ℝ),
    (∀ i, ∃! (four_points : List (Fin n)), 
      List.length four_points = 4 ∧ (∀ j ∈ four_points, dist (p i) (p j) = 1)))
    ↔ n = 9 :=
by 
  sorry

end minimum_nine_points_distance_l724_724436


namespace sum_of_differences_eq_68896_l724_724462

def T : Finset ℕ := Finset.range 9 |>.map (λ x => 3 ^ x)

noncomputable def M : ℕ := 
  let addends := Finset.range 9 |>.map (λ x => (8 - x) * 3 ^ x + x * 3 ^ (8 - x))
  addends.sum

theorem sum_of_differences_eq_68896 : M = 68896 := by
  sorry

end sum_of_differences_eq_68896_l724_724462


namespace quadratic_inequality_false_iff_range_of_a_l724_724428

theorem quadratic_inequality_false_iff_range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) ↔ (-1 < a ∧ a < 3) :=
sorry

end quadratic_inequality_false_iff_range_of_a_l724_724428


namespace pile_limit_exists_l724_724054

noncomputable def log_floor (b x : ℝ) : ℤ :=
  Int.floor (Real.log x / Real.log b)

theorem pile_limit_exists (k : ℝ) (hk : k < 2) : ∃ Nk : ℤ, 
  Nk = 2 * (log_floor (2 / k) 2 + 1) := 
  by
    sorry

end pile_limit_exists_l724_724054


namespace least_number_to_make_divisible_l724_724570

def least_common_multiple (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem least_number_to_make_divisible (n : ℕ) (a : ℕ) (b : ℕ) (c : ℕ) : 
  least_common_multiple a b = 77 → 
  (n % least_common_multiple a b) = 40 →
  c = (least_common_multiple a b - (n % least_common_multiple a b)) →
  c = 37 :=
by
sorry

end least_number_to_make_divisible_l724_724570


namespace total_games_equal_684_l724_724597

-- Define the number of players
def n : Nat := 19

-- Define the formula to calculate the total number of games played
def total_games (n : Nat) : Nat := n * (n - 1) * 2

-- The proposition asserting the total number of games equals 684
theorem total_games_equal_684 : total_games n = 684 :=
by
  sorry

end total_games_equal_684_l724_724597


namespace num_diagonals_octagon_l724_724642

def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem num_diagonals_octagon : num_diagonals 8 = 20 :=
by
  sorry

end num_diagonals_octagon_l724_724642


namespace rotation_locus_l724_724650

-- Definitions for points and structure of the cube
structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Cube :=
(A : Point3D) (B : Point3D) (C : Point3D) (D : Point3D)
(E : Point3D) (F : Point3D) (G : Point3D) (H : Point3D)

-- Function to perform the required rotations and return the locus geometrical representation
noncomputable def locus_points_on_surface (c : Cube) : Set Point3D :=
sorry

-- Mathematical problem rephrased in Lean 4 statement
theorem rotation_locus (c : Cube) :
  locus_points_on_surface c = {c.D, c.A} ∪ {c.A, c.C} ∪ {c.C, c.D} :=
sorry

end rotation_locus_l724_724650


namespace max_value_of_function_l724_724380

theorem max_value_of_function (x : ℝ) (hx : 0 < x ∧ x < 1 / 2) : 
  ∃ y, y = x * (1 - 2 * x) ∧ y ≤ 1 / 8 :=
by
  use x * (1 - 2 * x)
  split
  . rfl
  . sorry

end max_value_of_function_l724_724380


namespace ratio_MN_AN_l724_724121

-- Define the necessary structures and axioms
variable {ABC : Triangle}
variable {A B C M N : Point}
variable {k : ℝ}

-- Given:
-- 1. ABC is an isosceles triangle with base AB.
-- 2. A circle is inscribed in triangle ABC touching BC at point M.
-- 3. Segment AM intersects the circle at point N.
-- 4. The ratio AB/BC = k.
axiom is_isosceles (ABC : Triangle) : ABC.isIsosceles A B C
axiom inscribed_circle_touches_BC_at_M (ABC : Triangle) (M : Point) : Circle.isInscribedInTriangleTouchingSideAtPoint ABC BC M
axiom AM_intersects_circle_at_N (A M N : Point) (circle : Circle) : Segment A M ∩ circle = {N}
axiom ratio_AB_BC (AB BC : ℝ) (k : ℝ) : AB / BC = k

-- To prove:
-- 1. The ratio MN/AN = 2(2 - k).
theorem ratio_MN_AN 
  (h_isosceles : ABC.isIsosceles A B C)
  (h_circle : Circle.isInscribedInTriangleTouchingSideAtPoint ABC BC M)
  (h_intersection : Segment A M ∩ Circle = {N})
  (h_ratio : AB / BC = k) :
  MN / AN = 2 * (2 - k) := 
  sorry

end ratio_MN_AN_l724_724121


namespace primes_with_ones_digit_3_l724_724738

theorem primes_with_ones_digit_3 :
  ∃ n : ℕ, n = 7 ∧ n = (List.filter Nat.Prime [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]).length :=
begin
  sorry
end

end primes_with_ones_digit_3_l724_724738


namespace simplify_fraction_l724_724104

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : 
  ( (x^2 + 1) / (x - 1) - (2*x) / (x - 1) ) = x - 1 :=
by
  -- Your proof steps would go here.
  sorry

end simplify_fraction_l724_724104


namespace xiaoliang_prob_correct_l724_724931

def initial_box_setup : List (Nat × Nat) := [(1, 2), (2, 2), (3, 2), (4, 2)]

def xiaoming_draw : List Nat := [1, 1, 3]

def remaining_balls_after_xiaoming : List (Nat × Nat) := [(1, 0), (2, 2), (3, 1), (4, 2)]

def remaining_ball_count (balls : List (Nat × Nat)) : Nat :=
  balls.foldl (λ acc ⟨_, count⟩ => acc + count) 0

theorem xiaoliang_prob_correct :
  (1 : ℚ) / (remaining_ball_count remaining_balls_after_xiaoming) = 1 / 5 :=
by
  sorry

end xiaoliang_prob_correct_l724_724931


namespace magic_8_ball_probability_l724_724796

def probability_positive (p_pos : ℚ) (questions : ℕ) (positive_responses : ℕ) : ℚ :=
  (Nat.choose questions positive_responses : ℚ) * (p_pos ^ positive_responses) * ((1 - p_pos) ^ (questions - positive_responses))

theorem magic_8_ball_probability :
  probability_positive (1/3) 7 3 = 560 / 2187 :=
by
  sorry

end magic_8_ball_probability_l724_724796


namespace max_possible_value_l724_724043

def sequence (a : ℕ → ℝ) : Prop :=
  (a 0 = 0) ∧
  (a 1 = 1) ∧
  (∀ n ≥ 2, ∃ k (hk : 1 ≤ k ∧ k ≤ n), a n = (∑ i in (finset.range k).map (λ m, n - m - 1), a i) / k)

theorem max_possible_value (a : ℕ → ℝ) (h : sequence a) :
  a 2018 - a 2017 = 2016 / (2017^2) :=
sorry

end max_possible_value_l724_724043


namespace diagonals_intersect_l724_724500

def Point := (ℝ × ℝ)

def midpoint (p1 p2 : Point) : Point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem diagonals_intersect (p1 p2 : Point) (h : p1 = (2, -3) ∧ p2 = (8, 9)) :
  midpoint p1 p2 = (5, 3) :=
by
  cases h with
  | intro h1 h2 =>
    rw [h1, h2]
    -- detailed proof goes here
sorry

end diagonals_intersect_l724_724500


namespace distinct_triangles_octahedron_l724_724737

theorem distinct_triangles_octahedron (V : Finset ℕ) (hV : V.card = 6) (h_collinear : ∀ (a b c : ℕ), a ∈ V → b ∈ V → c ∈ V → ¬ (a = b ∧ b = c ∧ c = a)) :
  (V.subset_powerset_len 3).card = 20 :=
sorry

end distinct_triangles_octahedron_l724_724737


namespace sum_odd_divisors_of_90_less_than_10_l724_724950

theorem sum_odd_divisors_of_90_less_than_10 :
  ∑ d in {d | d ∣ 90 ∧ odd d ∧ d < 10}.toFinset, d = 18 :=
by
  sorry

end sum_odd_divisors_of_90_less_than_10_l724_724950


namespace find_blue_chips_l724_724998

def num_chips_satisfies (n m : ℕ) : Prop :=
  (n > m) ∧ (n + m > 2) ∧ (n + m < 50) ∧
  (n * (n - 1) + m * (m - 1)) = 2 * n * m

theorem find_blue_chips (n : ℕ) :
  (∃ m : ℕ, num_chips_satisfies n m) → 
  n = 3 ∨ n = 6 ∨ n = 10 ∨ n = 15 ∨ n = 21 ∨ n = 28 :=
by
  sorry

end find_blue_chips_l724_724998


namespace quadrilateral_AD_length_l724_724015

variables (A B C D P M : Type) [ordered_field A]
variable (distance : A → A → A)
variable (midpoint : A → A → A)
variable (perp_distance : A → A → A)

-- Given conditions:
def conditions (ABCD : A) (AC BD : A) (intersect_AT_P : Prop) (midpoint_AD : Prop) 
  (length_CM : A) (distance_P_BC : A) (length_AP : A) (circumcircle_exists : Prop) : Prop :=
  intersect_AT_P ∧ midpoint_AD ∧ length_CM = 5 / 4 ∧ distance_P_BC = 1 / 2 ∧ length_AP = 1 ∧ circumcircle_exists

-- Proof statement:
theorem quadrilateral_AD_length (ABCD : A) (AC BD : A) (intersect_AT_P : Prop) (midpoint_AD : Prop) 
  (length_CM : A) (distance_P_BC : A) (length_AP : A) (circumcircle_exists : Prop)
  (h : conditions ABCD AC BD intersect_AT_P midpoint_AD length_CM distance_P_BC length_AP circumcircle_exists):
  ∃ AD : A, AD = 3 * real.sqrt 6 - 2 :=
by sorry

end quadrilateral_AD_length_l724_724015


namespace tammy_speed_proof_l724_724228

noncomputable def tammy_average_speed_second_day (v t : ℝ) :=
  v + 0.5

theorem tammy_speed_proof :
  ∃ v t : ℝ, 
    t + (t - 2) = 14 ∧
    v * t + (v + 0.5) * (t - 2) = 52 ∧
    tammy_average_speed_second_day v t = 4 :=
by
  sorry

end tammy_speed_proof_l724_724228


namespace pages_torn_l724_724914

theorem pages_torn (n : ℕ) (H1 : n = 185) (H2 : ∃ m, m = 518 ∧ (digits 10 m = digits 10 n) ∧ (m % 2 = 0)) : 
  ∃ k, k = ((518 - 185 + 1) / 2) ∧ k = 167 :=
by sorry

end pages_torn_l724_724914


namespace larger_number_of_ratio_and_lcm_l724_724768

theorem larger_number_of_ratio_and_lcm (x : ℕ) (h1 : (2 * x) % (5 * x) = 160) : (5 * x) = 160 := by
  sorry

end larger_number_of_ratio_and_lcm_l724_724768


namespace angle_between_vectors_minimum_value_ta_minus_b_l724_724733

-- Definitions of vectors and given conditions
variables (a b : ℝ^3) (t : ℝ)
variable h1 : ‖a‖ = real.sqrt 2
variable h2 : ‖b‖ = 4
variable h3 : a ⬝ (b - a) = 2

-- Proof Statement 1: Find the angle between vectors a and b
theorem angle_between_vectors : 
  let θ := real.arccos ((a ⬝ b) / (real.sqrt 2 * 4)) in θ = real.pi / 4 :=
by sorry

-- Proof Statement 2: Find the minimum value of ‖t • a - b‖ and the value of t
theorem minimum_value_ta_minus_b : 
  let squared_norm := λ t, ‖t • a - b‖^2 in 
  ∃ t0, squared_norm t0 = 2 * (t - 2)^2 + 8 ∧ t0 = 2 ∧ ‖t • a - b‖ = 2 * real.sqrt 2 :=
by sorry

end angle_between_vectors_minimum_value_ta_minus_b_l724_724733


namespace arithmetic_sequence_sum_l724_724875

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℤ) (d : ℤ),
    a 1 = 1 →
    d ≠ 0 →
    (a 2 = a 1 + d) →
    (a 3 = a 1 + 2 * d) →
    (a 6 = a 1 + 5 * d) →
    (a 3)^2 = (a 2) * (a 6) →
    (1 + 2 * d)^2 = (1 + d) * (1 + 5 * d) →
    (6 / 2) * (2 * a 1 + (6 - 1) * d) = -24 := 
by intros a d h1 h2 h3 h4 h5 h6 h7
   sorry

end arithmetic_sequence_sum_l724_724875


namespace exists_max_pile_division_l724_724052

theorem exists_max_pile_division (k : ℝ) (hk : k < 2) : 
  ∃ (N_k : ℕ), ∀ (A : Multiset ℝ) (m : ℝ), (∀ a ∈ A, a < 2 * m) → 
    ¬(∃ B : Multiset ℝ, B.card > N_k ∧ (∀ b ∈ B, b ∈ A ∧ b < 2 * m)) :=
sorry

end exists_max_pile_division_l724_724052


namespace train_cross_time_l724_724272

def length_of_train : ℕ := 165
def length_of_bridge : ℕ := 275
def speed_of_train_kmph : ℕ := 45

def total_distance : ℕ := length_of_train + length_of_bridge
def speed_of_train_mps : ℝ := (speed_of_train_kmph * 1000) / 3600
def time_to_cross_bridge : ℝ := total_distance / speed_of_train_mps

theorem train_cross_time :
  time_to_cross_bridge = 35.2 :=
by
  /-
    Proof steps go here
  -/
  sorry

end train_cross_time_l724_724272


namespace radius_of_circle_D_l724_724646

theorem radius_of_circle_D : 
  ∃ R : ℝ, (∃ m n : ℕ, R = Real.sqrt m - n ∧ m + n = 254) ∧ 
  (let r := R / 4 in
  (let C_radius := 4 in
  r = R / 4 ∧
  E_tangent_C := C_radius - r ∧ 
  E_tangent_AB := ∃ F : ℝ, ∀ (x : ℝ), x = r ∧ 
  CF_squared := (C_radius - r)^2 - r^2 ∧ 
  R = Real.sqrt CF_squared )) := sorry

end radius_of_circle_D_l724_724646


namespace total_length_T_l724_724463

def T := {p : ℝ × ℝ | 
  let (x, y) := p in 
  abs (abs (abs x - 3) - 2) + abs (abs (abs y - 3) - 2) = 2}

theorem total_length_T : 
  let lines_length := 128 in
  ∑ p in T, length_of_lines(p) = lines_length :=
sorry

end total_length_T_l724_724463


namespace range_of_a_l724_724334

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 * a - 1) * x + 3 * a else a^x

theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → ((f a x₁ - f a x₂) / (x₁ - x₂)) < 0) ↔ 
  (1/4 ≤ a ∧ a < 1/2) :=
by 
  sorry -- The proof is omitted

end range_of_a_l724_724334


namespace exponential_decreasing_l724_724396

noncomputable def a_range : set ℝ :=
  {a | ∀ x y : ℝ, x < y → (a - 1) ^ x > (a - 1) ^ y}

theorem exponential_decreasing (a : ℝ) (h : ∀ x y : ℝ, x < y → (a - 1) ^ x > (a - 1) ^ y) :
  1 < a ∧ a < 2 :=
sorry

end exponential_decreasing_l724_724396


namespace quadrilateral_in_unit_circle_max_m_l724_724047

noncomputable def maximum_value_of_m (A B C D: ℝ) (rays_AB AD: ℝ) (m: ℝ) : ℝ :=
  if hyp : (A * A + B * B ≤ 1) ∧ (A * A + D * D ≤ 1) ∧ (∠BAD = 30) then
    2
  else
    sorry

theorem quadrilateral_in_unit_circle_max_m {A B C D: ℝ} (rays_AB AD: ℝ) (m: ℝ) :
  (A * A + B * B ≤ 1) ∧ (A * A + D * D ≤ 1) ∧ (∠BAD = 30) → 
  m ≥ CP + PQ + CQ → 
  maximum_value_of_m A B C D rays_AB AD m = 2 :=
sorry

end quadrilateral_in_unit_circle_max_m_l724_724047


namespace complex_triangle_inequality_l724_724504

theorem complex_triangle_inequality
  (α : ℕ → ℂ)
  (h1 : α 1 ≠ 0 ∧ α 2 ≠ 0 ∧ α 3 ≠ 0)
  (h2 : α 1 ≠ α 2 ∧ α 2 ≠ α 3 ∧ α 3 ≠ α 1)
  (h3 : α 1 + α 2 + α 3 = 0) :
  (∑ i in {1, 2, 3}, (|α (i % 3 + 1) - α ((i + 1) % 3 + 1)| / |α i|.sqrt) * ((1 / |α (i % 3 + 1)|.sqrt) + (1 / |α ((i + 1) % 3 + 1)|.sqrt) - (2 / |α i|.sqrt))) ≤ 0 := 
sorry

end complex_triangle_inequality_l724_724504


namespace interior_and_exterior_angles_of_regular_dodecagon_l724_724565

-- Definition of a regular dodecagon
def regular_dodecagon_sides : ℕ := 12

-- The sum of the interior angles of a regular polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Measure of one interior angle of a regular polygon
def one_interior_angle (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Measure of one exterior angle of a regular polygon (180 degrees supplementary to interior angle)
def one_exterior_angle (n : ℕ) : ℕ := 180 - one_interior_angle n

-- The theorem to prove
theorem interior_and_exterior_angles_of_regular_dodecagon :
  one_interior_angle regular_dodecagon_sides = 150 ∧ one_exterior_angle regular_dodecagon_sides = 30 :=
by
  sorry

end interior_and_exterior_angles_of_regular_dodecagon_l724_724565


namespace tree_growth_rate_l724_724874

-- Given conditions
def currentHeight : ℝ := 52
def futureHeightInches : ℝ := 1104
def oneFootInInches : ℝ := 12
def years : ℝ := 8

-- Prove the yearly growth rate in feet
theorem tree_growth_rate:
  (futureHeightInches / oneFootInInches - currentHeight) / years = 5 := 
by
  sorry

end tree_growth_rate_l724_724874


namespace angle_XZY_45_degree_l724_724459

variables {Point : Type} [metric_space Point] [inner_product_space ℝ Point] 
variables (A B F Z X Y : Point)
variables (FA AB AZ BY: ℝ)

-- Conditions
def midpoint (M P Q : Point) := dist M P = dist M Q
def on_segment (P Q R : Point) := dist P Q + dist Q R = dist P R
def perpendicular (L P Q : Point) := ∠ (P - L) (Q - L) = π / 2

-- Assume F is the midpoint of AB
@[assume h1 : midpoint F A B]
-- Assume Z is on AF
@[assume h2 : on_segment A Z F]
-- Assume FX = FA
@[assume h3 : dist F X = dist F A]
-- Assume BY = AZ
@[assume h4 : dist B Y = dist A Z]
-- Assume X, Y are on the same side of AB
@[assume h5 : (X - F) = (Y - B) ∧ (X - A) = (Y - A)]

-- Proof of final statement
theorem angle_XZY_45_degree : ∠ (X - Z) (Y - Z) = π / 4 := sorry

end angle_XZY_45_degree_l724_724459


namespace pages_torn_and_sheets_calculation_l724_724900

theorem pages_torn_and_sheets_calculation : 
  (∀ (n : ℕ), (sheet_no n) = (n + 1) / 2 → (2 * (n + 1) / 2) - 1 = n ∨ 2 * (n + 1) / 2 = n) →
  let first_page := 185 in
  let last_page := 518 in
  last_page = 518 → 
  ((last_page - first_page + 1) / 2) = 167 := 
by
  sorry

end pages_torn_and_sheets_calculation_l724_724900


namespace round_robin_equal_points_l724_724001

theorem round_robin_equal_points
  (n : ℕ) (h : n > 2)
  (points : Fin n → ℕ)
  (victory_points : ℕ := 1)
  (loss_points : ℕ := 0)
  (coeff : (Fin n) → ℕ)
  (h_points : ∀ i : Fin n, coeff i = ∑ j in (Finset.filter (λ j => points j < points i) Finset.univ), points j)
  (h_coeff_equal : ∀ i j : Fin n, coeff i = coeff j)
  : ∀ i j : Fin n, points i = points j :=
by
  sorry

end round_robin_equal_points_l724_724001


namespace incorrect_statement_C_l724_724321

def linear_function (x : ℝ) : ℝ := -2 * x + 1

theorem incorrect_statement_C : 
  (∀ x : ℝ, linear_function x = -2 * x + 1) ∧
  (∃ x : ℝ, linear_function x = 1) ∧
  (∃ x : ℝ, linear_function x = 0.5 * x) ∧
  (¬ ∃ x : ℝ, linear_function x > linear_function (x + 1)) ∧
  (¬ (∀ x : ℝ, linear_function x (x < 0 ∨ x > 0)))
  → false :=
begin
  sorry
end

end incorrect_statement_C_l724_724321


namespace nails_remaining_l724_724251

theorem nails_remaining (nails_initial : ℕ) (kitchen_fraction : ℚ) (fence_fraction : ℚ) (nails_used_kitchen : ℕ) (nails_remaining_after_kitchen : ℕ) (nails_used_fence : ℕ) (nails_remaining_final : ℕ) 
  (h1 : nails_initial = 400) 
  (h2 : kitchen_fraction = 0.30) 
  (h3 : nails_used_kitchen = kitchen_fraction * nails_initial) 
  (h4 : nails_remaining_after_kitchen = nails_initial - nails_used_kitchen) 
  (h5 : fence_fraction = 0.70) 
  (h6 : nails_used_fence = fence_fraction * nails_remaining_after_kitchen) 
  (h7 : nails_remaining_final = nails_remaining_after_kitchen - nails_used_fence) :
  nails_remaining_final = 84 := by
sorry

end nails_remaining_l724_724251


namespace kenny_books_l724_724808

def lawns_mowed := 35
def charge_per_lawn := 15
def video_game_price := 45
def book_price := 5
def desired_video_games := 5

theorem kenny_books :
  let total_earnings := lawns_mowed * charge_per_lawn in
  let total_video_game_cost := desired_video_games * video_game_price in
  let remaining_money := total_earnings - total_video_game_cost in
  let num_books := remaining_money / book_price in
  num_books = 60 :=
by
  sorry

end kenny_books_l724_724808


namespace circumcircle_passing_condition_l724_724003

structure Triangle (α : Type) :=
(A B C : α)

structure Point (α : Type) :=
(x y : α)

variable {α : Type} [Field α]

def is_excircle_touching (T : Triangle α) (D E F : Point α) (BC AB AC : α) : Prop :=
  -- definition of the excircle touching points
  sorry

def is_projection (D : Point α) (EF : → α) (P : Point α) : Prop :=
  -- definition of the projection
  sorry

def is_circumcircle (k : α) (T : Triangle α) (P : Point α) (M : Point α) : Prop :=
  -- definition of the circumcircle
  sorry

def is_midpoint (E F M : Point α) : Prop :=
  -- definition of midpoint
  sorry

theorem circumcircle_passing_condition (T : Triangle α) (BC AB AC : α)
  (D E F P M : Point α) (k : α) :
  is_excircle_touching T D E F BC AB AC → 
  is_projection D (E, F) P →
  is_midpoint E F M →
  (is_circumcircle k T P M ↔ is_circumcircle k T M) :=
begin
  sorry
end

end circumcircle_passing_condition_l724_724003


namespace amount_in_paise_l724_724754

theorem amount_in_paise (a : ℝ) (h_a : a = 170) (percentage_value : ℝ) (h_percentage : percentage_value = 0.5 / 100) : 
  (percentage_value * a * 100) = 85 := 
by
  sorry

end amount_in_paise_l724_724754


namespace possible_values_l724_724495

def sequence := List.range 1598

def part_mean (n : ℕ) (part : List ℕ) : ℕ :=
  part.sum / part.length

noncomputable def valid_n : ℕ → Prop :=
  λ n, ∃ k : ℕ, k * n = 799

theorem possible_values (n : ℕ) : valid_n n ↔ n = 1 ∨ n = 17 ∨ n = 47 ∨ n = 799 :=
sorry

end possible_values_l724_724495


namespace T_perimeter_is_14_l724_724533

-- Define the dimensions of the rectangles
def horizontal_rect_length : ℝ := 6
def horizontal_rect_height : ℝ := 1
def vertical_rect_length : ℝ := 4
def vertical_rect_height : ℝ := 3

-- Define the exposed and visible parts derived from the placement
def exposed_side_length : ℝ := (horizontal_rect_length - vertical_rect_height) / 2
def visible_vertical_top_length : ℝ := horizontal_rect_height 
def visible_vertical_bottom_length : ℝ := vertical_rect_length - horizontal_rect_height

-- Define the total exposed lengths for the horizontal and vertical parts
def total_horizontal_exposed_length : ℝ := horizontal_rect_length + 2 * exposed_side_length
def total_vertical_visible_length : ℝ := 2 * visible_vertical_bottom_length + visible_vertical_top_length 

-- Final perimeter calculation
def T_perimeter : ℝ := total_horizontal_exposed_length + total_vertical_visible_length

-- Prove that the perimeter is 14 inches
theorem T_perimeter_is_14 : T_perimeter = 14 := by
  sorry

end T_perimeter_is_14_l724_724533


namespace simplify_fraction_l724_724107

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : 
  ( (x^2 + 1) / (x - 1) - (2*x) / (x - 1) ) = x - 1 :=
by
  -- Your proof steps would go here.
  sorry

end simplify_fraction_l724_724107


namespace least_gamma_l724_724457

theorem least_gamma (n : ℕ) (hn : n ≥ 2)
    (x : Fin n → ℝ) (hx : (∀ i, x i > 0) ∧ (∑ i, x i = 1))
    (y : Fin n → ℝ) (hy : (∀ i, 0 ≤ y i ∧ y i ≤ 1/2) ∧ (∑ i, y i = 1)) :
    ∃ γ, γ = 1 / (2 * (n - 1)^(n - 1)) ∧ (∏ i, x i) ≤ γ * (∑ i, x i * y i) := 
by
  sorry

end least_gamma_l724_724457


namespace drawings_on_last_page_l724_724181

theorem drawings_on_last_page :
  let n_notebooks := 10 
  let p_pages := 50
  let d_original := 5
  let d_new := 8
  let total_drawings := n_notebooks * p_pages * d_original
  let total_pages_new := total_drawings / d_new
  let filled_complete_pages := 6 * p_pages
  let drawings_on_last_page := total_drawings - filled_complete_pages * d_new - 40 * d_new
  drawings_on_last_page == 4 :=
  sorry

end drawings_on_last_page_l724_724181


namespace proof_ab_greater_ac_l724_724331

theorem proof_ab_greater_ac (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : 
  a * b > a * c :=
by sorry

end proof_ab_greater_ac_l724_724331


namespace least_score_to_play_final_l724_724433

-- Definitions based on given conditions
def num_teams := 2021

def match_points (outcome : String) : ℕ :=
  match outcome with
  | "win"  => 3
  | "draw" => 1
  | "loss" => 0
  | _      => 0

def brazil_won_first_match : Prop := True

def ties_advantage (bfc_score other_team_score : ℕ) : Prop :=
  bfc_score = other_team_score

-- Theorem statement
theorem least_score_to_play_final (bfc_has_tiebreaker : (bfc_score other_team_score : ℕ) → ties_advantage bfc_score other_team_score)
  (bfc_first_match_won : brazil_won_first_match) :
  ∃ (least_score : ℕ), least_score = 2020 := sorry

end least_score_to_play_final_l724_724433


namespace degree_of_g_l724_724417

theorem degree_of_g 
  (f : Polynomial ℤ)
  (g : Polynomial ℤ) 
  (h₁ : f = -9 * Polynomial.X^5 + 4 * Polynomial.X^3 - 2 * Polynomial.X + 6)
  (h₂ : (f + g).degree = 2) :
  g.degree = 5 :=
sorry

end degree_of_g_l724_724417


namespace triangle_area_of_parabola_l724_724725

theorem triangle_area_of_parabola (p : ℝ) (hp : p > 0) (A B : ℝ × ℝ) (F : ℝ × ℝ) :
    let parabola := λ x y : ℝ, y^2 = 2 * p * x
    let focus := (p / 2, 0)
    let line := λ x y : ℝ, y = √3 * (x - p / 2)
    A.2 > 0  -- A is above the x-axis
    ∧ parabola A.1 A.2
    ∧ parabola B.1 B.2
    ∧ line A.1 A.2
    ∧ line B.1 B.2
    ∧ F = focus
    → let S := 1 / 2 * (A.1 - 0) * A.2
    in S = ( √3 / 4 ) * p^2 :=
by
  sorry

end triangle_area_of_parabola_l724_724725


namespace digit_difference_l724_724580

variable (X Y : ℕ)

theorem digit_difference (h : 10 * X + Y - (10 * Y + X) = 27) : X - Y = 3 :=
by
  sorry

end digit_difference_l724_724580


namespace determine_digit_l724_724656

theorem determine_digit (Θ : ℕ) (hΘ : Θ > 0 ∧ Θ < 10) (h : 630 / Θ = 40 + 3 * Θ) : Θ = 9 :=
sorry

end determine_digit_l724_724656


namespace triangle_area_QPO_correct_l724_724008

noncomputable def parallelogram_area_k (k : ℝ) : Prop :=
  ∃ (A B C D P Q N M O : ℝ × ℝ),
  is_parallelogram A B C D ∧
  trisects_segment D P B C N ∧
  extends_meeting_line D P A B P ∧
  trisects_segment C Q A D M ∧
  extends_meeting_line C Q A B Q ∧
  intersection_point D P C Q O ∧
  parallelogram_area A B C D = k

noncomputable def triangle_area_QPO (k : ℝ) : ℝ :=
  8 * k / 9

theorem triangle_area_QPO_correct (k : ℝ) :
  parallelogram_area_k k → triangle_area_QPO k = 8 * k / 9 :=
by
  intros hpk
  sorry

end triangle_area_QPO_correct_l724_724008


namespace reciprocal_of_neg_three_l724_724153

theorem reciprocal_of_neg_three : ∃ x : ℚ, (-3) * x = 1 ∧ x = (-1) / 3 := sorry

end reciprocal_of_neg_three_l724_724153


namespace largest_expression_is_d_l724_724663

def expr_a := 3 + 0 + 4 + 8
def expr_b := 3 * 0 + 4 + 8
def expr_c := 3 + 0 * 4 + 8
def expr_d := 3 + 0 + 4 * 8
def expr_e := 3 * 0 * 4 * 8
def expr_f := (3 + 0 + 4) / 8

theorem largest_expression_is_d : 
  expr_d = 35 ∧ 
  expr_a = 15 ∧ 
  expr_b = 12 ∧ 
  expr_c = 11 ∧ 
  expr_e = 0 ∧ 
  expr_f = 7 / 8 ∧
  35 > 15 ∧ 
  35 > 12 ∧ 
  35 > 11 ∧ 
  35 > 0 ∧ 
  35 > 7 / 8 := 
by
  sorry

end largest_expression_is_d_l724_724663


namespace triangle_ADE_perimeter_l724_724374

noncomputable def ellipse_perimeter (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) (e : ℝ) (h₃ : e = (1 / 2)) 
(F₁ F₂ : ℝ × ℝ) (h₄ : F₁ ≠ F₂) (D E : ℝ × ℝ) (h₅ : |D - E| = 6) : ℝ :=
  let c := (sqrt (a ^ 2 - b ^ 2)) in
  let A := (0, b) in
  let AD := sqrt ((fst D) ^ 2 + (snd D - b) ^ 2) in
  let AE := sqrt ((fst E) ^ 2 + (snd E - b) ^ 2) in
  AD + AE + |D - E|

theorem triangle_ADE_perimeter (a b : ℝ) (h₁ : a > b > 0) (e : ℝ) (h₂ : e = (1 / 2))
(F₁ F₂ : ℝ × ℝ) (h₃ : F₁ ≠ F₂)
(D E : ℝ × ℝ) (h₄ : |D - E| = 6) : 
  ellipse_perimeter a b (and.left h₁) (and.right h₁) e h₂ F₁ F₂ h₃ D E h₄ = 19 :=
sorry

end triangle_ADE_perimeter_l724_724374


namespace not_always_two_colors_not_always_three_colors_l724_724173

-- a) Definitions based on conditions
def match (α : Type) := α × α
def color := Nat -- assuming 0 represents one color and 1 represents the other for 2 coloring, and 0, 1, 2 for 3 coloring

def ends_different_colors (c : color → Prop) (m : match color) : Prop :=
  c m.1 ≠ c m.2

def touching_ends_same_color (c : color → Prop) (m1 m2 : match color) : Prop :=
  c m1.2 = c m2.1

-- b) Lean 4 statements for the two questions

theorem not_always_two_colors (m : list (match α)) (c : color → Prop) :
  ¬ (∀ (m : match color), ends_different_colors c m ∧
    ∀ (m1 m2 : match color), touching_ends_same_color c m1 m2) :=
sorry

theorem not_always_three_colors (m : list (match α)) (c : color → Prop) :
  ¬ (∀ (m : match color), ends_different_colors c m ∧
    ∀ (m1 m2 : match color), touching_ends_same_color c m1 m2) :=
sorry

end not_always_two_colors_not_always_three_colors_l724_724173


namespace height_to_top_floor_l724_724860

def total_height : ℕ := 1454
def antenna_spire_height : ℕ := 204

theorem height_to_top_floor : (total_height - antenna_spire_height) = 1250 := by
  sorry

end height_to_top_floor_l724_724860


namespace work_on_monday_l724_724073

variable (Tuesday Wednesday Thursday Friday TotalHours Monday : ℚ)

axiom tuesday_hours : Tuesday = 1 / 2
axiom wednesday_hours : Wednesday = 2 / 3
axiom thursday_hours : Thursday = 5 / 6
axiom friday_hours : Friday = 75 / 60
axiom total_hours : TotalHours = 4

theorem work_on_monday :
  Monday = TotalHours - (Tuesday + Wednesday + Thursday + Friday) → Monday = 3 / 4 := sorry

end work_on_monday_l724_724073


namespace find_complex_numbers_l724_724667

theorem find_complex_numbers (z : ℂ) (h : z^2 = -100 - 48 * complex.i) :
  z = 2 - 12 * complex.i ∨ z = -2 + 12 * complex.i :=
sorry

end find_complex_numbers_l724_724667


namespace ages_of_people_l724_724277

-- Define types
variable (A M B C : ℕ)

-- Define conditions as hypotheses
def conditions : Prop :=
  A = 2 * M ∧
  A = 4 * B ∧
  M = A - 10 ∧
  C = B + 3 ∧
  C = M / 2

-- Define what we want to prove
theorem ages_of_people :
  (conditions A M B C) →
  A = 20 ∧
  M = 10 ∧
  B = 2 ∧
  C = 5 :=
by
  sorry

end ages_of_people_l724_724277


namespace length_bc_is_radius_l724_724771

noncomputable def problem_conditions : Prop :=
  let O := (0 : ℝ)
  let A := (1 : ℝ)  -- using arbitrary positions, we'll focus on properties.
  let D := (2 : ℝ)
  let B := (3 : ℝ)
  let C := (4 : ℝ) in
  let radius := 8 in
  let angle_abo := 90 in
  -- Central angle subtended by arc CD
  let central_angle_cd := 90 in
  (D - A = 2 * radius) ∧ -- AD is a diameter, hence twice radius
  (C - O = radius) ∧    -- O is the center and CO is the radius
  (B - O = radius) ∧    -- O is the center and BO is the radius
  (angle_abo / 2 = 45) ∧ -- Inscribed angle theorem
  (central_angle_cd = 90) -- Central angle is 90 degrees

theorem length_bc_is_radius : ∀ (O A D B C : ℝ), problem_conditions → 
  let radius := 8 in
  let length_bc := B - C in
  length_bc = radius :=
sorry

end length_bc_is_radius_l724_724771


namespace complete_wall_in_time_l724_724452

theorem complete_wall_in_time (avery_time : ℝ) (tom_time : ℝ) (initial_work_duration : ℝ) : 
  avery_time = 3 → tom_time = 2.5 → initial_work_duration = 1 → 
  (let avery_rate := 1 / avery_time in
   let tom_rate := 1 / tom_time in
   let combined_rate := avery_rate + tom_rate in
   let work_done := combined_rate * initial_work_duration in
   let remaining_work := 1 - work_done in
   let tom_remaining_time := remaining_work / tom_rate in
   tom_remaining_time = 2 / 3) :=
by
  intros h1 h2 h3
  let avery_rate := 1 / avery_time
  let tom_rate := 1 / tom_time
  let combined_rate := avery_rate + tom_rate
  let work_done := combined_rate * initial_work_duration
  let remaining_work := 1 - work_done
  let tom_remaining_time := remaining_work / tom_rate
  have : tom_remaining_time = 2 / 3 := sorry
  exact this

end complete_wall_in_time_l724_724452


namespace base_five_product_correct_l724_724207

-- Define a function that multiplies two base five numbers represented as natural numbers
noncomputable def base_five_mult (a b : Nat) : Nat :=
  -- Convert to decimal, multiply, and convert back to base five representation
  let a_dec := Nat.ofDigits 5 (a.digits 10)
  let b_dec := Nat.ofDigits 5 (b.digits 10)
  let product_dec := a_dec * b_dec
  product_dec.ofDigits 5

-- Define specific base five numbers
def num1 := 1 * 5^2 + 2 * 5^1 + 1 * 5^0 -- 121_5 in decimal
def num2 := 1 * 5^1 + 1 * 5^0 -- 11_5 in decimal

-- Define the expected product base five number
def expected_product := 1 * 5^3 + 3 * 5^2 + 3 * 5^1 + 1 * 5^0 -- 1331_5 in decimal

-- The main statement to prove
theorem base_five_product_correct : base_five_mult num1 num2 = expected_product := by
  sorry

end base_five_product_correct_l724_724207


namespace range_of_a_l724_724871

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  real.log x - x / a

theorem range_of_a (a : ℝ) (h : 0 < a) :
  (∃ x₀ : ℝ, ∀ x₁ : ℝ, 1 ≤ x₁ ∧ x₁ ≤ 2 → f x₁ a < f x₀ a) →
    (a ∈ Ioo 0 1 ∨ a ∈ Ioi 2) :=
by
  sorry

end range_of_a_l724_724871


namespace share_of_B_l724_724510

noncomputable def problem_statement (A B C : ℝ) : Prop :=
  A + B + C = 595 ∧ A = (2/3) * B ∧ B = (1/4) * C

theorem share_of_B (A B C : ℝ) (h : problem_statement A B C) : B = 105 :=
by
  -- Proof omitted
  sorry

end share_of_B_l724_724510


namespace value_of_a6_l724_724333

def sequence (n : ℕ) : ℤ :=
  if n = 1 then 3 else if n = 2 then 6 else sequence (n - 1) - sequence (n - 2)

theorem value_of_a6 : sequence 6 = -3 := by
  sorry

end value_of_a6_l724_724333


namespace triangle_inequality_l724_724465

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (triangle_cond : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2*(b + c - a) + b^2*(c + a - b) + c^2*(a + b - c) ≤ 3*a*b*c :=
by
  sorry

end triangle_inequality_l724_724465


namespace painting_cost_l724_724805

-- Define contributions
def JudsonContrib := 500
def KennyContrib := JudsonContrib + 0.20 * JudsonContrib
def CamiloContrib := KennyContrib + 200

-- Define total cost
def TotalCost := JudsonContrib + KennyContrib + CamiloContrib

-- Theorem to prove
theorem painting_cost : TotalCost = 1900 :=
by 
  -- Calculate Kenny's contribution
  have hK : KennyContrib = 600 := by 
    simp [KennyContrib, JudsonContrib]
    sorry -- additional steps would go here, we use sorry to skip details

  -- Calculate Camilo's contribution
  have hC : CamiloContrib = 800 := by 
    simp [CamiloContrib, hK]
    sorry -- additional steps would go here, we use sorry to skip details

  -- Calculate total cost
  simp [TotalCost, JudsonContrib, hK, hC]
  sorry -- additional steps would go here, we use sorry to skip details

end painting_cost_l724_724805


namespace distinct_absolute_differences_impossible_l724_724644

theorem distinct_absolute_differences_impossible :
  ∀ (a_1 a_2 b_1 b_2 b_3 c_1 c_2 d_1 d_2 d_3 : ℕ),
  a_1 ≠ a_2 ∧ a_1 ≠ b_1 ∧ a_1 ≠ b_2 ∧ a_1 ≠ b_3 ∧ a_2 ≠ b_1 ∧ a_2 ≠ b_2 ∧ a_2 ≠ b_3 ∧
  c_1 ≠ c_2 ∧ c_1 ≠ d_1 ∧ c_1 ≠ d_2 ∧ c_1 ≠ d_3 ∧ c_2 ≠ d_1 ∧ c_2 ≠ d_2 ∧ c_2 ≠ d_3 ∧
  a_1 ≠ c_1 ∧ a_2 ≠ c_2 ∧ b_1 ≠ b_2 ∧ b_1 ≠ b_3 ∧ b_2 ≠ b_3 ∧ d_1 ≠ d_2 ∧ d_1 ≠ d_3 ∧ d_2 ≠ d_3 ∧
  (∀ n ∈ {a_1, a_2, b_1, b_2, b_3, c_1, c_2, d_1, d_2, d_3}, n ≤ 14) →
  ¬ (list.pairwise (≠) [|a_1 - b_1|, |a_1 - b_2|, |a_1 - b_3|,
                        |a_2 - b_1|, |a_2 - b_2|, |a_2 - b_3|,
                        |c_1 - d_1|, |c_1 - d_2|, |c_1 - d_3|,
                        |c_2 - d_1|, |c_2 - d_2|, |c_2 - d_3|,
                        |a_1 - c_1|, |a_2 - c_2|])
) := sorry

end distinct_absolute_differences_impossible_l724_724644


namespace decimal_to_fraction_l724_724964

theorem decimal_to_fraction :
  (368 / 100 : ℚ) = (92 / 25 : ℚ) := by
  sorry

end decimal_to_fraction_l724_724964


namespace part1_part2_part3_l724_724595

-- Definitions for the conditions
def not_divisible_by_2_or_3 (k : ℤ) : Prop :=
  ¬(k % 2 = 0 ∨ k % 3 = 0)

def form_6n1_or_6n5 (k : ℤ) : Prop :=
  ∃ (n : ℤ), k = 6 * n + 1 ∨ k = 6 * n + 5

-- Part 1
theorem part1 (k : ℤ) (h : not_divisible_by_2_or_3 k) : form_6n1_or_6n5 k :=
sorry

-- Part 2
def form_6n1 (a : ℤ) : Prop :=
  ∃ (n : ℤ), a = 6 * n + 1

def form_6n5 (a : ℤ) : Prop :=
  ∃ (n : ℤ), a = 6 * n + 5

theorem part2 (a b : ℤ) (ha : form_6n1 a ∨ form_6n5 a) (hb : form_6n1 b ∨ form_6n5 b) :
  form_6n1 (a * b) :=
sorry

-- Part 3
theorem part3 (a b : ℤ) (ha : form_6n1 a) (hb : form_6n5 b) :
  form_6n5 (a * b) :=
sorry

end part1_part2_part3_l724_724595


namespace DebateClubOfficerSelection_l724_724862

-- Definitions based on the conditions
def members : Finset ℕ := Finset.range 25 -- Members are indexed from 0 to 24
def Simon := 0
def Rachel := 1
def John := 2

-- Conditions regarding the officers
def is_officer (x : ℕ) (pres sec tre : ℕ) : Prop := 
  x = pres ∨ x = sec ∨ x = tre

def Simon_condition (pres sec tre : ℕ) : Prop :=
  (is_officer Simon pres sec tre) → (is_officer Rachel pres sec tre)

def Rachel_condition (pres sec tre : ℕ) : Prop :=
  (is_officer Rachel pres sec tre) → (is_officer Simon pres sec tre) ∨ (is_officer John pres sec tre)

-- Statement of the problem in Lean
theorem DebateClubOfficerSelection : ∃ (pres sec tre : ℕ), 
  pres ≠ sec ∧ sec ≠ tre ∧ pres ≠ tre ∧ 
  pres ∈ members ∧ sec ∈ members ∧ tre ∈ members ∧ 
  Simon_condition pres sec tre ∧
  Rachel_condition pres sec tre :=
sorry

end DebateClubOfficerSelection_l724_724862


namespace sum_geometric_sequence_n_terms_l724_724713

-- Given conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 1

def sum_first_n_terms (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, x i

-- Problem statement
theorem sum_geometric_sequence_n_terms (x : ℕ → ℝ) (log_x : ℕ → ℝ) (x1 : ℝ) 
  (h1 : x = λ n, x1 * 2 ^ n)
  (h2 : log_x = λ n, real.log (x n) / real.log 2)
  (h_arithmetic : arithmetic_sequence log_x)
  (h_sum_first_100 : sum_first_n_terms x 100 = 100) : 
  sum_first_n_terms x 200 = 100 * (1 + 2 ^ 100) :=
sorry

end sum_geometric_sequence_n_terms_l724_724713


namespace negation_of_universal_is_existential_l724_724132

theorem negation_of_universal_is_existential :
  ¬ (∀ x : ℝ, x^2 - 2 * x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2 * x + 4 > 0) :=
by
  sorry

end negation_of_universal_is_existential_l724_724132


namespace simplification_l724_724513

theorem simplification (a b c : ℤ) :
  (12 * a + 35 * b + 17 * c) + (13 * a - 15 * b + 8 * c) - (8 * a + 28 * b - 25 * c) = 17 * a - 8 * b + 50 * c :=
by
  sorry

end simplification_l724_724513


namespace sum_x_y_l724_724756

-- Introduce the conditions as variables and hypotheses
variables (x y : ℝ)
hypothesis h1 : 0.65 * 800 = 0.35 * x
hypothesis h2 : x = 2 * y

-- The goal is to prove that x + y = 2228.57 given the conditions
theorem sum_x_y (x y : ℝ) (h1 : 0.65 * 800 = 0.35 * x) (h2 : x = 2 * y) : x + y = 2228.57 :=
sorry

end sum_x_y_l724_724756


namespace jelly_beans_count_l724_724792

theorem jelly_beans_count :
  (let large_glass_beans := 50 in
  let small_glass_beans := 25 in
  let large_glasses := 5 in
  let small_glasses := 3 in
  (large_glass_beans * large_glasses + small_glass_beans * small_glasses) = 325) :=
by
  let large_glass_beans := 50
  let small_glass_beans := 25
  let large_glasses := 5
  let small_glasses := 3
  have h_large : large_glass_beans * large_glasses = 250 := by norm_num
  have h_small : small_glass_beans * small_glasses = 75 := by norm_num
  have h_total : 250 + 75 = 325 := by norm_num
  exact h_total

end jelly_beans_count_l724_724792


namespace at_least_two_in_front_of_correct_name_card_one_sit_at_correct_place_no_more_possible_l724_724596

-- Proof Problem 1
theorem at_least_two_in_front_of_correct_name_card :
  ∃ (r : ℕ) (r < 15), rotate_table r ∧ sits_in_front_of_own_name_card ≥ 2 :=
by
sorry

-- Proof Problem 2
theorem one_sit_at_correct_place_no_more_possible :
  ∃ (arrangement : list ℕ) (h : length arrangement = 15),
    (∀ r < 15, rotate_table r ∧ sits_in_front_of_own_name_card = 1) :=
by
sorry

end at_least_two_in_front_of_correct_name_card_one_sit_at_correct_place_no_more_possible_l724_724596


namespace sin_theta_smallest_angle_l724_724456

variable (a b c : ℝ) (θ : ℝ)

theorem sin_theta_smallest_angle (h1 : a^2 + b^2 = c^2) (h2 : θ = real.arcsin (a/c)) (h3 : (1/a)^2 + (1/b)^2 = (1/c)^2) :
  real.sin θ = (real.sqrt 5 - 1) / 2 := 
sorry

end sin_theta_smallest_angle_l724_724456


namespace two_digit_number_conditions_l724_724212

-- Definitions for two-digit number and its conditions
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ := n / 10
def units_digit (n : ℕ) : ℕ := n % 10
def sum_of_digits (n : ℕ) : ℕ := tens_digit n + units_digit n

-- The proof problem statement in Lean 4
theorem two_digit_number_conditions (N : ℕ) (c d : ℕ) :
  is_two_digit_number N ∧ N = 10 * c + d ∧ N' = N + 7 ∧ 
  N = 6 * sum_of_digits (N + 7) →
  N = 24 ∨ N = 78 :=
by
  sorry

end two_digit_number_conditions_l724_724212


namespace largest_inscribed_square_l724_724789

noncomputable def side_length_of_largest_inscribed_square (side_len : ℝ) : ℝ :=
  let s : ℝ := (20 * real.sqrt 2) / (1 + 2 * real.sqrt 3) in
  let y : ℝ := 10 - (10 / (1 + 2 * real.sqrt 3)) in
  y

theorem largest_inscribed_square (side_len : ℝ) (h : side_len = 20) :
  side_length_of_largest_inscribed_square side_len = 10 - (10 / (1 + 2 * real.sqrt 3)) :=
by
  sorry

end largest_inscribed_square_l724_724789


namespace projection_of_AB_onto_AC_is_correct_l724_724782

-- Define the points A, B, and C
def pointA : ℝ × ℝ := (3, 4)
def pointB : ℝ × ℝ := (1, 8)
def pointC : ℝ × ℝ := (-1, 6)

-- Define the vector subtraction function
def vector_sub (p q : ℝ × ℝ) : ℝ × ℝ := (p.1 - q.1, p.2 - q.2)

-- Define the dot product function
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the scalar multiplication of a vector
def scalar_mul (r : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (r * v.1, r * v.2)

-- Define the magnitude squared of a vector
def mag_squared (v : ℝ × ℝ) : ℝ := dot_product v v

-- Define vectors AB and AC
def vectorAB := vector_sub pointB pointA
def vectorAC := vector_sub pointC pointA

-- Calculate the projection vector
def projection_vector : ℝ × ℝ :=
  scalar_mul (dot_product vectorAB vectorAC / mag_squared vectorAC) vectorAC

-- Statement to be proven
theorem projection_of_AB_onto_AC_is_correct :
  projection_vector = (-16 / 5, 8 / 5) :=
by
  sorry

end projection_of_AB_onto_AC_is_correct_l724_724782


namespace simplify_expression_l724_724095

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
    ((x^2 + 1) / (x - 1)) - (2 * x / (x - 1)) = x - 1 := 
by
    sorry

end simplify_expression_l724_724095


namespace problem_1_problem_2_l724_724734

open Real

-- Define the vectors a, b, c based on the given conditions
def vector_a (x : ℝ) : ℝ × ℝ := (cos (3/2 * x), sin (3/2 * x))
def vector_b (x : ℝ) : ℝ × ℝ := (cos (x / 2), -sin (x / 2))
def vector_c : ℝ × ℝ := (1, -1)

-- Condition: x ∈ [-π/2, π/2]
def x_in_interval (x : ℝ) : Prop := -π/2 ≤ x ∧ x ≤ π/2

-- Problem 1: Prove orthogonality of the vectors
theorem problem_1 (x : ℝ) (hx : x_in_interval x) :
  let a := vector_a x
  let b := vector_b x
  (a.1 + b.1) * (a.1 - b.1) + (a.2 + b.2) * (a.2 - b.2) = 0 :=
sorry

-- Problem 2: Find the maximum and minimum values of f(x)
def f (x : ℝ) : ℝ :=
  let a := vector_a x
  let b := vector_b x
  let c := vector_c
  (norm (a.1 + c.1, a.2 + c.2) ^ 2 - 3) * (norm (b.1 + c.1, b.2 + c.2) ^ 2 - 3)

theorem problem_2 (x : ℝ) (hx : x_in_interval x) :
  (f x ≤ 9 / 2 ∧ f x ≥ -8) :=
sorry

end problem_1_problem_2_l724_724734


namespace product_of_roots_is_100_l724_724814

theorem product_of_roots_is_100 :
  let a := real.cbrt 11
  let b := real.cbrt 121
  let P := Polynomial.X^3 - 100 : Polynomial ℚ
  root_of_P := a + b
  (P.root root_of_P)
  ∃ roots, (∀ r ∈ roots, P.root r) ∧ (roots.prod id = 100) :=
by
  let a := real.cbrt 11
  let b := real.cbrt 121
  let P := Polynomial.X^3 - 100 : Polynomial ℚ
  let root_of_P := a + b
  have h := P.root root_of_P
  use a + b, exp(iota xi.sum within between before after χ (≤ 100 imply id))
  sorry

end product_of_roots_is_100_l724_724814


namespace cube_volume_l724_724953

theorem cube_volume (side_area : ℝ) (h : side_area = 64) : 
  let side_length := Real.sqrt side_area in
  let volume := side_length ^ 3 in
  volume = 512 := 
by
  -- Define the side area and assume it equals 64
  let side_length := Real.sqrt side_area
  have h_side_length : side_length = 8 := by
    rw [h, Real.sqrt_eq_rpow]
    norm_num
  -- Calculate the volume as side_length ^ 3
  let volume := side_length ^ 3
  have h_volume : volume = 512 := by
    rw [h_side_length]
    norm_num
  -- Conclude with the volume
  exact h_volume

end cube_volume_l724_724953


namespace coefficient_of_x10_in_expansion_l724_724865

theorem coefficient_of_x10_in_expansion : 
  (C 10 6) = 210 :=
by
  sorry

end coefficient_of_x10_in_expansion_l724_724865


namespace decimal_to_fraction_simplify_l724_724969

theorem decimal_to_fraction_simplify (d : ℚ) (h : d = 3.68) : d = 92 / 25 :=
by
  rw h
  sorry

end decimal_to_fraction_simplify_l724_724969


namespace min_impact_distance_l724_724189

theorem min_impact_distance :
  ∃ x ∈ Ioo 0 20, (∀ y ∈ Ioo 0 x, y >= 4 * Real.sqrt 10 * y) ∧ (∀ z ∈ Ioo x 20, z >= 4 * Real.sqrt 10 * z) :=
sorry

end min_impact_distance_l724_724189


namespace machineB_produces_100_parts_in_40_minutes_l724_724478

-- Define the given conditions
def machineA_rate := 50 / 10 -- Machine A's rate in parts per minute
def machineB_rate := machineA_rate / 2 -- Machine B's rate in parts per minute

-- Machine A produces 50 parts in 10 minutes
def machineA_50_parts_time : ℝ := 10

-- Machine B's time to produce 100 parts (The question)
def machineB_100_parts_time : ℝ := 40

-- Proving that Machine B takes 40 minutes to produce 100 parts
theorem machineB_produces_100_parts_in_40_minutes :
    machineB_100_parts_time = 40 :=
by
  sorry

end machineB_produces_100_parts_in_40_minutes_l724_724478


namespace exists_quadratic_polynomial_with_constant_term_neg2_l724_724976

-- Define what it means to be a quadratic polynomial with the mentioned conditions
def is_quadratic_polynomial (p : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c d e : ℝ), p = λ x y, a * x^2 + b * x * y + c * y^2 + d * x + e * y + (-2)

-- The main proposition we need to prove
theorem exists_quadratic_polynomial_with_constant_term_neg2 :
  ∃ p : ℝ → ℝ → ℝ, is_quadratic_polynomial p :=
begin
  use (λ x y, 15 * x^2 - y - 2),
  unfold is_quadratic_polynomial,
  use [15, 0, 0, 0, -1],
  simp,
end

end exists_quadratic_polynomial_with_constant_term_neg2_l724_724976


namespace inequality_for_sum_of_powers_l724_724045

theorem inequality_for_sum_of_powers
  {n : ℕ} (n_ge_3 : 3 ≤ n)
  {x : Fin n → ℝ} (hx : ∀ i j, i ≠ j → x i ≠ x j)
  (h_pos : ∀ i, 0 < x i) 
  (h_sum : ∑ i, x i = n)
  {k t : ℝ} (k_gt_t : k > t) (t_gt_1 : t > 1) :
  (∑ i, x i ^ k - n) / (k - 1) ≥ (∑ i, x i ^ t - n) / (t - 1) :=
sorry

end inequality_for_sum_of_powers_l724_724045


namespace pages_torn_and_sheets_calculation_l724_724902

theorem pages_torn_and_sheets_calculation : 
  (∀ (n : ℕ), (sheet_no n) = (n + 1) / 2 → (2 * (n + 1) / 2) - 1 = n ∨ 2 * (n + 1) / 2 = n) →
  let first_page := 185 in
  let last_page := 518 in
  last_page = 518 → 
  ((last_page - first_page + 1) / 2) = 167 := 
by
  sorry

end pages_torn_and_sheets_calculation_l724_724902


namespace smallest_possible_knight_liar_pairs_l724_724840

theorem smallest_possible_knight_liar_pairs :
  ∃ (N : ℕ), (∀ (knights liars : ℕ), knights = 100 ∧ liars = 100 →
  (∀ (residents : list (bool × list ℕ)), 
    length residents = 200 ∧ 
    (∀ r ∈ residents, (r.fst = tt ∨ r.fst = ff) ∧ r.snd ≠ []) ∧
    (countp (λ r, all_friends_are_knights r residents) residents = 100) ∧ 
    (countp (λ r, all_friends_are_liars r residents) residents = 100) →
    count_knight_liar_pairs residents ≥ 50 ∧
    N = 50)) :=
begin
  sorry
end

def all_friends_are_knights (res : bool × list ℕ) (residents : list (bool × list ℕ)) : bool :=
  ∀ id ∈ res.snd, (residents.nth id).map prod.fst = some tt

def all_friends_are_liars (res : bool × list ℕ) (residents : list (bool × list ℕ)) : bool :=
  ∀ id ∈ res.snd, (residents.nth id).map prod.fst = some ff

def count_knight_liar_pairs (residents : list (bool × list ℕ)) : ℕ :=
  residents.sum (λ res, res.snd.count (λ friend_id, 
    match residents.nth friend_id with
    | some (tt, _) => res.fst = ff
    | some (ff, _) => res.fst = tt
    | none => false
    end))

end smallest_possible_knight_liar_pairs_l724_724840


namespace soccer_team_selection_l724_724620

theorem soccer_team_selection (players : Finset ℕ) (quadruplets : Finset ℕ)
  (h_players : players.card = 15) (h_quadruplets : quadruplets.card = 4)
  (h_quadruplets_subset : quadruplets ⊆ players) (h_starters : Finset ℕ)
  (h_starters.card = 7) (h_quadruplets_in_starters : (quadruplets ∩ h_starters).card = 2) :
  (Nat.choose 4 2) * (Nat.choose 11 5) = 2772 :=
by sorry

end soccer_team_selection_l724_724620


namespace Eugene_buys_two_pairs_of_shoes_l724_724002

theorem Eugene_buys_two_pairs_of_shoes :
  let tshirt_price : ℕ := 20
  let pants_price : ℕ := 80
  let shoes_price : ℕ := 150
  let discount_rate : ℕ := 10
  let discounted_price (price : ℕ) := price - (price * discount_rate / 100)
  let total_price (count1 count2 count3 : ℕ) (price1 price2 price3 : ℕ) :=
    (count1 * price1) + (count2 * price2) + (count3 * price3)
  let total_amount_paid : ℕ := 558
  let tshirts_bought : ℕ := 4
  let pants_bought : ℕ := 3
  let amount_left := total_amount_paid - discounted_price (tshirts_bought * tshirt_price + pants_bought * pants_price)
  let shoes_bought := amount_left / discounted_price shoes_price
  shoes_bought = 2 := 
sorry

end Eugene_buys_two_pairs_of_shoes_l724_724002


namespace three_point_sixty_eight_as_fraction_l724_724974

theorem three_point_sixty_eight_as_fraction : 3.68 = 92 / 25 := 
by 
  sorry

end three_point_sixty_eight_as_fraction_l724_724974


namespace g_x_plus_3_l724_724471

def g (x : ℝ) : ℝ := (x * (x + 3)) / 3

theorem g_x_plus_3 (x : ℝ) : g (x + 3) = (x^2 + 9 * x + 18) / 3 :=
by
  sorry

end g_x_plus_3_l724_724471


namespace probability_A_fires_l724_724996

theorem probability_A_fires
  (p_A_fires_first_attempt : ℚ := 1/6)
  (p_A_fires_third_attempt : ℚ := (5/6)^2 * (1/6))
  (p_A_fires_fifth_attempt : ℚ := (5/6)^4 * (1/6)) :
  ∑ k : ℕ, (5/6)^(2*k) * (1/6) = 6/11 :=
by
  sorry

end probability_A_fires_l724_724996


namespace lucia_outfits_l724_724069

theorem lucia_outfits (shoes : ℕ) (dresses : ℕ) (hats : ℕ)
                      (h_shoes : shoes = 3) (h_dresses : dresses = 5) (h_hats : hats = 4) : 
                      shoes * dresses * hats = 60 :=
by
  rw [h_shoes, h_dresses, h_hats]
  exact Nat.mul_eq_mul_right_iff.mpr (Or.inl (by norm_num))


end lucia_outfits_l724_724069


namespace last_two_digits_x_2012_l724_724473

noncomputable def x : ℕ → ℤ
| 1 := 1
| 2 := 1
| n := if h : n ≥ 3 then x (n-1) * y (n-2) + x (n-2) * y (n-1) else 0

noncomputable def y : ℕ → ℤ
| 1 := 1
| 2 := 1
| n := if h : n ≥ 3 then y (n-1) * y (n-2) - x (n-1) * x (n-2) else 0

def last_two_digits (n : ℤ) : ℤ :=
n % 100

theorem last_two_digits_x_2012 :
last_two_digits (abs (x 2012)) = 84 :=
sorry

end last_two_digits_x_2012_l724_724473


namespace magnitude_of_z_l724_724340

theorem magnitude_of_z (z : ℂ) (h : z * complex.I = 1 + complex.I) : complex.abs z = real.sqrt 2 :=
sorry

end magnitude_of_z_l724_724340


namespace perimeter_of_triangle_ADE_l724_724367

theorem perimeter_of_triangle_ADE
  (a b : ℝ) (F1 F2 A : ℝ × ℝ) (D E : ℝ × ℝ) 
  (h_ellipse : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1)
  (h_a_gt_b : a > b)
  (h_b_gt_0 : b > 0)
  (h_eccentricity : ∃ c, c / a = 1 / 2 ∧ a^2 - b^2 = c^2)
  (h_F1_F2 : ∀ F1 F2, distance F1 (0, 0) = distance F2 (0, 0) ∧ F1 ≠ F2 ∧ 
                       ∀ P : ℝ × ℝ, (distance P F1 + distance P F2 = 2 * a) ↔ (x : ℝ)(y : ℝ) (h_ellipse x y))
  (h_line_DE : ∃ k, ∃ c, ∀ x F1 A, (2 * a * x/(sqrt k^2 + 1)) = |DE|
  (h_length_DE : |DE| = 6)
  (h_A_vertex : A = (0, b))
  : ∃ perim : ℝ, perim = 13 :=
sorry

end perimeter_of_triangle_ADE_l724_724367


namespace simplify_expression_l724_724092

variable (x : ℝ)

theorem simplify_expression (h : x ≠ 1) : (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 :=
by sorry

end simplify_expression_l724_724092


namespace josephine_milk_containers_l724_724074

theorem josephine_milk_containers :
  ∃ x : ℝ, (3 * x + 1.5 + 2.5 = 10) ∧ (x = 2) :=
by
  use 2
  split
  · calc
    3 * 2 + 1.5 + 2.5 = 6 + 4 : by decide
    ... = 10 : by decide
  · rfl

end josephine_milk_containers_l724_724074


namespace max_ballpoint_pens_l724_724999

theorem max_ballpoint_pens (x y z : ℕ) (hx : x + y + z = 15)
  (hy : 10 * x + 40 * y + 60 * z = 500) (hz : x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1) :
  x ≤ 6 :=
sorry

end max_ballpoint_pens_l724_724999


namespace perimeter_of_triangle_ADE_l724_724364

noncomputable def ellipse_perimeter (a b : ℝ) (h : a > b) (e : ℝ) (he : e = 1/2) (h_ellipse : ∀ (x y : ℝ), 
                            x^2 / a^2 + y^2 / b^2 = 1) : ℝ :=
13 -- we assert that the perimeter is 13

theorem perimeter_of_triangle_ADE 
  (a b : ℝ) (h : a > b) (e : ℝ) (he : e = 1/2) 
  (C_eq : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1) 
  (upper_vertex_A : ℝ × ℝ)
  (focus_F1 : ℝ × ℝ)
  (focus_F2 : ℝ × ℝ)
  (line_through_F1_perpendicular_to_AF2 : ∀ x y, y = (√3 / 3) * (x + focus_F1.1))
  (points_D_E_on_ellipse : ∃ D E : ℝ × ℝ, line_through_F1_perpendicular_to_AF2 D.1 D.2 = true ∧
    line_through_F1_perpendicular_to_AF2 E.1 E.2 = true ∧ 
    (dist D E = 6)) :
  ∃ perimeter : ℝ, perimeter = ellipse_perimeter a b h e he C_eq :=
sorry

end perimeter_of_triangle_ADE_l724_724364


namespace minimum_value_of_expression_l724_724585

-- Definitions based on the problem conditions
def expr (x y : ℝ) : ℝ :=
  (3 * real.sqrt (2 * (1 + real.cos 2 * x)) - real.sqrt (8 - 4 * real.sqrt 3) * real.sin x + 2) *
  (3 + 2 * real.sqrt (11 - real.sqrt 3) * real.cos y - real.cos (2 * y))

-- Statement of the proof problem
theorem minimum_value_of_expression :
  ∃ m : ℝ, ∀ x y : ℝ, expr x y ≥ -33 ∧ m = -33 :=
sorry

end minimum_value_of_expression_l724_724585


namespace problem_1_problem_2_problem_3_problem_4_l724_724957

-- Problem 1
theorem problem_1 (e : ℝ × ℝ) (h : e = (-1, real.sqrt 3)) : 
  ∃ θ : ℝ, θ = 2 * real.pi / 3 ∧ (real.tan θ = e.2 / e.1) :=
sorry

-- Problem 2
theorem problem_2 (a : ℝ) : 
  ¬ (∃ a : ℝ, (∀ x y : ℝ, (a^2 * x - y + 1 = 0 ↔ x - a * y - 2 = 0) ↔ a = -1)) :=
sorry

-- Problem 3
theorem problem_3 (a : ℝ) : 
  (∀ a : ℝ, (∀ x y : ℝ, (a * x + 2 * y - 1 = 0 ↔ 8 * x + a * y + 2 - a = 0) ↔ a = -4)) :=
sorry

-- Problem 4
theorem problem_4 (θ : ℝ) (α : ℝ): 
  ∃ θ : ℝ, (0 ≤ θ ∧ θ < real.pi / 4) ∨ (3 * real.pi / 4 ≤ θ ∧ θ < real.pi) ∧ (real.tan θ = -real.sin α) :=
sorry

end problem_1_problem_2_problem_3_problem_4_l724_724957


namespace choose_signs_l724_724039

variables {n : ℕ} (hn : n > 1) (odd_n : n % 2 = 1)
variables {a : ℕ → ℝ}
variables (distinct_a : ∀ i j, i ≠ j → a i ≠ a j)
noncomputable def M := finset.max' (finset.range n) (finset.nonempty_of_ne_empty (ne_empty_of_card_pos hn))
noncomputable def m := finset.min' (finset.range n) (finset.nonempty_of_ne_empty (ne_empty_of_card_pos hn))

theorem choose_signs (hn : n > 1) (odd_n : n % 2 = 1) (distinct_a : ∀ i j, i ≠ j → a i ≠ a j) :
  ∃ (sign : ℕ → ℤ), m < ∑ i in finset.range n, sign i * a i ∧ ∑ i in finset.range n, sign i * a i < M :=
sorry

end choose_signs_l724_724039


namespace largest_circle_circumference_l724_724923

-- Conditions
def side_length : ℝ := 4
def radius : ℝ := (side_length * sqrt 2) / 2
def circumference : ℝ := 2 * π * radius

-- Statement
theorem largest_circle_circumference : circumference = 4 * π * sqrt 10 :=
by
  sorry

end largest_circle_circumference_l724_724923


namespace profit_ratio_l724_724981

theorem profit_ratio (p_investment q_investment : ℕ) (proportional_profits : ∀ r1 r2, r1 * p_investment = r2 * q_investment) 
(h_p_investment : p_investment = 60000) (h_q_investment : q_investment = 90000) : 
  (p_investment : ℚ) / q_investment = 2 / 3 := 
by 
  rw [h_p_investment, h_q_investment]
  norm_num
  sorry

end profit_ratio_l724_724981


namespace distance_from_point_to_line_l724_724505

open Real

noncomputable def point_to_line_distance (a b c x0 y0 : ℝ) : ℝ :=
  abs (a * x0 + b * y0 + c) / sqrt (a^2 + b^2)

theorem distance_from_point_to_line (a b c x0 y0 : ℝ) :
  point_to_line_distance a b c x0 y0 = abs (a * x0 + b * y0 + c) / sqrt (a^2 + b^2) :=
by
  sorry

end distance_from_point_to_line_l724_724505


namespace bicycle_cost_correct_l724_724640

def pay_rate : ℕ := 5
def hours_p_week : ℕ := 2 + 1 + 3
def weeks : ℕ := 6
def bicycle_cost : ℕ := 180

theorem bicycle_cost_correct :
  pay_rate * hours_p_week * weeks = bicycle_cost :=
by
  sorry

end bicycle_cost_correct_l724_724640


namespace geometric_seq_a3_eq_neg2_l724_724382

noncomputable def geometric_sequence (a : ℕ → ℝ) := 
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_seq_a3_eq_neg2
  (a : ℕ → ℝ)
  (p : ℝ)
  (h_geometric : geometric_sequence a)
  (h_eqn_roots : a 1 * a 5 = 4 ∧ a 1 + a 5 = p)
  (h_p_neg : p < 0) :
  a 3 = -2 :=
begin
  -- Proof will go here
  sorry
end

end geometric_seq_a3_eq_neg2_l724_724382


namespace max_size_of_F_l724_724317

open Set

-- Define D(x, y) according to the conditions provided
def D (x y : ℝ) : ℤ :=
  if x = y then 0 else int.floor (real.log (abs (x - y)) / real.log 2)

-- Define the conditions for scales in a set
def scales (F : Set ℝ) (x : ℝ) : Set ℤ :=
  {d | ∃ y ∈ F, x ≠ y ∧ D x y = d}

-- Define the main theorem
theorem max_size_of_F (F : Set ℝ) (k : ℕ) (hk : ∀ x ∈ F, |scales F x| ≤ k) :
  F.finite → F.to_finset.card ≤ 2^k := by
  sorry

end max_size_of_F_l724_724317


namespace inheritance_first_nonzero_digit_l724_724574

theorem inheritance_first_nonzero_digit (S : ℝ) (h : S ≠ 0) : 
  ∃ d : ℕ, d = 3 ∧ 
    (∃ n : ℕ, n > 0 ∧ fractional_part (n * (1 / 97)) * 10^n = d) := 
sorry

end inheritance_first_nonzero_digit_l724_724574


namespace mappings_from_A_to_B_mappings_from_A_to_B_with_preimage_l724_724511

section Problem1

variable (A B : Type)
variable [Fintype A] [Fintype B] [DecidableEq A] [DecidableEq B]
variable (hA : Fintype.card A = 4)
variable (hB : Fintype.card B = 3)

-- The number of mappings from A to B is B^A, which should be 81
theorem mappings_from_A_to_B : Fintype.card (A → B) = 3^4 := by
  rw [hA, hB]
  exact sorry -- This is to skip the proof

end Problem1

section Problem2

variable (A B : Type)
variable [Fintype A] [Fintype B] [DecidableEq A] [DecidableEq B]
variable (hA : Fintype.card A = 4)
variable (hB : Fintype.card B = 3)

-- The number of mappings such that every element in B has a preimage in A is 4
theorem mappings_from_A_to_B_with_preimage : 
  {f : A → B // Function.Surjective f}.card = 4 := by
  rw [hA, hB]
  exact sorry -- This is to skip the proof

end Problem2

end mappings_from_A_to_B_mappings_from_A_to_B_with_preimage_l724_724511


namespace probability_log_interval_correct_l724_724614

noncomputable def probability_log_interval : ℝ :=
  let lower_bound := 0
  let upper_bound := 2
  let event_lower_bound := -1
  let event_upper_bound := 1
  let transformation (x : ℝ) := x + 1 / 2
  let log_base := 1 / 2
  let probability := (λ (a b : ℝ), (b - a) / (upper_bound - lower_bound))
  probability (
    max lower_bound (log_base^event_upper_bound - 1 / 2) 
  ) (
    min upper_bound (log_base^event_lower_bound - 1 / 2)
  )

theorem probability_log_interval_correct : 
  probability_log_interval = 3 / 4 := by
  sorry

end probability_log_interval_correct_l724_724614


namespace sum_of_coordinates_of_intersection_l724_724018

theorem sum_of_coordinates_of_intersection :
  let A := (0, 4)
  let B := (6, 0)
  let C := (9, 3)
  let D := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let E := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let line_AE := (fun x : ℚ => (-1/3) * x + 4)
  let line_CD := (fun x : ℚ => (1/6) * x + 1/2)
  let F_x := (21 : ℚ) / 3
  let F_y := line_AE F_x
  F_x + F_y = 26 / 3 := sorry

end sum_of_coordinates_of_intersection_l724_724018


namespace jovana_shells_l724_724802

variable (initial_shells : Nat) (additional_shells : Nat)

theorem jovana_shells (h1 : initial_shells = 5) (h2 : additional_shells = 12) : initial_shells + additional_shells = 17 := 
by 
  sorry

end jovana_shells_l724_724802


namespace perimeter_triangle_ADA_l724_724359

open Real

noncomputable def eccentricity : ℝ := 1 / 2

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

noncomputable def foci_distance (a b : ℝ) : ℝ :=
  (a^2 - b^2).sqrt

noncomputable def line_passing_through_focus_perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  sorry

noncomputable def distance_de (d e : ℝ) : ℝ := 6

theorem perimeter_triangle_ADA
  (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : foci_distance a b = c)
  (h4 : eccentricity * a = c) (h5 : distance_de 6 6) :
  4 * a = 13 :=
by sorry

end perimeter_triangle_ADA_l724_724359


namespace candle_height_after_half_time_l724_724237

-- Initial candle height and burning time per centimeter
def initial_height : ℕ := 150
def burning_time (k : ℕ) : ℕ := 10 * k + 5

-- Total burning time for the candle
noncomputable def total_time : ℕ := 
  ∑ k in finset.range (initial_height + 1), burning_time k

-- Time after half of the total burning time
def half_time : ℕ := total_time / 2

-- Proof problem: Prove the candle height after half the total burning time
theorem candle_height_after_half_time :
  let m := (initial_height - 75) in
  ∑ k in finset.range (m + 1), burning_time k ≤ half_time ∧
  half_time < ∑ k in finset.range (m + 2), burning_time k ∧
  initial_height - m = 75 :=
by sorry

end candle_height_after_half_time_l724_724237


namespace option_A_implies_right_triangle_option_B_implies_right_triangle_option_D_implies_right_triangle_l724_724710

variable (A B C a b c : ℝ)

def is_right_triangle (A B C : ℝ) : Prop :=
  A + B + C = π ∧ (A = π / 2 ∨ B = π/2 ∨ C = π / 2)

theorem option_A_implies_right_triangle 
  (hA : tan A * tan B = 1) : is_right_triangle A B C :=
  sorry

theorem option_B_implies_right_triangle 
  (hB : a * cos C + c * cos A = a * sin B) : is_right_triangle A B C :=
  sorry

theorem option_D_implies_right_triangle 
  (hD : sin (2 * A) + sin (2 * B) = sin (2 * C)) : is_right_triangle A B C :=
  sorry

end option_A_implies_right_triangle_option_B_implies_right_triangle_option_D_implies_right_triangle_l724_724710


namespace determinant_matrix_eq_2D_l724_724458

-- Define vectors a, b, c
variables (a b c : ℝ^3)

-- Define the given determinant D
def D := a ⬝ (b × c)

-- Define the new vectors
def u := 2 • a + b
def v := b + c
def w := c + a

-- State the theorem
theorem determinant_matrix_eq_2D (a b c : ℝ^3) : 
  matrix.det ![u, v, w] = 2 * D :=
sorry

end determinant_matrix_eq_2D_l724_724458


namespace exists_other_wrapping_infinitely_many_wrappings_l724_724275

-- Definitions and assumptions from the problem
def is_wrapping (length width : ℝ) : Prop :=
  length * width = 2

def is_valid_wrapping (length width : ℝ) : Prop :=
  is_wrapping length width ∧ length > 0 ∧ width > 0 ∧ (length ≠ 2 ∨ width ≠ 1) ∧ (length ≠ real.sqrt 2 ∨ width ≠ real.sqrt 2)

def inf_wrappings : Prop :=
  ∀ n : ℕ, ∃ length width : ℝ, n > 0 ∧ is_wrapping length width

-- Part (a): There exists at least one other wrapping
theorem exists_other_wrapping :
  ∃ length width : ℝ, is_valid_wrapping length width := 
  sorry

-- Part (b): There are infinitely many wrappings
theorem infinitely_many_wrappings : 
  inf_wrappings :=
  sorry

end exists_other_wrapping_infinitely_many_wrappings_l724_724275


namespace necessary_but_not_sufficient_condition_l724_724702

open Real

-- Define α as an internal angle of a triangle
def is_internal_angle (α : ℝ) : Prop := (0 < α ∧ α < π)

-- Given conditions
axiom α : ℝ
axiom h1 : is_internal_angle α

-- Prove: if (α ≠ π / 6) then (sin α ≠ 1 / 2) is a necessary but not sufficient condition 
theorem necessary_but_not_sufficient_condition : 
  (α ≠ π / 6) ∧ ¬((α ≠ π / 6) → (sin α ≠ 1 / 2)) ∧ ((sin α ≠ 1 / 2) → (α ≠ π / 6)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l724_724702


namespace hyperbola_represents_range_of_m_l724_724762

open Real

theorem hyperbola_represents_range_of_m (m : ℝ) :
  (∃ x y : ℝ, (x^2 / (|m| - 1) + y^2 / (2 - m) = 1)) ↔ (m ∈ set.Ioo (-1 : ℝ) 1 ∪ set.Ioi 2) := 
sorry

end hyperbola_represents_range_of_m_l724_724762


namespace sum_abs_a_10_eq_58_l724_724542

-- Define the arithmetic sequence summation function and the sequence element
def S (n : ℕ) : ℤ := 6 * n - n ^ 2
def a (n : ℕ) : ℤ := 7 - 2 * n

-- Define the absolute value sequence and its summation for specific terms
def abs_a (n : ℕ) : ℤ := abs (a n)
def sum_abs_a (n : ℕ) : ℤ := (Finset.range n).sum abs_a

theorem sum_abs_a_10_eq_58 : sum_abs_a 10 = 58 :=
by
  -- By the problem condition, S_3 = 9 and S_10 = -40
  have h1 : S 3 = 9 := by
    sorry
  have h2 : S 10 = -40 := by
    sorry
  -- Using the relationship of sums 2S_3 - S_10
  have : 2 * S 3 - S 10 = 58 := by
    sorry
  -- Conclude sum of absolute values
  exact this

end sum_abs_a_10_eq_58_l724_724542


namespace reciprocal_of_neg3_l724_724160

theorem reciprocal_of_neg3 : ∃ x : ℝ, -3 * x = 1 ∧ x = -1/3 :=
by
  sorry

end reciprocal_of_neg3_l724_724160


namespace pages_torn_and_sheets_calculation_l724_724899

theorem pages_torn_and_sheets_calculation : 
  (∀ (n : ℕ), (sheet_no n) = (n + 1) / 2 → (2 * (n + 1) / 2) - 1 = n ∨ 2 * (n + 1) / 2 = n) →
  let first_page := 185 in
  let last_page := 518 in
  last_page = 518 → 
  ((last_page - first_page + 1) / 2) = 167 := 
by
  sorry

end pages_torn_and_sheets_calculation_l724_724899


namespace distance_between_cars_l724_724940

-- Define initial conditions
def initial_distance : ℝ := 150
def first_car_segment_1 : ℝ := 25
def first_car_turn_1 : ℝ := 15
def first_car_segment_2 : ℝ := 25
def first_car_turn_2 : ℝ := 15
def second_car_distance : ℝ := 35

-- Theorem to prove the distance between the two cars at this point
theorem distance_between_cars : 
  let total_first_car_distance := first_car_segment_1 + first_car_turn_1 + first_car_segment_2 + first_car_turn_2 in
  let remaining_distance := initial_distance - total_first_car_distance in
  remaining_distance - second_car_distance = 35 :=
by {
  sorry
}

end distance_between_cars_l724_724940


namespace u_lt_v_l724_724820

def U (x : ℝ) : ℝ := (x * (x^9 - 1) / (x - 1)) + 10 * x^9

def V (x : ℝ) : ℝ := (x * (x^11 - 1) / (x - 1)) + 10 * x^11

theorem u_lt_v (u v : ℝ) 
  (hu : U u = 8) 
  (hv : V v = 8) : 
  u < v := 
sorry

end u_lt_v_l724_724820


namespace walkway_area_296_l724_724780

theorem walkway_area_296 :
  let bed_length := 4
  let bed_width := 3
  let num_rows := 4
  let num_columns := 3
  let walkway_width := 2
  let total_bed_area := num_rows * num_columns * bed_length * bed_width
  let total_garden_width := num_columns * bed_length + (num_columns + 1) * walkway_width
  let total_garden_height := num_rows * bed_width + (num_rows + 1) * walkway_width
  let total_garden_area := total_garden_width * total_garden_height
  let total_walkway_area := total_garden_area - total_bed_area
  total_walkway_area = 296 :=
by 
  sorry

end walkway_area_296_l724_724780


namespace hyperbola_eccentricity_l724_724387

theorem hyperbola_eccentricity 
  (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (asymptote : b = sqrt 3 * a)
  (h_b_asymptote : b = sqrt 3 * a)
  : let c := sqrt (a^2 + b^2) in
    let e := c / a in
    e = 2 :=
by
  -- assumptions based on given conditions
  have h_a_pos : a > 0 := h_a_pos,
  have h_b_pos : b > 0 := h_b_pos,
  have asymptote_eq : b = sqrt 3 * a := h_b_asymptote,

  -- prove the main statement
  sorry

end hyperbola_eccentricity_l724_724387


namespace hillary_activities_l724_724285

-- Define the conditions
def swims_every : ℕ := 6
def runs_every : ℕ := 4
def cycles_every : ℕ := 16

-- Define the theorem to prove
theorem hillary_activities : Nat.lcm (Nat.lcm swims_every runs_every) cycles_every = 48 :=
by
  -- Provide a placeholder for the proof
  sorry

end hillary_activities_l724_724285


namespace smallest_k_for_inequality_l724_724209

theorem smallest_k_for_inequality :
  ∃ k : ℕ, (∀ m : ℕ, m < k → 64^m ≤ 7) ∧ 64^k > 7 :=
by
  sorry

end smallest_k_for_inequality_l724_724209


namespace gcd_102_238_l724_724190

def gcd (a b : ℕ) : ℕ := if b = 0 then a else gcd b (a % b)

theorem gcd_102_238 : gcd 102 238 = 34 :=
by
  sorry

end gcd_102_238_l724_724190


namespace lejas_theorem_l724_724773

theorem lejas_theorem (n : ℕ) (r : ℝ) (points : Fin n → ℝ × ℝ)
  (h : ∀ i j k : Fin n, ∃ c : ℝ × ℝ, ∀ p ∈ {i, j, k}, (points p).dist c = r) :
  ∃ c : ℝ × ℝ, ∀ p : Fin n, (points p).dist c = r := sorry

end lejas_theorem_l724_724773


namespace torn_out_sheets_count_l724_724882

theorem torn_out_sheets_count :
  ∃ (sheets : ℕ), (first_page = 185 ∧
                   last_page = 518 ∧
                   pages_torn_out = last_page - first_page + 1 ∧ 
                   sheets = pages_torn_out / 2 ∧
                   sheets = 167) :=
by
  sorry

end torn_out_sheets_count_l724_724882


namespace sin_beta_value_l724_724345

theorem sin_beta_value (α β : ℝ) (h₀ : 0 < α ∧ α < π / 2) (h₁ : 0 < β ∧ β < π / 2)
    (h₂ : cos α = 2 * sqrt 5 / 5) (h₃ : sin (α - β) = -3 / 5) :
    sin β = 2 * sqrt 5 / 5 := 
sorry

end sin_beta_value_l724_724345


namespace false_statement_about_circles_l724_724405

variable (P Q : Type) [MetricSpace P] [MetricSpace Q]
variable (p q : ℝ)
variable (dist_PQ : ℝ)

theorem false_statement_about_circles 
  (hA : p - q = dist_PQ → false)
  (hB : p + q = dist_PQ → false)
  (hC : p + q < dist_PQ → false)
  (hD : p - q < dist_PQ → false) : 
  false :=
by sorry

end false_statement_about_circles_l724_724405


namespace A_time_to_cover_distance_is_45_over_y_l724_724992

variable (y : ℝ)
variable (h0 : y > 0)
variable (h1 : (45 : ℝ) / (y - 2 / 3) - (45 : ℝ) / y = 3 / 4)

theorem A_time_to_cover_distance_is_45_over_y :
  45 / y = 45 / y :=
by
  sorry

end A_time_to_cover_distance_is_45_over_y_l724_724992


namespace probability_white_ball_l724_724932

variable (P : Type → ℝ)

noncomputable def P_R : ℝ := 0.3
noncomputable def P_B : ℝ := 0.5

theorem probability_white_ball :
  (P bool) := P(true) + P(false) = 1 →
  (P(true) = P_R) ∧ (P(false) = P_B) →
  P(friend White ball) = 0.2 := 
sorry

end probability_white_ball_l724_724932


namespace arithmetic_mean_of_first_n_even_numbers_l724_724522

theorem arithmetic_mean_of_first_n_even_numbers (n : ℕ) :
  let S_n := 2 * (n * (n + 1)) / 2
  in S_n / n = n + 1 :=
by
  sorry

end arithmetic_mean_of_first_n_even_numbers_l724_724522


namespace torn_out_sheets_count_l724_724884

theorem torn_out_sheets_count :
  ∃ (sheets : ℕ), (first_page = 185 ∧
                   last_page = 518 ∧
                   pages_torn_out = last_page - first_page + 1 ∧ 
                   sheets = pages_torn_out / 2 ∧
                   sheets = 167) :=
by
  sorry

end torn_out_sheets_count_l724_724884


namespace sum_of_digits_from_1_to_billion_l724_724949

-- Define the sum of digits function
def sum_of_digits (n : Nat) : Nat := n.digits.sum

-- Total number of digits sum from 1 to 1,000,000,000
def sum_digits_one_to_billion : Nat :=
  (List.range 1000000000).map sum_of_digits |> List.sum

theorem sum_of_digits_from_1_to_billion :
  sum_digits_one_to_billion = 40500000001 := by
  sorry

end sum_of_digits_from_1_to_billion_l724_724949


namespace ring_has_P_iff_field_l724_724947

-- Definition of a ring is provided by Mathlib.
-- For conciseness, let's refer to the property (P) within the scope.

variables {A : Type*} [Ring A]

-- Definition of Property (P)
-- A ring A has property (P) if any non-zero element can be written uniquely as
-- the sum of an invertible element and a non-invertible element.

def has_property_P (A : Type*) [Ring A] : Prop :=
  ∀ (a : A), a ≠ 0 → ∃! (u v : A), is_unit u ∧ ¬ is_unit v ∧ a = u + v

-- Condition 1 + 1 = 0
variable (h : (1 : A) + 1 = 0)

-- Statement to be proven: A has property (P) if and only if A is a field.
theorem ring_has_P_iff_field (h : (1 : A) + 1 = 0) :
  has_property_P A ↔ is_field A :=
sorry

end ring_has_P_iff_field_l724_724947


namespace prob_white_ball_is_0_25_l724_724000

-- Let's define the conditions and the statement for the proof
variable (P_red P_white P_yellow : ℝ)

-- The given conditions 
def prob_red_or_white : Prop := P_red + P_white = 0.65
def prob_yellow_or_white : Prop := P_yellow + P_white = 0.6

-- The statement we want to prove
theorem prob_white_ball_is_0_25 (h1 : prob_red_or_white P_red P_white)
                               (h2 : prob_yellow_or_white P_yellow P_white) :
  P_white = 0.25 :=
sorry

end prob_white_ball_is_0_25_l724_724000


namespace surface_area_of_sphere_l724_724787

-- Define the entities and conditions
variables {A B C A₁ B₁ C₁ O : Type}
variables [metric_space O] [euclidean_space O] [has_dist O (real)]
variables (AC BC : real) (angleACB : real) (C₁O : real)

-- Conditions
def AC_eq_one : Prop := AC = 1
def BC_eq_three : Prop := BC = 3
def angleACB_eq_sixty : Prop := angleACB = 60
def C₁C_eq_two_sqrt_three : Prop := C₁O = 2 * real.sqrt 3

-- Prove the surface area
theorem surface_area_of_sphere 
  (h1 : AC_eq_one AC) 
  (h2 : BC_eq_three BC) 
  (h3 : angleACB_eq_sixty angleACB) 
  (h4 : C₁C_eq_two_sqrt_three C₁O)
  : 4 * π * (4 * real.sqrt 3 / 3)^2 = 64 * π / 3 :=
begin
  sorry
end

end surface_area_of_sphere_l724_724787


namespace exists_max_pile_division_l724_724050

theorem exists_max_pile_division (k : ℝ) (hk : k < 2) : 
  ∃ (N_k : ℕ), ∀ (A : Multiset ℝ) (m : ℝ), (∀ a ∈ A, a < 2 * m) → 
    ¬(∃ B : Multiset ℝ, B.card > N_k ∧ (∀ b ∈ B, b ∈ A ∧ b < 2 * m)) :=
sorry

end exists_max_pile_division_l724_724050


namespace unique_parallel_through_external_point_l724_724959

/--
Statement: There is only one line passing through a point outside a line that is parallel to the given line.

Conditions:
1. In the same plane, two lines perpendicular to the same line are perpendicular to each other.
2. Two equal angles must be vertical angles.
3. There is only one line passing through a point outside a line that is parallel to the given line.
4. Equal alternate interior angles indicate parallel lines.
-/
theorem unique_parallel_through_external_point 
  (P L : Type) [Plane P] [Line L]
  (point_on_plane : P)
  (line_in_plane : L)
  (not_on_line : point_on_plane ∉ line_in_plane)
  (unique_parallel : (∃! parallel_line : L, parallel_line ∥ line_in_plane ∧ point_on_plane ∈ parallel_line)) :
  true :=
by
  sorry

end unique_parallel_through_external_point_l724_724959


namespace arrow_symmetry_l724_724281

theorem arrow_symmetry (n k : ℕ) (h : 2 * k ≤ n) :
  (∃ (enter enter_vertices exit_vertices : Fin n → ℕ), 
    (∀ v, enter v = 2 ↔ v ∈ enter_vertices)
    ∧
    (exit_vertices = {v | enter v = 0})
    ∧
    (card enter_vertices = k)
    ∧
    (card exit_vertices = k)) :=
begin
  sorry
end

end arrow_symmetry_l724_724281


namespace ticket_distribution_l724_724548

theorem ticket_distribution :
  let tickets := {1, 2, 3, 4, 5, 6} in
  let people := {A, B, C, D} in
  ∃ (distribution : tickets → option (option people)),
  (∀ t, distribution t ≠ none) ∧
  (∀ p, 1 ≤ (set.count (λ t, distribution t = some p) tickets) ∧ (set.count (λ t, distribution t = some p) tickets) ≤ 2) ∧
  (∀ t1 t2, distribution t1 = distribution t2 → abs (t1 - t2) ≤ 1) →
  (finset.card (set.ticket_distributions tickets people 144) = 144) :=
sorry

end ticket_distribution_l724_724548


namespace possible_last_three_digits_product_l724_724559

def lastThreeDigits (n : ℕ) : ℕ := n % 1000

theorem possible_last_three_digits_product (a b c : ℕ) (ha : a > 1000) (hb : b > 1000) (hc : c > 1000)
  (h1 : (a + b) % 10 = c % 10)
  (h2 : (a + c) % 10 = b % 10)
  (h3 : (b + c) % 10 = a % 10) :
  lastThreeDigits (a * b * c) = 0 ∨ lastThreeDigits (a * b * c) = 250 ∨ lastThreeDigits (a * b * c) = 500 ∨ lastThreeDigits (a * b * c) = 750 := 
sorry

end possible_last_three_digits_product_l724_724559


namespace problem_solution_l724_724390

open Real

def prob_max_min (y : ℝ → ℝ) (a b : ℝ) :=
  (∀ x, y x = a - b * cos (2 * x + π / 6)) ∧
  (∃ x, y x = 3) ∧
  (∃ x, y x = -1)

def find_a_b (a b : ℝ) := a = 1 ∧ b = 2

def range_g (g : ℝ → ℝ) := {y | ∃ x, π / 4 ≤ x ∧ x ≤ 5 * π / 6 ∧ y = g x}

theorem problem_solution :
  ∀ y g a b,
    prob_max_min y a b →
    find_a_b a b ∧
    ∃ r, r = range_g g ∧ r = set.Icc (-2 * sqrt 3) 4 :=
by
  intros y g a b h
  sorry

end problem_solution_l724_724390


namespace transformed_shape_is_square_l724_724423

theorem transformed_shape_is_square (L W : ℕ) 
  (h_perimeter : 2 * L + 2 * W = 50)
  (h_area_transform : L * W = (L - 4) * (W + 3)) :
  (L - 4) = 12 ∧ (W + 3) = 12 :=
begin
  sorry
end

end transformed_shape_is_square_l724_724423


namespace simplify_fraction_l724_724103

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : 
  ( (x^2 + 1) / (x - 1) - (2*x) / (x - 1) ) = x - 1 :=
by
  -- Your proof steps would go here.
  sorry

end simplify_fraction_l724_724103


namespace cube_volume_ratio_l724_724545

theorem cube_volume_ratio 
  (A B C D A₁ B₁ C₁ D₁ F O : ℝ^3)
  [IsUnitCube A B C D A₁ B₁ C₁ D₁]
  (hF : F = midpoint B C)
  (hO : O = center D C C₁ D₁) :
  volume_ratio A F O = 7 / 29 := 
sorry

end cube_volume_ratio_l724_724545


namespace monotonic_intervals_minimum_integer_a_l724_724397

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + a / x - x + 1 - a 

theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x, 0 < x ∧ x < (1 + Real.sqrt(1 - 4 * a)) / 2 → f x a ≤ f (x + ε) a) ∧
  (0 < a ∧ a < 1/4 →
    ∀ x, ((0 < x ∧ x < (1 - Real.sqrt(1 - 4 * a)) / 2) ∨ 
          ((1 - Real.sqrt(1 - 4 * a)) / 2 < x ∧ 
           x < (1 + Real.sqrt(1 - 4 * a)) / 2)) → 
    f x a ≤ f (x + ε) a) ∧
  (a ≥ 1/4 → ∀ x, 0 < x → f x a ≤ f (x + ε) a) :=
sorry

theorem minimum_integer_a :
  (∃ x, 1 < x ∧ f x 5 + x < (1 - x) / x) →
  ∀ a, (∃ x, 1 < x ∧ f x a + x < (1 - x) / x) → integer a ∧ a ≥ 5 :=
sorry

end monotonic_intervals_minimum_integer_a_l724_724397


namespace proof_1_over_a_squared_sub_1_over_b_squared_eq_1_over_ab_l724_724022

variable (a b : ℝ)

-- Condition
def condition : Prop :=
  (1 / a) - (1 / b) = 1 / (a + b)

-- Proof statement
theorem proof_1_over_a_squared_sub_1_over_b_squared_eq_1_over_ab (h : condition a b) :
  (1 / a^2) - (1 / b^2) = 1 / (a * b) :=
sorry

end proof_1_over_a_squared_sub_1_over_b_squared_eq_1_over_ab_l724_724022


namespace planes_parallel_l724_724936

theorem planes_parallel (L1 L2 : ℝ^3 → ℝ^3) :
  (∃ P : ℝ^3 → ℝ^3, (P ⊇ L1) ∧ (∀ x : ℝ^3, P x = P (L2 x))) → (0 ∨ 1 ∨ ∞) := 
sorry

end planes_parallel_l724_724936


namespace reciprocal_of_neg_three_l724_724152

theorem reciprocal_of_neg_three : ∃ x : ℚ, (-3) * x = 1 ∧ x = (-1) / 3 := sorry

end reciprocal_of_neg_three_l724_724152


namespace product_of_x_and_y_l724_724446

theorem product_of_x_and_y :
  ∀ (x y : ℝ), (∀ p : ℝ × ℝ, (p = (x, 6) ∨ p = (10, y)) → p.2 = (1 / 2) * p.1) → x * y = 60 :=
by
  intros x y h
  have hx : 6 = (1 / 2) * x := by exact h (x, 6) (Or.inl rfl)
  have hy : y = (1 / 2) * 10 := by exact h (10, y) (Or.inr rfl)
  sorry

end product_of_x_and_y_l724_724446


namespace square_area_l724_724583

theorem square_area (x y : ℝ) (h1 : distance (20, 20) (20, 9) = 11) : 
  (20 - 20) ^ 2 + (9 - 20) ^ 2 = 11 ^ 2 → 
  (9 - y) ^ 2 + (20 - 9) ^ 2 = 11 ^ 2 →
  (x - 20) ^ 2 + (y - 9) ^ 2 = 11 ^ 2 →
  let side_length := 11 in
  side_length * side_length = 121 := by {
  sorry
}

end square_area_l724_724583


namespace janice_class_girls_l724_724430

theorem janice_class_girls : ∃ (g b : ℕ), (3 * b = 4 * g) ∧ (g + b + 2 = 32) ∧ (g = 13) := by
  sorry

end janice_class_girls_l724_724430


namespace magic_8_ball_probability_l724_724799

open ProbabilityTheory
noncomputable theory

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (choose n k) * (p^k) * ((1 - p)^(n - k))

theorem magic_8_ball_probability :
  binomial_probability 7 3 (1/3) = 560 / 2187 :=
by
  sorry

end magic_8_ball_probability_l724_724799


namespace intersection_of_A_B_l724_724066

open Set

def A : Set ℝ := {x | x^2 - 4 * x - 5 < 0}
def B : Set ℝ := {x | x < 3}

theorem intersection_of_A_B : A ∩ B = {x | -1 < x ∧ x < 3} :=
by
  sorry

end intersection_of_A_B_l724_724066


namespace minimum_value_of_f_l724_724717

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x + 1) + 2 * x
def domain (x: ℝ) : Prop := x ≥ -1

theorem minimum_value_of_f : ∀ x : ℝ, domain x → (f x ≥ -2) :=
sorry

end minimum_value_of_f_l724_724717


namespace reciprocal_of_neg_three_l724_724155

theorem reciprocal_of_neg_three : ∃ (x : ℚ), (-3 * x = 1) ∧ (x = -1 / 3) :=
by
  use (-1 / 3)
  split
  . rw [mul_comm]
    norm_num 
  . norm_num

end reciprocal_of_neg_three_l724_724155


namespace simplify_expression_l724_724090

variable (x : ℝ)

theorem simplify_expression (h : x ≠ 1) : (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 :=
by sorry

end simplify_expression_l724_724090


namespace A_at_one_zero_l724_724821

variables {R : Type*} [CommRing R]

-- Definitions for the polynomials A, B, C, and D
variables (A B C D : R[X])

-- The given condition for all real x
axiom given_condition :
  ∀ x : R, A.eval (x^5) + x * B.eval (x^5) + x^2 * C.eval (x^5) = (1 + x + x^2 + x^3 + x^4) * D.eval x

-- Proof goal: Show that A(1) = 0
theorem A_at_one_zero : A.eval 1 = 0 :=
sorry

end A_at_one_zero_l724_724821


namespace minimum_value_of_function_l724_724673

theorem minimum_value_of_function :
  let f (x y : ℝ) := (x * y) / (x^2 + y^2)
  ∃ (x y : ℝ), 
    (1/4 ≤ x ∧ x ≤ 2/3) ∧ 
    (1/5 ≤ y ∧ y ≤ 1/2) ∧ 
    (∀ (a b : ℝ), 
      (1/4 ≤ a ∧ a ≤ 2/3) ∧ 
      (1/5 ≤ b ∧ b ≤ 1/2) → 
      f(x, y) ≤ f(a, b)) ∧ 
    f(x, y) = 2/5 := 
begin
  sorry
end

end minimum_value_of_function_l724_724673


namespace B_minus_C_equals_pi_over_2_area_of_triangle_l724_724769

variable (A B C a b c : Real)

-- Assumptions
axiom angle_A : A = π / 4
axiom side_relation : b * Real.cos (π / 4 - C) - c * Real.sin (π / 4 + B) = a

-- Question (1)
theorem B_minus_C_equals_pi_over_2 (h1 : b * Real.cos (π / 4 - C) - c * Real.sin (π / 4 + B) = a)
  (h2 : A = π / 4) : B - C = π / 2 :=
sorry

-- Question (2)
theorem area_of_triangle (h1 : b * Real.cos (π / 4 - C) - c * Real.sin (π / 4 + B) = a)
  (h2 : A = π / 4) (h3 : a = 2): 
  0.5 * b * c * Real.sin A = 1 :=
sorry


end B_minus_C_equals_pi_over_2_area_of_triangle_l724_724769


namespace calc_expression_eq_3_solve_quadratic_eq_l724_724593

-- Problem 1
theorem calc_expression_eq_3 :
  (-1 : ℝ) ^ 2020 + (- (1 / 2)⁻¹) - (3.14 - Real.pi) ^ 0 + abs (-3) = 3 :=
by
  sorry

-- Problem 2
theorem solve_quadratic_eq {x : ℝ} :
  (3 * x * (x - 1) = 2 - 2 * x) ↔ (x = 1 ∨ x = -2 / 3) :=
by
  sorry

end calc_expression_eq_3_solve_quadratic_eq_l724_724593


namespace total_distance_is_3_miles_l724_724488

-- Define conditions
def running_speed := 6   -- mph
def walking_speed := 2   -- mph
def running_time := 20 / 60   -- hours
def walking_time := 30 / 60   -- hours

-- Define total distance
def total_distance := (running_speed * running_time) + (walking_speed * walking_time)

theorem total_distance_is_3_miles : total_distance = 3 :=
by
  sorry

end total_distance_is_3_miles_l724_724488


namespace intersection_of_A_and_B_l724_724085

def A : Set ℝ := { x | 0 < x ∧ x < 2 }
def B : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

theorem intersection_of_A_and_B : A ∩ B = { x | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_of_A_and_B_l724_724085


namespace angle_adb_l724_724243

theorem angle_adb (A B C D : Point) (r : ℝ) (h_r : r = 10)
  (h_eq : triangle A B C)
  (h_center : center_of_circle C r)
  (h_oncircle : on_circle A r ∧ on_circle B r ∧ on_circle D r)
  (h_dia : C = midpoint D A)
  : ∠ A D B = 90 :=
  sorry

end angle_adb_l724_724243


namespace find_b_l724_724429

theorem find_b :
  (∃ b, ∀ x y, (x + y - 2 = 0 ∧ x - 2y + 4 = 0) → (y = 3 * x + b)) ↔ b = 2 :=
by
  sorry

end find_b_l724_724429


namespace tabitha_initial_money_l724_724116

def initial_money (initial left mom_cost items_cost inv_fraction : ℝ) (n_items : ℕ) : ℝ :=
  let after_mom := initial - mom_cost
  let after_invest := after_mom * (1 - inv_fraction)
  after_invest - n_items * items_cost

theorem tabitha_initial_money :
  ∀ (mom_cost items_cost left inv_fraction : ℝ) (n_items : ℕ),
  mom_cost = 8 →
  items_cost = 0.5 →
  left = 6 →
  inv_fraction = 0.5 →
  n_items = 5 →
  initial_money initial -8 0.5 0.5 = 25 := 
by
  intros
  sorry

end tabitha_initial_money_l724_724116


namespace center_to_side_AB_distance_l724_724261

variable {P : Type} [MetricSpace P] [InnerProductSpace ℝ P]

structure CircleInscribedQuadrilateral (P : Type) [MetricSpace P] [InnerProductSpace ℝ P] :=
  (A B C D O : P)
  (inscribed : isInscribedCircle A B C D O)
  (perpendicular_diagonals : ∠AOC = π/2 ∧ ∠BOD = π/2)
  (CD_length : dist C D = 8)

theorem center_to_side_AB_distance
  (Q : CircleInscribedQuadrilateral P) :
  ∃ d : ℝ, d = 4 ∧ ∀ p, p ∈ line Q.A Q.B → dist Q.O p = d :=
sorry

end center_to_side_AB_distance_l724_724261


namespace buffet_dressings_l724_724603

theorem buffet_dressings :
  ∃ (caesar italian thousandisland : ℕ),
  let ranch := 28 in
  (ranch * 2 / 7 = caesar) ∧
  (caesar * 3 = italian) ∧
  (italian * 2 / 3 = thousandisland) ∧
  caesar = 8 ∧
  italian = 24 ∧
  thousandisland = 16 :=
by
  sorry

end buffet_dressings_l724_724603


namespace rollins_ratio_l724_724187

noncomputable def proof_problem : Prop :=
  let johnson_amount := 2300
  let sutton_amount := (2300 : ℚ) / 2
  let rollins_amount := sutton_amount * 8
  let total_after_fees := 27048
  let total_school_raised := total_after_fees / 0.98
  (rollins_amount / total_school_raised ≈ 1 / 3)

theorem rollins_ratio : proof_problem :=
by
  sorry

end rollins_ratio_l724_724187


namespace sheets_torn_out_l724_724896

-- Define the conditions as given in the problem
def first_torn_page : Nat := 185
def last_torn_page : Nat := 518
def pages_per_sheet : Nat := 2

-- Calculate the total number of pages torn out
def total_pages_torn_out : Nat :=
  last_torn_page - first_torn_page + 1

-- Calculate the number of sheets torn out
def number_of_sheets_torn_out : Nat :=
  total_pages_torn_out / pages_per_sheet

-- Prove that the number of sheets torn out is 167
theorem sheets_torn_out :
  number_of_sheets_torn_out = 167 :=
by
  unfold number_of_sheets_torn_out total_pages_torn_out
  rw [Nat.sub_add_cancel (Nat.le_of_lt (Nat.lt_of_le_of_ne
    (Nat.le_add_left _ _) (Nat.ne_of_lt (Nat.lt_add_one 184))))]
  rw [Nat.div_eq_of_lt (Nat.lt.base 333)] 
  sorry -- proof steps are omitted

end sheets_torn_out_l724_724896


namespace gcd_102_238_is_34_l724_724199

noncomputable def gcd_102_238 : ℕ :=
  Nat.gcd 102 238

theorem gcd_102_238_is_34 : gcd_102_238 = 34 := by
  -- Conditions based on the Euclidean algorithm
  have h1 : 238 = 2 * 102 + 34 := by norm_num
  have h2 : 102 = 3 * 34 := by norm_num
  have h3 : Nat.gcd 102 34 = 34 := by
    rw [Nat.gcd, Nat.gcd_rec]
    exact Nat.gcd_eq_left h2

  -- Conclusion
  show gcd_102_238 = 34 from
    calc gcd_102_238 = Nat.gcd 102 238 : rfl
                  ... = Nat.gcd 34 102 : Nat.gcd_comm 102 34
                  ... = Nat.gcd 34 (102 % 34) : by rw [Nat.gcd_rec]
                  ... = Nat.gcd 34 34 : by rw [Nat.mod_eq_of_lt (by norm_num : 34 < 102)]
                  ... = 34 : Nat.gcd_self 34

end gcd_102_238_is_34_l724_724199


namespace car_rental_total_cost_l724_724857

theorem car_rental_total_cost
  (daily_rental_cost : ℕ → ℕ) 
  (cost_per_mile : ℕ → ℝ) 
  (days_rented : ℕ) 
  (miles_driven : ℕ) 
  (total_cost : ℝ) 
  (h1 : daily_rental_cost 1 = 30) 
  (h2 : cost_per_mile 1 = 0.25) 
  (h3 : days_rented = 5) 
  (h4 : miles_driven = 350) 
  (h5 : total_cost = daily_rental_cost 1 * days_rented + cost_per_mile 1 * miles_driven) :
  total_cost = 237.5 :=
begin
  sorry
end

end car_rental_total_cost_l724_724857


namespace max_ab_over_c2_l724_724005

variables {A B C a b c : ℝ}
hypothesis h1 : ∀ A B C : ℝ, a = sin A ∧ b = sin B ∧ c = sin C
hypothesis h2 : ∀ A B C : ℝ, (1 / tan A) + (1 / tan B) = 1 / tan C

theorem max_ab_over_c2 : ∀ a b c : ℝ, (∀ A B C : ℝ, a = sin A ∧ b = sin B ∧ c = sin C) →
  (∀ A B C : ℝ, (1 / tan A) + (1 / tan B) = 1 / tan C) → 
  (∃ m : ℝ, ∀ a b c : ℝ, (ab / c^2) ≤ m ∧ m = 3 / 2) :=
sorry

end max_ab_over_c2_l724_724005


namespace correct_operation_l724_724955

theorem correct_operation :
  (2 * Real.sqrt 3 - Real.sqrt 3 ≠ 1) ∧
  (Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5) ∧
  (Real.sqrt 27 / Real.sqrt 3 = 3) ∧
  (Real.sqrt ((-6)^2) ≠ -6) :=
by
  -- Condition for 2sqrt(3) - sqrt(3) ≠ 1
  have h1: 2 * Real.sqrt 3 - Real.sqrt 3 ≠ 1 := sorry,
  -- Condition for sqrt(2) + sqrt(3) ≠ sqrt(5)
  have h2: Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5 := sorry,
  -- Condition for sqrt(27) / sqrt(3) = 3
  have h3: Real.sqrt 27 / Real.sqrt 3 = 3 := sorry,
  -- Condition for sqrt((-6)^2) ≠ -6
  have h4: Real.sqrt ((-6)^2) ≠ -6 := sorry,
  exact ⟨h1, h2, h3, h4⟩

end correct_operation_l724_724955


namespace prob_three_or_more_expected_value_is_2_point_12_l724_724431

noncomputable def prob_group_A : ℝ := 0.6
noncomputable def prob_group_B : ℝ := 0.5
noncomputable def prob_group_C : ℝ := 0.5
noncomputable def prob_group_D : ℝ := 0.4

def prob_exactly_k (k : ℕ) : ℕ → ℝ 
| 0 => (1 - prob_group_A) * (1 - prob_group_B) * (1 - prob_group_C) * (1 - prob_group_D)
| 1 => prob_group_A * (1 - prob_group_B) * (1 - prob_group_C) * (1 - prob_group_D) +
       (1 - prob_group_A) * prob_group_B * (1 - prob_group_C) * (1 - prob_group_D) +
       (1 - prob_group_A) * (1 - prob_group_B) * prob_group_C * (1 - prob_group_D) +
       (1 - prob_group_A) * (1 - prob_group_B) * (1 - prob_group_C) * prob_group_D
| 2 => sorry  -- Calculation needs to involve selecting 2 out of 4 groups.
| 3 => sorry  -- Calculation needs to involve selecting 3 out of 4 groups.
| 4 => prob_group_A * prob_group_B * prob_group_C * prob_group_D
| _ => 0

theorem prob_three_or_more :
  prob_exactly_k 3 4 + prob_exactly_k 4 4 = 0.48 :=
sorry

def expected_value : ℝ :=
0 * prob_exactly_k 0 4 + 
1 * prob_exactly_k 1 4 + 
2 * prob_exactly_k 2 4 + 
3 * prob_exactly_k 3 4 + 
4 * prob_exactly_k 4 4

theorem expected_value_is_2_point_12 :
  expected_value = 2.12 :=
sorry

end prob_three_or_more_expected_value_is_2_point_12_l724_724431


namespace problem_proof_l724_724221

theorem problem_proof:
  (∃ n : ℕ, 25 = n ^ 2) ∧
  (Prime 31) ∧
  (¬ ∀ p : ℕ, Prime p → p >= 3 → p = 2) ∧
  (∃ m : ℕ, 8 = m ^ 3) ∧
  (∃ a b : ℕ, Prime a ∧ Prime b ∧ 15 = a * b) :=
by
  sorry

end problem_proof_l724_724221


namespace chord_line_equation_l724_724242

/-- Given the parametric curve defined by x = 4 cos θ and y = 4 sin θ,
    and a point M(2,1) on this curve, prove that the line passing through M
    and making M the midpoint of a chord has the equation y - 1 = -2(x - 2). -/
theorem chord_line_equation (θ : ℝ) (x y : ℝ) (M : ℝ × ℝ) (hM : M = (2, 1))
  (hx : x = 4 * real.cos θ) (hy : y = 4 * real.sin θ) :
  ∃ (m b : ℝ), line_eqn = y - 1 = -2 * (x - 2) := 
sorry

end chord_line_equation_l724_724242


namespace range_of_reciprocal_sums_l724_724337

noncomputable def f (a x : ℝ) : ℝ := a^x + x - 4
noncomputable def g (a x : ℝ) : ℝ := log a x + x - 4

theorem range_of_reciprocal_sums (a m n : ℝ) (h1 : a > 1)
  (hm : f a m = 0) (hn : g a n = 0) : 
  Set.range (λ p : ℝ, ∃ m n, (f a m = 0) ∧ (g a n = 0) ∧ p = 1/m + 1/n) = Set.Ici 1 :=
sorry

end range_of_reciprocal_sums_l724_724337


namespace torn_out_sheets_count_l724_724880

theorem torn_out_sheets_count :
  ∃ (sheets : ℕ), (first_page = 185 ∧
                   last_page = 518 ∧
                   pages_torn_out = last_page - first_page + 1 ∧ 
                   sheets = pages_torn_out / 2 ∧
                   sheets = 167) :=
by
  sorry

end torn_out_sheets_count_l724_724880


namespace min_area_convex_quadrilateral_l724_724445

theorem min_area_convex_quadrilateral (T1 T3 : ℝ) (hT1 : T1 = 4) (hT3 : T3 = 9) :
  ∃ (T2 T4 : ℝ), 
    T1 * T3 = T2 * T4 ∧
    T2 * T4 = 36 ∧
    T2 + T4 ≥ 12 ∧
    T1 + T2 + T3 + T4 = 25 :=
by
  use [T2, T4]
  sorry

end min_area_convex_quadrilateral_l724_724445


namespace inequality_holds_l724_724079

variable {x y : ℝ}

theorem inequality_holds (x : ℝ) (y : ℝ) (hy : y ≥ 5) : 
  x^2 - 2 * x * Real.sqrt (y - 5) + y^2 + y - 30 ≥ 0 := 
sorry

end inequality_holds_l724_724079


namespace slope_of_line_in_range_l724_724403

theorem slope_of_line_in_range (α : ℝ) (h₁ : 60 * Real.pi / 180 < α) (h₂ : α ≤ 135 * Real.pi / 180) :
    (tan α) ∈ set.Iic (-1) ∪ set.Ioi (Real.sqrt 3) :=
sorry

end slope_of_line_in_range_l724_724403


namespace valid_two_digit_numbers_l724_724165

-- Define the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + (n / 10 % 10) + (n / 100 % 10) + (n / 1000 % 10)

-- Prove the statement about two-digit numbers satisfying the condition
theorem valid_two_digit_numbers :
  {A : ℕ | 10 ≤ A ∧ A ≤ 99 ∧ (sum_of_digits A)^2 = sum_of_digits (A^2)} =
  {10, 11, 12, 13, 20, 21, 22, 30, 31} :=
by
  sorry

end valid_two_digit_numbers_l724_724165


namespace cos_half_pi_plus_alpha_l724_724392

theorem cos_half_pi_plus_alpha (α : ℝ) (h : ∃ (P : ℝ × ℝ), P = (-3, -4) ∧ α = real.arctan2 P.2 P.1) : 
  real.cos (real.pi / 2 + α) = 4 / 5 :=
by { sorry }

end cos_half_pi_plus_alpha_l724_724392


namespace isosceles_triangle_perimeter_l724_724778

theorem isosceles_triangle_perimeter (a b : ℕ) (h₁ : a = 6) (h₂ : b = 3) (h₃ : a > b) : a + a + b = 15 :=
by
  sorry

end isosceles_triangle_perimeter_l724_724778


namespace geometric_sequence_b_value_l724_724544

theorem geometric_sequence_b_value 
  (b : ℝ)
  (h1 : b > 0)
  (h2 : ∃ r : ℝ, 160 * r = b ∧ b * r = 1)
  : b = 4 * Real.sqrt 10 := 
sorry

end geometric_sequence_b_value_l724_724544


namespace part1_maximum_value_part1_set_of_x_part2_minimum_a_l724_724830

noncomputable def f (x : ℝ) : ℝ := 
  cos (2 * x - 4 * Real.pi / 3) + 2 * (cos x)^2

theorem part1_maximum_value :
  ∃ (x : ℝ), (∀ (y : ℝ), f y ≤ 2) ∧ f x = 2 :=
sorry

theorem part1_set_of_x :
  {x : ℝ | ∃ (k : ℤ), x = k * Real.pi - Real.pi / 6} = 
  {x : ℝ | f x = 2} :=
sorry

theorem part2_minimum_a (B C : ℝ) (b c : ℝ) :
  (f (B + C) = 3 / 2 ∧ b + c = 2) → 
  ∃ (a : ℝ), a = 1 :=
sorry

end part1_maximum_value_part1_set_of_x_part2_minimum_a_l724_724830


namespace find_a_b_l724_724686

theorem find_a_b (a b x y : ℝ) (h₀ : a + b = 10) (h₁ : a / x + b / y = 1) (h₂ : x + y = 16) (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) :
    (a = 1 ∧ b = 9) ∨ (a = 9 ∧ b = 1) :=
by
  sorry

end find_a_b_l724_724686


namespace hyperbola_standard_eq_l724_724402

variables (a b c : ℝ)
variables (x y : ℝ)
variables (eccentricity : ℝ)
variables (area_OMN : ℝ)

-- Define the hyperbola
def hyperbola (x y a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Conditions
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : eccentricity = Real.sqrt 5
axiom h4 : area_OMN = 20

-- Proof that the equation of the hyperbola is the given one
theorem hyperbola_standard_eq (a b : ℝ) : a > 0 ∧ b > 0 ∧ 
     (Real.sqrt(5) = Real.sqrt 5) ∧
     (4 * a^2 * (1/2) * Real.sqrt(5) * 4 * Real.sqrt(5) = 20)
     → hyperbola x y a b → (a = Real.sqrt 2) →  (b = Real.sqrt 8) → hyperbola x y (Real.sqrt 2) (Real.sqrt 8) :=
by
  intros ha hb heq
  intros hper h_a h_b
  have ha2 : a^2 = 2 := sorry
  have hb8 : b^2 = 8 := sorry
  exact sorry

end hyperbola_standard_eq_l724_724402


namespace regular_triangular_pyramid_surface_area_l724_724170

-- Define the surface area of a regular triangular pyramid with given conditions
theorem regular_triangular_pyramid_surface_area (a : ℝ) :
  (∃ base side : ℝ, 
  base = a ∧ side = a ∧ 
  (∀ b, 2 * b^2 = a^2) ∧ 
  ∃ S_base S_lateral S_total, 
  S_base = (sqrt 3 / 4) * a^2 ∧ 
  S_lateral = (3 / 4) * a^2 ∧ 
  S_total = S_base + S_lateral) →
  ∃ S_total, S_total = ((3 + sqrt 3) / 4) * a^2 := 
by
  intro conditions
  -- Sorry to skip the proof
  sorry

end regular_triangular_pyramid_surface_area_l724_724170


namespace red_candies_l724_724222

theorem red_candies (R Y B : ℕ) 
  (h1 : Y = 3 * R - 20)
  (h2 : B = Y / 2)
  (h3 : R + B = 90) :
  R = 40 :=
by
  sorry

end red_candies_l724_724222


namespace monotonic_range_k_l724_724872

theorem monotonic_range_k (k : ℝ) :
  (∀ x ∈ set.Icc (1 : ℝ) 2, deriv (λ x : ℝ, x^2 - k*x + 1) x ≥ 0) ∨
  (∀ x ∈ set.Icc (1 : ℝ) 2, deriv (λ x : ℝ, x^2 - k*x + 1) x ≤ 0) ↔
  (k ≤ 2) ∨ (k ≥ 4) :=
sorry

end monotonic_range_k_l724_724872


namespace triangle_side_length_l724_724020

theorem triangle_side_length 
  (X Y Z : Type) [triangle X Y Z]
  (angle_X : ∠X = 90°)
  (YZ : length Y Z = 16)
  (tan_Z_eq_3sin_Z : tan ∠Z = 3 * sin ∠Z) :
  length X Z = 16 / 3 :=
sorry

end triangle_side_length_l724_724020


namespace hyperbola_eccentricity_of_M_l724_724651

theorem hyperbola_eccentricity_of_M :
  ∀ (b : ℝ), 
  (∀ (M : ℝ → ℝ → Prop), 
  (∀ x y : ℝ, M x y ↔ x^2 - (y^2 / b^2) = 1) → 
  ∃ (A B C : ℝ × ℝ), A = (-1, 0) ∧
  B.1 = x1 ∧ B.2 = y1 ∧
  C.1 = x2 ∧ C.2 = y2 ∧
  (y1 - 0 = x1 + 1) ∧
  (y2 - 0 = x2 + 1) ∧
  |B.1 - A.1| + |C.1 - B.1| = |A.1 - C.1| ∧
  (y1 - 0) = 1*(x1 + 1) ∧ (y2 - 0) = 1*(x2 + 1) ∧
  real.sqrt (1 + 10) = 10) → 
  real.sqrt (1 - (b^2 - 10)) = b :=
begin
  sorry
end

end hyperbola_eccentricity_of_M_l724_724651


namespace reciprocal_of_neg3_l724_724163

theorem reciprocal_of_neg3 : ∃ x : ℝ, -3 * x = 1 ∧ x = -1/3 :=
by
  sorry

end reciprocal_of_neg3_l724_724163


namespace sheets_torn_out_l724_724892

-- Define the conditions as given in the problem
def first_torn_page : Nat := 185
def last_torn_page : Nat := 518
def pages_per_sheet : Nat := 2

-- Calculate the total number of pages torn out
def total_pages_torn_out : Nat :=
  last_torn_page - first_torn_page + 1

-- Calculate the number of sheets torn out
def number_of_sheets_torn_out : Nat :=
  total_pages_torn_out / pages_per_sheet

-- Prove that the number of sheets torn out is 167
theorem sheets_torn_out :
  number_of_sheets_torn_out = 167 :=
by
  unfold number_of_sheets_torn_out total_pages_torn_out
  rw [Nat.sub_add_cancel (Nat.le_of_lt (Nat.lt_of_le_of_ne
    (Nat.le_add_left _ _) (Nat.ne_of_lt (Nat.lt_add_one 184))))]
  rw [Nat.div_eq_of_lt (Nat.lt.base 333)] 
  sorry -- proof steps are omitted

end sheets_torn_out_l724_724892


namespace simplify_expression_l724_724097

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
    ((x^2 + 1) / (x - 1)) - (2 * x / (x - 1)) = x - 1 := 
by
    sorry

end simplify_expression_l724_724097


namespace incorrect_statement_l724_724788

def population : ℕ := 13000
def sample_size : ℕ := 500
def academic_performance (n : ℕ) : Type := sorry

def statement_A (ap : Type) : Prop := 
  ap = academic_performance population

def statement_B (ap : Type) : Prop := 
  ∀ (u : ℕ), u ≤ population → ap = academic_performance 1

def statement_C (ap : Type) : Prop := 
  ap = academic_performance sample_size

def statement_D : Prop := 
  sample_size = 500

theorem incorrect_statement : ¬ (statement_B (academic_performance 1)) :=
sorry

end incorrect_statement_l724_724788


namespace tetrahedron_sphere_surface_area_l724_724632

noncomputable def surface_area_of_circumscribed_sphere (R : ℝ) : ℝ := 
  4 * real.pi * R^2

theorem tetrahedron_sphere_surface_area (S A B C O : Type)
  [MetricSpace S] [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]
  (r : ℝ) (S_O_sub : S ⊆ Sphere O r) 
  (A_O_sub : A ⊆ Sphere O r) 
  (B_O_sub : B ⊆ Sphere O r) 
  (C_O_sub : C ⊆ Sphere O r) 
  (SA_perp_ABC : S ⊥ plane A B C)
  (AB_perp_AC : A B ⊥ A C)
  (SA_length : dist S A = 1)
  (AB_length : dist A B = 1)
  (AC_length : dist A C = 1) :
  surface_area_of_circumscribed_sphere (sqrt 3 / 2) = 3 * real.pi := 
sorry

end tetrahedron_sphere_surface_area_l724_724632


namespace shift_sin2x_to_cos2x_l724_724180

noncomputable def transform_graph_shift (x : ℝ) : Prop :=
  let sin_2x := Math.sin (2 * x)
  let cos_2x := Math.cos (2 * x)
  (cos_2x = Math.cos (2 * (x + (Real.pi / 4))))

theorem shift_sin2x_to_cos2x :
  ∀ (x : ℝ), (Math.sin (2 * x) = Math.cos (2 * (x - Real.pi / 4))) :=
by
  sorry

end shift_sin2x_to_cos2x_l724_724180


namespace second_job_hourly_wage_l724_724838

-- Definitions based on conditions
def total_wages : ℕ := 160
def first_job_wages : ℕ := 52
def second_job_hours : ℕ := 12

-- Proof statement
theorem second_job_hourly_wage : 
  (total_wages - first_job_wages) / second_job_hours = 9 :=
by
  sorry

end second_job_hourly_wage_l724_724838


namespace locus_of_vertices_locus_of_vertices_l724_724672

-- Definitions based on the given problem
def O : Point := sorry
def H : Point := sorry
def C0 : Point := midpoint (A, B)

-- Given conditions
axiom condition_CH_eq_2OC0 : dist C H = 2 * dist O C0
axiom condition_C0_in_circumcircle : C0 ∈ circumcircle O A B
axiom condition_distance_relationship : dist C H < 2 * dist O C

-- We now state our theorem
theorem locus_of_vertices_locus_of_vertices (O H : Point) (C : Point) : 
  ∃ M : Point, ∃ M' : Point,
  M = midpoint(O, H) ∧
  M' = point_symmetric H O ∧
  ∀ C, C ∉ circle_with_diameter M M' ∧ C ∉ circumference_with_diameter M H ∧ C = H :=
sorry

end locus_of_vertices_locus_of_vertices_l724_724672


namespace f_2010_l724_724399

noncomputable def f : ℝ → ℝ := λ x, x * exp x

def f_i (i : ℕ) : ℝ → ℝ :=
  Nat.recOn i f (λ n fn x, deriv fn x)

theorem f_2010 (x : ℝ) : f_i 2010 x = 2010 * exp x + x * exp x := 
by sorry

end f_2010_l724_724399


namespace passed_in_both_is_50_percent_l724_724578

variables (total_students failed_in_hindi failed_in_english failed_in_both : ℕ)

-- Define the conditions as given in the problem
def percentage_failed_in_hindi : Prop := failed_in_hindi = total_students * 25 / 100
def percentage_failed_in_english : Prop := failed_in_english = total_students * 50 / 100
def percentage_failed_in_both : Prop := failed_in_both = total_students * 25 / 100

-- Define the statement to be proved
theorem passed_in_both_is_50_percent 
  (h1 : percentage_failed_in_hindi)
  (h2 : percentage_failed_in_english)
  (h3 : percentage_failed_in_both)
  (total_students = 100) : 
  (total_students - (failed_in_hindi + failed_in_english - failed_in_both)) * 100 / total_students = 50 :=
sorry

end passed_in_both_is_50_percent_l724_724578


namespace angles_in_range_l724_724977

-- Define the set S of angles having the same terminal side as 370°23'
def S : Set ℝ := {x | ∃ k ∈ Int, x = k * 360 + (370 + 23 / 60)}

-- Main theorem statement
theorem angles_in_range : 
  ∀ x ∈ S, -720 ≤ x ∧ x ≤ 360 → x = -709.6166666666667 ∨ x = -349.6166666666667 ∨ x = 10.383333333333333 :=
by
  sorry

end angles_in_range_l724_724977


namespace quadrilateral_angle_ABC_l724_724680

theorem quadrilateral_angle_ABC (A B C D : Type*) [metric_space A] [metric_space B]
  [metric_space C] [metric_space D] [metric_space (triangle A B C D)] :
  (BAC = 60) ∧ (CAD = 60) ∧ (AB + AD = AC) ∧ (ACD = 23) → ∠ABC = 83 := 
by
  sorry

end quadrilateral_angle_ABC_l724_724680


namespace range_of_w_l724_724833

noncomputable def w (x : ℝ) : ℝ := x^4 - 6 * x^2 + 9

theorem range_of_w : set.range w = set.Ici 0 :=
  sorry

end range_of_w_l724_724833


namespace problem_equiv_solution_l724_724654

noncomputable def determine_x (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) : ℝ :=
  Classical.choose (Exists.rec_on
    (det_eq_zero_iff_exists_smul_add_self _ _ _ (show Matrix n n ℝ, sorry )) $
      λ x H, ⟨x, sorry⟩)

theorem problem_equiv_solution {a b c : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  ∃ x, 
  det ![![x + a, x + b, x + c], ![x + b, x + c, x + a], ![x + c, x + a, x + b]] = 0 :=
begin
  use determine_x a b c ha hb hc hab hac hbc,
  sorry,
end

end problem_equiv_solution_l724_724654


namespace MichaelRobots_l724_724837

-- Define the known conditions as constants
constant TomRobots : ℕ := 16
constant twice_as_many : ∀ (M : ℕ), TomRobots = 2 * M

-- Define the final assertion about Michael's robots
theorem MichaelRobots : ∃ (M : ℕ), TomRobots = 2 * M ∧ M = 8 :=
by
  use 8
  split
  · exact twice_as_many 8
  · rfl

end MichaelRobots_l724_724837


namespace Ptolemy_inequality_l724_724590

   theorem Ptolemy_inequality {A B C D : Point} : 
     dist A B * dist C D + dist B C * dist A D ≥ dist A C * dist B D := 
   sorry
   
end Ptolemy_inequality_l724_724590


namespace amphibians_frogs_count_l724_724434

/-- In a magical swamp, five amphibians (Brian, Chris, LeRoy, Mike, David) make the following statements:
    - Brian: "Mike and I are different species."
    - Chris: "LeRoy is a frog."
    - LeRoy: "Chris is a frog."
    - Mike: "Of the five of us, at least three are toads."
    - David: "Brian is a toad and Chris is a frog."
    Prove that the number of frogs among these five amphibians is 2.
-/
theorem amphibians_frogs_count : 
  ∀ (Brian_toad Chris_toad LeRoy_toad Mike_toad David_toad : Prop),
  (Brian_toad ↔ ¬Mike_toad) →
  (¬Chris_toad ↔ LeRoy_toad) →
  (¬LeRoy_toad ↔ Chris_toad) →
  (¬Mike_toad ↔ Brian_toad ∧ Chris_toad) →
  (David_toad ↔ Brian_toad ∧ ¬Chris_toad) →
  (¬David_toad ↔ Brian_toad ∧ Chris_toad) →
  (Brian_toad ∧ ¬Chris_toad ∧ LeRoy_toad ∧ ¬Mike_toad ∧ ¬David_toad →
  ∑ e in ([Brian_toad, Chris_toad, LeRoy_toad, Mike_toad, David_toad].erase true), 1 = 2) :=
begin
  sorry
end

end amphibians_frogs_count_l724_724434


namespace find_base_of_isosceles_triangle_l724_724136

-- Definitions and conditions based on the given problem
def isosceles_triangle (a b c : ℕ) := (a = b ∨ b = c ∨ c = a)

def valid_triangle (a b c : ℕ) := (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

def perimeter (a b c : ℕ) := a + b + c

-- The proof statement to show the base can be either 3 or 5
theorem find_base_of_isosceles_triangle (a b c : ℕ) (h_iso : isosceles_triangle a b c)
    (h_perimeter : perimeter a b c = 11) (h_side_3 : a = 3 ∨ b = 3 ∨ c = 3) :
    (a = 3 ∨ b = 3 ∨ c = 3) ∧ (a = 5 ∨ b = 5 ∨ c = 5) ∨
    (a = 3 ∨ b = 3 ∨ c = 3) ∧ (a = 4 ∨ b = 4 ∨ c = 4) := 

begin 
  sorry 
end

end find_base_of_isosceles_triangle_l724_724136


namespace percentage_workday_in_meetings_l724_724290

theorem percentage_workday_in_meetings :
  let workday_minutes := 10 * 60
  let first_meeting := 30
  let second_meeting := 2 * first_meeting
  let third_meeting := first_meeting + second_meeting
  let total_meeting_minutes := first_meeting + second_meeting + third_meeting
  (total_meeting_minutes * 100) / workday_minutes = 30 :=
by
  sorry

end percentage_workday_in_meetings_l724_724290


namespace marbles_difference_l724_724549

theorem marbles_difference (t a : ℕ) (h_t : t = 72) (h_a : a = 42) : 
  let b := t - a in
  let d := a - b in
  d = 12 :=
by 
  sorry

end marbles_difference_l724_724549


namespace total_money_spent_correct_l724_724480

def money_spent_at_mall : Int := 250

def cost_per_movie : Int := 24
def number_of_movies : Int := 3
def money_spent_at_movies := cost_per_movie * number_of_movies

def cost_per_bag_of_beans : Float := 1.25
def number_of_bags : Int := 20
def money_spent_at_market := cost_per_bag_of_beans * number_of_bags

def total_money_spent := money_spent_at_mall + money_spent_at_movies + money_spent_at_market

theorem total_money_spent_correct : total_money_spent = 347 := by
  sorry

end total_money_spent_correct_l724_724480


namespace gcd_102_238_l724_724191

def gcd (a b : ℕ) : ℕ := if b = 0 then a else gcd b (a % b)

theorem gcd_102_238 : gcd 102 238 = 34 :=
by
  sorry

end gcd_102_238_l724_724191


namespace point_in_second_quadrant_l724_724760

theorem point_in_second_quadrant (a b : ℝ) (ha : a < 0) (hb : b > 0) : M ∈ second_quadrant :=
sorry

end point_in_second_quadrant_l724_724760


namespace min_modulus_of_z_l724_724046

theorem min_modulus_of_z (z : ℂ) (h : |z + 3 * I| + |z - (5 + 2 * I)| = 7) : 
  ∃ z : ℂ, |z| = 5 / real.sqrt 29 ∧ (|z + 3 * I| + |z - (5 + 2 * I)| = 7) :=
sorry

end min_modulus_of_z_l724_724046


namespace bob_cleaning_time_l724_724793

theorem bob_cleaning_time (alice_time : ℕ) (fraction : ℚ) (bob_time : ℕ) 
  (h1 : alice_time = 30) 
  (h2 : fraction = 1 / 3) 
  (h3 : bob_time = fraction.to_nat * alice_time) : 
  bob_time = 10 := 
sorry

end bob_cleaning_time_l724_724793


namespace product_increase_false_l724_724573

theorem product_increase_false (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) : 
  ¬ (a * b = a * (10 * b) / 10 ∧ a * (10 * b) / 10 = 10 * (a * b)) :=
by 
  sorry

end product_increase_false_l724_724573


namespace decimal_to_fraction_l724_724965

theorem decimal_to_fraction :
  (368 / 100 : ℚ) = (92 / 25 : ℚ) := by
  sorry

end decimal_to_fraction_l724_724965


namespace exists_max_piles_l724_724063

theorem exists_max_piles (k : ℝ) (hk : k < 2) : 
  ∃ Nk : ℕ, ∀ A : Multiset ℝ, 
    (∀ a ∈ A, ∃ m ∈ A, a ≤ k * m) → 
    A.card ≤ Nk :=
sorry

end exists_max_piles_l724_724063


namespace power_function_decreasing_l724_724427

theorem power_function_decreasing (m : ℤ) (h : ∀ x > 0, (x ^ (m ^ 2 - m - 2)) < x) : m = 0 ∨ m = 1 := 
sorry

end power_function_decreasing_l724_724427


namespace minimum_positive_period_l724_724535

noncomputable def y (x : ℝ) : ℝ := sin x * cos x

theorem minimum_positive_period (T : ℝ) :
  (∀ x : ℝ, y (x + T) = y x) → T = π :=
sorry

end minimum_positive_period_l724_724535


namespace correct_statement_l724_724220

-- Conditions as definitions
def statementA : Prop := 
  ∀ (n : ℕ), (frequency n event) → (stabilize_at (probability event))

def statementB : Prop :=
  ∃ (tickets : ℕ), (prob_winning tickets) = 1

def statementC : Prop :=
  ∀ (tosses : ℕ), (tails : ℕ), (tosses = 100) ∧ (tails = 49) → (probability_tail tosses tails) = 49 / 100

def statementD : Prop :=
  (prob_precip 70) → (expert_opinion_precipitation 70 30)

-- Proof problem requiring the correct statement
theorem correct_statement : statementA ∧ ¬statementB ∧ ¬statementC ∧ ¬statementD :=
by sorry

end correct_statement_l724_724220


namespace juan_picked_oranges_l724_724624

variable (total_oranges : ℕ) (del_oranges_per_day : ℕ) (days : ℕ)
variable (total_oranges_picked : total_oranges = 107)
variable (del_picked_daily : del_oranges_per_day = 23)
variable (del_picked_days : days = 2)

theorem juan_picked_oranges (total_oranges_picked : total_oranges = 107)
  (del_picked_daily : del_oranges_per_day = 23)
  (del_picked_days : days = 2) :
  let del_total := del_oranges_per_day * days 
  let juan_oranges := total_oranges - del_total
  juan_oranges = 61 :=
by
  let del_total := del_oranges_per_day * days
  let juan_oranges := total_oranges - del_total
  have h1 : del_total = 46 := by rw [del_picked_daily, del_picked_days]; norm_num
  have h2 : juan_oranges = 61 := by rw [total_oranges_picked, h1]; norm_num
  exact h2

end juan_picked_oranges_l724_724624


namespace complement_of_M_in_U_l724_724728

noncomputable def U : Set ℝ := { x | x^2 - 2 * x - 3 ≤ 0 }
noncomputable def M : Set ℝ := { y | ∃ x, x^2 + y^2 = 1 }

theorem complement_of_M_in_U :
  (U \ M) = { x | 1 < x ∧ x ≤ 3 } :=
by
  sorry

end complement_of_M_in_U_l724_724728


namespace number_of_papers_l724_724631

-- Define the conditions
def folded_pieces (folds : ℕ) : ℕ := 2 ^ folds
def notes_per_day : ℕ := 10
def days_per_notepad : ℕ := 4
def notes_per_notepad : ℕ := notes_per_day * days_per_notepad
def notes_per_paper (folds : ℕ) : ℕ := folded_pieces folds

-- Lean statement for the proof problem
theorem number_of_papers (folds : ℕ) (h_folds : folds = 3) :
  (notes_per_notepad / notes_per_paper folds) = 5 :=
by
  rw [h_folds]
  simp [notes_per_notepad, notes_per_paper, folded_pieces]
  sorry

end number_of_papers_l724_724631


namespace construct_regular_tetrahedron_l724_724637

-- Defining the structure and required properties
structure EquilateralTriangle (s : ℝ) :=
(a : ℝ) (b : ℝ) (c : ℝ)
(h1 : a = s)
(h2 : b = s)
(h3 : c = s)

structure RegularTetrahedron (s : ℝ) :=
(face1 : EquilateralTriangle s)
(face2 : EquilateralTriangle s)
(face3 : EquilateralTriangle s)
(face4 : EquilateralTriangle s)
(h : ∃ l1 l2 l3 l4 l5 l6 : ℝ, 
  ∀ (li : ℝ) (h : ∃ i : ℕ, li = i), li = s)

-- The main statement to be proven based on the given problem and conditions
theorem construct_regular_tetrahedron : 
  ∃ (s : ℝ) (matches : Fin 6 → ℝ) 
  (tetra : RegularTetrahedron s), 
  (∀ (i : Fin 6), matches i = s) :=
sorry

end construct_regular_tetrahedron_l724_724637


namespace monikaTotalSpending_l724_724482

-- Define the conditions as constants
def mallSpent : ℕ := 250
def movieCost : ℕ := 24
def movieCount : ℕ := 3
def beanCost : ℚ := 1.25
def beanCount : ℕ := 20

-- Define the theorem to prove the total spending
theorem monikaTotalSpending : mallSpent + (movieCost * movieCount) + (beanCost * beanCount) = 347 :=
by
  sorry

end monikaTotalSpending_l724_724482


namespace find_matrix_N_l724_724669

def N : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 6, -4]

theorem find_matrix_N :
  (N ⬝ ![![2], ![1]] = ![![7], ![8]]) ∧ 
  (N ⬝ ![![-2], ![6]] = ![![-1], ![23]]) :=
by
  sorry

end find_matrix_N_l724_724669


namespace age_difference_l724_724582

variable (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 16) : C + 16 = A := 
by
  sorry

end age_difference_l724_724582


namespace polynomial_remainder_l724_724818

noncomputable def g (x : ℤ) : ℤ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

theorem polynomial_remainder (x : ℤ) : 
  let remainder := (g x^12) % (g x)
  remainder = 7 :=
begin
  sorry
end

end polynomial_remainder_l724_724818


namespace torn_out_sheets_count_l724_724879

theorem torn_out_sheets_count :
  ∃ (sheets : ℕ), (first_page = 185 ∧
                   last_page = 518 ∧
                   pages_torn_out = last_page - first_page + 1 ∧ 
                   sheets = pages_torn_out / 2 ∧
                   sheets = 167) :=
by
  sorry

end torn_out_sheets_count_l724_724879


namespace Petya_tore_out_sheets_l724_724886

theorem Petya_tore_out_sheets (n m : ℕ) (h1 : n = 185) (h2 : m = 518)
  (h3 : m.digits = n.digits) : (m - n + 1) / 2 = 167 :=
by
  sorry

end Petya_tore_out_sheets_l724_724886


namespace find_length_of_street_l724_724521

-- Definitions based on conditions
def area_street (L : ℝ) : ℝ := L^2
def area_forest (L : ℝ) : ℝ := 3 * (area_street L)
def num_trees (L : ℝ) : ℝ := 4 * (area_forest L)

-- Statement to prove
theorem find_length_of_street (L : ℝ) (h : num_trees L = 120000) : L = 100 := by
  sorry

end find_length_of_street_l724_724521


namespace max_chips_with_constraints_l724_724850

theorem max_chips_with_constraints (n : ℕ) (h1 : n > 0) 
  (h2 : ∀ i j : ℕ, (i < n) → (j = i + 10 ∨ j = i + 15) → ((i % 25) = 0 ∨ (j % 25) = 0)) :
  n ≤ 25 := 
sorry

end max_chips_with_constraints_l724_724850


namespace Jonas_initial_socks_l724_724454

noncomputable def pairsOfSocks(Jonas_pairsOfShoes : ℕ) (Jonas_pairsOfPants : ℕ) 
                              (Jonas_tShirts : ℕ) (Jonas_pairsOfNewSocks : ℕ) : ℕ :=
    let individualShoes := Jonas_pairsOfShoes * 2
    let individualPants := Jonas_pairsOfPants * 2
    let individualTShirts := Jonas_tShirts
    let totalWithoutSocks := individualShoes + individualPants + individualTShirts
    let totalToDouble := (totalWithoutSocks + Jonas_pairsOfNewSocks * 2) / 2
    (totalToDouble * 2 - totalWithoutSocks) / 2

theorem Jonas_initial_socks (Jonas_pairsOfShoes : ℕ) (Jonas_pairsOfPants : ℕ) 
                             (Jonas_tShirts : ℕ) (Jonas_pairsOfNewSocks : ℕ) 
                             (h1 : Jonas_pairsOfShoes = 5)
                             (h2 : Jonas_pairsOfPants = 10)
                             (h3 : Jonas_tShirts = 10)
                             (h4 : Jonas_pairsOfNewSocks = 35) :
    pairsOfSocks Jonas_pairsOfShoes Jonas_pairsOfPants Jonas_tShirts Jonas_pairsOfNewSocks = 15 :=
by
    subst h1
    subst h2
    subst h3
    subst h4
    sorry

end Jonas_initial_socks_l724_724454


namespace calculate_expression_l724_724289

theorem calculate_expression :
  500 * 1986 * 0.3972 * 100 = 20 * 1986^2 :=
by sorry

end calculate_expression_l724_724289


namespace tangent_line_at_point_l724_724528

-- Conditions definition
def f (x : ℝ) (f'1 : ℝ) (f0 : ℝ) : ℝ :=
  (f'1 / Real.exp 1) * Real.exp x - f0 * x + (1 / 2) * x^2

-- The final proof statement 
theorem tangent_line_at_point (f'1 f0 : ℝ) (h : f'1 = Real.exp 1) :
  let f_1 := f 1 f'1 f0
  ∃ (m b : ℝ), m = Real.exp 1 ∧ b = - (1 / 2) ∧ ∀ y x : ℝ, y = m * x + b :=
  
sorry

end tangent_line_at_point_l724_724528


namespace count_primes_with_ones_digit_3_lt_150_eq_9_l724_724416

def is_prime (n : ℕ) : Prop := nat.prime n

def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3

def primes_with_ones_digit_3_lt_150 : list ℕ :=
  [3, 13, 23, 43, 53, 73, 83, 103, 113]

theorem count_primes_with_ones_digit_3_lt_150_eq_9 :
  list.length primes_with_ones_digit_3_lt_150 = 9 :=
by
  -- A proof would go here.
  sorry

end count_primes_with_ones_digit_3_lt_150_eq_9_l724_724416


namespace equivalence_of_min_perimeter_and_cyclic_quadrilateral_l724_724034

-- Definitions for points P, Q, R, S on sides of quadrilateral ABCD
-- Function definitions for conditions and equivalence of stated problems

variable {A B C D P Q R S : Type*} 

def is_on_side (P : Type*) (A B : Type*) : Prop := sorry
def is_interior_point (P : Type*) (A B : Type*) : Prop := sorry
def is_convex_quadrilateral (A B C D : Type*) : Prop := sorry
def is_cyclic_quadrilateral (A B C D : Type*) : Prop := sorry
def has_circumcenter_interior (A B C D : Type*) : Prop := sorry
def has_minimal_perimeter (P Q R S : Type*) : Prop := sorry

theorem equivalence_of_min_perimeter_and_cyclic_quadrilateral 
  (h1 : is_convex_quadrilateral A B C D) 
  (hP : is_on_side P A B ∧ is_interior_point P A B) 
  (hQ : is_on_side Q B C ∧ is_interior_point Q B C) 
  (hR : is_on_side R C D ∧ is_interior_point R C D) 
  (hS : is_on_side S D A ∧ is_interior_point S D A) :
  (∃ P' Q' R' S', has_minimal_perimeter P' Q' R' S') ↔ (is_cyclic_quadrilateral A B C D ∧ has_circumcenter_interior A B C D) :=
sorry

end equivalence_of_min_perimeter_and_cyclic_quadrilateral_l724_724034


namespace distribute_stickers_l724_724278

theorem distribute_stickers :
  let n := 9 -- number of stickers
  let k := 3 -- number of sheets
  (finset.filter (λ p: Π i : fin k, ℕ, finset.univ.sum p = n) finset.univ).card = 55 :=
by
  let n := 9
  let k := 3
  sorry  -- Proof omitted

end distribute_stickers_l724_724278


namespace antenna_spire_height_l724_724520

theorem antenna_spire_height : 
  ∀ (top_floor_height total_height : ℕ), 
  top_floor_height = 1250 → 
  total_height = 1454 → 
  total_height - top_floor_height = 204 := 
by 
  intros top_floor_height total_height h1 h2
  rw [h1, h2]
  norm_num
  sorry

end antenna_spire_height_l724_724520


namespace arithmetic_sequence_property_l724_724346

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 0 + a (n - 1))) / 2

def condition (S : ℕ → ℝ) : Prop :=
  (S 8 - S 5) * (S 8 - S 4) < 0

-- Theorem to prove
theorem arithmetic_sequence_property {a : ℕ → ℝ} {S : ℕ → ℝ}
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms S a)
  (h_cond : condition S) :
  |a 5| > |a 6| := 
sorry

end arithmetic_sequence_property_l724_724346


namespace smaller_box_glasses_l724_724283

theorem smaller_box_glasses :
  ∃ (x : ℕ), x + 16 * 17 = 480 ∧ (480 = 15 * (1 + 17)) ∧ x = 208 :=
by
  existsi 208
  split
  . trivial
  . split
  . trivial
  . trivial
  sorry

end smaller_box_glasses_l724_724283


namespace reflection_across_x_axis_reflection_across_origin_l724_724527

def P := (1, 2)
def reflect_x_axis (point : ℕ × ℕ) : ℕ × ℕ := (point.1, -point.2)
def reflect_origin (point : ℕ × ℕ) : ℕ × ℕ := (-point.1, -point.2)

theorem reflection_across_x_axis :
  reflect_x_axis P = (1, -2) :=
by
  sorry

theorem reflection_across_origin :
  reflect_origin P = (-1, -2) :=
by
  sorry

end reflection_across_x_axis_reflection_across_origin_l724_724527


namespace num_solutions_l724_724038

theorem num_solutions (n : ℕ) (a : ℝ) : n > 0 → a > 0 → (∑ i in Finset.range n, x_i^2 + (a - x_i)^2 = n * a^2) → 
  (∀ i, 0 ≤ x_i ∧ x_i ≤ a) → 
  (∀ i, x_i = 0 ∨ x_i = a) → 
  ∑ (i : ℕ) in Finset.range n, if 0 ≤ x_i ∧ x_i ≤ a then 2^n else 0 :=
sorry

end num_solutions_l724_724038


namespace Diaz_age_20_years_from_now_l724_724601

open Nat

theorem Diaz_age_20_years_from_now:
  (∃ (diaz_age : ℕ) (sierra_age : ℕ),
    sierra_age = 30 ∧
    40 + 10 * diaz_age = 20 + 10 * sierra_age ∧
    diaz_age + 20 = 56) :=
begin
  sorry
end

end Diaz_age_20_years_from_now_l724_724601


namespace fare_for_100_miles_l724_724629

noncomputable def fare_for_distance (d : ℕ) : ℕ :=
  let S := 20
  let fare_80 := 200
  let variable_fare_per_mile := (fare_80 - S) / 80
  S + variable_fare_per_mile * d

theorem fare_for_100_miles : fare_for_distance 100 = 245 := by
  let S := 20
  let fare_80 := 200
  let variable_fare_per_mile := (fare_80 - S) / 80
  calc
    fare_for_distance 100 = S + variable_fare_per_mile * 100 := rfl
                        ... = 20 + 225 := by sorry
                        ... = 245 := rfl

end fare_for_100_miles_l724_724629


namespace triangle_ADE_perimeter_l724_724372

noncomputable def ellipse_perimeter (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) (e : ℝ) (h₃ : e = (1 / 2)) 
(F₁ F₂ : ℝ × ℝ) (h₄ : F₁ ≠ F₂) (D E : ℝ × ℝ) (h₅ : |D - E| = 6) : ℝ :=
  let c := (sqrt (a ^ 2 - b ^ 2)) in
  let A := (0, b) in
  let AD := sqrt ((fst D) ^ 2 + (snd D - b) ^ 2) in
  let AE := sqrt ((fst E) ^ 2 + (snd E - b) ^ 2) in
  AD + AE + |D - E|

theorem triangle_ADE_perimeter (a b : ℝ) (h₁ : a > b > 0) (e : ℝ) (h₂ : e = (1 / 2))
(F₁ F₂ : ℝ × ℝ) (h₃ : F₁ ≠ F₂)
(D E : ℝ × ℝ) (h₄ : |D - E| = 6) : 
  ellipse_perimeter a b (and.left h₁) (and.right h₁) e h₂ F₁ F₂ h₃ D E h₄ = 19 :=
sorry

end triangle_ADE_perimeter_l724_724372


namespace area_of_triangle_l724_724700

-- Define the ellipse and its properties
def ellipse (x y : ℝ) : Prop := (x^2) / 16 + (y^2) / 4 = 1

-- Define the foci F1 and F2 based on the ellipse parameters
def F1 : ℝ × ℝ := (-2 * Real.sqrt 3, 0)
def F2 : ℝ × ℝ := (2 * Real.sqrt 3, 0)

-- Define the point P on the ellipse
variables (P : ℝ × ℝ)

-- Define the perpendicular condition
def perp (P F1 F2 : ℝ × ℝ) : Prop :=
let PF1 := Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) in
let PF2 := Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) in
PF1 * PF2 = 16

-- The main proof statement
theorem area_of_triangle
  (h1 : ellipse P.1 P.2)
  (h2 : perp P F1 F2) :
  Real.sqrt ((P.1 - F1.1) * (P.2 - F2.2)) = 4 :=
sorry

end area_of_triangle_l724_724700


namespace simplified_fraction_of_num_l724_724961

def num : ℚ := 368 / 100

theorem simplified_fraction_of_num : num = 92 / 25 := by
  sorry

end simplified_fraction_of_num_l724_724961


namespace square_area_in_second_configuration_l724_724550

theorem square_area_in_second_configuration 
  (area_first_config : ℝ)
  (isosceles_right_triangle : Prop)
  (inscribe_square_first : Prop)
  (inscribe_square_second : Prop)
  (hypotenuse : ℝ)
  (legs : ℝ)
  (side_first_square : ℝ)
  (side_second_square : ℝ)
  (area_second_square : ℝ) :
  (isosceles_right_triangle ∧ inscribe_square_first ∧ area_first_config = 484 ∧ 
   inscribe_square_second ∧ hypotenuse = side_first_square * real.sqrt 2 ∧ 
   legs = hypotenuse / real.sqrt 2 ∧ 
   side_first_square = real.sqrt area_first_config ∧ 
   3 * side_second_square = legs ∧ 
   area_second_square = side_second_square^2) → 
   area_second_square = 968 / 9 :=
begin
  sorry
end

end square_area_in_second_configuration_l724_724550


namespace painting_house_cost_l724_724803

theorem painting_house_cost 
  (judson_contrib : ℕ := 500)
  (kenny_contrib : ℕ := judson_contrib + (judson_contrib * 20) / 100)
  (camilo_contrib : ℕ := kenny_contrib + 200) :
  judson_contrib + kenny_contrib + camilo_contrib = 1900 :=
by
  sorry

end painting_house_cost_l724_724803


namespace product_approximation_l724_724208

theorem product_approximation : 
  let x := 53.8 - 0.08 in
  let y := 2.4 * x in
  let z := y * 1.2 in
  |z - 155| < 1 :=
by
  let x := 53.8 - 0.08
  let y := 2.4 * x
  let z := y * 1.2
  sorry

end product_approximation_l724_724208


namespace simplify_expression_l724_724100

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
  ((x^2 + 1) / (x - 1) - 2 * x / (x - 1)) = x - 1 :=
by
  -- Proof goes here.
  sorry

end simplify_expression_l724_724100


namespace sequence_geometric_not_arithmetic_l724_724469

noncomputable def x : ℝ := (Real.sqrt 5 + 1) / 2
def floor_x : ℤ := Int.floor x
def frac_x : ℝ := x - floor_x

theorem sequence_geometric_not_arithmetic :
  let a := frac_x,
      b := floor_x,
      c := x in
  (a, b, c) = (frac_x, floor_x, x) ∧
  a * c = b ^ 2 ∧
  a + b + c ≠ (a + b + c) / 3 :=
by
  have : x = (Real.sqrt 5 + 1) / 2 := rfl
  have floor_x_eq : floor_x = 1 := 
    by
      simp [floor_x, x]
      apply Int.floor_eq 1
      norm_num
      linarith [Real.sqrt_pos.mpr (by norm_num : (0 : ℝ) < 5)]
  have frac_x_eq : frac_x = (Real.sqrt 5 - 1) / 2 :=
    by
      dsimp [frac_x]
      rw [floor_x_eq]
      refl
  exact false.elim sorry -- placeholder for further proof

end sequence_geometric_not_arithmetic_l724_724469


namespace perimeter_of_triangle_ADE_l724_724360

noncomputable def ellipse_perimeter (a b : ℝ) (h : a > b) (e : ℝ) (he : e = 1/2) (h_ellipse : ∀ (x y : ℝ), 
                            x^2 / a^2 + y^2 / b^2 = 1) : ℝ :=
13 -- we assert that the perimeter is 13

theorem perimeter_of_triangle_ADE 
  (a b : ℝ) (h : a > b) (e : ℝ) (he : e = 1/2) 
  (C_eq : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1) 
  (upper_vertex_A : ℝ × ℝ)
  (focus_F1 : ℝ × ℝ)
  (focus_F2 : ℝ × ℝ)
  (line_through_F1_perpendicular_to_AF2 : ∀ x y, y = (√3 / 3) * (x + focus_F1.1))
  (points_D_E_on_ellipse : ∃ D E : ℝ × ℝ, line_through_F1_perpendicular_to_AF2 D.1 D.2 = true ∧
    line_through_F1_perpendicular_to_AF2 E.1 E.2 = true ∧ 
    (dist D E = 6)) :
  ∃ perimeter : ℝ, perimeter = ellipse_perimeter a b h e he C_eq :=
sorry

end perimeter_of_triangle_ADE_l724_724360


namespace subsequence_of_length_11_l724_724134

theorem subsequence_of_length_11 (a : Fin 101 → ℕ) :
  ∃ S : Finset (Fin 101), S.card = 11 ∧ 
  ((∀ i j ∈ S, i < j → a i < a j) ∨ (∀ i j ∈ S, i < j → a i > a j)) :=
sorry

end subsequence_of_length_11_l724_724134


namespace max_distance_thm_curve_C1_rect_eq_l724_724786

section

-- Defining the parametric equations of line l
def line_l (t : ℝ) : ℝ × ℝ :=
  (1 - 2 * sqrt 5 / 5 * t, 1 + sqrt 5 / 5 * t)

-- Defining the polar coordinate equation of curve C1
def curve_C1 (theta : ℝ) : ℝ :=
  4 * cos theta

-- Defining the parametric equations of curve C2
def curve_C2 (alpha : ℝ) : ℝ × ℝ :=
  (cos alpha, sin alpha)

-- Defining the transformation τ to get curve C3
def transform_curve_C3 (alpha : ℝ) : ℝ × ℝ :=
  (2 * cos alpha, sin alpha)

-- Defining the midpoint M of segment PQ
def midpoint_M (alpha : ℝ) : ℝ × ℝ :=
  ((2 + 2 * cos alpha) / 2, (2 + sin alpha) / 2)

-- Equation of line l in rectangular coordinates
def line_l_eq (x y : ℝ) : Prop :=
  x + 2*y - 3 = 0

-- Polar and rectangular conversion for curve C1
def curve_C1_rect (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 4

-- Maximum distance formula
def max_distance (alpha : ℝ) : ℝ :=
  abs (sqrt 2 * sin (alpha + π / 4)) / sqrt 5

theorem max_distance_thm : ∃ α : ℝ, α = 3 * π / 4 ∧ max_distance α = sqrt 10 / 5 :=
  sorry

theorem curve_C1_rect_eq (x y : ℝ) : curve_C1_rect x y ↔ (x - 2)^2 + y^2 = 4 :=
  sorry

end

end max_distance_thm_curve_C1_rect_eq_l724_724786


namespace number_of_members_in_league_l724_724071

-- Define the costs of the items considering the conditions
def sock_cost : ℕ := 6
def tshirt_cost : ℕ := sock_cost + 3
def shorts_cost : ℕ := sock_cost + 2

-- Define the total cost for one member
def total_cost_one_member : ℕ := 
  2 * (sock_cost + tshirt_cost + shorts_cost)

-- Given total expenditure
def total_expenditure : ℕ := 4860

-- Define the theorem to be proved
theorem number_of_members_in_league :
  total_expenditure / total_cost_one_member = 106 :=
by 
  sorry

end number_of_members_in_league_l724_724071


namespace Petya_tore_out_sheets_l724_724889

theorem Petya_tore_out_sheets (n m : ℕ) (h1 : n = 185) (h2 : m = 518)
  (h3 : m.digits = n.digits) : (m - n + 1) / 2 = 167 :=
by
  sorry

end Petya_tore_out_sheets_l724_724889


namespace equal_sequences_l724_724037

theorem equal_sequences {n : ℕ} (h_pos : 0 < n) 
  (a b : Fin n → ℝ) 
  (nondecreasing_a : ∀ i j, i ≤ j → a i ≤ a j) 
  (nondecreasing_b : ∀ i j, i ≤ j → b i ≤ b j)
  (sum_le : ∀ i, (Finset.range i).sum (λ j, a j) ≤ (Finset.range i).sum (λ j, b j))
  (sum_eq : (Finset.range n).sum (λ j, a j) = (Finset.range n).sum (λ j, b j))
  (pair_equal : ∀ m, (Fin (n * n)).countp (λ p, let ⟨i, j⟩ := FinTuple_promote p in a i - a j = m) =
                    (Fin (n * n)).countp (λ p, let ⟨k, l⟩ := FinTuple_promote p in b k - b l = m)) :
  ∀ i, a i = b i :=
by
  sorry

end equal_sequences_l724_724037


namespace probability_all_three_same_group_l724_724029

theorem probability_all_three_same_group
  (num_students : ℕ)
  (num_groups : ℕ)
  (group_size : ℕ)
  (h_num_students : num_students = 900)
  (h_num_groups : num_groups = 4)
  (h_group_size : group_size = num_students / num_groups)
  (h_friends : 3 ≤ num_students) :
  (group_size = 225) → 
  (∃ prob : ℚ, prob = 1 / 16) :=
by
  intro h_group_size_225
  use (1 / 16)
  sorry

end probability_all_three_same_group_l724_724029


namespace binomial_coefficient_10_3_l724_724287

theorem binomial_coefficient_10_3 : nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l724_724287


namespace maximum_a_value_l724_724705

noncomputable def g (t m : ℝ) : ℝ := Real.exp t * (t^3 - 6 * t^2 + 3 * t + m)

def condition (a : ℝ) : Prop :=
  ∃ (m : ℝ), (0 ≤ m ∧ m ≤ 5) ∧ ∀ (t : ℝ), (1 ≤ t ∧ t ≤ a) → g(t,m) ≤ t

theorem maximum_a_value : condition 5 ∧ ∀ (a : ℝ), condition a → a ≤ 5 :=
by
  sorry

end maximum_a_value_l724_724705


namespace car_fewer_minutes_than_bus_l724_724633

-- Conditions translated into Lean definitions
def bus_time_to_beach : ℕ := 40
def car_round_trip_time : ℕ := 70

-- Derived condition
def car_one_way_time : ℕ := car_round_trip_time / 2

-- Theorem statement to be proven
theorem car_fewer_minutes_than_bus : car_one_way_time = bus_time_to_beach - 5 := by
  -- This is the placeholder for the proof
  sorry

end car_fewer_minutes_than_bus_l724_724633


namespace exists_composite_evaluation_l724_724978

noncomputable def is_composite (n : ℕ) : Prop :=
  ∃ (p q : ℕ), p > 1 ∧ q > 1 ∧ n = p * q

theorem exists_composite_evaluation (n : ℕ) (F : fin n → ℤ[X]) :
  ∃ a : ℤ, ∀ i : fin n, is_composite (eval a (F i).natAbs) := by
  sorry

end exists_composite_evaluation_l724_724978


namespace binomial_coeff_sum_l724_724812

theorem binomial_coeff_sum :
  let a := (λ x: ℝ, (1 - 2*x)^10)
  let b (n : ℕ) := (a x).coeff n
  b 0 = 1 →
  (1 + (b 1 / 2) + (b 2 / (2^2)) + (b 3 / (2^3)) + (b 4 / (2^4))
    + (b 5 / (2^5)) + (b 6 / (2^6)) + (b 7 / (2^7)) + (b 8 / (2^8))
    + (b 9 / (2^9)) + (b 10 / (2^10))) = 0 →
  (b 1 / 2) + (b 2 / (2^2)) + (b 3 / (2^3)) + (b 4 / (2^4))
    + (b 5 / (2^5)) + (b 6 / (2^6)) + (b 7 / (2^7)) + (b 8 / (2^8))
    + (b 9 / (2^9)) + (b 10 / (2^10)) = -1 := by
  sorry

end binomial_coeff_sum_l724_724812


namespace simplified_fraction_of_num_l724_724962

def num : ℚ := 368 / 100

theorem simplified_fraction_of_num : num = 92 / 25 := by
  sorry

end simplified_fraction_of_num_l724_724962


namespace irrational_infinitely_many_approximations_l724_724087

theorem irrational_infinitely_many_approximations (x : ℝ) (hx : Irrational x) (hx_pos : 0 < x) :
  ∃ᶠ (q : ℕ) in at_top, ∃ p : ℤ, |x - p / q| < 1 / q^2 :=
sorry

end irrational_infinitely_many_approximations_l724_724087


namespace perimeter_of_triangle_ADE_l724_724348

noncomputable def ellipse (x y a b : ℝ) : Prop :=
  (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1

def foci_distance (a : ℝ) : ℝ := a / 2

def line_through_f1_perpendicular_to_af2 (x y c : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * (x + c)

def distance_between_points (x1 x2 y1 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem perimeter_of_triangle_ADE
  (a b c : ℝ)
  (h_ellipse : ∀ (x y : ℝ), ellipse x y a b)
  (h_eccentricity : b = Real.sqrt 3 * c)
  (h_foci_distance : foci_distance a = c)
  (h_line : ∀ (x y : ℝ), line_through_f1_perpendicular_to_af2 x y c)
  (h_DE : ∀ (x1 y1 x2 y2 : ℝ), distance_between_points x1 x2 y1 y2 = 6) :
  perimeter_of_triangle_ADE = 13 := sorry

end perimeter_of_triangle_ADE_l724_724348


namespace problem_statement_l724_724701

theorem problem_statement :
  let a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ
  in (∀ x : ℝ, (x + 2)^8 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + a_4 * (x + 1)^4 + a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + a_7 * (x + 1)^7 + a_8 * (x + 1)^8)
  → a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 + 6 * a_6 + 7 * a_7 + 8 * a_8 = 1024 := by
  sorry

end problem_statement_l724_724701


namespace josh_and_fred_age_l724_724801

theorem josh_and_fred_age
    (a b k : ℕ)
    (h1 : 10 * a + b > 10 * b + a)
    (h2 : 99 * (a^2 - b^2) = k^2)
    (ha : a ≥ 0 ∧ a ≤ 9)
    (hb : b ≥ 0 ∧ b ≤ 9) : 
    10 * a + b = 65 ∧ 
    10 * b + a = 56 := 
sorry

end josh_and_fred_age_l724_724801


namespace perimeter_of_triangle_ADE_l724_724371

theorem perimeter_of_triangle_ADE
  (a b : ℝ) (F1 F2 A : ℝ × ℝ) (D E : ℝ × ℝ) 
  (h_ellipse : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1)
  (h_a_gt_b : a > b)
  (h_b_gt_0 : b > 0)
  (h_eccentricity : ∃ c, c / a = 1 / 2 ∧ a^2 - b^2 = c^2)
  (h_F1_F2 : ∀ F1 F2, distance F1 (0, 0) = distance F2 (0, 0) ∧ F1 ≠ F2 ∧ 
                       ∀ P : ℝ × ℝ, (distance P F1 + distance P F2 = 2 * a) ↔ (x : ℝ)(y : ℝ) (h_ellipse x y))
  (h_line_DE : ∃ k, ∃ c, ∀ x F1 A, (2 * a * x/(sqrt k^2 + 1)) = |DE|
  (h_length_DE : |DE| = 6)
  (h_A_vertex : A = (0, b))
  : ∃ perim : ℝ, perim = 13 :=
sorry

end perimeter_of_triangle_ADE_l724_724371


namespace log_seq_not_ap_gp_l724_724041

variables {a b c n d : ℕ}
hypothesis h1 : a < b
hypothesis h2 : b < c
hypothesis h3 : a > 1
hypothesis h4 : b = a + d
hypothesis h5 : c = a + 2 * d
hypothesis h6 : n > 1

def log_seq_is_not_ap_gp : Prop :=
  ¬ (is_arithmetic_prog (log a n) (log b n) (log c n) ∨ 
     is_geometric_prog (log a n) (log b n) (log c n))

theorem log_seq_not_ap_gp :
  log_seq_is_not_ap_gp :=
sorry

end log_seq_not_ap_gp_l724_724041


namespace decimal_to_fraction_l724_724966

theorem decimal_to_fraction :
  (368 / 100 : ℚ) = (92 / 25 : ℚ) := by
  sorry

end decimal_to_fraction_l724_724966


namespace gcd_max_1001_l724_724169

theorem gcd_max_1001 (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1001) : 
  ∃ d, d = Nat.gcd a b ∧ d ≤ 143 := 
sorry

end gcd_max_1001_l724_724169


namespace sequence_equality_l724_724735

theorem sequence_equality (a : ℕ → ℝ) (h₁ : ∀ n : ℕ, a n > 0) (h₂ : ∀ n : ℕ, (∑ j in finset.range n, (a j) ^ 3) = (∑ j in finset.range n, a j) ^ 2) :
  ∀ n : ℕ, a n = n :=
by
  sorry

end sequence_equality_l724_724735


namespace min_value_expression_l724_724466

/-- Prove that for integers a, b, c satisfying 1 ≤ a ≤ b ≤ c ≤ 5, the minimum value of the expression 
  (a - 2)^2 + ((b + 1) / a - 1)^2 + ((c + 1) / b - 1)^2 + (5 / c - 1)^2 is 1.2595. -/
theorem min_value_expression (a b c : ℤ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  ∃ (min_val : ℝ), min_val = ((a - 2)^2 + ((b + 1) / a - 1)^2 + ((c + 1) / b - 1)^2 + (5 / c - 1)^2) ∧ min_val = 1.2595 :=
by
  sorry

end min_value_expression_l724_724466


namespace equilateral_triangle_side_length_l724_724499

theorem equilateral_triangle_side_length
  (P Q R S : Point)
  (A B C : Point)
  (h1 : PQ = 1)
  (h2 : PR = 2)
  (h3 : PS = 3)
  (h4 : PointInsideTriangle P A B C)
  (h₅ : PerpendicularFoot P A B Q)
  (h₆ : PerpendicularFoot P B C R)
  (h₇ : PerpendicularFoot P C A S)
  (h₈ : EquilateralTriangle A B C) :
  side_length A B = 4 * sqrt 3 :=
sorry

end equilateral_triangle_side_length_l724_724499


namespace intervals_of_monotonicity_and_extremum_range_of_a_l724_724831

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem intervals_of_monotonicity_and_extremum :
  (∀ x, f'(x) < 0 ↔ x < -1) ∧ (∀ x, f'(x) > 0 ↔ x > -1) ∧
  (f(-1) = -1/Real.exp 1) ∧ (¬ ∃ x, f(x) = ⊤) :=
by sorry

theorem range_of_a :
  ∃ (a : ℝ), (∀ x1 x2, a < x1 → x1 < x2 → (f(x2) - f(a)) / (x2 - a) > (f(x1) - f(a)) / (x1 - a)) ∧ 
  a ∈ Icc (-2) (⊤) :=
by sorry

end intervals_of_monotonicity_and_extremum_range_of_a_l724_724831


namespace rational_root_even_coeff_l724_724845

theorem rational_root_even_coeff {a b c : ℤ} (h_rational_root : ∃ x : ℚ, a * x^2 + b * x + c = 0) (h_a_nonzero : a ≠ 0) :
  even a ∨ even b ∨ even c :=
sorry

end rational_root_even_coeff_l724_724845


namespace area_of_circle_r_is_16_percent_of_circle_s_l724_724226

open Real

variables (Ds Dr Rs Rr As Ar : ℝ)

def circle_r_is_40_percent_of_circle_s (Ds Dr : ℝ) := Dr = 0.40 * Ds
def radius_of_circle (D : ℝ) (R : ℝ) := R = D / 2
def area_of_circle (R : ℝ) (A : ℝ) := A = π * R^2
def percentage_area (As Ar : ℝ) (P : ℝ) := P = (Ar / As) * 100

theorem area_of_circle_r_is_16_percent_of_circle_s :
  ∀ (Ds Dr Rs Rr As Ar : ℝ),
    circle_r_is_40_percent_of_circle_s Ds Dr →
    radius_of_circle Ds Rs →
    radius_of_circle Dr Rr →
    area_of_circle Rs As →
    area_of_circle Rr Ar →
    percentage_area As Ar 16 := by
  intros Ds Dr Rs Rr As Ar H1 H2 H3 H4 H5
  sorry

end area_of_circle_r_is_16_percent_of_circle_s_l724_724226


namespace line_standard_equation_curve_cartesian_equation_distance_between_intersections_l724_724011

theorem line_standard_equation : 
  ∀ t : ℝ, 
  (1 + ( √2 / 2) * t) - (( √2 / 2) * t) - 1 = 0 := 
by 
  sorry

theorem curve_cartesian_equation : 
  ∀ ρ θ : ℝ, 
  (ρ = 4 * cos θ) → 
  ( (4 * cos θ)^2 + (ρ * sin θ)^2 - 4 * (ρ * cos θ) = 0 ) := 
by 
  sorry

theorem distance_between_intersections :
  ∀ (t1 t2 : ℝ), 
  (t1 ^ 2 - √2 * t1 - 3 = 0) ∧ 
  (t2 ^ 2 - √2 * t2 - 3 = 0) → 
  abs (t1 - t2) = √14 :=
by 
  sorry

end line_standard_equation_curve_cartesian_equation_distance_between_intersections_l724_724011


namespace cube_div_identity_l724_724205

theorem cube_div_identity (a b : ℕ) (h₁ : a = 6) (h₂ : b = 3) :
  (a^3 + b^3) / (a^2 - a * b + b^2) = 9 := by
  sorry

end cube_div_identity_l724_724205


namespace numbers_removed_19th_92nd_l724_724081

/-- Prove that the 19th and 92nd numbers removed are 225 and 6084 respectively,
    given the removal of all perfect squares and perfect cubes from the set of natural numbers. -/
theorem numbers_removed_19th_92nd :
  let natural_numbers := {n : ℕ | true} sub {m : ℕ | ∃ k : ℕ, k^2 = m} sub {p : ℕ | ∃ q : ℕ, q^3 = p},
  removed_numbers := list.diff (list.of_fn (λ x, x + 1)) (list.of_fn (λ n, n^2) ++ list.of_fn (λ n, n^3)),
  n_19 := list.nth_le removed_numbers 18 (by sorry),
  n_92 := list.nth_le removed_numbers 91 (by sorry)
in
  n_19 = 225 ∧ n_92 = 6084 :=
sorry

end numbers_removed_19th_92nd_l724_724081


namespace cost_per_sqm_plastering_l724_724268

-- Define the given constants and conditions
def tank_length : ℝ := 25
def tank_width : ℝ := 12
def tank_depth : ℝ := 6
def total_cost : ℝ := 558

-- Define the expressions for surface areas
def longer_walls_area : ℝ := 2 * (tank_length * tank_depth)
def shorter_walls_area : ℝ := 2 * (tank_width * tank_depth)
def bottom_area : ℝ := tank_length * tank_width
def total_surface_area : ℝ := longer_walls_area + shorter_walls_area + bottom_area

-- Define the expected cost per square meter
def cost_per_sq_m : ℝ := total_cost / total_surface_area

-- Math proof problem statement
theorem cost_per_sqm_plastering : cost_per_sq_m = 0.75 := sorry

end cost_per_sqm_plastering_l724_724268


namespace product_expression_zero_l724_724305

theorem product_expression_zero : 
  (∏ n in Finset.range 50, (1 - (1 / (n + 1)))) = 0 :=
by
  sorry

end product_expression_zero_l724_724305


namespace smallest_possible_value_l724_724012

-- Definitions of the digits
def P := 1
def A := 9
def B := 2
def H := 8
def O := 3

-- Expression for continued fraction T
noncomputable def T : ℚ :=
  P + 1 / (A + 1 / (B + 1 / (H + 1 / O)))

-- The goal is to prove that T is the smallest possible value given the conditions
theorem smallest_possible_value : T = 555 / 502 :=
by
  -- The detailed proof would be done here, but for now we use sorry because we only need the statement
  sorry

end smallest_possible_value_l724_724012


namespace num_correct_statements_is_one_l724_724870

-- Definitions based on conditions mentioned in part a)
def condition1 (L1 L2 L3 : Set Point) : Prop := 
  ∃ p1 p2 p3, p1 ∈ L1 ∧ p2 ∈ L2 ∧ p3 ∈ L3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
  (∀ p, p ∈ L1 → p ∈ L2 → p ∈ L3 → False)

def condition2 (L1 L2 L3 : Set Point) : Prop := 
  ∀ p q, (p ∉ L1 ∨ q ∉ L2 ∨ parallel p q) ∧ (p ∉ L2 ∨ q ∉ L3 ∨ parallel p q) ∧ (p ∉ L3 ∨ q ∉ L1 ∨ parallel p q)

def condition3 (L1 L2 L3 : Set Point) : Prop := 
  ∃ p, p ∈ L1 ∧ p ∈ L2 ∧ p ∈ L3

def condition4 (L1 L2 L3 : Set Point) : Prop := 
  ∃ p q, (p ∈ L1 ∧ perpendicular p q ∧ q ∈ L2 ∧ ∃ r, r ∈ L3 ∧ perpendicular q r)

-- Main theorem statement
theorem num_correct_statements_is_one (L1 L2 L3 : Set Point) : 
  (condition1 L1 L2 L3 ∨ condition2 L1 L2 L3 ∨ condition3 L1 L2 L3 ∨ condition4 L1 L2 L3) → 
  (condition1 L1 L2 L3 → ¬ condition2 L1 L2 L3 ∧ ¬ condition3 L1 L2 L3 ∧ ¬ condition4 L1 L2 L3) ∧ 
  (condition2 L1 L2 L3 → ¬ condition1 L1 L2 L3 ∧ ¬ condition3 L1 L2 L3 ∧ ¬ condition4 L1 L2 L3) ∧ 
  (condition3 L1 L2 L3 → ¬ condition1 L1 L2 L3 ∧ ¬ condition2 L1 L2 L3 ∧ ¬ condition4 L1 L2 L3) ∧ 
  (condition4 L1 L2 L3 → ¬ condition1 L1 L2 L3 ∧ ¬ condition2 L1 L2 L3 ∧ ¬ condition3 L1 L2 L3) ∧ 
  1 = 1 := 
sorry

end num_correct_statements_is_one_l724_724870


namespace gcd_102_238_l724_724193

def gcd (a b : ℕ) : ℕ := if b = 0 then a else gcd b (a % b)

theorem gcd_102_238 : gcd 102 238 = 34 :=
by
  sorry

end gcd_102_238_l724_724193


namespace combined_tax_rate_correct_l724_724227

noncomputable def john_income : ℕ := 57000
noncomputable def ingrid_income : ℕ := 72000
noncomputable def john_tax_rate : ℝ := 0.30
noncomputable def ingrid_tax_rate : ℝ := 0.40

noncomputable def combined_tax_rate : ℝ :=
  (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income) * 100

theorem combined_tax_rate_correct : combined_tax_rate ≈ 35.58 := 
  sorry

end combined_tax_rate_correct_l724_724227


namespace find_m_range_l724_724394

noncomputable def ellipse_symmetric_points_range (m : ℝ) : Prop :=
  -((2:ℝ) * Real.sqrt (13:ℝ) / 13) < m ∧ m < ((2:ℝ) * Real.sqrt (13:ℝ) / 13)

theorem find_m_range :
  ∃ m : ℝ, ellipse_symmetric_points_range m :=
sorry

end find_m_range_l724_724394


namespace solution_set_of_inequality_l724_724757

variable (f : ℝ → ℝ)
variable (x : ℝ)
variable (h1 : ∀ x : ℝ, f(x) + f'(x) > 1)
variable (h2 : f(0) = 4)

theorem solution_set_of_inequality : (f x > 3 / real.exp x + 1) ↔ (x > 0) :=
sorry

end solution_set_of_inequality_l724_724757


namespace sum_of_special_primes_l724_724567

open Nat

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def reverse_digits (n : ℕ) : ℕ :=
(n % 10) * 10 + n / 10

def two_digit_primes_between_20_and_90 := { n : ℕ | is_prime n ∧ 20 < n ∧ n < 90 }

theorem sum_of_special_primes : ∑ n in two_digit_primes_between_20_and_90, is_prime (reverse_digits n) → n = 291 :=
by
  sorry

end sum_of_special_primes_l724_724567


namespace melissa_remaining_bananas_l724_724836

theorem melissa_remaining_bananas :
  let initial_bananas := 88
  let shared_bananas := 4
  initial_bananas - shared_bananas = 84 :=
by
  sorry

end melissa_remaining_bananas_l724_724836


namespace pentagon_area_is_correct_l724_724259

-- Definitions for the conditions given in the problem
def pentagon_sides : list ℕ := [17, 23, 28, 29, 35]

noncomputable def side_a := 23
noncomputable def side_b := 35
noncomputable def side_c := 28
noncomputable def side_d := 17
noncomputable def side_e := 35

-- Pythagorean triple condition
def r := side_c
def s := side_b
def e := side_e

axiom pythagorean_triple : r^2 + s^2 = e^2

-- Area calculation
noncomputable def pentagon_area : ℕ :=
  side_c * side_b - (1 / 2 : ℚ) * (r * s).to_rat

-- Proof statement
theorem pentagon_area_is_correct : pentagon_area.to_int = 574 := by
  -- Ensuring the set contains exactly these sides
  have h1 : pentagon_sides.perm [side_a, side_b, side_c, side_d, side_e] := by
    sorry
  -- Using the Pythagorean triple condition
  have h2 : pythagorean_triple := by
    sorry
  -- Finally proving the area
  have h_area : pentagon_area.to_int = 574 := by
    sorry
  exact h_area

end pentagon_area_is_correct_l724_724259


namespace product_floor_ceil_expression_l724_724662

theorem product_floor_ceil_expression : 
  (∏ n in finset.range 5, let m := (n + 6) in (Int.floor (-(m + 0.5)) * Int.ceil (m + 0.5))) = -3074593760 :=
by
  sorry

end product_floor_ceil_expression_l724_724662


namespace cost_price_of_coat_l724_724245

variable (x : ℝ)

-- Condition 1: The coat is marked up by 25% from its cost price.
def markup (x : ℝ) : ℝ := 1.25 * x

-- Condition 2: The selling price after markup is 275 yuan.
def selling_price : ℝ := 275

-- Prove that the cost price of the coat, given the conditions, is 220 yuan.
theorem cost_price_of_coat : x = 220 :=
by
  have h1 : markup x = 275 := sorry
  have h2 : 1.25 * x = 275 := by
    rw [markup] at h1
    exact h1
  have h3 : x = 220 := by
    rw [← div_eq_iff (ne_of_gt (by norm_num : 1.25 > 0)), div_self (ne_of_gt (by norm_num : 1.25 > 0))] at h2
    convert h2.symm
    norm_num
  exact h3

end cost_price_of_coat_l724_724245


namespace shelby_rain_minutes_l724_724086

/-- 
Shelby's driving speed in miles per hour.
-/
def speed_non_rain : ℕ := 40

/-- 
Shelby's driving speed in the rain in miles per hour.
-/
def speed_rain : ℕ := 25

/-- 
Total distance driven in miles.
-/
def total_distance : ℕ := 20

/-- 
Total time driven in minutes.
-/
def total_time : ℕ := 36

/--
Convert speed from miles per hour to miles per minute.
-/
def speed_to_miles_per_min (speed : ℕ) : ℚ := speed / 60

/-- 
Prove that Shelby drove in the rain for 16 minutes.
-/
theorem shelby_rain_minutes : 
  ∃ (x : ℕ),  
    let distance_non_rain := speed_to_miles_per_min speed_non_rain * (total_time - x)
        distance_rain := speed_to_miles_per_min speed_rain * x
    in 
      distance_non_rain + distance_rain = total_distance ∧ 
      x = 16 :=
by 
  sorry

end shelby_rain_minutes_l724_724086


namespace decimal_to_fraction_simplify_l724_724970

theorem decimal_to_fraction_simplify (d : ℚ) (h : d = 3.68) : d = 92 / 25 :=
by
  rw h
  sorry

end decimal_to_fraction_simplify_l724_724970


namespace triangle_ADE_perimeter_l724_724375

noncomputable def ellipse_perimeter (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) (e : ℝ) (h₃ : e = (1 / 2)) 
(F₁ F₂ : ℝ × ℝ) (h₄ : F₁ ≠ F₂) (D E : ℝ × ℝ) (h₅ : |D - E| = 6) : ℝ :=
  let c := (sqrt (a ^ 2 - b ^ 2)) in
  let A := (0, b) in
  let AD := sqrt ((fst D) ^ 2 + (snd D - b) ^ 2) in
  let AE := sqrt ((fst E) ^ 2 + (snd E - b) ^ 2) in
  AD + AE + |D - E|

theorem triangle_ADE_perimeter (a b : ℝ) (h₁ : a > b > 0) (e : ℝ) (h₂ : e = (1 / 2))
(F₁ F₂ : ℝ × ℝ) (h₃ : F₁ ≠ F₂)
(D E : ℝ × ℝ) (h₄ : |D - E| = 6) : 
  ellipse_perimeter a b (and.left h₁) (and.right h₁) e h₂ F₁ F₂ h₃ D E h₄ = 19 :=
sorry

end triangle_ADE_perimeter_l724_724375


namespace pages_torn_and_sheets_calculation_l724_724901

theorem pages_torn_and_sheets_calculation : 
  (∀ (n : ℕ), (sheet_no n) = (n + 1) / 2 → (2 * (n + 1) / 2) - 1 = n ∨ 2 * (n + 1) / 2 = n) →
  let first_page := 185 in
  let last_page := 518 in
  last_page = 518 → 
  ((last_page - first_page + 1) / 2) = 167 := 
by
  sorry

end pages_torn_and_sheets_calculation_l724_724901


namespace trigonometric_identity_l724_724594

theorem trigonometric_identity :
  sin (-1071 * real.pi / 180) * sin (99 * real.pi / 180) +
  sin (-171 * real.pi / 180) * sin (-261 * real.pi / 180) +
  tan (-1089 * real.pi / 180) * tan (-540 * real.pi / 180) = 0 :=
by
  sorry

end trigonometric_identity_l724_724594


namespace smallest_portion_is_five_thirds_l724_724591

theorem smallest_portion_is_five_thirds
    (a1 a2 a3 a4 a5 : ℚ)
    (h1 : a2 = a1 + 1)
    (h2 : a3 = a1 + 2)
    (h3 : a4 = a1 + 3)
    (h4 : a5 = a1 + 4)
    (h_sum : a1 + a2 + a3 + a4 + a5 = 100)
    (h_cond : (1 / 7) * (a3 + a4 + a5) = a1 + a2) :
    a1 = 5 / 3 :=
by
  sorry

end smallest_portion_is_five_thirds_l724_724591


namespace AM_GM_Inequality_geq_2_pow_n_l724_724692

theorem AM_GM_Inequality_geq_2_pow_n (n : ℕ) (a : Fin n → ℝ) 
  (h_pos : ∀ i, 0 < a i)
  (h_prod : (∏ i, a i) = 1) : 
  (∏ i, 1 + a i) ≥ 2^n := 
by sorry

end AM_GM_Inequality_geq_2_pow_n_l724_724692


namespace relationship_oil_distance_remaining_oil_at_100_remaining_oil_at_200_remaining_oil_at_300_remaining_oil_at_400_remaining_oil_at_350_distance_when_oil_8_l724_724238

variables (y x : ℝ)

def initial_oil := 56
def consumption_rate := 0.08

theorem relationship_oil_distance (x : ℝ) (h : 0 ≤ x):
  y = initial_oil - consumption_rate * x :=
sorry

theorem remaining_oil_at_100 :
  y = initial_oil - consumption_rate * 100 :=
sorry

theorem remaining_oil_at_200:
  y = 40 :=
sorry

theorem remaining_oil_at_300 :
  y = initial_oil - consumption_rate * 300 :=
sorry

theorem remaining_oil_at_400 :
  y = 24 :=
sorry

theorem remaining_oil_at_350 :
  y = initial_oil - consumption_rate * 350 :=
sorry

theorem distance_when_oil_8 :
  initial_oil - consumption_rate * x = 8 → x = 600 :=
sorry

end relationship_oil_distance_remaining_oil_at_100_remaining_oil_at_200_remaining_oil_at_300_remaining_oil_at_400_remaining_oil_at_350_distance_when_oil_8_l724_724238


namespace shipping_cost_per_unit_l724_724247

noncomputable def fixed_monthly_costs : ℝ := 16500
noncomputable def production_cost_per_component : ℝ := 80
noncomputable def production_quantity : ℝ := 150
noncomputable def selling_price_per_component : ℝ := 193.33

theorem shipping_cost_per_unit :
  ∀ (S : ℝ), (production_quantity * production_cost_per_component + production_quantity * S + fixed_monthly_costs) ≤ (production_quantity * selling_price_per_component) → S ≤ 3.33 :=
by
  intro S
  sorry

end shipping_cost_per_unit_l724_724247


namespace problem_solution_l724_724952

theorem problem_solution :
  (12345 * 5 + 23451 * 4 + 34512 * 3 + 45123 * 2 + 51234 * 1 = 400545) :=
by
  sorry

end problem_solution_l724_724952


namespace marks_history_and_government_l724_724622

/-- Define the problem conditions -/
def marks_geography : ℝ := 56
def marks_art : ℝ := 72
def marks_computer_science : ℝ := 85
def marks_modern_literature : ℝ := 80
def avg_marks : ℝ := 70.6
def num_subjects : ℕ := 5

/-- Calculate total marks from average marks and number of subjects -/
def total_marks : ℝ := avg_marks * num_subjects

/-- Given the problem conditions, prove that the marks obtained in history and government is 60 -/
theorem marks_history_and_government : 
  let m_h := total_marks - (marks_geography + marks_art + marks_computer_science + marks_modern_literature) in
  m_h = 60 :=
by
  sorry

end marks_history_and_government_l724_724622


namespace inequality_solution_l724_724110

theorem inequality_solution (x y : ℝ) (h1 : y ≥ x^2 + 1) :
    2^y - 2 * Real.cos x + Real.sqrt (y - x^2 - 1) ≤ 0 ↔ x = 0 ∧ y = 1 :=
by
  sorry

end inequality_solution_l724_724110


namespace range_of_f_l724_724688

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then |x| - 1 else Real.sin x ^ 2

theorem range_of_f : Set.range f = Set.Ioi (-1) := 
  sorry

end range_of_f_l724_724688


namespace modulus_of_z_l724_724758

theorem modulus_of_z (z : ℂ) (h : (3 - 4 * complex.i) * z = 1) : complex.abs z = 1 / 5 :=
sorry

end modulus_of_z_l724_724758


namespace find_Finley_age_l724_724509

-- Definitions based on conditions
def Jill_age : ℕ := 20
def Roger_age (J : ℕ) : ℕ := 2 * J + 5
def age_diff_in_15_years (J R F : ℕ) : Prop := (R + 15) - (J + 15) = F - 30

-- The problem statement to prove
theorem find_Finley_age : ∃ F : ℕ, let J := Jill_age in let R := Roger_age J in age_diff_in_15_years J R F ∧ F = 55 :=
by
  -- The solution steps and actual proof go here
  existsi 55
  unfold Jill_age
  unfold Roger_age
  unfold age_diff_in_15_years
  sorry

end find_Finley_age_l724_724509


namespace vessel_width_l724_724249

-- Define the problem conditions
def edge_length := 16    -- in cm
def base_length := 20    -- in cm
def water_rise := 13.653333333333334 -- in cm
def volume_cube := edge_length * edge_length * edge_length

-- Prove the width of the base of the vessel
theorem vessel_width (W : ℝ) (h : volume_cube = base_length * W * water_rise) : W = 15 := by
  -- Insert the proof here
  sorry

end vessel_width_l724_724249


namespace general_term_given_cond1_general_term_given_cond2_compare_sum_given_S_formula_l724_724330

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Conditions
def condition_one (a : ℕ → ℕ) := a 2 = 1 ∧ 2 * (a 5) - a 3 = 11
def condition_two (a : ℕ → ℕ) (S : ℕ → ℕ) := a 2 = 1 ∧ S 4 = 8

-- General term of the sequence
def general_term (a : ℕ → ℕ) := ∀ n, a n = 2 * n - 3

-- Sum of the first n terms of the sequence a_n
def sum_of_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) := 
  ∀ n, S n = n * (n - 2)

-- Comparison theorem
def compare_sum (b : ℕ → ℝ) (T : ℕ → ℝ) :=
  ∀ n, let Tn := ∑ i in range n, b i + 2 in Tn < 3 / 4

-- Proof statements
theorem general_term_given_cond1 : 
  (condition_one a) → (general_term a) := by
  sorry

theorem general_term_given_cond2 : 
  (condition_two a S) → (general_term a) := by
  sorry

theorem compare_sum_given_S_formula (S : ℕ → ℕ) :
  (sum_of_first_n_terms a S) →
  ∀ n, let b := (λ n, 1 / (S (n + 2) : ℝ)) in 
  let T := summation n (b) in 
  compare_sum b T := by
  sorry

end general_term_given_cond1_general_term_given_cond2_compare_sum_given_S_formula_l724_724330


namespace shift_graph_necessary_l724_724562

noncomputable def shift_graph (x : ℝ) : ℝ := x + π/8

theorem shift_graph_necessary :
  (∀ x : ℝ, 3 * sin (2 * x) = 3 * sin (2 * (shift_graph x) - π/4)) :=
by {
  intro x,
  simp [shift_graph],
  sorry
}

end shift_graph_necessary_l724_724562


namespace direction_vector_value_l724_724343

theorem direction_vector_value
  (P Q : ℝ × ℝ)
  (P_eq : P = (-3, 6))
  (Q_eq : Q = (2, -1))
  (dir_vec : ℝ × ℝ)
  (dir_vec_eq : dir_vec = (b, -1)) :
  b = 5/7 :=
begin
  cases P_eq,
  cases Q_eq,
  cases dir_vec_eq,
  sorry,
end

end direction_vector_value_l724_724343


namespace priyas_age_l724_724844

/-- 
  Let P be Priya's current age, and F be her father's current age. 
  Given:
  1. F = P + 31
  2. (P + 8) + (F + 8) = 69
  Prove: Priya's current age P is 11.
-/
theorem priyas_age 
  (P F : ℕ) 
  (h1 : F = P + 31) 
  (h2 : (P + 8) + (F + 8) = 69) 
  : P = 11 :=
by
  sorry

end priyas_age_l724_724844


namespace cheezit_bag_weight_l724_724795

-- Definitions based on the conditions of the problem
def cheezit_bags : ℕ := 3
def calories_per_ounce : ℕ := 150
def run_minutes : ℕ := 40
def calories_per_minute : ℕ := 12
def excess_calories : ℕ := 420

-- Main theorem stating the question with the solution
theorem cheezit_bag_weight (x : ℕ) : 
  (calories_per_ounce * cheezit_bags * x) - (run_minutes * calories_per_minute) = excess_calories → 
  x = 2 :=
by
  sorry

end cheezit_bag_weight_l724_724795


namespace distance_between_points_l724_724447

noncomputable def distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2)

theorem distance_between_points :
  let A := (1, 0, -2 : ℝ × ℝ × ℝ);
  let B := (-2, 4, 3 : ℝ × ℝ × ℝ) in
  distance A B = 5 * real.sqrt 2 :=
by {
  sorry
}

end distance_between_points_l724_724447


namespace not_sum_six_odd_squares_l724_724645

-- Definition stating that a number is odd.
def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

-- Given that the square of any odd number is 1 modulo 8.
lemma odd_square_mod_eight (n : ℕ) (h : is_odd n) : (n^2) % 8 = 1 :=
sorry

-- Main theorem stating that 1986 cannot be the sum of six squares of odd numbers.
theorem not_sum_six_odd_squares : ¬ ∃ n1 n2 n3 n4 n5 n6 : ℕ, 
    is_odd n1 ∧ is_odd n2 ∧ is_odd n3 ∧ is_odd n4 ∧ is_odd n5 ∧ is_odd n6 ∧
    n1^2 + n2^2 + n3^2 + n4^2 + n5^2 + n6^2 = 1986 :=
sorry

end not_sum_six_odd_squares_l724_724645


namespace focal_distance_equal_l724_724868

-- Definitions of the curves and the focal distances
def curve1 (x y : ℝ) := x^2 / 16 + y^2 / 12 = 1
def curve2 (x y : ℝ) (k : ℝ) := 12 < k ∧ k < 16 ∧ x^2 / (16 - k) + y^2 / (12 - k) = 1
noncomputable def focal_distance_ellipse := real.sqrt (16 - 12)
noncomputable def focal_distance_hyperbola (k : ℝ) := real.sqrt ((16 - k) - (12 - k))

-- The proof statement
theorem focal_distance_equal (k : ℝ) (hk : 12 < k ∧ k < 16) :
  focal_distance_ellipse = focal_distance_hyperbola k := by
  sorry

end focal_distance_equal_l724_724868


namespace polynomial_value_at_2_l724_724203

noncomputable def f (x : ℝ) : ℝ := x^5 + 2*x^4 - 3*x^2 + 7*x - 2

theorem polynomial_value_at_2 : f 2 = 64 :=
by
  let v0 := 1
  let v1 := 2 + 2
  have v1_eq : v1 = 4 := rfl
  let v2 := v1 * 2
  have v2_eq : v2 = 8 := by rw [v1_eq, mul_two]
  let v3 := v2 * 2 - 3
  have v3_eq : v3 = 13 := by rw [v2_eq, mul_two, sub_eq_add_neg, add_neg]
  let v4 := v3 * 2 + 7
  have v4_eq : v4 = 33 := by rw [v3_eq, mul_two, add_comm]
  let v5 := v4 * 2 - 2
  have v5_eq : v5 = 64 := by rw [v4_eq, mul_two, sub_eq_add_neg, add_neg]
  show f 2 = 64
  rw [v5_eq]
  sorry


end polynomial_value_at_2_l724_724203


namespace probability_of_real_root_l724_724848

noncomputable def has_real_root_probability :=
  let interval_1 := set.Icc (-4 : ℝ) (-2)
  let interval_2 := set.Icc (2 : ℝ) (3)
  let allowable_m_values := interval_1 ∪ interval_2
  (set.card allowable_m_values) / (set.card (set.Icc (-4 : ℝ) (3)))

theorem probability_of_real_root : 
  has_real_root_probability = 3 / 7 := 
sorry

end probability_of_real_root_l724_724848


namespace find_f0_sum_condition_l724_724810

-- Assume the function f : ℕ ∪ {0} → ℕ ∪ {0}
def f : ℕ → ℕ

-- Given condition:
axiom functional_eq (a b : ℕ) (h: a ≠ b) : f a + f b - f (a + b) = 2019

-- Problem 1: Prove that f(0) = 2019
theorem find_f0 : f 0 = 2019 := sorry

-- Problem 2: Prove the sum condition for 100 distinct positive integers
theorem sum_condition (a : Fin 100 → ℕ) (h: ∀ i j, i ≠ j → a i ≠ a j) :
  (∑ i, f (a i)) - f (∑ i, a i) = 2019 * 99 := sorry

end find_f0_sum_condition_l724_724810


namespace time_for_B_to_complete_work_l724_724979

theorem time_for_B_to_complete_work (A B : ℝ) (h1 : (1 / 32 : ℝ) = A) (h2 : (A + B = 1 / 16 : ℝ)) : (1 / B) = 32 :=
by
  sorry

end time_for_B_to_complete_work_l724_724979


namespace max_special_points_spherical_planet_l724_724076

noncomputable def max_special_points (S : Type) [metric_space S] (co1 co2 co3 co4 : S) (radius : ℝ) 
    (is_special : S → Prop) : ℕ :=
  sorry

theorem max_special_points_spherical_planet :
  ∀ (S : Type) [metric_space S] [compact_space S] [complete_space S] (co1 co2 co3 co4 : S),
  (∀ (x : S),
   is_in_tetrahedron co1 co2 co3 co4 x →
   ∃ continents: finset S, continents.card = 4 ∧
   ∀ continent ∈ continents, sontent if and only if it is cos of a spherical cap of radius ε < r ∧ is_special x) →
   max_special_points S co1 co2 co3 co4 (tetrahedron_radius co1 co2 co3 co4 fact) 
     (special_point_condition co1 co2 co3 co4 (tetrahedron_radius co1 co2 co3 co4 fact)) = 4 :=
sorry

end max_special_points_spherical_planet_l724_724076


namespace k_plus_a_equals_three_halves_l724_724422

theorem k_plus_a_equals_three_halves :
  ∃ (k a : ℝ), (2 = k * 4 ^ a) ∧ (k + a = 3 / 2) :=
sorry

end k_plus_a_equals_three_halves_l724_724422


namespace pages_torn_l724_724912

theorem pages_torn (n : ℕ) (H1 : n = 185) (H2 : ∃ m, m = 518 ∧ (digits 10 m = digits 10 n) ∧ (m % 2 = 0)) : 
  ∃ k, k = ((518 - 185 + 1) / 2) ∧ k = 167 :=
by sorry

end pages_torn_l724_724912


namespace correct_transformation_option_c_l724_724954

theorem correct_transformation_option_c (x : ℝ) (h : (x / 2) - (x / 3) = 1) : 3 * x - 2 * x = 6 :=
by
  sorry

end correct_transformation_option_c_l724_724954


namespace odd_function_decreasing_l724_724696

theorem odd_function_decreasing (f : ℝ → ℝ) (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x y, x < y → y < 0 → f x > f y) :
  ∀ x y, 0 < x → x < y → f y < f x :=
by
  sorry

end odd_function_decreasing_l724_724696


namespace sum_of_reciprocals_of_primes_lt_10_l724_724040

theorem sum_of_reciprocals_of_primes_lt_10 (n : ℕ) (primes : Fin n → ℕ) (h : ∀ i, prime (primes i) ∧ primes i < 2^100) :
  (∑ i in Finset.range n, 1 / (primes i : ℝ)) < 10 :=
sorry

end sum_of_reciprocals_of_primes_lt_10_l724_724040


namespace proof_statement_l724_724301

noncomputable def problem_statement : Prop :=
  let z1 := 6 * real.cos (real.pi / 3) - 12 * real.sin (real.pi / 3) * complex.I in
  let z2 := 5 * real.cos (real.pi / 6) + 10 * real.sin (real.pi / 6) * complex.I in
  complex.abs (z1 * z2) = 15 * real.sqrt 91 / 2

theorem proof_statement : problem_statement :=
  sorry

end proof_statement_l724_724301


namespace total_filets_from_catch_l724_724223

theorem total_filets_from_catch :
  let ben_fish := [(5, "Bluefish"), (9, "Bluefish"), (9, "Yellowtail"), (9, "Yellowtail")]
  let judy_fish := [(11, "Red Snapper")]
  let billy_fish := [(6, "Bluefish"), (6, "Yellowtail"), (10, "Yellowtail")]
  let jim_fish := [(4, "Red Snapper"), (8, "Bluefish")]
  let susie_fish := [(3, "Red Snapper"), (7, "Yellowtail"), (12, "Yellowtail"), (12, "Bluefish"), (12, "Bluefish")]
  let fish := ben_fish ++ judy_fish ++ billy_fish ++ jim_fish ++ susie_fish
  let limits := [("Bluefish", 7), ("Yellowtail", 6), ("Red Snapper", 8)]
  let validFish := λ (size : ℕ) (species : String) => 
    match limits.lookup species with
    | some minSize => size >= minSize
    | none => false
  let validCatches := fish.filter (λ (size, species) => validFish size species)
  let numFilets := validCatches.length * 2
  numFilets = 22 :=
by
  sorry

end total_filets_from_catch_l724_724223


namespace min_sum_sqrt3_l724_724767

theorem min_sum_sqrt3 (a b c : ℝ) 
  (h1 : a + 2 * b > 0)
  (h2 : 12 - 4 * (a + 2 * b) * (a + 2 * c) ≤ 0) :
  a + b + c ≥ sqrt 3 :=
sorry

end min_sum_sqrt3_l724_724767


namespace floor_sum_eq_l724_724994

theorem floor_sum_eq (n : ℕ) (x : ℝ) : 
  Int.floor (n * x) = Int.floor x + ∑ r in Finset.range n, Int.floor (x + r / n) := 
sorry

end floor_sum_eq_l724_724994


namespace determine_company_of_vladimir_l724_724284

-- Representing conditions as variables
variable (Alexei Boris Vladimir : Prop)
variable is_from_index : Prop     -- whether someone is from 'Index'
variable is_from_zugl : Prop      -- whether someone is from 'Zugl'

/-- Conditions: --/
-- Representatives of the same company always tell the truth to each other and lie to their competitors.
-- Here, we can infer that if someone tells the truth to another, then they are from the same company.
axiom same_company (A B : Prop) : (A = B) ↔ (A ∧ B) ∨ (¬A ∧ ¬B)

-- Alexei's statement: "I am from the 'Index' company."
axiom alexei_says_index : Alexei ↔ is_from_index

-- Boris's reply: "You and Vladimir work for the same company!"
axiom boris_says_same_company : Boris ↔ same_company Alexei Vladimir

-- Proof statement
theorem determine_company_of_vladimir : Vladimir = is_from_index :=
by
  sorry

end determine_company_of_vladimir_l724_724284


namespace exists_max_piles_l724_724062

theorem exists_max_piles (k : ℝ) (hk : k < 2) : 
  ∃ Nk : ℕ, ∀ A : Multiset ℝ, 
    (∀ a ∈ A, ∃ m ∈ A, a ≤ k * m) → 
    A.card ≤ Nk :=
sorry

end exists_max_piles_l724_724062


namespace train_length_proof_l724_724274

noncomputable def speed_km_per_hr : ℝ := 108
noncomputable def time_seconds : ℝ := 9
noncomputable def length_of_train : ℝ := 270
noncomputable def km_to_m : ℝ := 1000
noncomputable def hr_to_s : ℝ := 3600

theorem train_length_proof : 
  (speed_km_per_hr * (km_to_m / hr_to_s) * time_seconds) = length_of_train :=
  by
  sorry

end train_length_proof_l724_724274


namespace find_p0_l724_724467

-- Definitions: polynomial of degree 6 and conditions on values
def polynomial_of_degree_6 (p : ℝ[X]) : Prop :=
  p.degree = 6

def condition_on_values (p : ℝ[X]) : Prop :=
  ∀ n : ℕ, n ≤ 6 → p.eval (3^n) = 1 / 3^n

-- Theorem: given the conditions, find the value of p(0)
theorem find_p0 (p : ℝ[X]) (h_degree : polynomial_of_degree_6 p) (h_conditions : condition_on_values p) : 
  p.eval 0 = 6560 / 2187 := by
  sorry

end find_p0_l724_724467


namespace problem_statement_l724_724048

variable {ι : Type*} {a : ι → ℝ} [decidable_eq ι]

theorem problem_statement 
  (h1 : ∀ i ∈ (finset.range 50), a i ≥ a (100 - i))
  (x : ℕ → ℝ) 
  (h2 : ∀ k ∈ (finset.range 99), x k = (k * a (k + 1)) / finset.sum (finset.range k) a)
  : finset.prod (finset.range 99) (λ k, x k ^ k) ≤ 1 := 
sorry

end problem_statement_l724_724048


namespace richard_boxes_l724_724083

-- Define the function f
def f : ℕ → ℕ
| 1       := 1
| 2       := 2
| (n + 1) := if n = 0 then 2 else f(n) + f(n - 1)

-- Statement of the problem
theorem richard_boxes : f 9 ≤ 89 ∧ f 10 ≤ 89 :=
by {
  sorry
}

end richard_boxes_l724_724083


namespace external_angle_bisector_midpoint_l724_724164

namespace Geometry

def is_midpoint (A B C : Point) : Prop :=
  B = midpoint_line_segment A C

theorem external_angle_bisector_midpoint :
  ∀ (A B C : Point),
  right_triangle A B C 3 4 5 → 
  let C2 := external_angle_bisector A B C
  let A2 := external_angle_bisector B C A
  let B2 := external_angle_bisector C A B
  is_midpoint C2 B2 A2 :=
sorry

end Geometry

end external_angle_bisector_midpoint_l724_724164


namespace triangle_medians_inequality_l724_724439

-- Define the parameters
variables {a b c t_a t_b t_c D : ℝ}

-- Assume the sides and medians of the triangle and the diameter of the circumcircle
axiom sides_of_triangle (a b c : ℝ) : Prop
axiom medians_of_triangle (t_a t_b t_c : ℝ) : Prop
axiom diameter_of_circumcircle (D : ℝ) : Prop

-- The theorem to prove
theorem triangle_medians_inequality
  (h_sides : sides_of_triangle a b c)
  (h_medians : medians_of_triangle t_a t_b t_c)
  (h_diameter : diameter_of_circumcircle D)
  : (a^2 + b^2) / t_c + (b^2 + c^2) / t_a + (c^2 + a^2) / t_b ≤ 6 * D :=
sorry -- proof omitted

end triangle_medians_inequality_l724_724439


namespace reciprocal_of_neg_three_l724_724143

theorem reciprocal_of_neg_three : -3 * (-1 / 3) = 1 := 
by
  sorry

end reciprocal_of_neg_three_l724_724143


namespace bamboo_tube_middle_capacity_l724_724501

-- Definitions and conditions
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + n * d

theorem bamboo_tube_middle_capacity:
  ∃ a d, (arithmetic_sequence a d 0 + arithmetic_sequence a d 1 + arithmetic_sequence a d 2 = 3.9) ∧
         (arithmetic_sequence a d 5 + arithmetic_sequence a d 6 + arithmetic_sequence a d 7 + arithmetic_sequence a d 8 = 3) ∧
         (arithmetic_sequence a d 4 = 1) :=
sorry

end bamboo_tube_middle_capacity_l724_724501


namespace exists_max_piles_l724_724061

theorem exists_max_piles (k : ℝ) (hk : k < 2) : 
  ∃ Nk : ℕ, ∀ A : Multiset ℝ, 
    (∀ a ∈ A, ∃ m ∈ A, a ≤ k * m) → 
    A.card ≤ Nk :=
sorry

end exists_max_piles_l724_724061


namespace top_weight_l724_724218

theorem top_weight (T : ℝ) : 
    (9 * 0.8 + 7 * T = 10.98) → T = 0.54 :=
by 
  intro h
  have H_sum := h
  simp only [mul_add, add_assoc, mul_assoc, mul_comm, add_comm, mul_comm 7] at H_sum
  sorry

end top_weight_l724_724218


namespace geoff_initial_percent_l724_724437

theorem geoff_initial_percent (votes_cast : ℕ) (win_percent : ℝ) (needed_more_votes : ℕ) (initial_votes : ℕ)
  (h1 : votes_cast = 6000)
  (h2 : win_percent = 50.5)
  (h3 : needed_more_votes = 3000)
  (h4 : initial_votes = 31) :
  (initial_votes : ℝ) / votes_cast * 100 = 0.52 :=
by
  sorry

end geoff_initial_percent_l724_724437


namespace reciprocal_of_neg_three_l724_724154

theorem reciprocal_of_neg_three : ∃ (x : ℚ), (-3 * x = 1) ∧ (x = -1 / 3) :=
by
  use (-1 / 3)
  split
  . rw [mul_comm]
    norm_num 
  . norm_num

end reciprocal_of_neg_three_l724_724154


namespace weight_of_new_person_l724_724229

theorem weight_of_new_person 
  (avg_weight_increase : ℝ)
  (old_weight : ℝ) 
  (num_people : ℕ)
  (new_weight_increase : ℝ)
  (total_weight_increase : ℝ)  
  (W : ℝ)
  (h1 : avg_weight_increase = 1.8)
  (h2 : old_weight = 69)
  (h3 : num_people = 6) 
  (h4 : new_weight_increase = num_people * avg_weight_increase) 
  (h5 : total_weight_increase = new_weight_increase)
  (h6 : W = old_weight + total_weight_increase)
  : W = 79.8 := 
by
  sorry

end weight_of_new_person_l724_724229


namespace similar_sizes_bound_l724_724060

theorem similar_sizes_bound (k : ℝ) (hk : k < 2) :
  ∃ (N_k : ℝ), ∀ (A : multiset ℝ), (∀ a ∈ A, a ≤ k * multiset.min A) → 
  A.card ≤ N_k := sorry

end similar_sizes_bound_l724_724060


namespace total_bugs_eaten_l724_724784

theorem total_bugs_eaten :
  let gecko_bugs := 12
  let lizard_bugs := gecko_bugs / 2
  let frog_bugs := lizard_bugs * 3
  let toad_bugs := frog_bugs + (frog_bugs / 2)
  gecko_bugs + lizard_bugs + frog_bugs + toad_bugs = 63 :=
by
  sorry

end total_bugs_eaten_l724_724784


namespace range_a_monotonically_increasing_l724_724421

def g (a x : ℝ) : ℝ := a * x^3 + a * x^2 + x

theorem range_a_monotonically_increasing (a : ℝ) : 
  (∀ x : ℝ, 3 * a * x^2 + 2 * a * x + 1 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 3) := 
sorry

end range_a_monotonically_increasing_l724_724421


namespace thirty_six_hundredths_is_decimal_l724_724555

namespace thirty_six_hundredths

-- Define the fraction representation of thirty-six hundredths
def fraction_thirty_six_hundredths : ℚ := 36 / 100

-- The problem is to prove that this fraction is equal to 0.36 in decimal form
theorem thirty_six_hundredths_is_decimal : fraction_thirty_six_hundredths = 0.36 := 
sorry

end thirty_six_hundredths

end thirty_six_hundredths_is_decimal_l724_724555


namespace num_palindromes_1000_to_3000_l724_724414

theorem num_palindromes_1000_to_3000 : 
  ∃ n : ℕ, n = 30 ∧ ∀ x : ℕ, 1000 ≤ x ∧ x ≤ 3000 ∧ 
  (∀ i j, i ≠ j → i.to_digits 10 x.nth (1+i) = x.nth (4-i)) 
  → x ∈ finset.range 1001 2999 :=
sorry

end num_palindromes_1000_to_3000_l724_724414


namespace sum_carolyn_removed_numbers_l724_724291

open Nat

def initial_list : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def game_carolyn_turn (curr_list : List ℕ) (n : ℕ) : List ℕ :=
  if (curr_list.contains n ∧ (isPrime n ∨ ∃ m ∈ curr_list, m ∣ n ∧ m ≠ n)) then curr_list.erase n else curr_list

def game_paul_turn (curr_list : List ℕ) (n : ℕ) : List ℕ :=
  let divisors := (curr_list.filter (λ m => m ∣ n))
  List.foldl List.erase curr_list divisors

def carolyn_removed_numbers : List ℕ := [3, 9, 7]

theorem sum_carolyn_removed_numbers :
  let curr_list := initial_list
  let curr_list1 := game_carolyn_turn curr_list 3
  let curr_list2 := game_paul_turn curr_list1 3
  let curr_list3 := game_carolyn_turn curr_list2 9
  let curr_list4 := game_paul_turn curr_list3 9
  let curr_list5 := game_carolyn_turn curr_list4 7
  let curr_list6 := game_paul_turn curr_list5 7
  curr_list5 = curr_list6 →
  sum carolyn_removed_numbers = 19 :=
by
  intros
  simp [carolyn_removed_numbers]
  sorry

end sum_carolyn_removed_numbers_l724_724291


namespace complex_fraction_evaluation_l724_724706

theorem complex_fraction_evaluation : (5 * complex.I) / (1 + 2 * complex.I) = 2 + complex.I :=
by
  sorry

end complex_fraction_evaluation_l724_724706


namespace proof_solution_l724_724464

variable (U : Set ℝ) (A : Set ℝ) (C_U_A : Set ℝ)
variables (a b : ℝ)

noncomputable def proof_problem : Prop :=
  (U = Set.univ) →
  (A = {x | a ≤ x ∧ x ≤ b}) →
  (C_U_A = {x | x > 4 ∨ x < 3}) →
  A = {x | 3 ≤ x ∧ x ≤ 4} ∧ a = 3 ∧ b = 4

theorem proof_solution : proof_problem U A C_U_A a b :=
by
  intro hU hA hCUA
  have hA_eq : A = {x | 3 ≤ x ∧ x ≤ 4} :=
    by { sorry }
  have ha : a = 3 :=
    by { sorry }
  have hb : b = 4 :=
    by { sorry }
  exact ⟨hA_eq, ha, hb⟩

end proof_solution_l724_724464


namespace sheets_torn_out_l724_724891

-- Define the conditions as given in the problem
def first_torn_page : Nat := 185
def last_torn_page : Nat := 518
def pages_per_sheet : Nat := 2

-- Calculate the total number of pages torn out
def total_pages_torn_out : Nat :=
  last_torn_page - first_torn_page + 1

-- Calculate the number of sheets torn out
def number_of_sheets_torn_out : Nat :=
  total_pages_torn_out / pages_per_sheet

-- Prove that the number of sheets torn out is 167
theorem sheets_torn_out :
  number_of_sheets_torn_out = 167 :=
by
  unfold number_of_sheets_torn_out total_pages_torn_out
  rw [Nat.sub_add_cancel (Nat.le_of_lt (Nat.lt_of_le_of_ne
    (Nat.le_add_left _ _) (Nat.ne_of_lt (Nat.lt_add_one 184))))]
  rw [Nat.div_eq_of_lt (Nat.lt.base 333)] 
  sorry -- proof steps are omitted

end sheets_torn_out_l724_724891


namespace exists_pairwise_coprime_composite_AP_l724_724697

theorem exists_pairwise_coprime_composite_AP (n : ℕ) (h : n ≥ 2) :
  ∃ a : ℕ → ℕ, 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → a k = k * nat.factorial n + nat.factorial (n+1)! + 1) ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (∃ m : ℕ, m ≥ 2 ∧ m * m ≤ a k)) ∧ 
  (∀ i j : ℕ, (1 ≤ i ∧ i ≤ n) ∧ (1 ≤ j ∧ j ≤ n) ∧ (i ≠ j) → nat.coprime (a i) (a j)) :=
sorry

end exists_pairwise_coprime_composite_AP_l724_724697


namespace diaz_age_twenty_years_later_l724_724599

theorem diaz_age_twenty_years_later (D S : ℕ) (h₁ : 10 * D - 40 = 10 * S + 20) (h₂ : S = 30) : D + 20 = 56 :=
sorry

end diaz_age_twenty_years_later_l724_724599


namespace unique_solution_a_inequality_k_2019_l724_724722

-- Define the function f
def f (x : ℝ) (a : ℝ) (k : ℕ) : ℝ := x^2 - 2 * a * (-1 : ℝ)^k * real.log x

-- Conditions
variables {a : ℝ} {k : ℕ} (h_a : a > 0)

-- Goal 1: Proving a = 1/2 when k = 2018 and f(x) = 2ax has a unique solution
theorem unique_solution_a (ha : a = 1/2) (hk : k = 2018) : 
  (∃! x, f x a k = 2 * a * x) :=
sorry

-- Goal 2: Proving the inequality for k = 2019
theorem inequality_k_2019 (hk : k = 2019) : 
  ∀ x ∈ (0 : ℝ, +∞), f x a k - x^2 > 2 * a * ((1 / real.exp x) - (2 / (real.exp 1 * x))) :=
sorry

end unique_solution_a_inequality_k_2019_l724_724722


namespace three_digit_multiples_of_36_eq_25_l724_724740

-- Definition: A three-digit number is between 100 and 999
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- Definition: A number is a multiple of both 4 and 9 if and only if it's a multiple of 36
def is_multiple_of_36 (n : ℕ) : Prop := n % 36 = 0

-- Definition: Count of three-digit integers that are multiples of 36
def count_multiples_of_36 : ℕ :=
  (999 / 36) - (100 / 36) + 1

-- Theorem: There are 25 three-digit integers that are multiples of 36
theorem three_digit_multiples_of_36_eq_25 : count_multiples_of_36 = 25 := by
  sorry

end three_digit_multiples_of_36_eq_25_l724_724740


namespace ms_warren_total_distance_l724_724485

-- Conditions as definitions
def running_speed : ℝ := 6 -- mph
def running_time : ℝ := 20 / 60 -- hours

def walking_speed : ℝ := 2 -- mph
def walking_time : ℝ := 30 / 60 -- hours

-- Total distance calculation
def distance_ran : ℝ := running_speed * running_time
def distance_walked : ℝ := walking_speed * walking_time
def total_distance : ℝ := distance_ran + distance_walked

-- Statement to be proved
theorem ms_warren_total_distance : total_distance = 3 := by
  sorry

end ms_warren_total_distance_l724_724485


namespace molecular_weight_of_dichromate_l724_724566

theorem molecular_weight_of_dichromate :
  let Cr_atomic_mass := 52.00
  let O_atomic_mass := 16.00
  let dichromate_molecular_weight := 2 * Cr_atomic_mass + 7 * O_atomic_mass
  let moles := 9
  dichromate_molecular_weight * moles = 1944.00 :=
by
  let Cr_atomic_mass := 52.00
  let O_atomic_mass := 16.00
  let dichromate_molecular_weight := 2 * Cr_atomic_mass + 7 * O_atomic_mass
  let moles := 9
  calc
    dichromate_molecular_weight * moles
      = (2 * Cr_atomic_mass + 7 * O_atomic_mass) * moles : by sorry
  ... = 1944.00 : by sorry

end molecular_weight_of_dichromate_l724_724566


namespace three_point_sixty_eight_as_fraction_l724_724975

theorem three_point_sixty_eight_as_fraction : 3.68 = 92 / 25 := 
by 
  sorry

end three_point_sixty_eight_as_fraction_l724_724975


namespace triangle_parallel_lines_sum_floor_value_l724_724729

theorem triangle_parallel_lines_sum_floor_value :
  let S := area ABC
  let n := 2009
  let each_area := S / n
  let h := height_from_vertex_C ABC
  let AiBi := λ i, base_length_of_subtriangle i ABC each_area h in
  floor (∑ i in (finset.range (2008)).filter (λ i, i > 0), (AiBi 1) / (2 * (AiBi (i+1)))) = 29985 :=
begin
  sorry
end

end triangle_parallel_lines_sum_floor_value_l724_724729


namespace coordinates_of_A_l724_724016

-- defining the hyperbola with a > 0
noncomputable def hyperbola (x y a : ℝ) : Prop := x ^ 2 / a ^ 2 - y ^ 2 / 4 = 1

-- defining the coordinates of points B
def B : ℝ × ℝ := (0, 2)

-- conditions for the foci and distances
def F1 (a : ℝ) : ℝ × ℝ := (-real.sqrt(a ^ 2 + 4), 0)
def F2 (a : ℝ) : ℝ × ℝ := (real.sqrt(a ^ 2 + 4), 0)
noncomputable def A (a : ℝ) : ℝ × ℝ := (a, 0)
noncomputable def dist (p q : ℝ × ℝ) : ℝ :=
  real.sqrt((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- main theorem
theorem coordinates_of_A (a : ℝ) (h1 : a > 0)
  (h2 : F1 a.1 = -real.sqrt(a ^ 2 + 4) ∧ F1 a.2 = 0)
  (h3 : F2 a.1 = real.sqrt(a ^ 2 + 4) ∧ F2 a.2 = 0)
  (h4 : dist B (F2 a) = real.sqrt(a ^ 2 + 8))
  (h5 : 2 * real.sqrt(a ^ 2 + 4) * (2 / real.sqrt(a ^ 2 + 4)) = real.sqrt(a ^ 2 + 8)) :
  A (2 * real.sqrt(2)) = (2 * real.sqrt(2), 0) :=
by
  sorry

end coordinates_of_A_l724_724016


namespace parabola_focus_directrix_l724_724655

theorem parabola_focus_directrix :
  (∃ a : ℝ, ∀ x y : ℝ, y = - (1 / 6) * x^2 → (focus x y = (0, -a / 4)) ∧ (directrix x y = a)) :=
by
  sorry

end parabola_focus_directrix_l724_724655


namespace problem_m_n_sum_l724_724815

-- Assume necessary conditions
variables {A B C H R S : Type}
variables {A B C H R S : Point}

-- Given values
def AB : ℝ := 2005
def AC : ℝ := 2003
def BC : ℝ := 2002

-- Distance between the points R and S
def RS : ℝ := (200:ℝ) / 401

-- Mathematical proof problem statement in Lean 4
theorem problem_m_n_sum :
  let m := 200 in let n := 401 in
  m + n = 601 :=
by
  let m := 200
  let n := 401
  exact rfl

end problem_m_n_sum_l724_724815


namespace probability_same_color_correct_l724_724744

-- conditions
def sides := ["maroon", "teal", "cyan", "sparkly"]
def die : Type := {v // v ∈ sides}
def maroon_count := 6
def teal_count := 9
def cyan_count := 10
def sparkly_count := 5
def total_sides := 30

-- calculate probabilities
def prob (count : ℕ) : ℚ := (count ^ 2) / (total_sides ^ 2)
def prob_same_color : ℚ :=
  prob maroon_count +
  prob teal_count +
  prob cyan_count +
  prob sparkly_count

-- statement
theorem probability_same_color_correct :
  prob_same_color = 121 / 450 :=
sorry

end probability_same_color_correct_l724_724744


namespace mathematical_proof_l724_724339

noncomputable def proof_problem (x y : ℝ) (hx_pos : y > 0) (hxy_gt2 : x + y > 2) : Prop :=
  (1 + x) / y < 2 ∨ (1 + y) / x < 2

theorem mathematical_proof (x y : ℝ) (hx_pos : y > 0) (hxy_gt2 : x + y > 2) : proof_problem x y hx_pos hxy_gt2 :=
by {
  sorry
}

end mathematical_proof_l724_724339


namespace min_c_for_expression_not_min_abs_c_for_expression_l724_724659

theorem min_c_for_expression :
  ∀ c : ℝ,
  (c - 3)^2 + (c - 4)^2 + (c - 8)^2 ≥ (5 - 3)^2 + (5 - 4)^2 + (5 - 8)^2 := 
by sorry

theorem not_min_abs_c_for_expression :
  ∃ c : ℝ, |c - 3| + |c - 4| + |c - 8| < |5 - 3| + |5 - 4| + |5 - 8| := 
by sorry

end min_c_for_expression_not_min_abs_c_for_expression_l724_724659


namespace perimeter_of_triangle_ADE_l724_724362

noncomputable def ellipse_perimeter (a b : ℝ) (h : a > b) (e : ℝ) (he : e = 1/2) (h_ellipse : ∀ (x y : ℝ), 
                            x^2 / a^2 + y^2 / b^2 = 1) : ℝ :=
13 -- we assert that the perimeter is 13

theorem perimeter_of_triangle_ADE 
  (a b : ℝ) (h : a > b) (e : ℝ) (he : e = 1/2) 
  (C_eq : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1) 
  (upper_vertex_A : ℝ × ℝ)
  (focus_F1 : ℝ × ℝ)
  (focus_F2 : ℝ × ℝ)
  (line_through_F1_perpendicular_to_AF2 : ∀ x y, y = (√3 / 3) * (x + focus_F1.1))
  (points_D_E_on_ellipse : ∃ D E : ℝ × ℝ, line_through_F1_perpendicular_to_AF2 D.1 D.2 = true ∧
    line_through_F1_perpendicular_to_AF2 E.1 E.2 = true ∧ 
    (dist D E = 6)) :
  ∃ perimeter : ℝ, perimeter = ellipse_perimeter a b h e he C_eq :=
sorry

end perimeter_of_triangle_ADE_l724_724362


namespace max_cubes_with_one_red_face_l724_724930

/-- 
  Define a structure for RectangularPrism with edge lengths and a painted face count indicator.
-/
structure RectangularPrism :=
(edge_length1 : ℕ)
(edge_length2 : ℕ)
(edge_length3 : ℕ)
(faces_painted : ℕ) -- Number of painted faces

/-- 
  Define a function to calculate small cubes with exactly one red face.
-/
def small_cubes_with_one_red_face (prism : RectangularPrism) : ℕ :=
  match prism.faces_painted with
  | 1 => prism.edge_length2 * prism.edge_length3
  | 2 => 4 * (prism.edge_length2 + prism.edge_length2 + prism.edge_length1 - 4)
  | 3 => 4 * (prism.edge_length2 + prism.edge_length2 + prism.edge_length1 - 4)
  | 4 => 4 * (prism.edge_length2 + prism.edge_length2 + prism.edge_length3 + prism.edge_length3 - 2 * prism.edge_length1)
  | 5 => (prism.edge_length1 - 2) * (prism.edge_length3 - 2) + (prism.edge_length2 - 1) * (2 * prism.edge_length2 + 2 * prism.edge_length3 - 2 * prism.edge_length1)
  | 6 => 2 * ((prism.edge_length1 - 2) * (prism.edge_length3 - 2) + (prism.edge_length3 - 2) * (prism.edge_length2 - 2) + (prism.edge_length2 - 2) * (prism.edge_length1 - 2))
  | _ => 0  -- Assuming 0 for cases where none or more faces are not in scope of conditions.
  end

/-- 
  The maximum number of small cubes with exactly one red face for the given conditions.
-/
theorem max_cubes_with_one_red_face (p1 p2 p3 p4 p5 p6 : RectangularPrism)
  (h1 : p1.faces_painted = 1) (h2 : p2.faces_painted = 2) (h3 : p3.faces_painted = 3)
  (h4 : p4.faces_painted = 4) (h5 : p5.faces_painted = 5) (h6 : p6.faces_painted = 6) :
  small_cubes_with_one_red_face p1 + small_cubes_with_one_red_face p2 + small_cubes_with_one_red_face p3 +
  small_cubes_with_one_red_face p4 + small_cubes_with_one_red_face p5 + small_cubes_with_one_red_face p6 = 20 :=
sorry

end max_cubes_with_one_red_face_l724_724930


namespace intersection_complement_A_with_B_C_subset_A_iff_a_in_range_l724_724409

noncomputable def U : Set ℝ := Set.univ
noncomputable def A : Set ℝ := {x | 1 < 2 * x - 1 ∧ 2 * x - 1 < 5}
noncomputable def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = (1 / 2) ^ x ∧ x ≥ -2}
noncomputable def C (a : ℝ) : Set ℝ := {x | a - 1 < x - a ∧ x - a < 1}

theorem intersection_complement_A_with_B :
  (U \ A) ∩ B = {x : ℝ | (0 < x ∧ x ≤ 1/2) ∨ (5/2 ≤ x ∧ x ≤ 4)} := sorry

theorem C_subset_A_iff_a_in_range (a : ℝ) :
  C a ⊆ A ↔ 1 ≤ a := sorry

end intersection_complement_A_with_B_C_subset_A_iff_a_in_range_l724_724409


namespace sum_even_indices_complex_angles_l724_724928

theorem sum_even_indices_complex_angles 
  (n : ℕ)
  (z : ℕ → ℂ)
  (θ : ℕ → ℝ)
  (h_eq : ∀ m, z m ^ 36 - z m ^ 12 - 1 = 0)
  (h_mod : ∀ m, complex.abs (z m) = 1)
  (h_form : ∀ m, z m = complex.of_real (real.cos (θ m)) + complex.I * complex.of_real (real.sin (θ m)))
  (h_range : ∀ i j, i < j → θ i < θ j)
  (h_theta_range : ∀ m, 0 ≤ θ m ∧ θ m < 360) :
  ∑ i in finset.range (2 * n + 1), if i % 2 = 1 then θ i else 0 = 1200 := sorry

end sum_even_indices_complex_angles_l724_724928


namespace triangles_congruent_l724_724178

open EuclideanGeometry

-- Definitions of points and conditions
variables {A B C D E : Point}

-- Conditions in the given problem
def AB_eq_BC : LineSegment A B = LineSegment B C := sorry
def AB_perp_BD : Perpendicular (Line A B) (Line B D) := sorry
def BE_perp_BC : Perpendicular (Line B E) (Line B C) := sorry

-- Statement we need to prove
theorem triangles_congruent :
  Triangle A B E ≅ Triangle B C D :=
by
  sorry

end triangles_congruent_l724_724178


namespace distance_between_poles_correct_l724_724263
noncomputable def distance_between_poles (L W : ℝ) (num_poles : ℕ) : ℝ := do
  let perimeter := 2 * (L + W)
  perimeter / (num_poles - 1)

theorem distance_between_poles_correct :
  distance_between_poles 90 50 70 ≈ 4.06 :=
by
  sorry

end distance_between_poles_correct_l724_724263


namespace systematic_sampling_5_out_of_50_l724_724325

def is_systematic_sampling (N n k : ℕ) (s : Fin n → Fin N) : Prop :=
  ∀ i : Fin n, s ⟨i.1 + 1, by linarith [i.2]⟩.1 - s i.1 = k

def sequence_5_missiles : Fin 5 → Fin 50
| ⟨0, _⟩ => ⟨2, by norm_num⟩
| ⟨1, _⟩ => ⟨12, by norm_num⟩
| ⟨2, _⟩ => ⟨22, by norm_num⟩
| ⟨3, _⟩ => ⟨32, by norm_num⟩
| ⟨4, _⟩ => ⟨42, by norm_num⟩

theorem systematic_sampling_5_out_of_50 : 
  ∃ s : Fin 5 → Fin 50, 
  is_systematic_sampling 50 5 10 s ∧ 
  (s 0 = 3 ∧ s 1 = 13 ∧ s 2 = 23 ∧ s 3 = 33 ∧ s 4 = 43) := 
sorry

end systematic_sampling_5_out_of_50_l724_724325


namespace pile_limit_exists_l724_724056

noncomputable def log_floor (b x : ℝ) : ℤ :=
  Int.floor (Real.log x / Real.log b)

theorem pile_limit_exists (k : ℝ) (hk : k < 2) : ∃ Nk : ℤ, 
  Nk = 2 * (log_floor (2 / k) 2 + 1) := 
  by
    sorry

end pile_limit_exists_l724_724056


namespace perimeter_triangle_ADA_l724_724355

open Real

noncomputable def eccentricity : ℝ := 1 / 2

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

noncomputable def foci_distance (a b : ℝ) : ℝ :=
  (a^2 - b^2).sqrt

noncomputable def line_passing_through_focus_perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  sorry

noncomputable def distance_de (d e : ℝ) : ℝ := 6

theorem perimeter_triangle_ADA
  (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : foci_distance a b = c)
  (h4 : eccentricity * a = c) (h5 : distance_de 6 6) :
  4 * a = 13 :=
by sorry

end perimeter_triangle_ADA_l724_724355


namespace prod_cos_eq_zero_or_one_l724_724044

theorem prod_cos_eq_zero_or_one (n : ℕ) (h_pos : n > 0) :
  (∏ k in finset.range (2 * n + 1).filter (λ k, k > 0 && k < (2 * n + 1)), 
    (1 + 2 * real.cos (2 * k * real.pi / (2 * n + 1)))) = 
  if (2 * n + 1) % 3 = 0 then 0 else 1 :=
sorry

end prod_cos_eq_zero_or_one_l724_724044


namespace negation_of_cos_proposition_l724_724406

variable (x : ℝ)

theorem negation_of_cos_proposition (h : ∀ x : ℝ, Real.cos x ≤ 1) : ∃ x₀ : ℝ, Real.cos x₀ > 1 :=
sorry

end negation_of_cos_proposition_l724_724406


namespace students_above_130_l724_724609

noncomputable def number_of_students_scoring_at_least (mean std_dev n : ℝ) (total_students : ℕ) : ℝ :=
  let prob := 0.5 * (1 - 0.75)
  prob * total_students

theorem students_above_130 (mean : ℝ) (σ : ℝ) (total_students : ℕ) :
  mean = 110 → 
  σ = σ → 
  total_students = 800 →
  number_of_students_scoring_at_least 110 σ 90 800 = 100 :=
by {
  intros,
  unfold number_of_students_scoring_at_least,
  rw [H, H_2],
  norm_num,
  sorry
}

end students_above_130_l724_724609


namespace simplify_fraction_l724_724106

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : 
  ( (x^2 + 1) / (x - 1) - (2*x) / (x - 1) ) = x - 1 :=
by
  -- Your proof steps would go here.
  sorry

end simplify_fraction_l724_724106


namespace old_clock_slow_l724_724129

theorem old_clock_slow (minute_hand_speed hour_hand_speed : ℝ)
  (overlap_old_clock duration_standard : ℝ) :
  (minute_hand_speed = 6) →
  (hour_hand_speed = 0.5) →
  (overlap_old_clock = 66) →
  (duration_standard = 24 * 60) →
  let relative_speed := minute_hand_speed - hour_hand_speed in
  let time_to_coincide := 360 / relative_speed in
  let intervals_per_day := duration_standard / time_to_coincide in
  let new_duration := intervals_per_day * overlap_old_clock in
  new_duration - duration_standard = 12 :=
sorry

end old_clock_slow_l724_724129


namespace sequence_bounded_l724_724030

def sequence {n : ℕ} (p q : ℕ) : ℕ → ℕ
| 0 => p
| 1 => q
| (k + 2) => Nat.min_fac (sequence k + sequence (k + 1) + 2016)

theorem sequence_bounded (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  ∃ M : ℝ, ∀ n : ℕ, (sequence p q n).toReal < M :=
by
  sorry

end sequence_bounded_l724_724030


namespace necessary_condition_of_and_is_or_l724_724679

variable (p q : Prop)

theorem necessary_condition_of_and_is_or (hpq : p ∧ q) : p ∨ q :=
by {
    sorry
}

end necessary_condition_of_and_is_or_l724_724679


namespace system_of_equations_solution_l724_724541

theorem system_of_equations_solution :
  ∃ (x y z : ℝ), 2 * x + y = 3 ∧ 3 * x - z = 7 ∧ x - y + 3 * z = 0 ∧ x = 2 ∧ y = -1 ∧ z = -1 :=
by
  exists 2
  exists -1
  exists -1
  split
  { norm_num }
  split
  { norm_num }
  split
  { norm_num }
  split
  { rfl }
  split
  { rfl }
  { rfl }

end system_of_equations_solution_l724_724541


namespace increasing_cubic_function_l724_724761

theorem increasing_cubic_function (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = a * x^3 + x) :
  (∀ x, (∂ f) x ≥ 0) → a > 0 :=
by
  sorry

end increasing_cubic_function_l724_724761


namespace john_average_speed_l724_724026

theorem john_average_speed 
  (distance : ℝ) 
  (start_time end_time : ℝ) :
  distance / (end_time - start_time) = 32.31 :=
by 
  assume 
    (h_distance : distance = 210)
    (h_start_time : start_time = (8 + 15/60))
    (h_end_time : end_time = (14 + 45/60))
  have h_time : end_time - start_time = 6.5 := sorry
  
  calc
    distance / (end_time - start_time) = 210 / 6.5 : by rw [h_distance, h_time]
    ... = 32.31 : sorry

end john_average_speed_l724_724026


namespace powers_of_k_l724_724036

theorem powers_of_k (k : ℕ) (hn : k > 1) 
  (a : ℕ → ℕ) (h_init1 : a 1 = 1) (h_init2 : a 2 = k)
  (h_recur : ∀ n > 1, a (n + 1) = (k + 1) * a n - a (n - 1))
  (m : ℕ) (n : ℕ) (h_power : a n = k ^ m) :
  n = 1 ∨ n = 2 :=
begin
  sorry
end

end powers_of_k_l724_724036


namespace gcd_102_238_is_34_l724_724201

noncomputable def gcd_102_238 : ℕ :=
  Nat.gcd 102 238

theorem gcd_102_238_is_34 : gcd_102_238 = 34 := by
  -- Conditions based on the Euclidean algorithm
  have h1 : 238 = 2 * 102 + 34 := by norm_num
  have h2 : 102 = 3 * 34 := by norm_num
  have h3 : Nat.gcd 102 34 = 34 := by
    rw [Nat.gcd, Nat.gcd_rec]
    exact Nat.gcd_eq_left h2

  -- Conclusion
  show gcd_102_238 = 34 from
    calc gcd_102_238 = Nat.gcd 102 238 : rfl
                  ... = Nat.gcd 34 102 : Nat.gcd_comm 102 34
                  ... = Nat.gcd 34 (102 % 34) : by rw [Nat.gcd_rec]
                  ... = Nat.gcd 34 34 : by rw [Nat.mod_eq_of_lt (by norm_num : 34 < 102)]
                  ... = 34 : Nat.gcd_self 34

end gcd_102_238_is_34_l724_724201


namespace perimeter_of_triangle_ADE_l724_724370

theorem perimeter_of_triangle_ADE
  (a b : ℝ) (F1 F2 A : ℝ × ℝ) (D E : ℝ × ℝ) 
  (h_ellipse : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1)
  (h_a_gt_b : a > b)
  (h_b_gt_0 : b > 0)
  (h_eccentricity : ∃ c, c / a = 1 / 2 ∧ a^2 - b^2 = c^2)
  (h_F1_F2 : ∀ F1 F2, distance F1 (0, 0) = distance F2 (0, 0) ∧ F1 ≠ F2 ∧ 
                       ∀ P : ℝ × ℝ, (distance P F1 + distance P F2 = 2 * a) ↔ (x : ℝ)(y : ℝ) (h_ellipse x y))
  (h_line_DE : ∃ k, ∃ c, ∀ x F1 A, (2 * a * x/(sqrt k^2 + 1)) = |DE|
  (h_length_DE : |DE| = 6)
  (h_A_vertex : A = (0, b))
  : ∃ perim : ℝ, perim = 13 :=
sorry

end perimeter_of_triangle_ADE_l724_724370


namespace harry_books_l724_724736

theorem harry_books : ∀ (H : ℝ), 
  (H + 2 * H + H / 2 = 175) → 
  H = 50 :=
by
  intros H h_sum
  sorry

end harry_books_l724_724736


namespace abigail_loss_l724_724628

theorem abigail_loss :
  let initial_amount := 50
  let rate1 := 0.85
  let spent1 := 13
  let rate2 := 0.82
  let spent2 := 7.2
  let remaining_amount := 16
  let usd_spent1 := spent1 / rate1
  let usd_spent2 := spent2 / rate2
  let total_spent_usd := usd_spent1 + usd_spent2
  let current_amount := remaining_amount + total_spent_usd
  let loss := initial_amount - current_amount
  loss ≈ 9.93 :=
by {
  sorry
}

end abigail_loss_l724_724628


namespace existence_of_solution_l724_724470

noncomputable def is_nonempty {α : Type*} (s : set α) : Prop :=
  ∃ x, x ∈ s

noncomputable def is_proper_subset {α : Type*} (s : set α) : Prop :=
  s ≠ set.univ ∧ s.nonempty

noncomputable def X_mul_X_add_c {α : Type*} [has_mul α] [has_add α] (X : set α) (c : α) : set α :=
  {z | ∃ x y ∈ X, z = x * y + c}

noncomputable def is_solution (c : ℝ) :=
  ∃ X : set ℝ, is_proper_subset X ∧ X_mul_X_add_c X c = X

theorem existence_of_solution :
  ∀ c : ℝ, is_solution c :=
  by
    intros c
    sorry

end existence_of_solution_l724_724470


namespace cell_with_two_red_two_blue_l724_724491

theorem cell_with_two_red_two_blue (n : ℕ) (h : n = 21) :
  ∃ (i j : ℕ), 
  i < n ∧ j < n ∧
  (cell_has_two_red_two_blue i j) :=
begin
  -- Conditions
  have top_red : ∀ (j : ℕ), j < n → vertex (0, j) = red, from sorry,
  have right_red : ∀ (i : ℕ), i < n - 1 → vertex (i, n-1) = red, from sorry,
  have other_edges_blue : ∀ (v : ℕ × ℕ), 
                             (v.1 = n ∧ v.2 < n-1 ∨ v.2 = 0 ∧ v.1 ≠ 0) → 
                             vertex v = blue, from sorry,
  exact sorry
end

end cell_with_two_red_two_blue_l724_724491


namespace snake_earnings_l724_724794

noncomputable def number_vipers := 3
noncomputable def number_cobras := 2
noncomputable def number_pythons := 1
noncomputable def number_anacondas := 1

noncomputable def eggs_per_viper := 3
noncomputable def eggs_per_cobra := 2
noncomputable def eggs_per_python := 4
noncomputable def eggs_per_anaconda := 5

noncomputable def price_per_baby_viper := 300
noncomputable def price_per_baby_cobra := 250
noncomputable def price_per_baby_python := 450
noncomputable def price_per_baby_anaconda := 500

noncomputable def discount_viper := 0.1
noncomputable def discount_cobra := 0.05
noncomputable def discount_python := 0.075
noncomputable def discount_anaconda := 0.12

theorem snake_earnings : 
  let total_earnings := 
    (number_vipers * eggs_per_viper * (price_per_baby_viper * (1 - discount_viper))) +
    (number_cobras * eggs_per_cobra * (price_per_baby_cobra * (1 - discount_cobra))) +
    (number_pythons * eggs_per_python * (price_per_baby_python * (1 - discount_python))) +
    (number_anacondas * eggs_per_anaconda * (price_per_baby_anaconda * (1 - discount_anaconda)))
  in total_earnings = 7245 := by
  sorry

end snake_earnings_l724_724794


namespace digit_last_digit_of_product_l724_724563

theorem digit_last_digit_of_product (k : ℕ) :
  ((3 ^ 65) * (6 ^ k) * (7 ^ 71)) % 10 = 4 :=
by
  have h3 : (3 ^ 65) % 10 = 3 := sorry
  have h6 : (6 ^ k) % 10 = 6 := sorry
  have h7 : (7 ^ 71) % 10 = 3 := sorry
  calc
    ((3 ^ 65) * (6 ^ k) * (7 ^ 71)) % 10
        = (3 * 6 * 3) % 10 : by rw [h3, h6, h7]
    ... = 54 % 10 : by norm_num
    ... = 4 : by norm_num

end digit_last_digit_of_product_l724_724563


namespace find_B_squared_l724_724306

noncomputable def g (x : ℝ) : ℝ :=
  real.sqrt 31 + 105 / x

theorem find_B_squared :
  let g_iter := (g ∘ g ∘ g ∘ g ∘ g)
  ∀ x, g_iter x = x → 
  let B := |(real.sqrt 31 + real.sqrt 451) / 2| + |-(real.sqrt 31 - real.sqrt 451) / 2| 
  B^2 = 451 :=
by
  sorry

end find_B_squared_l724_724306


namespace triangle_ADE_perimeter_l724_724376

noncomputable def ellipse_perimeter (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) (e : ℝ) (h₃ : e = (1 / 2)) 
(F₁ F₂ : ℝ × ℝ) (h₄ : F₁ ≠ F₂) (D E : ℝ × ℝ) (h₅ : |D - E| = 6) : ℝ :=
  let c := (sqrt (a ^ 2 - b ^ 2)) in
  let A := (0, b) in
  let AD := sqrt ((fst D) ^ 2 + (snd D - b) ^ 2) in
  let AE := sqrt ((fst E) ^ 2 + (snd E - b) ^ 2) in
  AD + AE + |D - E|

theorem triangle_ADE_perimeter (a b : ℝ) (h₁ : a > b > 0) (e : ℝ) (h₂ : e = (1 / 2))
(F₁ F₂ : ℝ × ℝ) (h₃ : F₁ ≠ F₂)
(D E : ℝ × ℝ) (h₄ : |D - E| = 6) : 
  ellipse_perimeter a b (and.left h₁) (and.right h₁) e h₂ F₁ F₂ h₃ D E h₄ = 19 :=
sorry

end triangle_ADE_perimeter_l724_724376


namespace daisy_dog_toys_l724_724490

theorem daisy_dog_toys (X : ℕ) (lost_toys : ℕ) (total_toys_after_found : ℕ) : 
    (X - lost_toys + (3 + 3) - lost_toys + 5 = total_toys_after_found) → total_toys_after_found = 13 → X = 5 :=
by
  intros h1 h2
  sorry

end daisy_dog_toys_l724_724490


namespace sufficient_not_necessary_condition_l724_724752

theorem sufficient_not_necessary_condition (a b : ℝ) (h1 : a > 1) (h2 : b > 2) : a + b > 3 :=
by
  sorry

end sufficient_not_necessary_condition_l724_724752


namespace sin_sum_identity_l724_724689

theorem sin_sum_identity
  (α β : Real)
  (h : sin (α - β) * cos α - cos (α - β) * sin α = 1 / 4) :
  sin ((3 * π / 2) + 2 * β) = - 7 / 8 :=
  sorry

end sin_sum_identity_l724_724689


namespace number_of_possible_concatenated_integers_l724_724177

theorem number_of_possible_concatenated_integers : 
  (finset.card (finset.image (λ (t : ℕ × ℕ × ℕ), t.1 * 10000 + t.2 * 100 + t.3) 
  (finset.product (finset.product (finset.range 100).filter (λ x, x > 0) 
  (finset.range 100).filter (λ y, y > 0)) 
  (finset.range 100).filter (λ z, z > 0)))) 
  = 825957 :=
sorry

end number_of_possible_concatenated_integers_l724_724177


namespace perimeter_triangle_ADA_l724_724357

open Real

noncomputable def eccentricity : ℝ := 1 / 2

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

noncomputable def foci_distance (a b : ℝ) : ℝ :=
  (a^2 - b^2).sqrt

noncomputable def line_passing_through_focus_perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  sorry

noncomputable def distance_de (d e : ℝ) : ℝ := 6

theorem perimeter_triangle_ADA
  (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : foci_distance a b = c)
  (h4 : eccentricity * a = c) (h5 : distance_de 6 6) :
  4 * a = 13 :=
by sorry

end perimeter_triangle_ADA_l724_724357


namespace root_exists_in_interval_l724_724993

def f (x : ℝ) : ℝ := 2 * x + x - 2

theorem root_exists_in_interval :
  (∃ x ∈ (Set.Ioo 0 1), f x = 0) :=
by
  sorry

end root_exists_in_interval_l724_724993


namespace graph_division_l724_724530

theorem graph_division (G : Graph)
    (hG : ¬ three_colorable G) :
    ∃ (M N : Graph), ¬ two_colorable M ∧ ¬ one_colorable N :=
by
    sorry

end graph_division_l724_724530


namespace three_digit_numbers_with_6_or_8_l724_724741

theorem three_digit_numbers_with_6_or_8 : 
  let total_three_digit_numbers := 900 in
  let no_6_or_8_hundreds := 7 in
  let no_6_or_8_tens_ones := 8 in
  let no_6_or_8_digits := no_6_or_8_hundreds * no_6_or_8_tens_ones * no_6_or_8_tens_ones in
  total_three_digit_numbers - no_6_or_8_digits = 452 :=
by
  let total_three_digit_numbers := 900
  let no_6_or_8_hundreds := 7
  let no_6_or_8_tens_ones := 8
  let no_6_or_8_digits := no_6_or_8_hundreds * no_6_or_8_tens_ones * no_6_or_8_tens_ones
  exact total_three_digit_numbers - no_6_or_8_digits = 452

end three_digit_numbers_with_6_or_8_l724_724741


namespace range_of_a_l724_724858

def f (x : ℝ) : ℝ :=
if x ≥ 0 then (1/2) * x - 1 else (1/x)

theorem range_of_a (a : ℝ) : f a > 1 ↔ a > 4 := by
  sorry

end range_of_a_l724_724858


namespace joe_final_expenditure_in_euros_l724_724025

noncomputable def final_cost_in_euros : ℝ :=
  let orange_cost := 3 * 4.50
  let juice_cost := 7 * 0.50
  let honey_cost := (2 * 5) -- two-for-one special
  let plant_cost := 2 * 18 -- 2 plants for $18, so 4 plants * 1 pair = 2 pairs
  
  let oranges_and_juices_cost := orange_cost + juice_cost
  let oranges_and_juices_discount := oranges_and_juices_cost * 0.10
  let oranges_and_juices_cost_after_discount := oranges_and_juices_cost - oranges_and_juices_discount
  
  let honey_discount := honey_cost * 0.05
  let honey_cost_after_discount := honey_cost - honey_discount
  
  let total_cost_before_tax := oranges_and_juices_cost_after_discount + honey_cost_after_discount + plant_cost
  let sales_tax := total_cost_before_tax * 0.08
  let final_cost_in_dollars := total_cost_before_tax + sales_tax
  
  final_cost_in_dollars * 0.85 -- convert to Euros

theorem joe_final_expenditure_in_euros : final_cost_in_euros ≈ 55.81 :=
begin
  simp [final_cost_in_euros],
  sorry
end

end joe_final_expenditure_in_euros_l724_724025


namespace exists_max_pile_division_l724_724049

theorem exists_max_pile_division (k : ℝ) (hk : k < 2) : 
  ∃ (N_k : ℕ), ∀ (A : Multiset ℝ) (m : ℝ), (∀ a ∈ A, a < 2 * m) → 
    ¬(∃ B : Multiset ℝ, B.card > N_k ∧ (∀ b ∈ B, b ∈ A ∧ b < 2 * m)) :=
sorry

end exists_max_pile_division_l724_724049


namespace speed_of_first_train_l724_724581

-- Definitions based on conditions
def ratio (a b : ℕ) : Prop := a / b = 7 / 8

def speed_of_second_train : ℕ := 100 -- 100 km/h as calculated from 400 km / 4 hours

-- Main proof problem
theorem speed_of_first_train (S1 : ℕ) (h : ratio S1 speed_of_second_train) : S1 = 87.5 := by
  sorry

end speed_of_first_train_l724_724581


namespace maximum_value_of_f_l724_724341

noncomputable def f (a x : ℝ) : ℝ := 1 - (a * x^2 / real.exp x)

theorem maximum_value_of_f (a : ℝ) : (∃ x : ℝ, f a x = 5) ↔ a = -real.exp 2 := sorry

end maximum_value_of_f_l724_724341


namespace sum_of_s_of_r_values_l724_724294

def r (x : ℝ) : ℝ := |x| + 1

def s (x : ℝ) : ℝ := -2 * |x|

def s_of_r (x : ℝ) : ℝ := s (r x)

theorem sum_of_s_of_r_values :
  (s_of_r (-5) + s_of_r (-4) + s_of_r (-3) + s_of_r (-2) + s_of_r (-1) +
   s_of_r 0 + s_of_r 1 + s_of_r 2 + s_of_r 3 + s_of_r 4 + s_of_r 5) = -62 := 
begin
  sorry
end

end sum_of_s_of_r_values_l724_724294


namespace decimal_to_fraction_simplify_l724_724968

theorem decimal_to_fraction_simplify (d : ℚ) (h : d = 3.68) : d = 92 / 25 :=
by
  rw h
  sorry

end decimal_to_fraction_simplify_l724_724968


namespace number_of_real_solutions_l724_724303

noncomputable def f (x : ℝ) : ℝ := ∑ i in Finset.range 101, (i + 2) / (x - (i + 1))

def g (x : ℝ) : ℝ := x - 1

theorem number_of_real_solutions : 
  {x : ℝ | f x = g x}.card = 101 :=
sorry

end number_of_real_solutions_l724_724303


namespace compare_logs_l724_724384

noncomputable def a := Real.log 6 / Real.log 3
noncomputable def b := Real.log 10 / Real.log 5
noncomputable def c := Real.log 14 / Real.log 7

theorem compare_logs : a > b ∧ b > c := by
  -- Proof will be written here, currently placeholder
  sorry

end compare_logs_l724_724384


namespace BX_eq_XC_l724_724827

theorem BX_eq_XC (A B C D X : Point) (θ : Real) 
    [convex_quad ABCD]
    (hθ : θ < 90) 
    (h1 : ∠ABC = θ) (h2 : ∠BCD = θ) 
    (h3 : ∠XAD = 90 - θ) (h4 : ∠XDA = 90 - θ) : 
    dist B X = dist X C :=
by
  sorry

end BX_eq_XC_l724_724827


namespace divisible_by_12_l724_724681

theorem divisible_by_12 (n : ℕ) (hn : n = 6) : (51470 + n) % 12 = 0 := by
  have h₁ : (51476 : ℕ) = 51470 + 6 := rfl
  rw [hn, h₁]
  norm_num
  sorry

end divisible_by_12_l724_724681


namespace find_q_l724_724772

noncomputable def geometric_sequence_common_ratio (a sequence: ℕ → ℝ)
    (S₃ S₆: ℝ)
    (hS₃ : S₃ = (sequence 0 + sequence 1 + sequence 2))
    (hS₆ : S₆ = (sequence 0 + sequence 1 + sequence 2 + sequence 3 + sequence 4 + sequence 5)) : 
    Prop :=
  ∃ q : ℝ, (q ≠ 1) ∧
          (sequence 1 = sequence 0 * q) ∧
          (sequence 2 = sequence 1 * q) ∧
          (S₃ = 4) ∧ (S₆ = 36) ∧ q = 2

theorem find_q (sequence: ℕ→ℝ) (a₀: ℝ):
    geometric_sequence_common_ratio a₀ sequence 4 36
    (by sorry) (by sorry) := 
      2 :=
sorry

end find_q_l724_724772


namespace prove_a_eq_neg2_solve_inequality_for_a_leq0_l724_724404

-- Problem 1: Proving that a = -2 given the solution set of the inequality
theorem prove_a_eq_neg2 (a : ℝ) (h : ∀ x : ℝ, (-1 < x ∧ x < -1/2) ↔ (ax - 1) * (x + 1) > 0) : a = -2 := sorry

-- Problem 2: Solving the inequality (ax-1)(x+1) > 0 for different conditions on a
theorem solve_inequality_for_a_leq0 (a x : ℝ) (h_a_le_0 : a ≤ 0) : 
  (ax - 1) * (x + 1) > 0 ↔ 
    if a < -1 then -1 < x ∧ x < 1/a
    else if a = -1 then false
    else if -1 < a ∧ a < 0 then 1/a < x ∧ x < -1
    else x < -1 := sorry

end prove_a_eq_neg2_solve_inequality_for_a_leq0_l724_724404


namespace total_yards_in_marathons_eq_495_l724_724613

-- Definitions based on problem conditions
def marathon_miles : ℕ := 26
def marathon_yards : ℕ := 385
def yards_in_mile : ℕ := 1760
def marathons_run : ℕ := 15

-- Main proof statement
theorem total_yards_in_marathons_eq_495
  (miles_per_marathon : ℕ := marathon_miles)
  (yards_per_marathon : ℕ := marathon_yards)
  (yards_per_mile : ℕ := yards_in_mile)
  (marathons : ℕ := marathons_run) :
  let total_yards := marathons * yards_per_marathon
  let remaining_yards := total_yards % yards_per_mile
  remaining_yards = 495 :=
by
  sorry

end total_yards_in_marathons_eq_495_l724_724613


namespace area_of_intersection_l724_724945

-- Define the circle centered at (3, 0) with radius 3
def circle1 (x y : ℝ) : Prop := (x - 3) ^ 2 + y ^ 2 = 9

-- Define the circle centered at (0, 3) with radius 3
def circle2 (x y : ℝ) : Prop := x ^ 2 + (y - 3) ^ 2 = 9

-- Defining the theorem to prove the area of intersection of these circles
theorem area_of_intersection : 
  let r := 3 in
  let a := (3, 0) in
  let b := (0, 3) in
  area_intersection (circle1) (circle2) = (9 * π - 18) / 2 := 
sorry

end area_of_intersection_l724_724945


namespace Diaz_age_20_years_from_now_l724_724600

open Nat

theorem Diaz_age_20_years_from_now:
  (∃ (diaz_age : ℕ) (sierra_age : ℕ),
    sierra_age = 30 ∧
    40 + 10 * diaz_age = 20 + 10 * sierra_age ∧
    diaz_age + 20 = 56) :=
begin
  sorry
end

end Diaz_age_20_years_from_now_l724_724600


namespace gcd_102_238_l724_724194

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end gcd_102_238_l724_724194


namespace exist_triangle_with_given_conditions_l724_724202

noncomputable section

open EuclideanGeometry

def construct_triangle (H O : Point) (l : Line) : Triangle :=
  sorry

-- Define the conditions as hypotheses
variable {H O : Point} (l : Line)
variable {ABC : Triangle}

-- Define a function that checks if a given triangle satisfies the required conditions
def satisfies_conditions (ABC : Triangle) : Prop :=
  is_orthocenter H ABC 
  ∧ is_circumcenter O ABC 
  ∧ side_of_triangle_lies_on_line ABC l

-- Statement of the theorem
theorem exist_triangle_with_given_conditions : ∃ ABC, satisfies_conditions ABC :=
  sorry

end exist_triangle_with_given_conditions_l724_724202


namespace direction_vector_equal_l724_724763

theorem direction_vector_equal (a : ℝ) : (∀ x y : ℝ, ax + 2y + 3 = 0 → 2x + ay - 1 = 0) → (a = 2 ∨ a = -2) :=
by
  sorry

end direction_vector_equal_l724_724763


namespace Petya_tore_out_sheets_l724_724887

theorem Petya_tore_out_sheets (n m : ℕ) (h1 : n = 185) (h2 : m = 518)
  (h3 : m.digits = n.digits) : (m - n + 1) / 2 = 167 :=
by
  sorry

end Petya_tore_out_sheets_l724_724887


namespace f_monotonicity_l724_724718

noncomputable def f (x : ℝ) : ℝ := Real.exp (x + 1) - Real.log (x + 2)

theorem f_monotonicity : (∀ (x : ℝ), x ∈ Ioo (-2 : ℝ) (-1 : ℝ) → f' x < 0) ∧ (∀ (x : ℝ), x > -1 → f' x > 0) :=
begin
  sorry
end

end f_monotonicity_l724_724718


namespace is_even_function_l724_724450

noncomputable def f (x : ℝ) : ℝ := log 10 (x ^ 2)

theorem is_even_function : ∀ x : ℝ, f x = f (-x) :=
by {
  -- proof steps are skipped
  sorry
}

end is_even_function_l724_724450


namespace sum_possible_values_l724_724539

theorem sum_possible_values (n : ℝ) (h1 : n ≠ 2) (h2 : n ≠ 5) (h3 : n ≠ 8) (h4 : n ≠ 11) 
(h5 : ∃ median, 
       (let s := ({2, 5, 8, 11} : set ℝ).insert n in 
        median = (s.to_list.nth 2).get_or_else 0 ∧ 
        ((2 + 5 + 8 + 11 + n) / 5 = median))): 
    6.5 + 14 + (-1) = 19.5 :=
by {
    sorry
}

end sum_possible_values_l724_724539


namespace coefficient_x9_in_expansion_zero_l724_724310

theorem coefficient_x9_in_expansion_zero :
  let f := (λ (x : ℤ), (x^3/3 - 3/x^2)^10) in
  polynomial.coeff (f x) 9 = 0 :=
by
  let x := sorry
  let f := (λ (x : ℤ), (x^3/3 - 3/x^2)^10)
  have h : polynomial.coeff (f x) 9 = 0, from sorry
  exact h

end coefficient_x9_in_expansion_zero_l724_724310


namespace parabola_problem_l724_724612

theorem parabola_problem (a x1 x2 y1 y2 : ℝ)
  (h1 : y1^2 = a * x1)
  (h2 : y2^2 = a * x2)
  (h3 : x1 + x2 = 8)
  (h4 : (x2 - x1)^2 + (y2 - y1)^2 = 144) : 
  a = 8 := 
sorry

end parabola_problem_l724_724612


namespace three_point_sixty_eight_as_fraction_l724_724972

theorem three_point_sixty_eight_as_fraction : 3.68 = 92 / 25 := 
by 
  sorry

end three_point_sixty_eight_as_fraction_l724_724972


namespace angle_AFE_eq_140_l724_724013

theorem angle_AFE_eq_140
  (A B C D E F : Point)
  (h1 : is_rectangle A B C D)
  (h2 : dist A B = 2 * dist C D)
  (h3 : E_on_extension_CD : ∃x, x > 1 ∧ (vector (A - D) ∧ (E = D + x * (C - D))))
  (h4 : angle_ade_100 : ∠ ADE = 100)
  (h5 : F_on_AD : ∃ y, F = A + y * (D - A))
  (h6 : dist E F = dist D F)
  (h7 : ¬ E_on_half_plane : E ∉ (set_of_points_on_half_plane A B D)) :
  ∠ AFE = 140 :=
by
  sorry

end angle_AFE_eq_140_l724_724013


namespace find_b_l724_724319

noncomputable def conjugate (z : ℂ) : ℂ := conj z

theorem find_b
  (a b c d : ℝ)
  (h1 : ∀ z : ℂ, (z^4 + a * z^3 + b * z^2 + c * z + d = 0) → z.im ≠ 0)
  (z w : ℂ)
  (h2 : z * w = 10 + 2 * complex.I)
  (h3 : conjugate z + conjugate w = 2 + 3 * complex.I)
  (i_squared : ∀ (i : ℂ), i * i = -1) :
  b = 33 :=
sorry

end find_b_l724_724319


namespace proposition1_proposition2_proposition3_proposition4_true_propositions_l724_724293

theorem proposition1 (α β : Plane) (l : Line) (a b : Line)
  (H1 : α ∩ β = l)
  (H2 : a ⊂ α)
  (H3 : b ⊂ β)
  (H4 : skew a b) :
  (intersect a l ∨ intersect b l)  :=
sorry

theorem proposition2 : 
  ∀ (a b : ℝ), a + b = 3 -> (2^a + 2^b ≥ 4*sqrt 2) :=
sorry

theorem proposition3 (x : ℝ) :
  (∃ (z : ℂ), z = (1 - x^2) + (1 + x)*i ∧ is_purely_imaginary z) -> 
  (log (abs x) ≠ 0) :=
sorry

theorem proposition4 (a : ℕ → ℝ) (Sn : ℕ → ℝ)
  (H1 : ∀ n, Sn n = (1 / 2) * (a n + 1 / a n)) :
  ∀ n ∈ ℕ+, a n = sqrt n - sqrt (n - 1) :=
sorry

theorem true_propositions :
  (proposition1) ∧ (proposition2) ∧ (proposition4) :=
sorry

end proposition1_proposition2_proposition3_proposition4_true_propositions_l724_724293


namespace A_sym_diff_B_l724_724320

-- Definitions of sets and operations
def set_diff (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}
def sym_diff (M N : Set ℝ) : Set ℝ := set_diff M N ∪ set_diff N M

def A : Set ℝ := {y | ∃ x : ℝ, y = 3^x}
def B : Set ℝ := {y | ∃ x : ℝ, y = -(x-1)^2 + 2}

-- The target equality to prove
theorem A_sym_diff_B : sym_diff A B = (({y | y ≤ 0}) ∪ ({y | y > 2})) :=
by
  sorry

end A_sym_diff_B_l724_724320


namespace num_correct_propositions_l724_724730

   /-- Given two different lines and two different planes, consider the following five propositions:
       Proposition 1: If a line m is perpendicular to a plane α, and a line l is perpendicular to a plane β, then l is perpendicular to α.
       Proposition 2: If a line m is perpendicular to a plane α, and a line l is parallel to m, and l intersects plane β, then α is perpendicular to β.
       Proposition 3: If plane α is parallel to plane β, and a line l is perpendicular to α, and l intersects plane β, then l is perpendicular to a line m that is parallel to β.
       Proposition 4: If plane α is parallel to plane β, and a line l is parallel to α, and l intersects plane β, then l is parallel to a line m that is parallel to β.
       Proposition 5: If plane α is perpendicular to plane β, and the intersection of α and β is line l, and a line m is perpendicular to l, then m is perpendicular to β. 
       The number of correct propositions is 2.
   -/
   theorem num_correct_propositions (α β : Plane) (l m : Line) : 
     (is_perpendicular m α → is_perpendicular l β → is_perpendicular l α) →
     (is_perpendicular m α → is_parallel l m → intersects l β → is_perpendicular α β) →
     (is_parallel α β → is_perpendicular l α → intersects l β → exists (m : Line), is_parallel m β ∧ is_perpendicular l m) →
     (is_parallel α β → is_parallel l α → intersects l β → exists (m : Line), is_parallel m β ∧ is_parallel l m) →
     (is_perpendicular α β → (intersection α β = l) → is_perpendicular m l → is_perpendicular m β) →
     2 := sorry
   
end num_correct_propositions_l724_724730


namespace subset_A_has_only_one_element_l724_724925

theorem subset_A_has_only_one_element (m : ℝ) :
  (∀ x y, (mx^2 + 2*x + 1 = 0) → (mx*y^2 + 2*y + 1 = 0) → x = y) →
  (m = 0 ∨ m = 1) :=
by
  sorry

end subset_A_has_only_one_element_l724_724925


namespace nails_remaining_proof_l724_724253

noncomputable
def remaining_nails (initial_nails kitchen_percent fence_percent : ℕ) : ℕ :=
  let kitchen_used := initial_nails * kitchen_percent / 100
  let remaining_after_kitchen := initial_nails - kitchen_used
  let fence_used := remaining_after_kitchen * fence_percent / 100
  let final_remaining := remaining_after_kitchen - fence_used
  final_remaining

theorem nails_remaining_proof :
  remaining_nails 400 30 70 = 84 := by
  sorry

end nails_remaining_proof_l724_724253


namespace pages_torn_and_sheets_calculation_l724_724898

theorem pages_torn_and_sheets_calculation : 
  (∀ (n : ℕ), (sheet_no n) = (n + 1) / 2 → (2 * (n + 1) / 2) - 1 = n ∨ 2 * (n + 1) / 2 = n) →
  let first_page := 185 in
  let last_page := 518 in
  last_page = 518 → 
  ((last_page - first_page + 1) / 2) = 167 := 
by
  sorry

end pages_torn_and_sheets_calculation_l724_724898


namespace cube_surface_area_increase_l724_724216

theorem cube_surface_area_increase (s : ℝ) :
  let original_surface_area := 6 * s^2
  let new_side_length := 1.8 * s
  let new_surface_area := 6 * (new_side_length)^2
  let percentage_increase := (new_surface_area - original_surface_area) / original_surface_area * 100
  percentage_increase = 1844 :=
by
  unfold original_surface_area
  unfold new_side_length
  unfold new_surface_area
  unfold percentage_increase
  sorry

end cube_surface_area_increase_l724_724216


namespace simplify_expression_l724_724091

variable (x : ℝ)

theorem simplify_expression (h : x ≠ 1) : (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 :=
by sorry

end simplify_expression_l724_724091


namespace sequence_1001st_term_l724_724265

theorem sequence_1001st_term (a b : ℤ) (h1 : b = 2 * a - 3) : 
  ∃ n : ℤ, n = 1001 → (a + 1000 * (20 * a - 30)) = 30003 := 
by 
  sorry

end sequence_1001st_term_l724_724265


namespace common_ratio_of_geometric_series_l724_724714

theorem common_ratio_of_geometric_series (a₁ q : ℝ) 
  (S_3 : ℝ) (S_2 : ℝ) 
  (hS3 : S_3 = a₁ * (1 - q^3) / (1 - q)) 
  (hS2 : S_2 = a₁ * (1 - q^2) / (1 - q)) 
  (h_ratio : S_3 / S_2 = 3 / 2) :
  q = 1 ∨ q = -1/2 :=
by
  -- Proof goes here.
  sorry

end common_ratio_of_geometric_series_l724_724714


namespace tank_capacity_l724_724269

theorem tank_capacity (C : ℝ) (h1 : 1/4 * C + 180 = 3/4 * C) : C = 360 :=
sorry

end tank_capacity_l724_724269


namespace find_a2_l724_724391

theorem find_a2 
  (a1 a2 a3 : ℝ)
  (h1 : a1 * a2 * a3 = 15)
  (h2 : (3 / (a1 * 3 * a2)) + (15 / (3 * a2 * 5 * a3)) + (5 / (5 * a3 * a1)) = 3 / 5) :
  a2 = 3 :=
sorry

end find_a2_l724_724391


namespace sequence_sum_l724_724042

theorem sequence_sum (x : ℕ → ℝ) (n : ℕ)
  (h : ∀ n : ℕ, (finset.range (n + 1)).sum (λ i, (x i)^3) = ((finset.range (n + 1)).sum (λ i, x i))^2) :
  ∃ m : ℕ, (finset.range (n + 1)).sum (λ i, x i) = m * (m + 1) / 2 :=
by sorry

end sequence_sum_l724_724042


namespace maximum_value_of_function_l724_724534

noncomputable def f (x: ℝ) : ℝ := x^3 - 12 * x + 16

theorem maximum_value_of_function : 
  (∀ x ∈ Icc (-2:ℝ) 3, f x ≤ 32) ∧
  (∃ x ∈ Icc (-2:ℝ) 3, f x = 32) :=
begin
  sorry
end

end maximum_value_of_function_l724_724534


namespace clothing_order_equation_l724_724605

open Real

-- Definitions and conditions
def total_pieces : ℕ := 720
def initial_rate : ℕ := 48
def days_earlier : ℕ := 5

-- Statement that we need to prove
theorem clothing_order_equation (x : ℕ) :
    (720 / 48 : ℝ) - (720 / (x + 48) : ℝ) = 5 := 
sorry

end clothing_order_equation_l724_724605


namespace magic_8_ball_probability_l724_724798

open ProbabilityTheory
noncomputable theory

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (choose n k) * (p^k) * ((1 - p)^(n - k))

theorem magic_8_ball_probability :
  binomial_probability 7 3 (1/3) = 560 / 2187 :=
by
  sorry

end magic_8_ball_probability_l724_724798


namespace park_tickets_l724_724790

theorem park_tickets (teachers students : ℕ) (ticket_cost total_money : ℕ)
  (h_teachers : teachers = 3)
  (h_students : students = 9)
  (h_ticket_cost : ticket_cost = 22)
  (h_total_money : total_money = 300) :
  total_money >= (teachers + students) * ticket_cost :=
by
  have h_people : teachers + students = 12, from eq.trans (by simp [h_teachers, h_students]) rfl
  have h_total_cost : 12 * ticket_cost = 264, from by simp [h_ticket_cost]
  have h_inequality : total_money >= 264, from by simp [h_total_money]
  exact h_inequality

end park_tickets_l724_724790


namespace pages_left_to_read_l724_724292

-- Define the given conditions
def total_pages : ℕ := 563
def pages_read : ℕ := 147

-- Define the proof statement
theorem pages_left_to_read : total_pages - pages_read = 416 :=
by
  -- The proof will be given here
  sorry

end pages_left_to_read_l724_724292


namespace irrational_sqrt_ten_l724_724956

-- Definitions for the conditions
def one_seventh := 1 / 7
def three_point_five := 3.5
def sqrt_ten := Real.sqrt 10
def repeating_decimal := -0.3030030003 -- This is simplified for demonstration

-- The proof statement
theorem irrational_sqrt_ten : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ sqrt_ten = a / b ∧ (sqrt_ten * sqrt_ten = 10) ∧
    (∀ (x : ℚ), x ≠ sqrt_ten) ∧ ∃ (x : ℚ), x = one_seventh ∨ x = three_point_five ∨ x = repeating_decimal :=
by sorry

end irrational_sqrt_ten_l724_724956


namespace torn_sheets_count_l724_724908

noncomputable def first_page_num : ℕ := 185
noncomputable def last_page_num : ℕ := 518
noncomputable def pages_per_sheet : ℕ := 2

theorem torn_sheets_count :
  last_page_num > first_page_num ∧
  last_page_num.digits = first_page_num.digits.rotate 1 ∧
  pages_per_sheet = 2 →
  (last_page_num - first_page_num + 1)/pages_per_sheet = 167 :=
by {
  sorry
}

end torn_sheets_count_l724_724908


namespace y_coords_diff_of_ellipse_incircle_area_l724_724395

theorem y_coords_diff_of_ellipse_incircle_area
  (x1 y1 x2 y2 : ℝ)
  (F1 F2 : ℝ × ℝ)
  (a b : ℝ)
  (h1 : a^2 = 25)
  (h2 : b^2 = 9)
  (h3 : F1 = (-4, 0))
  (h4 : F2 = (4, 0))
  (h5 : 4 * (|y1 - y2|) = 20)
  (h6 : ∃ (x : ℝ), (x / 25)^2 + (y1 / 9)^2 = 1 ∧ (x / 25)^2 + (y2 / 9)^2 = 1) :
  |y1 - y2| = 5 :=
sorry

end y_coords_diff_of_ellipse_incircle_area_l724_724395


namespace number_of_loafers_sold_l724_724661

noncomputable theory

def commission_rate : ℝ := 0.15
def suit_price : ℝ := 700
def num_suits : ℕ := 2
def shirt_price : ℝ := 50
def num_shirts : ℕ := 6
def loafer_price : ℝ := 150
def total_commission : ℝ := 300

theorem number_of_loafers_sold :
  ∃ (num_loafers : ℕ), 
  let commission_per_suit := commission_rate * suit_price,
      commission_per_shirt := commission_rate * shirt_price,
      commission_per_loafer := commission_rate * loafer_price,
      total_sales_commission := (num_suits * commission_per_suit) + (num_shirts * commission_per_shirt),
      loafers_commission := total_commission - total_sales_commission
  in loafers_commission / commission_per_loafer = num_loafers ∧ num_loafers = 2 :=
by 
  sorry

end number_of_loafers_sold_l724_724661


namespace painting_cost_l724_724806

-- Define contributions
def JudsonContrib := 500
def KennyContrib := JudsonContrib + 0.20 * JudsonContrib
def CamiloContrib := KennyContrib + 200

-- Define total cost
def TotalCost := JudsonContrib + KennyContrib + CamiloContrib

-- Theorem to prove
theorem painting_cost : TotalCost = 1900 :=
by 
  -- Calculate Kenny's contribution
  have hK : KennyContrib = 600 := by 
    simp [KennyContrib, JudsonContrib]
    sorry -- additional steps would go here, we use sorry to skip details

  -- Calculate Camilo's contribution
  have hC : CamiloContrib = 800 := by 
    simp [CamiloContrib, hK]
    sorry -- additional steps would go here, we use sorry to skip details

  -- Calculate total cost
  simp [TotalCost, JudsonContrib, hK, hC]
  sorry -- additional steps would go here, we use sorry to skip details

end painting_cost_l724_724806


namespace solution_set_inequality_l724_724921

theorem solution_set_inequality (x : ℝ) : 
  (∃ x, (x-1)/((x^2) - x - 30) > 0) ↔ (x > -5 ∧ x < 1) ∨ (x > 6) :=
by
  sorry

end solution_set_inequality_l724_724921


namespace area_of_trapezium_l724_724124

-- Define the conditions of the problem
variables {A B C D E F G : Type*}
variables (AD BC CE : ℝ)

-- Given initial conditions
def initial_conditions (h1 : AD * 3 = BC)
                       (h2 : CE * 3 = BC)
                       (h3 : 0 < AD)
                       (h4 : 0 < CE)
                       (area_ΔGCE : ℝ := 15) :=
  True

-- Lean proof statement for the area of trapezium ABCD
theorem area_of_trapezium (h1 : AD * 3 = BC)
                          (h2 : CE * 3 = BC)
                          (h3 : 0 < AD)
                          (h4 : 0 < CE)
                          (h5 : initial_conditions h1 h2 h3 h4) :
  let area_trap : ℝ := 360 in
  True := sorry

end area_of_trapezium_l724_724124


namespace find_n_given_sum_l724_724213

theorem find_n_given_sum (n : ℕ) :
  (∃ (e : ℕ → ℕ), (∀ k, k ≥ 1 → e k = 2 * (3 * k) - 1) ∧ 
                   (∑ i in finset.range (n + 1), e (i + 1)) = 597) → n = 13 :=
by
  sorry

end find_n_given_sum_l724_724213


namespace intersection_A_B_l724_724727

def set_A : Set ℝ := { x | abs (x - 1) < 2 }
def set_B : Set ℝ := { x | Real.log x / Real.log 2 > Real.log x / Real.log 3 }

theorem intersection_A_B : set_A ∩ set_B = {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l724_724727


namespace direct_proportion_graph_is_straight_line_l724_724873

-- Defining the direct proportion function
def direct_proportion_function (k x : ℝ) : ℝ := k * x

-- Theorem statement
theorem direct_proportion_graph_is_straight_line (k : ℝ) :
  ∀ x : ℝ, ∃ y : ℝ, y = direct_proportion_function k x ∧ 
    ∀ (x1 x2 : ℝ), 
    ∃ a b : ℝ, b ≠ 0 ∧ 
    (a * x1 + b * (direct_proportion_function k x1)) = (a * x2 + b * (direct_proportion_function k x2)) :=
by
  sorry

end direct_proportion_graph_is_straight_line_l724_724873


namespace trajectory_is_ellipse_l724_724171

theorem trajectory_is_ellipse (z : ℂ) (h : abs (z + complex.i) + abs (z - complex.i) = 4) : 
  ∃ a b c : ℝ, a ≠ b ∧ ∀ (w : ℂ), abs (w + complex.i) + abs (w - complex.i) = 4 ↔ (c * ((w.re - 0)^2 / a^2) + (w.im - 0)^2 / b^2) = 1 :=
sorry

end trajectory_is_ellipse_l724_724171


namespace triangle_single_lattice_point_centroid_l724_724926

theorem triangle_single_lattice_point_centroid (A B C D : ℤ × ℤ)
  (hA : ∃ z1 z2 : ℤ, A = (z1, z2))
  (hB : ∃ z1 z2 : ℤ, B = (z1, z2))
  (hC : ∃ z1 z2 : ℤ, C = (z1, z2))
  (hD : ∃ z1 z2 : ℤ, D = (z1, z2))
  (h_inside : D ≠ A ∧ D ≠ B ∧ D ≠ C)
  (h_convex_comb :
    ∃ λ μ ν : ℚ, 
      0 < λ ∧ 0 < μ ∧ 0 < ν ∧ λ + μ + ν = 1 ∧ 
      ((λ : ℚ) * A.1 + (μ : ℚ) * B.1 + (ν : ℚ) * C.1 = D.1 ∧
       (λ : ℚ) * A.2 + (μ : ℚ) * B.2 + (ν : ℚ) * C.2 = D.2))
  (h_single_inside : ∀ P : ℤ × ℤ, 
    P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ 
    ∃ λ μ ν : ℚ, 
      0 ≤ λ ∧ 0 ≤ μ ∧ 0 ≤ ν ∧ λ + μ + ν = 1 ∧ 
      ((λ : ℚ) * A.1 + (μ : ℚ) * B.1 + (ν : ℚ) * C.1 = P.1 ∧
       (λ : ℚ) * A.2 + (μ : ℚ) * B.2 + (ν : ℚ) * C.2 = P.2) → P = D) :
  D = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3) :=
sorry

end triangle_single_lattice_point_centroid_l724_724926


namespace chromium_percentage_A5_correct_l724_724432

-- Define the initial conditions for the problem
def weight_A1 : ℝ := 15
def chromium_percentage_A1 : ℝ := 12 / 100
def chromium_A1 := chromium_percentage_A1 * weight_A1

def weight_A2 : ℝ := 35
def chromium_percentage_A2 : ℝ := 10 / 100
def chromium_A2 := chromium_percentage_A2 * weight_A2

def weight_A3 : ℝ := 25
def chromium_percentage_A3 : ℝ := 8 / 100
def chromium_A3 := chromium_percentage_A3 * weight_A3

def weight_A4 : ℝ := 10
def chromium_percentage_A4 : ℝ := 15 / 100
def chromium_A4 := chromium_percentage_A4 * weight_A4

def total_chromium : ℝ := chromium_A1 + chromium_A2 + chromium_A3 + chromium_A4
def total_weight : ℝ := weight_A1 + weight_A2 + weight_A3 + weight_A4
def chromium_percentage_A5 : ℝ := (total_chromium / total_weight) * 100

-- Statement of the proof problem
theorem chromium_percentage_A5_correct : chromium_percentage_A5 = 10.35 := by
  sorry

end chromium_percentage_A5_correct_l724_724432


namespace widgets_unloaded_l724_724980
-- We import the necessary Lean library for general mathematical purposes.

-- We begin the lean statement for our problem.
theorem widgets_unloaded (n_doo n_geegaw n_widget n_yamyam : ℕ) :
  (2^n_doo) * (11^n_geegaw) * (5^n_widget) * (7^n_yamyam) = 104350400 →
  n_widget = 2 := by
  -- Placeholder for proof
  sorry

end widgets_unloaded_l724_724980


namespace next_four_customers_cases_l724_724551

theorem next_four_customers_cases (total_people : ℕ) (first_eight_cases : ℕ) (last_eight_cases : ℕ) (total_cases : ℕ) :
    total_people = 20 →
    first_eight_cases = 24 →
    last_eight_cases = 8 →
    total_cases = 40 →
    (total_cases - (first_eight_cases + last_eight_cases)) / 4 = 2 :=
by
  intro h1 h2 h3 h4
  -- Fill in the proof steps using h1, h2, h3, and h4
  sorry

end next_four_customers_cases_l724_724551


namespace general_term_arithmetic_sequence_l724_724704

variable {α : Type*}
variables (a_n a : ℕ → ℕ) (d a_1 a_2 a_3 a_4 n : ℕ)

-- Define the arithmetic sequence condition
def arithmetic_sequence (a_n : ℕ → ℕ) (d : ℕ) :=
  ∀ n, a_n (n + 1) = a_n n + d

-- Define the inequality solution condition 
def inequality_solution_set (a_1 a_2 : ℕ) (x : ℕ) :=
  a_1 ≤ x ∧ x ≤ a_2

theorem general_term_arithmetic_sequence :
  arithmetic_sequence a_n d ∧ (d ≠ 0) ∧ 
  (∀ x, x^2 - a_3 * x + a_4 ≤ 0 ↔ inequality_solution_set a_1 a_2 x) →
  a_n = 2 * n :=
by
  sorry

end general_term_arithmetic_sequence_l724_724704


namespace polar_coordinates_standard_representation_l724_724441

theorem polar_coordinates_standard_representation :
  ∀ (r θ : ℝ), (r, θ) = (-4, 5 * Real.pi / 6) → (∃ (r' θ' : ℝ), r' > 0 ∧ (r', θ') = (4, 11 * Real.pi / 6))
:= by
  sorry

end polar_coordinates_standard_representation_l724_724441


namespace total_profit_is_80000_l724_724983

variable (P : ℝ)
variable (majority_share partners_share remaining_profit : ℝ)

-- Conditions
def majority_owner_receives : majority_share = 0.25 * P := by sorry
def partners_receive : partners_share = 0.1875 * P := by sorry
def remaining_percentage : remaining_profit = 0.75 * P := by sorry
def combined_receives_50000 : majority_share + 2 * partners_share = 50000 := by sorry

-- Statement to prove
theorem total_profit_is_80000 : P = 80000 :=
by
  have h1 : majority_share = 0.25 * P := majority_owner_receives
  have h2 : partners_share = 0.1875 * P := partners_receive
  have h3 : majority_share + 2 * partners_share = 50000 := combined_receives_50000
  have h4 : 0.25 * P + 0.375 * P = 50000 := by
    rw [h1, h2]
    exact h3
  have h5 : 0.625 * P = 50000 := by
    linarith
  have h6 : P = 50000 / 0.625 := by
    linarith
  have h7 : P = 80000 := by
    norm_num [h6]
  exact h7

end total_profit_is_80000_l724_724983


namespace find_multiple_of_benjy_peaches_l724_724479

theorem find_multiple_of_benjy_peaches
(martine_peaches gabrielle_peaches : ℕ)
(benjy_peaches : ℕ)
(m : ℕ)
(h1 : martine_peaches = 16)
(h2 : gabrielle_peaches = 15)
(h3 : benjy_peaches = gabrielle_peaches / 3)
(h4 : martine_peaches = m * benjy_peaches + 6) :
m = 2 := by
sorry

end find_multiple_of_benjy_peaches_l724_724479


namespace ball_returns_to_bella_after_13_throws_l724_724554

theorem ball_returns_to_bella_after_13_throws :
  ∃ n : ℕ, n = 13 ∧
  let positions := list.range 1 14  -- positions = [1, 2, ..., 13]
  in ∃ (f : ℕ → ℕ), (∀ i < n, f (list.nth positions i) = list.nth positions ((i + 5) % 13)) ∧
    (list.nth (list.iterate f n 1) 13 = 1)  :=
begin
  sorry
end

end ball_returns_to_bella_after_13_throws_l724_724554


namespace decimal_to_fraction_simplify_l724_724971

theorem decimal_to_fraction_simplify (d : ℚ) (h : d = 3.68) : d = 92 / 25 :=
by
  rw h
  sorry

end decimal_to_fraction_simplify_l724_724971


namespace sum_unique_prime_factors_195195_eq_39_l724_724568

theorem sum_unique_prime_factors_195195_eq_39 :
  (∀ (p : ℕ), p ∣ 195195 → prime p → p ∈ {3, 5, 7, 11, 13}) → (∑ (p : ℕ) in {3, 5, 7, 11, 13}, p) = 39 :=
by
  intro h
  sorry

end sum_unique_prime_factors_195195_eq_39_l724_724568


namespace symmetric_point_reflection_l724_724182

-- Defining the coordinates for point A
def A := (3, 2)

-- Function to translate a point along the x-axis
def translate_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

-- Function to find the reflection of a point with respect to y-axis
def reflect_over_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

-- Main theorem stating the proof goal
theorem symmetric_point_reflection :
  reflect_over_y (translate_left A 4) = (1, 2) :=
by
  -- Lean proof goes here (using sorry for now)
  sorry

end symmetric_point_reflection_l724_724182


namespace valid_two_digit_numbers_l724_724166

-- Define the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + (n / 10 % 10) + (n / 100 % 10) + (n / 1000 % 10)

-- Prove the statement about two-digit numbers satisfying the condition
theorem valid_two_digit_numbers :
  {A : ℕ | 10 ≤ A ∧ A ≤ 99 ∧ (sum_of_digits A)^2 = sum_of_digits (A^2)} =
  {10, 11, 12, 13, 20, 21, 22, 30, 31} :=
by
  sorry

end valid_two_digit_numbers_l724_724166


namespace exists_max_pile_division_l724_724051

theorem exists_max_pile_division (k : ℝ) (hk : k < 2) : 
  ∃ (N_k : ℕ), ∀ (A : Multiset ℝ) (m : ℝ), (∀ a ∈ A, a < 2 * m) → 
    ¬(∃ B : Multiset ℝ, B.card > N_k ∧ (∀ b ∈ B, b ∈ A ∧ b < 2 * m)) :=
sorry

end exists_max_pile_division_l724_724051


namespace Petya_tore_out_sheets_l724_724885

theorem Petya_tore_out_sheets (n m : ℕ) (h1 : n = 185) (h2 : m = 518)
  (h3 : m.digits = n.digits) : (m - n + 1) / 2 = 167 :=
by
  sorry

end Petya_tore_out_sheets_l724_724885


namespace emptying_tank_time_l724_724623

theorem emptying_tank_time :
  let V := 30 * 12^3 -- volume of the tank in cubic inches
  let r_in := 3 -- rate of inlet pipe in cubic inches per minute
  let r_out1 := 12 -- rate of first outlet pipe in cubic inches per minute
  let r_out2 := 6 -- rate of second outlet pipe in cubic inches per minute
  let net_rate := r_out1 + r_out2 - r_in
  V / net_rate = 3456 := by
sorry

end emptying_tank_time_l724_724623


namespace reciprocal_of_neg_three_l724_724142

theorem reciprocal_of_neg_three : -3 * (-1 / 3) = 1 := 
by
  sorry

end reciprocal_of_neg_three_l724_724142


namespace simplify_expression_l724_724088

variable (x : ℝ)

theorem simplify_expression (h : x ≠ 1) : (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 :=
by sorry

end simplify_expression_l724_724088


namespace DE_value_l724_724835

theorem DE_value
  {D E F P Q G : Type}
  (DP EQ DE : ℝ)
  (h1 : DP = 15)
  (h2 : EQ = 20)
  (h3 : ∀ {T : Triangle P Q G}, T.isMedian D P Eq T.isMedian E Q)
  (h4 : ∀ {T : Triangle D E F}, T.isRightAngle E) :
  DE = 50 / 3 := by
  sorry

end DE_value_l724_724835


namespace david_mowing_hours_l724_724138

variable (h : ℕ) -- defining h as a natural number since time in hours should be non-negative

-- defining all the conditions as hypotheses in Lean
theorem david_mowing_hours 
(rate : ℕ) -- rate for mowing per hour
(days : ℕ) -- total number of days David mowed
(half_spent_on_shoes : ℕ) -- half the money spent on shoes
(half_given_to_mom : ℕ) -- half the remaining money given to mom
(money_left : ℕ) -- money left after all other expenditures
(M : ℕ) -- total money David earned

(hypo1 : rate = 14)
(hypo2 : days = 7)
(hypo3 : M = rate * days * h) -- total money earned M
(hypo4 : half_spent_on_shoes = 1 / 2 * M) -- half the money spent on shoes
(hypo5 : half_given_to_mom = 1 / 2 * (M - half_spent_on_shoes)) -- half the remaining money given to mom
(hypo6 : 49 =  1 / 2 * (M - half_spent_on_shoes)) -- David had $49 left after giving half the remaining money to mom

:
h = 2 :=
begin
  sorry
end

end david_mowing_hours_l724_724138


namespace grade_point_average_one_third_l724_724126

theorem grade_point_average_one_third :
  ∃ (x : ℝ), 55 = (1/3) * x + (2/3) * 60 ∧ x = 45 :=
by
  sorry

end grade_point_average_one_third_l724_724126


namespace torn_sheets_count_l724_724905

noncomputable def first_page_num : ℕ := 185
noncomputable def last_page_num : ℕ := 518
noncomputable def pages_per_sheet : ℕ := 2

theorem torn_sheets_count :
  last_page_num > first_page_num ∧
  last_page_num.digits = first_page_num.digits.rotate 1 ∧
  pages_per_sheet = 2 →
  (last_page_num - first_page_num + 1)/pages_per_sheet = 167 :=
by {
  sorry
}

end torn_sheets_count_l724_724905


namespace chocoBites_mod_l724_724665

theorem chocoBites_mod (m : ℕ) (hm : m % 8 = 5) : (4 * m) % 8 = 4 :=
by
  sorry

end chocoBites_mod_l724_724665


namespace correct_conclusions_l724_724323

-- Define the quadratic function with given constant constraints
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Given conditions
variables (a b c m n : ℝ)
-- a != 0
-- y(-1) = m, y(0) = 2, y(1) = 2, y(2) = n
-- y(3/2) < 0
def y_neg1 := quadratic a b c (-1) = m
def y_0 := quadratic a b c 0 = 2
def y_1 := quadratic a b c 1 = 2
def y_2 := quadratic a b c 2 = n
def y_3halfs := quadratic a b c (3/2) < 0

-- Proven conclusions
theorem correct_conclusions :
    y_neg1 → y_0 → y_1 → y_2 → y_3halfs →
    (¬ (a * b * c > 0)) ∧
    (∀ x : ℝ, x ≤ 0 → quadratic a b c x ≤ quadratic a b c (x + 1)) ∧
    (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ 0 < x₂ ∧ quadratic a b c x₁ = 0 ∧ quadratic a b c x₂ = 0 ∧ (-1/2 < x₁ ∧ x₁ < 0)) ∧
    (3 * m - n < -20/3) :=
by
  intros
  sorry

end correct_conclusions_l724_724323


namespace sum_of_factorial_multiples_l724_724315

open BigOperators

theorem sum_of_factorial_multiples (n : ℕ) :
  (∑ k in Finset.range (n+1), k * k!) = (n+1)! - 1 :=
by
  -- Since we are only required to state, we use sorry to skip the proof
  sorry

end sum_of_factorial_multiples_l724_724315


namespace new_average_daily_production_l724_724322

-- Definition of conditions
def initial_average : ℕ → ℕ := λ n, 50
def today_production : ℕ := 60
def n : ℕ := 1

-- Statement to prove
theorem new_average_daily_production : 
  ((n * initial_average n + today_production) / (n + 1)) = 55 :=
by
  sorry

end new_average_daily_production_l724_724322


namespace find_x_l724_724308

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : ⌈x⌉₊ * x = 198) : x = 13.2 :=
by
  sorry

end find_x_l724_724308


namespace perimeter_of_triangle_ADE_l724_724366

theorem perimeter_of_triangle_ADE
  (a b : ℝ) (F1 F2 A : ℝ × ℝ) (D E : ℝ × ℝ) 
  (h_ellipse : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1)
  (h_a_gt_b : a > b)
  (h_b_gt_0 : b > 0)
  (h_eccentricity : ∃ c, c / a = 1 / 2 ∧ a^2 - b^2 = c^2)
  (h_F1_F2 : ∀ F1 F2, distance F1 (0, 0) = distance F2 (0, 0) ∧ F1 ≠ F2 ∧ 
                       ∀ P : ℝ × ℝ, (distance P F1 + distance P F2 = 2 * a) ↔ (x : ℝ)(y : ℝ) (h_ellipse x y))
  (h_line_DE : ∃ k, ∃ c, ∀ x F1 A, (2 * a * x/(sqrt k^2 + 1)) = |DE|
  (h_length_DE : |DE| = 6)
  (h_A_vertex : A = (0, b))
  : ∃ perim : ℝ, perim = 13 :=
sorry

end perimeter_of_triangle_ADE_l724_724366


namespace right_triangles_count_l724_724739

theorem right_triangles_count : 
  (∃ (n : ℕ), n = 10 ∧ ∀ (a b : ℕ), b < 100 ∧ a^2 + b^2 = (b + 2)^2) :=
begin
  sorry
end

end right_triangles_count_l724_724739


namespace odd_handshake_even_count_l724_724299

theorem odd_handshake_even_count {n : ℕ} (d : Fin n → ℕ) 
  (h_sum_even : ∑ i, d i % 2 = 0) :
  (Finset.card (Finset.filter (λ i, d i % 2 = 1) Finset.univ) % 2 = 0) :=
sorry

end odd_handshake_even_count_l724_724299


namespace projection_of_AB_on_AC_coordinates_of_point_D_l724_724410

noncomputable def point := (ℝ × ℝ)

def A : point := (0, 3)
def B : point := (2, 2)
def C : point := (-4, 6)

def vector_sub (p q : point) : point := (p.1 - q.1, p.2 - q.2)

def dot_product (v w : point) : ℝ := v.1 * w.1 + v.2 * w.2

def vector_length (v : point) : ℝ := real.sqrt (v.1^2 + v.2^2)

def vector_div (v : point) (k : ℝ) : point := (v.1 / k, v.2 / k)

def projection (v w : point) : ℝ := dot_product v (vector_div w (vector_length w))

theorem projection_of_AB_on_AC : 
  projection (vector_sub B A) (vector_sub C A) = -11 / 5 := sorry

def D : point := (-22 / 5, 26 / 5)

def vector_parallel (v w : point) : Prop := v.1 * w.2 = v.2 * w.1

def vector_perpendicular (v w : point) : Prop := dot_product v w = 0

theorem coordinates_of_point_D : 
  (vector_parallel (vector_sub B A) (vector_sub D A)) ∧ 
  (vector_perpendicular (vector_sub B A) (vector_sub D C)) := 
  sorry

end projection_of_AB_on_AC_coordinates_of_point_D_l724_724410


namespace find_2alpha_minus_beta_l724_724684

theorem find_2alpha_minus_beta (α β : ℝ) (tan_diff : Real.tan (α - β) = 1 / 2) 
  (cos_β : Real.cos β = -7 * Real.sqrt 2 / 10) (α_range : 0 < α ∧ α < Real.pi) 
  (β_range : 0 < β ∧ β < Real.pi) : 2 * α - β = -3 * Real.pi / 4 :=
sorry

end find_2alpha_minus_beta_l724_724684


namespace beaker_filling_l724_724577

theorem beaker_filling (C : ℝ) (hC : 0 < C) :
    let small_beaker_salt := (1/2) * C
    let large_beaker_capacity := 5 * C
    let large_beaker_fresh := large_beaker_capacity / 5
    let large_beaker_total_fill := large_beaker_fresh + small_beaker_salt
    (large_beaker_total_fill / large_beaker_capacity) = 3 / 10 :=
by
    let small_beaker_salt := (1/2) * C
    let large_beaker_capacity := 5 * C
    let large_beaker_fresh := large_beaker_capacity / 5
    let large_beaker_total_fill := large_beaker_fresh + small_beaker_salt
    show (large_beaker_total_fill / large_beaker_capacity) = 3 / 10
    sorry

end beaker_filling_l724_724577


namespace num_divisors_in_ξ_l724_724517

-- Define S_n as the set of positive divisors of n.
def S (n : ℕ) : Finset ℕ := (Finset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0)

-- Define the conditions for a set to be in the set ξ
def is_in_ξ (n : ℕ) : Prop :=
  let sn := S n in 
  let elems := sn.toList in
  sn.card ≥ 20 ∧ sn.card % 2 = 0 ∧ 
  ∃ (m : ℕ) (a b : ℕ), (1 ≤ m ∧ m ≤ sn.card / 2 ∧ 
    ∀ i, 1 ≤ i ∧ i ≤ m → (gcd (elems.nth_le (2 * i - 2) sorry) (elems.nth_le (2 * i - 1) sorry) = 1) ∧
    ∃ j, 1 ≤ j ∧ j ≤ m ∧ 6 ∣ ((elems.nth_le (2 * j - 2) sorry)^2 + (elems.nth_le (2 * j - 1) sorry)^2 + 1))

-- The main theorem to prove
theorem num_divisors_in_ξ : 
  let divisors := S (nat.factorial 24) in 
  (divisors.filter (λ d, is_in_ξ d)).card = 64 :=
sorry

end num_divisors_in_ξ_l724_724517


namespace pages_torn_l724_724909

theorem pages_torn (n : ℕ) (H1 : n = 185) (H2 : ∃ m, m = 518 ∧ (digits 10 m = digits 10 n) ∧ (m % 2 = 0)) : 
  ∃ k, k = ((518 - 185 + 1) / 2) ∧ k = 167 :=
by sorry

end pages_torn_l724_724909


namespace expression_value_l724_724418

theorem expression_value (x y z : ℕ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  3 * x - 2 * y + 4 * z = 21 :=
by
  subst hx
  subst hy
  subst hz
  sorry

end expression_value_l724_724418


namespace simplify_fraction_l724_724105

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : 
  ( (x^2 + 1) / (x - 1) - (2*x) / (x - 1) ) = x - 1 :=
by
  -- Your proof steps would go here.
  sorry

end simplify_fraction_l724_724105


namespace geometric_sequence_nec_suff_l724_724866

theorem geometric_sequence_nec_suff (a b c : ℝ) : (b^2 = a * c) ↔ (∃ r : ℝ, b = a * r ∧ c = b * r) :=
sorry

end geometric_sequence_nec_suff_l724_724866


namespace allocate_teaching_positions_l724_724743

theorem allocate_teaching_positions :
  ∃ (ways : ℕ), ways = 10 ∧ 
    (∃ (a b c : ℕ), a + b + c = 8 ∧ 1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ 2 ≤ a) := 
sorry

end allocate_teaching_positions_l724_724743


namespace area_between_is_correct_l724_724775

noncomputable def radius_innermost := 1.5 -- derived from 2 * radius_innermost = 3 feet
def radius_middle := 3 * radius_innermost
def radius_outermost := 2 * radius_middle
def area_middle := Real.pi * (radius_middle ^ 2)
def area_outermost := Real.pi * (radius_outermost ^ 2)
def area_region := area_outermost - area_middle

theorem area_between_is_correct :
  area_region = 60.75 * Real.pi := by
  sorry

end area_between_is_correct_l724_724775


namespace plane_equation_l724_724589

open Function -- helps use basic function properties
open Real -- for real number operations
open Locale

variable {A B C : EuclideanSpace ℝ (Fin 3)}

def A : EuclideanSpace ℝ (Fin 3) := ![2, 5, -3]
def B : EuclideanSpace ℝ (Fin 3) := ![7, 8, -1]
def C : EuclideanSpace ℝ (Fin 3) := ![9, 7, 4]

theorem plane_equation (X : EuclideanSpace ℝ (Fin 3)) 
    (h1 : X = A) 
    (h2 : ∃ u v, (u, v) ∈ Set.Icc (0 : ℝ) 1 ∧ X = u • B + v • C) :
  2 * (X 0 - 2) - (X 1 - 5) + 5 * (X 2 + 3) = 0 :=
by sorry

end plane_equation_l724_724589


namespace geometric_sequence_find_Sn_l724_724817

-- Definitions for the sequence and sum function
def a (n : ℕ) : ℕ
def S (n : ℕ) : ℕ

-- Given conditions
axiom seq_condition (n : ℕ) : S (n + 1) = 3 * S n + 2 * n + 4
axiom a1_condition : a 1 = 4

-- Part (1): Prove that {a_n + 1} forms a geometric sequence
theorem geometric_sequence : ∃ r : ℕ, ∃ b : ℕ, ∀ n : ℕ, a (n + 1) + 1 = r * (a n + 1) :=
sorry

-- Part (2): Find the sum S_n
theorem find_Sn (n : ℕ) : S n = (5 * (3^n - 1)) / 2 - n :=
sorry

end geometric_sequence_find_Sn_l724_724817


namespace intersection_area_two_circles_l724_724942

theorem intersection_area_two_circles :
  let r : ℝ := 3
  let center1 : ℝ × ℝ := (3, 0)
  let center2 : ℝ × ℝ := (0, 3)
  let intersection_area := (9 * Real.pi - 18) / 2
  (∃ x y : ℝ, (x - center1.1)^2 + y^2 = r^2 ∧ x^2 + (y - center2.2)^2 = r^2) →
  (∃ (a : ℝ), a = intersection_area) :=
by
  sorry

end intersection_area_two_circles_l724_724942


namespace gcd_102_238_l724_724195

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end gcd_102_238_l724_724195


namespace train_crossing_time_is_correct_l724_724412

-- Define the constant values
def train_length : ℝ := 350        -- Train length in meters
def train_speed : ℝ := 20          -- Train speed in m/s
def crossing_time : ℝ := 17.5      -- Time to cross the signal post in seconds

-- Proving the relationship that the time taken for the train to cross the signal post is as calculated
theorem train_crossing_time_is_correct : (train_length / train_speed) = crossing_time :=
by
  sorry

end train_crossing_time_is_correct_l724_724412


namespace num_digits_abc_l724_724419

theorem num_digits_abc (a b c : ℕ) (n : ℕ) (h_a : 10^(n-1) ≤ a ∧ a < 10^n) (h_b : 10^(n-1) ≤ b ∧ b < 10^n) (h_c : 10^(n-1) ≤ c ∧ c < 10^n) :
  ¬ ((Int.natAbs ((10^(n-1) : ℕ) * (10^(n-1) : ℕ) * (10^(n-1) : ℕ)) + 1 = 3*n) ∧
     (Int.natAbs ((10^(n-1) : ℕ) * (10^(n-1) : ℕ) * (10^(n-1) : ℕ)) + 1 = 3*n - 1) ∧
     (Int.natAbs ((10^(n-1) : ℕ) * (10^(n-1) : ℕ) * (10^(n-1) : ℕ)) + 1 = 3*n - 2)) :=
sorry

end num_digits_abc_l724_724419


namespace points_earned_l724_724004

-- Definitions from conditions
def points_per_enemy : ℕ := 8
def total_enemies : ℕ := 7
def enemies_not_destroyed : ℕ := 2

-- The proof statement
theorem points_earned :
  points_per_enemy * (total_enemies - enemies_not_destroyed) = 40 := 
by
  sorry

end points_earned_l724_724004


namespace least_number_to_add_for_divisibility_by_nine_l724_724571

theorem least_number_to_add_for_divisibility_by_nine : ∃ x : ℕ, (4499 + x) % 9 = 0 ∧ x = 1 :=
by
  sorry

end least_number_to_add_for_divisibility_by_nine_l724_724571


namespace simplify_expression_l724_724089

variable (x : ℝ)

theorem simplify_expression (h : x ≠ 1) : (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 :=
by sorry

end simplify_expression_l724_724089


namespace simplify_and_evaluate_l724_724515

-- Definitions of a and b
def a : ℝ := Real.sqrt 2
def b : ℝ := Real.sqrt 6

-- Main theorem statement
theorem simplify_and_evaluate : (a - b)^2 + b * (3 * a - b) - a^2 = 2 * Real.sqrt 3 := by 
  sorry

end simplify_and_evaluate_l724_724515


namespace inspection_team_combinations_l724_724627

theorem inspection_team_combinations (total_members : ℕ)
                                     (men : ℕ)
                                     (women : ℕ)
                                     (team_size : ℕ)
                                     (men_in_team : ℕ)
                                     (women_in_team : ℕ)
                                     (h_total : total_members = 15)
                                     (h_men : men = 10)
                                     (h_women : women = 5)
                                     (h_team_size : team_size = 6)
                                     (h_men_in_team : men_in_team = 4)
                                     (h_women_in_team : women_in_team = 2) :
  nat.choose men men_in_team * nat.choose women women_in_team = nat.choose 10 4 * nat.choose 5 2 := 
by {
    rw [h_men, h_women, h_men_in_team, h_women_in_team],
    sorry
}

end inspection_team_combinations_l724_724627


namespace envelopes_problem_l724_724258

theorem envelopes_problem
  (first_machine_time : ℕ)
  (combined_time : ℕ)
  (x : ℚ)
  (h1 : first_machine_time = 8)
  (h2 : combined_time = 2) :
  1 / (first_machine_time : ℚ) + 1 / x = 1 / combined_time :=
begin
  -- Proof omitted
  sorry
end

end envelopes_problem_l724_724258


namespace SDR_count_l724_724413

noncomputable def alpha_1 : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def alpha_2 : ℝ := (1 - Real.sqrt 5) / 2

theorem SDR_count (n : ℕ) (hn : 6 ≤ n) :
  ∑ (6 * (alpha_1 ^ (n - 3) + alpha_2 ^ (n - 3) + 2)) = 
  6 * ((1 + Real.sqrt 5) / 2) ^ (n - 3) + 6 * ((1 - Real.sqrt 5) / 2) ^ (n - 3) + 12 := sorry

end SDR_count_l724_724413


namespace solution_sets_intersection_l724_724123

variable {α : Type} [LinearOrder α]
variable (f g : α → ℝ) (F G : Set α)

-- Conditions
def f_geq_0 : Real.Set := {x : α | f x ≥ 0}
def g_lt_0 : Real.Set := {x : α | g x < 0}
def f_solution_set : F = f_geq_0 f := sorry
def g_solution_set : G = g_lt_0 g := sorry

-- Question: Prove the solution set of the system of inequalities is as stated
theorem solution_sets_intersection :
  {x : α | f x < 0} ∩ {x : α | g x ≥ 0} = (Real.Set.compl F) ∩ (Real.Set.compl G) :=
by
  sorry

end solution_sets_intersection_l724_724123


namespace exist_valid_permutation_l724_724791

open List

def valid_adj_pair (a b : ℕ) : Prop :=
  (a - b = 2 ∨ b - a = 2 ∨ a = 2 * b ∨ b = 2 * a)

def valid_permutation (l : List ℕ) : Prop :=
  ∀ i, i + 1 < l.length → valid_adj_pair (nthLe l i sorry) (nthLe l (i + 1) sorry)

theorem exist_valid_permutation :
  ∃ l : List ℕ, (l.perm (range' 1 100)) ∧ valid_permutation l :=
sorry

end exist_valid_permutation_l724_724791


namespace area_of_region_inside_hexagon_outside_semicircles_l724_724438

theorem area_of_region_inside_hexagon_outside_semicircles :
  let s := 4
  let R1 := s / 2  -- radius of semicircles equal to side of the hexagon
  let R2 := s / 4  -- radius of semicircles equal to half the side of the hexagon
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let large_semicircles_area := 3 * (1 / 2 * Real.pi * R1^2)
  let small_semicircles_area := 3 * (1 / 2 * Real.pi * R2^2)
  let shaded_region_area := hexagon_area - large_semicircles_area - small_semicircles_area
  in shaded_region_area = 24 * Real.sqrt 3 - 15 * Real.pi / 2 :=
by
  sorry

end area_of_region_inside_hexagon_outside_semicircles_l724_724438


namespace product_of_roots_of_quadratic_l724_724314

theorem product_of_roots_of_quadratic :
  ∀ (x : ℝ), 12 * x^2 + 28 * x - 315 = 0 → 
  (∃ r1 r2 : ℝ, 12 * (x - r1) * (x - r2) = 12 * x^2 + 28 * x - 315 ∧ r1 * r2 = -105 / 4) :=
begin
  intros x hx,
  have h : 12 * x^2 + 28 * x - 315 = 12 * x^2 + 28 * x - 315 := by simp,
  use [r1, r2],
  split,
  { linarith, },  -- Assuming r1 and r2 are the roots of the given quadratic equation.
  { sorry, }      -- Prove r1 * r2 = -105 / 4
end

end product_of_roots_of_quadratic_l724_724314


namespace find_two_digit_numbers_l724_724168

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem find_two_digit_numbers :
  ∀ (A : ℕ), (10 ≤ A ∧ A ≤ 99) →
    (sum_of_digits A)^2 = sum_of_digits (A^2) →
    (A = 11 ∨ A = 12 ∨ A = 13 ∨ A = 20 ∨ A = 21 ∨ A = 22 ∨ A = 30 ∨ A = 31 ∨ A = 50) :=
by sorry

end find_two_digit_numbers_l724_724168


namespace arrange_books_l724_724007

-- Assuming a finite type of books, let us model our conditions
inductive BookType
| math : BookType
| history : BookType

-- Given the two types of books and the defined number of each
def num_math_books : Nat := 4
def num_history_books : Nat := 4

-- Function to compute factorial
noncomputable def factorial : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * factorial n

-- The main theorem stating the problem
theorem arrange_books : 
  ∑(n : Fin num_math_books), ∑(m : Fin num_math_books - 1), 
    (n ≠ m)→ factorial (num_math_books + num_history_books - 2) = 8640 := 
by
  sorry

end arrange_books_l724_724007


namespace find_angle_CDB_l724_724442

variables (A B C D E : Type)
variables [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C] [LinearOrderedField D] [LinearOrderedField E]

noncomputable def angle := ℝ -- Define type for angles

variables (AB AD AC ACB ACD : angle)
variables (BAD BEA CDB : ℝ)

-- Define the given angles and conditions in Lean
axiom AB_eq_AD : AB = AD
axiom angle_ACD_eq_angle_ACB : AC = ACD
axiom angle_BAD_eq_140 : BAD = 140
axiom angle_BEA_eq_110 : BEA = 110

theorem find_angle_CDB (AB_eq_AD : AB = AD)
                       (angle_ACD_eq_angle_ACB : AC = ACD)
                       (angle_BAD_eq_140 : BAD = 140)
                       (angle_BEA_eq_110 : BEA = 110) :
                       CDB = 50 :=
by
  sorry

end find_angle_CDB_l724_724442


namespace a_sufficient_but_not_necessary_l724_724460

-- Conditions
def M : Set ℕ := {1, 2}
def N (a : ℤ) : Set ℤ := {a^2}

-- Problem Statement
theorem a_sufficient_but_not_necessary (a : ℤ) : 
  (N a ⊆ M) ↔ (a = 1) ∨ (a = -1) := 
by sorry

end a_sufficient_but_not_necessary_l724_724460


namespace sin_double_angle_given_tangent_l724_724750

theorem sin_double_angle_given_tangent :
  ∀ α : ℝ, tan (α + π / 4) = 2 → sin (2 * α) = 3 / 5 :=
by
  intro α h
  sorry

end sin_double_angle_given_tangent_l724_724750


namespace eval_complex_magnitude_product_squared_l724_724302

def z1 : ℂ := 7 - 5 * complex.I
def z2 : ℂ := 5 + 12 * complex.I

theorem eval_complex_magnitude_product_squared :
  (complex.abs (z1 * z2))^2 = 12506 :=
by
  sorry

end eval_complex_magnitude_product_squared_l724_724302


namespace sheets_torn_out_l724_724893

-- Define the conditions as given in the problem
def first_torn_page : Nat := 185
def last_torn_page : Nat := 518
def pages_per_sheet : Nat := 2

-- Calculate the total number of pages torn out
def total_pages_torn_out : Nat :=
  last_torn_page - first_torn_page + 1

-- Calculate the number of sheets torn out
def number_of_sheets_torn_out : Nat :=
  total_pages_torn_out / pages_per_sheet

-- Prove that the number of sheets torn out is 167
theorem sheets_torn_out :
  number_of_sheets_torn_out = 167 :=
by
  unfold number_of_sheets_torn_out total_pages_torn_out
  rw [Nat.sub_add_cancel (Nat.le_of_lt (Nat.lt_of_le_of_ne
    (Nat.le_add_left _ _) (Nat.ne_of_lt (Nat.lt_add_one 184))))]
  rw [Nat.div_eq_of_lt (Nat.lt.base 333)] 
  sorry -- proof steps are omitted

end sheets_torn_out_l724_724893


namespace simplified_fraction_of_num_l724_724963

def num : ℚ := 368 / 100

theorem simplified_fraction_of_num : num = 92 / 25 := by
  sorry

end simplified_fraction_of_num_l724_724963


namespace strictly_increasing_interval_l724_724876

noncomputable def f (x : ℝ) := log (1 / 2) (x ^ 2 - 5 * x + 6)

theorem strictly_increasing_interval :
  ∀ x y: ℝ, x < y → (f x < f y) ↔ (x, y) ∈ (set.Ioo -∞ 2) :=
sorry

end strictly_increasing_interval_l724_724876


namespace nails_remaining_proof_l724_724254

noncomputable
def remaining_nails (initial_nails kitchen_percent fence_percent : ℕ) : ℕ :=
  let kitchen_used := initial_nails * kitchen_percent / 100
  let remaining_after_kitchen := initial_nails - kitchen_used
  let fence_used := remaining_after_kitchen * fence_percent / 100
  let final_remaining := remaining_after_kitchen - fence_used
  final_remaining

theorem nails_remaining_proof :
  remaining_nails 400 30 70 = 84 := by
  sorry

end nails_remaining_proof_l724_724254


namespace pile_limit_exists_l724_724053

noncomputable def log_floor (b x : ℝ) : ℤ :=
  Int.floor (Real.log x / Real.log b)

theorem pile_limit_exists (k : ℝ) (hk : k < 2) : ∃ Nk : ℤ, 
  Nk = 2 * (log_floor (2 / k) 2 + 1) := 
  by
    sorry

end pile_limit_exists_l724_724053


namespace chord_intersection_sum_l724_724523

theorem chord_intersection_sum 
(A B C A' B' C' S : ℝ)
(hAz: A = 6) (hBz: B = 3) (hCz: C = 2) 
(vol_ratio: 2 / 9 = (volume_of_simplex S A B C) / (volume_of_simplex S A' B' C')) :
(A' = 3) ∧ (B' = 6) ∧ (C' = 9) ∧ (A' + B' + C' = 18) :=
by
  sorry

end chord_intersection_sum_l724_724523


namespace option_A_not_algebraic_expression_option_B_is_algebraic_expression_option_C_is_algebraic_expression_option_D_is_algebraic_expression_l724_724219

def is_algebraic_expression (e : Expr) : Prop :=
  match e with
  | Expr.sqrt x => x ≥ -2
  | Expr.const c => True
  | Expr.div num denom => denom ≠ 0
  | _ => False

inductive Expr
| sqrt   : Expr → Expr
| const  : ℝ → Expr
| div    : Expr → Expr → Expr

theorem option_A_not_algebraic_expression :
  ¬ is_algebraic_expression (Expr.const 5 + Expr.const 8 = 7) := sorry

theorem option_B_is_algebraic_expression :
  is_algebraic_expression (Expr.sqrt (Expr.const 2 + Expr.const x)) := sorry

theorem option_C_is_algebraic_expression :
  is_algebraic_expression (Expr.const 2022) := sorry

theorem option_D_is_algebraic_expression :
  is_algebraic_expression (Expr.div (Expr.const b + Expr.const 2) (Expr.const 3 * Expr.const a - 1)) := sorry

end option_A_not_algebraic_expression_option_B_is_algebraic_expression_option_C_is_algebraic_expression_option_D_is_algebraic_expression_l724_724219


namespace negation_proof_l724_724878

theorem negation_proof :
  ¬ (∀ x : ℝ, 0 < x ∧ x < (π / 2) → x > Real.sin x) ↔ 
  ∃ x : ℝ, 0 < x ∧ x < (π / 2) ∧ x ≤ Real.sin x := 
sorry

end negation_proof_l724_724878


namespace find_ff_neg1_l724_724719

def f : ℝ → ℝ :=
  λ x, if x > 0 then Real.sqrt x else (x + 1 / 2) ^ 4

theorem find_ff_neg1 :
  f (f (-1)) = 1 / 4 := by
  -- Proof omitted
  sorry

end find_ff_neg1_l724_724719


namespace candy_left_l724_724186

theorem candy_left (total_candy : ℕ) (ate_each : ℕ) : total_candy = 68 → ate_each = 4 → total_candy - 2 * ate_each = 60 :=
by
  intros h1 h2
  rw [h1, h2]
  dsimp
  norm_num
  done

end candy_left_l724_724186


namespace hyperbola_eccentricity_cond_l724_724592

def hyperbola_eccentricity_condition (m : ℝ) : Prop :=
  let a := Real.sqrt m
  let b := Real.sqrt 3
  let c := Real.sqrt (m + 3)
  let e := 2
  (e * e) = (c * c) / (a * a)

theorem hyperbola_eccentricity_cond (m : ℝ) :
  hyperbola_eccentricity_condition m ↔ m = 1 :=
by
  sorry

end hyperbola_eccentricity_cond_l724_724592


namespace universal_proposition_is_B_l724_724958

theorem universal_proposition_is_B :
  (∀ n : ℤ, (2 * n % 2 = 0)) = True :=
sorry

end universal_proposition_is_B_l724_724958


namespace enclosing_circle_radius_l724_724556

/--
If three circles each with a radius of 1 are placed such that each circle touches the other two circles, but none of the circles overlap, then the exact value of the radius of the smallest circle that will enclose all three circles is \( 1 + \frac{2}{\sqrt{3}} \).
-/
theorem enclosing_circle_radius (r : ℝ) (h : ∀ (A B C O : Point),
  (dist A B = 2) ∧ (dist B C = 2) ∧ (dist C A = 2) ∧
  (dist O A = r - 1) ∧ (dist O B = r - 1) ∧ (dist O C = r - 1))
  : r = 1 + 2 / Real.sqrt 3 :=
sorry

end enclosing_circle_radius_l724_724556


namespace decimal_to_fraction_l724_724967

theorem decimal_to_fraction :
  (368 / 100 : ℚ) = (92 / 25 : ℚ) := by
  sorry

end decimal_to_fraction_l724_724967


namespace tangent_ellipse_hyperbola_l724_724636

-- Definitions of the curves
def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y+3)^2 = 1

-- Condition for tangency: the curves must meet and the discriminant must be zero
noncomputable def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Prove the given curves are tangent at some x and y for m = 8/9
theorem tangent_ellipse_hyperbola : 
    (∃ x y : ℝ, ellipse x y ∧ hyperbola x y (8 / 9)) ∧ 
    quadratic_discriminant ((8 / 9) + 9) (6 * (8 / 9)) ((-8/9) * (8 * (8/9)) - 8) = 0 :=
sorry

end tangent_ellipse_hyperbola_l724_724636


namespace minimal_surface_area_height_l724_724834

noncomputable def height_of_box {x : ℝ} (hx : x ≥ 10/3) : ℝ :=
  x + 5

theorem minimal_surface_area_height :
  ∃ x : ℝ, x ≥ 10/3 ∧ height_of_box (x := x) (hx := by linarith) = 25/3 :=
begin
  use 10/3,
  split,
  { linarith, },
  { dsimp [height_of_box],
    norm_num }
end

end minimal_surface_area_height_l724_724834


namespace bridge_length_correct_l724_724271

noncomputable def train_length : ℝ := 110
noncomputable def train_speed_km_per_hr : ℝ := 72
noncomputable def crossing_time : ℝ := 12.399008079353651

-- converting train speed from km/hr to m/s
noncomputable def train_speed_m_per_s : ℝ := train_speed_km_per_hr * (1000 / 3600)

-- total length the train covers to cross the bridge
noncomputable def total_length : ℝ := train_speed_m_per_s * crossing_time

-- length of the bridge
noncomputable def bridge_length : ℝ := total_length - train_length

theorem bridge_length_correct :
  bridge_length = 137.98 :=
by 
  sorry

end bridge_length_correct_l724_724271


namespace sum_of_prime_factors_2145_l724_724211

theorem sum_of_prime_factors_2145 : 
  let prime_factors (n : ℕ) := [3, 5, 11, 13] in
  (prime_factors 2145).sum = 32 := 
by 
  sorry

end sum_of_prime_factors_2145_l724_724211


namespace plan_A_per_minute_charge_l724_724241

-- Definition of charges based on the conditions
def plan_A_charge (x : ℝ) (t : ℝ) : ℝ :=
  if t <= 5 then 0.60
  else 0.60 + (t - 5) * x

def plan_B_charge (t : ℝ) : ℝ := t * 0.08

-- Given conditions
def call_duration : ℝ := 14.999999999999996
def cost_under_planA (x : ℝ) : ℝ := plan_A_charge x call_duration
def cost_under_planB : ℝ := plan_B_charge call_duration

-- Lean proof statement
theorem plan_A_per_minute_charge :
  ∃ x : ℝ, cost_under_planA x = cost_under_planB → x = 0.06 :=
by
  sorry

end plan_A_per_minute_charge_l724_724241


namespace geometric_sequence_arithmetic_sum_condition_l724_724014

variable {a_1 q : ℝ}
variable {n : ℕ}

-- Definition of the sum of the first n terms of a geometric sequence
def S (n : ℕ) : ℝ := a_1 * (1 - q^n) / (1 - q)

-- Hypothesis that S_{n+1}, S_n, and S_{n+2} form an arithmetic sequence
theorem geometric_sequence_arithmetic_sum_condition (hq : q ≠ 1) (h : S (n+1) + S (n+1) = 2 * S (n+1)) : q = -1 :=
sorry

end geometric_sequence_arithmetic_sum_condition_l724_724014


namespace solve_quadratic_roots_l724_724918

theorem solve_quadratic_roots (x : ℝ) : (x - 3) ^ 2 = 3 - x ↔ x = 3 ∨ x = 2 :=
by
  sorry

end solve_quadratic_roots_l724_724918


namespace min_typeA_buses_l724_724935

theorem min_typeA_buses (capacityA capacityB totalPeople totalBuses : ℕ) (h₁ : capacityA = 45)
  (h₂ : capacityB = 30) (h₃ : totalPeople = 300) (h₄ : totalBuses = 8) :
  ∃ x : ℕ, x ≥ 4 ∧ 45 * x + 30 * (8 - x) ≥ 300 :=
by
  use (4 : ℕ)
  split
  · exact le_refl 4
  · simp [Nat.mul_sub_left_distrib]
    linarith

end min_typeA_buses_l724_724935


namespace repeating_decimal_fraction_sum_l724_724666

theorem repeating_decimal_fraction_sum :
  let y := 1.4747474747 in
  let frac := 146/99 in
  (146 + 99) = 245 :=
by
  sorry

end repeating_decimal_fraction_sum_l724_724666


namespace last_digit_8_is_last_to_appear_l724_724117

def tribonacci_sequence : ℕ → ℕ
| 0 := 2
| 1 := 3
| 2 := 4
| (n+3) := tribonacci_sequence n + tribonacci_sequence (n+1) + tribonacci_sequence (n+2)

def last_digit (n: ℕ) : ℕ := (tribonacci_sequence n) % 10

theorem last_digit_8_is_last_to_appear :
  (∀ d: ℕ, d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] → (∃ n: ℕ, last_digit n = d)) ∧
  (¬ ∃ n: ℕ, last_digit n = 8 → ∀ m: ℕ, last_digit (n + m) ≠ 8) := sorry

end last_digit_8_is_last_to_appear_l724_724117


namespace triangle_DM_eq_DN_l724_724021

theorem triangle_DM_eq_DN
  (A B C D M N : Type)
  [HasPoint A] [HasPoint B] [HasPoint C] [HasPoint D] [HasPoint M] [HasPoint N]
  (triangle_ABC : Triangle A B C)
  (D_on_BC : OnLine D B C)
  (AD_bisects_ABC : IsAngleBisector A D B C)
  (b_line : LineThrough B)
  (c_line : LineThrough C)
  (parallel_bc : ParallelLines b_line c_line)
  (equidistant_from_A : ∀ P, OnLine P b_line → ∃ Q, OnLine Q c_line ∧ (Distance A P = Distance A Q))
  (AB_bisects_DM : IsSegmentBisector A B D M)
  (AC_bisects_DN : IsSegmentBisector A C D N) :
  Distance D M = Distance D N :=
sorry

end triangle_DM_eq_DN_l724_724021


namespace exists_composite_nm_plus_one_l724_724502

theorem exists_composite_nm_plus_one (n : ℕ) : ∃ m : ℕ, m = n + 2 ∧ ¬nat.prime (n * m + 1) :=
by
  sorry

end exists_composite_nm_plus_one_l724_724502


namespace widget_production_l724_724114

theorem widget_production (p q r s t : ℕ) :
  (s * q * t) / (p * r) = (sqt / pr) := 
sorry

end widget_production_l724_724114


namespace b_charges_l724_724225

theorem b_charges (total_cost : ℕ) (a_hours b_hours c_hours : ℕ)
  (h_total_cost : total_cost = 720)
  (h_a_hours : a_hours = 9)
  (h_b_hours : b_hours = 10)
  (h_c_hours : c_hours = 13) :
  (total_cost * b_hours / (a_hours + b_hours + c_hours)) = 225 :=
by
  sorry

end b_charges_l724_724225


namespace probability_obtuse_angle_l724_724078

theorem probability_obtuse_angle (A B C D E F : Point)
(hA : A = (0, 3))
(hB : B = (5, 0))
(hC : C = (2 * π + 2, 0))
(hD : D = (2 * π + 2, 5))
(hE : E = (0, 5))
(hF : F = (3, 5))
(H_in_hex : ∀ Q : Point, Q ∈ hexagon A B C D E F)
(R_inside_hex : semicircle_center_midpoint_radius A B (√34) ⊆ hexagon A B C D E F) :
let area_hexagon := 10 * π + 30,
    area_semicircle := 17 * π in
probability_angle_obtuse A B Q = area_semicircle / area_hexagon :=
by
sufficient_assumptions -- insert the required assumptions here
sorry    -- to be filled in with the actual proof

end probability_obtuse_angle_l724_724078


namespace reciprocal_of_neg_three_l724_724156

theorem reciprocal_of_neg_three : ∃ (x : ℚ), (-3 * x = 1) ∧ (x = -1 / 3) :=
by
  use (-1 / 3)
  split
  . rw [mul_comm]
    norm_num 
  . norm_num

end reciprocal_of_neg_three_l724_724156


namespace ceil_minus_floor_eq_one_imp_ceil_minus_x_l724_724855

variable {x : ℝ}

theorem ceil_minus_floor_eq_one_imp_ceil_minus_x (H : ⌈x⌉ - ⌊x⌋ = 1) : ∃ (n : ℤ) (f : ℝ), (x = n + f) ∧ (0 < f) ∧ (f < 1) ∧ (⌈x⌉ - x = 1 - f) := sorry

end ceil_minus_floor_eq_one_imp_ceil_minus_x_l724_724855


namespace find_d_l724_724641

theorem find_d
  (a b c d : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_c_pos : c > 0)
  (h_d_pos : d > 0)
  (h_max : a * 1 + d = 5)
  (h_min : a * (-1) + d = -3) :
  d = 1 := 
sorry

end find_d_l724_724641


namespace range_of_a_l724_724916
noncomputable def exponential_quadratic (a : ℝ) : Prop :=
  ∃ x : ℝ, 0 < x ∧ (1/4)^x + (1/2)^(x-1) + a = 0

theorem range_of_a (a : ℝ) : exponential_quadratic a ↔ -3 < a ∧ a < 0 :=
sorry

end range_of_a_l724_724916


namespace find_a_value_l724_724518

-- Define the conditions
def inverse_variation (a b : ℝ) : Prop := ∃ k : ℝ, a * b^3 = k

-- Define the proof problem
theorem find_a_value
  (a b : ℝ)
  (h1 : inverse_variation a b)
  (h2 : a = 4)
  (h3 : b = 1) :
  ∃ a', a' = 1 / 2 ∧ inverse_variation a' 2 := 
sorry

end find_a_value_l724_724518


namespace candy_left_l724_724185

theorem candy_left (total_candy : ℕ) (ate_each : ℕ) : total_candy = 68 → ate_each = 4 → total_candy - 2 * ate_each = 60 :=
by
  intros h1 h2
  rw [h1, h2]
  dsimp
  norm_num
  done

end candy_left_l724_724185


namespace pages_torn_l724_724913

theorem pages_torn (n : ℕ) (H1 : n = 185) (H2 : ∃ m, m = 518 ∧ (digits 10 m = digits 10 n) ∧ (m % 2 = 0)) : 
  ∃ k, k = ((518 - 185 + 1) / 2) ∧ k = 167 :=
by sorry

end pages_torn_l724_724913


namespace find_n_for_arithmetic_sequence_l724_724693

variable {a : ℕ → ℤ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (a₁ : ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a n = a₁ + n * d

theorem find_n_for_arithmetic_sequence (h_arith : is_arithmetic_sequence a (-1) 2)
  (h_nth_term : ∃ n : ℕ, a n = 15) : ∃ n : ℕ, n = 9 :=
by
  sorry

end find_n_for_arithmetic_sequence_l724_724693


namespace square_digits_conditions_l724_724922

-- Define the conditions and the final conclusion based on the problem statement

def S : ℕ := (10^101 + 5) / 3

theorem square_digits_conditions : 
  ∃ S : ℕ, 
    (S = (10^101 + 5) / 3) ∧ 
    (nat.digits 10 (S ^ 2)).length = 202 ∧ 
    (nat.take 100 (nat.digits 10 (S ^ 2))) = [1,1,...,1] ∧ 
    (nat.drop 100 (nat.digits 10 (S ^ 2))) = [2,2,...,2] ∧ 
    (S ^ 2 % 10 = 5) :=
begin
  use S,
  split,
  { exact rfl },
  sorry  -- Proof needed to verify all conditions
end

end square_digits_conditions_l724_724922


namespace pizza_area_increase_l724_724118

/-- Given a circular pizza with radius 7 inches and a square pizza with side length 5 inches,
prove that the percentage increase in the area of the circular pizza over the square pizza 
is closest to 515. --/
theorem pizza_area_increase (r s : ℝ) (π : ℝ) (h₁ : r = 7) (h₂ : s = 5) (h₃ : π = 3.14) :
  let A_circular := π * r^2,
      A_square := s^2,
      A_diff := A_circular - A_square,
      percentage_increase := (A_diff / A_square) * 100 in
  abs (percentage_increase - 515) < 1 :=
by {
  sorry
}

end pizza_area_increase_l724_724118


namespace old_clock_slow_l724_724128

theorem old_clock_slow (minute_hand_speed hour_hand_speed : ℝ)
  (overlap_old_clock duration_standard : ℝ) :
  (minute_hand_speed = 6) →
  (hour_hand_speed = 0.5) →
  (overlap_old_clock = 66) →
  (duration_standard = 24 * 60) →
  let relative_speed := minute_hand_speed - hour_hand_speed in
  let time_to_coincide := 360 / relative_speed in
  let intervals_per_day := duration_standard / time_to_coincide in
  let new_duration := intervals_per_day * overlap_old_clock in
  new_duration - duration_standard = 12 :=
sorry

end old_clock_slow_l724_724128


namespace minimum_fuel_consumption_l724_724561

def fuel_consumption_per_hour (x : ℝ) : ℝ :=
  (1 / 120000) * x^3 - (1 / 50) * x + (18 / 5)

noncomputable def fuel_consumption (x : ℝ) : ℝ :=
  fuel_consumption_per_hour x * (100 / x)

theorem minimum_fuel_consumption :
  (0 < x ∧ x ≤ 100) →
  f x = (1 / 120000) * x^3 - (1 / 50) * x + (18 / 5) * (100 / x) →
  f 60 = 7 :=
begin
  -- proof here
  sorry
end

end minimum_fuel_consumption_l724_724561


namespace locus_of_C_locus_of_M_l724_724774

-- Part (1): Equation of the Locus of the Right Angle Vertex C
theorem locus_of_C (x y : ℝ) :
  (A : ℝ × ℝ) := (-1, 0)
  (B : ℝ × ℝ) := (3, 0)
  (H : (x - 1)^2 + y^2 = 4) : 
  (C x y) → 
  (x ≠ 3) → 
  (x ≠ -1) → 
  (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 :=
sorry

-- Part (2): Equation of the Locus of the Midpoint M
theorem locus_of_M (x y : ℝ) :
  (A : ℝ × ℝ) := (-1, 0)
  (B : ℝ × ℝ) := (3, 0)
  (H : (x - 2)^2 + y^2 = 1) : 
  (M (x y)) →
  (x ≠ 3) → 
  (x ≠ 1) → 
  (2*M.1 - 3)(2 * M.2):
sorry

end locus_of_C_locus_of_M_l724_724774


namespace sheets_torn_out_l724_724894

-- Define the conditions as given in the problem
def first_torn_page : Nat := 185
def last_torn_page : Nat := 518
def pages_per_sheet : Nat := 2

-- Calculate the total number of pages torn out
def total_pages_torn_out : Nat :=
  last_torn_page - first_torn_page + 1

-- Calculate the number of sheets torn out
def number_of_sheets_torn_out : Nat :=
  total_pages_torn_out / pages_per_sheet

-- Prove that the number of sheets torn out is 167
theorem sheets_torn_out :
  number_of_sheets_torn_out = 167 :=
by
  unfold number_of_sheets_torn_out total_pages_torn_out
  rw [Nat.sub_add_cancel (Nat.le_of_lt (Nat.lt_of_le_of_ne
    (Nat.le_add_left _ _) (Nat.ne_of_lt (Nat.lt_add_one 184))))]
  rw [Nat.div_eq_of_lt (Nat.lt.base 333)] 
  sorry -- proof steps are omitted

end sheets_torn_out_l724_724894


namespace cube_surface_area_increase_l724_724217

theorem cube_surface_area_increase (s : ℝ) :
  let original_surface_area := 6 * s^2
  let new_side_length := 1.8 * s
  let new_surface_area := 6 * (new_side_length)^2
  let percentage_increase := (new_surface_area - original_surface_area) / original_surface_area * 100
  percentage_increase = 1844 :=
by
  unfold original_surface_area
  unfold new_side_length
  unfold new_surface_area
  unfold percentage_increase
  sorry

end cube_surface_area_increase_l724_724217


namespace remy_gallons_l724_724084

noncomputable def gallons_used (R : ℝ) : ℝ :=
  let remy := 3 * R + 1
  let riley := (R + remy) - 2
  let ronan := riley / 2
  R + remy + riley + ronan

theorem remy_gallons : ∃ R : ℝ, gallons_used R = 60 ∧ (3 * R + 1) = 18.85 :=
by
  sorry

end remy_gallons_l724_724084


namespace coefficient_of_x3_in_expansion_l724_724525

theorem coefficient_of_x3_in_expansion :
  (∀ (x : ℝ), (Polynomial.coeff ((Polynomial.C x - 1)^5) 3) = 10) :=
by
  sorry

end coefficient_of_x3_in_expansion_l724_724525


namespace cos_five_theta_l724_724749

theorem cos_five_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : 
  Real.cos (5 * θ) = (125 * Real.sqrt 15 - 749) / 1024 := 
  sorry

end cos_five_theta_l724_724749


namespace max_incircle_circumcircle_ratio_l724_724564

theorem max_incircle_circumcircle_ratio (c : ℝ) (α : ℝ) 
  (hα : 0 < α ∧ α < π / 2) :
  let a := c * Real.cos α
  let b := c * Real.sin α
  let R := c / 2
  let r := (a + b - c) / 2
  (r / R <= Real.sqrt 2 - 1) :=
by
  sorry

end max_incircle_circumcircle_ratio_l724_724564


namespace trader_car_profit_l724_724270

theorem trader_car_profit
  (P : ℝ) -- Original price of the car
  (h1 : 0 < P) -- The price should be positive
  (P1 := 0.80 * P) -- The purchase price after 20% discount
  (depreciation_factor : ℝ := 0.95) -- Depreciation factor per year
  (P_depreciated := depreciation_factor^2 * P1) -- Value after 2 years depreciation
  (tax_rate : ℝ := 0.03)
  (C := P_depreciated + tax_rate * P1) -- Cost after adding tax
  (increase_factor : ℝ := 1.70)
  (S := increase_factor * C) -- Selling price after 70% increase
  : (S / P - 1) * 100 = 27.03 :=
begin
  sorry -- Proof goes here
end

end trader_car_profit_l724_724270


namespace f_is_odd_f_is_increasing_range_of_m_l724_724398

noncomputable def f (x : ℝ) := 2^x - (1 / 2^x)

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  intro x
  unfold f
  -- proof goes here
  sorry

theorem f_is_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 :=
by
  intros x1 x2 h
  unfold f
  -- proof goes here
  sorry

theorem range_of_m : ∀ m : ℝ, (m > 1 ∧ m ≤ 3/2) ↔ (∀ x ∈ Ioo (-1 : ℝ) 1, f (1-m) + f (2-m) ≥ 0) :=
by
  intro m
  unfold f
  -- proof goes here
  sorry

end f_is_odd_f_is_increasing_range_of_m_l724_724398


namespace distance_travelled_l724_724625

theorem distance_travelled
  (d : ℝ)                   -- distance in kilometers
  (train_speed : ℝ)         -- train speed in km/h
  (ship_speed : ℝ)          -- ship speed in km/h
  (time_difference : ℝ)     -- time difference in hours
  (h1 : train_speed = 48)
  (h2 : ship_speed = 60)
  (h3 : time_difference = 2) :
  d = 480 := 
by
  sorry

end distance_travelled_l724_724625


namespace eight_faucets_fill_30_gallon_in_60_seconds_l724_724324

def time_to_fill (gallons: ℕ) (rate_per_faucet: ℕ) (num_faucets: ℕ): ℕ :=
  gallons / (rate_per_faucet * num_faucets) * 60

theorem eight_faucets_fill_30_gallon_in_60_seconds:
  let rate_per_faucet := 15 in
  time_to_fill 30 rate_per_faucet 8 = 60 :=
by
  sorry

end eight_faucets_fill_30_gallon_in_60_seconds_l724_724324


namespace arithmetic_sequence_sum_condition_l724_724832

noncomputable def sum_first_n_terms (a_1 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a_1 + (n * (n - 1)) / 2 * d

theorem arithmetic_sequence_sum_condition (a_1 d : ℤ) :
  sum_first_n_terms a_1 d 3 = 3 →
  sum_first_n_terms a_1 d 6 = 15 →
  (a_1 + 9 * d) + (a_1 + 10 * d) + (a_1 + 11 * d) = 30 :=
by
  intros h1 h2
  sorry

end arithmetic_sequence_sum_condition_l724_724832


namespace true_propositions_1_and_3_l724_724934

theorem true_propositions_1_and_3 :
  (∀ x y : ℝ, (x + y = 0) ↔ (x = -y)) ∧
  ¬(∀ (Δ1 Δ2 : triangle), (congruent Δ1 Δ2) ↔ (area Δ1 = area Δ2)) ∧
  (∀ q : ℝ, (q > 1) ↔ (¬ (∃ x : ℝ, x^2 + 2 * x + q = 0))) ∧
  ¬(∀ (Δ : triangle), (has_two_acute_angles Δ) ↔ (is_right_triangle Δ)) :=
begin
  sorry
end

end true_propositions_1_and_3_l724_724934


namespace loan_duration_l724_724610

theorem loan_duration (P R SI : ℝ) (hP : P = 20000) (hR : R = 12) (hSI : SI = 7200) : 
  ∃ T : ℝ, T = 3 :=
by
  sorry

end loan_duration_l724_724610


namespace pile_limit_exists_l724_724055

noncomputable def log_floor (b x : ℝ) : ℤ :=
  Int.floor (Real.log x / Real.log b)

theorem pile_limit_exists (k : ℝ) (hk : k < 2) : ∃ Nk : ℤ, 
  Nk = 2 * (log_floor (2 / k) 2 + 1) := 
  by
    sorry

end pile_limit_exists_l724_724055


namespace perimeter_triangle_ADA_l724_724354

open Real

noncomputable def eccentricity : ℝ := 1 / 2

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

noncomputable def foci_distance (a b : ℝ) : ℝ :=
  (a^2 - b^2).sqrt

noncomputable def line_passing_through_focus_perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  sorry

noncomputable def distance_de (d e : ℝ) : ℝ := 6

theorem perimeter_triangle_ADA
  (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : foci_distance a b = c)
  (h4 : eccentricity * a = c) (h5 : distance_de 6 6) :
  4 * a = 13 :=
by sorry

end perimeter_triangle_ADA_l724_724354


namespace part1_part2_l724_724400

open Real

def f (x : ℝ) : ℝ := |x - 5| - |x - 2|

theorem part1 (m : ℝ) : (∃ x : ℝ, f x ≤ m) ↔ m ≥ -3 := 
sorry

theorem part2 : {x : ℝ | x^2 - 8 * x + 15 + f x ≤ 0} = {x : ℝ | 5 - sqrt 3 ≤ x ∧ x ≤ 6} :=
sorry

end part1_part2_l724_724400


namespace parallelogram_area_l724_724670

theorem parallelogram_area (base height : ℕ) (h1 : base = 24) (h2 : height = 16) : base * height = 384 := 
by
  rw [h1, h2]
  norm_num
  -- Sorry to skip the detailed proof calculation steps

-- The noncomputable part is not needed here since base*height is a concrete calculation

end parallelogram_area_l724_724670


namespace positive_integers_satisfy_condition_l724_724415

theorem positive_integers_satisfy_condition :
  ∃ (S : Finset ℕ), S.card = 6 ∧ ∀ n ∈ S, (n - 40000) % 80 = 0 ∧ (n + 2000) / 80 = int.floor (Real.sqrt n) :=
by
  sorry

end positive_integers_satisfy_condition_l724_724415


namespace reciprocal_of_neg3_l724_724159

theorem reciprocal_of_neg3 : ∃ x : ℝ, -3 * x = 1 ∧ x = -1/3 :=
by
  sorry

end reciprocal_of_neg3_l724_724159


namespace positive_solution_conditions_l724_724109

theorem positive_solution_conditions
  (x y z a m : ℝ)
  (h1 : x + y - z = 2a)
  (h2 : x^2 + y^2 = z^2)
  (h3 : m * (x + y) = x * y)
  (h4 : x > 0) (h5 : y > 0) (h6 : z > 0) :
  (a > 0) ∧ (m >= a / 2 * (2 + Real.sqrt 2)) ∧ (m <= 2 * a) :=
sorry

end positive_solution_conditions_l724_724109


namespace sqrt_inequality_l724_724080

theorem sqrt_inequality : sqrt 3 + sqrt 7 < 2 * sqrt 5 :=
sorry

end sqrt_inequality_l724_724080


namespace reciprocal_of_neg_three_l724_724140

theorem reciprocal_of_neg_three : -3 * (-1 / 3) = 1 := 
by
  sorry

end reciprocal_of_neg_three_l724_724140


namespace triangle_sides_and_area_l724_724497

variable {a r varrho b c t : ℝ}

-- Given conditions
axiom a_value : a = 6
axiom r_value : r = 5
axiom varrho_value : varrho = 2

-- Prove that
theorem triangle_sides_and_area : 
  (b = 8 ∧ c = 10 ∧ t = 24) ∧ 
  let s := (a + b + c) / 2 in 
  let area_formula := r * t = (a * b * c) / 4 in 
  let semiperimeter := t = varrho * s in 
  let herons_formula := t^2 = s * (s - a) * (s - b) * (s - c) in 
  a = 6 ∧ r = 5 ∧ varrho = 2 :=
by
  sorry

end triangle_sides_and_area_l724_724497


namespace min_value_fraction_l724_724822

noncomputable theory

open Real

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + 3 * b = 1) :
  (2 / a + 3 / b) ≥ 25 :=
sorry

end min_value_fraction_l724_724822


namespace reciprocal_of_neg_three_l724_724151

theorem reciprocal_of_neg_three : ∃ x : ℚ, (-3) * x = 1 ∧ x = (-1) / 3 := sorry

end reciprocal_of_neg_three_l724_724151


namespace quadratic_function_expression_l724_724765

def g (x : ℝ) : ℝ := 3 * x^2 - 2 * x

theorem quadratic_function_expression :
  (g 1 = 1) ∧ (g (-1) = 5) ∧ (g 0 = 0) :=
by
  split;
  sorry

end quadratic_function_expression_l724_724765


namespace abes_age_l724_724543

theorem abes_age (A : ℕ) (h : A + (A - 7) = 29) : A = 18 :=
by
  sorry

end abes_age_l724_724543


namespace dress_design_count_l724_724250

-- Definitions of the given conditions
def number_of_colors : Nat := 4
def number_of_patterns : Nat := 5

-- Statement to prove the total number of unique dress designs
theorem dress_design_count :
  number_of_colors * number_of_patterns = 20 := by
  sorry

end dress_design_count_l724_724250


namespace find_n_and_p_l724_724825

theorem find_n_and_p (p : ℝ → ℝ) (n : ℕ)
  (h_deg : ∀ x, p x ≠ 0 → degree p = 2 * n)
  (h_zeros : ∀ k, 0 ≤ k ∧ k ≤ n → p(2 * k) = 0)
  (h_twos : ∀ k, 0 ≤ k ∧ k < n → p(2 * k + 1) = 2)
  (h_final : p(2 * n + 1) = -30) :
  n = 2 ∧ p = λ x, -2 * x^2 + 4 * x :=
by
  sorry

end find_n_and_p_l724_724825


namespace problem_statement_l724_724279

theorem problem_statement : 
  let A := 10^6 + 10^6
  let B := (2^10 * 5^10)^2
  let C := (2 * 5 * 10^5) * 10^6
  let D := (10^3)^3 in
  A ≠ 10^12 ∧ B ≠ 10^12 ∧ C = 10^12 ∧ D ≠ 10^12 :=
by
  let A := 10^6 + 10^6
  let B := (2^10 * 5^10)^2
  let C := (2 * 5 * 10^5) * 10^6
  let D := (10^3)^3
  sorry

end problem_statement_l724_724279


namespace solve_diff_eqn_l724_724313

noncomputable def diff_eqn (y : ℝ → ℝ) : Prop :=
  ∀ x, deriv y x = 2 + y x

def initial_condition (y : ℝ → ℝ) : Prop :=
  y 0 = 3

def particular_solution (y : ℝ → ℝ) : Prop :=
  y = λ x, 5 * Real.exp x - 2

theorem solve_diff_eqn :
  ∃ y : ℝ → ℝ, diff_eqn y ∧ initial_condition y ∧ particular_solution y :=
sorry

end solve_diff_eqn_l724_724313


namespace old_clock_duration_l724_724130

noncomputable def hour_hand_speed : ℝ := 0.5 -- degrees per minute
noncomputable def minute_hand_speed : ℝ := 6 -- degrees per minute
noncomputable def coincidence_period : ℝ := 66 -- minutes

theorem old_clock_duration:
  let T := 720 / 11 in 
  let standard_duration := 24 * 60 in 
  let old_clock_duration_in_standard := (standard_duration * coincidence_period) / T in
  old_clock_duration_in_standard = 1452 :=
sorry

end old_clock_duration_l724_724130


namespace monikaTotalSpending_l724_724483

-- Define the conditions as constants
def mallSpent : ℕ := 250
def movieCost : ℕ := 24
def movieCount : ℕ := 3
def beanCost : ℚ := 1.25
def beanCount : ℕ := 20

-- Define the theorem to prove the total spending
theorem monikaTotalSpending : mallSpent + (movieCost * movieCount) + (beanCost * beanCount) = 347 :=
by
  sorry

end monikaTotalSpending_l724_724483


namespace focus_of_parabola_l724_724295

theorem focus_of_parabola : 
  ∃(h k : ℚ), ((∀ x : ℚ, -2 * x^2 - 6 * x + 1 = -2 * (x + 3 / 2)^2 + 11 / 2) ∧ 
  (∃ a : ℚ, (a = -2 / 8) ∧ (h = -3/2) ∧ (k = 11/2 + a)) ∧ 
  (h, k) = (-3/2, 43 / 8)) :=
sorry

end focus_of_parabola_l724_724295


namespace slope_angle_of_tangent_line_l724_724540

theorem slope_angle_of_tangent_line 
  (x : ℝ) (y : ℝ) (curve : ℝ → ℝ)
  (point_x : ℝ) (point_y : ℝ)
  (slope : ℝ) (angle : ℝ) :
  (curve x = (1 / 3) * x^3 - 2) →
  (point_x = 1) →
  (point_y = -5 / 3) →
  (y = curve point_x) →
  (slope = 1) →
  (angle = 45) :=
by
  intros h_curve h_point_x h_point_y h_y h_slope
  have h_deriv : ∀ x, derivative (λ x, (1 / 3) * x^3 - 2) x = x^2 := sorry
  have h_slope_at_point : slope = (derivative (λ x, (1 / 3) * x^3 - 2) point_x) := sorry
  have h_slope_eq_1 : slope = 1 := sorry
  have h_angle_45 : angle = 45 := sorry
  exact h_angle_45

end slope_angle_of_tangent_line_l724_724540


namespace sum_first_15_terms_l724_724444

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2

-- Conditions
def a_7 := 1
def a_9 := 5

-- Prove that S_15 = 45
theorem sum_first_15_terms : 
  ∃ (a d : ℤ), 
    (arithmetic_sequence a d 7 = a_7) ∧ 
    (arithmetic_sequence a d 9 = a_9) ∧ 
    (sum_first_n_terms a d 15 = 45) :=
sorry

end sum_first_15_terms_l724_724444


namespace muffin_mix_buyers_l724_724240

def num_buyers : ℕ := 100
def cake_mix_buyers : ℕ := 50
def both_cake_muffin_buyers : ℕ := 15
def neither_cake_muffin_buyers : ℕ := (0.25 * 100).toNat

theorem muffin_mix_buyers : ℕ :=
  num_buyers - (cake_mix_buyers - both_cake_muffin_buyers) - neither_cake_muffin_buyers

example : muffin_mix_buyers = 40 :=
  by sorry

end muffin_mix_buyers_l724_724240


namespace incorrect_proposition_l724_724635

-- Definition of propositions based on given conditions
def prop_A : Prop := ∀ (P₁ P₂ : Plane) (l : Line), P₁ ∥ l → P₂ ∥ l → P₁ ∥ P₂
def prop_B : Prop := ∀ (P₁ P₂ P₃ : Plane), P₁ ∥ P₃ → P₂ ∥ P₃ → P₁ ∥ P₂
def prop_C : Prop := ∀ (l : Line) (P₁ P₂ : Plane), P₁ ∥ P₂ → angle_between l P₁ = angle_between l P₂
def prop_D : Prop := ∀ (l : Line) (P₁ P₂ : Plane), P₁ ∥ P₂ → l ⊥ P₁ → l ⊥ P₂

-- The main theorem stating Proposition A is incorrect
theorem incorrect_proposition (hA : prop_A) (hB : prop_B) (hC : prop_C) (hD : prop_D) : ¬ prop_A := 
    by 
    sorry -- Proof is not required

end incorrect_proposition_l724_724635


namespace minimum_shots_required_l724_724067

noncomputable def minimum_shots_to_sink_boat : ℕ := 4000

-- Definitions for the problem conditions.
structure Boat :=
(square_side : ℕ)
(base1 : ℕ)
(base2 : ℕ)
(rotatable : Bool)

def boat : Boat := { square_side := 1, base1 := 1, base2 := 3, rotatable := true }

def grid_size : ℕ := 100

def shot_covers_triangular_half : Prop := sorry -- Assumption: Define this appropriately

-- Problem statement in Lean 4
theorem minimum_shots_required (boat_within_grid : Bool) : 
  Boat → grid_size = 100 → boat_within_grid → minimum_shots_to_sink_boat = 4000 :=
by
  -- Here you would do the full proof which we assume is "sorry" for now
  sorry

end minimum_shots_required_l724_724067


namespace average_of_last_four_numbers_l724_724119

theorem average_of_last_four_numbers
  (avg_seven : ℝ) (avg_first_three : ℝ) (avg_last_four : ℝ)
  (h1 : avg_seven = 62) (h2 : avg_first_three = 55) :
  avg_last_four = 67.25 := 
by
  sorry

end average_of_last_four_numbers_l724_724119


namespace victoria_donuts_cost_l724_724946

theorem victoria_donuts_cost (n : ℕ) (cost_per_dozen : ℝ) (total_donuts_needed : ℕ) 
  (dozens_needed : ℕ) (actual_total_donuts : ℕ) (total_cost : ℝ) :
  total_donuts_needed ≥ 550 ∧ cost_per_dozen = 7.49 ∧ (total_donuts_needed = 12 * dozens_needed) ∧
  (dozens_needed = Nat.ceil (total_donuts_needed / 12)) ∧ 
  (actual_total_donuts = 12 * dozens_needed) ∧ actual_total_donuts ≥ 550 ∧ 
  (total_cost = dozens_needed * cost_per_dozen) →
  total_cost = 344.54 :=
by
  sorry

end victoria_donuts_cost_l724_724946


namespace arithmetic_mean_divisors_condition_l724_724492

theorem arithmetic_mean_divisors_condition (n : ℕ) : 
  (n ∣ 799) ↔ 
  (∃ (k m : ℕ), m = 1598 ∧ 
                k * n = m / 2 ∧ 
                ∀ i, i ∈ set.range k → 
                     ∑ j in finset.Ico (2 * i * n) ((2 * i + 1) * n), 
                     (j + 1) / n = i + 1) := 
sorry

end arithmetic_mean_divisors_condition_l724_724492


namespace pencil_arrangements_l724_724606

theorem pencil_arrangements (yellow red blue : ℕ) (hy : yellow = 6) (hr : red = 3) (hb : blue = 4) : 
  let total_unrestricted := (13.factorial) / ((6.factorial) * (3.factorial) * (4.factorial))
  let blue_block := (10.factorial) / ((6.factorial) * (3.factorial) * (1.factorial))
  let valid_arrangements := total_unrestricted - blue_block
  in valid_arrangements = 274400 :=
by
  sorry

end pencil_arrangements_l724_724606


namespace find_k_values_l724_724176

theorem find_k_values (k : ℝ) : 
  (∃ (x y : ℝ), x + 2 * y - 1 = 0 ∧ x + 1 = 0 ∧ x + k * y = 0) → 
  (k = 0 ∨ k = 1 ∨ k = 2) ∧
  (k = 0 ∨ k = 1 ∨ k = 2 → ∃ (x y : ℝ), x + 2 * y - 1 = 0 ∧ x + 1 = 0 ∧ x + k * y = 0) :=
by
  sorry

end find_k_values_l724_724176


namespace find_point_B_l724_724783

def line_segment_parallel_to_x_axis (A B : (ℝ × ℝ)) : Prop :=
  A.snd = B.snd

def length_3 (A B : (ℝ × ℝ)) : Prop :=
  abs (A.fst - B.fst) = 3

theorem find_point_B (A B : (ℝ × ℝ))
  (h₁ : A = (3, 2))
  (h₂ : line_segment_parallel_to_x_axis A B)
  (h₃ : length_3 A B) :
  B = (0, 2) ∨ B = (6, 2) :=
sorry

end find_point_B_l724_724783


namespace sign_of_f_given_angle_C_l724_724708

noncomputable def sides : Type := {a b c : ℝ}

theorem sign_of_f_given_angle_C (a b c R r : ℝ) (C : ℝ) (h1 : a ≤ b ∧ b ≤ c) (h2 : 0 < C ∧ C < π) 
    (h3 : R = sorry) (h4 : r = sorry) :
    let f := a + b - 2 * R - 2 * r in
    if C < π / 2 then f > 0
    else if C = π / 2 then f = 0
    else f < 0 :=
begin
    sorry
end

end sign_of_f_given_angle_C_l724_724708


namespace sum_three_smallest_solutions_l724_724316

def fractional_part (x : ℝ) : ℝ := x - (x.floor : ℝ)

def equation (x : ℝ) : Prop :=
  fractional_part x = 1 / (x.floor + 1)

def smallest_positive_solutions : List ℝ :=
  [1.5, 2 + 1/3, 3 + 1/4]

def sum_solutions (sols : List ℝ) : ℝ :=
  sols.foldr (· + ·) 0

theorem sum_three_smallest_solutions :
  sum_solutions smallest_positive_solutions = 85 / 12 := by
  sorry

end sum_three_smallest_solutions_l724_724316


namespace find_number_l724_724995

theorem find_number (x : ℝ) (h : 5020 - 502 / x = 5015) : x = 100.4 :=
by
  sorry

end find_number_l724_724995


namespace simplify_expression_l724_724102

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
  ((x^2 + 1) / (x - 1) - 2 * x / (x - 1)) = x - 1 :=
by
  -- Proof goes here.
  sorry

end simplify_expression_l724_724102


namespace torn_out_sheets_count_l724_724883

theorem torn_out_sheets_count :
  ∃ (sheets : ℕ), (first_page = 185 ∧
                   last_page = 518 ∧
                   pages_torn_out = last_page - first_page + 1 ∧ 
                   sheets = pages_torn_out / 2 ∧
                   sheets = 167) :=
by
  sorry

end torn_out_sheets_count_l724_724883


namespace max_min_sum_eq_two_l724_724685

open Real

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 1

theorem max_min_sum_eq_two (a : ℝ) (h : a > 0) : 
  let M := (λ x, f x).sup (Icc (-a) a)
  let N := (λ x, f x).inf (Icc (-a) a)
  M + N = 2 :=
by
  sorry

end max_min_sum_eq_two_l724_724685


namespace total_distance_bug_travels_l724_724236

theorem total_distance_bug_travels : 
  let start_pos := -3
  let mid_pos := -8
  let end_pos := 7
  abs(mid_pos - start_pos) + abs(end_pos - mid_pos) = 20 := 
by
  let start_pos := -3
  let mid_pos := -8
  let end_pos := 7
  show abs(mid_pos - start_pos) + abs(end_pos - mid_pos) = 20
  sorry

end total_distance_bug_travels_l724_724236


namespace simplify_expression_l724_724094

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
    ((x^2 + 1) / (x - 1)) - (2 * x / (x - 1)) = x - 1 := 
by
    sorry

end simplify_expression_l724_724094


namespace inradius_of_triangle_l724_724915

theorem inradius_of_triangle (P A : ℝ) (hP : P = 40) (hA : A = 50) : 
  ∃ r : ℝ, r = 2.5 ∧ A = r * (P / 2) :=
by
  sorry

end inradius_of_triangle_l724_724915


namespace similar_sizes_bound_l724_724058

theorem similar_sizes_bound (k : ℝ) (hk : k < 2) :
  ∃ (N_k : ℝ), ∀ (A : multiset ℝ), (∀ a ∈ A, a ≤ k * multiset.min A) → 
  A.card ≤ N_k := sorry

end similar_sizes_bound_l724_724058


namespace divisors_count_l724_724823

open BigOperators

def n : ℕ := 2^25 * 3^17

-- We are to prove that the number of positive integer divisors of n^2 that are less than n but do not divide n is 424.
theorem divisors_count (n := 2^25 * 3^17) : 
  (finset.filter (λ d, d < n ∧ ∀ m, d = n * m → m = 1) (finset.divisors (n^2))).card = 424 := 
sorry

end divisors_count_l724_724823


namespace max_n_minus_m_l724_724712

/-- The function defined with given parameters. -/
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b

theorem max_n_minus_m (a b : ℝ) (h1 : -a / 2 = 1)
    (h2 : ∀ x, f x a b ≥ 2)
    (h3 : ∃ m n, (∀ x, f x a b ≤ 6 → m ≤ x ∧ x ≤ n) ∧ (n = 3 ∧ m = -1)) : 
    (∀ m n, (m ≤ n) → (n - m ≤ 4)) :=
by sorry

end max_n_minus_m_l724_724712


namespace trip_correct_graph_l724_724072

-- Define a structure representing the trip
structure Trip :=
  (initial_city_traffic_duration : ℕ)
  (highway_duration_to_mall : ℕ)
  (shopping_duration : ℕ)
  (highway_duration_from_mall : ℕ)
  (return_city_traffic_duration : ℕ)

-- Define the conditions about the trip
def conditions (t : Trip) : Prop :=
  t.shopping_duration = 1 ∧ -- Shopping for one hour
  t.initial_city_traffic_duration < t.highway_duration_to_mall ∧ -- Travel more rapidly on the highway
  t.return_city_traffic_duration < t.highway_duration_from_mall -- Return more rapidly on the highway

-- Define the graph representation of the trip
inductive Graph
| A | B | C | D | E

-- Define the property that graph B correctly represents the trip
def correct_graph (t : Trip) (g : Graph) : Prop :=
  g = Graph.B

-- The theorem stating that given the conditions, the correct graph is B
theorem trip_correct_graph (t : Trip) (h : conditions t) : correct_graph t Graph.B :=
by
  sorry

end trip_correct_graph_l724_724072


namespace segment_count_l724_724846

def number_segments (n : ℕ) (λ : Type → ℕ) (P : Type → Prop) :=
  ∑ λ P - n

theorem segment_count (n : ℕ) (P : Type → Prop) (λ : Type → ℕ) :
  (∀ P, (λ P) = number of lines through P ) →
  (∑ P, λ P ) = ∑ of points of intersection P → -- ensure P passes through all intersection points
  number_segments (n λ P) = -n + (∑ λ P) :=
begin
  sorry
end

end segment_count_l724_724846


namespace quadrilateral_in_graph_l724_724691

open Classical

noncomputable def num_edges (n : ℕ) : ℝ :=
  (1 / 4) * (n * (1 + real.sqrt (4 * n - 3)))

theorem quadrilateral_in_graph (n m : ℕ)
  (h_points_non_collinear : ∀ (A B C : ℝ × ℝ), A ≠ B → B ≠ C → C ≠ A → collinear ℝ {A, B, C} → false)
  (h_edges : m > num_edges n) :
  ∃ (G : SimpleGraph (fin n)), ∃ (C : Set (fin n)), G.Cycle C ∧ C.card = 4 :=
  sorry

end quadrilateral_in_graph_l724_724691


namespace total_distance_is_3_miles_l724_724487

-- Define conditions
def running_speed := 6   -- mph
def walking_speed := 2   -- mph
def running_time := 20 / 60   -- hours
def walking_time := 30 / 60   -- hours

-- Define total distance
def total_distance := (running_speed * running_time) + (walking_speed * walking_time)

theorem total_distance_is_3_miles : total_distance = 3 :=
by
  sorry

end total_distance_is_3_miles_l724_724487


namespace JohnnyTV_percentage_l724_724028

theorem JohnnyTV_percentage (P : ℕ) :
  (LJ_productions_per_year = 220) →
  (five_years_combined = 2475) →
  (johnny_tv_more_percentage LJ_productions_per_year five_years_combined P = 25) :=
by
  assume LJ_productions_per_year_eq: LJ_productions_per_year = 220,
  assume five_years_combined_eq: five_years_combined = 2475,
  sorry

def LJ_productions_per_year : ℕ := 220

def five_years_combined : ℕ := 2475

noncomputable def johnny_tv_more_percentage (LJ_productions_per_year : ℕ) 
    (five_years_combined : ℕ) (P : ℕ) : ℕ :=
  let one_year_combined := (LJ_productions_per_year + LJ_productions_per_year * (100 + P) / 100)
  in if 5 * one_year_combined = five_years_combined
    then P
    else 0

end JohnnyTV_percentage_l724_724028


namespace find_an_find_Tn_l724_724344

open Real

def seq_an (n : ℕ) : ℝ := 
  if n = 0 then 0 else (1 / 4^n)

def seq_condition (n : ℕ) : Prop := 
  (∑ i in Finset.range n, 4^i * (seq_an (i+1))) = n / 4

def seq_bn (n : ℕ) : ℝ := 
  4^n * (seq_an n) / (2 * n + 1)

theorem find_an (n : ℕ) (hn : n > 0) : 
  seq_condition n → seq_an n = (1 / (4^n)) := by 
  sorry

theorem find_Tn (n : ℕ) (hn : n > 0) : 
  (∑ i in Finset.range n, seq_bn i * seq_bn (i + 1)) = (n / (6 * n + 9)) := by 
  sorry

end find_an_find_Tn_l724_724344


namespace reciprocal_of_neg3_l724_724161

theorem reciprocal_of_neg3 : ∃ x : ℝ, -3 * x = 1 ∧ x = -1/3 :=
by
  sorry

end reciprocal_of_neg3_l724_724161


namespace simplify_fractions_l724_724514

theorem simplify_fractions :
  (20 / 19) * (15 / 28) * (76 / 45) = 95 / 84 :=
by
  sorry

end simplify_fractions_l724_724514


namespace sheets_torn_out_l724_724895

-- Define the conditions as given in the problem
def first_torn_page : Nat := 185
def last_torn_page : Nat := 518
def pages_per_sheet : Nat := 2

-- Calculate the total number of pages torn out
def total_pages_torn_out : Nat :=
  last_torn_page - first_torn_page + 1

-- Calculate the number of sheets torn out
def number_of_sheets_torn_out : Nat :=
  total_pages_torn_out / pages_per_sheet

-- Prove that the number of sheets torn out is 167
theorem sheets_torn_out :
  number_of_sheets_torn_out = 167 :=
by
  unfold number_of_sheets_torn_out total_pages_torn_out
  rw [Nat.sub_add_cancel (Nat.le_of_lt (Nat.lt_of_le_of_ne
    (Nat.le_add_left _ _) (Nat.ne_of_lt (Nat.lt_add_one 184))))]
  rw [Nat.div_eq_of_lt (Nat.lt.base 333)] 
  sorry -- proof steps are omitted

end sheets_torn_out_l724_724895


namespace range_of_m_l724_724328

-- Given conditions
variable {x y m : ℝ}

-- Hypotheses
axiom h1 : 1 < x
axiom h2 : x < 3
axiom h3 : -3 < y
axiom h4 : y < 1

-- Definition of m
def m := x - 3 * y

-- Goal
theorem range_of_m : -2 < m ∧ m < 12 :=
by
  simp only [m]
  sorry

end range_of_m_l724_724328


namespace emilia_water_requirements_l724_724938

-- Definitions for the conditions
def water_for_flour (flour_ml : ℕ) : ℕ := (flour_ml / 300) * 90
def additional_water : ℕ := 50

-- Main theorem statement
theorem emilia_water_requirements :
  ∀ (flour_ml : ℕ), flour_ml = 900 → water_for_flour flour_ml + additional_water = 320 :=
by
  intro flour_ml
  intro h
  have h1 : water_for_flour flour_ml = 270 := by
    rw [h]
    simp [water_for_flour]
  rw [h1]
  simp [additional_water]
  sorry

end emilia_water_requirements_l724_724938


namespace ethanol_in_full_tank_l724_724280

/-- An empty fuel tank with a capacity of 200 gallons was filled partially with fuel A and then to capacity with fuel B. 
    Fuel A contains 12% ethanol by volume and fuel B contains 16% ethanol by volume. The full fuel tank contains a certain amount of ethanol.
    Given that 49.99999999999999 gallons of fuel A were added, prove that the total amount of ethanol in the full fuel tank is 30 gallons. -/
theorem ethanol_in_full_tank : 
  ∀ (capacity tank : ℕ) (ethanolA_percent ethanolB_percent : ℚ) (volumeA : ℚ),
    capacity = 200 →
    ethanolA_percent = 12/100 →
    ethanolB_percent = 16/100 →
    volumeA = 49.99999999999999 →
    let volumeB := capacity - volumeA in
    let ethanolA := ethanolA_percent * volumeA in
    let ethanolB := ethanolB_percent * volumeB in
    let total_ethanol := ethanolA + ethanolB in
    total_ethanol = 30 :=
by sorry

end ethanol_in_full_tank_l724_724280


namespace perimeter_of_triangle_ADE_l724_724369

theorem perimeter_of_triangle_ADE
  (a b : ℝ) (F1 F2 A : ℝ × ℝ) (D E : ℝ × ℝ) 
  (h_ellipse : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1)
  (h_a_gt_b : a > b)
  (h_b_gt_0 : b > 0)
  (h_eccentricity : ∃ c, c / a = 1 / 2 ∧ a^2 - b^2 = c^2)
  (h_F1_F2 : ∀ F1 F2, distance F1 (0, 0) = distance F2 (0, 0) ∧ F1 ≠ F2 ∧ 
                       ∀ P : ℝ × ℝ, (distance P F1 + distance P F2 = 2 * a) ↔ (x : ℝ)(y : ℝ) (h_ellipse x y))
  (h_line_DE : ∃ k, ∃ c, ∀ x F1 A, (2 * a * x/(sqrt k^2 + 1)) = |DE|
  (h_length_DE : |DE| = 6)
  (h_A_vertex : A = (0, b))
  : ∃ perim : ℝ, perim = 13 :=
sorry

end perimeter_of_triangle_ADE_l724_724369


namespace P_and_S_could_not_be_fourth_l724_724516

-- Define the relationships between the runners using given conditions
variables (P Q R S T U : ℕ)

axiom P_beats_Q : P < Q
axiom Q_beats_R : Q < R
axiom R_beats_S : R < S
axiom T_after_P_before_R : P < T ∧ T < R
axiom U_before_R_after_S : S < U ∧ U < R

-- Prove that P and S could not be fourth
theorem P_and_S_could_not_be_fourth : ¬((Q < U ∧ U < P) ∨ (Q > S ∧ S < P)) :=
by sorry

end P_and_S_could_not_be_fourth_l724_724516


namespace trapezoid_inscribed_circle_radius_l724_724626

theorem trapezoid_inscribed_circle_radius (a b h : ℤ) (area1 area2 : ℤ) 
  (h_integer_sides : a ∈ ℤ ∧ b ∈ ℤ ∧ h ∈ ℤ)
  (h_area_conditions : area1 = 15 ∧ area2 = 30)
  (h_total_area : (a + b) * h = 90) 
  (h_area_splitting : (a + b) * h / 2 = area1 + area2) : 
  let radius := (a + b) * h / 180 in radius = 5 / 2 := 
sorry

end trapezoid_inscribed_circle_radius_l724_724626


namespace pythagoras_school_students_l724_724853

theorem pythagoras_school_students 
  (x : ℕ)
  (h_math : x / 2)
  (h_music : x / 4)
  (h_rest : x / 7)
  (h_eq : x = x / 2 + x / 4 + x / 7 + 3) : x = 28 :=
by
  sorry

end pythagoras_school_students_l724_724853


namespace simplify_expression_l724_724101

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
  ((x^2 + 1) / (x - 1) - 2 * x / (x - 1)) = x - 1 :=
by
  -- Proof goes here.
  sorry

end simplify_expression_l724_724101


namespace overall_percentage_of_savings_l724_724257

-- Defining the housewife's savings and percentage conditions
def kitchen_appliance_savings : ℝ := 8
def kitchen_appliance_discount : ℝ := 0.2

def home_decor_savings : ℝ := 12
def home_decor_discount : ℝ := 0.15

def gardening_tool_savings : ℝ := 4
def gardening_tool_discount : ℝ := 0.1

def total_spent : ℝ := 95

-- Stating the theorem for the overall percentage of savings
theorem overall_percentage_of_savings :
  let original_price_kitchen := kitchen_appliance_savings / kitchen_appliance_discount,
      original_price_home_decor := home_decor_savings / home_decor_discount,
      original_price_gardening := gardening_tool_savings / gardening_tool_discount,
      total_original_price := original_price_kitchen + original_price_home_decor + original_price_gardening,
      total_savings := total_original_price - total_spent,
      overall_percentage := (total_savings / total_original_price) * 100
  in overall_percentage = 40.625 := by
  sorry

end overall_percentage_of_savings_l724_724257


namespace intersection_M_N_l724_724408

  open Set

  def M : Set ℝ := {x | Real.log x > 0}
  def N : Set ℝ := {x | x^2 ≤ 4}

  theorem intersection_M_N : M ∩ N = {x | 1 < x ∧ x ≤ 2} :=
  by
    sorry
  
end intersection_M_N_l724_724408


namespace polynomial_sum_of_roots_l724_724951

theorem polynomial_sum_of_roots :
  let p := Polynomial.Coeff 4 0 * X^3 + Polynomial.Coeff 5 0 * X^2 - Polynomial.Coeff 8 0 * X in
  p.sum_roots = -1.25 :=
by sorry

end polynomial_sum_of_roots_l724_724951


namespace distance_between_planes_is_zero_l724_724671

variable {R : Type*} [Field R] [LinearOrderedField R]

def plane1 : AffineSubspace R (EuclideanSpace R (Fin 3)) :=
  { carrier := {p | 2 * (p 0) - 4 * (p 1) + 4 * (p 2) = 10},
    smul_mem' := sorry,
    add_mem' := sorry,
    zero_mem' := sorry }

def plane2 : AffineSubspace R (EuclideanSpace R (Fin 3)) :=
  { carrier := {p | 4 * (p 0) - 8 * (p 1) + 8 * (p 2) = 20},
    smul_mem' := sorry,
    add_mem' := sorry,
    zero_mem' := sorry }

theorem distance_between_planes_is_zero :
  ∀ (p1 p2 : AffineSubspace R (EuclideanSpace R (Fin 3))), 
    (p1.carrier = plane1.carrier ∧ p2.carrier = plane2.carrier) → distance p1 p2 = 0 :=
by
  sorry

end distance_between_planes_is_zero_l724_724671


namespace basketball_free_throws_l724_724137

theorem basketball_free_throws:
  ∀ (a b x : ℕ),
    3 * b = 4 * a →
    x = 2 * a →
    2 * a + 3 * b + x = 65 →
    x = 18 := 
by
  intros a b x h1 h2 h3
  sorry

end basketball_free_throws_l724_724137


namespace similar_sizes_bound_l724_724057

theorem similar_sizes_bound (k : ℝ) (hk : k < 2) :
  ∃ (N_k : ℝ), ∀ (A : multiset ℝ), (∀ a ∈ A, a ≤ k * multiset.min A) → 
  A.card ≤ N_k := sorry

end similar_sizes_bound_l724_724057


namespace part1_part2_l724_724329

namespace Proof

def A (a b : ℝ) : ℝ := 3 * a ^ 2 - 4 * a * b
def B (a b : ℝ) : ℝ := a ^ 2 + 2 * a * b

theorem part1 (a b : ℝ) : 2 * A a b - 3 * B a b = 3 * a ^ 2 - 14 * a * b := by
  sorry
  
theorem part2 (a b : ℝ) (h : |3 * a + 1| + (2 - 3 * b) ^ 2 = 0) : A a b - 2 * B a b = 5 / 3 := by
  have ha : a = -1 / 3 := by
    sorry
  have hb : b = 2 / 3 := by
    sorry
  rw [ha, hb]
  sorry

end Proof

end part1_part2_l724_724329


namespace no_family_of_lines_exists_l724_724841

theorem no_family_of_lines_exists (k : ℕ → ℝ) (a b : ℕ → ℝ):
  (∀ n : ℕ, (1:ℝ) = k n * 1 - k n + 1) ∧
  (∀ n : ℕ, k (n+1) ≥ (1 - 1 / (k n)) - (1 - k n)) ∧
  (∀ n : ℕ, k n * k (n+1) ≥ 0) →
  false :=
begin
  sorry,
end

end no_family_of_lines_exists_l724_724841


namespace annual_rent_per_square_foot_l724_724230

-- Given conditions
def dimensions_length : ℕ := 10
def dimensions_width : ℕ := 10
def monthly_rent : ℕ := 1300

-- Derived conditions
def area : ℕ := dimensions_length * dimensions_width
def annual_rent : ℕ := monthly_rent * 12

-- The problem statement as a theorem in Lean 4
theorem annual_rent_per_square_foot :
  annual_rent / area = 156 := by
  sorry

end annual_rent_per_square_foot_l724_724230


namespace number_of_puppies_l724_724745

theorem number_of_puppies (total_ears : ℕ) (ears_per_puppy : ℕ) (h1 : total_ears = 210) (h2 : ears_per_puppy = 2) :
  total_ears / ears_per_puppy = 105 :=
by
  rw [h1, h2]
  norm_num

end number_of_puppies_l724_724745


namespace circumference_of_smaller_circle_l724_724524

noncomputable def pi : ℝ := 3.141592653589793

noncomputable def circumference (r : ℝ) : ℝ := 2 * pi * r
noncomputable def area (r : ℝ) : ℝ := pi * r^2

constant R : ℝ
constant r : ℝ
constant C_larger : ℝ := 704
constant delta_area : ℝ := 26960.847359767075

axiom radius_of_larger_circle : R = C_larger / (2 * pi)
axiom difference_in_area : area R - area r = delta_area

theorem circumference_of_smaller_circle : circumference r = 396 := 
by
  sorry

end circumference_of_smaller_circle_l724_724524


namespace remainder_of_7529_div_by_9_is_not_divisible_by_11_l724_724948

theorem remainder_of_7529_div_by_9 : 7529 % 9 = 5 := by
  sorry

theorem is_not_divisible_by_11 : ¬ (7529 % 11 = 0) := by
  sorry

end remainder_of_7529_div_by_9_is_not_divisible_by_11_l724_724948


namespace part_a_l724_724538

variable {n : ℕ}

theorem part_a (a : Fin n → ℝ) (h1 : ∀ i, |a i| ≤ 1) (h2 : (∑ i, a i) = 0) :
  ∃ k : Fin n, |∑ i in Finset.range k.succ, (i + 1) * a ⟨i, k.is_lt.trans_le (Nat.le_of_lt_succ i.2)⟩| ≤ (2 * (k + 1) + 1) / 4 := sorry

end part_a_l724_724538


namespace cos_eq_solutions_l724_724407

theorem cos_eq_solutions (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ π → cos (2 * x) + 4 * a * sin x + a - 2 = 0) ↔
  ((3/5 < a ∧ a ≤ 1) ∨ a = 1/2) :=
sorry

end cos_eq_solutions_l724_724407


namespace simplify_expression_l724_724093

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
    ((x^2 + 1) / (x - 1)) - (2 * x / (x - 1)) = x - 1 := 
by
    sorry

end simplify_expression_l724_724093


namespace sum_of_sines_l724_724232

def f (x : ℝ) : ℝ := Real.sin (Real.pi / 3 * x)

theorem sum_of_sines : (∑ i in Finset.range 2018, f (i+1)) = (Real.sqrt 3) / 2 :=
by
  sorry

end sum_of_sines_l724_724232


namespace reciprocal_of_neg_three_l724_724157

theorem reciprocal_of_neg_three : ∃ (x : ℚ), (-3 * x = 1) ∧ (x = -1 / 3) :=
by
  use (-1 / 3)
  split
  . rw [mul_comm]
    norm_num 
  . norm_num

end reciprocal_of_neg_three_l724_724157


namespace part_I_tangent_line_part_II_range_of_m_l724_724721

noncomputable def f (m x : ℝ) : ℝ :=
  Real.exp (2 * x) + m * x

theorem part_I_tangent_line :
  let m := -1
  ∃ T : ℝ → ℝ, (∀ x, T x = f m 0 + (f m 0) * x) :=
by
  let m := -1
  use (λ x, x + 1)
  sorry

theorem part_II_range_of_m :
  (∀ x: ℝ, f ∶ ℝ → ℝ → ℝ f x > 0) → (-2 * Real.exp 1 < m ∧ m ≤ 0) :=
by
  sorry

end part_I_tangent_line_part_II_range_of_m_l724_724721


namespace pages_torn_and_sheets_calculation_l724_724897

theorem pages_torn_and_sheets_calculation : 
  (∀ (n : ℕ), (sheet_no n) = (n + 1) / 2 → (2 * (n + 1) / 2) - 1 = n ∨ 2 * (n + 1) / 2 = n) →
  let first_page := 185 in
  let last_page := 518 in
  last_page = 518 → 
  ((last_page - first_page + 1) / 2) = 167 := 
by
  sorry

end pages_torn_and_sheets_calculation_l724_724897


namespace second_number_less_than_twice_first_l724_724920

theorem second_number_less_than_twice_first (x y z : ℤ) (h1 : y = 37) (h2 : x + y = 57) (h3 : y = 2 * x - z) : z = 3 :=
by
  sorry

end second_number_less_than_twice_first_l724_724920


namespace coffee_consumption_l724_724133

variables (h w g : ℝ)

theorem coffee_consumption (k : ℝ) 
  (H1 : ∀ h w g, h * g = k * w)
  (H2 : h = 8 ∧ g = 4.5 ∧ w = 2)
  (H3 : h = 4 ∧ w = 3) : g = 13.5 :=
by {
  sorry
}

end coffee_consumption_l724_724133


namespace cosine_of_acute_angle_l724_724709

theorem cosine_of_acute_angle (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : Real.sin α = 4 / 5) : Real.cos α = 3 / 5 :=
by
  sorry

end cosine_of_acute_angle_l724_724709


namespace perimeter_of_triangle_ADE_l724_724351

noncomputable def ellipse (x y a b : ℝ) : Prop :=
  (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1

def foci_distance (a : ℝ) : ℝ := a / 2

def line_through_f1_perpendicular_to_af2 (x y c : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * (x + c)

def distance_between_points (x1 x2 y1 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem perimeter_of_triangle_ADE
  (a b c : ℝ)
  (h_ellipse : ∀ (x y : ℝ), ellipse x y a b)
  (h_eccentricity : b = Real.sqrt 3 * c)
  (h_foci_distance : foci_distance a = c)
  (h_line : ∀ (x y : ℝ), line_through_f1_perpendicular_to_af2 x y c)
  (h_DE : ∀ (x1 y1 x2 y2 : ℝ), distance_between_points x1 x2 y1 y2 = 6) :
  perimeter_of_triangle_ADE = 13 := sorry

end perimeter_of_triangle_ADE_l724_724351


namespace identify_correct_graph_l724_724634

theorem identify_correct_graph (m n : ℝ) (h₁ : m ≠ 0) (h₂ : n ≠ 0) :
  let parabola := fun x y => m * x + n * y ^ 2 = 0,
      conic := fun x y => m * x ^ 2 + n * y ^ 2 = 1;
  (∀ x y, parabola x y ↔ (1 : ℝ)) ∧ (∀ x y, conic x y ↔ (-1 : ℝ)) → 
  ∃ (g : ℕ), g = 1 := 
by 
  sorry

end identify_correct_graph_l724_724634


namespace half_AB_equals_l724_724732

-- Define vectors OA and OB
def vector_OA : ℝ × ℝ := (3, 2)
def vector_OB : ℝ × ℝ := (4, 7)

-- Prove that (1 / 2) * (OB - OA) = (1 / 2, 5 / 2)
theorem half_AB_equals :
  (1 / 2 : ℝ) • ((vector_OB.1 - vector_OA.1), (vector_OB.2 - vector_OA.2)) = (1 / 2, 5 / 2) := 
  sorry

end half_AB_equals_l724_724732


namespace derek_walk_time_l724_724453

theorem derek_walk_time (x : ℕ) :
  (∀ y : ℕ, (y = 9) → (∀ d₁ d₂ : ℕ, (d₁ = 20 ∧ d₂ = 60) →
    (20 * x = d₁ * y + d₂))) → x = 12 :=
by
  intro h
  sorry

end derek_walk_time_l724_724453


namespace infinite_graph_ten_colorable_l724_724455

noncomputable def ten_colorable (G : SimpleGraph (V : Type)) := 
  ∃ (c : V → Fin 10), G.IsColoring c

theorem infinite_graph_ten_colorable (G : SimpleGraph ℕ) (h : ∀ (V' : Finset ℕ), 
  ten_colorable (G.inducedSubgraph V')) : 
  ten_colorable G :=
by
  sorry

end infinite_graph_ten_colorable_l724_724455


namespace perimeter_of_triangle_ADE_l724_724363

noncomputable def ellipse_perimeter (a b : ℝ) (h : a > b) (e : ℝ) (he : e = 1/2) (h_ellipse : ∀ (x y : ℝ), 
                            x^2 / a^2 + y^2 / b^2 = 1) : ℝ :=
13 -- we assert that the perimeter is 13

theorem perimeter_of_triangle_ADE 
  (a b : ℝ) (h : a > b) (e : ℝ) (he : e = 1/2) 
  (C_eq : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1) 
  (upper_vertex_A : ℝ × ℝ)
  (focus_F1 : ℝ × ℝ)
  (focus_F2 : ℝ × ℝ)
  (line_through_F1_perpendicular_to_AF2 : ∀ x y, y = (√3 / 3) * (x + focus_F1.1))
  (points_D_E_on_ellipse : ∃ D E : ℝ × ℝ, line_through_F1_perpendicular_to_AF2 D.1 D.2 = true ∧
    line_through_F1_perpendicular_to_AF2 E.1 E.2 = true ∧ 
    (dist D E = 6)) :
  ∃ perimeter : ℝ, perimeter = ellipse_perimeter a b h e he C_eq :=
sorry

end perimeter_of_triangle_ADE_l724_724363


namespace log_condition_l724_724924

theorem log_condition (x : ℝ) : x < 1 ↔ (log (1/2) x > 0) → (0 < x ∧ x < 1) :=
by
  intro h
  split
  case mp =>
    intro hlog
    obtain ⟨hpos, hlt⟩ := 
      have : 0 < x ∧ x < 1,
      sorry -- The details of this proof step are omitted
    exact this
  case mpr =>
    intro hgt0
    rcases hgt0 with ⟨hpos, hlt⟩
    exact lt_trans hlt (by linarith) -- The details of this proof step are omitted

end log_condition_l724_724924


namespace polynomial_inequality_l724_724586

open Real

noncomputable def polynomial (n : ℕ) (a : Fin (n+1) → ℝ) : (ℝ → ℝ) :=
  fun x => ∑ i in Finset.range (n+1), a ⟨i, by linarith⟩ * x ^ i

theorem polynomial_inequality
    (n : ℕ)
    (a : Fin (n+1) → ℝ)
    (b : Fin n → ℝ)
    (h_n : 2 ≤ n)
    (h_root : ∀ i, polynomial n a b i = 0)
    (x : ℝ)
    (hx : x > Finset.max' (Finset.image (subtype.val) (Finset.univ : Finset (Fin n)))) :
  polynomial n a (x + 1) ≥ (2 * n^2) / (∑ i in Finset.univ, 1 / (x - b i)) :=
  sorry

end polynomial_inequality_l724_724586


namespace correct_quadratic_equation_l724_724779

open Real

theorem correct_quadratic_equation 
  (roots1_sum : 7 + 3 = 10) 
  (roots2_product : -12 * 3 = -36) : 
  ∃ b c, (b = -10 ∧ c = -36) ∧ (∀ x : ℝ, x^2 + b * x + c = x^2 - 10 * x - 36) := 
by
  use [-10, -36]
  simp
  exact And.intro rfl rfl

end correct_quadratic_equation_l724_724779


namespace triangle_CN_angle_l724_724188

/-- Triangle ABC is isosceles with AC = BC.
  ∠ACB = 120°. Point N is in the interior of the triangle such that
  ∠NAC = 18° and ∠NCA = 42°. Prove ∠CNB = 150°. -/
theorem triangle_CN_angle :
  ∀ (A B C N : Type) [EuclideanGeometry] (h: is_triangle ABC)
  (h_isosceles: AC = BC) 
  (angle_ACB: ∠ACB = 120)
  (h_interior: point_in_triangle ABC N)
  (angle_NAC: ∠NAC = 18)
  (angle_NCA: ∠NCA = 42), 
  ∠CNB = 150 := sorry

end triangle_CN_angle_l724_724188


namespace gcd_102_238_l724_724196

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end gcd_102_238_l724_724196


namespace cube_surface_area_increase_l724_724214

theorem cube_surface_area_increase (s : ℝ) :
  let A_original := 6 * s^2
  let s' := 1.8 * s
  let A_new := 6 * s'^2
  (A_new - A_original) / A_original * 100 = 224 :=
by
  -- Definitions from the conditions
  let A_original := 6 * s^2
  let s' := 1.8 * s
  let A_new := 6 * s'^2
  -- Rest of the proof; replace sorry with the actual proof
  sorry

end cube_surface_area_increase_l724_724214


namespace solve_equation_l724_724988

theorem solve_equation : ∃! x : ℕ, 3^x = x + 2 := by
  sorry

end solve_equation_l724_724988


namespace torn_sheets_count_l724_724904

noncomputable def first_page_num : ℕ := 185
noncomputable def last_page_num : ℕ := 518
noncomputable def pages_per_sheet : ℕ := 2

theorem torn_sheets_count :
  last_page_num > first_page_num ∧
  last_page_num.digits = first_page_num.digits.rotate 1 ∧
  pages_per_sheet = 2 →
  (last_page_num - first_page_num + 1)/pages_per_sheet = 167 :=
by {
  sorry
}

end torn_sheets_count_l724_724904


namespace subcommittee_count_l724_724607

theorem subcommittee_count (n : ℕ) (hn : n = 30) : 
  (∑ k in finset.range(n), (nat.choose (n - 1) 2)) = 12180 :=
by {
  rw hn,
  sorry
}

end subcommittee_count_l724_724607


namespace junior_titles_in_sample_l724_724246

noncomputable def numberOfJuniorTitlesInSample (totalEmployees: ℕ) (juniorEmployees: ℕ) (sampleSize: ℕ) : ℕ :=
  (juniorEmployees * sampleSize) / totalEmployees

theorem junior_titles_in_sample (totalEmployees juniorEmployees intermediateEmployees seniorEmployees sampleSize : ℕ) 
  (h_total : totalEmployees = 150) 
  (h_junior : juniorEmployees = 90) 
  (h_intermediate : intermediateEmployees = 45) 
  (h_senior : seniorEmployees = 15) 
  (h_sampleSize : sampleSize = 30) : 
  numberOfJuniorTitlesInSample totalEmployees juniorEmployees sampleSize = 18 := by
  sorry

end junior_titles_in_sample_l724_724246


namespace cakes_served_today_l724_724618

def lunch_cakes := 6
def dinner_cakes := 9
def total_cakes := lunch_cakes + dinner_cakes

theorem cakes_served_today : total_cakes = 15 := by
  sorry

end cakes_served_today_l724_724618


namespace correct_propositions_l724_724819

variables (l m : Line) (α β : Plane)

theorem correct_propositions :
  -- Propositions stated as conditions
  (¬ (m ⟂ α ∧ l ⟂ m → l ‖ α)) ∧
  (¬ (α ⟂ β ∧ α ∩ β = l ∧ m ⟂ l → m ⟂ β)) ∧
  (α ‖ β ∧ l ⟂ α ∧ m ‖ β → l ⟂ m) ∧
  (¬ (α ‖ β ∧ l ‖ α ∧ m ⊆ β → l ‖ m)) → NumberOfCorrectPropositions = 1 :=
begin
  sorry
end

end correct_propositions_l724_724819


namespace probability_of_winning_l724_724537

theorem probability_of_winning (P_lose P_tie P_win : ℚ) (h_lose : P_lose = 5/11) (h_tie : P_tie = 1/11)
  (h_total : P_lose + P_win + P_tie = 1) : P_win = 5/11 := 
by
  sorry

end probability_of_winning_l724_724537


namespace calculate_expression_l724_724643

theorem calculate_expression (a : ℤ) (h : a = -2) : a^3 - a^2 = -12 := 
by
  sorry

end calculate_expression_l724_724643


namespace circle_radius_k_value_l724_724941

/-- Two circles are centered at the origin. Point P(5, 12) is on the larger circle and point S(0, k) is on the smaller circle. If QR = 5, prove that k = 8. -/
theorem circle_radius_k_value :
  let origin : ℝ × ℝ := (0, 0)
  let P : ℝ × ℝ := (5, 12)
  let r_large := dist origin P
  let QR : ℝ := 5
  let r_small := r_large - QR
  let S : ℝ × ℝ := (0, k)
  dist origin S = r_small → k = 8 :=
by
  let origin := (0: ℝ, 0: ℝ)
  let P := (5: ℝ, 12: ℝ)
  have r_large : ℝ := dist origin P -- radius of the larger circle
  have QR : ℝ := 5 -- given QR = 5
  have r_small : ℝ := r_large - QR -- radius of the smaller circle
  let S := (0: ℝ, k: ℝ)
  intro h
  rw [dist_eq_zero_right_iff_eq, ←h]
  rw [dist, norm_eq] at r_small
  sorry

end circle_radius_k_value_l724_724941


namespace cylinder_total_surface_area_l724_724619

-- Definitions based on the conditions given
def radius : ℝ := 3
def height : ℝ := 7
def lateral_surface_area (r : ℝ) (h : ℝ) : ℝ := 2 * real.pi * r * h
def base_area (r : ℝ) : ℝ := real.pi * r^2

-- Total surface area of the right cylinder
def total_surface_area (r : ℝ) (h : ℝ) : ℝ := lateral_surface_area r h + 2 * base_area r

-- The problem statement: prove that the total surface area is 60π
theorem cylinder_total_surface_area : total_surface_area radius height = 60 * real.pi :=
by sorry -- proof to be filled in

end cylinder_total_surface_area_l724_724619


namespace perimeter_of_triangle_ADE_l724_724352

noncomputable def ellipse (x y a b : ℝ) : Prop :=
  (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1

def foci_distance (a : ℝ) : ℝ := a / 2

def line_through_f1_perpendicular_to_af2 (x y c : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * (x + c)

def distance_between_points (x1 x2 y1 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem perimeter_of_triangle_ADE
  (a b c : ℝ)
  (h_ellipse : ∀ (x y : ℝ), ellipse x y a b)
  (h_eccentricity : b = Real.sqrt 3 * c)
  (h_foci_distance : foci_distance a = c)
  (h_line : ∀ (x y : ℝ), line_through_f1_perpendicular_to_af2 x y c)
  (h_DE : ∀ (x1 y1 x2 y2 : ℝ), distance_between_points x1 x2 y1 y2 = 6) :
  perimeter_of_triangle_ADE = 13 := sorry

end perimeter_of_triangle_ADE_l724_724352


namespace maximum_root_l724_724699

noncomputable def max_root (α β γ : ℝ) : ℝ := 
  if α ≥ β ∧ α ≥ γ then α 
  else if β ≥ α ∧ β ≥ γ then β 
  else γ

theorem maximum_root :
  ∃ α β γ : ℝ, α + β + γ = 14 ∧ α^2 + β^2 + γ^2 = 84 ∧ α^3 + β^3 + γ^3 = 584 ∧ max_root α β γ = 8 :=
by
  sorry

end maximum_root_l724_724699


namespace total_students_l724_724933

noncomputable def total_students_in_gym (F : ℕ) (T : ℕ) : Prop :=
  T = 26

theorem total_students (F T : ℕ) (h1 : 4 = T - F) (h2 : F / (F + 4) = 11 / 13) : total_students_in_gym F T :=
by sorry

end total_students_l724_724933


namespace glacier_discovery_overtakes_denali_star_in_60_seconds_l724_724982

theorem glacier_discovery_overtakes_denali_star_in_60_seconds:
  ∀ (speed_ds speed_gd length_ds length_gd : ℝ),
    speed_ds = 50 → 
    speed_gd = 70 → 
    length_ds = 1 / 6 → 
    length_gd = 1 / 6 →
  let combined_length := length_ds + length_gd in
  let relative_speed := speed_gd - speed_ds in
  let relative_speed_per_minute := relative_speed / 60 in
  let time_to_overtake := combined_length / relative_speed_per_minute in
  let time_in_seconds := time_to_overtake * 60 in
  time_in_seconds = 60 := 
by 
  intros speed_ds speed_gd length_ds length_gd h_ds h_gd h_len_ds h_len_gd
  let combined_length := length_ds + length_gd
  let relative_speed := speed_gd - speed_ds
  let relative_speed_per_minute := relative_speed / 60
  let time_to_overtake := combined_length / relative_speed_per_minute
  let time_in_seconds := time_to_overtake * 60
  sorry

end glacier_discovery_overtakes_denali_star_in_60_seconds_l724_724982


namespace product_of_fractions_l724_724288

theorem product_of_fractions : (2 : ℚ) / 9 * (4 : ℚ) / 5 = 8 / 45 :=
by 
  sorry

end product_of_fractions_l724_724288


namespace constant_term_of_P_l724_724311

-- Define the polynomial P(x)
noncomputable def P (x : ℤ) : ℤ := sorry

-- Conditions
axiom P_int_coefficients : ∀ x : ℤ, ∃ c : ℤ, P(x) = c
axiom P_19 : P(19) = 1994
axiom P_94 : P(94) = 1994
axiom abs_constant_term_lt_1000 : ∀ a₀ : ℤ, (∃ Q : ℤ → ℤ, P = λ x, x * Q x + a₀) → |a₀| < 1000

-- The proof statement
theorem constant_term_of_P : ∃ (a₀ : ℤ), (∀ (Q : ℤ → ℤ), P = λ x, x * Q x + a₀) ∧ a₀ = 208 :=
by {
  -- Inserting "sorry" to indicate the proof will be constructed here
  sorry
}

end constant_term_of_P_l724_724311


namespace sum_of_s_r_l724_724468

def rDomain := {-2, -1, 0, 1}
def rRange := {-3, 0, 3, 6}
def sDomain := {0, 3, 6, 9}

def s (x : ℕ) : ℕ := x + 2

theorem sum_of_s_r (r : ℤ → ℤ) :
  (∀ x ∈ rDomain, r x ∈ rRange) →
  r '' rDomain ∩ sDomain ⊆ sDomain →
  finset.sum (finset.image s (finset.filter (λ y, y ∈ sDomain) (finset.image r (rDomain : finset ℤ)))) = 15 :=
by sorry

end sum_of_s_r_l724_724468


namespace probability_of_same_club_l724_724854

noncomputable theory

def total_choices : ℕ := 3
def total_events : ℕ := total_choices * total_choices
def same_club_events : ℕ := 3
def probability_same_club : ℚ := same_club_events / total_events

theorem probability_of_same_club :
  probability_same_club = 1 / 3 :=
by
  -- skip proof
  sorry

end probability_of_same_club_l724_724854


namespace workers_time_to_complete_job_l724_724682

theorem workers_time_to_complete_job (D E Z H k : ℝ) (h1 : 1 / D + 1 / E + 1 / Z + 1 / H = 1 / (D - 8))
  (h2 : 1 / D + 1 / E + 1 / Z + 1 / H = 1 / (E - 2))
  (h3 : 1 / D + 1 / E + 1 / Z + 1 / H = 3 / Z) :
  E = 10 → Z = 3 * (E - 2) → k = 120 / 19 :=
by
  intros hE hZ
  sorry

end workers_time_to_complete_job_l724_724682


namespace find_t_l724_724687

variable (a : ℝ × ℝ) (b : ℝ × ℝ) (a1 a2 : ℝ) (b1 t : ℝ)

-- Define the vectors with given components
def vector_a : ℝ × ℝ := (5, -7)
def vector_b := (-6, t)

-- Dot product definition
def dot_product (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2

-- Given condition for the dot product
def condition := dot_product (5, -7) (-6, t) = -2

-- Theorem to prove the value of t
theorem find_t (h : condition) : t = -4 :=
by
  -- proof would go here
  sorry

end find_t_l724_724687


namespace probability_two_different_color_chips_l724_724997

theorem probability_two_different_color_chips (blue yellow red : ℕ) (h_total : blue = 6) (h_yellow : yellow = 4) (h_red : red = 2) :
    let total := blue + yellow + red in
    total = 12 →
    (6 / 12 * 4 / 12 + 4 / 12 * 6 / 12) + (6 / 12 * 2 / 12 + 2 / 12 * 6 / 12) + (4 / 12 * 2 / 12 + 2 / 12 * 4 / 12) = 11 / 18 :=
by
  intros
  sorry

end probability_two_different_color_chips_l724_724997


namespace math_problem_prime_quadruples_l724_724811

theorem math_problem_prime_quadruples (p q r n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hq3 : ¬ (p + q) % 3 = 0) :
  p + q = r * (p - q) ^ n ↔ 
  (p = 2 ∧ q = 3 ∧ r = 5 ∧ ∃ k, n = 2 * k) ∨
  (p = 3 ∧ q = 2 ∧ r = 5 ∧ n > 0) ∨
  (p = 5 ∧ q = 3 ∧ r = 1 ∧ n = 3) ∨
  (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 2) ∨
  (p = 5 ∧ q = 3 ∧ r = 8 ∧ n = 1) ∨
  (p = 3 ∧ q = 5 ∧ r = -1 ∧ n = 3) ∨
  (p = 3 ∧ q = 5 ∧ r = -2 ∧ n = 2) ∨
  (p = 3 ∧ q = 5 ∧ r = -8 ∧ n = 1) := by sorry

end math_problem_prime_quadruples_l724_724811


namespace exists_n_for_m_l724_724816

def π (x : ℕ) : ℕ := sorry -- Placeholder for the prime counting function

theorem exists_n_for_m (m : ℕ) (hm : m > 1) : ∃ n : ℕ, n > 1 ∧ n / π n = m :=
by sorry

end exists_n_for_m_l724_724816


namespace determine_xyz_l724_724379

theorem determine_xyz (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 35)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) : 
  x * y * z = 23 / 3 := 
by { sorry }

end determine_xyz_l724_724379


namespace prime_divisors_of_p3_plus_3_l724_724746

theorem prime_divisors_of_p3_plus_3
  (p: ℕ) (hp: Nat.Prime p) 
  (hp2_plus_2: Nat.Prime (p^2 + 2)) : 
  (p = 3 ∧ Nat.PrimeFactors (p^3 + 3).card = 3) :=
by
  sorry

end prime_divisors_of_p3_plus_3_l724_724746


namespace torn_out_sheets_count_l724_724881

theorem torn_out_sheets_count :
  ∃ (sheets : ℕ), (first_page = 185 ∧
                   last_page = 518 ∧
                   pages_torn_out = last_page - first_page + 1 ∧ 
                   sheets = pages_torn_out / 2 ∧
                   sheets = 167) :=
by
  sorry

end torn_out_sheets_count_l724_724881


namespace triangle_ADE_perimeter_l724_724373

noncomputable def ellipse_perimeter (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) (e : ℝ) (h₃ : e = (1 / 2)) 
(F₁ F₂ : ℝ × ℝ) (h₄ : F₁ ≠ F₂) (D E : ℝ × ℝ) (h₅ : |D - E| = 6) : ℝ :=
  let c := (sqrt (a ^ 2 - b ^ 2)) in
  let A := (0, b) in
  let AD := sqrt ((fst D) ^ 2 + (snd D - b) ^ 2) in
  let AE := sqrt ((fst E) ^ 2 + (snd E - b) ^ 2) in
  AD + AE + |D - E|

theorem triangle_ADE_perimeter (a b : ℝ) (h₁ : a > b > 0) (e : ℝ) (h₂ : e = (1 / 2))
(F₁ F₂ : ℝ × ℝ) (h₃ : F₁ ≠ F₂)
(D E : ℝ × ℝ) (h₄ : |D - E| = 6) : 
  ellipse_perimeter a b (and.left h₁) (and.right h₁) e h₂ F₁ F₂ h₃ D E h₄ = 19 :=
sorry

end triangle_ADE_perimeter_l724_724373


namespace unique_solution_l724_724677

def d (n : ℕ) : ℕ :=
  ∑ i in (finset.range n).filter (λ i, n % (i + 1) = 0), 1

def is_solution (f : ℕ → ℕ) : Prop :=
    (∀ x : ℕ, d (f x) = x) ∧
    (∀ x y : ℕ, f (x * y) ∣ (x - 1) * y^(x * y - 1) * f x)

def expected_function (f : ℕ → ℕ) : Prop :=
  f 1 = 1 ∧
  ∀ n : ℕ, ∀ (p_a_i : List (ℕ × ℕ)), -- list of prime, exponent pairs
    n = p_a_i.foldl (λ acc pa, acc * pa.1 ^ pa.2) 1 →
    f n = p_a_i.foldl (λ acc pa, acc * pa.1 ^ (pa.1 ^ pa.2 - 1)) 1

theorem unique_solution (f : ℕ → ℕ) :
    is_solution f ↔ expected_function f := sorry

end unique_solution_l724_724677


namespace triangle_area_proof_l724_724120

noncomputable def area_of_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
    (AC AF CQ : ℝ) (hAC : AC = 20) (hAF : AF = 18) (hCQ : CQ = 24) : ℝ :=
  1/2 * 20 * 48

theorem triangle_area_proof (A B C : Type) [metric_space A] [metric_space B] [metric_space C] 
    (AC AF CQ : ℝ) (hAC : AC = 20) (hAF : AF = 18) (hCQ : CQ = 24) : 
    area_of_triangle A B C AC AF CQ hAC hAF hCQ = 288 :=
begin
  sorry
end

end triangle_area_proof_l724_724120


namespace reciprocal_of_neg3_l724_724162

theorem reciprocal_of_neg3 : ∃ x : ℝ, -3 * x = 1 ∧ x = -1/3 :=
by
  sorry

end reciprocal_of_neg3_l724_724162


namespace number_of_paths_l724_724604

-- Definitions of the vertices and paths in the graph.
structure Graph :=
  (vertices : Type)
  (edges : vertices → vertices → Prop)

-- The specific vertices.
inductive Vertices
| A : Vertices
| B : Vertices
| C : Vertices
| D : Vertices
| H : Vertices

open Vertices

-- The specific graph of the problem.
def bugGraph : Graph :=
{ vertices := Vertices,
  edges := λ v1 v2, 
    (v1 = A ∧ v2 = H) ∨ 
    (v1 = H ∧ v2 = A) ∨ 
    (v1 = H ∧ v2 = B) ∨ (v1 = B ∧ v2 = H) ∨
    (v1 = H ∧ v2 = C) ∨ (v1 = C ∧ v2 = H) ∨
    (v1 = H ∧ v2 = D) ∨ (v1 = D ∧ v2 = H) }

-- Counting the number of paths based on the conditions.
def count_paths (g : Graph) : ℕ :=
  let paths_from_A_to_H := 4 in
  let paths_from_H_to_B := 3 in
  let paths_from_H_to_C := 3 in
  let paths_from_H_to_D := 3 in
  let total_paths_to_each := paths_from_A_to_H * paths_from_H_to_B +
                             paths_from_A_to_H * paths_from_H_to_C +
                             paths_from_A_to_H * paths_from_H_to_D in
  total_paths_to_each

-- The theorem we want to prove.
theorem number_of_paths : count_paths bugGraph = 36 :=
by {
  sorry
}

end number_of_paths_l724_724604


namespace social_studies_score_l724_724856

-- Step d): Translate to Lean 4
theorem social_studies_score 
  (K E S SS : ℝ)
  (h1 : (K + E + S) / 3 = 89)
  (h2 : (K + E + S + SS) / 4 = 90) :
  SS = 93 :=
by
  -- We'll leave the mathematics formal proof details to Lean.
  sorry

end social_studies_score_l724_724856


namespace intersection_area_two_circles_l724_724943

theorem intersection_area_two_circles :
  let r : ℝ := 3
  let center1 : ℝ × ℝ := (3, 0)
  let center2 : ℝ × ℝ := (0, 3)
  let intersection_area := (9 * Real.pi - 18) / 2
  (∃ x y : ℝ, (x - center1.1)^2 + y^2 = r^2 ∧ x^2 + (y - center2.2)^2 = r^2) →
  (∃ (a : ℝ), a = intersection_area) :=
by
  sorry

end intersection_area_two_circles_l724_724943


namespace max_variance_l724_724256

theorem max_variance (a b : ℝ) (h1 : a + b + 1/9 = 1) :
  let E_X := b + 2/9 in
  let E_X2 := b + 4/9 in
  let D_X := E_X2 - (E_X)^2 in
  D_X ≤ 17/36 :=
by
  let E_X := b + 2/9
  let E_X2 := b + 4/9
  let D_X := E_X2 - (E_X)^2
  have h2 : a + b = 8/9 := by linarith [h1]
  have h3 : ∀ (x : ℝ), - (x - 5/18)^2 ≤ 0 := by intro; linarith
  have D_X_max : D_X = -(b - 5/18)^2 + 17/36 := by
    calc 
      D_X = (b + 4/9) - (b + 2/9) * (b + 2/9) : rfl
      ... = (b + 4/9) - (b^2 + 4b/9 + 4/81) : by ring
      ... = -(b^2 - 5b/9 + 25/81) + 17/36 : by ring
  calc
    D_X ≤ 17/36 : by linarith [h3 b]

end max_variance_l724_724256


namespace all_ones_after_S_iterations_l724_724986

def S (A : List ℤ) : List ℤ := 
  List.zipWith (*) A (A.tail ++ [A.head])

theorem all_ones_after_S_iterations (A : List ℤ) (n : ℕ) 
  (h1 : A.length = 2^n) 
  (h2 : ∀ i ∈ A, i = 1 ∨ i = -1) : 
  ∃ k : ℕ, (S^[2^k] A) = List.replicate (2^n) 1 := 
by
  sorry

end all_ones_after_S_iterations_l724_724986


namespace find_j_l724_724174

-- Definitions of the conditions in a)
variables {V : Type*} [inner_product_space ℝ V]
variables (u v w : V)
variable (j : ℝ)

-- Condition that u - v + w = 0
def condition (u v w : V) : Prop := u - v + w = 0

-- Main statement to prove that J is 0 under the conditions
theorem find_j (h : condition u v w) : j * (u × v) + u × w + w × v = (0 : V) → j = 0 :=
by sorry

end find_j_l724_724174


namespace exists_max_piles_l724_724064

theorem exists_max_piles (k : ℝ) (hk : k < 2) : 
  ∃ Nk : ℕ, ∀ A : Multiset ℝ, 
    (∀ a ∈ A, ∃ m ∈ A, a ≤ k * m) → 
    A.card ≤ Nk :=
sorry

end exists_max_piles_l724_724064


namespace original_price_doubled_l724_724764

variable (P : ℝ)

-- Given condition: Original price plus 20% equals 351
def price_increased (P : ℝ) : Prop :=
  P + 0.20 * P = 351

-- The goal is to prove that 2 times the original price is 585
theorem original_price_doubled (P : ℝ) (h : price_increased P) : 2 * P = 585 :=
sorry

end original_price_doubled_l724_724764


namespace emily_subtracts_99_from_50_squared_l724_724937

theorem emily_subtracts_99_from_50_squared :
  (50 - 1) ^ 2 = 50 ^ 2 - 99 := by
  sorry

end emily_subtracts_99_from_50_squared_l724_724937


namespace system1_solution_system2_solution_l724_724111

theorem system1_solution : ∀ x y : ℝ, (y = 2 * x - 3 ∧ 3 * x + 2 * y = 8) ↔ (x = 2 ∧ y = 1) :=
by
  intros x y
  split
  {
    intro h
    cases h with h1 h2
    subst y
    linarith
  }
  {
    intro h
    cases h with h1 h2
    exact ⟨by linarith, by linarith⟩
  }

theorem system2_solution : ∀ x y : ℝ, (5 * x + 2 * y = 25 ∧ 3 * x + 4 * y = 15) ↔ (x = 5 ∧ y = 0) :=
by
  intros x y
  split
  {
    intro h
    cases h with h1 h2
    subst y
    linarith
  }
  {
    intro h
    cases h with h1 h2
    exact ⟨by linarith, by linarith⟩
  }

end system1_solution_system2_solution_l724_724111


namespace find_angle_CLA1_l724_724777

structure Triangle :=
  (A B C : Point)
  (is_acute : acute_angle_triangle A B C)

structure Heights :=
  (A1 B1 C1 : Point)
  (heightA : is_height A A1)
  (heightB : is_height B B1)
  (heightC : is_height C C1)

structure Circle :=
  (center : Point)
  (radius : ℝ)
  (circumcircle : circumcircle A B C center radius)

structure TangentPoint (circumcircle : Circle) :=
  (T : Point)
  (tangentA : is_tangent T A circumcircle)
  (tangentB : is_tangent T B circumcircle)
  (center : Point)
  (is_center : circumcircle.center = center)

structure ParallelogramPoint (T : Point) (A1 : Point) (B1 : Point) (C1 : Point) :=
  (K : Point)
  (L : Point)
  (drop_perpendicular_T_A1B1 : is_perpendicular T (line A1 B1) K)
  (line_parallel_through_C1_intersects_C_O : parallel (line C1 L) (line  circumcircle.center K))

noncomputable def angle_CLA1 := 90

theorem find_angle_CLA1 (A B C A1 B1 C1 T center K L : Point)
  (h_triangle_ABC : acute_angle_triangle A B C)
  (h_heights : Heights A B C A1 B1 C1)
  (h_circle : Circle A B C center (circumcircle_radius))
  (h_tangent : TangentPoint circumcircle.center T)
  (h_perpendicular : ParallelogramPoint T A1 B1 C1 K L)
  : angle (line L A1) (line C L) = 90 := sorry

end find_angle_CLA1_l724_724777


namespace no_possible_right_triangle_l724_724068

theorem no_possible_right_triangle (x : Real) (h : x > 0)
    (h1 : 3 * x = Math.sqrt ((x) ^ 2 + (2 * x) ^ 2)) : False := by
  sorry

end no_possible_right_triangle_l724_724068


namespace farmer_adds_goats_l724_724075

theorem farmer_adds_goats : 
  (current_cows = 2) → 
  (current_pigs = 3) → 
  (current_goats = 6) → 
  (planned_cows = 3) → 
  (planned_pigs = 5) → 
  (total_animals_after_adding = 21) → 
  (planned_goats = 2) :=
by
  intros current_cows current_pigs current_goats planned_cows planned_pigs total_animals_after_adding planned_goats
  sorry

end farmer_adds_goats_l724_724075


namespace perimeter_triangle_ADA_l724_724358

open Real

noncomputable def eccentricity : ℝ := 1 / 2

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

noncomputable def foci_distance (a b : ℝ) : ℝ :=
  (a^2 - b^2).sqrt

noncomputable def line_passing_through_focus_perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  sorry

noncomputable def distance_de (d e : ℝ) : ℝ := 6

theorem perimeter_triangle_ADA
  (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : foci_distance a b = c)
  (h4 : eccentricity * a = c) (h5 : distance_de 6 6) :
  4 * a = 13 :=
by sorry

end perimeter_triangle_ADA_l724_724358


namespace unique_colorings_of_cube_l724_724529

-- Definition of the problem and conditions
def bottom_corners := ['red, 'green, 'blue, 'purple]
def faces_have_different_colored_corners (cube: vector (vector char 3) 6) : Prop :=
  ∀ face ∈ cube, (distinct face.to_list)

-- Statement of the problem
theorem unique_colorings_of_cube :
  ∃! (top_corners: vector char 4), 
    faces_have_different_colored_corners (bottom_corners ++ top_corners) :=
sorry

end unique_colorings_of_cube_l724_724529
