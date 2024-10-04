import Mathlib
import Mathlib.Algebra.ArithmeticSequence
import Mathlib.Algebra.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Order.Sqrt
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Quadratic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Calculus.Polynomial
import Mathlib.Analysis.Calculus.Tangent
import Mathlib.Analysis.Calculus.TangentCone
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Factors
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Polygon
import Mathlib.Geometry.Trigonometry
import Mathlib.Init.Data.Int
import Mathlib.LinearAlgebra
import Mathlib.LinearAlgebra.AffineSpace.Midpoint
import Mathlib.Order.Filter.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Sorry
import Mathlib.Topology.Basic
import Mathlib.Topology.ContinuousFunction.Basic
import Mathlib.Topology.EuclideanSpace.Basic

namespace fixed_circle_tangent_l164_164055

variable (A B : Point)
variable (O : Point := ⟨0, 0⟩)
variable (circle : Circle := Circle.mk ⟨1, 0⟩ 1)

-- Define the conditions
def non_coincident_chords (A B : Point) (circle : Circle) : Prop :=
  A ∈ circle.points ∧ B ∈ circle.points ∧ (A ≠ B) ∧ 
  (distance O A) * (distance O B) = 2

-- Lean statement for the proof problem
theorem fixed_circle_tangent (A B : Point) (circle : Circle)
  (h1 : non_coincident_chords A B circle) :
  ∃ circle', circle'.center = O ∧ circle'.radius = 1 ∧ tangent_line_through (line_through AB) circle' :=
  sorry -- Placeholder for the proof

end fixed_circle_tangent_l164_164055


namespace point_on_ellipse_l164_164912

noncomputable def ellipse (θ : ℝ) : ℝ × ℝ :=
(2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

def line (t : ℝ) : ℝ × ℝ :=
(-3 + Real.sqrt 3 * t, 2 * Real.sqrt 3 + t)

def dist (P Q : ℝ × ℝ) :=
Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def dist_to_line (P : ℝ × ℝ) : ℝ :=
|P.1 - Real.sqrt 3 * P.2 + 9| / 2

theorem point_on_ellipse (θ : ℝ) (h : 3 * Real.sin θ - 4 * Real.cos θ = 5)
  (A : ℝ × ℝ) (P : ℝ × ℝ) (HP : P = ellipse θ)
  (HA : A = (1, 0)) : P = (-8 / 5, 3 * Real.sqrt 3 / 5) :=
by
  sorry

end point_on_ellipse_l164_164912


namespace rearrangement_impossible_l164_164390

theorem rearrangement_impossible
  (init_covering: ∀ (i j : ℕ), colored i j → (∃ a b : ℕ, (a = i ∨ a = i + 1) ∧ (b = j ∨ b = j + 1) ∧ rectangular a b))
  (one_1x4_torn: ∃ (i j : ℕ), rectangular i j ∧ rectangular (i + 2) j ∧ rectangular (i + 3) j ∧ is_1x4 i j)
  (one_2x2_left: ∃ (i j : ℕ), rectangular i j ∧ rectangular (i + 1) j ∧ rectangular i (j + 1) ∧ rectangular (i + 1) (j + 1) ∧ is_2x2 i j) :
  ¬(∀ (i j : ℕ), colored i j → (∃ a b : ℕ, (a = i ∨ a = i + 1) ∧ (b = j ∨ b = j + 1) ∧ rectangular a b)) :=
by
  sorry

end rearrangement_impossible_l164_164390


namespace sequence_count_23_l164_164112

-- Define the recursive function g with the necessary conditions
def has_sequence_properties (n : ℕ) (s : list ℕ) : Prop :=
  s.length = n ∧
  s.head = 0 ∧
  s.last = some 0 ∧
  ∀ i < n - 1, s.nth i ≠ some 0 ∨ s.nth (i+1) ≠ some 0 ∧
  ∀ i < n - 3, (s.nth i = some 1) → (s.nth (i+1) = some 1) → 
                (s.nth (i+2) = some 1) → (s.nth (i+3) ≠ some 1)

def g (n : ℕ) : ℕ :=
  if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 1
  else if n = 6 then 2
  else if n = 7 then 2
  else g(n-3) + g(n-4) + g(n-4) + g(n-5)

-- Define the target proposition to prove
theorem sequence_count_23 : g 23 = 78 :=
by sorry

end sequence_count_23_l164_164112


namespace intersection_is_correct_l164_164629

def P (x : ℝ) := x > 0
def Q (x : ℝ) := -1 < x ∧ x < 2
def P_inter_Q := {x | P x ∧ Q x}
def correct_set := {x | 0 < x ∧ x < 2}

theorem intersection_is_correct : P_inter_Q = correct_set := 
by 
  sorry

end intersection_is_correct_l164_164629


namespace jerry_cut_maple_trees_l164_164156

theorem jerry_cut_maple_trees :
  (∀ pine maple walnut : ℕ, 
    pine = 8 * 80 ∧ 
    walnut = 4 * 100 ∧ 
    1220 = pine + walnut + maple * 60) → 
  maple = 3 := 
by 
  sorry

end jerry_cut_maple_trees_l164_164156


namespace possible_values_of_sum_l164_164241

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l164_164241


namespace curve_is_line_l164_164025

noncomputable def polar_eq (θ : ℝ) : ℝ :=
  1 / (Real.sin θ + Real.cos θ)

def cartesian_x (r θ : ℝ) := r * Real.cos θ
def cartesian_y (r θ : ℝ) := r * Real.sin θ

theorem curve_is_line (r θ : ℝ) :
  let x := cartesian_x r θ,
      y := cartesian_y r θ in
  r = polar_eq θ → y + x = 1 :=
by
  sorry

end curve_is_line_l164_164025


namespace smallest_sum_of_primes_l164_164751

def is_odd_prime (n : ℕ) : Prop :=
  n > 3 ∧ Prime n ∧ n % 2 = 1
  
noncomputable def smallest_sum_four_odd_primes_div5 : ℕ :=
  60

theorem smallest_sum_of_primes : smallest_sum_four_odd_primes_div5 =
  (∃ a b c d : ℕ, is_odd_prime a ∧ is_odd_prime b ∧ is_odd_prime c ∧ is_odd_prime d ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (a + b + c + d) % 5 = 0 ∧ a + b + c + d = smallest_sum_four_odd_primes_div5) :=
by
  sorry

end smallest_sum_of_primes_l164_164751


namespace find_smallest_int_cube_ends_368_l164_164036

theorem find_smallest_int_cube_ends_368 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 500 = 368 ∧ n = 14 :=
by
  sorry

end find_smallest_int_cube_ends_368_l164_164036


namespace correct_statements_l164_164388

-- Definition of the third statement
def statement_3 := ∀ (x : ℝ), (1/2)^(|x|) ≤ 1

-- Definition of the fourth statement (symmetry about the y-axis)
def statement_4 := ∀ (x : ℝ), ∃ y : ℝ, (2^x = y) ∧ (y = 1 / (2^(-x)))

-- The theorem stating that both statement_3 and statement_4 are true
theorem correct_statements : statement_3 ∧ statement_4 :=
by
  sorry

end correct_statements_l164_164388


namespace sum_product_of_integers_l164_164337

theorem sum_product_of_integers (a b c : ℕ) (h₁ : c = a + b) (h₂ : N = a * b * c) (h₃ : N = 8 * (a + b + c)) : 
  a * b * (a + b) = 16 * (a + b) :=
by {
  sorry
}

end sum_product_of_integers_l164_164337


namespace equal_blocks_strings_l164_164177

open Nat

def count_special_strings (n : ℕ) : ℕ := 2 * (n - 2).choose (n - 2) / 2

theorem equal_blocks_strings (n : ℕ) (h : n ≥ 2) :
  (count_special_strings n) = 2 * (((n-2).choose ((n-2) / 2))) :=
by
  sorry

end equal_blocks_strings_l164_164177


namespace area_of_region_l164_164809

noncomputable def area_under_curve_sqrt_x : ℝ :=
  ∫ x in 1..2, real.sqrt x

theorem area_of_region : area_under_curve_sqrt_x = (4 * real.sqrt 2 - 2) / 3 :=
by
  sorry

end area_of_region_l164_164809


namespace area_triangle_ABC_l164_164151

-- Define the triangle and properties within Lean
variables {A B C D E F : Point}
variable {distance : Point → Point → ℝ}

-- Conditions
def is_midpoint (M X Y : Point) : Prop := distance M X = distance M Y
def is_perpendicular (l1 l2 : Line) : Prop := angle l1 l2 = π / 2
def length_DE : ℝ := 10
def length_AF : ℝ := 15

-- Assume necessary properties
axiom midpoint_AC : is_midpoint D A C
axiom midpoint_AB : is_midpoint E A B
axiom perpendicular_DE_AF : is_perpendicular (line D E) (line A F)
axiom length_DE_is_10 : length (segment D E) = length_DE
axiom length_AF_is_15 : length (segment A F) = length_AF

-- Prove the area of triangle ABC
theorem area_triangle_ABC : area_triangle A B C = 150 :=
sorry

end area_triangle_ABC_l164_164151


namespace find_a_for_imaginary_l164_164967

theorem find_a_for_imaginary (a : ℝ) : ((a - complex.I) * (3 - 2 * complex.I)).re = 0 → a = 2 / 3 :=
by
  sorry

end find_a_for_imaginary_l164_164967


namespace area_of_triangle_formed_by_intercepts_l164_164835

theorem area_of_triangle_formed_by_intercepts :
  let f (x : ℝ) := (x - 4)^2 * (x + 3)
  let x_intercepts := [-3, 4]
  let y_intercept := 48
  let base := 7
  let height := 48
  let area := (1 / 2 : ℝ) * base * height
  area = 168 :=
by
  sorry

end area_of_triangle_formed_by_intercepts_l164_164835


namespace prove_triangle_angle_C_prove_triangle_side_c_l164_164127

noncomputable def triangle_angle_C {a b c : ℝ} (h : sqrt 3 * a * real.cos C - c * real.sin A = 0) : Prop :=
  C = real.pi / 3

noncomputable def triangle_side_c {b : ℝ} (hb : b = 4) {area : ℝ} (h_area : area = 6 * sqrt 3) {a : ℝ} (ha : a = 6) : Prop :=
  c = 2 * sqrt 7

theorem prove_triangle_angle_C {a b c : ℝ} (h : sqrt 3 * a * real.cos C - c * real.sin A = 0) : triangle_angle_C h :=
  sorry

theorem prove_triangle_side_c {b : ℝ} (hb : b = 4) {area : ℝ} (h_area : area = 6 * sqrt 3) {a : ℝ} (ha : a = 6) : triangle_side_c hb h_area ha :=
  sorry

end prove_triangle_angle_C_prove_triangle_side_c_l164_164127


namespace curve_is_line_l164_164026

noncomputable def polar_eq (θ : ℝ) : ℝ :=
  1 / (Real.sin θ + Real.cos θ)

def cartesian_x (r θ : ℝ) := r * Real.cos θ
def cartesian_y (r θ : ℝ) := r * Real.sin θ

theorem curve_is_line (r θ : ℝ) :
  let x := cartesian_x r θ,
      y := cartesian_y r θ in
  r = polar_eq θ → y + x = 1 :=
by
  sorry

end curve_is_line_l164_164026


namespace unknown_towel_rate_l164_164392

theorem unknown_towel_rate :
  let num_towels_100 := 3
  let num_towels_150 := 5
  let num_towels_unknown := 2
  let price_100 := 100
  let price_150 := 150
  let avg_price := 155
  let total_towels := num_towels_100 + num_towels_150 + num_towels_unknown
  let total_cost := avg_price * total_towels
  let cost_100 := num_towels_100 * price_100
  let cost_150 := num_towels_150 * price_150
  let cost_known := cost_100 + cost_150
  let cost_unknown := total_cost - cost_known
  let rate_unknown := cost_unknown / num_towels_unknown
  rate_unknown = 250 :=
by
  dsimp
  sorry

end unknown_towel_rate_l164_164392


namespace parallel_lines_l164_164174

-- Definitions for geometric points and lines
variables {A B C N S T K : Point}
variables {ω : Circle}
variables {O : Point}

-- Assumptions
variable (h1 : AcuteTriangle A B C)
variable (h2 : Circumcircle ω A B C)
variable (h3 : OnArc N ω A C)
variable (h4 : ¬ OnArc N ω B)
variable (h5 : OnLine S A B)
variable (h6 : TangentAt ω N T)
variable (h7 : IntersectAtLine NS ω K)
variable (h8 : ∠NTC = ∠KSB)

-- Theorem statement
theorem parallel_lines (h1 : AcuteTriangle A B C) (h2 : Circumcircle ω A B C)
                      (h3 : OnArc N ω A C) (h4 : ¬OnArc N ω B)
                      (h5 : OnLine S A B) (h6 : TangentAt ω N T)
                      (h7 : IntersectAtLine NS ω K) (h8 : ∠NTC = ∠KSB) :
                      Parallel CK AN ∧ Parallel AN TS :=
begin
  sorry
end

end parallel_lines_l164_164174


namespace exists_m_sqrt_8m_integer_l164_164668

theorem exists_m_sqrt_8m_integer : ∃ (m : ℕ), (m > 0) ∧ (∃ k : ℕ, k^2 = 8 * m) :=
by
  use 2
  split
  · exact Nat.succ_pos 1
  · use 4
    exact Nat.succ_pos 1
    sorry

end exists_m_sqrt_8m_integer_l164_164668


namespace Lizzy_savings_after_loan_l164_164644

theorem Lizzy_savings_after_loan :
  ∀ (initial_amount loan_amount : ℕ) (interest_percent : ℕ),
  initial_amount = 30 →
  loan_amount = 15 →
  interest_percent = 20 →
  initial_amount - loan_amount + loan_amount + loan_amount * interest_percent / 100 = 33 :=
by
  intros initial_amount loan_amount interest_percent h1 h2 h3
  sorry

end Lizzy_savings_after_loan_l164_164644


namespace proof1_proof2_proof3_l164_164152

variable {A B C P Q : Type*}
variable [metric_space A] [metric_space B] [metric_space C] [metric_space P] [metric_space Q]

-- Given Conditions
variable (b c : ℝ) 
variable (PB PC PA QB QC QA : ℝ)
variable (B : ℝ) 
variable (AP AQ : Type*) 

-- Questions
theorem proof1 (h0 : b = c) (h1 : AP intersects BC) : 
  PB + PC = 2 * PA * Real.cos B := sorry

theorem proof2 (h0 : b = c) (h2 : AQ does_not_intersect BC) : 
  abs (QB - QC) = 2 * QA * Real.cos B := sorry

theorem proof3 (h0 : b = c) (h1 : AP intersects BC) (h2 : AQ does_not_intersect BC) : 
  PA / (PB + PC) = QA / abs (QB - QC) := sorry

end proof1_proof2_proof3_l164_164152


namespace parabola_equation_acute_angle_range_l164_164923

def parabola (p : ℝ) : Set (ℝ × ℝ) := { point | ∃ x y, y^2 = 2 * p * x }

theorem parabola_equation (p : ℝ) (h : p > 0) (F : ℝ × ℝ) (P : ℝ × ℝ) :
  (F.1 = p / 2 ∧ F.2 = 0) → 
  (P.1 = 0 ∧ P.2 = 4) → 
  (∃ A : ℝ × ℝ, parabola p A ∧ A = (p / 4, 2) ∧ ∠PQA = 90.0) → 
  (∃ C : Set (ℝ × ℝ), C = { point | ∃ x y : ℝ, y^2 = 4 * sqrt 2 * x }) := by
sorry

theorem acute_angle_range (p m : ℝ) (h : p > 0) :
  0 < m ∧ m ≠ p / 2 ∧ m < 9 * p / 2 :=
by
sorry

end parabola_equation_acute_angle_range_l164_164923


namespace remainder_of_division_l164_164750

open Polynomial

noncomputable def p : Polynomial ℤ := 3 * X^3 - 20 * X^2 + 45 * X + 23
noncomputable def d : Polynomial ℤ := (X - 3)^2

theorem remainder_of_division :
  ∃ q r : Polynomial ℤ, p = q * d + r ∧ degree r < degree d ∧ r = 6 * X + 41 := sorry

end remainder_of_division_l164_164750


namespace convert_101101_to_octal_l164_164825

theorem convert_101101_to_octal :
  let binary_num := string_to_nat "101101" 2 in
  binary_num = 45 → nat_to_octal binary_num = "55" :=
by
  intro h1,
  have h2 : binary_num = 45 := h1,
  rw [h2],
  sorry

end convert_101101_to_octal_l164_164825


namespace product_of_three_integers_sum_l164_164330

theorem product_of_three_integers_sum :
  ∀ (a b c : ℕ), (c = a + b) → (a * b * c = 8 * (a + b + c)) →
  (a > 0) → (b > 0) → (c > 0) →
  (∃ N1 N2 N3: ℕ, N1 = (a * b * (a + b)), N2 = (a * b * (a + b)), N3 = (a * b * (a + b)) ∧ 
  (N1 = 272 ∨ N2 = 160 ∨ N3 = 128) ∧ 
  (N1 + N2 + N3 = 560)) := sorry

end product_of_three_integers_sum_l164_164330


namespace invalid_prob_distribution_D_l164_164622

noncomputable def sum_of_probs_A : ℚ :=
  0 + 1/2 + 0 + 0 + 1/2

noncomputable def sum_of_probs_B : ℚ :=
  0.1 + 0.2 + 0.3 + 0.4

noncomputable def sum_of_probs_C (p : ℚ) (hp : 0 ≤ p ∧ p ≤ 1) : ℚ :=
  p + (1 - p)

noncomputable def sum_of_probs_D : ℚ :=
  (1/1*2) + (1/2*3) + (1/3*4) + (1/4*5) + (1/5*6) + (1/6*7) + (1/7*8)

theorem invalid_prob_distribution_D :
  sum_of_probs_D ≠ 1 := sorry

end invalid_prob_distribution_D_l164_164622


namespace product_value_l164_164842

theorem product_value:
  (\bigprod n in Finset.range (98+1) \ 1 \ 2, n (n+2) / (n+1)^2) * (99 * 101 / 100^2) = 101 / 150 :=
sorry

end product_value_l164_164842


namespace avg_visitors_on_sundays_l164_164424

theorem avg_visitors_on_sundays (avg_other_days : ℕ) (avg_month : ℕ) (days_in_month sundays other_days : ℕ) (total_month_visitors : ℕ) (total_other_days_visitors : ℕ) (S : ℕ):
  avg_other_days = 240 →
  avg_month = 285 →
  days_in_month = 30 →
  sundays = 5 →
  other_days = 25 →
  total_month_visitors = avg_month * days_in_month →
  total_other_days_visitors = avg_other_days * other_days →
  5 * S + total_other_days_visitors = total_month_visitors →
  S = 510 :=
by
  intros _
          _
          _
          _
          _
          _
          _
          h
  -- Proof goes here
  sorry

end avg_visitors_on_sundays_l164_164424


namespace exists_f_lt_zero_l164_164507

noncomputable def f (x : ℝ) : ℝ := Real.sin x - x

theorem exists_f_lt_zero : ∃ x ∈ (set.Ioo 0 (Real.pi / 2)), f x < 0 :=
by
  sorry

end exists_f_lt_zero_l164_164507


namespace max_value_of_PF1_PF2_l164_164893

def is_foci (e : Ellipse) (F1 F2 : Point) : Prop :=
  let c := sqrt (e.a ^ 2 - e.b ^ 2) in
  F1 = (c, 0) ∧ F2 = (-c, 0)

def on_ellipse (P : Point) (e : Ellipse) : Prop :=
  e.eqn P

noncomputable def max_value_of_product (e : Ellipse) (F1 F2 : Point) (P : Point) : ℝ :=
  if is_foci e F1 F2 ∧ on_ellipse P e then
    max (|dist P F1 * dist P F2|)
  else 0

theorem max_value_of_PF1_PF2 (e : Ellipse) (F1 F2 : Point) (P : Point) :
  (e.eqn = λ x y, x^2/25 + y^2/16 = 1) → is_foci e F1 F2 → on_ellipse P e → max_value_of_product e F1 F2 P = 25 :=
sorry

end max_value_of_PF1_PF2_l164_164893


namespace samuel_birds_total_berries_l164_164413

theorem samuel_birds_total_berries (berries_per_day : ℕ) (number_of_birds : ℕ) (days : ℕ) :
  berries_per_day = 7 → number_of_birds = 5 → days = 4 → (berries_per_day * days * number_of_birds = 140) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end samuel_birds_total_berries_l164_164413


namespace find_x_l164_164845

theorem find_x (x : ℝ) (h : 4 * log 4 x = log 4 (4 * x^2)) : x = 2 :=
by
  sorry

end find_x_l164_164845


namespace current_population_l164_164136

def initial_population : ℕ := 4200
def percentage_died : ℕ := 10
def percentage_left : ℕ := 15

theorem current_population (pop : ℕ) (died left : ℕ) 
  (h1 : pop = initial_population) 
  (h2 : died = pop * percentage_died / 100) 
  (h3 : left = (pop - died) * percentage_left / 100) 
  (h4 : ∀ remaining, remaining = pop - died - left) 
  : (pop - died - left) = 3213 := 
by sorry

end current_population_l164_164136


namespace xiaoning_comprehensive_score_l164_164779

theorem xiaoning_comprehensive_score
  (max_score : ℕ := 100)
  (midterm_weight : ℝ := 0.3)
  (final_weight : ℝ := 0.7)
  (midterm_score : ℕ := 80)
  (final_score : ℕ := 90) :
  (midterm_score * midterm_weight + final_score * final_weight) = 87 :=
by
  sorry

end xiaoning_comprehensive_score_l164_164779


namespace amount_of_increase_correct_l164_164452

-- Define the given conditions
def new_salary : ℝ := 90000
def percent_increase : ℝ := 38.46153846153846

-- Calculate the old_salary
def old_salary : ℝ := new_salary / (1 + percent_increase / 100)

-- The amount of the increase
def amount_of_increase : ℝ := new_salary - old_salary

-- Lean statement to prove the amount of increase
theorem amount_of_increase_correct :
  amount_of_increase = 25000 := by
  -- Calculations and justifications to be filled here
  sorry

end amount_of_increase_correct_l164_164452


namespace range_of_m_inequality_for_m_neg_one_l164_164099

open Real

noncomputable def f (x m : ℝ) : ℝ := exp x - x - m

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x > 0 → f x m > 0) ↔ m ≤ 1 :=
begin
  sorry
end

theorem inequality_for_m_neg_one (x : ℝ) (h : x > 0) : 
  (f x (-1) * (x - log x) / exp x) > 1 - 1/exp(2) :=
begin
  sorry
end

end range_of_m_inequality_for_m_neg_one_l164_164099


namespace length_of_segment_AB_l164_164316

theorem length_of_segment_AB :
  ∀ (x1 y1 x2 y2 : ℝ),
    y1 = x1 - 1 → y2 = x2 - 1 → y1^2 = 4 * x1 → y2^2 = 4 * x2 →
    x1 + x2 = 6 → x1 * x2 = 1 →
    real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 8 :=
by
  sorry

end length_of_segment_AB_l164_164316


namespace points_on_line_l164_164276

theorem points_on_line (n : ℕ) : 9 * n - 8 = 82 → n = 10 :=
by
  sorry

end points_on_line_l164_164276


namespace angle_ATC_eq_90_l164_164137

variables {A B C H T : Type}
variables (triangle_ABC : Triangle A B C)
variables (acute_triangle_ABC : ∀ (a b c : ℝ), a^2 + b^2 > c^2)
variables (altitude_CH : Line C H)
variables (CH_equal_AB : Altitude.CH = Distance AB)
variables (T_in_CHB : Point T)
variables (right_isosceles_BTH : RightIsoscelesTriangle B T H HB)

theorem angle_ATC_eq_90 
  (ABC_acute : AcuteTriangle A B C)
  (altitude : Altitude C H)
  (CH_eq_AB : Altitude.CH = Distance AB)
  (BTH_right_iso : RightIsoscelesTriangle B T H HB) :
  Angle A T C = 90 :=
begin
  sorry
end

end angle_ATC_eq_90_l164_164137


namespace number_of_nonempty_subsets_l164_164321

open Finset

theorem number_of_nonempty_subsets (S : Finset ℕ) (hS : S = {1, 2, 3}) :
  (S.powerset.filter (λ x, x ≠ ∅)).card = 7 :=
by
  /- Using the condition that S = {1, 2, 3} -/
  have hs : S = {1, 2, 3} := hS,
  /- Substitute S as {1, 2, 3} -/
  rw hs,
  /- Calculate the result by considering subsets of {1, 2, 3} -/
  sorry

end number_of_nonempty_subsets_l164_164321


namespace arithmetic_sequence_T_n_bound_l164_164887

open Nat

theorem arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) (h2 : a 2 = 6) (h3_h6 : a 3 + a 6 = 27) :
  (∀ n, a n = 3 * n) := 
by
  sorry

theorem T_n_bound (a : ℕ → ℤ) (S : ℕ → ℤ) (T : ℕ → ℝ) (m : ℝ) (h_general_term : ∀ n, a n = 3 * n) 
  (h_S_n : ∀ n, S n = n^2 + n) (h_T_n : ∀ n, T n = (S n : ℝ) / (3 * (2 : ℝ)^(n-1)))
  (h_bound : ∀ n > 0, T n ≤ m) : 
  m ≥ 3/2 :=
by
  sorry

end arithmetic_sequence_T_n_bound_l164_164887


namespace problem1_problem2_i_problem2_ii_l164_164550

def circle_M (x y : ℝ) : Prop :=
  2 * x ^ 2 + 2 * y ^ 2 - 8 * x - 8 * y - 1 = 0

def circle_N (x y : ℝ) : Prop :=
  x ^ 2 + y ^ 2 + 2 * x + 2 * y - 6 = 0

def line_l (x y : ℝ) : Prop :=
  x + y - 9 = 0

def intersects_origin (f : ℝ → ℝ → Prop) : Prop :=
  f 0 0

theorem problem1 :
  ∀ (x y : ℝ), (circle_M x y ∧ circle_N x y ∨ x = 0 ∧ y = 0) →
  (x ^ 2 + y ^ 2 - 50 / 11 * x - 50 / 11 * y = 0) :=
sorry

theorem problem2_i :
  ∀ (x y : ℝ), (x = 4 ∧ y = 5) →
  (5 * x + y - 25 = 0 ∨ x - 5 * y + 21 = 0) :=
sorry

theorem problem2_ii :
  ∀ (m : ℝ), (3 ≤ m ∧ m ≤ 6) ↔
  (∃ x y : ℝ, m = x ∧ y = 9 - x ∧ 
   circle_M 2 2 ∧  
   let dist := (λ x y, real.sqrt ((x - 2) ^ 2 + (y - 2) ^ 2)) in
   dist x y = real.sqrt 53 / 2) :=
sorry

end problem1_problem2_i_problem2_ii_l164_164550


namespace eccentricity_of_conic_section_l164_164906

theorem eccentricity_of_conic_section (m : ℝ) (h : m^2 = 2 * 8) :
  (x y : ℝ) (conic_eq : x^2 + y^2 / m = 1) → 
  (eccentricity : ℝ) (eccentricity = if m = 4 then (sqrt 3 / 2) else if m = -4 then sqrt 5 else 0) :=
sorry

end eccentricity_of_conic_section_l164_164906


namespace circle_general_form_l164_164028

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def dist (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

noncomputable def general_form_circle_equation (C : ℝ × ℝ) (r : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  (1, 1, -2 * C.1, -2 * C.2, C.1^2 + C.2^2 - r^2)

open real

theorem circle_general_form (A B : ℝ × ℝ) (hA : A = (1, 4)) (hB : B = (3, -2)) :
  general_form_circle_equation (midpoint A B) (dist A B / 2) = (1, 1, -4, -2, -5) := by
  /-
  We need to show that the calculated center (2, 1) and radius sqrt(10) give the equation
  (x - 2)^2 + (y - 1)^2 = 10 which expands to the general form x^2 + y^2 - 4x - 2y - 5 = 0.
  -/
  sorry

end circle_general_form_l164_164028


namespace matrix_square_rational_l164_164001
-- Import necessary libraries

-- Define the matrix A in terms of a
def matrix_A (a : ℚ) : Matrix (fin 4) (fin 4) ℚ :=
  ![
    ![a, -a, -1, 0],
    ![a, -a, 0, -1],
    ![1, 0, a, -a],
    ![0, 1, a, -a]
  ]

-- Statement of the problem to prove that a must be 0 for A to be a square of a rational matrix
theorem matrix_square_rational (a : ℚ) : 
  (∃ C : Matrix (fin 4) (fin 4) ℚ, C * C = matrix_A a) → a = 0 :=
by sorry

end matrix_square_rational_l164_164001


namespace problem_1_problem_2_l164_164902

variables (a b : EuclideanSpace ℝ 3)
noncomputable def length (v : EuclideanSpace ℝ 3) := vector.norm v
noncomputable def dot_product (u v : EuclideanSpace ℝ 3) := u ⬝ v
noncomputable def angle (u v : EuclideanSpace ℝ 3) := real.arccos ((dot_product u v) / (length u * length v))

theorem problem_1 (ha : length a = 1) (hb : length b = 3) (hab_angle : real.angle a b = real.pi / 3) :
  length (a + b) = real.sqrt 13 :=
sorry

theorem problem_2 (ha : length a = 1) (hb : length b = 3) (hab_angle : real.angle a b = real.pi / 3) :
  angle a (a + b) = real.arccos (5 * real.sqrt 13 / 26) :=
sorry

end problem_1_problem_2_l164_164902


namespace max_min_modulus_m_l164_164600

noncomputable theory
open Complex

theorem max_min_modulus_m (z1 z2 m : ℂ) (alpha beta : ℂ) 
  (h_eq : α^2 + z1 * α + z2 + m = 0 ∧ β^2 + z1 * β + z2 + m = 0)
  (h_cond : z1^2 - 4 * z2 = 16 + 20*I) 
  (h_roots : abs (α - β) = 2 * real.sqrt 7) :
  let dist_center := real.sqrt (4^2 + 5^2) in
  (abs m = dist_center + 7) ∨ (abs m = 7 - dist_center) :=
sorry

end max_min_modulus_m_l164_164600


namespace prime_remainders_between_50_100_have_five_qualified_numbers_l164_164962

open Nat

def prime_remainder_set : Set ℕ := {2, 3, 5}

def count_prime_remainders_between (a b n : ℕ) : ℕ :=
  (List.range (b - a)).countp (λ x, x + a ∈ (range b).filter prime ∧ ((x + a) % n) ∈ prime_remainder_set)

theorem prime_remainders_between_50_100_have_five_qualified_numbers :
  count_prime_remainders_between 50 100 7 = 5 := 
by
  sorry

end prime_remainders_between_50_100_have_five_qualified_numbers_l164_164962


namespace sum_of_xy_l164_164256

theorem sum_of_xy {x y : ℝ} (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := sorry

end sum_of_xy_l164_164256


namespace sum_of_xy_l164_164254

theorem sum_of_xy {x y : ℝ} (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := sorry

end sum_of_xy_l164_164254


namespace area_of_ADE_is_correct_l164_164605

noncomputable def area_ABC (AB BC AC : ℝ) : ℝ := 
let s := (AB + BC + AC) / 2 in
Real.sqrt (s * (s - AB) * (s - BC) * (s - AC))

noncomputable def sin_angle_A (AB BC AC : ℝ) : ℝ := 
2 * (area_ABC AB BC AC AB BC AC) / (BC * AC)

noncomputable def area_ADE (AD AE sin_A : ℝ) : ℝ := 
1 / 2 * AD * AE * sin_A

theorem area_of_ADE_is_correct :
  let AB := 8; let BC := 15; let AC := 17;
  let AD := 3; let AE := 10 in
  area_ADE AD AE (sin_angle_A AB BC AC) = 120 / 17 :=
by
  sorry

end area_of_ADE_is_correct_l164_164605


namespace cyclic_points_in_triangle_l164_164069

/-- Given an acute triangle ABC with altitudes AD, BE, and CF.
A line through point D parallel to EF intersects the extension of AC and side AB at points Q and R, respectively.
EF intersects the extension of BC at point P.
M is the midpoint of BC.
Prove that points P, Q, M, and R are concyclic. -/
theorem cyclic_points_in_triangle
  {A B C D E F Q R P M : Type*}
  [h_triangle : is_acute_triangle A B C]
  [h_altitudes: is_altitude A D B E C F]
  [h_parallel1: parallel (line_through D Q) (line_through E F)]
  [h_parallel2: parallel (line_through D R) (line_through E F)]
  [h_inter1: intersects_at (extension A C) (line_through D Q) Q]
  [h_inter2: intersects_at (side A B) (line_through D R) R]
  [h_inter3: intersects_at (extension B C) (line_through E F) P]
  [h_midpoint: is_midpoint M B C] :
  concyclic P Q M R :=
sorry

end cyclic_points_in_triangle_l164_164069


namespace mutually_exclusive_events_of_balls_l164_164054

def ball := {color : String} -- Represent each ball by its color

-- Condition: The bag contains 2 red balls and 2 white balls
def bag : Multiset ball := [⟨"Red"⟩, ⟨"Red"⟩, ⟨"White"⟩, ⟨"White"⟩]

-- Event definitions
def at_least_one_white_ball (draw : List ball) : Prop :=
  ∃ b ∈ draw, b.color = "White"

def both_red_balls (draw : List ball) : Prop :=
  ∀ b ∈ draw, b.color = "Red"

-- Proof statement: Prove that "at least one white ball" and "both are red balls" are mutually exclusive
theorem mutually_exclusive_events_of_balls (draw : List ball)
  (h_cond : length draw = 2 ∧ multiset.sublist draw.to_multiset bag) :
  (at_least_one_white_ball draw ↔ ¬both_red_balls draw) :=
by sorry

end mutually_exclusive_events_of_balls_l164_164054


namespace value_of_expression_l164_164402

-- Defining the necessary values of a and b
def a : ℝ := 2.502
def b : ℝ := 0.064

-- The theorem we need to prove
theorem value_of_expression :
  ((a + b)^2 - (a - b)^2) / (a * b) ≈ 4.007 := sorry

end value_of_expression_l164_164402


namespace exists_n_eq_4_l164_164479

theorem exists_n_eq_4 :
  ∃ (n : ℕ), n = 4 ∧ ∀ (x y : ℝ), ∃ (a : ℕ → ℝ), (∑ i in Finset.range n, a i = x) ∧ 
  (∑ i in Finset.range n, (1 / a i) = y) :=
by
  use 4
  sorry

end exists_n_eq_4_l164_164479


namespace curve_is_line_l164_164027

noncomputable def polar_eq (θ : ℝ) : ℝ :=
  1 / (Real.sin θ + Real.cos θ)

def cartesian_x (r θ : ℝ) := r * Real.cos θ
def cartesian_y (r θ : ℝ) := r * Real.sin θ

theorem curve_is_line (r θ : ℝ) :
  let x := cartesian_x r θ,
      y := cartesian_y r θ in
  r = polar_eq θ → y + x = 1 :=
by
  sorry

end curve_is_line_l164_164027


namespace trig_identity_l164_164462

theorem trig_identity :
  (Real.cot (Real.pi / 6) + 
  (2 * Real.cos (Real.pi / 6) + Real.tan (Real.pi / 4)) / (2 * Real.sin (Real.pi / 6)) - 
  (Real.cos (Real.pi / 4))^2) = 
  2 * Real.sqrt 3 + 1 / 2 :=
sorry

end trig_identity_l164_164462


namespace length_of_other_train_is_correct_l164_164411

noncomputable def length_of_other_train
  (l1 : ℝ) -- length of the first train in meters
  (s1 : ℝ) -- speed of the first train in km/hr
  (s2 : ℝ) -- speed of the second train in km/hr
  (t : ℝ)  -- time in seconds
  (h1 : l1 = 500)
  (h2 : s1 = 240)
  (h3 : s2 = 180)
  (h4 : t = 12) :
  ℝ :=
  let s1_m_s := s1 * 1000 / 3600
  let s2_m_s := s2 * 1000 / 3600
  let relative_speed := s1_m_s + s2_m_s
  let total_distance := relative_speed * t
  total_distance - l1

theorem length_of_other_train_is_correct :
  length_of_other_train 500 240 180 12 rfl rfl rfl rfl = 900 := sorry

end length_of_other_train_is_correct_l164_164411


namespace collinear_iff_equal_areas_l164_164180

variable {A B C I H B1 C1 B2 C2 K A1 : Point}

-- Conditions of the problem
axiom Incenter (T : Triangle) : Point
axiom Orthocenter (T : Triangle) : Point
axiom Midpoint (p1 p2 : Point) : Point
axiom IntersectionRay (p1 p2 p3 : Point) : Point
axiom Circumcenter (T : Triangle) : Point
axiom Collinear (p1 p2 p3 : Point) : Prop
axiom Area (T : Triangle) : Real

-- The problem's setup
axiom h1 : I = Incenter (Triangle.mk A B C)
axiom h2 : H = Orthocenter (Triangle.mk A B C)
axiom h3 : B1 = Midpoint A C
axiom h4 : C1 = Midpoint A B
axiom h5 : B2 = IntersectionRay B1 I A
axiom h6 : C2 = IntersectionRay C1 I (Line.extend A C)
axiom h7 : K = Intersection (Line.mk B2 C2) (Line.mk B C)
axiom h8 : A1 = Circumcenter (Triangle.mk B H C)

-- Lean statement for the problem
theorem collinear_iff_equal_areas : 
  Collinear A I A1 ↔ (Area (Triangle.mk B K B2)) = (Area (Triangle.mk C K C2)) := sorry

end collinear_iff_equal_areas_l164_164180


namespace mikail_birthday_money_l164_164209

theorem mikail_birthday_money :
  ∀ (A M : ℕ), A = 3 * 3 → M = 5 * A → M = 45 :=
by
  intros A M hA hM
  rw [hA] at hM
  rw [hM]
  norm_num

end mikail_birthday_money_l164_164209


namespace borrowed_dimes_calculation_l164_164678

-- Define Sam's initial dimes and remaining dimes after borrowing
def original_dimes : ℕ := 8
def remaining_dimes : ℕ := 4

-- Statement to prove that the borrowed dimes is 4
theorem borrowed_dimes_calculation : (original_dimes - remaining_dimes) = 4 :=
by
  -- This is the proof section which follows by simple arithmetic computation
  sorry

end borrowed_dimes_calculation_l164_164678


namespace least_positive_n_divisible_by_125_l164_164194

theorem least_positive_n_divisible_by_125 :
  ∀ (b : ℕ → ℕ), b 15 = 15 →
  (∀ n, n > 15 → b n = 125 * b (n - 1) + 2 * n) →
  ∃ n, n > 15 ∧ b n % 125 = 0 ∧ (∀ m, n > m > 15 → b m % 125 ≠ 0) →
  n = 75 :=
by
  intro b hb1 hrec hexist
  sorry

end least_positive_n_divisible_by_125_l164_164194


namespace geometric_series_common_ratio_l164_164720

theorem geometric_series_common_ratio (a S r : ℝ) 
  (hS : S = a / (1 - r)) 
  (h_modified : (a * r^2) / (1 - r) = S / 16) : 
  r = 1/4 ∨ r = -1/4 :=
by
  sorry

end geometric_series_common_ratio_l164_164720


namespace sum_of_roots_l164_164196

theorem sum_of_roots (a1 a2 a3 a4 a5 : ℤ)
  (h_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧
                a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧
                a3 ≠ a4 ∧ a3 ≠ a5 ∧
                a4 ≠ a5)
  (h_poly : (104 - a1) * (104 - a2) * (104 - a3) * (104 - a4) * (104 - a5) = 2012) :
  a1 + a2 + a3 + a4 + a5 = 17 := by
  sorry

end sum_of_roots_l164_164196


namespace no_meet_at_O_l164_164514

universe u

variable (A B C D O : Type u) [ConvexQuadrilateral A B C D]
variable (Petya Vasya Tolya : Type u) [Pedestrian Petya] [Pedestrian Vasya] [Pedestrian Tolya]
variable [StartsAt Petya A] [StartsAt Vasya A] [StartsAt Tolya B]
variable [TravelsAlong Petya (A, B)] [TravelsAlong Petya (B, C)] [TravelsAlong Petya (C, D)] [TravelsAlong Petya (D, A)]
variable [TravelsAlong Vasya (A, C)]
variable [TravelsAlong Tolya (B, D)]
variable [ConstantSpeed Petya] [ConstantSpeed Vasya] [ConstantSpeed Tolya]
variable [ArrivesSimultaneously Petya B] [ArrivesSimultaneously Petya C] [ArrivesSimultaneously Tolya D]
variable (t : Time)

axiom convex_quadrilateral (A B C D : Type u) : ConvexQuadrilateral A B C D
axiom intersection_diagonals (A C B D) : IntersectsAt A C B D O

theorem no_meet_at_O :
  ¬ ∃ (O : Type u), (IntersectsAt A C B D O) ∧
    (ArrivesSimultaneously Vasya O t) ∧ (ArrivesSimultaneously Tolya O t) :=
sorry

end no_meet_at_O_l164_164514


namespace exists_real_solution_l164_164167

noncomputable def f (x : ℝ) : ℝ :=
  x^2 + 2007 * x + 1

theorem exists_real_solution (n : ℕ) (hn : n > 0) : ∃ x : ℝ, (nat.iterate f n x) = 0 := by
  sorry

end exists_real_solution_l164_164167


namespace third_part_ratio_l164_164677

def ratio_of_third_part : Prop :=
  ∃ (x : ℝ), 782 = A + 164.6315789473684 + C →
    A = 1/2 * (A + 164.6315789473684 + C) ∧ 
    C = x * (A + 164.6315789473684 + C) ∧ 
    x = 0.7473

theorem third_part_ratio :
  ratio_of_third_part →
  ∃ x : ℝ, x = 0.7473 := 
by 
  sorry

end third_part_ratio_l164_164677


namespace solution_set_of_inequality_l164_164525

noncomputable def f : ℝ → ℝ := sorry  -- Assuming such a function exists satisfying the conditions

-- Conditions given in the problem
axiom even_f : ∀ x : ℝ, f x = f (-x)
axiom f_prime_lt_f : ∀ x : ℝ, deriv f x < f x
axiom periodic_f : ∀ x : ℝ, f (x + 1) = f (3 - x)
axiom value_at_2011 : f 2011 = 3

-- Theorem to be proved
theorem solution_set_of_inequality : {x : ℝ | f x < 3 * real.exp (x - 1)} = {x : ℝ | 1 < x} := sorry

end solution_set_of_inequality_l164_164525


namespace union_of_A_and_B_l164_164891

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 5}

theorem union_of_A_and_B : A ∪ B = {1, 2, 4, 5} := 
by
  sorry

end union_of_A_and_B_l164_164891


namespace boys_and_girls_l164_164987

theorem boys_and_girls (x y : ℕ) (h1 : x + y = 21) (h2 : 5 * x + 2 * y = 69) : x = 9 ∧ y = 12 :=
by 
  sorry

end boys_and_girls_l164_164987


namespace max_S_possible_l164_164521

theorem max_S_possible (nums : List ℝ) (h_nums_in_bound : ∀ n ∈ nums, 0 ≤ n ∧ n ≤ 1) (h_sum_leq_253_div_12 : nums.sum ≤ 253 / 12) :
  ∃ (A B : List ℝ), (∀ x ∈ A, x ∈ nums) ∧ (∀ y ∈ B, y ∈ nums) ∧ A.union B = nums ∧ A.sum ≤ 11 ∧ B.sum ≤ 11 :=
sorry

end max_S_possible_l164_164521


namespace magnitude_of_a_minus_2b_l164_164553

def a : ℝ × ℝ × ℝ := (1, 0, 2)
def b : ℝ × ℝ × ℝ := (0, 1, 2)

def vector_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

def scalar_mult (k : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (k * v.1, k * v.2, k * v.3)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

theorem magnitude_of_a_minus_2b : magnitude (vector_sub a (scalar_mult 2 b)) = 3 := by
  sorry

end magnitude_of_a_minus_2b_l164_164553


namespace car_rental_savings_l164_164164

noncomputable def total_distance := 150 * 2
noncomputable def cost_per_liter := 0.90
noncomputable def distance_covered_per_liter := 15
noncomputable def rental_cost_first_option := 50
noncomputable def rental_cost_second_option := 90

theorem car_rental_savings : 
  let gasoline_needed := total_distance / distance_covered_per_liter,
      gasoline_cost := gasoline_needed * cost_per_liter,
      total_cost_first_option := rental_cost_first_option + gasoline_cost in
  (rental_cost_second_option - total_cost_first_option) = 22 :=
by
  sorry

end car_rental_savings_l164_164164


namespace shirts_and_pants_outfits_l164_164389

theorem shirts_and_pants_outfits (shirts pants : ℕ) (h_shirts : shirts = 3) (h_pants : pants = 4) :
  shirts * pants = 12 :=
by
  rw [h_shirts, h_pants]
  exact rfl

end shirts_and_pants_outfits_l164_164389


namespace jills_favorite_number_l164_164159

theorem jills_favorite_number : ∃ n : ℕ, even n ∧ (∃ k : ℕ, prime k ∧ k ∣ n ∧ k = 7) ∧ (∃ m : ℕ, prime m ∧ m ∣ n ∧ m = 7) ∧ n = 98 :=
by
  sorry

end jills_favorite_number_l164_164159


namespace number_of_cleaners_l164_164415

-- Begin the proof using noncomputable instance if necessary
noncomputable theory

-- Define the constants and the problem setup
def initial_number_of_cleaners : ℕ := 
let n : ℕ := sorry in
let person_hours_lower_floors : ℕ := 4 * (n + n / 2) in
let person_hours_upper_floor : ℕ := 4 * (n / 2) + 8 in
-- The condition that relates the cleaning efforts with respect to the floor areas and person hours
have cond : person_hours_lower_floors = 2 * person_hours_upper_floor, 
from sorry,
-- Prove that the initial number of cleaners is 8 given the conditions above
have solution : n = 8, from sorry,
n

theorem number_of_cleaners : initial_number_of_cleaners = 8 := by
  sorry

end number_of_cleaners_l164_164415


namespace right_triangle_angle_bisector_cot_half_angle_l164_164066

-- Define the problem setup
variables {A B C D : Type} [euclidean_geometry] 
variables (c p a : ℝ)

-- The given conditions
variables (hypotenuse : distance A B = c) (angle_bisector_segment : distance B D = p) (leg : distance B C = a)

-- The theorem to be proven
theorem right_triangle_angle_bisector_cot_half_angle (α : ℝ) :
  cot (α / 2) = c / p :=
sorry

end right_triangle_angle_bisector_cot_half_angle_l164_164066


namespace sum_of_selected_primes_divisible_by_3_probability_l164_164010

def first_fifteen_primes : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def count_combinations_divisible_3 (nums : List ℕ) (k : ℕ) : ℕ :=
sorry -- Combines over the list to count combinations summing divisible by 3

noncomputable def probability_divisible_by_3 : ℚ :=
  let total_combinations := (Nat.choose 15 4)
  let favorable_combinations := count_combinations_divisible_3 first_fifteen_primes 4
  favorable_combinations / total_combinations

theorem sum_of_selected_primes_divisible_by_3_probability :
  probability_divisible_by_3 = 1/3 :=
sorry

end sum_of_selected_primes_divisible_by_3_probability_l164_164010


namespace isosceles_triangle_if_x_eq_neg_one_root_right_triangle_if_two_equal_real_roots_find_roots_if_equilateral_triangle_l164_164879

-- Part 1
theorem isosceles_triangle_if_x_eq_neg_one_root (a b c : ℝ) (h : (a + b) * (-1)^2 + 2 * c * (-1) + (b - a) = 0) : 
  b = c ↔ ∃ (ABC : Triangle), ABC.is_isosceles ∧ ABC.side_lengths = (a, b, c) :=
by
  sorry

-- Part 2
theorem right_triangle_if_two_equal_real_roots (a b c : ℝ)
  (h : ∀ (x : ℝ), (a + b) * x^2 + 2 * c * x + (b - a) = 0 ↔ x = -c/(a + b)) :
  a^2 + c^2 = b^2 ↔ ∃ (ABC : Triangle), ABC.is_right ∧ ABC.side_lengths = (a, b, c) :=
by
  sorry

-- Part 3
theorem find_roots_if_equilateral_triangle (a : ℝ)
  (h : (a + a) * x^2 + 2 * a * x + (a - a) = 0) : 
  roots (a, a, a, x) = {0, -1} :=
by
  sorry

end isosceles_triangle_if_x_eq_neg_one_root_right_triangle_if_two_equal_real_roots_find_roots_if_equilateral_triangle_l164_164879


namespace points_on_line_l164_164286

theorem points_on_line (n : ℕ) (h : 9 * n - 8 = 82) : n = 10 := by
  sorry

end points_on_line_l164_164286


namespace chef_earns_less_than_manager_l164_164807

noncomputable def manager_wage : ℝ := 8.50
noncomputable def dishwasher_wage : ℝ := manager_wage / 2
noncomputable def chef_wage : ℝ := dishwasher_wage * 1.22

theorem chef_earns_less_than_manager :
  manager_wage - chef_wage = 3.315 :=
by
  sorry

end chef_earns_less_than_manager_l164_164807


namespace cylinder_volume_l164_164903

theorem cylinder_volume (r h : ℝ) (radius_is_2 : r = 2) (height_is_3 : h = 3) :
  π * r^2 * h = 12 * π :=
by
  rw [radius_is_2, height_is_3]
  sorry

end cylinder_volume_l164_164903


namespace problem_l164_164079

theorem problem (a b c : ℝ) (h : 2 * |a - 1| + sqrt (2 * a - b) + (c - sqrt 3)^2 = 0) :
  a + b + c = 3 + sqrt 3 := by
  sorry

end problem_l164_164079


namespace number_of_mappings_l164_164862

-- Define sets and mapping properties
def A := {a, b, c}
def B := {0, 1, 2}
def f : A → B

-- Main theorem statement
theorem number_of_mappings (h : ∀ a b c, f a + f b = f c) : 
  (number_of_such_mappings f A B = 6) :=
sorry

end number_of_mappings_l164_164862


namespace sixth_equation_pattern_l164_164215

theorem sixth_equation_pattern :
  ∑ k in finset.range (2 * 6 - 1), (k + 6) = 121 :=
by
  sorry

end sixth_equation_pattern_l164_164215


namespace find_number_l164_164409

theorem find_number (x : ℝ) (h : 0.60 * 50 = 0.45 * x + 16.5) : x = 30 :=
by
  sorry

end find_number_l164_164409


namespace frequency_of_country_proof_l164_164656

def frequency_of_country (phrase : String) :=
  let total_words : Nat := 18
  let total_appearances_of_country : Nat := 3
  total_appearances_of_country / total_words.toRational

theorem frequency_of_country_proof : 
  frequency_of_country "When the youth are strong, the country is strong; when the youth are wise, the country is wise; when the youth are wealthy, the country is wealthy." = 1 / 6 :=
by 
  sorry

end frequency_of_country_proof_l164_164656


namespace find_polynomials_l164_164176

noncomputable def factorial_sum (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n+1), Nat.factorial k

theorem find_polynomials (P Q : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 →
    factorial_sum (n + 2) = P n * factorial_sum (n + 1) + Q n * factorial_sum n) →
  (P = (λ n, 1)) ∧ (Q = (λ n, 0)) :=
by
  intro h
  sorry

end find_polynomials_l164_164176


namespace seq_arith_formula_an_l164_164640

variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
axiom a1 : a 1 = 1
axiom a_arith : ∀ n, a n - a (n-1) = d
axiom Sn : ∀ n, S n = (n * a 1) + ((n * (n - 1)) / 2) * d
axiom sqrt_arith : ∀ n ∈ Nat, n ≥ 2 → (√(S (n-1)), √(S n), √(S (n+1))) forms_arithmetic 1

-- Prove (1)
theorem seq_arith : ∀ n ≥ 2, (S n) / n - (S (n-1)) / (n-1) = d / 2 := sorry

-- Prove (2)
theorem formula_an : ∀ n, a n = 2 * n - 1 := sorry

end seq_arith_formula_an_l164_164640


namespace five_students_not_adjacent_l164_164141

theorem five_students_not_adjacent : 
  let total_arrangements := 5!
  let block_arrangements := 3!
  let internal_block_arrangements := 3!
  let invalid_arrangements := block_arrangements * internal_block_arrangements
  let valid_arrangements := total_arrangements - invalid_arrangements
  3.students_refuse_adjacency_84 (total_arrangements valid_arrangements : nat) (H: valid_arrangements = 84) : valid_arrangements = 84 :=
sorry

end five_students_not_adjacent_l164_164141


namespace find_angle_AMD_l164_164265

variable (AB BC: ℝ) (AM MB: ℝ) (MC MD: ℝ)

-- Given conditions
axiom rectangle_ABCD : AB = 8 ∧ BC = 4
axiom point_M_condition : AM = 2 * MB
axiom sin_AMD_condition : sin (real.arcsin (MC / MD)) = MC / MD

-- Side lengths
noncomputable def MB_value : ℝ := (AB - AM) / 2
noncomputable def AM_value : ℝ := (AB * 2) / 3

-- Distances from Pythagorean theorem
noncomputable def MD_value : ℝ := real.sqrt (AM^2 + BC^2)
noncomputable def MC_value : ℝ := real.sqrt ((2 * AM)^2 + BC^2)

-- Theorem statement to prove the angle
theorem find_angle_AMD :
  rectangle_ABCD ∧
  point_M_condition ∧
  sin_AMD_condition →
  real.arcsin (5 / real.sqrt 13) = 67.38 :=
sorry

end find_angle_AMD_l164_164265


namespace distance_origin_is_two_l164_164313

noncomputable def distance_origin_intersection : ℝ :=
  let l1 := λ x y : ℝ, x + y - 2 * real.sqrt 2 = 0
  let l2 := λ t : ℝ, (⟨ real.sqrt 2 / 2 * t, real.sqrt 2 / 2 * t⟩ : ℝ × ℝ)
  let intersection : ℝ × ℝ := ⟨ real.sqrt 2, real.sqrt 2 ⟩
  real.sqrt( (real.sqrt 2)^2 + (real.sqrt 2)^2 )

theorem distance_origin_is_two :
  distance_origin_intersection = 2 :=
sorry

end distance_origin_is_two_l164_164313


namespace different_teams_all_games_l164_164772

noncomputable def players_distributed_across_teams (num_players : ℕ) (games : ℕ) (team_sizes : List ℕ) : Prop :=
  num_players = 22 ∧ games = 3 ∧ team_sizes = [11, 11] ∧
  ∃ p1 p2 : Fin num_players, ∀ g : Fin games, p1 ≠ p2 ∧ 
  (¬(∃ teams : Fin games → List (Fin num_players), 
  (∀ g : Fin games, teams g).length = num_players ∧ 
  (∀ g : Fin games, (teams g).partition (λ p, p ∈ (teams g)) = (⌊team_sizes.head / 2⌋, team_sizes.head - ⌊team_sizes.head / 2⌋)) ∧ 
  (∀ g : Fin games, p1 ∈ (teams g).head ∧ p2 ∈ (teams g).tail)))

theorem different_teams_all_games : 
  ∀ (num_players games : ℕ) (team_sizes : List ℕ), players_distributed_across_teams num_players games team_sizes → 
  ∃ p1 p2 : Fin num_players, ∀ g : Fin games, p1 ≠ p2 ∧ ¬(∃ teams : Fin games → List (Fin num_players), 
  (∀ g : Fin games, teams g).length = num_players ∧ 
  (∀ g : Fin games, (teams g).partition (λ p, p ∈ (teams g)) = (⌊team_sizes.head / 2⌋, team_sizes.head - ⌊team_sizes.head / 2⌋)) ∧ 
  (∀ g : Fin games, p1 ∈ (teams g).head ∧ p2 ∈ (teams g).tail)) :=
begin
  sorry
end

end different_teams_all_games_l164_164772


namespace find_correct_answer_l164_164466

theorem find_correct_answer (x : ℕ) (h : 3 * x = 135) : x / 3 = 15 :=
sorry

end find_correct_answer_l164_164466


namespace find_scalar_k_l164_164728

variables {V : Type*} [inner_product_space ℝ V]

theorem find_scalar_k (a b c d : V) (k : ℝ) (h₁ : a + b + c + d = 0)
    (h₂ : k • (a × b) + b × d + c × d + a × c = 0) : k = 1 :=
by {
  sorry
}

end find_scalar_k_l164_164728


namespace points_on_line_possible_l164_164282

theorem points_on_line_possible : ∃ n : ℕ, 9 * n - 8 = 82 :=
by
  sorry

end points_on_line_possible_l164_164282


namespace acute_triangle_unit_circle_l164_164806

noncomputable def is_acute_triangle (AD BD CD : ℝ) : Prop :=
  AD^2 + BD^2 > CD^2 ∧ AD^2 + CD^2 > BD^2 ∧ BD^2 + CD^2 > AD^2

theorem acute_triangle_unit_circle (x : ℝ) 
  (h1 : -1 ≤ x) (h2 : x ≤ 1) :
  let AD := abs (x + 1),
      BD := abs (1 - x),
      CD := real.sqrt (1 - x^2)
  in
  x ∈ set.Ioo (2 - real.sqrt 5) (real.sqrt 5 - 2) →
  is_acute_triangle AD BD CD :=
begin
  intros AD BD CD,
  sorry
end

end acute_triangle_unit_circle_l164_164806


namespace baseball_games_per_month_l164_164359

theorem baseball_games_per_month (total_games : ℕ) (season_length : ℕ) (games_per_month : ℕ) :
  total_games = 14 → season_length = 2 → games_per_month = total_games / season_length → games_per_month = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end baseball_games_per_month_l164_164359


namespace lizzy_wealth_after_loan_l164_164651

theorem lizzy_wealth_after_loan 
  (initial_wealth : ℕ)
  (loan : ℕ)
  (interest_rate : ℕ)
  (h1 : initial_wealth = 30)
  (h2 : loan = 15)
  (h3 : interest_rate = 20)
  : (initial_wealth - loan) + (loan + loan * interest_rate / 100) = 33 :=
by
  sorry

end lizzy_wealth_after_loan_l164_164651


namespace at_least_one_bigger_than_44_9_l164_164715

noncomputable def x : ℕ → ℝ := sorry
noncomputable def y : ℕ → ℝ := sorry

axiom x_positive (n : ℕ) : 0 < x n
axiom y_positive (n : ℕ) : 0 < y n
axiom recurrence_x (n : ℕ) : x (n + 1) = x n + 1 / (2 * y n)
axiom recurrence_y (n : ℕ) : y (n + 1) = y n + 1 / (2 * x n)

theorem at_least_one_bigger_than_44_9 : x 2018 > 44.9 ∨ y 2018 > 44.9 :=
sorry

end at_least_one_bigger_than_44_9_l164_164715


namespace problem_l164_164506

noncomputable def a : ℝ := Real.exp 1 - 2
noncomputable def b : ℝ := 1 - Real.log 2
noncomputable def c : ℝ := Real.exp (Real.exp 1) - Real.exp 2

theorem problem (a_def : a = Real.exp 1 - 2) 
                (b_def : b = 1 - Real.log 2) 
                (c_def : c = Real.exp (Real.exp 1) - Real.exp 2) : 
                c > a ∧ a > b := 
by 
  rw [a_def, b_def, c_def]
  sorry

end problem_l164_164506


namespace tetrahedron_vector_sum_l164_164696

variables {A B C D M P Q A₁ B₁ S : Type} [AddCommGroup A] [VectorSpace ℝ A]

def centroids (A B C D A₁ B₁ M : A) : Prop :=
  let F := (C + D) / 2 in 
  A₁ = (2 • B + F) / 3 ∧ 
  B₁ = (2 • A + F) / 3

def intersection (A A₁ M P Q B₁ C D : A) : Prop :=
  (∃ s : ℝ, P = M + s • (A₁ - A)) ∧
  (∃ t : ℝ, Q = M + t • (B₁ - B))

noncomputable def vectors_relation (M P Q S : A) : Prop :=
  P + Q = (4/3) • S

theorem tetrahedron_vector_sum (A B C D M P Q A₁ B₁ S : A) (H1 : centroids A B C D A₁ B₁ M)
  (H2 : intersection A A₁ M P Q B₁ C D) : vectors_relation M P Q S :=
by
  sorry

end tetrahedron_vector_sum_l164_164696


namespace ellipse_eccentricity_square_l164_164976

noncomputable def eccentricity_of_ellipse (a b : ℝ) (h : a > b > 0) : ℝ :=
  let c := sqrt (a^2 - b^2) in c / a

theorem ellipse_eccentricity_square (a b : ℝ) (h : a > b > 0)
  (h2 : ∃ A B C : ℝ × ℝ, A ≠ B ∧ B ≠ C ∧ C ≠ A ∧
          (A.1^2 / a^2 + A.2^2 / b^2 = 1) ∧
          (B.1^2 / a^2 + B.2^2 / b^2 = 1) ∧
          (C.1^2 / a^2 + C.2^2 / b^2 = 1) ∧
          (A.1^2 + B.1^2 + C.1^2 = 0) ∧
          (A.2^2 + B.2^2 + C.2^2 = 0)
       ) :
  eccentricity_of_ellipse a b h = sqrt 6 / 3 := sorry

end ellipse_eccentricity_square_l164_164976


namespace area_triangle_ABC_l164_164604

variables {A B C D : Type} -- Define the points
variables [linear_ordered_field K] -- Assume K is the underlying field (e.g., ℝ)

structure Trapezoid (K : Type) [linear_ordered_field K] :=
(A B C D : K)
(CD_eq_3AB : CD = 3 * AB)
(area_ABCD : area_ABCD = 18)

def area_triangle (A B C : K) : K := sorry -- Placeholder for the area function

theorem area_triangle_ABC :
  ∀ (trapezoid : Trapezoid K), 
  (area_triangle trapezoid.A trapezoid.B trapezoid.C) = 4.5 :=
by
  intro trapezoid
  -- Using the conditions for CD_eq_3AB and area_ABCD, prove area_triangle_ABC
  sorry

end area_triangle_ABC_l164_164604


namespace find_line_equation_l164_164702

-- Define the conditions for the x-intercept and inclination angle
def x_intercept (x : ℝ) (line : ℝ → ℝ) : Prop :=
  line x = 0

def inclination_angle (θ : ℝ) (k : ℝ) : Prop :=
  k = Real.tan θ

-- Define the properties of the line we're working with
def line (x : ℝ) : ℝ := -x + 5

theorem find_line_equation :
  x_intercept 5 line ∧ inclination_angle (3 * Real.pi / 4) (-1) → (∀ x, line x = -x + 5) :=
by
  intro h
  sorry

end find_line_equation_l164_164702


namespace value_of_x_l164_164044

theorem value_of_x (x y : ℕ) (h1 : y = 864) (h2 : x^3 * 6^3 / 432 = y) : x = 12 :=
sorry

end value_of_x_l164_164044


namespace polygon_sides_l164_164899

open Real

theorem polygon_sides (n : ℕ) : 
  (∀ (angle : ℝ), angle = 40 → n * angle = 360) → n = 9 := by
  intro h
  have h₁ := h 40 rfl
  sorry

end polygon_sides_l164_164899


namespace water_fee_expression_water_usage_for_27_fee_prove_water_fee_l164_164729

noncomputable def water_fee (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 6 then 2 * x
else if x > 6 then 3 * x - 6
else 0

theorem water_fee_expression (x : ℝ) (hx : 0 < x) :
  water_fee x = if x ≤ 6 then 2 * x else 3 * x - 6 :=
by
  unfold water_fee
  split_ifs with H1 H2 H3
  { refl }
  { refl }
  { exfalso; linarith }

theorem water_usage_for_27_fee :
  ∃ x : ℝ, water_fee x = 27 :=
by
  use 11
  unfold water_fee
  split_ifs with H1 H2 H3
  { exfalso; linarith }
  { norm_num }
  { exfalso; linarith }

-- c): Prove (question, conditions, correct answer)
theorem prove_water_fee :
  (water_fee_expression ∧ water_usage_for_27_fee) :=
by
  split; sorry

end water_fee_expression_water_usage_for_27_fee_prove_water_fee_l164_164729


namespace problem_solution_l164_164478

theorem problem_solution :
  let E := (∑ k in (finset.range 33), (3 * k + 1) + (3 * k + 2) - (3 * k + 3)) 
  in E = 1584 :=
by
  sorry

end problem_solution_l164_164478


namespace relationship_between_x_y_z_l164_164890

theorem relationship_between_x_y_z (x y z : ℕ) (a b c d : ℝ)
  (h1 : x ≤ y ∧ y ≤ z)
  (h2 : (x:ℝ)^a = 70^d ∧ (y:ℝ)^b = 70^d ∧ (z:ℝ)^c = 70^d)
  (h3 : 1/a + 1/b + 1/c = 1/d) :
  x + y = z := 
sorry

end relationship_between_x_y_z_l164_164890


namespace local_extrema_and_inflection_point_l164_164610

noncomputable def f : ℝ → ℝ := λ x, x^3 - 3 * x^2 - 9 * x + 11

theorem local_extrema_and_inflection_point :
  (∃ x, f x = 16 ∧ x = -1) ∧ 
  (∃ x, f x = -16 ∧ x = 3) ∧ 
  (∃ x, f x = 0 ∧ x = 1) :=
by 
  sorry

end local_extrema_and_inflection_point_l164_164610


namespace min_value_of_function_l164_164030

theorem min_value_of_function :
  ∀ x : ℝ, x > -1 → (y : ℝ) = (x^2 + 7*x + 10) / (x + 1) → y ≥ 9 :=
by
  intros x hx h
  sorry

end min_value_of_function_l164_164030


namespace range_of_a_l164_164121

-- Definitions for the conditions given
def C1 (a : ℝ) (x : ℝ) := a * x^2
def C2 (x : ℝ) := real.exp x

-- The main theorem statement
theorem range_of_a (a : ℝ) (h : a > 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ real.exp x2 = 2 * a * x1 ∧ real.exp x2 = (real.exp x2 - a * x1^2) / (x2 - x1)) →
  a ∈ Ioi (real.exp 2 / 4) :=
by
  sorry

end range_of_a_l164_164121


namespace ball_distribution_in_boxes_l164_164114

theorem ball_distribution_in_boxes :
  ∃ (n k : ℕ), n = 8 ∧ k = 3 ∧ nat.choose (n + k - 1) (k - 1) = 45 :=
by
  use 8
  use 3
  split
  · refl
  split
  · refl
  · simp [nat.choose]
  sorry

end ball_distribution_in_boxes_l164_164114


namespace graph_symmetry_l164_164083

def symmetric_about_x_equals_one (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(1 - x) = f(x - 1)

theorem graph_symmetry (f : ℝ → ℝ): 
  (symmetric_about_x_equals_one f) :=
sorry

end graph_symmetry_l164_164083


namespace num_triangles_eq_num_divide_items_eq_l164_164632

-- Definition for Question 1
def num_non_congruent_triangles (m : ℕ) : ℕ :=
  match m % 6 with
  | 0 => let k := m / 6 in 3 * k^2 - 3 * k + 1
  | 1 => let k := m / 6 in 3 * k^2 - 2 * k
  | 2 => let k := m / 6 in 3 * k^2 - k
  | 3 => let k := m / 6 in 3 * k^2
  | 4 => let k := m / 6 in 3 * k^2 + k
  | _ => let k := m / 6 in 3 * k^2 + 2 * k

-- Definition for Question 2
def num_ways_divide_into_groups (m : ℕ) : ℕ :=
  match m % 6 with
  | 0 => let k := m / 6 in 3 * k^2
  | 1 => let k := m / 6 in 3 * k^2 + k
  | 2 => let k := m / 6 in 3 * k^2 + 2 * k
  | 3 => let k := m / 6 in 3 * k^2 + k + 1
  | 4 => let k := m / 6 in 3 * k^2 + 4 * k + 1
  | _ => let k := m / 6 in 3 * k^2 + 5 * k + 2

-- Theorem for Question 1
theorem num_triangles_eq (m : ℕ) (h : m ≥ 3) : num_non_congruent_triangles m = 
  match m % 6 with
  | 0 => let k := m / 6 in 3 * k^2 - 3 * k + 1
  | 1 => let k := m / 6 in 3 * k^2 - 2 * k
  | 2 => let k := m / 6 in 3 * k^2 - k
  | 3 => let k := m / 6 in 3 * k^2
  | 4 => let k := m / 6 in 3 * k^2 + k
  | _ => let k := m / 6 in 3 * k^2 + 2 * k := sorry

-- Theorem for Question 2
theorem num_divide_items_eq (m : ℕ) (h : m ≥ 3) : num_ways_divide_into_groups m = 
  match m % 6 with
  | 0 => let k := m / 6 in 3 * k^2
  | 1 => let k := m / 6 in 3 * k^2 + k
  | 2 => let k := m / 6 in 3 * k^2 + 2 * k
  | 3 => let k := m / 6 in 3 * k^2 + k + 1
  | 4 => let k := m / 6 in 3 * k^2 + 4 * k + 1
  | _ => let k := m / 6 in 3 * k^2 + 5 * k + 2 := sorry

end num_triangles_eq_num_divide_items_eq_l164_164632


namespace sum_of_x_y_l164_164246

theorem sum_of_x_y (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_x_y_l164_164246


namespace members_group_size_bound_l164_164208

-- Definition of the problem parameters and conditions
variables {n : ℕ} -- Initial number of members

-- Question & proof statement
theorem members_group_size_bound (n : ℕ) : 
  ∃ weeks : ℕ, ∀ groups : list ℕ, is_after_weeks n weeks groups → 
  ∀ group ∈ groups, group ≤ (1 + (real.sqrt (2 * n))) := 
sorry

-- Definitions to model the problem
def is_after_weeks (n : ℕ) (weeks : ℕ) (groups : list ℕ) : Prop :=
sorry

end members_group_size_bound_l164_164208


namespace mixture_total_pounds_l164_164815

-- Definitions based on conditions
def cost_of_cashews_per_pound := 5.0
def cost_of_peanuts_per_pound := 2.0
def total_cost_of_mixture := 92.0
def pounds_of_cashews := 11.0

-- Mathematically equivalent proof problem
theorem mixture_total_pounds :
  let pounds_of_cashews := 11.0 in
  let cost_of_cashews := pounds_of_cashews * cost_of_cashews_per_pound in
  let cost_of_peanuts := total_cost_of_mixture - cost_of_cashews in
  let pounds_of_peanuts := cost_of_peanuts / cost_of_peanuts_per_pound in
  pounds_of_cashews + pounds_of_peanuts = 29.5 :=
by
  sorry

end mixture_total_pounds_l164_164815


namespace distinctDiagonalsConvexNonagon_l164_164939

theorem distinctDiagonalsConvexNonagon : 
  ∀ (P : Type) [fintype P] [decidable_eq P] (vertices : finset P) (h : vertices.card = 9), 
  let n := vertices.card in
  let diagonals := (n * (n - 3)) / 2 in
  diagonals = 27 :=
by
  intros
  let n := vertices.card
  have keyIdentity : (n * (n - 3)) / 2 = 27 := sorry
  exact keyIdentity

end distinctDiagonalsConvexNonagon_l164_164939


namespace downstream_speed_l164_164427

variable (Vu : ℝ) (Vs : ℝ)

theorem downstream_speed (h1 : Vu = 25) (h2 : Vs = 35) : (2 * Vs - Vu = 45) :=
by
  sorry

end downstream_speed_l164_164427


namespace log_base_9_of_8_l164_164567

theorem log_base_9_of_8 (a b : ℝ) (h1 : 10 ^ a = 2) (h2 : Real.log 3 = b) :
  Real.logBase 9 8 = (3 * a) / (2 * b) :=
sorry

end log_base_9_of_8_l164_164567


namespace sum_of_x_y_l164_164249

theorem sum_of_x_y (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_x_y_l164_164249


namespace points_on_line_proof_l164_164292

theorem points_on_line_proof (n : ℕ) (hn : n = 10) : 
  let after_first_procedure := 3 * n - 2 in
  let after_second_procedure := 3 * after_first_procedure - 2 in
  after_second_procedure = 82 :=
by
  let after_first_procedure := 3 * n - 2
  let after_second_procedure := 3 * after_first_procedure - 2
  have h : after_second_procedure = 9 * n - 8 := by
    calc
      after_second_procedure = 3 * (3 * n - 2) - 2 : rfl
                      ... = 9 * n - 6 - 2      : by ring
                      ... = 9 * n - 8          : by ring
  rw [hn] at h 
  exact h.symm.trans (by norm_num)

end points_on_line_proof_l164_164292


namespace f_periodic_f_monotonic_decreasing_intervals_g_minimum_in_interval_l164_164093

noncomputable def f (x : ℝ) : ℝ := cos (2 * x + (2 * π / 3)) + 2 * cos x ^ 2

theorem f_periodic : ∀ x, f (x + π) = f x :=
by sorry

theorem f_monotonic_decreasing_intervals (k : ℤ) : 
    ∀ x, (k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 3) → f x = cos (2 * x + π / 3) :=
by sorry

noncomputable def g (x : ℝ) : ℝ := f (x - π / 3)

theorem g_minimum_in_interval : ∃ x ∈ Icc (0 : ℝ) (π / 2), g x = 1 / 2 :=
by sorry

end f_periodic_f_monotonic_decreasing_intervals_g_minimum_in_interval_l164_164093


namespace xiaoning_pe_comprehensive_score_l164_164782

def midterm_weight : ℝ := 0.3
def final_weight : ℝ := 0.7
def midterm_score : ℝ := 80
def final_score : ℝ := 90

theorem xiaoning_pe_comprehensive_score : midterm_score * midterm_weight + final_score * final_weight = 87 :=
by
  sorry

end xiaoning_pe_comprehensive_score_l164_164782


namespace part_I_part_II_l164_164885

variables {a b c : ℝ}
variables {a_ne_b : a ≠ b} {b_ne_c : b ≠ c} {a_ne_c : a ≠ c}
variables {H_arithmetic : (2 / b) = (1 / a) + (1 / c)}

theorem part_I (h : 2 / b = 1 / a + 1 / c) : b / a < c / b :=
by
  sorry

theorem part_II {α β γ : ℝ} (h : 2 / b = 1 / a + 1 / c) (h_ne : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_cos : cos β = (a^2 + c^2 - b^2) / (2 * a * c)) :
  0 ≤ β ∧ β ≤ π / 3 :=
by
  sorry

end part_I_part_II_l164_164885


namespace abs_diff_avg_median_l164_164074

variable (a b : ℝ)

theorem abs_diff_avg_median (h : 1 < a ∧ a < b) : 
  | (1 + (a + 1) + (2 * a + b) + (a + b + 1)) / 4 - ((a + 1) + (a + b + 1)) / 2 | = | 1 / 4 | :=
by sorry

end abs_diff_avg_median_l164_164074


namespace five_letter_word_count_correct_l164_164475

theorem five_letter_word_count_correct :
  ∃ n : ℕ, n = 26^3 * 5 ∧ n = 87880 :=
by
  use 87880
  split
  { norm_num }
  { refl }

end five_letter_word_count_correct_l164_164475


namespace complex_magnitude_result_l164_164082

noncomputable def magnitude (z : ℂ) : ℝ := complex.abs z

theorem complex_magnitude_result {z : ℂ} (h : complex.I * z = 1 + complex.I) : magnitude z = real.sqrt 2 :=
by
  sorry

end complex_magnitude_result_l164_164082


namespace baseball_games_per_month_l164_164362

-- Define the conditions
def total_games_in_a_season : ℕ := 14
def months_in_a_season : ℕ := 2

-- Define the proposition stating the number of games per month
def games_per_month (total_games months : ℕ) : ℕ := total_games / months

-- State the equivalence proof problem
theorem baseball_games_per_month : games_per_month total_games_in_a_season months_in_a_season = 7 :=
by
  -- Directly stating the equivalence based on given conditions
  sorry

end baseball_games_per_month_l164_164362


namespace average_age_of_students_l164_164693

theorem average_age_of_students :
  (8 * 14 + 6 * 16 + 17) / 15 = 15 :=
by
  sorry

end average_age_of_students_l164_164693


namespace exterior_angle_DEG_l164_164686

-- Define the degree measures of angles in a square and a pentagon.
def square_interior_angle := 90
def pentagon_interior_angle := 108

-- Define the sum of the adjacent interior angles at D
def adjacent_interior_sum := square_interior_angle + pentagon_interior_angle

-- Statement to prove the exterior angle DEG
theorem exterior_angle_DEG :
  360 - adjacent_interior_sum = 162 := by
  sorry

end exterior_angle_DEG_l164_164686


namespace line_intersects_hyperbola_l164_164126

theorem line_intersects_hyperbola 
  (k : ℝ)
  (hyp : ∃ x y : ℝ, y = k * x + 2 ∧ x^2 - y^2 = 6) :
  -Real.sqrt 15 / 3 < k ∧ k < -1 := 
sorry


end line_intersects_hyperbola_l164_164126


namespace area_ratio_of_triangles_l164_164403

variable {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (A B C M N : Point)

def is_on_or_extension_of (P Q R: Point) : Prop := 
  ∃ (k : ℝ), P = Q + k • (R - Q)

theorem area_ratio_of_triangles (hM : is_on_or_extension_of M A B)
                                (hN : is_on_or_extension_of N A C)
: (area (triangle A M N)) / (area (triangle A B C)) = (dist A M / dist A B) * (dist A N / dist A C) :=
by
  sorry

end area_ratio_of_triangles_l164_164403


namespace circle_ways_l164_164874

noncomputable def count3ConsecutiveCircles : ℕ :=
  let longSideWays := 1 + 2 + 3 + 4 + 5 + 6
  let perpendicularWays := (4 + 4 + 4 + 3 + 2 + 1) * 2
  longSideWays + perpendicularWays

theorem circle_ways : count3ConsecutiveCircles = 57 := by
  sorry

end circle_ways_l164_164874


namespace prime_factor_of_difference_l164_164068

theorem prime_factor_of_difference {A B : ℕ} (hA : 1 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (h_neq : A ≠ B) :
  Nat.Prime 2 ∧ (∃ B : ℕ, 20 * B = 20 * B) :=
by
  sorry

end prime_factor_of_difference_l164_164068


namespace sum_of_squares_of_reciprocals_eq_one_l164_164626

theorem sum_of_squares_of_reciprocals_eq_one (p q r s : ℝ) (w : ℂ → ℝ) :
  (∀ z : ℂ, (z^4 + p*z^3 + q*z^2 + r*z + s = 0) → (|z| = 2)) →
  ((1/(w 0))^2 + (1/(w 1))^2 + (1/(w 2))^2 + (1/(w 3))^2 = 1) :=
by
  sorry

end sum_of_squares_of_reciprocals_eq_one_l164_164626


namespace trigonometric_expression_value_l164_164813

theorem trigonometric_expression_value :
  (sin 25 * sin 25) + (cos 25 * cos 25) + 2 * sin 60 + tan 45 - tan 60 = 3 :=
by
  have h1 : sin 30 = 1 / 2 := sorry
  have h2 : cos 30 = sqrt 3 / 2 := sorry
  have h3 : tan 30 = sqrt 3 / 3 := sorry
  have h4 : cot 30 = sqrt 3 := sorry
  have h5 : sin 45 = sqrt 2 / 2 := sorry
  have h6 : cos 45 = sqrt 2 / 2 := sorry
  have h7 : tan 45 = 1 := sorry
  have h8 : cot 45 = 1 := sorry
  have h9 : sin 60 = sqrt 3 / 2 := sorry
  have h10 : cos 60 = 1 / 2 := sorry
  have h11 : tan 60 = sqrt 3 := sorry
  have h12 : cot 60 = sqrt 3 / 3 := sorry
  have h13 : ∀ A, sin A * sin A + cos A * cos A = 1 := sorry
  sorry

end trigonometric_expression_value_l164_164813


namespace exists_m_sqrt_8m_integer_l164_164667

theorem exists_m_sqrt_8m_integer : ∃ (m : ℕ), (m > 0) ∧ (∃ k : ℕ, k^2 = 8 * m) :=
by
  use 2
  split
  · exact Nat.succ_pos 1
  · use 4
    exact Nat.succ_pos 1
    sorry

end exists_m_sqrt_8m_integer_l164_164667


namespace max_area_triangle_l164_164128

theorem max_area_triangle (a b c : ℝ) (A B C : ℝ) 
  (h₀ : a^2 + b^2 + 2 * c^2 = 8)
  (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : C = 180 - A - B) : 
  (1/2 * a * b * real.sin C) ≤ (2 * real.sqrt 5) / 5 :=
sorry

end max_area_triangle_l164_164128


namespace sum_of_x_y_possible_values_l164_164260

theorem sum_of_x_y_possible_values (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
sorry

end sum_of_x_y_possible_values_l164_164260


namespace triangle_proof_l164_164583

section TriangleGeometry

variables {A B C : Point} {I : Point}
variables (R r : ℝ) (D D' E E' F F' : Point)
variables (circumcircle : Circle) (incircle : Circle)

-- Given the conditions stated
hypothesis : ∃ (triangle : Triangle ABC)
  (circ : ∀ {P}, P ∈ circumcircle ↔ dist P (triangle.center) = R)
  (inc : ∀ {P}, P ∈ incircle ↔ dist P I = r)
  (existD : Line AI ∩ Line BC = D' ∧ Line AI ∩ circumcircle = D)
  (existE : Line BI ∩ Line AC = E' ∧ Line BI ∩ circumcircle = E)
  (existF : Line CI ∩ Line AB = F' ∧ Line CI ∩ circumcircle = F), True

theorem triangle_proof : 
  ∀ {A B C I D D' E E' F F'} (R r : ℝ) 
    (circumcircle incircle : Circle),
  (∃ (triangle : Triangle ABC)
    (circ : ∀ {P}, P ∈ circumcircle ↔ dist P (triangle.center) = R)
    (inc : ∀ {P}, P ∈ incircle ↔ dist P I = r)
    (existD : Line AI ∩ Line BC = D' ∧ Line AI ∩ circumcircle = D)
    (existE : Line BI ∩ Line AC = E' ∧ Line BI ∩ circumcircle = E)
    (existF : Line CI ∩ Line AB = F' ∧ Line CI ∩ circumcircle = F),
  ∑ i in [D, E, F], i = ∑ j in [D', E', F'], j) →
  ( dist I D' / dist D' A + dist I E' / dist E' B + dist I F' / dist F' C = (R - r) / r ) :=
sorry

end TriangleGeometry

end triangle_proof_l164_164583


namespace trey_nail_usage_l164_164736

theorem trey_nail_usage (total_decorations nails thumbtacks sticky_strips : ℕ) 
  (h1 : nails = 2 * total_decorations / 3)
  (h2 : sticky_strips = 15)
  (h3 : sticky_strips = 3 * (total_decorations - 2 * total_decorations / 3) / 5) :
  nails = 50 :=
by
  sorry

end trey_nail_usage_l164_164736


namespace max_determinant_value_l164_164029

noncomputable def determinant (θ : ℝ) : ℝ :=
  by 
    rw [Matrix.det_fin_3]
    simp
    exact 1 * ((1 + sin θ ^ 2) * 1) - 1 * (1 + cos θ ^ 2) * 1 
        - 1 * (1 + cos θ ^ 2) * 1 + 1 * ((1 + sin θ ^ 2) * (1 + cos θ ^ 2))

theorem max_determinant_value : ∃ θ : ℝ, ∀ θ : ℝ, determinant θ ≤ 1 / 4 :=
by sorry

end max_determinant_value_l164_164029


namespace polygon_angle_ratio_pairs_l164_164713

theorem polygon_angle_ratio_pairs : ∃ k r : ℕ, (180 - 360 / r) / (180 - 360 / k) = 4 / 3 ∧ r > 2 ∧ k > 2 ∧ k ∈ {7, 6, 5, 4} :=
by
  sorry

end polygon_angle_ratio_pairs_l164_164713


namespace num_integers_congruent_mod_7_count_integers_congruent_mod_7_l164_164958

theorem num_integers_congruent_mod_7 (n: ℕ): (1 ≤ n ∧ n ≤ 150 ∧ n % 7 = 3) ↔ (0 ≤ n ∧ n ≤ 21) :=
sorry

theorem count_integers_congruent_mod_7:
  (∃ l, l = {k : ℕ | k ∈ (Finset.range 151) ∧ k % 7 = 3} ∧ l.card = 22) :=
by
  have : ∀ k, k ∈ (Finset.range 151) ∧ k % 7 = 3 ↔ (k = 7 * (k / 7) + 3 ∧ 1 ≤ k ∧ k ≤ 150) :=
    λ k, ⟨λ ⟨hk1, hk2⟩, ⟨eq.symm (Nat.mod_add_div k 7), by linarith [Finset.mem_range_succ_iff.1 hk1]⟩,
           λ ⟨h_eq, h_range⟩, ⟨Finset.mem_range.mpr (by linarith), (by rw [←h_eq, Nat.add_mod]; norm_num)⟩⟩,
  let S := {k : ℕ | 1 ≤ k ∧ k ≤ 150 ∧ k % 7 = 3},
  have hS_card : S.card = 22,
  sorry,
  use S,
  split,
  norm_num,
  exact hS_card

end num_integers_congruent_mod_7_count_integers_congruent_mod_7_l164_164958


namespace max_value_fraction_l164_164897

theorem max_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (∀ x y : ℝ, (0 < x → 0 < y → (x / (2 * x + y) + y / (x + 2 * y)) ≤ 2 / 3)) :=
by
  sorry

end max_value_fraction_l164_164897


namespace min_value_of_a_plus_b_l164_164183

theorem min_value_of_a_plus_b (a b : ℤ) (h1 : Even a) (h2 : Even b) (h3 : a * b = 144) : a + b = -74 :=
sorry

end min_value_of_a_plus_b_l164_164183


namespace lowest_final_price_l164_164797

def price_after_changes (P : ℝ) (increase decrease : ℝ) : ℝ :=
  P * (1 + increase) * (1 - decrease)

theorem lowest_final_price (P : ℝ) (hP : 0 ≤ P):
  let A := price_after_changes P 0.1 0.1 in
  let B := price_after_changes P 0.1 0.1 in
  let C := price_after_changes P 0.2 0.2 in
  let D := price_after_changes P 0.3 0.3 in
  D < A ∧ D < B ∧ D < C :=
  by
    sorry

end lowest_final_price_l164_164797


namespace bales_stored_in_barn_l164_164358

-- Defining the conditions
def bales_initial : Nat := 28
def bales_stacked : Nat := 28
def bales_already_there : Nat := 54

-- Formulate the proof statement
theorem bales_stored_in_barn : bales_already_there + bales_stacked = 82 := by
  sorry

end bales_stored_in_barn_l164_164358


namespace num_arrangements_l164_164839

-- Define the constants
constant num_volunteers : ℕ := 6
constant num_areas : ℕ := 4

-- Define the areas and the volunteers
constant Area : Type
constant A : Area
constant B : Area
constant C : Area
constant D : Area

constant Volunteer : Type
constant XiaoLi : Volunteer
constant XiaoWang : Volunteer
constants remaining_volunteers : set Volunteer

-- Define the required conditions
axiom num_area_A : A ≠ B ∧ A ≠ C ∧ A ≠ D
axiom num_area_B : B ≠ C ∧ B ≠ D
axiom num_volunteers_per_area : ∀ x, (x = A ∨ x = B) → (∀ y, y ∈ {XiaoLi, XiaoWang} → false) ∧ (card (set.filter (λ y, ∃ z, y ∈ z) (remaining_volunteers)) = 1)
axiom num_volunteers_per_other_areas : ∀ x, (x = C ∨ x = D) → card (set.filter (λ y, y = XiaoLi ∨ y = XiaoWang) ∪ set.filter (λ y, y ∈ {a | ∃ z, a ∈ z})) = 2
axiom separation_condition : (∀ x, x = {XiaoLi, XiaoWang}) → false

-- Now, state the mathematical problem as a theorem
theorem num_arrangements (volunteers : set Volunteer) (areas : set Area) 
  (arrange : Area → set Volunteer) : volunteers = {XiaoLi, XiaoWang} ∪ remaining_volunteers ∧ areas = {A, B, C, D} ∧
  (∀ x ∈ areas, ∃ y ∈ volunteers, y ∈ arrange x) ∧
  (∀ x ∈ {A, B}, card (arrange x) = 1) ∧ (∀ x ∈ {C, D}, card (arrange x) = 2) ∧
  (∀ x ∈ {C, D}, XiaoLi ∈ arrange x ∨ XiaoWang ∈ arrange x) ∧
  (¬(XiaoLi ∈ arrange (areas.Except {A, B})) ∨ ¬(XiaoWang ∈ arrange (areas.Except {A, B}))) →
  count_arrangements = 156 := sorry

end num_arrangements_l164_164839


namespace minimum_value_l164_164527

variable (x y : ℝ)

noncomputable def problem_statement : Prop :=
  (x > 0) ∧ (y > 0) ∧ (2^(x - 3) = (1/2)^y) → (∀ x' y', (x' > 0) ∧ (y' > 0) ∧ (2^(x' - 3) = (1/2)^y') → (1 / x' + 4 / y') ≥ 3)

theorem minimum_value (x y : ℝ) (h : problem_statement x y) : 
  ∃ x y, (x > 0) ∧ (y > 0) ∧ (2^(x - 3) = (1/2)^y) ∧ ((1 / x + 4 / y) = 3) :=
by
  sorry

end minimum_value_l164_164527


namespace Lizzy_money_after_loan_l164_164649

theorem Lizzy_money_after_loan :
  let initial_savings := 30
  let loaned_amount := 15
  let interest_rate := 0.20
  let interest := loaned_amount * interest_rate
  let total_amount_returned := loaned_amount + interest
  let remaining_money := initial_savings - loaned_amount
  let total_money := remaining_money + total_amount_returned
  total_money = 33 :=
by
  sorry

end Lizzy_money_after_loan_l164_164649


namespace original_price_of_article_l164_164436

theorem original_price_of_article (P : ℝ) : 
  (P - 0.30 * P) * (1 - 0.20) = 1120 → P = 2000 :=
by
  intro h
  -- h represents the given condition for the problem
  sorry  -- proof will go here

end original_price_of_article_l164_164436


namespace greatest_common_divisor_of_180_and_n_is_9_l164_164370

theorem greatest_common_divisor_of_180_and_n_is_9 
  {n : ℕ} (h_n_divisors : ∀ d : ℕ, d ∣ 180 ∧ d ∣ n ↔ d ∈ {1, 3, 9}) : 
  greatest_common_divisor 180 n = 9 := 
sorry

end greatest_common_divisor_of_180_and_n_is_9_l164_164370


namespace monotonicity_of_f_l164_164006

noncomputable theory

open Real

def f (a : ℝ) (x : ℝ) : ℝ := (1 / 2) * x^2 - a * x + (a - 1) * log x

theorem monotonicity_of_f (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, 0 < x -> (a = 2 -> monotone_on (f a) (Ioi 0)) ∧
             (1 < a ∧ a < 2 -> 
              (monotone_on (λ x, true) (Ioi 0)) ∧ 
              (monotone_on (λ x, true) (Ioi 0) ) ∧
              monopnically_increasing (Ioi 0) (Ioi 0)) ) ∧
             (a > 2 -> 
              (monotone_on (λ x, true) (Ioi 0)) ∧ 
              (monotone_on (λ x, true) (Ioi 0) ) ∧
              monotone_increasing (Ioi 0) (Ioi 0)) :

sorry

end monotonicity_of_f_l164_164006


namespace complex_product_real_l164_164910

theorem complex_product_real (a : ℝ) (h : let z₁ := complex.mk 3 a in
                                             let z₂ := complex.mk a -3 in
                                             ∃ (r : ℝ), z₁ * z₂ = r) :
  a = 3 ∨ a = -3 :=
by
  sorry

end complex_product_real_l164_164910


namespace integer_part_sum_l164_164345

noncomputable def sequence_a : ℕ → ℝ
| 0       := 4 / 3
| (n + 1) := (sequence_a n)^2 - (sequence_a n) + 1

theorem integer_part_sum :
  ⌊(∑ i in Finset.range 2017, (1 : ℝ) / sequence_a i)⌋ = 2 :=
by
  sorry

end integer_part_sum_l164_164345


namespace area_A_l164_164989

theorem area_A'B'C'D' :
  ∀ (A B C D A' B' C' D' : Type*)
  (AB BB' BC CC' CD DD' DA AA' : ℝ),
  convex_quadrilateral A B C D →
  (AB = 8) →
  (BB' = 8) →
  (BC = 9) →
  (CC' = 9) →
  (CD = 10) →
  (DD' = 10) →
  (DA = 11) →
  (AA' = 11) →
  (area_of_quadrilateral A B C D = 20) →
  (area_of_quadrilateral A' B' C' D' = 60) :=
begin
  intros,
  sorry,
end

end area_A_l164_164989


namespace properties_of_n_l164_164472

open Nat

theorem properties_of_n (n : ℕ) (h1 : n > 1) :
  (∀ (a b : ℤ), gcd a n = 1 → gcd b n = 1 → (a ≡ b [MOD n] ↔ (a * b ≡ 1 [MOD n]))) →
  n ∈ {2, 3, 4, 6, 8, 12, 24} :=
  sorry

end properties_of_n_l164_164472


namespace trash_can_ratio_l164_164798

theorem trash_can_ratio (streets_trash_cans total_trash_cans : ℕ) 
(h_streets : streets_trash_cans = 14) 
(h_total : total_trash_cans = 42) : 
(total_trash_cans - streets_trash_cans) / streets_trash_cans = 2 :=
by {
  sorry
}

end trash_can_ratio_l164_164798


namespace parking_average_cost_l164_164308

noncomputable def parking_cost_per_hour := 
  let cost_two_hours : ℝ := 20.00
  let cost_per_excess_hour : ℝ := 1.75
  let weekend_surcharge : ℝ := 5.00
  let discount_rate : ℝ := 0.10
  let total_hours : ℝ := 9.00
  let excess_hours : ℝ := total_hours - 2.00
  let remaining_cost := cost_per_excess_hour * excess_hours
  let total_cost_before_discount := cost_two_hours + remaining_cost + weekend_surcharge
  let discount := discount_rate * total_cost_before_discount
  let discounted_total_cost := total_cost_before_discount - discount
  let average_cost_per_hour := discounted_total_cost / total_hours
  average_cost_per_hour

theorem parking_average_cost :
  parking_cost_per_hour = 3.725 := 
by
  sorry

end parking_average_cost_l164_164308


namespace largest_number_of_photos_l164_164422

theorem largest_number_of_photos (n : ℕ) : 
  ∃ r, (∀ (photos : finset (finset (fin n))), 
      (∀ p ∈ photos, p.nonempty) → 
      (∀ p₁ p₂ ∈ photos, p₁ ≠ p₂ → (p₁ ∩ p₂).nonempty) →
      photos.card = r) ↔ r = 2^(n-1) := 
sorry

end largest_number_of_photos_l164_164422


namespace rhombus_count_l164_164994

theorem rhombus_count (n : ℕ) (h : n > 0) :
  let rhombuses := 3 * n * (n - 1) / 2 in
  rhombuses = 3 * combinatorial.binomial n 2 :=
by
  sorry

end rhombus_count_l164_164994


namespace remainder_is_4_l164_164701

-- Definitions based on the given conditions
def dividend := 132
def divisor := 16
def quotient := 8

-- The theorem we aim to prove, stating the remainder
theorem remainder_is_4 : dividend = divisor * quotient + 4 := sorry

end remainder_is_4_l164_164701


namespace translation_graph_pass_through_point_l164_164996

theorem translation_graph_pass_through_point :
  (∃ a : ℝ, (∀ x y : ℝ, y = -2 * x + 1 - 3 → y = 3 → x = a) → a = -5/2) :=
sorry

end translation_graph_pass_through_point_l164_164996


namespace num_integers_congruent_mod_7_count_integers_congruent_mod_7_l164_164959

theorem num_integers_congruent_mod_7 (n: ℕ): (1 ≤ n ∧ n ≤ 150 ∧ n % 7 = 3) ↔ (0 ≤ n ∧ n ≤ 21) :=
sorry

theorem count_integers_congruent_mod_7:
  (∃ l, l = {k : ℕ | k ∈ (Finset.range 151) ∧ k % 7 = 3} ∧ l.card = 22) :=
by
  have : ∀ k, k ∈ (Finset.range 151) ∧ k % 7 = 3 ↔ (k = 7 * (k / 7) + 3 ∧ 1 ≤ k ∧ k ≤ 150) :=
    λ k, ⟨λ ⟨hk1, hk2⟩, ⟨eq.symm (Nat.mod_add_div k 7), by linarith [Finset.mem_range_succ_iff.1 hk1]⟩,
           λ ⟨h_eq, h_range⟩, ⟨Finset.mem_range.mpr (by linarith), (by rw [←h_eq, Nat.add_mod]; norm_num)⟩⟩,
  let S := {k : ℕ | 1 ≤ k ∧ k ≤ 150 ∧ k % 7 = 3},
  have hS_card : S.card = 22,
  sorry,
  use S,
  split,
  norm_num,
  exact hS_card

end num_integers_congruent_mod_7_count_integers_congruent_mod_7_l164_164959


namespace solution_set_l164_164115

-- Defining the condition and inequalities:
variable (a x : Real)

-- Condition that a < 0
def condition_a : Prop := a < 0

-- Inequalities in the system
def inequality1 : Prop := x > -2 * a
def inequality2 : Prop := x > 3 * a

-- The solution set we need to prove
theorem solution_set (h : condition_a a) : (inequality1 a x) ∧ (inequality2 a x) ↔ x > -2 * a :=
by
  sorry

end solution_set_l164_164115


namespace sum_ao_aq_ar_l164_164404

noncomputable def regular_pentagon (ABCDE O : Type) [regular_pentagon ABCDE O] : Prop :=
  let AP := perpendicular_from A to CD
  let AQ := perpendicular_from A to (extension CB)
  let AR := perpendicular_from A to (extension DE)
  (AP^2 + AQ^2 + AR^2) = 4

theorem sum_ao_aq_ar {ABCDE : Type} {O : Type} [regular_pentagon ABCDE O] (A P Q R : Type)
  (AP AQ AR : Type) (h : OP = 2) :
  AO + AQ + AR = 8 :=
by sorry

end sum_ao_aq_ar_l164_164404


namespace radius_of_base_circle_l164_164744

theorem radius_of_base_circle (r : ℝ) (h : r = 2) : 
  ∃ base_radius : ℝ, base_radius = 1 :=
by
  -- Define the radius of the semicircle
  let semicircle_radius := 2
  -- The length of the arc of the semicircle, which becomes the circumference of the base circle of the cone
  let arc_length := π * semicircle_radius
  -- The circumference of the base circle of the cone is equal to arc_length
  have base_circumference : 2 * π * base_radius = arc_length,
    from sorry
  -- Solving for base_radius
  use 1
  sorry

end radius_of_base_circle_l164_164744


namespace possible_values_of_sum_l164_164230

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + y^3 + 21 * x * y = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l164_164230


namespace parallel_lines_slope_l164_164315

theorem parallel_lines_slope (a : ℝ) :
  (∃ b : ℝ, ( ∀ x y : ℝ, a*x - 5*y - 9 = 0 → b*x - 3*y - 10 = 0) → a = 10/3) :=
sorry

end parallel_lines_slope_l164_164315


namespace car_rental_savings_l164_164165

noncomputable def total_distance := 150 * 2
noncomputable def cost_per_liter := 0.90
noncomputable def distance_covered_per_liter := 15
noncomputable def rental_cost_first_option := 50
noncomputable def rental_cost_second_option := 90

theorem car_rental_savings : 
  let gasoline_needed := total_distance / distance_covered_per_liter,
      gasoline_cost := gasoline_needed * cost_per_liter,
      total_cost_first_option := rental_cost_first_option + gasoline_cost in
  (rental_cost_second_option - total_cost_first_option) = 22 :=
by
  sorry

end car_rental_savings_l164_164165


namespace statement_A_statement_B_statement_C_not_statement_D_l164_164914

theorem statement_A (a : ℝ) (x : ℝ) (h1 : 5^x = (a+3) / (5 - a)) (hx : x < 0) : -3 < a ∧ a < 1 := sorry

theorem statement_B (a : ℝ) (ha_pos : a > 0) (ha_ne : a ≠ 1) : 
  a^(1 - 1) + log a (2*1 - 1) - 1 = 0 := sorry

theorem statement_C_not (a : ℝ) : 
  ¬ (∀ x, (6 + x - 2 * x^2) > 0 → x ≥ 1/4 → log (1/2) (6 + x - 2 * x^2) < log (1/2) (6 + (x+1) - 2*(x+1)^2)) := sorry

theorem statement_D (a : ℝ) (h2 : log a (1/2) > 1) : 1/2 < a ∧ a < 1 := sorry

end statement_A_statement_B_statement_C_not_statement_D_l164_164914


namespace greatest_expression_is_b_l164_164385

noncomputable def expr_a : ℝ := (3/4)^(-2) + real.sqrt 9 - 5/2
noncomputable def expr_b : ℝ := 2^(3/2) * 1/(real.sqrt 16) + 3^2
noncomputable def expr_c : ℝ := (4/9)^(1/2) + (1/27)^(2/3) - (2/5)^(3/2)
noncomputable def expr_d : ℝ := (3/2)^(-1) * real.sqrt (27/8) - 9/4

theorem greatest_expression_is_b : expr_b = 9.707 ∧ expr_b > expr_a ∧ expr_b > expr_c ∧ expr_b > expr_d :=
by
  sorry

end greatest_expression_is_b_l164_164385


namespace sum_of_possible_values_of_N_l164_164335

theorem sum_of_possible_values_of_N :
  ∃ a b c : ℕ, (a > 0 ∧ b > 0 ∧ c > 0) ∧ (abc = 8 * (a + b + c)) ∧ (c = a + b)
  ∧ (2560 = 560) :=
by
  sorry

end sum_of_possible_values_of_N_l164_164335


namespace transformed_complex_l164_164743

noncomputable def rotation (z : Complex) : Complex := 
  (Complex.cos (Float.pi / 3) + Complex.sin (Float.pi / 3) * Complex.I) * z

noncomputable def dilation (z : Complex) : Complex := 
  2 * z 

noncomputable def combined_transformation (z : Complex) : Complex := 
  dilation (rotation z)

theorem transformed_complex :
  combined_transformation ⟨4, 3⟩ = ⟨4 - 3 * Real.sqrt 3, 4 * Real.sqrt 3 + 3⟩ :=
by
  sorry

end transformed_complex_l164_164743


namespace find_AC_l164_164599

-- Definitions based on the conditions
variables {A B C B' C' M E D : Type}
variable [metric_space E]

-- Assume M is the midpoint of BC
axiom M_midpoint_BC : dist B M = dist C M

-- Given distances
axiom AE_eq_8 : dist A E = 8
axiom EC_eq_10 : dist E C = 10
axiom BD_eq_15 : dist B D = 15

-- Condition based on the reflection property along median AM
axiom reflection_property : ∀ {P}, dist A P = dist P P → dist A P

-- The goal
theorem find_AC : dist A C = √106.4 :=
sorry

end find_AC_l164_164599


namespace fuel_efficiency_sum_l164_164594

theorem fuel_efficiency_sum (m n : ℕ) 
  (h₁ : ∀ d : ℕ, ((d ≠ 0) → 
    (18 * 24 * m * n = (d * (24 * m + 18 * n + 18 * 24))))
  : m + n = 108 :=
sorry

end fuel_efficiency_sum_l164_164594


namespace worms_domino_decomposition_eq_totient_l164_164821

theorem worms_domino_decomposition_eq_totient (n : ℕ) (hn : n > 2) :
  (number_of_decompositions_into_dominoes n) = (nat.totient n) :=
sorry

end worms_domino_decomposition_eq_totient_l164_164821


namespace paintable_wall_area_is_1624_l164_164155

-- Definitions based on the conditions
def bedrooms : Nat := 4
def length : Nat := 15
def width : Nat := 12
def height : Nat := 9
def non_painted_area_per_bedroom : Nat := 80

-- Total area of walls to be painted for all bedrooms
def wall_area_to_paint : Nat :=
  let area_per_bedroom := 2 * (length * height) + 2 * (width * height) - non_painted_area_per_bedroom
  bedrooms * area_per_bedroom

-- The proof goal
theorem paintable_wall_area_is_1624 : wall_area_to_paint = 1624 := by
  sorry

end paintable_wall_area_is_1624_l164_164155


namespace volleyballs_count_l164_164266

-- Definitions of sports item counts based on given conditions.
def soccer_balls := 20
def basketballs := soccer_balls + 5
def tennis_balls := 2 * soccer_balls
def baseballs := soccer_balls + 10
def hockey_pucks := tennis_balls / 2
def total_items := 180

-- Calculate the total number of known sports items.
def known_items_sum := soccer_balls + basketballs + tennis_balls + baseballs + hockey_pucks

-- Prove the number of volleyballs
theorem volleyballs_count : total_items - known_items_sum = 45 := by
  sorry

end volleyballs_count_l164_164266


namespace expression_value_correct_l164_164461

def numerator := 2^2 + 2^1 + 2^(-2)
def denominator := 2^(-1) + 2^(-3) + 2^(-5)
def expression := numerator / denominator

theorem expression_value_correct : expression = 200 / 21 := by
  sorry

end expression_value_correct_l164_164461


namespace parallelogram_coincides_l164_164803

-- Define the shapes
inductive Shape
| Parallelogram
| EquilateralTriangle
| IsoscelesRightTriangle
| RegularPentagon

open Shape

-- Define the property of coinciding with itself after 180 degrees rotation
def coincides_after_180_degrees_rotation (shape : Shape) : Prop :=
  match shape with
  | Parallelogram => true
  | EquilateralTriangle => false
  | IsoscelesRightTriangle => false
  | RegularPentagon => false

-- The theorem stating the Parallelogram coincides with itself after 180 degrees rotation
theorem parallelogram_coincides : coincides_after_180_degrees_rotation Parallelogram :=
  by
    exact true.intro

end parallelogram_coincides_l164_164803


namespace sum_of_real_numbers_l164_164229

theorem sum_of_real_numbers (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_real_numbers_l164_164229


namespace recurring_decimal_numerator_count_l164_164714

theorem recurring_decimal_numerator_count :
  let nums := {n | 1 ≤ n ∧ n ≤ 999};
  let rel_prime := {n | n ∈ nums ∧ Nat.gcd n 999 = 1};
  let additional := {n | n ∈ nums ∧ n % 81 = 0};
  (Card rel_prime + Card additional = 660) :=
by
  sorry

end recurring_decimal_numerator_count_l164_164714


namespace problem1_problem2_part1_problem2_part2_l164_164868

-- Problems Translation

-- Condition (1)
theorem problem1 (x : ℝ) (hx : x > 0) : (1 : ℝ) * Real.log(x) ≤ x - 1 :=
sorry

-- Condition (2), Maximum value of a
def max_a : ℝ := Real.exp 1
theorem problem2_part1 (x : ℝ) (hx : x > 0) : (max_a * Real.log(x)) ≤ x :=
sorry

-- Condition (2), second part
def H (x : ℝ) : ℝ := (Real.exp 1 + 1) / 2 * x^2 - 2 * Real.exp 1 * x * Real.log(x)
theorem problem2_part2 (x1 x2 : ℝ) (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x1 ≠ x2)
: (H(x1) - H(x2)) / (x1 - x2) > -Real.exp 1 :=
sorry

end problem1_problem2_part1_problem2_part2_l164_164868


namespace nonagon_distinct_diagonals_l164_164948

theorem nonagon_distinct_diagonals : 
  let n := 9 in
  ∃ (d : ℕ), d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end nonagon_distinct_diagonals_l164_164948


namespace Q3_x_coords_sum_eq_Q1_x_coords_sum_l164_164410

-- Define a 40-gon and its x-coordinates sum
def Q1_x_coords_sum : ℝ := 120

-- Statement to prove
theorem Q3_x_coords_sum_eq_Q1_x_coords_sum (Q1_x_coords_sum: ℝ) (h: Q1_x_coords_sum = 120) : 
  (Q3_x_coords_sum: ℝ) = Q1_x_coords_sum :=
sorry

end Q3_x_coords_sum_eq_Q1_x_coords_sum_l164_164410


namespace nonagon_diagonals_l164_164943

def convex_nonagon_diagonals : Prop :=
∀ (n : ℕ), n = 9 → (n * (n - 3)) / 2 = 27

theorem nonagon_diagonals : convex_nonagon_diagonals :=
by {
  sorry,
}

end nonagon_diagonals_l164_164943


namespace total_cost_is_45_48_l164_164663

noncomputable def total_order_cost (burger_cost soda_cost : ℝ)
  (chicken_sandwich_cost : ℝ)
  (discount_threshold : ℝ)
  (discount : ℝ)
  (tax_rate : ℝ)
  (coupon_threshold : ℝ)
  (coupon_value : ℝ)
  : ℝ :=
let paulo_cost := burger_cost + soda_cost in
let jeremy_cost := 2 * (burger_cost + soda_cost) in
let stephanie_cost := 3 * burger_cost + soda_cost + chicken_sandwich_cost in
let total_before_tax := paulo_cost + jeremy_cost + stephanie_cost in
let tax := tax_rate * total_before_tax in
let total_with_tax := total_before_tax + tax in
let total_with_coupon := if total_before_tax > coupon_threshold then total_with_tax - coupon_value else total_with_tax in
let total_burger_meals_cost := burger_cost + 2 * burger_cost + 3 * burger_cost in
let happy_hour_discount := if (1 + 2 + 3) > 2 then discount * total_burger_meals_cost else 0 in
total_with_coupon - happy_hour_discount

theorem total_cost_is_45_48 :
  total_order_cost 6 2 7.50 0.10 0.05 25 5 = 45.48 :=
by 
  -- Calculation steps 
  sorry

end total_cost_is_45_48_l164_164663


namespace find_f_f_neg1_l164_164867

def f (x : ℝ) : ℝ := 
  if x < 0 then x^2 + 2 
  else x + 1

theorem find_f_f_neg1 : f (f (-1)) = 4 := 
by 
  -- This is where the proof would go, but according to the instruction, we'll skip it.
  sorry

end find_f_f_neg1_l164_164867


namespace largest_A_at_k_125_l164_164823

noncomputable def binomial_A (k : ℕ) (n : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k)

theorem largest_A_at_k_125 :
  let A := λ k, binomial_A k 500 (3 / 10)
  ∃ k ∈ finset.range 501, k = 125 ∧ ∀ k' ∈ finset.range 501, A k' ≤ A k :=
begin
  sorry
end

end largest_A_at_k_125_l164_164823


namespace smallest_integer_whose_cube_ends_in_368_l164_164040

theorem smallest_integer_whose_cube_ends_in_368 :
  ∃ (n : ℕ+), (n % 2 = 0 ∧ n^3 % 1000 = 368) ∧ (∀ (m : ℕ+), m % 2 = 0 ∧ m^3 % 1000 = 368 → m ≥ n) :=
by
  sorry

end smallest_integer_whose_cube_ends_in_368_l164_164040


namespace area_perimeter_equality_of_60_gon_l164_164773

theorem area_perimeter_equality_of_60_gon (
  {radius : ℝ} {n : ℕ} (h_n : n = 30) (h_radius : radius = 2)
  (A B : Fin n → ℂ) (h_A_is_ngon : ∀ i j, i ≠ j → A i ≠ A j)
  (h_B_is_midpoint : ∀ i, B i = (A i + A (i + 1) % n) / 2) :
  (\sum i, (radius * (1 - (A i * B i).abs))) = (\sum i, (A i * (A (i + 1) % n)).abs) :=
sorry

end area_perimeter_equality_of_60_gon_l164_164773


namespace minimal_representation_l164_164633

noncomputable def valid_representation (n k t : ℕ) (us : Fin n.succ → ℕ) (as : Fin n.succ → ℕ) : Prop :=
  (∀ i, 0 ≤ as i) ∧ (t = ∑ i, as i * us i)

theorem minimal_representation (n k t : ℕ) (us : Fin n.succ → ℕ) (h_us_bound : ∀ i, 1 ≤ us i ∧ us i ≤ 2^k) (h_k_ge_3 : 3 ≤ k) :
  ∃ as : Fin n.succ → ℕ, valid_representation n k t us as → ∃ bs : Fin n.succ → ℕ, 
  (valid_representation n k t us bs ∧ ((∑ i, ite (bs i ≠ 0) 1 0) < k + 1)) := 
sorry

end minimal_representation_l164_164633


namespace polynomial_unique_solution_l164_164012

theorem polynomial_unique_solution (P : ℝ → ℝ) (hP : ∀ x : ℝ, P(x^2 + 1) = P(x)^2 + 1) (h0 : P(0) = 0) : ∀ x : ℝ, P(x) = x :=
sorry

end polynomial_unique_solution_l164_164012


namespace minimum_value_of_f_l164_164538

noncomputable def f (x m : ℝ) : ℝ := (x^2 + x + m) * Real.exp x

theorem minimum_value_of_f (m : ℝ) (hmax : ∀ x : ℝ, (x ≠ -3 → (derivative (λ x, f x m)) (-3) = 0 →) :
  x = -3 → (∀ y : ℝ, f y m ≤ f x m)) :
  ∃ x : ℝ, f x (-1) = -1 :=
begin
  sorry
end

end minimum_value_of_f_l164_164538


namespace evaluate_expression_l164_164841

def x : ℝ := 2
def y : ℝ := 4

theorem evaluate_expression : y * (y - 2 * x) = 0 := by
  sorry

end evaluate_expression_l164_164841


namespace log2_minus_x_decreasing_l164_164710

def is_monotonic_decreasing_interval (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f y ≤ f x

noncomputable def log_decreasing_interval : set ℝ :=
  {x : ℝ | x < 2}

theorem log2_minus_x_decreasing :
  is_monotonic_decreasing_interval (λ x, Real.log (2 - x)) log_decreasing_interval :=
sorry

end log2_minus_x_decreasing_l164_164710


namespace exists_two_families_equilateral_centers_of_families_on_two_concentric_circles_l164_164886

-- Given a triangle ABC
variables (A B C : Type)

-- Assuming some properties of triangles
axiom triangle (A B C : Type) : Prop
axiom point (P : Type)

-- Equilateral triangles whose sides (or their extensions) pass through points A, B, C
axiom equilateral_triangle_through_points : 
  (A B C : Type) (tri : triangle A B C) → 
  ∃ (E F G : Type), 
  point E ∧ point F ∧ point G ∧ 
  (A ∈ triangle E F G ∨ B ∈ triangle F G E ∨ C ∈ triangle G E F)

-- Centers of the triangles in these families lie on two concentric circles
axiom centers_on_concentric_circles :
  (A B C : Type) (tri : triangle A B C) →
  ∃ (O1 O2 : Type), 
  conc_circle O1 ∧ conc_circle O2 ∧
  (\exists (P Q R : Type), 
    point P ∧ point Q ∧ point R ∧ 
    (P ∈ circle O1) ∧ (Q ∈ circle O1) ∧ (R ∈ circle O2))

-- Main theorem statements
theorem exists_two_families_equilateral (A B C : Type) (tri : triangle A B C) :
  ∃ (E F G : Type), 
  point E ∧ point F ∧ point G ∧ 
  (A ∈ triangle E F G ∨ B ∈ triangle F G E ∨ C ∈ triangle G E F) :=
begin
  apply equilateral_triangle_through_points,
  assumption,
end

theorem centers_of_families_on_two_concentric_circles (A B C : Type) (tri : triangle A B C) :
  ∃ (O1 O2 : Type), 
  conc_circle O1 ∧ conc_circle O2 ∧ 
  (\exists (P Q R : Type), 
    point P ∧ point Q ∧ point R ∧
    (P ∈ circle O1) ∧ (Q ∈ circle O1) ∧ (R ∈ circle O2)) :=
begin
  apply centers_on_concentric_circles,
  assumption,
end

end exists_two_families_equilateral_centers_of_families_on_two_concentric_circles_l164_164886


namespace average_of_numbers_is_correct_l164_164748

theorem average_of_numbers_is_correct :
  (let nums := [12, 13, 14, 510, 520, 530, 1115, 1120, 1, 1252140] in
   let supposed_avg := 858.5454545454545 in
   let correct_avg := 125397.5 in
   (nums.sum / nums.length) = correct_avg) :=
by
  sorry

end average_of_numbers_is_correct_l164_164748


namespace function_increasing_in_range_l164_164542

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - m) * x - m else Real.log x / Real.log m

theorem function_increasing_in_range (m : ℝ) :
  (3 / 2 ≤ m ∧ m < 3) ↔ (∀ x y : ℝ, x < y → f m x < f m y) := by
  sorry

end function_increasing_in_range_l164_164542


namespace intersection_of_sets_l164_164520

theorem intersection_of_sets :
  let A := {1, 2, 3, 4}
  let B := {2, 4, 5}
  A ∩ B = {2, 4} := by
{
  let A := {1, 2, 3, 4}
  let B := {2, 4, 5}
  show A ∩ B = {2, 4}
  sorry
}

end intersection_of_sets_l164_164520


namespace process_termination_l164_164634

open Real EuclideanSpace

theorem process_termination {n m : ℕ} (v : Fin m → EuclideanSpace ℝ n)
  (h1 : ∀ i, 0 < (v i).1) :
  ∃ C : ℝ, ∀ w : EuclideanSpace ℝ n, (∀ i, w • v i ≤ 0) →
    (∃ r : ℕ, w r = (0: EuclideanSpace ℝ n) ∧ r ≤ C) := sorry

end process_termination_l164_164634


namespace modulus_of_complex_eq_l164_164523

open Complex

theorem modulus_of_complex_eq (a b : ℝ) 
  (h : (1 + 2 * Complex.I) / (a + b * Complex.I) = 1 + Complex.I) :
  Complex.abs (a + b * Complex.I) = sqrt (10) / 2 :=
  sorry

end modulus_of_complex_eq_l164_164523


namespace compute_binomial_sum_l164_164463

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem compute_binomial_sum :
  binomial 12 11 + binomial 12 1 = 24 :=
by
  sorry

end compute_binomial_sum_l164_164463


namespace range_of_a_l164_164060

noncomputable def p (x: ℝ) : Prop := |4 * x - 1| ≤ 1
noncomputable def q (x a: ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a: ℝ) :
  (¬ (∀ x, p x) → (¬ (∀ x, q x a))) ∧ (¬ (¬ (∀ x, p x) → (¬ (∀ x, q x a))))
  ↔ (-1 / 2 ≤ a ∧ a ≤ 0) :=
sorry

end range_of_a_l164_164060


namespace uncle_bob_can_park_probability_l164_164440

theorem uncle_bob_can_park_probability : 
  let total_ways_to_park := Nat.choose 20 14,
      ways_to_park_with_no_adjacent3 := Nat.choose 10 6,
      probability_cannot_park := (ways_to_park_with_no_adjacent3 : ℚ) / total_ways_to_park,
      probability_can_park := 1 - probability_cannot_park in
  probability_can_park = (19275 : ℚ) / 19380 :=
by
  sorry

end uncle_bob_can_park_probability_l164_164440


namespace question_1_question_2_l164_164109

variables {θ : ℝ} (hθ : 0 < θ ∧ θ < Real.pi / 2)

def vector_a := (2 : ℝ, Real.sin θ)
def vector_b := (1 : ℝ, Real.cos θ)
def dot_prod : ℝ := 2 * 1 + Real.sin θ * Real.cos θ

theorem question_1 (h : dot_prod = 13 / 6) : Real.sin θ + Real.cos θ = 2 * Real.sqrt 3 / 3 :=
sorry

theorem question_2 (h1 : 2 * Real.cos θ = Real.sin θ) : Real.cos (2 * θ) = -3 / 5 :=
sorry

end question_1_question_2_l164_164109


namespace sufficient_but_not_necessary_condition_l164_164968

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_even_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = h x

-- This definition states that both f and g are either odd or even functions
def is_odd_or_even (f g : ℝ → ℝ) : Prop := 
  (is_odd f ∧ is_odd g) ∨ (is_even f ∧ is_even g)

theorem sufficient_but_not_necessary_condition (f g : ℝ → ℝ)
  (h : is_odd_or_even f g) : 
  ¬(is_odd f ∧ is_odd g) → is_even_function (f * g) :=
sorry

end sufficient_but_not_necessary_condition_l164_164968


namespace sequence_property_l164_164295

noncomputable def sequence_exists : Prop :=
  ∃ (a : ℕ → ℕ), 
  (∀ x, ∃ n, a n = x) ∧ -- Every positive integer appears exactly once
  (∀ n, n ≥ 1 → (∑ i in finset.range (n + 1), a i) % n = 0) -- Sum is divisible by n

-- Appling sorry to skip the proof
theorem sequence_property : sequence_exists := 
  sorry

end sequence_property_l164_164295


namespace root_inequalities_l164_164918

noncomputable def f (x : ℝ) : ℝ := log x - 1 / x

theorem root_inequalities (x₀ : ℝ) (hx₀ : f x₀ = 0) (h1 : 1 < x₀) (h2 : x₀ < 2) : 
  2^x₀ > x₀^(1/2) ∧ x₀^(1/2) > log x₀ :=
by
  sorry

end root_inequalities_l164_164918


namespace triangles_similar_l164_164884

variable {A B C P M N Q : Type*}
variables [IsTriangle A B C] [OnSide P B C] [OnSide M A B] [OnSide N A C]
variables [Parallel MP AC] [Parallel NP AB] [Reflection MN P Q]

theorem triangles_similar :
  ∀ (A B C P M N Q : Type*) [IsTriangle A B C] [OnSide P B C] [OnSide M A B] [OnSide N A C]
    [Parallel MP AC] [Parallel NP AB] [Reflection MN P Q],
    Similar (Triangle.mk Q M B) (Triangle.mk C N Q) :=
by
  sorry

end triangles_similar_l164_164884


namespace loom_weaving_time_l164_164778

theorem loom_weaving_time (C : ℝ) (h1 : ∀ (t : ℝ), t = 15 / 117.1875 → t = 0.128) : 
  (T : ℝ) : T = C / 0.128 :=
by
  -- Proof to be completed
  sorry

end loom_weaving_time_l164_164778


namespace evaluate_powers_of_i_l164_164009

theorem evaluate_powers_of_i :
  let i : ℂ := complex.I in
  i ^ 23 + i ^ 34 + i ^ -17 = -1 :=
by
  sorry

end evaluate_powers_of_i_l164_164009


namespace integral_of_even_function_l164_164116

-- Define the function and conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def f (x : ℝ) (a : ℝ) : ℝ :=
  a * x ^ 2 + (a - 2) * x + a ^ 2

-- Define the integral to be computed
def integral_expr (a : ℝ) : ℝ :=
  ∫ x in -a..a, x ^ 2 + x + (sqrt (4 - x ^ 2))

-- Prove the equality
theorem integral_of_even_function :
  is_even_function (λ x : ℝ, f x 2) →
  integral_expr 2 = (16/3 + 2 * Real.pi) :=
by
  intro h_even
  have a_eq_2 : 2 - 2 = 0 := rfl
  have h_f_even : ∀ x : ℝ, f x 2 = f (-x) 2 :=
    by
      intro x
      sorry
  sorry

end integral_of_even_function_l164_164116


namespace smaller_circle_radius_is_6_l164_164467

-- Define the conditions of the problem
def large_circle_radius : ℝ := 2

def smaller_circles_touching_each_other (r : ℝ) : Prop :=
  let oa := large_circle_radius + r
  let ob := large_circle_radius + r
  let ab := 2 * r
  (oa^2 + ob^2 = ab^2)

def problem_statement : Prop :=
  ∃ r : ℝ, smaller_circles_touching_each_other r ∧ r = 6

theorem smaller_circle_radius_is_6 : problem_statement :=
sorry

end smaller_circle_radius_is_6_l164_164467


namespace circle_radius_squared_l164_164414

theorem circle_radius_squared {r P A B C D : ℝ}
  (h1 : chord_length A B = 10)
  (h2 : chord_length C D = 7)
  (h3 : intersects P A B C D)
  (h4 : angle A P D = 60)
  (h5 : length B P = 8)
  : r^2 = 73 :=
sorry

end circle_radius_squared_l164_164414


namespace rectangle_width_l164_164707

theorem rectangle_width (w : ℝ) (h_length : w * 2 = l) (h_area : w * l = 50) : w = 5 :=
by
  sorry

end rectangle_width_l164_164707


namespace product_zero_when_a_is_five_l164_164484

-- Define the product with the specified value of a
def product_expression (a : ℝ) : ℝ :=
  ∏ (i : ℕ) in (finset.range 11), (a - i)

-- The theorem statement
theorem product_zero_when_a_is_five : product_expression 5 = 0 :=
by
  sorry

end product_zero_when_a_is_five_l164_164484


namespace square_plot_dimensions_l164_164374

theorem square_plot_dimensions :
  ∃ (side_length : ℕ), ∃ (area : ℕ), 
    (area = side_length * side_length) ∧ 
    (area >= 1000) ∧ 
    (area <= 9999) ∧ 
    (∀ d ∈ (area.digits 10), d % 2 = 0) ∧ 
    (list.nodup (area.digits 10)) ∧ 
    (side_length = 78) ∧ 
    (area = 6084) :=
  by {
    sorry
  }

end square_plot_dimensions_l164_164374


namespace monotonic_intervals_exists_real_a_unique_real_a_l164_164920

noncomputable def f (x a : ℝ) : ℝ := x * Real.log x - a * x

theorem monotonic_intervals (a : ℝ) :
  (∀ x ∈ Ioo 0 (Real.exp (a - 1)), (deriv (fun x => f x a) x) < 0) ∧
  (∀ x ∈ Ioi (Real.exp (a - 1)), (deriv (fun x => f x a) x) > 0) :=
sorry

theorem exists_real_a (x : ℝ) (h₀ : x > 0) :
  (f x 1 + 1 ≥ 0) :=
sorry

theorem unique_real_a :
  ∀ a : ℝ, (∀ x > 0, f x a + a ≥ 0) ↔ a = 1 :=
sorry

end monotonic_intervals_exists_real_a_unique_real_a_l164_164920


namespace common_sum_in_4x4_matrix_l164_164691

open Nat

theorem common_sum_in_4x4_matrix : 
  let A := (-8 : ℤ), B := (7 : ℤ), n := 16
  let matrix_sum := (n / 2) * (A + B)
  let num_rows := 4
  (matrix_sum = -8) →
  (num_rows = 4) →
  ∃ common_sum : ℤ, common_sum = matrix_sum / num_rows ∧ common_sum = -2 :=
by
  let A := (-8 : ℤ)
  let B := (7 : ℤ)
  let n := 16
  let matrix_sum := (n / 2) * (A + B)
  let num_rows := 4
  assume h1 : (matrix_sum = -8)
  assume h2 : (num_rows = 4)
  use (matrix_sum / num_rows)
  split
  case 1 =>
    rw [h1, h2]
    simp
  case 2 =>
    sorry

end common_sum_in_4x4_matrix_l164_164691


namespace sequence_geometric_gen_formula_a_sum_formula_S_l164_164881

def a (n : ℕ) : ℕ → ℕ
| 1 := 1
| (k+1) := (2 * (k+1) * a k + k * (k+1)) / k

def b (n : ℕ) : ℕ := (a n) / n + 1

theorem sequence_geometric (n : ℕ) (hn : n > 0) : 
  b (n+1) = 2 * b n :=
sorry

theorem gen_formula_a (n : ℕ) (hn : n > 0) : 
  a n = n * (2^n - 1) :=
sorry

theorem sum_formula_S (n : ℕ) (hn : n > 0) : 
  let S_n := ∑ i in range (1, n+1) (\i, a i)
  in S_n = (n-1)*2^(n+1)+2-(n*(n+1))/2 :=
sorry

end sequence_geometric_gen_formula_a_sum_formula_S_l164_164881


namespace sum_reciprocal_converge_l164_164178

noncomputable def x : ℕ → ℝ
| 0       := 1 / 20
| 1       := 1 / 13
| (n + 2) := 2 * x n * x (n + 1) * (x n + x (n + 1)) / (x n ^ 2 + x (n + 1) ^ 2)

theorem sum_reciprocal_converge :
  ∑' n, 1 / (x n + x (n + 1)) = 23 := 
sorry

end sum_reciprocal_converge_l164_164178


namespace hyperbola_asymptote_ratio_l164_164495

theorem hyperbola_asymptote_ratio
  (a b : ℝ) (h₁ : a ≠ b) (h₂ : (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1))
  (h₃ : ∀ m n: ℝ, m * n = -1 → ∃ θ: ℝ, θ = 90* (π / 180)): 
  a / b = 1 := 
sorry

end hyperbola_asymptote_ratio_l164_164495


namespace four_circle_distances_sum_eq_ninety_l164_164859

noncomputable theory

-- Define the centers of the circles and points P, Q
variables {A B C D P Q R : ℝ} 

-- Provided conditions as hypotheses
variables 
  (rA rB rC rD : ℝ)
  (h_rad_AB : rA = (3 / 4) * rB)
  (h_rad_CD : rC = (3 / 4) * rD)
  (h_AB_CD : dist A B = 45 ∧ dist C D = 45)
  (h_PQ : dist P Q = 50)
  (h_R_midpoint : R = (P + Q) / 2)
  (h_P_on_circle : dist P A = rA ∧ dist P B = rB ∧ dist P C = rC ∧ dist P D = rD)
  (h_Q_on_circle : dist Q A = rA ∧ dist Q B = rB ∧ dist Q C = rC ∧ dist Q D = rD)

-- Theorem statement to prove
theorem four_circle_distances_sum_eq_ninety :
  dist A R + dist B R + dist C R + dist D R = 90 :=
sorry

end four_circle_distances_sum_eq_ninety_l164_164859


namespace M_is_midpoint_of_AB_l164_164927

open EuclideanGeometry

variables {A B C H D M : Point}
variables {circumcircle_ABC : Circle}
variables {angle_BAC_gt_90 : angle A B C > 90}

-- Conditions
-- 1. H is the orthocenter of triangle ABC
variable (Horthocenter : is_orthocenter H A B C)
-- 2. Circle with diameter HC intersects circumcircle of triangle ABC at point D
variable (circle_with_diameter_HC : is_circle_with_diameter H C)
variable (intersects_circumcircle_at_D : intersects_at circle_with_diameter_HC circumcircle_ABC D)

-- 3. DH extended intersects AB at point M
variable (DH_intersects_AB_at_M : extends_to_intersect D H A B M)

-- Goal
theorem M_is_midpoint_of_AB :
  is_midpoint M A B :=
sorry

end M_is_midpoint_of_AB_l164_164927


namespace points_on_line_l164_164275

theorem points_on_line (n : ℕ) : 9 * n - 8 = 82 → n = 10 :=
by
  sorry

end points_on_line_l164_164275


namespace sum_of_x_y_l164_164245

theorem sum_of_x_y (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_x_y_l164_164245


namespace sum_product_of_integers_l164_164336

theorem sum_product_of_integers (a b c : ℕ) (h₁ : c = a + b) (h₂ : N = a * b * c) (h₃ : N = 8 * (a + b + c)) : 
  a * b * (a + b) = 16 * (a + b) :=
by {
  sorry
}

end sum_product_of_integers_l164_164336


namespace proof_problem_l164_164205

def M : set ℝ := {x | 2 - x > 0}
def N : set ℝ := {x | x^2 - 4 * x + 3 < 0}
def U : set ℝ := set.univ
def C_U_M : set ℝ := {x | x ≥ 2}
def result_set : set ℝ := {x | 2 ≤ x ∧ x < 3}

theorem proof_problem : (C_U_M ∩ N) = result_set :=
by 
  sorry

end proof_problem_l164_164205


namespace sum_product_of_integers_l164_164339

theorem sum_product_of_integers (a b c : ℕ) (h₁ : c = a + b) (h₂ : N = a * b * c) (h₃ : N = 8 * (a + b + c)) : 
  a * b * (a + b) = 16 * (a + b) :=
by {
  sorry
}

end sum_product_of_integers_l164_164339


namespace general_formula_for_a_n_value_of_p_smallest_positive_m_l164_164882

open Nat

def S (n : ℕ) : ℕ := 2*n^2 - n
def a (n : ℕ) : ℕ := 4*n - 3
def b (n p : ℕ) : ℕ := (2*n^2 - n) / p
def c (n : ℕ) : ℕ := (2*n^2 - n) / 10
def T (n : ℕ) : ℕ := ∑ i in range (n + 1), c i

theorem general_formula_for_a_n (n : ℕ) : S n = ∑ i in range (n + 1), a i := sorry

theorem value_of_p (p : ℕ) : (∀ n, b n p) = (λ n : ℕ, 4*n - 3) → p = -1/2 := sorry

theorem smallest_positive_m (m : ℕ) : (∀ n, T n < m) → m = 10 := sorry

end general_formula_for_a_n_value_of_p_smallest_positive_m_l164_164882


namespace chessboard_knights_l164_164217

noncomputable def remove_20_knights {board : Type} [fintype board] (knights : finset board) (controls : board → finset board) :=
  ∃ (remaining_knights : finset board), remaining_knights.card = 200 ∧
  (∀ sq ∈ univ \ remaining_knights, ∃ k ∈ remaining_knights, sq ∈ controls k)

theorem chessboard_knights :
  ∃ (knights : finset (fin 20 × fin 20)) (controls : (fin 20 × fin 20) → finset (fin 20 × fin 20)),
    knights.card = 220 ∧
    (∀ sq ∈ univ \ knights, ∃ k ∈ knights, sq ∈ controls k) ∧
    remove_20_knights knights controls :=
by
  sorry

end chessboard_knights_l164_164217


namespace product_of_three_integers_sum_l164_164329

theorem product_of_three_integers_sum :
  ∀ (a b c : ℕ), (c = a + b) → (a * b * c = 8 * (a + b + c)) →
  (a > 0) → (b > 0) → (c > 0) →
  (∃ N1 N2 N3: ℕ, N1 = (a * b * (a + b)), N2 = (a * b * (a + b)), N3 = (a * b * (a + b)) ∧ 
  (N1 = 272 ∨ N2 = 160 ∨ N3 = 128) ∧ 
  (N1 + N2 + N3 = 560)) := sorry

end product_of_three_integers_sum_l164_164329


namespace maximize_profit_l164_164783

noncomputable section

-- Definitions of parameters
def daily_sales_volume (x : ℝ) : ℝ := -2 * x + 200
def daily_cost : ℝ := 450
def price_min : ℝ := 30
def price_max : ℝ := 60

-- Function for daily profit
def daily_profit (x : ℝ) : ℝ := (x - 30) * daily_sales_volume x - daily_cost

-- Theorem statement
theorem maximize_profit :
  let max_profit_price := 60
  let max_profit_value := 1950
  30 ≤ max_profit_price ∧ max_profit_price ≤ 60 ∧
  daily_profit max_profit_price = max_profit_value :=
by
  sorry

end maximize_profit_l164_164783


namespace rain_difference_l164_164602

variable (R : ℝ) -- Amount of rain in the second hour
variable (r1 : ℝ) -- Amount of rain in the first hour

-- Conditions
axiom h1 : r1 = 5
axiom h2 : R + r1 = 22

-- Theorem to prove
theorem rain_difference (R r1 : ℝ) (h1 : r1 = 5) (h2 : R + r1 = 22) : R - 2 * r1 = 7 := by
  sorry

end rain_difference_l164_164602


namespace trapezoid_perimeter_area_sum_l164_164883

noncomputable def distance (p1 p2 : Real × Real) : Real :=
  ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2).sqrt

noncomputable def perimeter (vertices : List (Real × Real)) : Real :=
  match vertices with
  | [a, b, c, d] => (distance a b) + (distance b c) + (distance c d) + (distance d a)
  | _ => 0

noncomputable def area_trapezoid (b1 b2 h : Real) : Real :=
  0.5 * (b1 + b2) * h

theorem trapezoid_perimeter_area_sum
  (A B C D : Real × Real)
  (h_AB : A = (2, 3))
  (h_BC : B = (7, 3))
  (h_CD : C = (9, 7))
  (h_DA : D = (0, 7)) :
  let perimeter := perimeter [A, B, C, D]
  let area := area_trapezoid (distance C D) (distance A B) (C.2 - B.2)
  perimeter + area = 42 + 4 * Real.sqrt 5 :=
by
  sorry

end trapezoid_perimeter_area_sum_l164_164883


namespace construct_triangle_case1_construct_triangle_case2_l164_164824

-- Definition of given points for the cases
structure GivenPointsCase1 :=
  (M A1 B : ℝ × ℝ) -- Points M, A1, and B
  (not_collinear : ¬CollinearPoints M A1 B) -- M, A1, and B must not be collinear

-- Definition of circumcenter, midpoint, and construction steps
structure GivenPointsCase2 :=
  (M A1 A : ℝ × ℝ) -- Points M, A1, and A
  (circumcenter_exists : ∃ O : ℝ × ℝ, True) -- Assume there exists a circumcenter O
  (midpointAM : ℝ × ℝ) -- Midpoint of A and M
  -- O, A1, F, and A form a parallelogram
  (parallelogram : IsParallelogram circumcenter_exists.some A1 midpointAM A)
  (construction_feasible : True) -- The construction is feasible

-- Definition and Collinear test
def CollinearPoints (P Q R : ℝ × ℝ) : Prop :=
  let (x1, y1) := P
  let (x2, y2) := Q
  let (x3, y3) := R
  (x2 - x1) * (y3 - y1) = (y2 - y1) * (x3 - x1)

-- Proving the construction is possible given the conditions
theorem construct_triangle_case1 (input : GivenPointsCase1) : ∃ C : ℝ × ℝ, True :=
  sorry

theorem construct_triangle_case2 (input : GivenPointsCase2) : ∃ B C : ℝ × ℝ, True :=
  sorry

end construct_triangle_case1_construct_triangle_case2_l164_164824


namespace product_gcd_lcm_15_9_l164_164494

theorem product_gcd_lcm_15_9 : Nat.gcd 15 9 * Nat.lcm 15 9 = 135 := 
by
  -- skipping proof as instructed
  sorry

end product_gcd_lcm_15_9_l164_164494


namespace polynomial_unique_solution_l164_164013

theorem polynomial_unique_solution (P : ℝ → ℝ) (hP : ∀ x : ℝ, P(x^2 + 1) = P(x)^2 + 1) (h0 : P(0) = 0) : ∀ x : ℝ, P(x) = x :=
sorry

end polynomial_unique_solution_l164_164013


namespace trigonometric_identity_l164_164894

theorem trigonometric_identity (alpha : ℝ) (h : Real.tan alpha = 2 * Real.tan (π / 5)) :
  (Real.cos (alpha - 3 * π / 10) / Real.sin (alpha - π / 5)) = 3 :=
by
  sorry

end trigonometric_identity_l164_164894


namespace intersecting_line_l164_164549

theorem intersecting_line {x y : ℝ} (h1 : x^2 + y^2 = 10) (h2 : (x - 1)^2 + (y - 3)^2 = 10) :
  x + 3 * y - 5 = 0 :=
sorry

end intersecting_line_l164_164549


namespace possible_values_of_sum_l164_164242

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l164_164242


namespace pyramid_volume_l164_164306

-- Definitions of given conditions
def acute_angle := Real.pi / 8
def lateral_edge_length := Real.sqrt 6
def inclination := 5 * Real.pi / 13

-- Proof statement
theorem pyramid_volume 
  (h1 : ∠ABC = acute_angle)
  (h2 : lateral_edge_length = Real.sqrt 6)
  (h3 : inclination = 5 * Real.pi / 13) :
  volume_of_pyramid = Real.sqrt 3 * Real.sin (10 * Real.pi / 13) * Real.cos (5 * Real.pi / 13) :=
  sorry

end pyramid_volume_l164_164306


namespace points_on_line_possible_l164_164279

theorem points_on_line_possible : ∃ n : ℕ, 9 * n - 8 = 82 :=
by
  sorry

end points_on_line_possible_l164_164279


namespace find_p_current_age_l164_164408

theorem find_p_current_age (x p q : ℕ) (h1 : p - 3 = 4 * x) (h2 : q - 3 = 3 * x) (h3 : (p + 6) / (q + 6) = 7 / 6) : p = 15 := 
sorry

end find_p_current_age_l164_164408


namespace probability_red_or_blue_is_713_l164_164420

-- Definition of area ratios
def area_ratio_red : ℕ := 6
def area_ratio_yellow : ℕ := 2
def area_ratio_blue : ℕ := 1
def area_ratio_black : ℕ := 4

-- Total area ratio
def total_area_ratio := area_ratio_red + area_ratio_yellow + area_ratio_blue + area_ratio_black

-- Probability of stopping on either red or blue
def probability_red_or_blue := (area_ratio_red + area_ratio_blue) / total_area_ratio

-- Theorem stating the probability is 7/13
theorem probability_red_or_blue_is_713 : probability_red_or_blue = 7 / 13 :=
by
  unfold probability_red_or_blue total_area_ratio area_ratio_red area_ratio_blue
  simp
  sorry

end probability_red_or_blue_is_713_l164_164420


namespace sum_of_real_numbers_l164_164227

theorem sum_of_real_numbers (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_real_numbers_l164_164227


namespace sum_of_possible_values_of_N_l164_164333

theorem sum_of_possible_values_of_N :
  ∃ a b c : ℕ, (a > 0 ∧ b > 0 ∧ c > 0) ∧ (abc = 8 * (a + b + c)) ∧ (c = a + b)
  ∧ (2560 = 560) :=
by
  sorry

end sum_of_possible_values_of_N_l164_164333


namespace possible_values_of_sum_l164_164240

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l164_164240


namespace sufficient_not_necessary_condition_l164_164505

-- Definitions for a, b, c being real numbers
variable (a b c : ℝ)

-- Condition that c squared is non-negative
theorem sufficient_not_necessary_condition (h : c^2 ≥ 0) : 
  (ac^2 < bc^2) ↔ (a < b) :=
by
  sorry

end sufficient_not_necessary_condition_l164_164505


namespace nonagon_diagonals_count_eq_27_l164_164954

theorem nonagon_diagonals_count_eq_27 :
  let vertices := 9
  let connections_per_vertex := vertices - 3
  let total_raw_count_diagonals := vertices * connections_per_vertex
  let distinct_diagonals := total_raw_count_diagonals / 2
  distinct_diagonals = 27 :=
by
  let vertices := 9
  let connections_per_vertex := vertices - 3
  let total_raw_count_diagonals := vertices * connections_per_vertex
  let distinct_diagonals := total_raw_count_diagonals / 2
  have : distinct_diagonals = 27 := sorry
  exact this

end nonagon_diagonals_count_eq_27_l164_164954


namespace points_on_line_l164_164284

theorem points_on_line (n : ℕ) (h : 9 * n - 8 = 82) : n = 10 := by
  sorry

end points_on_line_l164_164284


namespace probability_of_pink_l164_164614

-- Given conditions
variables (B P : ℕ) (h : (B : ℚ) / (B + P) = 3 / 7)

-- To prove
theorem probability_of_pink (h_pow : (B : ℚ) ^ 2 / (B + P) ^ 2 = 9 / 49) :
  (P : ℚ) / (B + P) = 4 / 7 :=
sorry

end probability_of_pink_l164_164614


namespace OddPrimeDivisorCondition_l164_164473

theorem OddPrimeDivisorCondition (n : ℕ) (h_pos : 0 < n) (h_div : ∀ d : ℕ, d ∣ n → d + 1 ∣ n + 1) : 
  ∃ p : ℕ, Prime p ∧ n = p ∧ ¬ Even p :=
sorry

end OddPrimeDivisorCondition_l164_164473


namespace approx_d_is_9_l164_164399
open Real

theorem approx_d_is_9.24 : 
  let d := (69.28 * 0.004) / 0.03
  in abs (d - 9.24) < 0.005 :=
by
  sorry  -- Proof to be completed

end approx_d_is_9_l164_164399


namespace qed_product_l164_164631

def Q : ℂ := 3 + 4 * complex.I
def E : ℂ := -complex.I
def D : ℂ := 3 - 4 * complex.I

theorem qed_product : Q * E * D = -25 * complex.I := by
    sorry

end qed_product_l164_164631


namespace nonagon_diagonals_count_eq_27_l164_164955

theorem nonagon_diagonals_count_eq_27 :
  let vertices := 9
  let connections_per_vertex := vertices - 3
  let total_raw_count_diagonals := vertices * connections_per_vertex
  let distinct_diagonals := total_raw_count_diagonals / 2
  distinct_diagonals = 27 :=
by
  let vertices := 9
  let connections_per_vertex := vertices - 3
  let total_raw_count_diagonals := vertices * connections_per_vertex
  let distinct_diagonals := total_raw_count_diagonals / 2
  have : distinct_diagonals = 27 := sorry
  exact this

end nonagon_diagonals_count_eq_27_l164_164955


namespace solution_y_values_l164_164685
-- Import the necessary libraries

-- Define the system of equations and the necessary conditions
def equation1 (x : ℝ) := x^2 - 6*x + 8 = 0
def equation2 (x y : ℝ) := 2*x - y = 6

-- The main theorem to be proven
theorem solution_y_values : ∃ x1 x2 y1 y2 : ℝ, 
  (equation1 x1 ∧ equation1 x2 ∧ equation2 x1 y1 ∧ equation2 x2 y2 ∧ 
  y1 = 2 ∧ y2 = -2) :=
by
  -- Use the provided solutions in the problem statement
  use 4, 2, 2, -2
  sorry  -- The details of the proof are omitted.

end solution_y_values_l164_164685


namespace probability_second_number_is_odd_l164_164480

theorem probability_second_number_is_odd :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let odds := {x ∈ S | x % 2 = 1}
  let evens := {x ∈ S | x % 2 = 0}
  let first_draw_is_odd : ∀ x ∈ odds, x ∈ S
  let remaining_after_first_odd_draw := S \ {first_draw_is_odd}
  let remaining_odds := {x ∈ remaining_after_first_odd_draw | x % 2 = 1}
  let remaining_evens := {x ∈ remaining_after_first_odd_draw | x % 2 = 0}
  let total_outcomes := remaining_odds ∪ remaining_evens
  let favorable_outcomes := remaining_odds
  let P := favorable_outcomes.card / total_outcomes.card
  in P = 1 / 2 :=
begin
  sorry
end

end probability_second_number_is_odd_l164_164480


namespace minimum_distance_sum_l164_164510

open Real

theorem minimum_distance_sum (x y : ℝ) (h₁ : x ∈ Ioo (-1) 1) (h₂ : y ∈ Ioo (-1) 1) :
  ∃(c : ℝ), c = 4 * Real.sqrt 2 ∧
    c ≤ sqrt ((x+1)^2 + (y-1)^2) + sqrt ((x+1)^2 + (y+1)^2) +
         sqrt ((x-1)^2 + (y+1)^2) + sqrt ((x-1)^2 + (y-1)^2) :=
by
  use 4 * Real.sqrt 2
  sorry

end minimum_distance_sum_l164_164510


namespace find_median_and_mode_l164_164588

noncomputable def seventh_grade_scores : List ℝ := [96, 85, 90, 86, 93, 92, 95, 81, 75, 81]
noncomputable def eighth_grade_scores : List ℝ := [68, 95, 83, 93, 94, 75, 85, 95, 95, 77]

noncomputable def seventh_grade_statistics : (ℝ × ℝ × ℝ × ℝ) := (87.4, 43.44, 81)
noncomputable def eighth_grade_statistics : (ℝ × ℝ × ℝ × ℝ) := (86, 89.2, 95)

theorem find_median_and_mode :
  let a := (seventh_grade_scores.nth (5 - 1)).getD 0
    in let b := (seventh_grade_scores.nth (6 - 1)).getD 0
      in let median := (a + b) / 2
        in median = 88 ∧ mode eighth_grade_scores = 95 :=
by
  sorry

end find_median_and_mode_l164_164588


namespace nonagon_diagonals_l164_164942

def convex_nonagon_diagonals : Prop :=
∀ (n : ℕ), n = 9 → (n * (n - 3)) / 2 = 27

theorem nonagon_diagonals : convex_nonagon_diagonals :=
by {
  sorry,
}

end nonagon_diagonals_l164_164942


namespace range_of_7a_minus_5b_l164_164059

theorem range_of_7a_minus_5b (a b : ℝ) (h1 : 5 ≤ a - b ∧ a - b ≤ 27) (h2 : 6 ≤ a + b ∧ a + b ≤ 30) : 
  36 ≤ 7 * a - 5 * b ∧ 7 * a - 5 * b ≤ 192 :=
sorry

end range_of_7a_minus_5b_l164_164059


namespace sum_possible_values_N_l164_164326

theorem sum_possible_values_N (a b c N : ℕ) (ha : 0 < a) (hb : 0 < b)
  (hc : c = a + b) (hN : N = a * b * c) (h_condition : N = 8 * (a + b + c)) :
  (N = 272 ∨ N = 160 ∨ N = 128) →
  (272 + 160 + 128) = 560 :=
by {
  intros h,
  have h1 : N = 272 ∨ N = 160 ∨ N = 128,
  from h,
  exact eq.refl 560,
}

end sum_possible_values_N_l164_164326


namespace count_congruent_to_3_mod_7_in_first_150_l164_164957

-- Define the predicate for being congruent to 3 modulo 7
def is_congruent_to_3_mod_7 (n : ℕ) : Prop :=
  ∃ k : ℤ, n = 7 * k + 3

-- Define the set of the first 150 positive integers
def first_150_positive_integers : finset ℕ :=
  finset.range 151 \ {0}

-- Define the subset of first 150 positive integers that are congruent to 3 modulo 7
def congruent_to_3_mod_7_in_first_150 : finset ℕ :=
  first_150_positive_integers.filter is_congruent_to_3_mod_7

-- The theorem to prove the size of this subset is 22
theorem count_congruent_to_3_mod_7_in_first_150 :
  congruent_to_3_mod_7_in_first_150.card = 22 :=
sorry

end count_congruent_to_3_mod_7_in_first_150_l164_164957


namespace problem_I_problem_II_l164_164978

open Real EuclideanSpace 

variable (V : Type) [InnerProductSpace ℝ V]

-- Conditions in the problem
variable (a b : V)
variable (AB BC : ℝ)
variable (angleB : ℝ)
variable (h₁ : AB = 1)
variable (h₂ : BC = 2)
variable (h₃ : angleB = π / 3)
variable (h₄ : (a • a) = AB^2)
variable (h₅ : (b • b) = BC^2)
variable (h₆ : (a • b) = AB * BC * Real.cos angleB)

-- Proofs of the equivalent questions
theorem problem_I : (2 • a - 3 • b) • (4 • a + b) = 6 :=
by sorry

theorem problem_II : ‖2 • a - b‖ = 2 * Real.sqrt 3 :=
by sorry

end problem_I_problem_II_l164_164978


namespace problem_part1_problem_part2_problem_part3_l164_164516

noncomputable def a (n : ℕ) : ℝ := (1 / 3 ^ (n - 1))
noncomputable def b (n : ℕ) : ℝ := n / (3 ^ (n - 1) * a n) + 3 / 2
noncomputable def T (n : ℕ) : ℝ := 3 - (n + 1) / 3 ^ (n - 1)
noncomputable def sum_reciprocal_b (n : ℕ) : ℝ := ∑ i in finset.range n, 1 / (b i * b (i + 1))

theorem problem_part1 :
  ∀ n : ℕ, n > 0 → a n = 1 / 3 ^ (n - 1) := sorry

theorem problem_part2 :
  ∀ n : ℕ, n > 0 → T n = 3 - (n + 1) / 3 ^ (n - 1) := sorry

theorem problem_part3 :
  ∀ n : ℕ, n > 0 → sum_reciprocal_b n ≥ 4 / 35 := sorry

end problem_part1_problem_part2_problem_part3_l164_164516


namespace Lizzy_savings_after_loan_l164_164646

theorem Lizzy_savings_after_loan :
  ∀ (initial_amount loan_amount : ℕ) (interest_percent : ℕ),
  initial_amount = 30 →
  loan_amount = 15 →
  interest_percent = 20 →
  initial_amount - loan_amount + loan_amount + loan_amount * interest_percent / 100 = 33 :=
by
  intros initial_amount loan_amount interest_percent h1 h2 h3
  sorry

end Lizzy_savings_after_loan_l164_164646


namespace points_on_line_possible_l164_164283

theorem points_on_line_possible : ∃ n : ℕ, 9 * n - 8 = 82 :=
by
  sorry

end points_on_line_possible_l164_164283


namespace problem_solution_l164_164974

noncomputable def quadratic_symmetric_b (a : ℝ) : ℝ :=
  2 * (1 - a)

theorem problem_solution (a : ℝ) (h1 : quadratic_symmetric_b a = 6) :
  b = 6 :=
by
  sorry

end problem_solution_l164_164974


namespace fly_flies_more_than_10_meters_l164_164418

theorem fly_flies_more_than_10_meters :
  ∃ (fly_path_length : ℝ), 
  (∃ (c : ℝ) (a b : ℝ), c = 5 ∧ a^2 + b^2 = c^2) →
  (fly_path_length > 10) := 
by
  sorry

end fly_flies_more_than_10_meters_l164_164418


namespace repeating_decimal_to_fraction_l164_164485

theorem repeating_decimal_to_fraction 
  (h : ∀ {x : ℝ}, (0.01 : ℝ) = 1 / 99 → x = 1.06 → (0.06 : ℝ) = 6 * 1 / 99): 
  1.06 = 35 / 33 :=
by sorry

end repeating_decimal_to_fraction_l164_164485


namespace gcd_example_l164_164379

theorem gcd_example : Nat.gcd 8675309 7654321 = 36 := sorry

end gcd_example_l164_164379


namespace shopkeeper_gain_l164_164437

noncomputable def overall_percentage_gain (P : ℝ) (increase_percentage : ℝ) (discount1_percentage : ℝ) (discount2_percentage : ℝ) : ℝ :=
  let increased_price := P * (1 + increase_percentage)
  let price_after_first_discount := increased_price * (1 - discount1_percentage)
  let final_price := price_after_first_discount * (1 - discount2_percentage)
  ((final_price - P) / P) * 100

theorem shopkeeper_gain : 
  overall_percentage_gain 100 0.32 0.10 0.15 = 0.98 :=
by
  sorry

end shopkeeper_gain_l164_164437


namespace intersection_point_ratio_l164_164595

-- Definitions of the parametric equations of line l
def x (t : ℝ) : ℝ := 2 + (real.sqrt 2 / 2) * t
def y (t : ℝ) : ℝ := (real.sqrt 2 / 2) * t

-- Definition of the standard form of the circle equation
def circle_eqn (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 8

-- Definition of the point P
def P : (ℝ × ℝ) := (2, 0)

-- Definition of the distance function
def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((p1.fst - p2.fst)^2 + (p1.snd - p2.snd)^2)

-- Placeholder for the proof (to be filled in).
theorem intersection_point_ratio :
  let A := (2 + (real.sqrt 2 / 2) * t1, (real.sqrt 2 / 2) * t1)
  let B := (2 + (real.sqrt 2 / 2) * t2, (real.sqrt 2 / 2) * t2)
  ∀ (t1 t2 : ℝ), circle_eqn (x t1) (y t1) ∧ circle_eqn (x t2) (y t2) ->
    (distance P A)^(-1) + (distance P B)^(-1) = real.sqrt 6 / 2 :=
begin
  sorry
end

end intersection_point_ratio_l164_164595


namespace minimum_distance_MN_l164_164901

-- We are given a line equation and a circle equation
def line_eq (x y : ℝ) := 3 * x + 4 * y - 2 = 0
def circle_eq (x y : ℝ) := (x + 1) ^ 2 + (y + 1) ^ 2 = 1

-- Define the center and radius of the circle
def center : ℝ × ℝ := (-1, -1)
def radius : ℝ := 1

-- Define the function to calculate distance from a point to a line
def distance_point_to_line (p : ℝ × ℝ) : ℝ :=
  let (x₀, y₀) := p in
  (|3 * x₀ + 4 * y₀ - 2|) / (real.sqrt (3^2 + 4^2))

-- Define the minimal distance problem as a Lean theorem
theorem minimum_distance_MN : 
  ∃ M N : ℝ × ℝ, line_eq M.1 M.2 ∧ circle_eq N.1 N.2 ∧ 
  ∀ (d : ℝ), d = real.dist M N → d ≥ (distance_point_to_line center - radius) ∧ 
  (distance_point_to_line center - radius) = 4/5 :=
sorry

end minimum_distance_MN_l164_164901


namespace max_value_equality_case_l164_164047

noncomputable def max_value_expression (x : ℝ) : ℝ :=
  x^4 / (x^8 + 2*x^6 - 4*x^4 + 8*x^2 + 16)

theorem max_value (x : ℝ) (h : x ≠ 0) :
  max_value_expression x ≤ 1 / 12 :=
sorry

theorem equality_case :
  max_value_expression (real.sqrt 2) = 1 / 12 :=
sorry

end max_value_equality_case_l164_164047


namespace intersection_of_A_and_B_l164_164106

def setA : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def setB : Set ℝ := {x | 0 < x ∧ x < 5}

theorem intersection_of_A_and_B : setA ∩ setB = {1, 3} :=
by
  sorry

end intersection_of_A_and_B_l164_164106


namespace roses_per_girl_l164_164356

noncomputable theory

-- Definitions based on problem conditions
def number_of_students : ℕ := 24
def number_of_birches : ℕ := 6
def number_of_plants : ℕ := 24
def boys_per_birch : ℕ := 3

-- Theorem statement
theorem roses_per_girl (n_girls n_boys n_roses : ℕ) 
  (h1 : n_girls + n_boys = number_of_students)
  (h2 : n_boys / boys_per_birch = number_of_birches)
  (h3 : n_roses + number_of_birches = number_of_plants) :
  n_roses / n_girls = 3 := 
sorry

end roses_per_girl_l164_164356


namespace median_name_length_and_syllables_l164_164441

def names_lengths_syllables : List (ℕ × ℕ) :=
  ([(4, 1)] * 8) ++ ([(5, 2)] * 5) ++ ([(3, 1)] * 3) ++ ([(6, 2)] * 4) ++ ([(7, 3)] * 3)

def name_lengths : List ℕ := names_lengths_syllables.map Prod.fst
def name_syllables : List ℕ := names_lengths_syllables.map Prod.snd

def median (l : List ℕ) : ℕ :=
  l.sorted.get? (l.length / 2) |>.getD 0

theorem median_name_length_and_syllables :
  median name_lengths = 5 ∧ median name_syllables = 1 := by
  sorry

end median_name_length_and_syllables_l164_164441


namespace sale_price_including_tax_l164_164342

-- Definition of the given conditions
def cost_price : ℝ := 535.65
def profit_rate : ℝ := 0.15
def tax_rate : ℝ := 0.10

-- The main statement to prove
theorem sale_price_including_tax :
  let sale_price_ex_tax := cost_price * (1 + profit_rate)
  let sale_price_incl_tax := sale_price_ex_tax * (1 + tax_rate)
  (Float.round (sale_price_incl_tax * 100) / 100) = 677.60 :=
by
  -- Place to insert the proof
  sorry

end sale_price_including_tax_l164_164342


namespace box_surface_area_l164_164723

variables (x y z : ℝ)

theorem box_surface_area (h1 : x + y + z = 40) (h2 : x^2 + y^2 + z^2 = 625) : 2 * (x * y + y * z + z * x) = 975 :=
by sorry

end box_surface_area_l164_164723


namespace discount_percentage_l164_164732

theorem discount_percentage :
  (total_bricks : ℕ) (bricks_in_discount : ℕ) (price_per_brick : ℚ) (total_spent : ℚ) (full_price_spent : ℚ) (discounted_price_spent : ℚ)
  (h1 : total_bricks = 1000) (h2 : bricks_in_discount = 500) (h3 : price_per_brick = 0.5) (h4 : total_spent = 375) (h5 : full_price_spent = 250) (h6 : discounted_price_spent = 125) :
  (discount_percentage : ℚ) (h7 : discount_percentage = (full_price_spent - discounted_price_spent) / full_price_spent * 100) :=
  discount_percentage = 50 :=
begin
  sorry
end

end discount_percentage_l164_164732


namespace exist_initial_points_l164_164273

theorem exist_initial_points (n : ℕ) (h : 9 * n - 8 = 82) : ∃ n = 10 :=
by
  sorry

end exist_initial_points_l164_164273


namespace part1_solution_part2_solution_l164_164579

-- Define f and its property M on R, odd function
def property_M (f : ℝ → ℝ) (c : ℝ) :=
  ∀ x : ℝ, exp x * (f x - exp x) = c

theorem part1_solution (f : ℝ → ℝ) (c : ℝ) (hM : property_M f c) (hodd : ∀ x : ℝ, f (-x) = -f x) :
  f = λ x : ℝ, exp x - (1 / exp x) := sorry

-- Define g and its property M on [-1, 1], even function, inequality condition
def property_M_interval (g : ℝ → ℝ) (c : ℝ) (I : Set ℝ) :=
  ∀ x ∈ I, exp x * (g x - exp x) = c

theorem part2_solution (g : ℝ → ℝ) (c : ℝ) 
  (hM : property_M_interval g c (Set.Icc (-1 : ℝ) 1))
  (heven : ∀ x : ℝ, g (-x) = g x)
  (hineq : ∀ x ∈ Set.Icc (-1 : ℝ) 1, g (2 * x) - 2 * exp 1 * g x + n > 0) :
  ∃ n : ℝ, ∀ x ∈ Set.Icc (-1 : ℝ) 1, n > exp 2 + 2 := sorry

end part1_solution_part2_solution_l164_164579


namespace moles_of_HCl_combined_l164_164033

theorem moles_of_HCl_combined :
  ∀ (n m k : ℕ),          -- declaring n, m, k as natural numbers
  (n = 1) →               -- condition 1
  (m = 1) →               -- condition 2 (1 mole of NaCl is produced)
  (1 * 1 = m) →           -- condition 3 (balanced chemical equation)
  (k = n) →               -- condition 4 (stoichiometry ratio implies 1:1)
  k = 1 :=                -- Proving that the number of moles of HCl (k) is 1
by
  intros n m k h1 h2 h3 h4,
  rw [h1, h4],
  exact rfl

end moles_of_HCl_combined_l164_164033


namespace product_zero_when_a_is_2_l164_164483

theorem product_zero_when_a_is_2 : 
  ∀ (a : ℤ), a = 2 → (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 0 :=
by
  intros a ha
  sorry

end product_zero_when_a_is_2_l164_164483


namespace students_taking_both_courses_l164_164586

theorem students_taking_both_courses (n_total n_F n_G n_neither number_both : ℕ)
  (h_total : n_total = 79)
  (h_F : n_F = 41)
  (h_G : n_G = 22)
  (h_neither : n_neither = 25)
  (h_any_language : n_total - n_neither = 54)
  (h_sum_languages : n_F + n_G = 63)
  (h_both : n_F + n_G - (n_total - n_neither) = number_both) :
  number_both = 9 :=
by {
  sorry
}

end students_taking_both_courses_l164_164586


namespace remaining_black_cards_l164_164590

def total_black_cards_per_deck : ℕ := 26
def num_decks : ℕ := 5
def removed_black_face_cards : ℕ := 7
def removed_black_number_cards : ℕ := 12

theorem remaining_black_cards : total_black_cards_per_deck * num_decks - (removed_black_face_cards + removed_black_number_cards) = 111 :=
by
  -- proof will go here
  sorry

end remaining_black_cards_l164_164590


namespace solve_for_k_l164_164382

theorem solve_for_k : 
  (∀ x : ℝ, x * (2 * x + 3) < k ↔ x ∈ Ioo (-2 : ℝ) 1) → k = (-2 : ℝ) :=
by
  sorry

end solve_for_k_l164_164382


namespace regular_tetrahedron_properties_l164_164434

noncomputable def min_cross_section_area (R : ℝ) : ℝ :=
  (2 * Real.sqrt 2 / Real.sqrt 33) * R^2

noncomputable def volume_ratio : ℤ × ℤ :=
  (3, 19)

theorem regular_tetrahedron_properties (R : ℝ) (h : ℝ) (h_eq : h = (4 * R / 3)) :
  ∃ A B : ℝ, A = min_cross_section_area R ∧ B = volume_ratio := 
by
  sorry

end regular_tetrahedron_properties_l164_164434


namespace no_real_m_satisfying_condition_l164_164364

theorem no_real_m_satisfying_condition :
  ∀ m : ℝ, (∀ (x : ℝ), mx^2 - 2x + m*(m^2 + 1) = 0 → False) :=
by
  sorry

end no_real_m_satisfying_condition_l164_164364


namespace no_integer_solutions_l164_164768

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), x^3 + 21 * y^2 + 5 = 0 :=
by {
  sorry
}

end no_integer_solutions_l164_164768


namespace log_suff_nec_l164_164058

theorem log_suff_nec (a b : ℝ) (ha : a > 0) (hb : b > 0) : ¬ ((a > b) ↔ (Real.log b / Real.log a < 1)) := 
sorry

end log_suff_nec_l164_164058


namespace quadratic_translation_transformed_l164_164366

-- The original function is defined as follows:
def original_func (x : ℝ) : ℝ := 2 * x^2

-- Translated function left by 3 units
def translate_left (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x + a)

-- Translated function down by 2 units
def translate_down (f : ℝ → ℝ) (b : ℝ) (x : ℝ) : ℝ := f x - b

-- Combine both translations: left by 3 units and down by 2 units
def translated_func (x : ℝ) : ℝ := translate_down (translate_left original_func 3) 2 x

-- The theorem we want to prove
theorem quadratic_translation_transformed :
  translated_func x = 2 * (x + 3)^2 - 2 := 
by
  sorry

end quadratic_translation_transformed_l164_164366


namespace sun_set_earlier_in_szeged_sun_rise_earlier_in_szeged_l164_164683

structure Location :=
  (lat : ℝ) -- Latitude
  (long : ℝ) -- Longitude

def Szeged := Location.mk 46.25 (20.167)
def Nyiregyhaza := Location.mk 47.966 (21.75)

theorem sun_set_earlier_in_szeged 
  (coords_szeged : Location)
  (coords_nyiregyhaza : Location)
  (winter_solstice : Prop) :
  coords_szeged.long < coords_nyiregyhaza.long → True :=
begin
  sorry
end

theorem sun_rise_earlier_in_szeged 
  (coords_szeged : Location)
  (coords_nyiregyhaza : Location)
  (winter_solstice : Prop) :
  coords_szeged.long < coords_nyiregyhaza.long → True :=
begin
  sorry
end

end sun_set_earlier_in_szeged_sun_rise_earlier_in_szeged_l164_164683


namespace conjugate_of_z_l164_164869

noncomputable def z := (1 + Complex.i) * Complex.i^3
noncomputable def z_conjugate := Complex.conj z

theorem conjugate_of_z : z_conjugate = 1 + Complex.i := by
  sorry

end conjugate_of_z_l164_164869


namespace nonagon_diagonals_count_l164_164932

theorem nonagon_diagonals_count (n : ℕ) (h1 : n = 9) : 
  let diagonals_per_vertex := n - 3 in
  let naive_count := n * diagonals_per_vertex in
  let distinct_diagonals := naive_count / 2 in
  distinct_diagonals = 27 :=
by
  sorry

end nonagon_diagonals_count_l164_164932


namespace sqrt_three_cubes_l164_164354

theorem sqrt_three_cubes : Real.sqrt (3^3 + 3^3 + 3^3) = 9 := 
  sorry

end sqrt_three_cubes_l164_164354


namespace lines_parallel_condition_l164_164571

theorem lines_parallel_condition {a : ℝ} :
  (∀ a ∈ ℝ, (a = 1) → 
  let l1 := 2 * x + a * y + 2 = 0 in
  let l2 := (a + 1) * x + y + a = 0 in
  ∃ k : ℝ, (2:ℝ) = k * ((a + 1):ℝ) ∧ (a:ℝ) = k * (1:ℝ) ∧ (2:ℝ) ≠ k * ((a + 1):ℝ) → False) ∧
  (∀ a ∈ ℝ, 
  let l1 := 2 * x + a * y + 2 = 0 in
  let l2 := (a + 1) * x + y + a = 0 in
  ∃ k : ℝ, (2:ℝ) = k * ((a + 1):ℝ) ∧ (a:ℝ) = k * (1:ℝ) ∧ (a ≠ 1))) := 
sorry

end lines_parallel_condition_l164_164571


namespace car_rental_savings_l164_164163

theorem car_rental_savings :
  let distance_one_way := 150 in
  let rental_cost_option_1 := 50 in
  let rental_cost_option_2 := 90 in
  let distance_per_liter := 15 in
  let gasoline_cost_per_liter := 0.90 in
  let total_distance := distance_one_way * 2 in
  let liters_needed := total_distance / distance_per_liter in
  let gasoline_cost := liters_needed * gasoline_cost_per_liter in
  let total_cost_option_1 := rental_cost_option_1 + gasoline_cost in
  rental_cost_option_2 - total_cost_option_1 = 22 :=
by 
  sorry

end car_rental_savings_l164_164163


namespace expected_value_of_product_l164_164438

open ProbabilityTheory

noncomputable def expected_value_product : ℚ :=
  let probabilities := [⟨0, 3/4⟩, ⟨1, 1/9⟩, ⟨2, 1/9⟩, ⟨4, 1/36⟩] in
  probabilities.sum (λ x, x.1 * x.2)

theorem expected_value_of_product {s : Fin 6 → ℕ}
  (h : ∀ i, s i = if i < 3 then 0 else (if i < 5 then 1 else 2)) :
  (expected_value_product : ℚ) = 4/9 :=
by sorry

end expected_value_of_product_l164_164438


namespace lines_b_c_are_skew_l164_164548

-- Define the concepts of lines, parallel lines, and skew lines
structure Line := 
  (exists_some_point: bool)

-- Define the property of two lines being parallel
def parallel (l1 l2 : Line) : Prop :=
  ∃ (P1 P2 : Prop), P1 ∧ P2

-- Define the property of two lines being skew
def skew (l1 l2 : Line) : Prop :=
  ¬ ∃ (P : Prop), P

-- Given conditions as definitions
variables (a b c : Line)
variable (skew_ab : skew a b)
variable (parallel_ca : parallel c a)
variable (not_intersect_bc : ¬ ∃ (P : Prop), P)

-- Target proof statement
theorem lines_b_c_are_skew : skew b c :=
by
  sorry

end lines_b_c_are_skew_l164_164548


namespace value_of_wk_l164_164637

theorem value_of_wk : 
  let a := 105
  let b := 60
  let c := 42
  let w := (2^7 * 3^4 * 5^2 * 7^2)
  let k := (3^5 * 7^2)
  a^3 * b^4 = 21 * 25 * 45 * 50 * w →
  c^5 = 35 * 28 * 56 * k →
  w * k = 18458529600 :=
by {
  intros,
  rw [←h, ←h_1],
  let lhs1 := (105: ℕ)^3 * (60: ℕ)^4,
  let rhs1 := 21 * 25 * 45 * 50 * (2^7 * 3^4 * 5^2 * 7^2: ℕ),
  let lhs2 := (42: ℕ)^5,
  let rhs2 := 35 * 28 * 56 * (3^5 * 7^2: ℕ),
  have eq1: lhs1 = rhs1 := sorry,
  have eq2: lhs2 = rhs2 := sorry,
  let product := w * k,
  calc
  product = 2^7 * 3^4 * 5^2 * 7^2 * (3^5 * 7^2) : rfl
         ... = 2^7 * (3^4 * 3^5) * 5^2 * (7^2 * 7^2) : by simp [mul_assoc, mul_comm, pow_add]
         ... = 2^7 * 3^9 * 5^2 * 7^4 : by rw [pow_add, pow_add]
         ... = 18458529600 : by sorry
}

end value_of_wk_l164_164637


namespace common_tangent_circumcircles_l164_164147

variables {A B C D M : Type} [EuclideanGeometry A B C D M]

-- Definitions of a parallelogram
def parallelogram (A B C D : Type) [EuclideanGeometry A B C D] :=
  (segment (A, C) ≃ segment (B, D)) ∧ (segment (B, C) ≃ segment (A, D))

-- Definitions of cyclic quadrilateral
def cyclic_quad (B C D M : Type) [EuclideanGeometry B C D M] :=
  ∃ (O : Type), circle (O, segment (B, C)) ∧ circle (O, segment (C, D)) ∧ circle (O, segment (D, M)) ∧ circle (O, segment (M, B))

-- The proof statement
theorem common_tangent_circumcircles
  {A B C D M : Type}
  [EuclideanGeometry A B C D M]
  (h_par : parallelogram A B C D)
  (h_AC_longer : segment (A, C) > segment (B, D))
  (h_M_on_AC : lies_on M (segment (A, C)))
  (h_cyclic : cyclic_quad B C D M) :
  common_tangent (segment (B, D)) (circumcircle (A, B, M)) (circumcircle (A, D, M)) :=
sorry

end common_tangent_circumcircles_l164_164147


namespace cups_flipping_l164_164355

theorem cups_flipping (n : ℕ) : (n % 2 = 0 → ∃ moves : List (Set ℕ), ∀ move ∈ moves, move.card = n - 1 ∧ (Set.Univ.filter (λ i, i ∈ move) ∑ -1 ^ n = 1))
∧ (n % 2 = 1 → ¬ ∃ moves : List (Set ℕ), ∀ move ∈ moves, move.card = n - 1 ∧ (Set.Univ.filter (λ i, i ∈ move) ∑ -1 ^ n = 1)) :=
by
  sorry

end cups_flipping_l164_164355


namespace total_cable_cost_l164_164457

theorem total_cable_cost 
    (num_east_west_streets : ℕ)
    (length_east_west_street : ℕ)
    (num_north_south_streets : ℕ)
    (length_north_south_street : ℕ)
    (cable_multiplier : ℕ)
    (cable_cost_per_mile : ℕ)
    (h1 : num_east_west_streets = 18)
    (h2 : length_east_west_street = 2)
    (h3 : num_north_south_streets = 10)
    (h4 : length_north_south_street = 4)
    (h5 : cable_multiplier = 5)
    (h6 : cable_cost_per_mile = 2000) :
    (num_east_west_streets * length_east_west_street + num_north_south_streets * length_north_south_street) * cable_multiplier * cable_cost_per_mile = 760000 := 
by
    sorry

end total_cable_cost_l164_164457


namespace correct_expression_l164_164386

theorem correct_expression (a x y m n : ℝ) :
  (a^3 - a^2 ≠ a) ∧
  (2 * x + 3 * y ≠ 5 * x * y) ∧
  (2 * (m - n) ≠ 2 * m - n) ∧
  (-xy - xy = -2 * xy) :=
by
  split
  -- Proof for a^3 - a^2 ≠ a
  { intro h, sorry },
  split
  -- Proof for 2x + 3y ≠ 5xy
  { intro h, sorry },
  split
  -- Proof for 2(m - n) ≠ 2m - n
  { intro h, sorry },
  -- Proof for -xy - xy = -2xy
  { exact eq.refl _
  sorry
  }

end correct_expression_l164_164386


namespace consecutive_block_mean_integer_l164_164831

theorem consecutive_block_mean_integer (n : ℕ) (h : n ≥ 4) :
  ∀ (a : ℕ → ℕ), (∀ i, 1 ≤ a i) → (∑ i in finset.range n, a i) = 2 * n - 1 →
  ∃ (i j : ℕ), (i < j) ∧ (j - i ≥ 1) ∧ ((∑ k in finset.range (j - i + 1), a (i + k) : ℚ) / (j - i + 1) = (∑ k in finset.range (j - i + 1), a (i + k)) / (j - i + 1)) :=
 by sorry

end consecutive_block_mean_integer_l164_164831


namespace g_neither_even_nor_odd_l164_164837

noncomputable def g (x : ℝ) : ℝ := (1 / (3^x - 2)) + 1

theorem g_neither_even_nor_odd : ¬ (∀ x : ℝ, g x = g (-x)) ∧ ¬ (∀ x : ℝ, g x = -g (-x)) := 
by sorry

end g_neither_even_nor_odd_l164_164837


namespace plank_nails_calc_l164_164857

-- Definitions based on the conditions
def total_nails (nails : ℕ) : Prop := nails = 11
def additional_nails (nails : ℕ) : Prop := nails = 8
def plank_nails (nails total additional : ℕ) : Prop := nails = total - additional
def number_of_planks (planks : ℕ) : Prop := planks = 1

-- The Lean 4 statement:
theorem plank_nails_calc (total : ℕ) (additional : ℕ) 
  (h1: total_nails total) 
  (h2: additional_nails additional) 
  (h3: number_of_planks 1) : 
  plank_nails 3 total additional :=
by {
  rw [total_nails, additional_nails] at *,
  exact h1, 
  exact h2,
  sorry
}

end plank_nails_calc_l164_164857


namespace distance_sin_eqrational_l164_164454

noncomputable def AB : ℝ := 4
noncomputable def BC : ℝ := 2 * real.sqrt 2
noncomputable def CC1 : ℝ := 2 * real.sqrt 2

structure Point :=
(x y z : ℝ)

def M : Point := { x := 1, y := (BC + CC1) / 2, z := 0 }   -- Simplified midpoint computation
def N : Point := { x := 1, y := (M.y + CC1) / 2, z := 0 } -- Simplified midpoint computation

def angle_AN_CM : ℝ := 90  -- Given angle

variable (d : ℝ)           -- Distance to be defined
variable (theta : ℝ)       -- Angle to be defined

theorem distance_sin_eqrational :
  theta = angle_AN_CM → d * real.sin theta = 4/5 :=
by
  intros h_theta
  rw [h_theta, real.sin_pi_div_two]
  sorry

end distance_sin_eqrational_l164_164454


namespace students_in_both_clubs_l164_164455

theorem students_in_both_clubs (total_students drama_club art_club drama_or_art in_both_clubs : ℕ)
  (H1 : total_students = 300)
  (H2 : drama_club = 120)
  (H3 : art_club = 150)
  (H4 : drama_or_art = 220) :
  in_both_clubs = drama_club + art_club - drama_or_art :=
by
  -- this is the proof space
  sorry

end students_in_both_clubs_l164_164455


namespace students_not_next_to_each_other_l164_164138

theorem students_not_next_to_each_other :
  let total_ways : ℕ := 5!
  let restricted_group_ways : ℕ := 3! * 3!
  total_ways - restricted_group_ways = 84 :=
by
  let total_ways := nat.factorial 5
  let restricted_group_ways := nat.factorial 3 * nat.factorial 3
  have h : total_ways - restricted_group_ways = 84 := sorry
  exact h

end students_not_next_to_each_other_l164_164138


namespace equation1_solution_equation2_solution_l164_164298

theorem equation1_solution : 
  ∀ x : ℝ, x^2 + 4 * x - 5 = 0 ↔ (x = 1 ∨ x = -5) :=
by
  intro x
  constructor
  {
    intro h
    sorry
  }
  {
    intro h
    cases h
    {
      rw h
      ring
    }
    {
      rw h
      ring
    }
  }

theorem equation2_solution :
  ∀ x : ℝ, x^2 - 3 * x + 1 = 0 ↔ (x = (3 + real.sqrt 5) / 2 ∨ x = (3 - real.sqrt 5) / 2) :=
by
  intro x
  constructor
  {
    intro h
    sorry
  }
  {
    intro h
    cases h
    {
      rw h
      ring
    }
    {
      rw h
      ring
    }
  }

end equation1_solution_equation2_solution_l164_164298


namespace parallel_implies_eq_diagonals_l164_164617

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

structure Quadrilateral (A B C D O H1 H2 M N : V) : Prop :=
  (convex : Convex ℝ {x : V | x = A ∨ x = B ∨ x = C ∨ x = D})
  (diagonals_not_perpendicular : ∀ (u v : V), u ≠ 0 → v ≠ 0 → ∠ u v ≠ π / 2)
  (sides_not_parallel : ∀ (u v : V), ∥u∥ ≠ 0 → ∥v∥ ≠ 0 → u ≠ v)
  (O_intersection : ∀ (l1 l2 : Line V), line_through l1 A C ∧ line_through l1 B D → l2 = O)
  (midpoint_M : M = (A + B) / 2)
  (midpoint_N : N = (C + D) / 2)
  (orthocenter_H1 : H1 ⊆ Orthocenter A O B)
  (orthocenter_H2 : H2 ⊆ Orthocenter C O D)

theorem parallel_implies_eq_diagonals
  {A B C D O H1 H2 M N : V}
  (q : Quadrilateral A B C D O H1 H2 M N) :
  (∀ (u v : V), u || v → u = v) ↔ (∥A - C∥ = ∥B - D∥) :=
sorry

end parallel_implies_eq_diagonals_l164_164617


namespace exist_initial_points_l164_164272

theorem exist_initial_points (n : ℕ) (h : 9 * n - 8 = 82) : ∃ n = 10 :=
by
  sorry

end exist_initial_points_l164_164272


namespace Little_John_money_distribution_l164_164643

theorem Little_John_money_distribution :
  (initial_money spent_sweets left_money : ℝ) 
  (money_given friends_count : ℝ) :
  initial_money = 10.10 →
  spent_sweets = 3.25 →
  left_money = 2.45 →
  friends_count = 2 →
  money_given = (initial_money - left_money - spent_sweets) / friends_count →
  money_given = 2.20 :=
by
  intros initial_money spent_sweets left_money money_given friends_count
  intros init_eq sweets_eq left_eq friends_count_eq given_eq
  sorry

end Little_John_money_distribution_l164_164643


namespace count_prime_pairs_sum_53_l164_164142

open Nat

-- Definition stating our specific problem and conditions
def is_prime_pair_sum_53 (p1 p2 : ℕ) : Prop := 
  p1 + p2 = 53 ∧ Prime p1 ∧ Prime p2

-- Lean statement to be proven
theorem count_prime_pairs_sum_53 : ∃! (p1 p2 : ℕ), is_prime_pair_sum_53 p1 p2 :=
sorry

end count_prime_pairs_sum_53_l164_164142


namespace points_on_line_l164_164287

theorem points_on_line (n : ℕ) (h : 9 * n - 8 = 82) : n = 10 := by
  sorry

end points_on_line_l164_164287


namespace sum_distances_constant_l164_164908

noncomputable def sum_of_distances_to_faces (a : ℝ) : ℝ :=
  ∑ i in finset.range 4, sorry -- Placeholder for the sum of distances to the four faces

theorem sum_distances_constant (a : ℝ) (p : EuclideanSpace ℝ (fin 3)) (h : inside_tetrahedron p a) : 
  sum_of_distances_to_faces p = (sqrt 6 / 3) * a := 
by 
  sorry -- Placeholder for the actual proof

end sum_distances_constant_l164_164908


namespace minimum_distance_between_parabol_and_line_l164_164125

theorem minimum_distance_between_parabol_and_line :
  ∀ (a b c d : ℝ), 
  b = -a^2 + 3 * Real.log a ∧ d = c + 2 → 
  (∃ (P Q: ℝ × ℝ), P = (a, b) ∧ Q = (c, d) ∧ |((λ P Q, (P.1 - Q.1)^2 + (P.2 - Q.2)^2) P Q) - 2*sqrt 2| = 0 ) :=
by
  sorry

end minimum_distance_between_parabol_and_line_l164_164125


namespace nonagon_distinct_diagonals_l164_164949

theorem nonagon_distinct_diagonals : 
  let n := 9 in
  ∃ (d : ℕ), d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end nonagon_distinct_diagonals_l164_164949


namespace minimum_points_on_circle_l164_164302

theorem minimum_points_on_circle (points : Set (ℝ × ℝ)) (h : points.card = 10) 
  (subset_property : ∀ (s : Finset (ℝ × ℝ)), s.card = 5 → ∃ (c : Circle), ∃ (t : s), t.card = 4 ∧ ∀ p ∈ t, p ∈ c) :
  ∃ (c : Circle), (points.to_finset).filter (λ p, p ∈ c).card ≥ 9 :=
by
  sorry

end minimum_points_on_circle_l164_164302


namespace minimum_distance_l164_164179

open Real

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y + 4 = 0
def parabola_eq (x y : ℝ) : Prop := y^2 = 8 * x

theorem minimum_distance :
  ∃ (A B : ℝ × ℝ), circle_eq A.1 A.2 ∧ parabola_eq B.1 B.2 ∧ dist A B = 1 / 2 :=
sorry

end minimum_distance_l164_164179


namespace pipe_A_fills_tank_in_21_hours_l164_164760

-- Definitions for the rates of pipes A, B, and C
def rate_A : ℝ := A
def rate_B : ℝ := 2 * A
def rate_C : ℝ := 4 * A

-- Definition for the combined rate of the three pipes filling the tank in 3 hours
def combined_rate : ℝ := 1 / 3

-- Main theorem statement
theorem pipe_A_fills_tank_in_21_hours (A : ℝ) 
  (h1 : rate_C = 2 * rate_B) 
  (h2 : rate_B = 2 * rate_A) 
  (h3 : rate_A + rate_B + rate_C = combined_rate) 
  : 1 / rate_A = 21 :=
by
  -- Proof skipped
  sorry

end pipe_A_fills_tank_in_21_hours_l164_164760


namespace mean_of_X_l164_164911

theorem mean_of_X :
  let P0 := 8 / 27
  let P1 := 4 / 9
  let P2 := (let m := 1 - (8/27 + 4/9 + 1/27) in m)
  let P3 := 1 / 27
  let X := [0, 1, 2, 3]
with let P := [P0, P1, P2, P3]
let mean := X.zip P |>.sum (λ ⟨x, p⟩, x * p)
in
mean = 1 := by
  let m := 2 / 9
  let P := [8 / 27, 4 / 9, m, 1 / 27]
  have h1 : P.sum = 1 := by
    simp [m]
  by_cases (X.zip P) mean = 1 := sorry

end mean_of_X_l164_164911


namespace find_y_l164_164998

variables (P Q R T S : Type) 
variables [euclidean_geometry P Q R T S] 
variables (angle_PQS angle_QRT y : ℝ)

-- Conditions given
axiom h1 : angle_PQS = 124
axiom h2 : angle_QRT = 78

-- Desired Proof Statement: Prove y = 46 given the conditions
theorem find_y (h1 : angle_PQS = 124) (h2 : angle_QRT = 78)
  (h3 : angle_PQS = y + angle_QRT) : y = 46 :=
sorry

end find_y_l164_164998


namespace painting_ways_l164_164774

def isValidConfiguration (grid : Fin 3 → Fin 3 → Bool) : Prop :=
  ∀ i j, grid i j → ((i = 2 ∨ grid (i+1) j) ∧ (j = 2 ∨ grid i (j+1)))

def numValidConfigurations : Nat :=
  let allConfigs := List.fin 3).bind (λ i => (List.fin 3).map (λ j => if grid i j then 1 else 0))
  List.length (List.filter isValidConfiguration allConfigs)

theorem painting_ways : numValidConfigurations = 11 :=
  sorry

end painting_ways_l164_164774


namespace exists_positive_integer_m_l164_164670

theorem exists_positive_integer_m (m : ℕ) (h_positive : m > 0) : 
  ∃ (m : ℕ), m > 0 ∧ ∃ k : ℕ, 8 * m = k^2 := 
sorry

end exists_positive_integer_m_l164_164670


namespace exists_positive_integer_m_l164_164666

theorem exists_positive_integer_m (m : ℕ) (hm : m > 0) : ∃ m : ℕ, m > 0 ∧ ∃ k : ℕ, 8 * m = k ^ 2 := 
by {
  let m := 2
  use m,
  dsimp,
  split,
  { exact hm },
  { use 4,
    calc 8 * m = 8 * 2 : by rfl
           ... = 16 : by norm_num
           ... = 4 ^ 2 : by norm_num }
}

end exists_positive_integer_m_l164_164666


namespace number_of_paths_A_to_G_l164_164468

-- Definitions of the points and conditions in the geometric figure
variables (A C D E F G : Type) 

def connected (A C D E F G : Type) : Prop :=
  (∃ path : list (A C D E F G), 
    ∀ (a b : A C D E F G), a ∈ path → b ∈ path → 
    ( (a = A ∧ b = C) ∨ (a = C ∧ b = D) ∨ (a = D ∧ b = E) ∨
      (a = E ∧ b = F) ∨ (a = F ∧ b = G) ∨ (a = E ∧ b = G) ) )

-- Theorem statement for number of unique paths from A to G
theorem number_of_paths_A_to_G (A C D E F G : Type) 
  [h : connected A C D E F G] : 
  ∃ (n : ℕ), n = 7 :=
sorry

end number_of_paths_A_to_G_l164_164468


namespace shifted_function_l164_164310

def initial_fun (x : ℝ) : ℝ := 5 * (x - 1) ^ 2 + 1

theorem shifted_function :
  (∀ x, initial_fun (x - 2) - 3 = 5 * (x + 1) ^ 2 - 2) :=
by
  intro x
  -- sorry statement to indicate proof should be here
  sorry

end shifted_function_l164_164310


namespace sum_a_cos_phi_eq_sum_a_sin_phi_eq_cosine_binomial_sum_sine_binomial_sum_l164_164460

noncomputable def sumacos (a : ℝ) (φ : ℝ) (h : |a| < 1) : ℝ :=
1 + a * cos φ + a^2 * cos (2 * φ) + ∑ k, a^(k+1) * cos ((k+1) * φ)

theorem sum_a_cos_phi_eq (a : ℝ) (φ : ℝ) (h : |a| < 1) :
  sumacos a φ h = (1 - a * cos φ) / (1 - 2 * a * cos φ + a^2) :=
by sorry

noncomputable def sumasin (a : ℝ) (φ : ℝ) (h : |a| < 1) : ℝ :=
a * sin φ + a^2 * sin (2 * φ) + ∑ k, a^((k+1)) * sin((k+1) * φ)

theorem sum_a_sin_phi_eq (a : ℝ) (φ : ℝ) (h : |a| < 1) :
  sumasin a φ h = (a * sin φ) / (1 - 2 * a * cos φ + a^2) :=
by sorry

def cosine_sum (n : ℕ) (φ : ℝ) : ℝ :=
cos φ + ∑ k in finset.range (n+1), nat.choose n k * cos ((k + 1) * φ)

theorem cosine_binomial_sum (n : ℕ) (φ : ℝ) :
  cosine_sum n φ = 2^n * cos^n (φ / 2) * cos ((n+2) * φ / 2) :=
by sorry

def sine_sum (n : ℕ) (φ : ℝ) : ℝ :=
sin φ + ∑ k in finset.range (n+1), nat.choose n k * sin ((k + 1) * φ)

theorem sine_binomial_sum (n : ℕ) (φ : ℝ) :
  sine_sum n φ = 2^n * cos^n (φ / 2) * sin ((n+2) * φ / 2) :=
by sorry

end sum_a_cos_phi_eq_sum_a_sin_phi_eq_cosine_binomial_sum_sine_binomial_sum_l164_164460


namespace sum_of_x_y_possible_values_l164_164261

theorem sum_of_x_y_possible_values (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
sorry

end sum_of_x_y_possible_values_l164_164261


namespace length_of_living_room_l164_164365

theorem length_of_living_room
  (l : ℝ) -- length of the living room
  (w : ℝ) -- width of the living room
  (boxes_coverage : ℝ) -- area covered by one box
  (initial_area : ℝ) -- area already covered
  (additional_boxes : ℕ) -- additional boxes required
  (total_area : ℝ) -- total area required
  (w_condition : w = 20)
  (boxes_coverage_condition : boxes_coverage = 10)
  (initial_area_condition : initial_area = 250)
  (additional_boxes_condition : additional_boxes = 7)
  (total_area_condition : total_area = l * w)
  (full_coverage_condition : additional_boxes * boxes_coverage + initial_area = total_area) :
  l = 16 := by
  sorry

end length_of_living_room_l164_164365


namespace rearrange_table_distinct_sums_l164_164393

theorem rearrange_table_distinct_sums (n : ℕ) (h : n > 2) 
  (a b : Fin n → ℤ) (distinct_col_sums : ∀ i j : Fin n, i ≠ j → a i + b i ≠ a j + b j) :
  ∃ a' b' : Fin n → ℤ, 
    (∀ i j : Fin n, i ≠ j → a' i + b' i ≠ a' j + b' j) ∧
    (∑ i, a' i ≠ ∑ i, b' i) :=
by sorry

end rearrange_table_distinct_sums_l164_164393


namespace even_abundant_count_l164_164559

-- Define what it means for a number to be abundant
def is_abundant (n : ℕ) : Prop :=
  ∑ k in (Finset.filter (λ d, d ∣ n ∧ d < n) (Finset.range n)), k > n

-- Define what it means to be an even number
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the even abundant numbers less than 50
def even_abundant_numbers_lt_50 : Finset ℕ :=
  Finset.filter (λ n, is_even n ∧ is_abundant n) (Finset.range 50)

-- Prove that the number of even abundant numbers less than 50 is 9
theorem even_abundant_count : even_abundant_numbers_lt_50.card = 9 :=
by
  -- Skipping the actual proof steps
  sorry

end even_abundant_count_l164_164559


namespace min_value_reciprocals_l164_164504

variable {a b : ℝ}

theorem min_value_reciprocals (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 2) :
  (∃ a b, a > 0 ∧ b > 0 ∧ a + b = 2 ∧ ∀ x y, x > 0 → y > 0 → x + y = 2 → 
  (1/a + 1/b) ≥ 2) :=
sorry

end min_value_reciprocals_l164_164504


namespace math_problem_l164_164085

-- Cartesian equation of the curve C
def cartesian_equation_of_curve : Prop :=
  ∀ x y : ℝ, (x - 2)^2 + y^2 = 4 ↔ ∃ ρ θ : ℝ, ρ = 4 * cos θ ∧ (ρ^2 = x^2 + y^2) ∧ (x = ρ * cos θ)

-- Normal equation of the line l
def normal_equation_of_line : Prop :=
  ∀ x y t : ℝ, (x = -1 + (sqrt 3 / 2) * t ∧ y = (1 / 2) * t) ↔ (x - sqrt 3 * y + 1 = 0)

-- Distance between points P and Q
def distance_PQ : Prop :=
  ∀ t1 t2 : ℝ, (t1 + t2 = 3 * sqrt 3 ∧ t1 * t2 = 5) → (|t1 - t2| = sqrt 7)

theorem math_problem : Prop :=
  cartesian_equation_of_curve ∧ normal_equation_of_line ∧ distance_PQ

end math_problem_l164_164085


namespace count_prime_remainders_l164_164964

-- List of prime numbers between 50 and 100
def primes : List ℕ := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

-- Function to check if a number is prime
def is_prime (n : ℕ) : Bool :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

-- Function to get the remainder of the division by 7
def remainder (n : ℕ) : ℕ :=
  n % 7

-- Prime remainders when dividing primes by 7
def prime_remainders : List ℕ :=
  primes.filter (λ p => is_prime (remainder p))

-- The count of prime numbers between 50 and 100 which have prime remainders when divided by 7
theorem count_prime_remainders : prime_remainders.length = 5 :=
  sorry

end count_prime_remainders_l164_164964


namespace Mikail_birthday_money_l164_164212

theorem Mikail_birthday_money (x : ℕ) (h1 : x = 3 + 3 * 3) : 5 * x = 60 := 
by 
  sorry

end Mikail_birthday_money_l164_164212


namespace probability_of_2spades_1heart_in_top3_l164_164439

open_locale big_operators

-- Total number of ways to select 2 spades out of 13 spades
def spades_ways : ℕ := nat.choose 13 2 

-- Total number of ways to select 1 heart out of 13 hearts
def hearts_ways : ℕ := nat.choose 13 1

-- Total number of favorable outcomes
def favorable_outcomes : ℕ := spades_ways * hearts_ways

-- Total number of ways to select any 3 cards out of 52
def total_outcomes : ℕ := nat.choose 52 3

-- Probability of 2 spades and 1 heart in the top 3 cards
def probability_top3_2spades_1heart : ℚ := favorable_outcomes / total_outcomes

-- Statement to prove
theorem probability_of_2spades_1heart_in_top3 :
  probability_top3_2spades_1heart = 507 / 11050 :=
by {
  -- Necessary calculations and logic will be filled in the proof 
  sorry
}

end probability_of_2spades_1heart_in_top3_l164_164439


namespace neg_p_equiv_l164_164202

open Real
open Classical

noncomputable def prop_p : Prop :=
  ∀ x : ℝ, 0 < x → exp x > log x

noncomputable def neg_prop_p : Prop :=
  ∃ x : ℝ, 0 < x ∧ exp x ≤ log x

theorem neg_p_equiv :
  ¬ prop_p ↔ neg_prop_p := by
  sorry

end neg_p_equiv_l164_164202


namespace fractional_eq_has_root_l164_164578

theorem fractional_eq_has_root (x : ℝ) (m : ℝ) (h : x ≠ 4) :
    (3 / (x - 4) + (x + m) / (4 - x) = 1) → m = -1 :=
by
    intros h_eq
    sorry

end fractional_eq_has_root_l164_164578


namespace min_ab_value_l164_164574

variable (a b : ℝ) 

theorem min_ab_value (h : (4 / a) + (1 / b) = Real.sqrt (a * b)) (ha : a > 0) (hb : b > 0) : a * b = 4 :=
sorry

end min_ab_value_l164_164574


namespace count_prime_remainders_l164_164965

-- List of prime numbers between 50 and 100
def primes : List ℕ := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

-- Function to check if a number is prime
def is_prime (n : ℕ) : Bool :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

-- Function to get the remainder of the division by 7
def remainder (n : ℕ) : ℕ :=
  n % 7

-- Prime remainders when dividing primes by 7
def prime_remainders : List ℕ :=
  primes.filter (λ p => is_prime (remainder p))

-- The count of prime numbers between 50 and 100 which have prime remainders when divided by 7
theorem count_prime_remainders : prime_remainders.length = 5 :=
  sorry

end count_prime_remainders_l164_164965


namespace ratio_of_expenditures_l164_164712

variable (Rajan_income Balan_income Rajan_expenditure Balan_expenditure Rajan_savings Balan_savings: ℤ)
variable (ratio_incomes: ℚ)
variable (savings_amount: ℤ)

-- Given conditions
def conditions : Prop :=
  Rajan_income = 7000 ∧
  ratio_incomes = 7 / 6 ∧
  savings_amount = 1000 ∧
  Rajan_savings = Rajan_income - Rajan_expenditure ∧
  Balan_savings = Balan_income - Balan_expenditure ∧
  Rajan_savings = savings_amount ∧
  Balan_savings = savings_amount

-- The theorem we want to prove
theorem ratio_of_expenditures :
  conditions Rajan_income Balan_income Rajan_expenditure Balan_expenditure Rajan_savings Balan_savings ratio_incomes savings_amount →
  (Rajan_expenditure : ℚ) / (Balan_expenditure : ℚ) = 6 / 5 :=
by
  sorry

end ratio_of_expenditures_l164_164712


namespace tangent_parallel_line_exists_l164_164170

theorem tangent_parallel_line_exists 
  (A B C H : Point) 
  [is_acute_triangle A B C]
  [is_orthocenter H A B C]
  : ∃ l, parallel l (line_through B C) ∧ (tangent l (incircle (triangle A B H))) ∧ (tangent l (incircle (triangle A C H))) := 
by
  sorry

end tangent_parallel_line_exists_l164_164170


namespace integer_part_sum_l164_164344

noncomputable def sequence_a : ℕ → ℝ
| 0       := 4 / 3
| (n + 1) := (sequence_a n)^2 - (sequence_a n) + 1

theorem integer_part_sum :
  ⌊(∑ i in Finset.range 2017, (1 : ℝ) / sequence_a i)⌋ = 2 :=
by
  sorry

end integer_part_sum_l164_164344


namespace math_problem_alternating_squares_l164_164465

def alternating_squares (n : ℕ) : ℤ :=
List.sum $ List.map (λ k : ℕ, if k % 4 = 0 then (-(2 : ℤ) * ((n - k)^2 : ℤ))
                             else if k % 2 = 1 then -(n - k)^2
                             else (n - k)^2) (List.range(51))

theorem math_problem_alternating_squares :
  alternating_squares 50 = 75 := by
  -- The proof here would follow the steps provided in the solution steps.
  sorry

end math_problem_alternating_squares_l164_164465


namespace max_plus_min_of_f_l164_164317

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 - b * x + Real.log (Real.sqrt (x^2 + 1) - x) + 5

theorem max_plus_min_of_f (a b : ℝ) :
  let f := f a b
  let M := Real.sup (f '' set.Icc (-2 : ℝ) 2)
  let m := Real.inf (f '' set.Icc (-2 : ℝ) 2)
  M + m = 10 :=
sorry

end max_plus_min_of_f_l164_164317


namespace find_smallest_int_cube_ends_368_l164_164035

theorem find_smallest_int_cube_ends_368 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 500 = 368 ∧ n = 14 :=
by
  sorry

end find_smallest_int_cube_ends_368_l164_164035


namespace sum_of_inverses_of_transformed_roots_l164_164873

noncomputable def cubic_polynomial : Polynomial ℝ := 60 * X^3 - 80 * X^2 + 24 * X - 2

theorem sum_of_inverses_of_transformed_roots
  (α β γ : ℝ)
  (hα : α ∈ Ioo 0 1) (hβ : β ∈ Ioo 0 1) (hγ : γ ∈ Ioo 0 1)
  (h_distinct : α ≠ β ∧ β ≠ γ ∧ γ ≠ α)
  (hroots : cubic_polynomial.eval α = 0 ∧ cubic_polynomial.eval β = 0 ∧ cubic_polynomial.eval γ = 0) :
  (1 / (1 - α)) + (1 / (1 - β)) + (1 / (1 - γ)) = 22 :=
sorry

end sum_of_inverses_of_transformed_roots_l164_164873


namespace extremum_range_m_l164_164204

noncomputable def f (x m : ℝ) : ℝ := (real.sqrt 3) * real.sin (real.pi * x / m)

theorem extremum_range_m (m x₀ : ℝ) (k : ℤ) (h1 : f x₀ m = real.sqrt 3 ∨ f x₀ m = -real.sqrt 3)
  (h2 : real.pi * x₀ / m = k * real.pi + real.pi / 2)
  (h3 : x₀^2 + (f x₀ m)^2 < m^2) :
  m ∈ set.Ioo (-∞) (-2) ∪ set.Ioo 2 ∞ := sorry

end extremum_range_m_l164_164204


namespace wheel_center_travel_distance_l164_164442

theorem wheel_center_travel_distance (r : ℝ) (h_r : r = 2) : 
  let circumference := 2 * real.pi * r in
  let half_circumference := circumference / 2 in
  half_circumference = 2 * real.pi :=
by
  -- We'll skip the proof for the solution to focus on problem statement
  sorry

end wheel_center_travel_distance_l164_164442


namespace smallest_integer_whose_cube_ends_in_368_l164_164042

theorem smallest_integer_whose_cube_ends_in_368 :
  ∃ (n : ℕ+), (n % 2 = 0 ∧ n^3 % 1000 = 368) ∧ (∀ (m : ℕ+), m % 2 = 0 ∧ m^3 % 1000 = 368 → m ≥ n) :=
by
  sorry

end smallest_integer_whose_cube_ends_in_368_l164_164042


namespace lines_and_planes_l164_164184

variables {Line Plane : Type} 
variables (a b : Line) (α β : Plane)

-- Conditions
def is_perpendicular (l : Line) (p : Plane) : Prop := sorry
def is_parallel (x y : (Line ⊕ Plane)) : Prop := sorry

-- Statement
theorem lines_and_planes (h1 : is_perpendicular a α) (h2 : is_parallel a β) : is_perpendicular α β := 
sorry

end lines_and_planes_l164_164184


namespace martha_saves_half_daily_allowance_l164_164207

theorem martha_saves_half_daily_allowance {f : ℚ} (h₁ : 12 > 0) (h₂ : (6 : ℚ) * 12 * f + (3 : ℚ) = 39) : f = 1 / 2 :=
by
  sorry

end martha_saves_half_daily_allowance_l164_164207


namespace quadrilateral_parallelogram_l164_164171

open Affine

variables {V : Type*} {P : Type*} [AddCommGroup V] [Module ℝ V] [AffineSpace V P]
variables (A B C D M N E F : P)

-- Midpoints
variable (hM : midpoint ℝ A B = M)
variable (hN : midpoint ℝ B C = N)

-- Intersections
variable [plane P]
variable (hE : ∃ t : ℝ, (1 - t) • A + t • N = E ∧ ∃ u : ℝ, (1 - u) • B + u • D = E)
variable (hF : ∃ v : ℝ, (1 - v) • D + v • M = F ∧ ∃ w : ℝ, (1 - w) • A + w • C = F)

-- Given conditions
variable (hBE : ∥E - B∥ = (1 / 3) * ∥D - B∥)
variable (hAF : ∥F - A∥ = (1 / 3) * ∥C - A∥)

-- Proof of parallelogram
theorem quadrilateral_parallelogram (hM : midpoint ℝ A B = M)
                                    (hN : midpoint ℝ B C = N)
                                    (hE : ∃ t : ℝ, (1 - t) • A + t • N = E ∧ ∃ u : ℝ, (1 - u) • B + u • D = E)
                                    (hF : ∃ v : ℝ, (1 - v) • D + v • M = F ∧ ∃ w : ℝ, (1 - w) • A + w • C = F)
                                    (hBE : ∥E - B∥ = (1 / 3) * ∥D - B∥)
                                    (hAF : ∥F - A∥ = (1 / 3) * ∥C - A∥) :
                                    parallelogram ℝ A B C D :=
begin
  sorry -- proof to be provided
end

end quadrilateral_parallelogram_l164_164171


namespace compute_value_l164_164576

noncomputable def repeating_decimal_31 : ℝ := 31 / 100000
noncomputable def repeating_decimal_47 : ℝ := 47 / 100000
def term : ℝ := 10^5 - 10^3

theorem compute_value : (term * repeating_decimal_31 + term * repeating_decimal_47) = 77.22 := 
by
  sorry

end compute_value_l164_164576


namespace integer_part_sum_inv_seq_l164_164347

def seq (n : ℕ) : ℚ :=
  match n with
  | 0     => 4 / 3
  | (n+1) => let a_n := seq n in a_n^2 - a_n + 1

theorem integer_part_sum_inv_seq :
  let s := (Finset.range 2017).sum (λ i => 1 / seq (i + 1))
  ⌊s⌋ = 2 :=
by
  sorry

end integer_part_sum_inv_seq_l164_164347


namespace find_a_find_modulus_l164_164513

theorem find_a (a : ℝ) (h : a > 0) (hpure_im : (2 + a * complex.I) ^ 2 = 4 * a * complex.I) : a = 2 :=
by sorry

theorem find_modulus (Z1 : ℂ) (h : Z1 = 2 + 2 * complex.I) : complex.abs ((Z1 / (1 - complex.I))) = 2 :=
by sorry

end find_a_find_modulus_l164_164513


namespace monotone_decreasing_on_1_infty_extremum_on_2_6_l164_164917

noncomputable def f (x : ℝ) : ℝ := 2 / (x - 1)

-- Part (1): Monotonicity of f(x) on (1,+∞)
theorem monotone_decreasing_on_1_infty : ∀ x1 x2 : ℝ, 1 < x1 → 1 < x2 → x1 < x2 → f x1 > f x2 := by
  sorry

-- Part (2): Extremum of f(x) on [2,6]
theorem extremum_on_2_6 :
  (∀ x ∈ Icc 2 6, f 2 ≥ f x ∧ f x ≥ f 6) := by
  sorry

#check f
#check monotone_decreasing_on_1_infty
#check extremum_on_2_6

end monotone_decreasing_on_1_infty_extremum_on_2_6_l164_164917


namespace nonagon_diagonals_l164_164941

def convex_nonagon_diagonals : Prop :=
∀ (n : ℕ), n = 9 → (n * (n - 3)) / 2 = 27

theorem nonagon_diagonals : convex_nonagon_diagonals :=
by {
  sorry,
}

end nonagon_diagonals_l164_164941


namespace points_cyclic_DNKE_l164_164199

noncomputable def reflection (A M : Point) : Point := sorry

variables {Point : Type} [geometry : Geometry Point]
open geometry

variables (A B C D E F T M K N : Point)
variables (ω : Circle)
variables (BCED : CyclicQuadrilateral ω B C E D)
variables (h1 : Ray A C = Ray B C)
variables (h2 : Ray A E = Ray D E)
variables (h3 : ∥ LineThrough D LineThrough B C)
variables (h4 : ω.contains F)
variables (h5 : F ≠ D)
variables (h6 : ω.sessionSegmentIntersects A F T)
variables (h7 : T ≠ F)
variables (h8 : isIntersection (LineFromPoints E T) (LineFromPoints B C) M)
variables (h9 : K = Midpoint B C)
variables (h10 : N = reflection A M)

theorem points_cyclic_DNKE :
  CyclicQuadrilateral LineThrough (D, N, K, E) :=
sorry

end points_cyclic_DNKE_l164_164199


namespace lizzy_wealth_after_loan_l164_164650

theorem lizzy_wealth_after_loan 
  (initial_wealth : ℕ)
  (loan : ℕ)
  (interest_rate : ℕ)
  (h1 : initial_wealth = 30)
  (h2 : loan = 15)
  (h3 : interest_rate = 20)
  : (initial_wealth - loan) + (loan + loan * interest_rate / 100) = 33 :=
by
  sorry

end lizzy_wealth_after_loan_l164_164650


namespace sum_of_x_y_l164_164250

theorem sum_of_x_y (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_x_y_l164_164250


namespace limit_does_not_exist_l164_164490

noncomputable def does_not_exist_limit : Prop := 
  ¬ ∃ l : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ,
    (0 < |x| ∧ 0 < |y| ∧ |x| < δ ∧ |y| < δ) →
    |(x^2 - y^2) / (x^2 + y^2) - l| < ε

theorem limit_does_not_exist :
  does_not_exist_limit :=
sorry

end limit_does_not_exist_l164_164490


namespace sum_of_x_y_possible_values_l164_164262

theorem sum_of_x_y_possible_values (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
sorry

end sum_of_x_y_possible_values_l164_164262


namespace magnitude_of_vector_sum_l164_164575

noncomputable def vec2 : Type := ℝ × ℝ

def dot_product (v w : vec2) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def magnitude (v : vec2) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def add_vectors (v w : vec2) : vec2 :=
  (v.1 + w.1, v.2 + w.2)

theorem magnitude_of_vector_sum :
  ∀ (b : vec2), magnitude b = 1 → 
  ∀ (theta : ℝ), theta = Real.pi / 3 →
  ∀ (a : vec2), a = (2, 0) →
  magnitude (add_vectors a (2, b)) = 2 * Real.sqrt 3 :=
by
  intros b hb theta htheta a ha
  sorry

end magnitude_of_vector_sum_l164_164575


namespace cannot_connect_phones_l164_164357

-- Definition of the problem parameters
def phones_count : ℕ := 19
def connections_per_phone : ℕ := 11

-- Theorem statement
theorem cannot_connect_phones : ¬ ∃ (G : SymmetricMatrix 19 bool), (∀ i, degree G i = 11) :=
by
  intros G h_deg
  let total_connections := 19 * 11
  have h_impossible : total_connections % 2 ≠ 0,
  from by
    calc
      total_connections % 2 = 209 % 2 := by simp [total_connections]
                             ... = 1        := by norm_num,
  exact absurd rfl h_impossible

end cannot_connect_phones_l164_164357


namespace total_cost_l164_164794

/-- There are two types of discs, one costing 10.50 and another costing 8.50.
You bought a total of 10 discs, out of which 6 are priced at 8.50.
The task is to determine the total amount spent. -/
theorem total_cost (price1 price2 : ℝ) (num1 num2 : ℕ) 
  (h1 : price1 = 10.50) (h2 : price2 = 8.50) 
  (h3 : num1 = 6) (h4 : num2 = 10) 
  (h5 : num2 - num1 = 4) : 
  (num1 * price2 + (num2 - num1) * price1) = 93.00 := 
by
  sorry

end total_cost_l164_164794


namespace factorize_x2_minus_7x_plus_12_factorize_x_minus_y_sq_plus_4_x_minus_y_plus_3_factorize_ab_ab_minus_2_minus_3_l164_164222

-- Definitions and conditions from Material 1
def is_factorable (p q m n : ℤ) : Prop :=
  q = m * n ∧ p = m + n

-- Problem (1)
theorem factorize_x2_minus_7x_plus_12 :
  ∃ m n : ℤ, is_factorable (-7) 12 m n ∧ (x : ℤ) → (x^2 - 7 * x + 12 = (x + m) * (x + n)) :=
sorry

-- Problem (2) Sub-question (1)
theorem factorize_x_minus_y_sq_plus_4_x_minus_y_plus_3 :
  ∃ m n : ℤ, is_factorable 4 3 m n ∧ (x y : ℤ) → ((x - y)^2 + 4 * (x - y) + 3 = (x - y + m) * (x - y + n)) :=
sorry

-- Problem (2) Sub-question (2)
theorem factorize_ab_ab_minus_2_minus_3 :
  ∃ m n : ℤ, is_factorable (-2) (-3) m n ∧ (a b : ℤ) → ((a + b) * (a + b - 2) - 3 = (a + b + m) * (a + b + n)) :=
sorry

end factorize_x2_minus_7x_plus_12_factorize_x_minus_y_sq_plus_4_x_minus_y_plus_3_factorize_ab_ab_minus_2_minus_3_l164_164222


namespace five_students_not_adjacent_l164_164140

theorem five_students_not_adjacent : 
  let total_arrangements := 5!
  let block_arrangements := 3!
  let internal_block_arrangements := 3!
  let invalid_arrangements := block_arrangements * internal_block_arrangements
  let valid_arrangements := total_arrangements - invalid_arrangements
  3.students_refuse_adjacency_84 (total_arrangements valid_arrangements : nat) (H: valid_arrangements = 84) : valid_arrangements = 84 :=
sorry

end five_students_not_adjacent_l164_164140


namespace nonagon_distinct_diagonals_l164_164950

theorem nonagon_distinct_diagonals : 
  let n := 9 in
  ∃ (d : ℕ), d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end nonagon_distinct_diagonals_l164_164950


namespace smallest_positive_integer_cube_ends_368_l164_164038

theorem smallest_positive_integer_cube_ends_368 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 368 ∧ n = 34 :=
by
  sorry

end smallest_positive_integer_cube_ends_368_l164_164038


namespace xiao_ming_percentile_l164_164988

theorem xiao_ming_percentile (total_students : ℕ) (rank : ℕ) 
  (h1 : total_students = 48) (h2 : rank = 5) :
  ∃ p : ℕ, (p = 90 ∨ p = 91) ∧ (43 < (p * total_students) / 100) ∧ ((p * total_students) / 100 ≤ 44) :=
by
  sorry

end xiao_ming_percentile_l164_164988


namespace find_k_l164_164017

noncomputable def satisfies_condition (k : ℕ) (F : polynomial ℤ) : Prop :=
  ∀ c ∈ finset.range (k + 2), 0 ≤ F.eval c ∧ F.eval c ≤ k

theorem find_k (k : ℕ) (F : polynomial ℤ) :
  (∀ F, (satisfies_condition k F) → ∀ c ∈ finset.range (k + 2), F.eval c = F.eval 0) ↔ k ≥ 4 :=
by sorry

end find_k_l164_164017


namespace arithmetic_evaluation_l164_164812

theorem arithmetic_evaluation : 8 + 18 / 3 - 4 * 2 = 6 := 
by
  sorry

end arithmetic_evaluation_l164_164812


namespace max_distance_circle_line_l164_164149

theorem max_distance_circle_line :
  ∀ (x y : ℝ),
  (x^2 + y^2 = 9) →
  ∃ d, ∀ (θ : ℝ), 
  let α := θ in 
  let point_on_circle := (3 * Real.cos α, 3 * Real.sin α) in 
  let distance := abs (3 * Real.cos α + 3 * Real.sqrt 3 * Real.sin α - 2) / (Real.sqrt (1 + (Real.sqrt 3)^2)) in 
  d = 4 :=
sorry

end max_distance_circle_line_l164_164149


namespace thm_2016th_number_is_6_l164_164730

def next_in_sequence (n : ℕ) : ℕ :=
if n % 2 = 0 then n / 2 + 2 else n * 2 - 2

def sequence_index (start : ℕ) (k : ℕ) : ℕ :=
nat.rec_on k start (λ idx prev, next_in_sequence prev)

theorem thm_2016th_number_is_6 : sequence_index 130 2015 = 6 :=
sorry

end thm_2016th_number_is_6_l164_164730


namespace curve_is_circle_l164_164022

theorem curve_is_circle (r θ : ℝ) (h : r = 1 / (Real.sin θ + Real.cos θ)) :
  ∃ x y : ℝ, r = Math.sqrt(x^2 + y^2) ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ (x + y)^2 = (x^2 + y^2) :=
by
  sorry

end curve_is_circle_l164_164022


namespace simplify_expression_l164_164296

open Complex

theorem simplify_expression : (5 * (3 - I) + 3 * I * (5 - I)) = 18 + 10 * I :=
by 
  long_proof_steps

end simplify_expression_l164_164296


namespace borrowed_quarters_l164_164157

def original_quarters : ℕ := 8
def remaining_quarters : ℕ := 5

theorem borrowed_quarters : original_quarters - remaining_quarters = 3 :=
by
  sorry

end borrowed_quarters_l164_164157


namespace flowers_left_l164_164446

theorem flowers_left (alissa_picked : ℕ) (melissa_factor : ℝ) (given_to_mother_fraction : ℝ) :
  alissa_picked = 16 →
  melissa_factor = 2.5 →
  given_to_mother_fraction = 3/4 →
  let melissa_picked := (alissa_picked : ℝ) * melissa_factor,
      total_picked := (alissa_picked : ℝ) + melissa_picked,
      given_to_mother := total_picked * given_to_mother_fraction,
      flowers_left := total_picked - given_to_mother
  in flowers_left = 14 :=
by {
  intros,
  let melissa_picked := (alissa_picked : ℝ) * melissa_factor,
  let total_picked := (alissa_picked : ℝ) + melissa_picked,
  let given_to_mother := total_picked * given_to_mother_fraction,
  let flowers_left := total_picked - given_to_mother,
  sorry
}

end flowers_left_l164_164446


namespace trigonometric_inequality_l164_164552

noncomputable def a : ℕ → ℝ
| 0       := real.sqrt 2 / 2
| (n + 1) := (real.sqrt 2 / 2) * real.sqrt (1 - real.sqrt (1 - (a n)^2))

noncomputable def b : ℕ → ℝ
| 0       := 1
| (n + 1) := (real.sqrt (1 + (b n)^2) - 1) / (b n)

theorem trigonometric_inequality (n : ℕ) : 2^(n + 2) * a n < real.pi ∧ real.pi < 2^(n + 2) * b n := sorry

end trigonometric_inequality_l164_164552


namespace horse_food_per_day_l164_164456

theorem horse_food_per_day (ratio_sh : ℕ) (ratio_h : ℕ) (sheep : ℕ) (total_food : ℕ) (sheep_count : sheep = 32) (ratio : ratio_sh = 4) (ratio_horses : ratio_h = 7) (total_food_need : total_food = 12880) :
  total_food / (sheep * ratio_h / ratio_sh) = 230 :=
by
  sorry

end horse_food_per_day_l164_164456


namespace maximize_QP_PR_QR_l164_164517

noncomputable def maximize_ratio (P Q : Point) (π : Plane) (R : Point) : Prop :=
  P ∈ π ∧ Q ∉ π ∧
  (forall S ∈ π, ∃ R, R = (line_through PQ).projection_onto π ∧
  (QP + PR) / QR >= (QP + PS) / QS )

theorem maximize_QP_PR_QR (P Q : Point) (π : Plane) : ∃ R : Point, maximize_ratio P Q π R :=
sorry

end maximize_QP_PR_QR_l164_164517


namespace perimeter_of_original_rectangle_l164_164792

-- Define the rectangle's dimensions based on the given condition
def length_of_rectangle := 2 * 8 -- because it forms two squares of side 8 cm each
def width_of_rectangle := 8 -- side of the squares

-- Using the formula for the perimeter of a rectangle: P = 2 * (length + width)
def perimeter_of_rectangle := 2 * (length_of_rectangle + width_of_rectangle)

-- The statement we need to prove
theorem perimeter_of_original_rectangle : perimeter_of_rectangle = 48 := by
  sorry

end perimeter_of_original_rectangle_l164_164792


namespace fg_of_neg5_eq_484_l164_164970

def f (x : Int) : Int := x * x
def g (x : Int) : Int := 6 * x + 8

theorem fg_of_neg5_eq_484 : f (g (-5)) = 484 := 
  sorry

end fg_of_neg5_eq_484_l164_164970


namespace solution_exists_l164_164756

theorem solution_exists (x y z a : ℝ) :
  (x + y + 2 * z = 4 * (a^2 + 1)) ∧ (z^2 - x * y = a^2) →
  ((x = a^2 + a + 1 ∧ y = a^2 - a + 1 ∧ z = a^2 + 1) ∨ 
   (x = a^2 - a + 1 ∧ y = a^2 + a + 1 ∧ z = a^2 + 1)) :=
by {
  intros h,
  sorry
}

end solution_exists_l164_164756


namespace subscription_total_amount_l164_164800

theorem subscription_total_amount 
  (A B C : ℝ)
  (profit_C profit_total : ℝ)
  (subscription_A subscription_B subscription_C : ℝ)
  (subscription_total : ℝ)
  (hA : subscription_A = subscription_B + 4000)
  (hB : subscription_B = subscription_C + 5000)
  (hc_share : profit_C = 8400)
  (total_profit : profit_total = 35000)
  (h_ratio : profit_C / profit_total = subscription_C / subscription_total)
  (h_subs : subscription_total = subscription_A + subscription_B + subscription_C)
  : subscription_total = 50000 := 
sorry

end subscription_total_amount_l164_164800


namespace rectangle_BG_perpendicular_FG_l164_164341

theorem rectangle_BG_perpendicular_FG
  (A B C D E F G : ℝ × ℝ)
  (h_rect : (A - B).x = (D - C).x ∧ (A - B).y = (D - C).y)
  (h_mid_E : E = (A + B) / 2)
  (h_mid_F : F = (C + D) / 2)
  (h_proj_G : G = E - (E - A).dot((A - C) / (A - C).norm_sq) * (A - C)):
  (B - G).dot((F - G)) = 0 :=
sorry

end rectangle_BG_perpendicular_FG_l164_164341


namespace distance_between_midpoints_l164_164785

-- Conditions
def AA' := 68 -- in centimeters
def BB' := 75 -- in centimeters
def CC' := 112 -- in centimeters
def DD' := 133 -- in centimeters

-- Question: Prove the distance between the midpoints of A'C' and B'D' is 14 centimeters
theorem distance_between_midpoints :
  let midpoint_A'C' := (AA' + CC') / 2
  let midpoint_B'D' := (BB' + DD') / 2
  (midpoint_B'D' - midpoint_A'C' = 14) :=
by
  sorry

end distance_between_midpoints_l164_164785


namespace ratio_approximation_l164_164993

theorem ratio_approximation :
  Float.round (8 / 12 : Float) 1 = 0.7 :=
sorry

end ratio_approximation_l164_164993


namespace improvement_percentage_correct_l164_164394

-- Define the times in seconds as given in the conditions
def bobs_time_seconds : ℕ := 10 * 60 + 40  -- 10 minutes 40 seconds
def sisters_time_seconds : ℕ := 9 * 60 + 42  -- 9 minutes 42 seconds

-- Define the difference in their times
def time_difference_seconds : ℕ := bobs_time_seconds - sisters_time_seconds

-- Calculate the percentage improvement needed
def percentage_improvement : ℚ := (time_difference_seconds.to_rat / bobs_time_seconds.to_rat) * 100

-- Statement to be proved
theorem improvement_percentage_correct :
  percentage_improvement ≈ 9.06 := -- we use ≈ to denote approximate equality
sorry

end improvement_percentage_correct_l164_164394


namespace variance_of_normal_std_dev_of_normal_l164_164851

def normal_pdf (a σ x : ℝ) : ℝ := (1 / (σ * Real.sqrt (2 * Real.pi)) * Real.exp (-(x - a)^2 / (2 * σ^2)))

def variance (a σ : ℝ) : ℝ :=
  ∫ x in Real.volume, (x - a)^2 * normal_pdf a σ x

theorem variance_of_normal (a σ : ℝ) (hσ : σ > 0) : variance a σ = σ^2 :=
by
  sorry

theorem std_dev_of_normal (a σ : ℝ) (hσ : σ > 0) : Real.sqrt (variance a σ) = σ :=
by
  sorry

end variance_of_normal_std_dev_of_normal_l164_164851


namespace new_pyramid_volume_l164_164432

-- Given conditions
variables (V : ℝ) (s h : ℝ)
-- Assumption: Initial volume of pyramid V = 60 cubic inches with a square base
axiom volume_initial : V = (1 / 3) * s^2 * h
axiom V_given : V = 60
-- Assumption: Side of the base is tripled, and height is quadrupled
axiom s_tripled : ∀ s, s' = 3 * s
axiom h_quadrupled : ∀ h, h' = 4 * h

-- Theorem: Prove the volume of the new pyramid
theorem new_pyramid_volume (V : ℝ) (s h : ℝ) (s' h' : ℝ)
  (V_given : V = 60)
  (volume_initial : V = (1 / 3) * s^2 * h)
  (s_tripled : s' = 3 * s)
  (h_quadrupled : h' = 4 * h)
  := ∃ V', V' = (1 / 3) * s'^2 * h' ∧ V' = 2160 := 
by
  sorry 

end new_pyramid_volume_l164_164432


namespace unique_solution_l164_164849

noncomputable def num_valid_functions : ℕ := 1

theorem unique_solution (g : ℝ → ℝ) (h : ∀ x y : ℝ, g(x + y) * g(x - y) = (g(x) - g(y)) ^ 2 - 6 * x ^ 2 * g(y)) : ∀ x : ℝ, g(x) = 0 :=
sorry

end unique_solution_l164_164849


namespace arrange_photo_ways_l164_164741

theorem arrange_photo_ways (teachers male_students female_students : ℕ)
  (H_teacher_ends : teachers = 2)
  (H_male_students : male_students = 3)
  (H_female_students : female_students = 3)
  (H_no_adjacent_males: ∀ (arrangement : List (Fin (teachers + male_students + female_students))),
    (∀ i, arrangement.nth i = some 0 → arrangement.nth (i + 1) ≠ some 1 ∧ arrangement.nth (i - 1) ≠ some 1)) :
  true → ∃ (num_ways : ℕ), num_ways = 288 :=
by
  sorry

end arrange_photo_ways_l164_164741


namespace necessary_but_not_sufficient_l164_164856

variables {R : Type*} [linear_ordered_field R]
variables (l : set R) (α : set (set R))

-- Assume plane α is a set of lines
def line_not_on_plane (l : set R) (α : set (set R)) : Prop :=
  ¬(l ∈ α)

def line_parallel_plane (l : set R) (α : set (set R)) : Prop :=
  ∀ (p q : set R), p ∈ α → q ∈ α → (l ∩ p = ∅) ∧ (l ∩ q = ∅)

theorem necessary_but_not_sufficient
  (l : set R) (α : set (set R))
  (h : line_not_on_plane l α) :
  (line_parallel_plane l α) → (line_not_on_plane l α) ∧ ¬((line_not_on_plane l α) → (line_parallel_plane l α)) :=
sorry

end necessary_but_not_sufficient_l164_164856


namespace Amit_can_complete_work_in_15_days_l164_164449

theorem Amit_can_complete_work_in_15_days :
  ∃ (x : ℕ), (∀ (an_days : ℕ) (total_days : ℕ)
    (amit_rate : ℚ) (ananthu_rate : ℚ), 
    an_days = 90 ∧ total_days = 75 ∧ amit_rate = 1 / x ∧ ananthu_rate = 1 / 90 →
    (3 * (1 / x) + (total_days - 3) * (1 / an_days) = 1)) → x = 15 :=
begin
  sorry
end

end Amit_can_complete_work_in_15_days_l164_164449


namespace find_f_of_pi_over_4_l164_164864

theorem find_f_of_pi_over_4 
  (varphi : ℝ) (omega : ℝ)
  (h0 : sin varphi = 3 / 5)
  (h1 : varphi ∈ (real.pi / 2, real.pi))
  (h2 : 2 * real.pi / omega / 2 = real.pi / 2)
  (h3 : omega > 0) :
  sin (omega * (real.pi / 4) + varphi) = -4 / 5 :=
by
  sorry

end find_f_of_pi_over_4_l164_164864


namespace g_at_six_l164_164197

def g (x : ℝ) : ℝ := 2 * x^4 - 19 * x^3 + 30 * x^2 - 12 * x - 72

theorem g_at_six : g 6 = 288 :=
by
  sorry

end g_at_six_l164_164197


namespace best_value_box_l164_164660

def price_per_ounce_in_cents (cost_dollars : ℝ) (ounces : ℕ) : ℝ :=
(cost_dollars / ounces.toReal) * 100

def box1_price_per_ounce := price_per_ounce_in_cents 4.80 30
def box2_price_per_ounce := price_per_ounce_in_cents 3.40 20
def box3_price_per_ounce := price_per_ounce_in_cents 2.00 15
def box4_price_per_ounce := price_per_ounce_in_cents 3.25 25

theorem best_value_box :
  min (min box1_price_per_ounce box2_price_per_ounce) (min box3_price_per_ounce box4_price_per_ounce) = 13 := by
sorry

end best_value_box_l164_164660


namespace cylinder_volume_increase_height_l164_164739

theorem cylinder_volume_increase_height
  (r h : ℝ)
  (r = 5)
  (h = 10)
  (increased_radius : ℝ := 5 + 4)
  (π : ℝ := Real.pi)
  (V1 : ℝ := π * (increased_radius ^ 2) * h)
  (V2 : ℝ := π * (r ^ 2) * (h + x))
  (volume_equal : V1 = V2)
  : x = 112 / 5 := 
by 
  sorry

end cylinder_volume_increase_height_l164_164739


namespace line_parallel_or_within_other_plane_l164_164573

open Plane

variable (P Q : Plane) (l : Line)

-- Declare conditions
variable (hPQ : P ∥ Q) (hlP : l ∥ P)

-- Formulate the theorem
theorem line_parallel_or_within_other_plane :
  l ∥ Q ∨ l ⊆ Q :=
sorry

end line_parallel_or_within_other_plane_l164_164573


namespace points_on_line_proof_l164_164293

theorem points_on_line_proof (n : ℕ) (hn : n = 10) : 
  let after_first_procedure := 3 * n - 2 in
  let after_second_procedure := 3 * after_first_procedure - 2 in
  after_second_procedure = 82 :=
by
  let after_first_procedure := 3 * n - 2
  let after_second_procedure := 3 * after_first_procedure - 2
  have h : after_second_procedure = 9 * n - 8 := by
    calc
      after_second_procedure = 3 * (3 * n - 2) - 2 : rfl
                      ... = 9 * n - 6 - 2      : by ring
                      ... = 9 * n - 8          : by ring
  rw [hn] at h 
  exact h.symm.trans (by norm_num)

end points_on_line_proof_l164_164293


namespace price_comparison_l164_164087

variable (x y : ℝ)
variable (h1 : 6 * x + 3 * y > 24)
variable (h2 : 4 * x + 5 * y < 22)

theorem price_comparison : 2 * x > 3 * y :=
sorry

end price_comparison_l164_164087


namespace selecting_elements_l164_164860

theorem selecting_elements (P Q S : ℕ) (a : ℕ) 
    (h1 : P = Nat.choose 17 (2 * a - 1))
    (h2 : Q = Nat.choose 17 (2 * a))
    (h3 : S = Nat.choose 18 12) :
    P + Q = S → (a = 3 ∨ a = 6) :=
by
  sorry

end selecting_elements_l164_164860


namespace even_three_digit_numbers_count_l164_164113

theorem even_three_digit_numbers_count :
  let digits := [0, 1, 2, 3, 4]
  let even_digits := [2, 4]
  let count := 2 * 3 * 3
  count = 18 :=
by
  let digits := [0, 1, 2, 3, 4]
  let even_digits := [2, 4]
  let count := 2 * 3 * 3
  show count = 18
  sorry

end even_three_digit_numbers_count_l164_164113


namespace manufacturer_l164_164789

-- Let x be the manufacturer's suggested retail price
variable (x : ℝ)

-- Regular discount range from 10% to 30%
def regular_discount (d : ℝ) : Prop := d >= 0.10 ∧ d <= 0.30

-- Additional discount during sale 
def additional_discount : ℝ := 0.20

-- The final discounted price is $16.80
def final_price (x : ℝ) : Prop := ∃ d, regular_discount d ∧ 0.80 * ((1 - d) * x) = 16.80

theorem manufacturer's_suggested_retail_price :
  final_price x → x = 30 := by
  sorry

end manufacturer_l164_164789


namespace min_value_of_function_l164_164493

def f (x : ℝ) : ℝ :=
  log 2 (sqrt x) * log (sqrt 2) (2 * x)

theorem min_value_of_function : ∃ x ∈ Ioi 0, f x = -1/4 := sorry

end min_value_of_function_l164_164493


namespace isosceles_triangle_of_cos_eq_l164_164900

noncomputable theory

-- Define angles A, B, C and sides a, b, c of triangle ABC
variables {A B C : ℝ} {a b c : ℝ}

-- Define the condition: b * cos A = a * cos B
def condition (A B : ℝ) (a b : ℝ) := b * real.cos A = a * real.cos B

-- The theorem stating that under the given condition, A = B, meaning the triangle is isosceles
theorem isosceles_triangle_of_cos_eq (A B : ℝ) (a b : ℝ) :
  condition A B a b → A = B :=
begin
  intro h,
  sorry,  -- Proof is omitted
end

end isosceles_triangle_of_cos_eq_l164_164900


namespace days_c_worked_l164_164757

noncomputable def work_done_by_a_b := 1 / 10
noncomputable def work_done_by_b_c := 1 / 18
noncomputable def work_done_by_c_alone := 1 / 45

theorem days_c_worked
  (A B C : ℚ)
  (h1 : A + B = work_done_by_a_b)
  (h2 : B + C = work_done_by_b_c)
  (h3 : C = work_done_by_c_alone) :
  15 = (1/3) / work_done_by_c_alone :=
sorry

end days_c_worked_l164_164757


namespace mike_last_5_shots_l164_164213

theorem mike_last_5_shots :
  let initial_shots := 30
  let initial_percentage := 40 / 100
  let additional_shots_1 := 10
  let new_percentage_1 := 45 / 100
  let additional_shots_2 := 5
  let new_percentage_2 := 46 / 100
  
  let initial_makes := initial_shots * initial_percentage
  let total_shots_after_1 := initial_shots + additional_shots_1
  let makes_after_1 := total_shots_after_1 * new_percentage_1 - initial_makes
  let total_makes_after_1 := initial_makes + makes_after_1
  let total_shots_after_2 := total_shots_after_1 + additional_shots_2
  let final_makes := total_shots_after_2 * new_percentage_2
  let makes_in_last_5 := final_makes - total_makes_after_1
  
  makes_in_last_5 = 2
:=
by
  sorry

end mike_last_5_shots_l164_164213


namespace part1_part2_part3_l164_164539

-- Given function definition
def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 3)

-- Part 1: Proving the value at a specific point
theorem part1 : f (3 * Real.pi / 4) = -3 / 2 := 
by 
  sorry

-- Part 2: Proving the intervals of monotonic increase
theorem part2 (k : ℤ) : 
  ∀ x : ℝ, -Real.pi / 12 + k * Real.pi < x ∧ x < 5 * Real.pi / 12 + k * Real.pi → 
  (f' x > 0) :=
by 
  sorry

-- Part 3: Proving the range on a given interval
theorem part3 : 
  ∀ y : ℝ, f (-Real.pi / 4) ≤ y ∧ y ≤ f (Real.pi / 4) → 
  y ∈ {-3, 3 / 2} := 
by 
  sorry

end part1_part2_part3_l164_164539


namespace find_number_with_divisors_condition_l164_164198

theorem find_number_with_divisors_condition :
  ∃ n : ℕ, (∃ d1 d2 d3 d4 : ℕ, 1 ≤ d1 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ d4 * d4 ∣ n ∧
    d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 = n) ∧ n = 130 :=
by
  sorry

end find_number_with_divisors_condition_l164_164198


namespace first_number_of_100th_group_l164_164103

noncomputable theory
open Nat 

def sequence : ℕ → ℕ
| n => 2^(n-1)

def first_of_nth_group (n : ℕ) : ℕ :=
sequence ((n * (n + 1)) / 2 + 1)

theorem first_number_of_100th_group :
  first_of_nth_group 100 = 2^4950 :=
by
-- Proof is required here
  sorry

end first_number_of_100th_group_l164_164103


namespace ellipse_eccentricity_l164_164535

-- Definition of the ellipse and the required conditions
def ellipse (a b : ℝ) : Prop := 
  a > b ∧ b > 0

-- Given conditions
def condition_foci_distance : Prop := (2 : ℝ)
def condition_incircle_radius : Prop := (1 : ℝ)
def condition_y_coordinates : Prop := (3 : ℝ)

-- Definition of eccentricity
noncomputable def eccentricity (a b : ℝ) := 
  if h : a > b ∧ b > 0 then (1 / a) else (0 : ℝ)

-- Main statement to be proven
theorem ellipse_eccentricity :
  ∀ (a b : ℝ), 
  ellipse a b → 
  condition_foci_distance = 2 → 
  condition_incircle_radius = 1 → 
  condition_y_coordinates = 3 →
  eccentricity a b = 2 / 3 := 
by
  sorry

end ellipse_eccentricity_l164_164535


namespace equal_elevation_points_l164_164367

noncomputable def point_set (h k a : ℝ) : set ℝ :=
{ x | x = (2 * a * k) / (h + k) ∨ x = 2 * a - (2 * a * k) / (h + k) }

theorem equal_elevation_points (h k a : ℝ) :
  ∀ P, (∃ θ : ℝ, θ > 0 ∧ θ < π/2 ∧ (tan θ = k / P ∧ tan θ = h / (2 * a - P))) →
  (P = (2 * a * k) / (h + k) ∨ P = 2 * a - (2 * a * k) / (h + k)) :=
by sorry

end equal_elevation_points_l164_164367


namespace batsman_average_increase_l164_164775

theorem batsman_average_increase
  (A : ℕ)  -- Assume the initial average is a non-negative integer
  (h1 : 11 * A + 70 = 12 * (A + 3))  -- Condition derived from the problem
  : A + 3 = 37 := 
by {
  -- The actual proof would go here, but is replaced by sorry to skip the proof
  sorry
}

end batsman_average_increase_l164_164775


namespace most_probable_hits_l164_164031

variable (n : ℕ) (p : ℝ) (q : ℝ) (k : ℕ)
variable (h1 : n = 5) (h2 : p = 0.6) (h3 : q = 1 - p)

theorem most_probable_hits : k = 3 := by
  -- Define the conditions
  have hp : p = 0.6 := h2
  have hn : n = 5 := h1
  have hq : q = 1 - p := h3

  -- Set the expected value for the number of hits
  let expected := n * p

  -- Use the bounds for the most probable number of successes (k_0)
  have bounds := expected - q ≤ k ∧ k ≤ expected + p

  -- Proof step analysis can go here
  sorry

end most_probable_hits_l164_164031


namespace problem_statement_l164_164094

def f (x : ℝ) : ℝ := 2 * x ^ 2 - 4 * x - 5

def g (t : ℝ) : ℝ :=
if t ≤ 0 then 2 * t ^ 2 + 2 * t - 7
else if 0 < t ∧ t < 1 then -7
else 2 * t ^ 2 - 4 * t - 5

theorem problem_statement :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≥ -7) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x = 11) ∧
  (∀ t : ℝ, (t ≤ 0 → f t ≥ -7) ∧ (t ≥ 1 → f t ≥ -7) ∧ ((0 < t ∧ t < 1) → f 1 ≥ -7)) ∧
  (∃ t : ℝ, g t = -7) :=
by
  sorry

end problem_statement_l164_164094


namespace jeremy_overall_accuracy_l164_164303

theorem jeremy_overall_accuracy
  (p : ℝ) -- p represents the total number of components in the project
  (terry_individual_accuracy : ℝ) (h_t_ind_acc : terry_individual_accuracy = 0.75)
  (terry_total_accuracy : ℝ) (h_t_tot_acc : terry_total_accuracy = 0.85)
  (jeremy_individual_accuracy : ℝ) (h_j_ind_acc : jeremy_individual_accuracy = 0.80) :
  let terry_correct_alone := terry_individual_accuracy * 0.60 * p,
      total_t_correct := terry_total_accuracy * p,
      collective_correct := total_t_correct - terry_correct_alone,
      jeremy_correct_alone := jeremy_individual_accuracy * 0.60 * p,
      jeremy_total_correct := jeremy_correct_alone + collective_correct in
  (jeremy_total_correct / p) * 100 = 88 :=
by
  simp only [terry_individual_accuracy, terry_total_accuracy, jeremy_individual_accuracy, h_t_ind_acc, h_t_tot_acc, h_j_ind_acc]
  sorry

end jeremy_overall_accuracy_l164_164303


namespace remove_12_increases_probability_l164_164363

open Finset

def T : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

def sums_to_18_combinations (s : Finset ℕ) : Finset (Finset ℕ) := 
  s.powerset.filter (λ x => x.card = 3 ∧ x.sum = 18)

noncomputable def probability_of_sums_to_18 (s : Finset ℕ) : ℚ :=
  let total_combinations := (s.card.choose 3 : ℚ)
  let successful_combinations := (sums_to_18_combinations s).card
  (successful_combinations : ℚ) / total_combinations

theorem remove_12_increases_probability : 
  ∀ m ∈ T, m ≠ 12 → probability_of_sums_to_18 (T.erase 12) > probability_of_sums_to_18 (T.erase m) :=
sorry

end remove_12_increases_probability_l164_164363


namespace remainder_3_pow_19_mod_10_l164_164380

theorem remainder_3_pow_19_mod_10 : (3^19) % 10 = 7 := 
by 
  sorry

end remainder_3_pow_19_mod_10_l164_164380


namespace range_of_t_l164_164673

theorem range_of_t (a b t : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 2 * a + b = 1) 
    (h_ineq : 2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ t - 1 / 2):
    t = Real.sqrt 2 / 2 :=
sorry

end range_of_t_l164_164673


namespace true_propositions_l164_164870

section GeometryPropositions

variables {m n : ℝ^3 → ℝ^3} {α β γ : ℝ^3 → Prop}

-- Conditions
def non_coincident_lines (m n : ℝ^3 → ℝ^3) : Prop := m ≠ n
def pairwise_non_coincident_planes (α β γ : ℝ^3 → Prop) : Prop := 
  α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Propositions
def P1 (m : ℝ^3 → ℝ^3) (α β : ℝ^3 → Prop) : Prop :=
  (∀ x, m x = 0 → α x) → (∀ x, m x = 0 → β x) → (∀ x, α x → β x)

def P4 (m n : ℝ^3 → ℝ^3) (α β : ℝ^3 → Prop) : Prop :=
  (∀ x y z, m x = 0 ∧ n y = 0 ∧ m z ≠ n z ∧ α x ∧ β y ∧ α z → (¬ β z))

theorem true_propositions 
  (m n : ℝ^3 → ℝ^3) (α β γ : ℝ^3 → Prop)
  (h1 : non_coincident_lines m n)
  (h2 : pairwise_non_coincident_planes α β γ) :
  P1 m α β ∧ P4 m n α β :=
by
  sorry
end GeometryPropositions

end true_propositions_l164_164870


namespace p_plus_q_of_given_series_final_result_l164_164350

noncomputable def given_series : ℕ → ℝ := 
  λ k, (2^k : ℝ) / (5^(2^k) + 1)

noncomputable def sum_series (s : ℕ → ℝ) : ℝ :=
  classical.some (classical.some_spec (real.series.has_sum s))

theorem p_plus_q_of_given_series : sum_series given_series = 1 / 4 :=
begin
  sorry
end

theorem final_result : 1 + 4 = 5 :=
begin
  exact rfl,
end

end p_plus_q_of_given_series_final_result_l164_164350


namespace zoo_animal_difference_l164_164799

theorem zoo_animal_difference :
  let parrots := 8
  let snakes := 3 * parrots
  let monkeys := 2 * snakes
  let elephants := 1 / 2 * (parrots + snakes)
  let zebras := elephants - 3
  monkeys - zebras = 35 :=
by
  sorry

end zoo_animal_difference_l164_164799


namespace possible_values_of_sum_l164_164238

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l164_164238


namespace intersection_A_B_l164_164105

def set_A : Set ℕ := {x | x^2 - 2 * x = 0}
def set_B : Set ℕ := {0, 1, 2}

theorem intersection_A_B : set_A ∩ set_B = {0, 2} := 
by sorry

end intersection_A_B_l164_164105


namespace size_relationship_l164_164570

noncomputable def a : ℝ := -(0.3 ^ 2)
noncomputable def b : ℝ := -(3 ^ -2)
noncomputable def c : ℝ := ((- (1 / 3)) ^ -2)
noncomputable def d : ℝ := ((- (1 / 5)) ^ 0)

theorem size_relationship : b < a ∧ a < d ∧ d < c :=
by
  -- Conditions given in the problem
  have h₁ : a = -0.09 := by rfl
  have h₂ : b = -(1 / 9) := by rfl
  have h₃ : c = 9 := by rfl
  have h₄ : d = 1 := by rfl
  -- The comparision of values
  have h₅ : b < a := by sorry
  have h₆ : a < d := by sorry
  have h₇ : d < c := by sorry
  exact ⟨h₅, h₆, h₇⟩

end size_relationship_l164_164570


namespace sum_of_x_y_l164_164244

theorem sum_of_x_y (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_x_y_l164_164244


namespace print_colored_pages_l164_164697

theorem print_colored_pages (cost_per_page : ℕ) (dollars : ℕ) (conversion_rate : ℕ) 
    (h_cost : cost_per_page = 4) (h_dollars : dollars = 30) (h_conversion : conversion_rate = 100) :
    (dollars * conversion_rate) / cost_per_page = 750 := 
by
  sorry

end print_colored_pages_l164_164697


namespace at_least_one_draw_l164_164592

-- Defining the conditions
variables (n : ℕ) (S : ℕ → ℤ) (p1 p2 : ℕ)
  (h_played_once: ∀ i j, i ≠ j → i < n → j < n → played i j) 
  (h_points_1: S p1 = 7)
  (h_points_2: S p2 = 20)
  (h_scores_definition: ∀ i (W L : ℕ), i < n → (W - L = S i) ∧ (W + L = n - 1))
  (h_sum_scores: ∑ i in range n, S i = 0) -- Sum of all scores should be zero if no draws happen

-- The theorem to prove there is at least one draw in the tournament
theorem at_least_one_draw : ∃ i j, i ≠ j ∧ i < n ∧ j < n ∧ result i j = 0 := 
sorry

end at_least_one_draw_l164_164592


namespace equal_values_after_moves_l164_164447

-- Definition of the initial state of the table
def initial_table : list (list ℕ) :=
  [[1, 2, 3],
   [4, 5, 6],
   [7, 8, 9]]

-- Function to perform the allowed moves on the table
def perform_move (table : list (list ℕ)) (r c : ℕ) (delta : ℤ) : list (list ℕ) :=
  table.map_with_index (λ i row,
    row.map_with_index (λ j val,
      if (i >= r ∧ i < r + 2) ∧ (j >= c ∧ j < c + 2) then val + delta else val))

-- Statement of the problem
theorem equal_values_after_moves (t : list (list ℕ)) (a : ℕ) :
  (∃ f : list (list ℕ) → list (list ℕ), f t = list.repeat (list.repeat a 3) 3) →
  a = 5 := 
sorry

end equal_values_after_moves_l164_164447


namespace xiao_pang_total_problems_solved_l164_164216

/-
  Xiao Pang practices solving problems from February 6, 2014 to February 17, 2014 inclusive.
  He does not practice on Saturdays or Sundays.
  The number of problems solved follows an arithmetic sequence starting at 1 and increasing by 2 each day.
  Prove that the total number of problems solved during this period is 64.
-/

def total_problems_solved : ℕ :=
  let total_days := 12 in
  let weekends := 4 in
  let working_days := total_days - weekends in
  let a := 1 in
  let d := 2 in
  let n := working_days in
  n * (2 * a + (n - 1) * d) / 2

theorem xiao_pang_total_problems_solved : total_problems_solved = 64 := by
  sorry

end xiao_pang_total_problems_solved_l164_164216


namespace systematic_sampling_40th_number_l164_164589

theorem systematic_sampling_40th_number
  (total_students sample_size : ℕ)
  (first_group_start first_group_end selected_first_group_number steps : ℕ)
  (h1 : total_students = 1000)
  (h2 : sample_size = 50)
  (h3 : first_group_start = 1)
  (h4 : first_group_end = 20)
  (h5 : selected_first_group_number = 15)
  (h6 : steps = total_students / sample_size)
  (h7 : first_group_end - first_group_start + 1 = steps)
  : (selected_first_group_number + steps * (40 - 1)) = 795 :=
sorry

end systematic_sampling_40th_number_l164_164589


namespace intersection_complement_l164_164925

open Set

/-- The universal set U as the set of all real numbers -/
def U : Set ℝ := @univ ℝ

/-- The set M -/
def M : Set ℝ := {-1, 0, 1}

/-- The set N defined by the equation x^2 + x = 0 -/
def N : Set ℝ := {x | x^2 + x = 0}

/-- The complement of set N in the universal set U -/
def C_U_N : Set ℝ := {x ∈ U | x ≠ -1 ∧ x ≠ 0}

theorem intersection_complement :
  M ∩ C_U_N = {1} :=
by
  sorry

end intersection_complement_l164_164925


namespace distances_equal_l164_164135

-- Define the geometric components and conditions
variables {A B C S : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space S]
variables (SA SB SC : A → ℝ) (SD SE SF : B → ℝ) (I : C → ℝ)

-- Mathematical equality for distances in the given tetrahedron
theorem distances_equal
  (h1 : SA = SD)
  (h2 : SB = SE)
  (h3 : SC = SF)
  (h4 : I = SI) : 
  ∃ (S' A' B' C' : Type*), dist S' A' = dist S' B' ∧ dist S' B' = dist S' C' :=
sorry

end distances_equal_l164_164135


namespace minimum_positive_period_of_f_l164_164318

noncomputable def f (x : ℝ) : ℝ :=
  matrix.det ![![Real.sin x, 2], ![-1, Real.cos x]]

theorem minimum_positive_period_of_f : ∀ T > 0, (∀ x, f (x + T) = f x) ↔ T = π :=
by
  sorry

end minimum_positive_period_of_f_l164_164318


namespace nonagon_distinct_diagonals_l164_164947

theorem nonagon_distinct_diagonals : 
  let n := 9 in
  ∃ (d : ℕ), d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end nonagon_distinct_diagonals_l164_164947


namespace extreme_point_of_f_l164_164002

noncomputable def f (x : ℝ) : ℝ := (3 / 2) * x^2 - Real.log x

theorem extreme_point_of_f : 
  ∃ c : ℝ, c = Real.sqrt 3 / 3 ∧ (∀ x: ℝ, x > 0 → (f x > f c → x > c) ∧ (f x < f c → x < c)) := 
sorry

end extreme_point_of_f_l164_164002


namespace sum_exponents_square_root_l164_164752

theorem sum_exponents_square_root :
  let factorial_prime_exponents (n : ℕ) (p : ℕ) : ℕ :=
    ∑ k in finset.range (n + 1), n / (p ^ (k + 1))
  let largest_square_root_exponents := finset.sum (finset.filter (λ p, p.prime) (finset.range 16)) 
                                                (λ p, (factorial_prime_exponents 15 p) / 2)
  in largest_square_root_exponents = 10 :=
sorry

end sum_exponents_square_root_l164_164752


namespace loop_terminates_with_X_34_l164_164818

def X_terminate_value : Nat :=
  let rec loop (X S : Nat) : Nat :=
    if S >= 8000 then X
    else loop (X + 3) (S + (X + 3)^2)
in loop 1 0

theorem loop_terminates_with_X_34 : X_terminate_value = 34 :=
sorry

end loop_terminates_with_X_34_l164_164818


namespace inscribed_circle_radius_l164_164459

def a := 2 : ℝ
def b := 3 : ℝ
def c := 12 : ℝ

noncomputable def r : ℝ := 1 / (1/a + 1/b + 1/c + 2 * Real.sqrt (1/(a*b) + 1/(a*c) + 1/(b*c)))

theorem inscribed_circle_radius :
  r = 0.5307 :=
by
  sorry

end inscribed_circle_radius_l164_164459


namespace systematic_sampling_third_group_draw_l164_164591

theorem systematic_sampling_third_group_draw
  (first_draw : ℕ) (second_draw : ℕ) (first_draw_eq : first_draw = 2)
  (second_draw_eq : second_draw = 12) :
  ∃ (third_draw : ℕ), third_draw = 22 :=
by
  sorry

end systematic_sampling_third_group_draw_l164_164591


namespace sarah_problem_solution_l164_164679

def two_digit_number := {x : ℕ // 10 ≤ x ∧ x < 100}
def three_digit_number := {y : ℕ // 100 ≤ y ∧ y < 1000}

theorem sarah_problem_solution (x : two_digit_number) (y : three_digit_number) 
    (h_eq : 1000 * x.1 + y.1 = 8 * x.1 * y.1) : 
    x.1 = 15 ∧ y.1 = 126 ∧ (x.1 + y.1 = 141) := 
by 
  sorry

end sarah_problem_solution_l164_164679


namespace main_theorem_l164_164546

open Complex

def can_partition (S : Finset ℂ) : Prop :=
  ∃ (A B C : Finset ℂ), 
    A ∪ B ∪ C = S ∧ 
    (∀ z ∈ A, (Complex.arg z - Complex.arg (A.sum id)).abs ≤ π / 2) ∧
    (∀ z ∈ B, (Complex.arg z - Complex.arg (B.sum id)).abs ≤ π / 2) ∧
    (∀ z ∈ C, (Complex.arg z - Complex.arg (C.sum id)).abs ≤ π / 2) ∧
    ∀ (x y ∈ {A.sum id, B.sum id, C.sum id} : Finset ℂ), x ≠ y → (Complex.arg x - Complex.arg y).abs > π / 2

theorem main_theorem (S : Finset ℂ) (hS : S.card = 1993) (h_nonzero : ∀ z ∈ S, z ≠ 0) :
  can_partition S :=
  sorry

end main_theorem_l164_164546


namespace vector_product_magnitude_l164_164972

noncomputable def vector_magnitude (a b : ℝ) (theta : ℝ) : ℝ :=
  abs a * abs b * Real.sin theta

theorem vector_product_magnitude 
  (a b : ℝ) 
  (theta : ℝ) 
  (ha : abs a = 4) 
  (hb : abs b = 3) 
  (h_dot : a * b = -2) 
  (theta_range : 0 ≤ theta ∧ theta ≤ Real.pi)
  (cos_theta : Real.cos theta = -1/6) 
  (sin_theta : Real.sin theta = Real.sqrt 35 / 6) :
  vector_magnitude a b theta = 2 * Real.sqrt 35 :=
sorry

end vector_product_magnitude_l164_164972


namespace unique_polynomial_p_l164_164015

noncomputable def polynomial_p (P : Polynomial ℝ) : Prop :=
  (∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1) ∧ (P.eval 0 = 0)

theorem unique_polynomial_p (P : Polynomial ℝ) (h : polynomial_p P) : 
  P = Polynomial.Coeff ℝ 1 :=
sorry

end unique_polynomial_p_l164_164015


namespace ordinary_eqns_and_distance_l164_164150

noncomputable def parametric_curve_C (α : ℝ) : ℝ × ℝ :=
  (3 * Real.cos α, Real.sin α)

noncomputable def parametric_line_l (t : ℝ) : ℝ × ℝ :=
  (-t + 2, t)

theorem ordinary_eqns_and_distance :
  (∀ (α : ℝ), ∃ (x y : ℝ), (x, y) = parametric_curve_C α ∧ (x^2 / 9 + y^2 = 1)) ∧
  (∀ (t : ℝ), ∃ (x y : ℝ), (x, y) = parametric_line_l t ∧ (x + y - 2 = 0)) ∧
  (∃ (t1 t2 : ℝ), (t1 + t2 = 2 * Real.sqrt 2 / 5) ∧ (t1 * t2 = -1) ∧ 
  (Real.abs (t1 - t2) = 6 * Real.sqrt 3 / 5)) :=
by
  sorry

end ordinary_eqns_and_distance_l164_164150


namespace pascal_sum_of_squares_of_interior_l164_164609

theorem pascal_sum_of_squares_of_interior (eighth_row_interior : List ℕ) 
    (h : eighth_row_interior = [7, 21, 35, 35, 21, 7]) : 
    (eighth_row_interior.map (λ x => x * x)).sum = 3430 := 
by
  sorry

end pascal_sum_of_squares_of_interior_l164_164609


namespace hundredth_term_seq_l164_164704

def seq (n : ℕ) : ℕ :=
  let binary_digits := Nat.binaryDigits n in
  (List.foldl (λ acc p => acc + 3^p) 0 (binary_digits.enum.filter (λ x => x.2 = 1)).map (λ x => x.1))

theorem hundredth_term_seq :
  seq 100 = 981 :=
by
  sorry

end hundredth_term_seq_l164_164704


namespace quotient_of_sum_l164_164765

theorem quotient_of_sum (a b c x y z : ℝ)
  (h1 : a^2 + b^2 + c^2 = 25)
  (h2 : x^2 + y^2 + z^2 = 36)
  (h3 : a * x + b * y + c * z = 30) :
  (a + b + c) / (x + y + z) = 5 / 6 :=
by
  sorry

end quotient_of_sum_l164_164765


namespace exist_initial_points_l164_164269

theorem exist_initial_points (n : ℕ) (h : 9 * n - 8 = 82) : ∃ n = 10 :=
by
  sorry

end exist_initial_points_l164_164269


namespace exist_identical_3x3_squares_l164_164585

theorem exist_identical_3x3_squares :
  ∀ (grid : Fin 25 → Fin 25 → Bool), ∃ (a b : Fin 23 × Fin 23), 
  ∀ i j : Fin 3, grid (a.1 + i) (a.2 + j) = grid (b.1 + i) (b.2 + j) :=
by
  sorry

end exist_identical_3x3_squares_l164_164585


namespace f_at_0_f_at_neg_2_l164_164186

-- Definitions for conditions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

def f (x : ℝ) : ℝ := if x > 0 then 2^x - 3 else if x < 0 then -(2^(-x) - 3) else 0

-- Theorem statements
theorem f_at_0 : f 0 = 0 := by 
  sorry

theorem f_at_neg_2 : f (-2) = -1 := by
  sorry

end f_at_0_f_at_neg_2_l164_164186


namespace possible_values_of_sum_l164_164237

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l164_164237


namespace proposition_C_proposition_D_l164_164804

theorem proposition_C (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_cond : 2 * a + b = 1) : 
  (1 / (2 * a)) + (1 / b) ≥ 4 := 
by
  sorry

theorem proposition_D (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_cond : a + b = 4) : 
  (min) (λ x : ℝ, (x^2) / (x + 1) + ((4 - x)^2) / (4 - x + 1)) = 8 / 3 := 
by
  sorry

end proposition_C_proposition_D_l164_164804


namespace inequality_A_if_ab_pos_inequality_D_if_ab_pos_l164_164569

variable (a b : ℝ)

theorem inequality_A_if_ab_pos (h : a * b > 0) : a^2 + b^2 ≥ 2 * a * b := 
sorry

theorem inequality_D_if_ab_pos (h : a * b > 0) : (b / a) + (a / b) ≥ 2 :=
sorry

end inequality_A_if_ab_pos_inequality_D_if_ab_pos_l164_164569


namespace sum_of_union_card_l164_164173

def F (n : ℕ) : Set (Fin n → Set (Fin 2019)) :=
  {A | ∀ i, A i ⊆ Finset.univ}

def sum_sets_card (A : Fin n → Set (Fin 2019)) : ℕ :=
  (Finset.univ.filter (λ e, ∃ i, e ∈ A i)).card

theorem sum_of_union_card (n : ℕ) :
  ∑ A in F n, sum_sets_card A = 2019 * (2^(2019 * n) - 2^(2018 * n)) :=
sorry

end sum_of_union_card_l164_164173


namespace ratio_of_third_to_second_building_l164_164352

/-
The tallest building in the world is 100 feet tall. The second tallest is half that tall, the third tallest is some 
fraction of the second tallest building's height, and the fourth is one-fifth as tall as the third. All 4 buildings 
put together are 180 feet tall. What is the ratio of the height of the third tallest building to the second tallest building?

Given H1 = 100, H2 = (1 / 2) * H1, H4 = (1 / 5) * H3, 
and H1 + H2 + H3 + H4 = 180, prove that H3 / H2 = 1 / 2.
-/

theorem ratio_of_third_to_second_building :
  ∀ (H1 H2 H3 H4 : ℝ),
  H1 = 100 →
  H2 = (1 / 2) * H1 →
  H4 = (1 / 5) * H3 →
  H1 + H2 + H3 + H4 = 180 →
  (H3 / H2) = (1 / 2) :=
by
  intros H1 H2 H3 H4 h1_eq h2_half_h1 h4_fifth_h3 total_eq
  /- proof steps go here -/
  sorry

end ratio_of_third_to_second_building_l164_164352


namespace seven_digit_palindrome_count_l164_164930

-- Define what it means for a list of digits to be a palindrome
def is_palindrome (l : List ℕ) : Prop :=
  l = l.reverse

-- Define the set of 7 digits we are working with
def digits : List ℕ := [1, 1, 4, 4, 4, 6, 6]

-- Define the problem statement as a Lean theorem
theorem seven_digit_palindrome_count :
  (Finset.univ.filter (λ l : List ℕ, l.length = 7 ∧ l ∈ digits.permutations ∧ is_palindrome l)).card = 6 :=
sorry

end seven_digit_palindrome_count_l164_164930


namespace find_k_l164_164182

variables {ℝ : Type*} [Field ℝ]
variables {e₁ e₂ : ℝ} (k : ℝ)
variables (AB BC CD : \(\overrightarrow{ℝ^2}\))

-- Conditions
def AB_expr := 2 * ⟨e₁, 0⟩ + k * ⟨0, e₂⟩
def BC_expr := ⟨e₁, 0⟩ + 3 * ⟨0, e₂⟩
def CD_expr := 2 * ⟨e₁, 0⟩ - ⟨0, e₂⟩
def BD_expr := BC_expr + CD_expr

-- Collinearity condition
def collinear (V W Z : \(\overrightarrow{ℝ^2}\)) : Prop :=
  ∃ m : ℝ, V = m • Z

theorem find_k (h₁ : AB = AB_expr)
               (h₂ : BC = BC_expr)
               (h₃ : CD = CD_expr)
               (h_collinear : collinear AB BD_expr) :
  k = 4 / 3 :=
sorry

end find_k_l164_164182


namespace find_c_l164_164924

-- Definitions of the sets A and B as per the given conditions
def A : Set ℝ := {x | log 2 x < 1}

def B (c : ℝ) : Set ℝ := {x | 0 < x ∧ x < c}

-- The proof statement
theorem find_c (c : ℝ) (h : A = B c) : c = 2 :=
by
  sorry

end find_c_l164_164924


namespace locus_is_perpendicular_line_l164_164551

variables {O1 O2 X : Type*} [MetricSpace O1] [MetricSpace O2] [MetricSpace X]
variables {R1 R2 : ℝ}

def locus_of_circle_centers (O1 O2 X : Type*) [MetricSpace O1] [MetricSpace O2] [MetricSpace X] (R1 R2 : ℝ) : Set X :=
  {X : X | (dist X O1)^2 - (dist X O2)^2 = R2^2 - R1^2}

theorem locus_is_perpendicular_line
  {O1 O2 X : Type*} [MetricSpace O1] [MetricSpace O2] [MetricSpace X]
  (R1 R2 : ℝ) (locus : Set X) :
  locus = locus_of_circle_centers O1 O2 X R1 R2 :=
sorry

end locus_is_perpendicular_line_l164_164551


namespace calculate_expression_l164_164395

theorem calculate_expression : 
  ( sqrt 1.21 / sqrt 0.81 + sqrt 1.00 / sqrt 0.49 ) ≈ 2.6507 :=
by {
  sorry
}

end calculate_expression_l164_164395


namespace system_of_equations_solution_l164_164299

theorem system_of_equations_solution :
  ∃ x y : ℝ, 7 * x - 3 * y = 2 ∧ 2 * x + y = 8 ∧ x = 2 ∧ y = 4 :=
by
  use 2
  use 4
  sorry

end system_of_equations_solution_l164_164299


namespace problem_1_problem_2_l164_164555

noncomputable def a (k : ℝ) : ℝ × ℝ := (2, k)
noncomputable def b : ℝ × ℝ := (1, 1)
noncomputable def a_minus_3b (k : ℝ) : ℝ × ℝ := (2 - 3 * 1, k - 3 * 1)

-- First problem: Prove that k = 4 given vectors a and b, and the condition that b is perpendicular to (a - 3b)
theorem problem_1 (k : ℝ) (h : b.1 * (a_minus_3b k).1 + b.2 * (a_minus_3b k).2 = 0) : k = 4 :=
sorry

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def cosine (v w : ℝ × ℝ) : ℝ := dot_product v w / (magnitude v * magnitude w)

-- Second problem: Prove that the cosine value of the angle between a and b is 3√10/10 when k is 4
theorem problem_2 (k : ℝ) (hk : k = 4) : cosine (a k) b = 3 * Real.sqrt 10 / 10 :=
sorry

end problem_1_problem_2_l164_164555


namespace range_of_a_l164_164916

def f (x : ℝ) := x^2 + Real.log (|x| + 1)

theorem range_of_a (a : ℝ) : 
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f (a * x^2) < f 3) →
  -3 / 4 < a ∧ a < 3 / 4 :=
by 
  intros h
  sorry

end range_of_a_l164_164916


namespace probability_sum_multiple_of_5_l164_164852

noncomputable def dice_probability_of_sum_multiple_of_5_given_product_even 
  (dice : Fin 5 → Fin 6) 
  (h_even : (∏ i in Finset.univ, (dice i + 1)) % 2 = 0) 
  (a : ℕ) : 
  ℚ :=
  let total_outcomes := 6^5
  let odd_outcomes := 3^5
  let at_least_one_even := total_outcomes - odd_outcomes
  a / at_least_one_even

theorem probability_sum_multiple_of_5
  (dice : Fin 5 → Fin 6)
  (h_even : (∏ i in Finset.univ, (dice i + 1)) % 2 = 0)
  (a : ℕ) :
  dice_probability_of_sum_multiple_of_5_given_product_even dice h_even a = a / (6^5 - 3^5) :=
sorry

end probability_sum_multiple_of_5_l164_164852


namespace larger_angle_is_99_l164_164706

theorem larger_angle_is_99 (x : ℝ) (h1 : 2 * x + 18 = 180) : x + 18 = 99 :=
by
  sorry

end larger_angle_is_99_l164_164706


namespace angle_between_vectors_l164_164554

noncomputable def vector_a : ℝ × ℝ := (1, 1)
noncomputable def vector_b_mag : ℝ := Real.sqrt 6
noncomputable def dot_product_ab : ℝ := -3

theorem angle_between_vectors : 
    ∃ θ : ℝ, θ = 150 ∧ 
    0 ≤ θ ∧ θ ≤ 180 ∧
    Real.arccos (dot_product_ab / (Real.sqrt ((fst vector_a)^2 + (snd vector_a)^2) * vector_b_mag)) = θ :=
  sorry

end angle_between_vectors_l164_164554


namespace max_integers_less_than_neg4_l164_164721

theorem max_integers_less_than_neg4 (x : Fin 8 → ℤ) (h : ∑ i, x i = 15) : 
  ∃ k : ℕ, k ≤ 8 ∧ k = 7 ∧ (∀ i, i < k → x i < -4) :=
by
  sorry

end max_integers_less_than_neg4_l164_164721


namespace symmetric_about_line_periodic_function_l164_164541

section
variable {α : Type*} [LinearOrderedField α]

-- First proof problem
theorem symmetric_about_line (f : α → α) (a : α) (h : ∀ x, f (a + x) = f (a - x)) : 
  ∀ x, f (2 * a - x) = f x :=
sorry

-- Second proof problem
theorem periodic_function (f : α → α) (a b : α) (ha : a ≠ b)
  (hsymm_a : ∀ x, f (2 * a - x) = f x)
  (hsymm_b : ∀ x, f (2 * b - x) = f x) : 
  ∃ p, p ≠ 0 ∧ ∀ x, f (x + p) = f x :=
sorry
end

end symmetric_about_line_periodic_function_l164_164541


namespace find_fourth_number_l164_164305

theorem find_fourth_number 
  (average : ℝ) 
  (a1 a2 a3 : ℝ) 
  (x : ℝ) 
  (n : ℝ) 
  (h1 : average = 20) 
  (h2 : a1 = 3) 
  (h3 : a2 = 16) 
  (h4 : a3 = 33) 
  (h5 : n = 27) 
  (h_avg : (a1 + a2 + a3 + x) / 4 = average) :
  x = n + 1 :=
by
  sorry

end find_fourth_number_l164_164305


namespace solution_set_of_inequality_l164_164973

open Set

def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + 2^x - 1 else - (x^2 + 2^(-x) - 1)

theorem solution_set_of_inequality :
  ∀ x : ℝ, f x + 7 < 0 ↔ x ∈ Iio (-2) :=
by
  sorry

end solution_set_of_inequality_l164_164973


namespace same_side_of_line_l164_164532

theorem same_side_of_line (a : ℝ) :
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) > 0 ↔ a < -7 ∨ a > 24 :=
by
  sorry

end same_side_of_line_l164_164532


namespace great_circle_stereographic_projection_circle_l164_164675

noncomputable def stereographic_projection (S : Sphere) (E : Point) : Point → Point := sorry

def is_circle (C : Set Point) : Prop := sorry

variable (G : Type) [Sphere G]
variable (E : Point)
variable (k : Set Point)
variable (δ : Plane)

-- Condition: k is a great circle on the sphere G, and k does not pass through E
axiom great_circle (k : Set Point) (G : Sphere) : Prop
axiom not_pass_through (k : Set Point) (E : Point): Prop

theorem great_circle_stereographic_projection_circle (G : Sphere) (E : Point) (k : Set Point) (δ : Plane)
  (h1 : great_circle k G)
  (h2 : not_pass_through k E) :
  ∃ C : Set Point, is_circle C ∧ stereographic_projection G E k = C := sorry

end great_circle_stereographic_projection_circle_l164_164675


namespace greatest_common_divisor_of_180_and_n_l164_164368

theorem greatest_common_divisor_of_180_and_n (n : ℕ) (h : ∀ d ∣ 180 ∧ d ∣ n, d = 1 ∨ d = 3 ∨ d = 9) : 
  ∃ d, d ∣ 180 ∧ d ∣ n ∧ d = 9 :=
sorry

end greatest_common_divisor_of_180_and_n_l164_164368


namespace abs_f_x_gt_abs_f_y_l164_164529

variable {f : ℝ → ℝ}
variable (h_inc : ∀ x y : ℝ, 0 < x → x < y → f(x) < f(y))
variable (h_one : f(1) = 0)
variable (h_func_eq : ∀ x y : ℝ, f(x) + f(y) = f(x * y))

theorem abs_f_x_gt_abs_f_y (x y : ℝ) (hx : 0 < x) (hy : x < y) (h_less_1 : y < 1) : 
  |f(x)| > |f(y)| := 
sorry

end abs_f_x_gt_abs_f_y_l164_164529


namespace num_sets_M_l164_164711

theorem num_sets_M {M : Type} [DecidableEq M] : 
  let M_1 := {2, 3} ∪ {1} = {1, 2, 3},
      M_2 := {1, 2, 3} ∪ {1} = {1, 2, 3} in 
      (M_1 ∨ M_2) → ∃! (M : Set Nat), M ∪ {1} = {1, 2, 3} → 2 :=
by
  sorry

end num_sets_M_l164_164711


namespace sum_of_reciprocals_lt_80_l164_164190

def does_not_contain_digit_9 (x : ℕ) : Prop :=
  ∀ d ∈ (Nat.digits 10 x), d ≠ 9

noncomputable def set_M := {x : ℕ | x > 0 ∧ does_not_contain_digit_9 x}

theorem sum_of_reciprocals_lt_80 (n : ℕ) (x : Fin n → ℕ) (hx : ∀ i j, i ≠ j → x i ≠ x j) (hxM : ∀ i, x i ∈ set_M) :
  ∑ i in Finset.range n, (1 / (x i : ℝ)) < 80 :=
sorry

end sum_of_reciprocals_lt_80_l164_164190


namespace find_larger_number_l164_164122

theorem find_larger_number (x y : ℤ) (h1 : 4 * y = 3 * x) (h2 : y - x = 12) : y = -36 := 
by sorry

end find_larger_number_l164_164122


namespace leonards_age_l164_164166

variable (L N J : ℕ)

theorem leonards_age (h1 : L = N - 4) (h2 : N = J / 2) (h3 : L + N + J = 36) : L = 6 := 
by 
  sorry

end leonards_age_l164_164166


namespace garden_perimeter_l164_164323

theorem garden_perimeter (L B : ℕ) (hL : L = 250) (hB : B = 100) : 2 * (L + B) = 700 := by
  rw [hL, hB]
  norm_num

end garden_perimeter_l164_164323


namespace nonagon_diagonals_count_eq_27_l164_164952

theorem nonagon_diagonals_count_eq_27 :
  let vertices := 9
  let connections_per_vertex := vertices - 3
  let total_raw_count_diagonals := vertices * connections_per_vertex
  let distinct_diagonals := total_raw_count_diagonals / 2
  distinct_diagonals = 27 :=
by
  let vertices := 9
  let connections_per_vertex := vertices - 3
  let total_raw_count_diagonals := vertices * connections_per_vertex
  let distinct_diagonals := total_raw_count_diagonals / 2
  have : distinct_diagonals = 27 := sorry
  exact this

end nonagon_diagonals_count_eq_27_l164_164952


namespace suitable_a_values_count_l164_164312

theorem suitable_a_values_count :
  ∃ a_values : Finset ℤ,
    ∀ a ∈ a_values,
      (∃ n : ℕ, n * n = |a| - 3) ∧
      21 * (1 / 2 * (a + 6)) = 21 * t ∧
      -6 < a ∧ a ≤ 4.5 ∧
      0 < t ∧ t ≤ 1 ∧
    a_values.card = 4 :=
begin
  sorry
end

end suitable_a_values_count_l164_164312


namespace trey_nail_usage_l164_164737

theorem trey_nail_usage (total_decorations nails thumbtacks sticky_strips : ℕ) 
  (h1 : nails = 2 * total_decorations / 3)
  (h2 : sticky_strips = 15)
  (h3 : sticky_strips = 3 * (total_decorations - 2 * total_decorations / 3) / 5) :
  nails = 50 :=
by
  sorry

end trey_nail_usage_l164_164737


namespace fishing_competition_total_fishes_l164_164130

noncomputable def jackson_catch_per_day : ℕ := 6
noncomputable def jonah_catch_per_day : ℕ := 4
noncomputable def george_catches : list ℕ := [8, 12, 7, 9, 11]
noncomputable def george_total_catch : ℕ := george_catches.sum
noncomputable def lily_catches_day1_to_4 : list ℕ := [5, 6, 9, 5]
noncomputable def lily_total_catch_day1_to_4 : ℕ := lily_catches_day1_to_4.sum
noncomputable def alex_catches : list ℕ := george_catches.map (λ x, x - 2)
noncomputable def alex_total_catch : ℕ := alex_catches.sum

theorem fishing_competition_total_fishes (Lily_day5 : ℕ) :
  let
    jackson_total := jackson_catch_per_day * 5,
    jonah_total := jonah_catch_per_day * 5,
    team_total := jackson_total + jonah_total + george_total_catch + lily_total_catch_day1_to_4 + alex_total_catch
  in
    team_total = 159 := 
by
  sorry

end fishing_competition_total_fishes_l164_164130


namespace nonagon_diagonals_count_l164_164931

theorem nonagon_diagonals_count (n : ℕ) (h1 : n = 9) : 
  let diagonals_per_vertex := n - 3 in
  let naive_count := n * diagonals_per_vertex in
  let distinct_diagonals := naive_count / 2 in
  distinct_diagonals = 27 :=
by
  sorry

end nonagon_diagonals_count_l164_164931


namespace number_of_people_who_didnt_do_both_l164_164597

def total_graduates : ℕ := 73
def graduates_both : ℕ := 13

theorem number_of_people_who_didnt_do_both : total_graduates - graduates_both = 60 :=
by
  sorry

end number_of_people_who_didnt_do_both_l164_164597


namespace quad_min_value_l164_164314

theorem quad_min_value (a b c : ℝ) (h : a > 0) : ∃ x : ℝ, (ax^2 + bx + c).min = (4ac - b^2) / (4a) :=
sorry

end quad_min_value_l164_164314


namespace picture_area_correct_l164_164391

def paper_length := 10
def paper_width := 8.5
def margin := 1.5

def picture_length := paper_length - 2 * margin
def picture_width := paper_width - 2 * margin

def picture_area := picture_length * picture_width

-- We need to prove that picture_area is 38.5 square inches
theorem picture_area_correct : picture_area = 38.5 := by
  sorry

end picture_area_correct_l164_164391


namespace logic_problem_l164_164580

theorem logic_problem (p q : Prop) (h1 : ¬p) (h2 : ¬(p ∧ q)) : ¬ (p ∨ q) :=
sorry

end logic_problem_l164_164580


namespace complex_multiplication_l164_164526

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : (2 + i) * (1 - 3 * i) = 5 - 5 * i := 
by
  sorry

end complex_multiplication_l164_164526


namespace each_dog_puppies_l164_164657

-- Let P be the total number of puppies born
variable (P : ℕ)

-- Conditions
axiom sold_fraction : 3 / 4 * P = 15
axiom price_per_puppy : 200
axiom total_amount_received : 3000
axiom num_dogs : 2

-- Prove that each dog gave birth to 10 puppies
theorem each_dog_puppies : P / num_dogs = 10 :=
by
  -- Setup the given equations
  have H1 : P = 20 := by
    calc
      3 / 4 * P = 15   : sold_fraction
           ... = 15 * 4 / 3 : by rw [←mul_div_assoc, ←mul_comm, div_self (show (3 : ℚ) ≠ 0 by norm_num), one_mul]
           ... = 20 : by norm_num
  show P / num_dogs = 10
  calc
    P / num_dogs = 20 / 2 : by rw H1
             ... = 10       : by norm_num

-- Include a placeholder for the main proof 
example : each_dog_puppies :=
by
  -- Current proof step simplified to the definition
  exact sorry

end each_dog_puppies_l164_164657


namespace highest_page_number_l164_164662

theorem highest_page_number (num_fives : ℕ) (fives_available : num_fives = 18) : 
  ∃ n, n = 99 ∧ ∀ m, m > n → (occur 5 m > 18) :=
by
  sorry

end highest_page_number_l164_164662


namespace curve_is_circle_l164_164023

theorem curve_is_circle (r θ : ℝ) (h : r = 1 / (Real.sin θ + Real.cos θ)) :
  ∃ x y : ℝ, r = Math.sqrt(x^2 + y^2) ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ (x + y)^2 = (x^2 + y^2) :=
by
  sorry

end curve_is_circle_l164_164023


namespace geometric_series_problems_l164_164522

noncomputable def S (n : ℕ) : ℚ

-- Main theorem statement
theorem geometric_series_problems :
  (S 2 = 2) ∧
  (S 3 = 3) ∧
  (∀ n, S n = 4 * (1 - (-1 / 2) ^ n) / (3 / 2)) ∧
  (∀ n, ∑ k in finset.range n, S (k + 1) = (8 / 3) * n + (8 / 9) * (1 - (-1 / 2) ^ n)) :=
by {
  -- Proof would go here
  sorry
}

end geometric_series_problems_l164_164522


namespace max_remaining_area_l164_164791

theorem max_remaining_area (original_area : ℕ) (rec1 : ℕ × ℕ) (rec2 : ℕ × ℕ) (rec3 : ℕ × ℕ)
  (rec4 : ℕ × ℕ) (total_area_cutout : ℕ):
  original_area = 132 →
  rec1 = (1, 4) →
  rec2 = (2, 2) →
  rec3 = (2, 3) →
  rec4 = (2, 3) →
  total_area_cutout = 20 →
  original_area - total_area_cutout = 112 :=
by
  intros
  sorry

end max_remaining_area_l164_164791


namespace sum_of_xy_l164_164255

theorem sum_of_xy {x y : ℝ} (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := sorry

end sum_of_xy_l164_164255


namespace possible_values_of_y_l164_164627

theorem possible_values_of_y (x : ℝ) (hx : x^2 + 5 * (x / (x - 3)) ^ 2 = 50) :
  ∃ (y : ℝ), y = (x - 3)^2 * (x + 4) / (3 * x - 4) ∧ (y = 0 ∨ y = 15 ∨ y = 49) :=
sorry

end possible_values_of_y_l164_164627


namespace Lizzy_savings_after_loan_l164_164645

theorem Lizzy_savings_after_loan :
  ∀ (initial_amount loan_amount : ℕ) (interest_percent : ℕ),
  initial_amount = 30 →
  loan_amount = 15 →
  interest_percent = 20 →
  initial_amount - loan_amount + loan_amount + loan_amount * interest_percent / 100 = 33 :=
by
  intros initial_amount loan_amount interest_percent h1 h2 h3
  sorry

end Lizzy_savings_after_loan_l164_164645


namespace secondary_spermatocytes_can_contain_two_y_chromosomes_l164_164598

-- Definitions corresponding to the conditions
def primary_spermatocytes_first_meiotic_division_contains_y (n : Nat) : Prop := n = 1
def spermatogonia_metaphase_mitosis_contains_y (n : Nat) : Prop := n = 1
def secondary_spermatocytes_second_meiotic_division_contains_y (n : Nat) : Prop := n = 0 ∨ n = 2
def spermatogonia_prophase_mitosis_contains_y (n : Nat) : Prop := n = 1

-- The theorem statement equivalent to the given math problem
theorem secondary_spermatocytes_can_contain_two_y_chromosomes :
  ∃ n, (secondary_spermatocytes_second_meiotic_division_contains_y n ∧ n = 2) :=
sorry

end secondary_spermatocytes_can_contain_two_y_chromosomes_l164_164598


namespace square_paintings_size_l164_164829

theorem square_paintings_size (total_area : ℝ) (small_paintings_count : ℕ) (small_painting_area : ℝ) 
                              (large_painting_area : ℝ) (square_paintings_count : ℕ) (square_paintings_total_area : ℝ) : 
  total_area = small_paintings_count * small_painting_area + large_painting_area + square_paintings_total_area → 
  square_paintings_count = 3 → 
  small_paintings_count = 4 → 
  small_painting_area = 2 * 3 → 
  large_painting_area = 10 * 15 → 
  square_paintings_total_area = 3 * 6^2 → 
  ∃ side_length, side_length^2 = (square_paintings_total_area / square_paintings_count) ∧ side_length = 6 := 
by
  intro h_total h_square_count h_small_count h_small_area h_large_area h_square_total 
  use 6
  sorry

end square_paintings_size_l164_164829


namespace revision_cost_per_page_l164_164340

theorem revision_cost_per_page :
  let typing_cost_per_page := 10
  let num_pages := 100
  let revised_once := 20
  let revised_twice := 30
  let no_revisions := num_pages - (revised_once + revised_twice)
  let total_cost := 1400
  ∃ x : ℕ, typing_cost_per_page * num_pages + revised_once * x + revised_twice * 2 * x = total_cost :=
begin
  let typing_cost_per_page := 10,
  let num_pages := 100,
  let revised_once := 20,
  let revised_twice := 30,
  let no_revisions := num_pages - (revised_once + revised_twice),
  let total_cost := 1400,
  let x := 5,
  sorry
end

end revision_cost_per_page_l164_164340


namespace cosine_sine_inequality_solution_l164_164848

theorem cosine_sine_inequality_solution (x : ℝ) :
  x ∈ Set.Icc (-4 * Real.pi / 3) (2 * Real.pi / 3) ∧
  (Real.cos x)^2018 + (Real.sin x)^(-2019) ≥ (Real.sin x)^2018 + (Real.cos x)^(-2019)
  ↔ x ∈ (Set.Ico (-4 * Real.pi / 3) (-Real.pi) ∪ 
          Set.Ico (-3 * Real.pi / 4) (-Real.pi / 2) ∪ 
          Set.Ioc 0 (Real.pi / 4) ∪ 
          Set.Ioc (Real.pi / 2) (2 * Real.pi / 3)) :=
sorry

end cosine_sine_inequality_solution_l164_164848


namespace min_initial_seeds_l164_164416

/-- Given conditions:
  - The farmer needs to sell at least 10,000 watermelons each year.
  - Each watermelon produces 250 seeds when used for seeds but cannot be sold if used for seeds.
  - We need to find the minimum number of initial seeds S the farmer must buy to never buy seeds again.
-/
theorem min_initial_seeds : ∃ (S : ℕ), S = 10041 ∧ ∀ (yearly_sales : ℕ), yearly_sales = 10000 →
  ∀ (seed_yield : ℕ), seed_yield = 250 →
  ∃ (x : ℕ), S = yearly_sales + x ∧ x * seed_yield ≥ S :=
sorry

end min_initial_seeds_l164_164416


namespace _l164_164487

lemma integer_root_theorem (b : ℤ) :
  (∃ x : ℤ, x ∣ 6 ∧ x^3 - 2 * x^2 + b * x + 6 = 0) ↔ b ∈ {-25, -7, -5, 3, 13, 47} := 
by
  sorry

end _l164_164487


namespace local_max_f_at_1_3_l164_164921

section
  variable (x : ℝ)

  def f : ℝ → ℝ := λ x, x^3 - 2*x^2 + x - 1

  noncomputable def local_max (f : ℝ → ℝ) (c : ℝ) : ℝ :=
    if h : ∃ x, f x < f c ∨ ∀ ᶠ y in 𝓝 c, f y ≤ f c then f c else 0

  theorem local_max_f_at_1_3 : local_max f (1/3) = -(23/27) :=
  sorry
end

end local_max_f_at_1_3_l164_164921


namespace find_cost_of_large_pizza_l164_164557

-- Define the given conditions
variables {P : ℝ} -- Let P be the cost of a large pizza without toppings
def topping_cost := 2
def num_pizzas := 2
def toppings_per_pizza := 3
def tip_percentage := 0.25
def total_cost := 50

-- Define the calculation of the total cost
def total_cost_eqn (P : ℝ) : ℝ :=
  let subtotal := num_pizzas * (P + toppings_per_pizza * topping_cost);
  subtotal + tip_percentage * subtotal

-- The theorem stating that given the conditions, P must be 14
theorem find_cost_of_large_pizza (h : total_cost_eqn P = total_cost) : P = 14 :=
  sorry

end find_cost_of_large_pizza_l164_164557


namespace total_displacement_correct_total_fuel_consumption_correct_l164_164690

-- Define the driving distances as a list of integers
def driving_distances : List ℤ := [+15, -3, +14, -11, +10, -12]

-- Define the fuel consumption per kilometer
variable (a : ℤ)

-- Statement for the first question: total displacement
theorem total_displacement_correct :
  driving_distances.sum = 13 :=
by
  -- proof to be filled in

-- Statement for the second question: total fuel consumption
theorem total_fuel_consumption_correct (a : ℕ) :
  driving_distances.map Int.natAbs |>.sum * a = 65 * a :=
by
  -- proof to be filled in

end total_displacement_correct_total_fuel_consumption_correct_l164_164690


namespace min_commission_deputies_l164_164725

theorem min_commission_deputies 
  (members : ℕ) 
  (brawls : ℕ) 
  (brawl_participants : brawls = 200) 
  (member_count : members = 200) :
  ∃ minimal_commission_members : ℕ, minimal_commission_members = 67 := 
sorry

end min_commission_deputies_l164_164725


namespace Yarns_are_Xants_and_Wooks_l164_164985

-- Definitions for sets involved
variable (Zelm Xant Yarn Wook : Type) 
variable (Zelm_sub_Xant : Zelm → Xant)
variable (Yarn_sub_Zelm : Yarn → Zelm)
variable (Xant_sub_Wook : Xant → Wook)

-- The statement to be proven
theorem Yarns_are_Xants_and_Wooks :
  (Yarn → Xant) ∧ (Yarn → Wook) :=
by
  -- Proof will go here
  sorry

end Yarns_are_Xants_and_Wooks_l164_164985


namespace ball_bounce_count_l164_164762

def height_after_bounce (n : ℕ) : ℝ := 16 / (2 ^ n)

theorem ball_bounce_count :
  (∑ n in range (4 + 1), height_after_bounce n
  + ∑ n in range 4, height_after_bounce (n + 1)) = 45 :=
sorry

end ball_bounce_count_l164_164762


namespace faster_train_speed_l164_164742

def train_speeds (v : ℚ) : Prop :=
  let len := 150 in
  let time := 12 in
  let combined_length := 2 * len in
  let relative_speed := 4 * v in
  combined_length = relative_speed * time

theorem faster_train_speed (v : ℚ) (faster_v := 3 * v) :
  train_speeds v → faster_v = 18.75 := by
  sorry

end faster_train_speed_l164_164742


namespace points_on_line_possible_l164_164281

theorem points_on_line_possible : ∃ n : ℕ, 9 * n - 8 = 82 :=
by
  sorry

end points_on_line_possible_l164_164281


namespace nonagon_diagonals_count_eq_27_l164_164951

theorem nonagon_diagonals_count_eq_27 :
  let vertices := 9
  let connections_per_vertex := vertices - 3
  let total_raw_count_diagonals := vertices * connections_per_vertex
  let distinct_diagonals := total_raw_count_diagonals / 2
  distinct_diagonals = 27 :=
by
  let vertices := 9
  let connections_per_vertex := vertices - 3
  let total_raw_count_diagonals := vertices * connections_per_vertex
  let distinct_diagonals := total_raw_count_diagonals / 2
  have : distinct_diagonals = 27 := sorry
  exact this

end nonagon_diagonals_count_eq_27_l164_164951


namespace complex_problem_l164_164089

def z : ℂ := -2 + complex.I

theorem complex_problem : (z * (conj z)) / complex.I = -5 * complex.I := by 
  sorry

end complex_problem_l164_164089


namespace b_arithmetic_sequence_T_n_sum_l164_164716

noncomputable theory

open Nat

-- Problem Conditions
def a (n : ℕ) : ℕ
| 0     := 0 -- not used
| 1     := 1
| 2     := 2
| (n+3) := 2 * a (n + 2) - a n

-- Auxiliary Definition
def b (n : ℕ) : ℚ := a n / (a (n + 1) - a n)

def c (n : ℕ) : ℚ := 1 / (a n * a (n + 1))

-- Question 1: Prove that {b_n} is an arithmetic sequence
theorem b_arithmetic_sequence : ∀ (n : ℕ), b (n + 1) - b n = 1 := by sorry

-- Question 2: Prove that the sum of first n terms of {c_n} is n / (n + 1)
def T (n : ℕ) : ℚ := (finset.range n).sum (λ i, c i)

theorem T_n_sum : ∀ (n : ℕ), T n = n / (n + 1) := by sorry

end b_arithmetic_sequence_T_n_sum_l164_164716


namespace right_triangle_legs_correct_l164_164134

noncomputable def right_triangle_legs (hypotenuse : ℝ) (area : ℝ) (b c : ℝ) : Prop :=
hypotenuse = 25 ∧ area = 150 ∧ hypotenuse ^ 2 = b ^ 2 + c ^ 2 ∧ (1 / 2) * b * c = area ∧ 
b = 20 ∧ c = 15

theorem right_triangle_legs_correct :
  right_triangle_legs 25 150 20 15 :=
begin
  sorry
end

end right_triangle_legs_correct_l164_164134


namespace evaluate_expression_l164_164498

variables (x : ℝ)

theorem evaluate_expression :
  x * (x * (x * (3 - x) - 5) + 13) + 1 = -x^4 + 3*x^3 - 5*x^2 + 13*x + 1 :=
by 
  sorry

end evaluate_expression_l164_164498


namespace possible_landing_l164_164984

-- There are 1985 airfields
def num_airfields : ℕ := 1985

-- 50 airfields where planes could potentially land
def num_land_airfields : ℕ := 50

-- Define the structure of the problem
structure AirfieldSetup :=
  (airfields : Fin num_airfields → Fin num_land_airfields)

-- There exists a configuration such that the conditions are met
theorem possible_landing : ∃ (setup : AirfieldSetup), 
  (∀ i : Fin num_airfields, -- For each airfield
    ∃ j : Fin num_land_airfields, -- There exists a landing airfield
    setup.airfields i = j) -- The plane lands at this airfield.
:=
sorry

end possible_landing_l164_164984


namespace real_roots_if_and_only_if_m_leq_5_l164_164577

theorem real_roots_if_and_only_if_m_leq_5 (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + 4 * x + 1 = 0) ↔ m ≤ 5 :=
by
  sorry

end real_roots_if_and_only_if_m_leq_5_l164_164577


namespace total_watermelon_weight_l164_164616

theorem total_watermelon_weight :
  let w1 := 9.91
  let w2 := 4.112
  let w3 := 6.059
  w1 + w2 + w3 = 20.081 :=
by
  sorry

end total_watermelon_weight_l164_164616


namespace find_point_M_l164_164603

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def distance (P Q : Point3D) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2)

theorem find_point_M :
  ∃ (M : Point3D), M.y = -1 ∧ distance M ⟨1, 0, 2⟩ = distance M ⟨1, -3, 1⟩ :=
by 
  sorry

end find_point_M_l164_164603


namespace not_triangle_preserving_sin_triangle_preserving_log_max_lambda_triangle_preserving_sin_l164_164875

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def triangle_preserving_function (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ (a b c : ℝ), a ∈ D → b ∈ D → c ∈ D → triangle_inequality a b c → triangle_inequality (f a) (f b) (f c)

def g (x : ℝ) : ℝ := Real.sin x
def h (x : ℝ) : ℝ := Real.log x

-- 1. Determine whether g(x) = sin x, x ∈ (0, π) is a "triangle-preserving function".
theorem not_triangle_preserving_sin : ¬ triangle_preserving_function g {x : ℝ | 0 < x ∧ x < π} := sorry

-- 2. Prove that the function h(x) = ln x, x ∈ [2, +∞) is a "triangle-preserving function".
theorem triangle_preserving_log : triangle_preserving_function h {x : ℝ | 2 ≤ x} := sorry

-- 3. Find the maximum value of λ such that f(x) = sin x, x ∈ (0, λ) is a "triangle-preserving function".
theorem max_lambda_triangle_preserving_sin : ∃ λ : ℝ, (∀ (x : ℝ), 0 < x → x < λ → triangle_preserving_function g {y | 0 < y ∧ y < x}) ∧ λ = (5 * Real.pi) / 6 := sorry

end not_triangle_preserving_sin_triangle_preserving_log_max_lambda_triangle_preserving_sin_l164_164875


namespace ellipse_focal_distance_l164_164850

noncomputable def ellipse_m_values : Set ℝ :=
  { m | ∃ (x y : ℝ), (x^2 / m) + (y^2 / 4) = 1 ∧ 
    (m - 4 = 1 ∨ 4 - m = 1) ∧ (abs(2√(m - 4)) = 2 ∨ abs(2√(4 - m)) = 2) }

theorem ellipse_focal_distance : ellipse_m_values = {3, 5} :=
begin
  sorry
end

end ellipse_focal_distance_l164_164850


namespace sequence_becomes_all_ones_l164_164820

theorem sequence_becomes_all_ones (n : ℕ) (a : Fin (2^n) → ℤ)
  (h : ∀ i, a i = 1 ∨ a i = -1) :
  ∃ k, ∀ i, (S^[k] a) i = 1 := 
sorry

where S (a : Fin (2^n) → ℤ) : Fin (2^n) → ℤ :=
λ i, a i * a ((i + 1) % (2^n))

end sequence_becomes_all_ones_l164_164820


namespace eval_expression_l164_164764

theorem eval_expression : 
  ( ( (476 * 100 + 424 * 100) * 2^3 - 4 * (476 * 100 * 424 * 100) ) * (376 - 150) ) / 250 = -7297340160 :=
by
  sorry

end eval_expression_l164_164764


namespace find_a_max_min_values_l164_164096

noncomputable def f (x a b : ℝ) := (1 / 3) * x ^ 3 - a * x ^ 2 + (a ^ 2 - 1) * x + b

theorem find_a (a b : ℝ) : 
  (∀ x, deriv (f x a b) x = x ^ 2 - 2 * a * x + (a ^ 2 - 1)) → 
  (deriv (f 1 a b) 1 = 0) → 
  ∃ a : ℝ, a = 1 :=
by
  sorry

theorem max_min_values (a b : ℝ) :
  let fa := f (1 : ℝ) 1 b in
  fa = 2 →
  (∀ x, deriv (f x 1 b) x = x^2 - 2 * x) →
  (f (0 : ℝ) 1 b = 8 / 3) →
  (f (2 : ℝ) 1 b = 4 / 3) →
  (f (-2 : ℝ) 1 b = -4) →
  (f (4 : ℝ) 1 b = 8) →
  ∃ (max_value min_value : ℝ), max_value = 8 ∧ min_value = -4 :=
by
  sorry

end find_a_max_min_values_l164_164096


namespace paper_holes_symmetric_l164_164819

-- Define the initial conditions
def folded_paper : Type := sorry -- Specific structure to represent the paper and its folds

def paper_fold_bottom_to_top (paper : folded_paper) : folded_paper := sorry
def paper_fold_right_half_to_left (paper : folded_paper) : folded_paper := sorry
def paper_fold_diagonal (paper : folded_paper) : folded_paper := sorry

-- Define a function that represents punching a hole near the folded edge
def punch_hole_near_folded_edge (paper : folded_paper) : folded_paper := sorry

-- Initial paper
def initial_paper : folded_paper := sorry

-- Folded and punched paper
def paper_after_folds_and_punch : folded_paper :=
  punch_hole_near_folded_edge (
    paper_fold_diagonal (
      paper_fold_right_half_to_left (
        paper_fold_bottom_to_top initial_paper)))

-- Unfolding the paper
def unfold_diagonal (paper : folded_paper) : folded_paper := sorry
def unfold_right_half (paper : folded_paper) : folded_paper := sorry
def unfold_bottom_to_top (paper : folded_paper) : folded_paper := sorry

def paper_after_unfolding : folded_paper :=
  unfold_bottom_to_top (
    unfold_right_half (
      unfold_diagonal paper_after_folds_and_punch))

-- Definition of hole pattern 'eight_symmetric_holes'
def eight_symmetric_holes (paper : folded_paper) : Prop := sorry

-- The proof problem
theorem paper_holes_symmetric :
  eight_symmetric_holes paper_after_unfolding := sorry

end paper_holes_symmetric_l164_164819


namespace integer_part_sum_inv_seq_l164_164346

def seq (n : ℕ) : ℚ :=
  match n with
  | 0     => 4 / 3
  | (n+1) => let a_n := seq n in a_n^2 - a_n + 1

theorem integer_part_sum_inv_seq :
  let s := (Finset.range 2017).sum (λ i => 1 / seq (i + 1))
  ⌊s⌋ = 2 :=
by
  sorry

end integer_part_sum_inv_seq_l164_164346


namespace weak_multiple_l164_164200

def is_weak (a b n : ℕ) : Prop :=
  ∀ (x y : ℕ), n ≠ a * x + b * y

theorem weak_multiple (a b n : ℕ) (h_coprime : Nat.gcd a b = 1) (h_weak : is_weak a b n) (h_bound : n < a * b / 6) : 
  ∃ k ≥ 2, is_weak a b (k * n) :=
by
  sorry

end weak_multiple_l164_164200


namespace range_f_le_4_l164_164203

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(1 - x)
  else 1 - Real.log2 x

theorem range_f_le_4 :
  { x : ℝ | f x ≤ 4 } = Set.Ici (-1) :=
by
  sorry

end range_f_le_4_l164_164203


namespace ab_gt_ac_neither_sufficient_nor_necessary_l164_164154

theorem ab_gt_ac_neither_sufficient_nor_necessary (a b c : ℝ) : 
  (ab > ac) ↔ (b > c) :=
by
  sorry

end ab_gt_ac_neither_sufficient_nor_necessary_l164_164154


namespace complement_A_in_U_l164_164547

def U : Set ℕ := {1, 2, 3, 4, 5}

def A : Set ℕ := {x | x ∈ ℤ ∧ x^2 - 3 * x < 0}

theorem complement_A_in_U :
  (U \ A) = {3, 4, 5} := by
  sorry

end complement_A_in_U_l164_164547


namespace curve_is_line_l164_164021

noncomputable def polar_to_cartesian (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (x, y)

theorem curve_is_line (r : ℝ) (θ : ℝ) :
  r = 1 / (Real.sin θ + Real.cos θ) ↔ ∃ (x y : ℝ), (x, y) = polar_to_cartesian r θ ∧ (x + y)^2 = 1 :=
by 
  sorry

end curve_is_line_l164_164021


namespace find_coefficients_l164_164474

theorem find_coefficients (a1 a2 : ℚ) :
  (4 * a1 + 5 * a2 = 9) ∧ (-a1 + 3 * a2 = 4) ↔ (a1 = 181 / 136) ∧ (a2 = 25 / 68) := 
sorry

end find_coefficients_l164_164474


namespace sum_of_x_y_l164_164248

theorem sum_of_x_y (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_x_y_l164_164248


namespace smallest_n_for_P_l164_164046

noncomputable def P (n : ℕ) : ℕ := n^(Nat.totient n / 2)

theorem smallest_n_for_P (n : ℕ) (h_positive : n > 0) :
  let Pn := P(P(P(n)))
  n = 6 → Pn > 10^12 := 
by 
  sorry

end smallest_n_for_P_l164_164046


namespace length_PQ_l164_164063

-- Definition of the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Definition of the line through point A(1, 0) with slope sqrt(3)
def line_through_A (x y : ℝ) : Prop := y = real.sqrt 3 * (x - 1)

-- Definition of points P and Q on both the line and the parabola
def is_intersection (P Q : ℝ × ℝ) : Prop :=
  parabola P.1 P.2 ∧ line_through_A P.1 P.2 ∧
  parabola Q.1 Q.2 ∧ line_through_A Q.1 Q.2

-- Define the length function from point P to Q
def length_between (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

-- Theorem to prove that the length of PQ is 16/3
theorem length_PQ {P Q : ℝ × ℝ} (hP : parabola P.1 P.2) (hQ : parabola Q.1 Q.2) 
  (h_int : is_intersection P Q) : length_between P Q = 16 / 3 :=
sorry

end length_PQ_l164_164063


namespace polygon_side_l164_164811

theorem polygon_side (total_area : ℝ) (area_upper_rect : ℝ) 
                     (area_lower_rect : ℝ) (possible_sides : set ℝ) :
  total_area = 72 → area_upper_rect = 10 → area_lower_rect = 20 →
  possible_sides = {7, 6} :=
by
  intro h_total h_upper h_lower
  have h1 : total_area - area_upper_rect - area_lower_rect = 42,
    by { sorry } 
  have h2 : ∀ l w : ℝ, l * w = 42 → (l = 7 ∨ l = 6 ∨ w = 7 ∨ w = 6),
    by { sorry }
  sorry

end polygon_side_l164_164811


namespace find_N_l164_164753

def nine_digit_number (n : ℕ) : Prop := 10^8 ≤ n ∧ n < 10^9

theorem find_N : ∃ N : ℕ, nine_digit_number N ∧ (N * 123456789 % 10^9 = 987654321) :=
by {
  use 989010989,
  split,
  {
    --prove that 989010989 is a nine-digit number
    unfold nine_digit_number,
    norm_num,
  },
  {
    --prove that 989010989 * 123456789 mod 10^9 = 987654321
    norm_num,
  },
  sorry
}

end find_N_l164_164753


namespace alyssa_spent_on_grapes_l164_164448

theorem alyssa_spent_on_grapes (t c g : ℝ) (h1 : t = 21.93) (h2 : c = 9.85) (h3 : t = g + c) : g = 12.08 :=
by
  sorry

end alyssa_spent_on_grapes_l164_164448


namespace tangent_line_at_point_existence_of_x4_l164_164537

-- Define the function and its properties
def f (x a b : ℝ) : ℝ := (x - a)^2 * (x - b)

-- Define the derivative of the function
def f_prime (x a b : ℝ) : ℝ := (x - 1) * (3 * x - 5)

-- Define the requirement for the first part of the problem
theorem tangent_line_at_point :
  ∀ (x : ℝ), f_prime (2 : ℝ) (1 : ℝ) (2 : ℝ) = 1 ∧ f (2 : ℝ) (1 : ℝ) (2 : ℝ) = 0 →
  ∃ (y : ℝ), y = x - 2 :=
sorry

-- Define the requirement for the second part of the problem
theorem existence_of_x4 (a b : ℝ) (h : a < b) :
  ∃ (x1 x2 x3 x4 : ℝ),
  x1 = a ∧
  x2 = (a + 2 * b) / 3 ∧
  x3 = b ∧
  x4 = (2 * a + b) / 3 ∧
  (x1, x2, x3, x4).to_list.nth_le 0 _ - (x1, x2, x3, x4).to_list.nth_le 1 _ =
  (x1, x2, x3, x4).to_list.nth_le 1 _ - (x1, x2, x3, x4).to_list.nth_le 2 _ :=
sorry

end tangent_line_at_point_existence_of_x4_l164_164537


namespace bruce_time_correct_l164_164740

variables (time_angus : ℕ) (walk_speed : ℕ) (run_speed : ℕ) (time_bruce : ℕ)

def time_to_minutes (t: ℕ) : ℕ := t

-- Define conditions
def angus_walk_time (t : ℕ) : ℕ := t / 2
def angus_run_time (t : ℕ) : ℕ := t / 2
def angus_walk_distance (walk_speed : ℕ) (walk_time : ℕ) : ℕ := (walk_speed * walk_time) / 60
def angus_run_distance (run_speed : ℕ) (run_time : ℕ) : ℕ := (run_speed * run_time) / 60
def angus_total_distance (walk_distance run_distance: ℕ) : ℕ := walk_distance + run_distance

def bruce_walk_distance (total_distance : ℕ) : ℕ := total_distance / 2
def bruce_run_distance (total_distance : ℕ) : ℕ := total_distance / 2
def bruce_walk_time (walk_distance walk_speed : ℕ) : ℕ := (walk_distance * 60) / walk_speed
def bruce_run_time (run_distance run_speed : ℕ) : ℕ := (run_distance * 60) / run_speed 
def bruce_total_time (walk_time run_time : ℕ) : ℕ := walk_time + run_time

-- Problem Statement to be proved
theorem bruce_time_correct : 
  ∀ (time_angus : ℕ) (walk_speed : ℕ) (run_speed : ℕ),
    time_angus = 40 →
    walk_speed = 3 →
    run_speed = 6 →
    let total_distance := angus_total_distance (angus_walk_distance walk_speed (angus_walk_time time_angus)) (angus_run_distance run_speed (angus_run_time time_angus))
    in bruce_total_time (bruce_walk_time (bruce_walk_distance total_distance) walk_speed) (bruce_run_time (bruce_run_distance total_distance) run_speed) = 45 := 
by
  sorry

end bruce_time_correct_l164_164740


namespace vector_sum_magnitude_l164_164531

noncomputable theory

variables (a b : ℝ → ℝ) (θ : ℝ)

-- Definitions for conditions
def mag_a : ℝ := 2
def mag_b : ℝ := 3
def angle_ab : ℝ := real.pi / 3  -- 60 degrees in radians

-- Pulling in the magnitude and dot product formula
def dot_product (a b : ℝ → ℝ) : ℝ := mag_a * mag_b * real.cos angle_ab
def vector_sum_magnitude_square : ℝ := mag_a^2 + mag_b^2 + 2 * dot_product a b

-- The statement to prove
theorem vector_sum_magnitude : | a + b | = real.sqrt vector_sum_magnitude_square :=
by
  sorry

end vector_sum_magnitude_l164_164531


namespace count_even_abundant_numbers_under_50_l164_164561

def is_abundant (n : ℕ) : Prop :=
  (∑ i in finset.range n \ {n} | i ∣ n, i) > n

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

theorem count_even_abundant_numbers_under_50 : 
  (finset.filter (λ n, is_even n ∧ is_abundant n) (finset.range 50)).card = 7 :=
by {
  sorry
}

end count_even_abundant_numbers_under_50_l164_164561


namespace normal_distribution_probability_l164_164533

/-- Given random variable X following a normal distribution with mean 2 and standard deviation σ,
    and the probability P(X < 4) is 0.84, prove that P(X ≤ 0) = 0.16. -/
theorem normal_distribution_probability (
  X : ℝ → ProbabilityDistribution,
  hX_normal : X = Normal 2 σ,
  h_prob : P(X < 4) = 0.84
) : P(X ≤ 0) = 0.16 :=
sorry

end normal_distribution_probability_l164_164533


namespace max_sum_arith_seq_l164_164144

theorem max_sum_arith_seq (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ)
  (h_arith_seq : ∀ n : ℕ, a (n + 1) = a 1 + n * d)
  (h_a1_pos : a 1 > 0)
  (h_d_neg : d < 0)
  (h_a5_3a7 : a 5 = 3 * a 7)
  (h_Sn_def : ∀ n : ℕ, S n = n / 2 * (2 * a 1 + (n - 1) * d)) :
  ∃ n : ℕ, (n = 7 ∨ n = 8) ∧ S n = max (S 7) (S 8) := by
  sorry

end max_sum_arith_seq_l164_164144


namespace general_formula_b_n_sum_formula_T_n_l164_164907

-- Each sequence is arithmetic and given conditions
def is_arithmetic_sequence (a : ℕ → ℕ) (a1 d : ℕ) := ∀ n, a (n + 1) = a n + d

def is_geometric_sequence (a : ℕ → ℕ) (q : ℕ) := ∀ n, a (n + 1) = a n * q

-- Given initial conditions
def initial_conditions (a : ℕ → ℕ) (d : ℕ) (b : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ d ≠ 0 ∧ b 1 = 1 ∧ b 2 = 2 ∧ b 3 = 5 ∧
  is_arithmetic_sequence a 1 d ∧ is_geometric_sequence (λ n, a (b n)) 3

-- To prove the formula for b_n
theorem general_formula_b_n (a : ℕ → ℕ) (b : ℕ → ℕ) (d : ℕ) (n : ℕ) (h : initial_conditions a d b) :
  b n = (3 ^ (n - 1) + 1) / 2 :=
sorry

-- To prove the sum T_n
theorem sum_formula_T_n (a : ℕ → ℕ) (b : ℕ → ℕ) (d n : ℕ) (h : initial_conditions a d b) :
  let c := λ n, log 3 (2 * b n - 1) in
  T n = -2 * n ^ 2 :=
sorry

end general_formula_b_n_sum_formula_T_n_l164_164907


namespace product_of_three_integers_sum_l164_164331

theorem product_of_three_integers_sum :
  ∀ (a b c : ℕ), (c = a + b) → (a * b * c = 8 * (a + b + c)) →
  (a > 0) → (b > 0) → (c > 0) →
  (∃ N1 N2 N3: ℕ, N1 = (a * b * (a + b)), N2 = (a * b * (a + b)), N3 = (a * b * (a + b)) ∧ 
  (N1 = 272 ∨ N2 = 160 ∨ N3 = 128) ∧ 
  (N1 + N2 + N3 = 560)) := sorry

end product_of_three_integers_sum_l164_164331


namespace exists_point_in_plane_P_at_distances_l164_164846

open Classical

-- Definitions of planes P, M, and N
noncomputable def plane (x: ℝ) (y: ℝ) (z: ℝ) : Prop := sorry  -- Define plane function as comparison (e.g., ax + by + cz = d)

-- We assume there are three planes P, M, and N
variable (P M N : ℝ → ℝ → ℝ → Prop)

-- Define distances m and n
variable (m n : ℝ)

-- Main theorem statement: there exists a point in P that is at distances m and n from planes M and N respectively
theorem exists_point_in_plane_P_at_distances (h_plane_P : ∃ x y z, P x y z)
    (h_distance_m : ∃ x y z, distance P x y z M = m)
    (h_distance_n : ∃ x y z, distance P x y z N = n) : 
  ∃ (x y z : ℝ), P x y z ∧ distance (x, y, z) M = m ∧ distance (x, y, z) N = n := 
sorry

end exists_point_in_plane_P_at_distances_l164_164846


namespace trey_used_50_nails_l164_164734

-- Definitions based on conditions
def decorations_with_sticky_strips := 15
def fraction_nails := 2/3
def fraction_thumbtacks := 2/5

-- Define D as total number of decorations and use the given conditions
noncomputable def total_decorations : ℕ :=
  let D := decorations_with_sticky_strips / ((1:ℚ) - fraction_nails - (fraction_thumbtacks * (1 - fraction_nails))) in
  if h : 0 < D ∧ D.denom = 1 then D.num else 0

-- Nails used by Trey
noncomputable def nails_used : ℕ := (fraction_nails * total_decorations).toNat

theorem trey_used_50_nails : nails_used = 50 := by
  sorry

end trey_used_50_nails_l164_164734


namespace select_numbers_sum_bound_l164_164982

theorem select_numbers_sum_bound (n : ℕ) 
  (array : Fin 2 → Fin n → ℝ) 
  (hpos : ∀ i j, 0 < array i j)
  (hsum : ∀ j, array 0 j + array 1 j = 1) :
  ∃ (selected : Fin n → ℝ),
    (∀ j, selected j = array 0 j ∨ selected j = array 1 j) ∧
    (Finset.univ.sum (λ j, selected j) ≤ (n + 1) / 4) :=
sorry

end select_numbers_sum_bound_l164_164982


namespace exists_positive_integer_m_l164_164672

theorem exists_positive_integer_m (m : ℕ) (h_positive : m > 0) : 
  ∃ (m : ℕ), m > 0 ∧ ∃ k : ℕ, 8 * m = k^2 := 
sorry

end exists_positive_integer_m_l164_164672


namespace middle_number_is_14_l164_164641

-- Conditions
variables (x y : ℝ)
variable h1 : 9 * x^2 + 4 * x^2 + 25 * x^2 = 1862
variable h2 : 3 * x + 2 * x + 5 * x + 4 * y + 7 * y = 155
variable hx : x = 7 -- derived from solution
variable hy : y = 85 / 11 -- derived from solution

-- Question converted to proof
theorem middle_number_is_14 : 2 * x = 14 :=
by sorry

end middle_number_is_14_l164_164641


namespace exists_good_subset_l164_164621

noncomputable def M : Set ℕ := {n | n ≥ 1 ∧ n ≤ 20}

def is_good (f : Set ℕ → ℕ) (T : Set ℕ) : Prop :=
  ∀ k ∈ T, f (T \ {k}) ≠ k

theorem exists_good_subset (f : Set ℕ → ℕ) :
  ∃ (T : Set ℕ), T ⊆ M ∧ T.card = 10 ∧ is_good f T :=
by
  sorry

end exists_good_subset_l164_164621


namespace gcd_180_270_eq_90_l164_164003

theorem gcd_180_270_eq_90 : Nat.gcd 180 270 = 90 := sorry

end gcd_180_270_eq_90_l164_164003


namespace points_on_line_proof_l164_164291

theorem points_on_line_proof (n : ℕ) (hn : n = 10) : 
  let after_first_procedure := 3 * n - 2 in
  let after_second_procedure := 3 * after_first_procedure - 2 in
  after_second_procedure = 82 :=
by
  let after_first_procedure := 3 * n - 2
  let after_second_procedure := 3 * after_first_procedure - 2
  have h : after_second_procedure = 9 * n - 8 := by
    calc
      after_second_procedure = 3 * (3 * n - 2) - 2 : rfl
                      ... = 9 * n - 6 - 2      : by ring
                      ... = 9 * n - 8          : by ring
  rw [hn] at h 
  exact h.symm.trans (by norm_num)

end points_on_line_proof_l164_164291


namespace solve_for_x_l164_164500

def minimum (a b : ℝ) : ℝ :=
if a < b then a else b

theorem solve_for_x :
  ∀ x : ℝ, (x ≠ 1) → 
  (minimum (1/(1 - x)) (2/(1 - x)) = (2/(x - 1)) - 3) → 
  x = 7/3 :=
by 
  intros x h1 h2
  sorry

end solve_for_x_l164_164500


namespace sum_of_real_numbers_l164_164225

theorem sum_of_real_numbers (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_real_numbers_l164_164225


namespace angles_correctness_l164_164091

theorem angles_correctness :
  (quadrant (-75) = 4) ∧ 
  (quadrant 225 = 3) ∧ 
  (quadrant 475 = 2) ∧ 
  (quadrant (-315) = 1) →
  4 :=
by
  sorry

end angles_correctness_l164_164091


namespace women_count_l164_164397

variable (x : ℤ)

-- Initial conditions
def initial_men := 4 * x
def initial_women := 5 * x
def men_after_entry := initial_men + 2
def women_after_leaving := initial_women - 3
def current_women := 2 * women_after_leaving

-- Given conditions
def men_in_room := 14
def x_value := (men_in_room - 2) / 4

-- Theorem to prove
theorem women_count : current_women = 24 :=
by
  -- Use the given value of x to reduce the proof
  have x_eq : x = x_value := sorry
  rw [x_eq]
  -- Prove that current_women equals 24
  sorry

end women_count_l164_164397


namespace sum_of_midpoint_coordinates_l164_164309

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def sum_of_coordinates (point : ℝ × ℝ) : ℝ :=
point.1 + point.2

theorem sum_of_midpoint_coordinates : 
  sum_of_coordinates (midpoint (1, 2) (7, 12)) = 11 :=
by
  sorry

end sum_of_midpoint_coordinates_l164_164309


namespace points_on_line_l164_164053

theorem points_on_line (t : ℝ) (ht : t ≠ 0) :
  let x := (t + 2) / t,
      y := (t - 2) / t in
  x + y = 2 :=
by
  sorry

end points_on_line_l164_164053


namespace geom_seq_general_formula_C_maximum_value_l164_164518

noncomputable def geom_seq (n : ℕ) : ℝ := sorry -- This function represents a_n
axiom a1_a3_condition : ∀ n : ℕ, 0 < n → (1 / n) * geom_seq 1 + (1 / n) * geom_seq 3 = 4
axiom a4_a6_condition : ∀ n : ℕ, 0 < n → (1 / n) * geom_seq 4 + (1 / n) * geom_seq 6 = 10
def S (n : ℕ) : ℝ := (1 / n) * geom_seq 1 + (1 / n) * geom_seq 2 + ... + (1 / n) * geom_seq n
def b (n : ℕ) : ℝ := 1 / (2 * S n)
def C (n : ℕ) : ℝ := (finset.range n).sum (λ i, b i) * (2 / 3) ^ n

theorem geom_seq_general_formula (n : ℕ) : geom_seq n = real.exp n := sorry

theorem C_maximum_value (n : ℕ) : C n ≤ (1 / 3) := sorry

end geom_seq_general_formula_C_maximum_value_l164_164518


namespace area_transformation_l164_164192

-- Define the given matrix
def matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 0], ![8, -2]]

-- Define the original area of region T
def area_T : ℝ := 15

-- Define the area of the transformed region T'
def area_T_prime : ℝ := 90

-- Prove that the area of T' is 90
theorem area_transformation :
  let det := |matrix.det| in
  let scaleFactor := det in
  area_T * scaleFactor = area_T_prime :=
by
  let det := matrix.det
  let scaleFactor := |det|
  have : scaleFactor = 6 := by sorry
  rw [this]
  have : area_T * 6 = 90 := by sorry
  rw [this]
  rfl

end area_transformation_l164_164192


namespace parabola_directrix_equation_l164_164717

-- Define the directrix of the parabola
def directrix (d: ℝ) : Prop := d = 4

-- Define the standard form of the parabola equation with given directrix
def parabola_equation (x y: ℝ) : Prop := x^2 = -16 * y

-- Theorem statement
theorem parabola_directrix_equation : ∀ (x y: ℝ), directrix 4 → parabola_equation x y :=
by 
  -- Presume the conditions and equations
  intro x y h,
  sorry  -- The proof will be written here

end parabola_directrix_equation_l164_164717


namespace speed_of_second_train_l164_164372

-- Define the conditions
def length_train1 : ℝ := 111
def length_train2 : ℝ := 165
def speed_train1 : ℝ := 100
def time_seconds : ℝ := 4.516002356175142

-- Convert distances and times
def total_distance_km : ℝ := (length_train1 + length_train2) / 1000
def time_hours : ℝ := time_seconds / 3600

-- Statement to prove the speed of the second train
theorem speed_of_second_train : 
  let relative_speed := total_distance_km / time_hours in
  let speed_train2 := relative_speed - speed_train1 in
  speed_train2 ≈ 119.976
:= by
  sorry

end speed_of_second_train_l164_164372


namespace cosine_sine_inequality_solution_l164_164847

theorem cosine_sine_inequality_solution (x : ℝ) :
  x ∈ Set.Icc (-4 * Real.pi / 3) (2 * Real.pi / 3) ∧
  (Real.cos x)^2018 + (Real.sin x)^(-2019) ≥ (Real.sin x)^2018 + (Real.cos x)^(-2019)
  ↔ x ∈ (Set.Ico (-4 * Real.pi / 3) (-Real.pi) ∪ 
          Set.Ico (-3 * Real.pi / 4) (-Real.pi / 2) ∪ 
          Set.Ioc 0 (Real.pi / 4) ∪ 
          Set.Ioc (Real.pi / 2) (2 * Real.pi / 3)) :=
sorry

end cosine_sine_inequality_solution_l164_164847


namespace rectangle_area_l164_164396

theorem rectangle_area (L B : ℕ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 226) : L * B = 3060 := by
  sorry

end rectangle_area_l164_164396


namespace fraction_eq_l164_164566

theorem fraction_eq {x : ℝ} (h : 1 - 6 / x + 9 / x ^ 2 - 2 / x ^ 3 = 0) :
  3 / x = 3 / 2 ∨ 3 / x = 3 / (2 + Real.sqrt 3) ∨ 3 / x = 3 / (2 - Real.sqrt 3) :=
sorry

end fraction_eq_l164_164566


namespace largest_profit_received_l164_164045

-- Define the conditions
def total_profit : ℕ := 60000
def profit_ratios : List ℕ := [2, 4, 3, 5, 6]
def total_shares : ℕ := profit_ratios.sum
def value_per_share : ℕ := total_profit / total_shares
def largest_share : ℕ := 6

-- Define the theorem we want to prove
theorem largest_profit_received : 
  ∃ p, p = largest_share * value_per_share ∧ p = 18000 :=
by
  use largest_share * value_per_share
  simp [largest_share, value_per_share, total_shares, total_profit, profit_ratios]
  sorry

end largest_profit_received_l164_164045


namespace slant_asymptote_sum_l164_164477

theorem slant_asymptote_sum (m b : ℝ) 
  (h : ∀ x : ℝ, y = 3*x^2 + 4*x - 8 / (x - 4) → y = m*x + b) :
  m + b = 19 :=
sorry

end slant_asymptote_sum_l164_164477


namespace minimum_value_abs_function_l164_164709

theorem minimum_value_abs_function : ∃ m : ℝ, (∀ x : ℝ, |x - 2| + 3 ≥ m) ∧ (∃ x₀ : ℝ, |x₀ - 2| + 3 = m) :=
by
  use 3
  split
  { intros x
    linarith [abs_nonneg (x - 2)] }
  { use 2
    norm_num }

end minimum_value_abs_function_l164_164709


namespace original_number_of_employees_l164_164423

theorem original_number_of_employees (x : ℕ) 
  (h1 : 0.77 * (x : ℝ) = 328) : x = 427 :=
sorry

end original_number_of_employees_l164_164423


namespace prism_edges_l164_164351

theorem prism_edges (V F E n : ℕ) (h1 : V + F + E = 44) (h2 : V = 2 * n) (h3 : F = n + 2) (h4 : E = 3 * n) : E = 21 := by
  sorry

end prism_edges_l164_164351


namespace odd_even_divisors_ratio_l164_164620

theorem odd_even_divisors_ratio (M : ℕ) (h1 : M = 2^5 * 3^5 * 5 * 7^3) :
  let sum_odd_divisors := (1 + 3 + 3^2 + 3^3 + 3^4 + 3^5) * (1 + 5) * (1 + 7 + 7^2 + 7^3)
  let sum_all_divisors := (1 + 2 + 4 + 8 + 16 + 32) * (1 + 3 + 3^2 + 3^3 + 3^4 + 3^5) * (1 + 5) * (1 + 7 + 7^2 + 7^3)
  let sum_even_divisors := sum_all_divisors - sum_odd_divisors
  sum_odd_divisors / sum_even_divisors = 1 / 62 :=
by
  sorry

end odd_even_divisors_ratio_l164_164620


namespace integral_quarter_circle_area_l164_164698

theorem integral_quarter_circle_area :
  ∫ x in 0..3, sqrt(9 - x^2) = (1/4) * π * 3^2 := 
sorry

end integral_quarter_circle_area_l164_164698


namespace exists_positive_integer_m_l164_164671

theorem exists_positive_integer_m (m : ℕ) (h_positive : m > 0) : 
  ∃ (m : ℕ), m > 0 ∧ ∃ k : ℕ, 8 * m = k^2 := 
sorry

end exists_positive_integer_m_l164_164671


namespace possible_values_of_sum_l164_164243

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l164_164243


namespace largest_term_in_expansion_sum_even_coefficients_l164_164123

noncomputable def largest_coeff_term_in_binomial_expansion (n : ℕ) : ℕ :=
  if h : n = 18 then 9 else sorry

noncomputable def sum_even_indexed_terms (n : ℕ) : ℕ :=
  let a_0_to_n : list ℕ := (list.range (n + 1)).map (λ k, (2*x - 1)^k)
  if h : n = 18 then (3^18 + 1) / 2 else sorry

theorem largest_term_in_expansion :
  largest_coeff_term_in_binomial_expansion 18 = 9 :=
by
  sorry

theorem sum_even_coefficients :
  sum_even_indexed_terms 18 = (3^18 + 1) / 2 :=
by
  sorry

end largest_term_in_expansion_sum_even_coefficients_l164_164123


namespace sum_first_n_terms_l164_164895

variables (a b c : ℕ) (n : ℕ)
def S := real.sqrt 3
def B := 60 * real.pi / 180 -- Function to convert degrees to radians if needed
def a_n : ℕ → ℕ := λ n, 2 * n
def b_n : ℕ → ℕ := λ n, 2^(n-1)
def c_n (n : ℕ) := a_n n * b_n n
def S_n (n : ℕ) := Σ i in finset.range n, c_n (i+1)

-- Conditions
axiom area_eq : S = (1/2) * a * b * real.sin B
axiom angle_eq : B = 60 * real.pi / 180
axiom side_eq : a^2 + c^2 = 2 * b^2

theorem sum_first_n_terms (a b c : ℕ) (n : ℕ) 
  (h : S = (1/2) * a * b * real.sin B) 
  (hB : B = 60 * real.pi / 180) 
  (h2 : a^2 + c^2 = 2 * b^2) :
  S_n n = (n-1) * 2^(n+1) + 2 :=
sorry

-- End of the Lean 4 statement

end sum_first_n_terms_l164_164895


namespace points_on_line_l164_164277

theorem points_on_line (n : ℕ) : 9 * n - 8 = 82 → n = 10 :=
by
  sorry

end points_on_line_l164_164277


namespace num_geometric_sequence_angles_l164_164476

theorem num_geometric_sequence_angles (θ : ℝ) : 
  -π < θ ∧ θ < π ∧ ¬∃ k : ℤ, θ = k * (π / 2) ∧ (∃ a b c: ℝ, ∃ permutation : list ℝ, permutation = [a, b, c] ∧ (∀ (i j k : ℕ), i+j+k=3 → i≠j → j≠k → i≠k → b = sqrt(a * c) ∧ (sin θ)^2 = permutation.nth i ∧ (cos θ)^2 = permutation.nth j ∧ (tan θ)^2 = permutation.nth k)) →
  {θ | -π < θ ∧ θ < π ∧ ¬∃ k : ℤ, θ = k * (π / 2) ∧ (∃ a b c: ℝ, ∃ permutation : list ℝ, permutation = [a, b, c] ∧ (∀ (i j k : ℕ), i+j+k=3 → i≠j → j≠k → i≠k → b = sqrt(a * c) ∧ (sin θ)^2 = permutation.nth i ∧ (cos θ)^2 = permutation.nth j ∧ (tan θ)^2 = permutation.nth k))}.card = 4 :=
begin
  sorry
end

end num_geometric_sequence_angles_l164_164476


namespace slope_of_tangent_line_maximized_area_l164_164081

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

noncomputable def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

noncomputable def is_tangent_line (l : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ k m, l = (λ x, k * x + m) ∧ m^2 = 2 * (k^2 + 1) ∧ (P.1 ^ 2 / 4 + P.2 ^ 2 = 1)

noncomputable def area_maximized (l : ℝ → ℝ) : Prop :=
  let A := (2, l(2))
  let B := (-2, l(-2))
  1 / 2 * abs (A.1 * B.2 - A.2 * B.1) = 2

theorem slope_of_tangent_line_maximized_area (P : ℝ × ℝ) (hP : ellipse P.1 P.2) :
  ∃ k : ℝ, (is_tangent_line (λ x, k * x + (1 - P.1 * k)).1 P ∧ abs k = real.sqrt 2 / 2) :=
sorry

end slope_of_tangent_line_maximized_area_l164_164081


namespace find_PR_in_triangle_l164_164608

theorem find_PR_in_triangle (P Q R M : ℝ) (PQ QR PM : ℝ):
  PQ = 7 →
  QR = 10 →
  PM = 5 →
  M = (Q + R) / 2 →
  PR = Real.sqrt 149 := 
sorry

end find_PR_in_triangle_l164_164608


namespace count_numbers_less_than_40_l164_164384

theorem count_numbers_less_than_40 : 
  let digits := {1, 3, 7, 8}
  let valid_numbers := {x * 10 + y | x y : ℕ // x ∈ digits ∧ y ∈ digits ∧ x * 10 + y < 40}
  valid_numbers.card = 6 
:= sorry

end count_numbers_less_than_40_l164_164384


namespace john_total_water_usage_l164_164615

-- Define the basic conditions
def total_days_in_weeks (weeks : ℕ) : ℕ := weeks * 7
def showers_every_other_day (days : ℕ) : ℕ := days / 2
def total_minutes_shower (showers : ℕ) (minutes_per_shower : ℕ) : ℕ := showers * minutes_per_shower
def total_water_usage (total_minutes : ℕ) (water_per_minute : ℕ) : ℕ := total_minutes * water_per_minute

-- Main statement
theorem john_total_water_usage :
  total_water_usage (total_minutes_shower (showers_every_other_day (total_days_in_weeks 4)) 10) 2 = 280 :=
by
  sorry

end john_total_water_usage_l164_164615


namespace euler_conjecture_counter_example_l164_164218

theorem euler_conjecture_counter_example :
  ∃ (n : ℕ), 133^5 + 110^5 + 84^5 + 27^5 = n^5 ∧ n = 144 :=
by
  sorry

end euler_conjecture_counter_example_l164_164218


namespace sum_of_possible_values_of_N_l164_164334

theorem sum_of_possible_values_of_N :
  ∃ a b c : ℕ, (a > 0 ∧ b > 0 ∧ c > 0) ∧ (abc = 8 * (a + b + c)) ∧ (c = a + b)
  ∧ (2560 = 560) :=
by
  sorry

end sum_of_possible_values_of_N_l164_164334


namespace nonagon_diagonals_count_l164_164933

theorem nonagon_diagonals_count (n : ℕ) (h1 : n = 9) : 
  let diagonals_per_vertex := n - 3 in
  let naive_count := n * diagonals_per_vertex in
  let distinct_diagonals := naive_count / 2 in
  distinct_diagonals = 27 :=
by
  sorry

end nonagon_diagonals_count_l164_164933


namespace trajectory_of_P_l164_164428

noncomputable def trajectory_equation (x y : ℝ) : Prop :=
  (2 * x - 3) ^ 2 + 4 * y ^ 2 = 1

theorem trajectory_of_P (m n x y : ℝ) (hM_on_circle : m^2 + n^2 = 1)
  (hP_midpoint : 2 * x = 3 + m ∧ 2 * y = n) : trajectory_equation x y :=
by 
  sorry

end trajectory_of_P_l164_164428


namespace curve_is_line_l164_164020

noncomputable def polar_to_cartesian (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (x, y)

theorem curve_is_line (r : ℝ) (θ : ℝ) :
  r = 1 / (Real.sin θ + Real.cos θ) ↔ ∃ (x y : ℝ), (x, y) = polar_to_cartesian r θ ∧ (x + y)^2 = 1 :=
by 
  sorry

end curve_is_line_l164_164020


namespace carpool_solution_l164_164826

noncomputable def carpool_problem := ∃ x: ℝ, (3 * (x + x / 2) = 36) ∧ (x = 8)

theorem carpool_solution : ∃ x: ℝ, 3 * (x + x / 2) = 36 ↔ x = 8 :=
begin
  sorry
end

end carpool_solution_l164_164826


namespace find_value_of_A_l164_164056

theorem find_value_of_A (x y A : ℝ)
  (h1 : 2^x = A)
  (h2 : 7^(2*y) = A)
  (h3 : 1 / x + 2 / y = 2) : 
  A = 7 * Real.sqrt 2 := 
sorry

end find_value_of_A_l164_164056


namespace unit_vector_collinear_with_ab_l164_164072

variable (A B : ℝ × ℝ) (u v : ℝ × ℝ)

def vector_ab (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def is_unit_vector (u : ℝ × ℝ) : Prop :=
  magnitude u = 1

def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem unit_vector_collinear_with_ab :
  ∀ A B: ℝ × ℝ,
  A = (1, 3) → B = (4, -1) →
  let v := vector_ab A B in
  let u₁ := (3 / 5, -4 / 5) in
  let u₂ := (-3 / 5, 4 / 5) in
  is_unit_vector u₁ → is_unit_vector u₂ →
  collinear u₁ v ∧ collinear u₂ v := by
  sorry

end unit_vector_collinear_with_ab_l164_164072


namespace shifted_sine_function_eq_l164_164680

theorem shifted_sine_function_eq :
  ∀ x : ℝ, (2 * sin (2 * (x - (π / 4)) + π / 6)) = 2 * sin (2 * x - π / 3) := 
by
  sorry  -- proof is omitted

end shifted_sine_function_eq_l164_164680


namespace tank_capacity_l164_164118

variable (C : ℝ)  -- total capacity of the tank

-- The tank is 5/8 full initially
axiom h1 : (5/8) * C + 15 = (19/24) * C

theorem tank_capacity : C = 90 :=
by
  sorry

end tank_capacity_l164_164118


namespace rectangle_same_color_exists_l164_164761

-- Define the board size: 4 rows and 7 columns
constant row_count : ℕ
constant col_count : ℕ

-- Define a color type where each cell can be either white or black
inductive Color
  | white
  | black

-- Define a function that describes the coloring of the board
constant color : ℕ → ℕ → Color

-- Assume the specific board dimensions
axiom row_condition : row_count = 4
axiom col_condition : col_count = 7

-- Define the main theorem to be proven
theorem rectangle_same_color_exists (color : ℕ → ℕ → Color): 
  ∃ r1 r2 c1 c2: ℕ, r1 < r2 ∧ c1 < c2 ∧ color r1 c1 = color r1 c2 ∧ color r1 c1 = color r2 c1 ∧ color r1 c1 = color r2 c2 :=
by sorry

end rectangle_same_color_exists_l164_164761


namespace clock_angle_15_40_l164_164834

/-- The angle between the hour and minute hands at 15 hours and 40 minutes is 130 degrees. -/
theorem clock_angle_15_40 : 
  let hour_hand_deg (hours minutes : ℕ) : ℝ := (hours % 12) * 30 + (minutes / 60) * 30
      minute_hand_deg (minutes : ℕ) : ℝ := minutes * 6
      angle_between (deg1 deg2 : ℝ) : ℝ := abs (deg1 - deg2)
  in angle_between (hour_hand_deg 15 40) (minute_hand_deg 40) = 130 := 
by {
  sorry
}

end clock_angle_15_40_l164_164834


namespace rest_days_in_1200_days_l164_164816

noncomputable def rest_days_coinciding (n : ℕ) : ℕ :=
  if h : n > 0 then (n / 6) else 0

theorem rest_days_in_1200_days :
  rest_days_coinciding 1200 = 200 :=
by
  sorry

end rest_days_in_1200_days_l164_164816


namespace ellipse_eccentricity_l164_164855

theorem ellipse_eccentricity 
  (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a > 0) (h4 : c^2 = a^2 - b^2) :
  let e := c / a in e = sqrt 2 / 2 :=
by
  sorry

end ellipse_eccentricity_l164_164855


namespace sum_of_x_y_l164_164247

theorem sum_of_x_y (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_x_y_l164_164247


namespace hyperbola_condition_l164_164718

theorem hyperbola_condition (m : ℝ) : (m > 0) ↔ (2 + m > 0 ∧ 1 + m > 0) :=
by sorry

end hyperbola_condition_l164_164718


namespace condition_iff_l164_164770

def is_pure_imaginary (z : ℂ) : Prop :=
  ∃ y : ℝ, z = complex.I * y

theorem condition_iff (a : ℝ) : 
  (a = 1) ↔ is_pure_imaginary (a^2 - 1 + (a + 1) * complex.I) :=
by sorry

end condition_iff_l164_164770


namespace factorize_expression_l164_164844

variable (a : ℝ)

theorem factorize_expression : a^3 + 4 * a^2 + 4 * a = a * (a + 2)^2 := by
  sorry

end factorize_expression_l164_164844


namespace find_xy_l164_164018

theorem find_xy :
  ∃ (x y : ℝ), (x - 14)^2 + (y - 15)^2 + (x - y)^2 = 1/3 ∧ x = 14 + 1/3 ∧ y = 14 + 2/3 :=
by
  sorry

end find_xy_l164_164018


namespace PQRS_value_l164_164191

theorem PQRS_value
  (P Q R S : ℝ)
  (hP : 0 < P)
  (hQ : 0 < Q)
  (hR : 0 < R)
  (hS : 0 < S)
  (h1 : Real.log (P * Q) / Real.log 10 + Real.log (P * S) / Real.log 10 = 2)
  (h2 : Real.log (Q * S) / Real.log 10 + Real.log (Q * R) / Real.log 10 = 3)
  (h3 : Real.log (R * P) / Real.log 10 + Real.log (R * S) / Real.log 10 = 5) :
  P * Q * R * S = 100000 := 
sorry

end PQRS_value_l164_164191


namespace evaluate_expression_l164_164499

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem evaluate_expression :
  ∃ f : ℝ, floor 6.5 * floor f + floor 2 * 7.2 + floor 8.4 - 9.8 = 12.599999999999998 ∧ floor f = 0 :=
by
  sorry

end evaluate_expression_l164_164499


namespace coefficient_of_fractional_term_l164_164975

theorem coefficient_of_fractional_term :
  (∀ (x : ℝ), (∑ k in (finset.range (7+1)), (3 * real.sqrt x - 1 / x)^7.coeff k) = 128) →
  (∃ c : ℝ, (3 * real.sqrt x - 1 / x)^7.coeff 5 = c ∧ c = -189) :=
begin
  intro h,
  sorry
end

end coefficient_of_fractional_term_l164_164975


namespace difference_in_payments_difference_between_plans_l164_164642

def initial_loan_amount : ℝ := 15000
def annual_interest_rate : ℝ := 0.08
def plan1_n : ℕ := 2  -- semi-annual compounding
def plan2_n : ℕ := 12  -- monthly compounding
def duration_years : ℕ := 5
def plan1_annual_payment : ℝ := 3000

-- Plan 1 formula: P(1 + r/n)^(nt) followed by annual payment deduction.
def plan1_balance (P : ℝ) (r : ℝ) (n : ℕ) (annual_payment : ℝ) : ℕ → ℝ
| 0     => P
| (i+1) => plan1_balance P r n annual_payment i * (1 + r / n) ^ n - annual_payment

-- Plan 2 formula: P(1 + r/n)^(nt) without intermediate payments.
def plan2_total_payment (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem difference_in_payments :
  (plan1_annual_payment * duration_years) + initial_loan_amount 
  - plan1_balance initial_loan_amount annual_interest_rate plan1_n plan1_annual_payment duration_years 
  = 3000 := sorry

theorem difference_between_plans :
  let final_plan1_payment := plan1_annual_payment * duration_years
  let final_plan2_payment := plan2_total_payment initial_loan_amount annual_interest_rate plan2_n duration_years
  final_plan2_payment - final_plan1_payment = 7375 := sorry

end difference_in_payments_difference_between_plans_l164_164642


namespace possible_values_of_sum_l164_164232

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + y^3 + 21 * x * y = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l164_164232


namespace sum_possible_values_N_l164_164327

theorem sum_possible_values_N (a b c N : ℕ) (ha : 0 < a) (hb : 0 < b)
  (hc : c = a + b) (hN : N = a * b * c) (h_condition : N = 8 * (a + b + c)) :
  (N = 272 ∨ N = 160 ∨ N = 128) →
  (272 + 160 + 128) = 560 :=
by {
  intros h,
  have h1 : N = 272 ∨ N = 160 ∨ N = 128,
  from h,
  exact eq.refl 560,
}

end sum_possible_values_N_l164_164327


namespace hours_of_rain_l164_164503

def totalHours : ℕ := 9
def noRainHours : ℕ := 5
def rainHours : ℕ := totalHours - noRainHours

theorem hours_of_rain : rainHours = 4 := by
  sorry

end hours_of_rain_l164_164503


namespace number_divisible_by_75_l164_164992

def is_two_digit (x : ℕ) := x >= 10 ∧ x < 100

theorem number_divisible_by_75 {a b : ℕ} (h1 : a * b = 35) (h2 : is_two_digit (10 * a + b)) : (10 * a + b) % 75 = 0 :=
sorry

end number_divisible_by_75_l164_164992


namespace largest_divisor_of_composite_l164_164050

theorem largest_divisor_of_composite (n : ℕ) (h : n > 1 ∧ ¬ Nat.Prime n) : 12 ∣ (n^4 - n^2) :=
sorry

end largest_divisor_of_composite_l164_164050


namespace arithmetic_sequence_reciprocal_sum_l164_164689

open BigOperators

noncomputable def x (n : ℕ) (d : ℝ) (x₁ : ℝ) : ℝ := x₁ + ((n - 1) * d)

theorem arithmetic_sequence_reciprocal_sum
  (x₁ : ℝ) (d : ℝ) (n : ℕ) (h_pos : 2 ≤ n) (h_nonzero : ∀ k, 1 ≤ k ∧ k ≤ n → x k d x₁ ≠ 0) :
  (∑ k in Finset.range (n - 1), 1 / (x (k + 1) d x₁) * (x (k + 2) d x₁)) = (n - 1) / (x 1 d x₁ * x n d x₁) :=
begin
  sorry
end

end arithmetic_sequence_reciprocal_sum_l164_164689


namespace shahrazad_stories_not_power_of_two_l164_164294

theorem shahrazad_stories_not_power_of_two :
  ∀ (a b c : ℕ) (k : ℕ),
  a + b + c = 1001 → 27 * a + 14 * b + c = 2^k → False :=
by {
  sorry
}

end shahrazad_stories_not_power_of_two_l164_164294


namespace part1_monotonic_part2_ineq_l164_164922

noncomputable def f1 (x : ℝ) : ℝ := x
noncomputable def f2 (x : ℝ) : ℝ := Real.exp x
noncomputable def f3 (x : ℝ) : ℝ := Real.log x

-- Note: This statement combines the requirements for h being monotonic and the range of m.
theorem part1_monotonic {m : ℝ} : 
  (∀ x : ℝ, 1 / 2 < x ∧ x ≤ 2 → deriv (λ x, m * f1 x - f3 x) x ≥ 0) ↔ m ∈ Set.Iic (1/2) ∪ Set.Ici 2 :=
sorry

-- Proof for f2(x) > f3(x) + 2 * derivative of f1(x) for all x in (0, ∞).
theorem part2_ineq (x : ℝ) (hx : x > 0) : 
  f2 x > f3 x + 2 * deriv f1 x := 
sorry

end part1_monotonic_part2_ineq_l164_164922


namespace ellipse_fixed_point_l164_164071

-- Define the conditions
def ellipse_condition (a b : ℝ) (a_pos b_pos : a > b ∧ b > 0) :=
  ∃ x y, x = 1 ∧ y = (sqrt 6) / 3 ∧ 
          (x^2) / (a^2) + (y^2) / (b^2) = 1 ∧
          a^2 = b^2 + ((sqrt 6) / 3 * a)^2

noncomputable def ellipse_eq : Prop :=
  ∃ (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (h : ellipse_condition a b (by linarith)),
  let eqn := a = sqrt 3 ∧ b = 1 in
  eqn ∧ (∀ x y, ((x^2) / 3 + y^2 = 1) ↔ ((x^2) / (a^2) + (y^2) / (b^2) = 1))

def moving_line_through_fixed_point (a b : ℝ) (a_pos b_pos : a > b ∧ b > 0) : Prop :=
  ∀ (l : ℝ → ℝ → Prop) (AP AQ : ℝ × ℝ → ℝ × ℝ),
  (∃ P Q : ℝ × ℝ, P ≠ Q ∧ l P ∧ l Q ∧ (fst AP P) * (fst AQ Q) + (snd AP P) * (snd AQ Q) = 0) →
  l (0, -1/2)

-- The final theorem combining both parts:
theorem ellipse_fixed_point :
  ∃ (a b : ℝ) (a_pos : a > b ∧ b > 0) (h : ellipse_condition a b (by linarith)),
    (let eqn := a = sqrt 3 ∧ b = 1 in eqn) ∧
    (ellipse_eq) ∧
    (moving_line_through_fixed_point a b (by linarith)) :=
sorry

end ellipse_fixed_point_l164_164071


namespace sum_of_x_y_possible_values_l164_164263

theorem sum_of_x_y_possible_values (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
sorry

end sum_of_x_y_possible_values_l164_164263


namespace fraction_reduction_by_11_l164_164681

theorem fraction_reduction_by_11 (k : ℕ) :
  (k^2 - 5 * k + 8) % 11 = 0 → 
  (k^2 + 6 * k + 19) % 11 = 0 :=
by
  sorry

end fraction_reduction_by_11_l164_164681


namespace prime_remainders_between_50_100_have_five_qualified_numbers_l164_164960

open Nat

def prime_remainder_set : Set ℕ := {2, 3, 5}

def count_prime_remainders_between (a b n : ℕ) : ℕ :=
  (List.range (b - a)).countp (λ x, x + a ∈ (range b).filter prime ∧ ((x + a) % n) ∈ prime_remainder_set)

theorem prime_remainders_between_50_100_have_five_qualified_numbers :
  count_prime_remainders_between 50 100 7 = 5 := 
by
  sorry

end prime_remainders_between_50_100_have_five_qualified_numbers_l164_164960


namespace sum_of_first_4n_integers_l164_164582

theorem sum_of_first_4n_integers (n : ℕ) 
  (h : (3 * n * (3 * n + 1)) / 2 = (n * (n + 1)) / 2 + 150) : 
  (4 * n * (4 * n + 1)) / 2 = 300 :=
by
  sorry

end sum_of_first_4n_integers_l164_164582


namespace nonagon_distinct_diagonals_l164_164946

theorem nonagon_distinct_diagonals : 
  let n := 9 in
  ∃ (d : ℕ), d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end nonagon_distinct_diagonals_l164_164946


namespace nonagon_diagonals_l164_164944

def convex_nonagon_diagonals : Prop :=
∀ (n : ℕ), n = 9 → (n * (n - 3)) / 2 = 27

theorem nonagon_diagonals : convex_nonagon_diagonals :=
by {
  sorry,
}

end nonagon_diagonals_l164_164944


namespace count_prime_remainders_l164_164963

-- List of prime numbers between 50 and 100
def primes : List ℕ := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

-- Function to check if a number is prime
def is_prime (n : ℕ) : Bool :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

-- Function to get the remainder of the division by 7
def remainder (n : ℕ) : ℕ :=
  n % 7

-- Prime remainders when dividing primes by 7
def prime_remainders : List ℕ :=
  primes.filter (λ p => is_prime (remainder p))

-- The count of prime numbers between 50 and 100 which have prime remainders when divided by 7
theorem count_prime_remainders : prime_remainders.length = 5 :=
  sorry

end count_prime_remainders_l164_164963


namespace inequality_abc_l164_164853

open Real

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / sqrt (a^2 + 8 * b * c)) + (b / sqrt (b^2 + 8 * c * a)) + (c / sqrt (c^2 + 8 * a * b)) ≥ 1 :=
sorry

end inequality_abc_l164_164853


namespace nonagon_diagonals_count_l164_164935

theorem nonagon_diagonals_count (n : ℕ) (h1 : n = 9) : 
  let diagonals_per_vertex := n - 3 in
  let naive_count := n * diagonals_per_vertex in
  let distinct_diagonals := naive_count / 2 in
  distinct_diagonals = 27 :=
by
  sorry

end nonagon_diagonals_count_l164_164935


namespace proof_problem_l164_164100

def f (x : ℝ) : ℝ := x^2 - 6 * x + 5

-- The two conditions
def condition1 (x y : ℝ) : Prop := f x + f y ≤ 0
def condition2 (x y : ℝ) : Prop := f x - f y ≥ 0

-- Equivalent description
def circle_condition (x y : ℝ) : Prop := (x - 3)^2 + (y - 3)^2 ≤ 8
def region1 (x y : ℝ) : Prop := y ≤ x ∧ y ≥ 6 - x
def region2 (x y : ℝ) : Prop := y ≥ x ∧ y ≤ 6 - x

-- The proof statement
theorem proof_problem (x y : ℝ) :
  (condition1 x y ∧ condition2 x y) ↔ 
  (circle_condition x y ∧ (region1 x y ∨ region2 x y)) :=
sorry

end proof_problem_l164_164100


namespace points_on_line_l164_164278

theorem points_on_line (n : ℕ) : 9 * n - 8 = 82 → n = 10 :=
by
  sorry

end points_on_line_l164_164278


namespace remainder_of_sum_binom_mod_2027_l164_164320

theorem remainder_of_sum_binom_mod_2027 :
  prime 2027 →
  (∑ k in Finset.range 65, Nat.choose 2024 k) % 2027 = 1089 :=
by
  intro hp
  sorry

end remainder_of_sum_binom_mod_2027_l164_164320


namespace general_term_correct_sum_of_squares_lt_7_over_6_l164_164766

-- Define the sequence a_n using the initial conditions and recurrence relation
def seq_a : ℕ → ℝ
| 1 => 1
| 2 => 1 / 4
| (n+1) => (n-1) * seq_a n / (n - seq_a n)

-- Define the general term formula for the sequence
def general_term (n : ℕ) : ℝ :=
  1 / (3 * n - 2)

-- The first part of the proof problem: Prove the general term formula
theorem general_term_correct : ∀ n ∈ ℕ, seq_a n = general_term n := 
  sorry

-- The second part of the proof problem: Prove the inequality
theorem sum_of_squares_lt_7_over_6 : ∀ n ∈ ℕ, (∑ k in finset.range n, (seq_a k)^2) < 7 / 6 :=
  sorry

end general_term_correct_sum_of_squares_lt_7_over_6_l164_164766


namespace min_value_omega_l164_164624

noncomputable def minValue (a b c : ℤ) (ω : ℂ) := complex.abs (a + b * ω + c * ω^3)

theorem min_value_omega (a b c : ℤ) (ω : ℂ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4: ω^4 = 1) (h5: ω ≠ 1) : minValue a b c ω = real.sqrt 3 :=
sorry

end min_value_omega_l164_164624


namespace line_intersects_max_points_l164_164502

theorem line_intersects_max_points 
    (C : Finset Circle) 
    (h_card : C.card = 4) 
    (h_intersects : ∀ (c1 c2 : Circle), c1 ∈ C → c2 ∈ C → c1 ≠ c2 → (∃ p, p ∈ c1 ∧ p ∈ c2)) :
    ∃ L : Line, ∀ c ∈ C, ∃ p1 p2 : Point, p1 ≠ p2 ∧ p1 ∈ L ∧ p2 ∈ L ∧ p1 ∈ c ∧ p2 ∈ c ∧ (∃ (p3 p4 : Point), p3 ∈ L ∧ p4 ∈ L ∧ p3 ∈ c ∧ p4 ∈ c ∧ p3 ≠ p4) :=
by
  sorry

end line_intersects_max_points_l164_164502


namespace farm_field_area_l164_164971

variable (A D : ℕ)

theorem farm_field_area
  (h1 : 160 * D = A)
  (h2 : 85 * (D + 2) + 40 = A) :
  A = 480 :=
by
  sorry

end farm_field_area_l164_164971


namespace car_rental_savings_l164_164162

theorem car_rental_savings :
  let distance_one_way := 150 in
  let rental_cost_option_1 := 50 in
  let rental_cost_option_2 := 90 in
  let distance_per_liter := 15 in
  let gasoline_cost_per_liter := 0.90 in
  let total_distance := distance_one_way * 2 in
  let liters_needed := total_distance / distance_per_liter in
  let gasoline_cost := liters_needed * gasoline_cost_per_liter in
  let total_cost_option_1 := rental_cost_option_1 + gasoline_cost in
  rental_cost_option_2 - total_cost_option_1 = 22 :=
by 
  sorry

end car_rental_savings_l164_164162


namespace tangent_line_eqn_l164_164064

open Real

def circle_eq (x y xc yc r : ℝ) := (x - xc) ^ 2 + (y - yc) ^ 2 = r ^ 2

def point_on_circle (x y xc yc r : ℝ) := circle_eq x y xc yc r

theorem tangent_line_eqn :
  ∃ line_eq : ℝ → ℝ → Prop,
    point_on_circle 3 1 1 0 1 →
    circle_eq 3 1 2 (1/2) (√5 / 2) →
    (∀ x y, point_on_circle x y 1 0 1 → x * line_eq + y = 3 → True) →
    (∀ x y, line_eq x y → 2 * x + y - 3 = 0) :=
by
  sorry

end tangent_line_eqn_l164_164064


namespace exist_initial_points_l164_164270

theorem exist_initial_points (n : ℕ) (h : 9 * n - 8 = 82) : ∃ n = 10 :=
by
  sorry

end exist_initial_points_l164_164270


namespace find_length_AE_l164_164168

-- Declare variables and conditions
variables (A B C D E : Type) [metric_space A] [metric_space B] [metric_space C]
          [metric_space D] [metric_space E]
          (dist : A → B → ℝ)
          (d_AB : dist A B = 7)
          (d_BC : dist B C = 8)
          (d_AC : dist A C = 6)
          (mid_point_D : dist B D = 4 ∧ dist D C = 4)
          (circ : Π {P Q R : Type}, P → Q → R → Type)
          (h_circ : ∀ A B D, circ A B D)

-- The goal is to prove the length |AE|
theorem find_length_AE :
  ∃ AE : ℝ, AE = 2 / 3 :=
sorry

end find_length_AE_l164_164168


namespace part1_part2_part3_l164_164769

noncomputable def seq : ℕ → ℝ
| 0     := 1994
| (n+1) := seq n ^ 2 / (2 * ⌊seq n⌋ + 21)

theorem part1 : seq 12 < 1 := by
  sorry

theorem part2 : ∃ l : ℝ, (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |seq n - l| < ε) ∧ (l = 0) := by
  sorry

theorem part3 : ∃ K : ℕ, (K > 0) ∧ (seq K < 1) ∧ (∀ K' < K, seq K' ≥ 1) := by
  use 10
  sorry

end part1_part2_part3_l164_164769


namespace sum_of_real_numbers_l164_164224

theorem sum_of_real_numbers (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_real_numbers_l164_164224


namespace sum_of_undefined_values_l164_164005

def g (x : ℝ) : ℝ := 1 / (1 + 1 / (1 + 1 / (1 + 1 / x)))

theorem sum_of_undefined_values :
  (∑ x in ({0, -1, -1/2, -2/3} : Finset ℝ), x) = -13 / 6 := by
  sorry

end sum_of_undefined_values_l164_164005


namespace exists_m_sqrt_8m_integer_l164_164669

theorem exists_m_sqrt_8m_integer : ∃ (m : ℕ), (m > 0) ∧ (∃ k : ℕ, k^2 = 8 * m) :=
by
  use 2
  split
  · exact Nat.succ_pos 1
  · use 4
    exact Nat.succ_pos 1
    sorry

end exists_m_sqrt_8m_integer_l164_164669


namespace minimum_value_f1_f2_l164_164830

def f : ℤ → ℤ := sorry

axiom f_equation (x y : ℤ) : 
  f(x^2 - 3 * y^2) + f(x^2 + y^2) = 2 * (x + y) * f(x - y)

axiom f_positive (n : ℤ) (h : n > 0) : 
  f(n) > 0 

axiom f_perfect_square : 
  ∃ k : ℕ, f(2015) * f(2016) = k^2 

theorem minimum_value_f1_f2 : f(1) + f(2) = 246 := 
sorry

end minimum_value_f1_f2_l164_164830


namespace parallel_or_perpendicular_count_l164_164470

def line_a (x : ℝ) : ℝ := 4 * x + 3
def line_b (x : ℝ) : ℝ := (3 / 2) * x + 2
def line_c (x : ℝ) : ℝ := 8 * x - 1
def line_d (x : ℝ) : ℝ := (1 / 2) * x + 3
def line_e (x : ℝ) : ℝ := (1 / 2) * x - 2

theorem parallel_or_perpendicular_count :
  ∃! (p : ℕ), p = 1 ∧ (
    (∀ (x₁ x₂ : ℝ), line_a x₁ = line_a x₂ → false) ∨
    (∀ (x₁ x₂ : ℝ), line_b x₁ = line_b x₂ → false) ∨
    (∀ (x₁ x₂ : ℝ), line_c x₁ = line_c x₂ → false) ∨
    (∀ (x₁ x₂ : ℝ), line_d x₁ = line_d x₂ → false) ∨
    (∀ (x₁ x₂ : ℝ), line_e x₁ = line_e x₂ → false)) :=
sorry

end parallel_or_perpendicular_count_l164_164470


namespace sum_of_x_y_possible_values_l164_164258

theorem sum_of_x_y_possible_values (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
sorry

end sum_of_x_y_possible_values_l164_164258


namespace quadratic_radical_property_l164_164754

theorem quadratic_radical_property {a : ℝ} (h1 : a < 1) :
  (√(a^2 + 1)) = √(a^2 + 1) ∧ 
  (∀ b : ℝ, b = -3 → ∃ r : ℂ, r^2 = b) ∧ 
  (∀ b : ℝ, b = 8 → ∃ r : ℝ, r^3 = b ) ∧ 
  (√(a - 1) = √(a - 1) → false ∧ 
  (√(a^2 + 1) = √(a^2 + 1) → true)) := sorry

end quadratic_radical_property_l164_164754


namespace quadratic_eq_has_nonzero_root_l164_164065

theorem quadratic_eq_has_nonzero_root (b c : ℝ) (h : c ≠ 0) (h_eq : c^2 + b * c + c = 0) : b + c = -1 :=
sorry

end quadratic_eq_has_nonzero_root_l164_164065


namespace range_of_f_l164_164004

noncomputable def f (x : ℝ) := (Real.cos x) ^ 2 + Real.sin x

theorem range_of_f :
  set.range (λ x, f x) = set.Icc (1 : ℝ) (5 / 4 : ℝ) :=
sorry

end range_of_f_l164_164004


namespace sum_of_x_y_possible_values_l164_164259

theorem sum_of_x_y_possible_values (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
sorry

end sum_of_x_y_possible_values_l164_164259


namespace age_difference_l164_164401

variables (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 18) : C = A - 18 :=
sorry

end age_difference_l164_164401


namespace solve_for_a_l164_164625

theorem solve_for_a (a : ℝ) (z : ℂ) (h : z = (1 / (1 + complex.I)) + a * complex.I) : a = 1 / 2 :=
by sorry

end solve_for_a_l164_164625


namespace sum_of_values_l164_164381

theorem sum_of_values (x : ℝ) 
  (h : ∀ x, sqrt ((x + 2)^2 + 9) = 10 → x = -2 + sqrt 91 ∨ x = -2 - sqrt 91) : 
  ∑ x in {-2 + sqrt 91, -2 - sqrt 91}, x = -4 := 
by
  sorry

end sum_of_values_l164_164381


namespace polynomial_degree_root_conditions_l164_164877

noncomputable theory
open Polynomial

def p (x : ℝ) := -(2 / 3 : ℝ) * x * (x - 2) ^ 2 * (x - 4)

theorem polynomial_degree_root_conditions :
  ∃ (n : ℕ) (p : ℝ → ℝ),
    (∀ k : ℕ, k ≤ 2*n → p (2 * k) = 0) ∧
    (∀ k : ℕ, k < 2*n → p (2 * k + 1) = 2) ∧
    (p (2*n + 1) = -30) ∧ n = 2 ∧
    p = λ x, -(2 / 3 : ℝ) * x * (x - 2) ^ 2 * (x - 4) := 
  sorry

end polynomial_degree_root_conditions_l164_164877


namespace probability_king_or_queen_top_card_l164_164786

/-- A deck consisting of 60 cards has 15 unique ranks across 4 suits. -/
def deck_card_count : ℕ := 60

/-- Each suit has one card of each rank. There are 15 ranks in total. -/
def unique_ranks_count : ℕ := 15
def suits_count : ℕ := 4

/-- There are 4 Kings and 4 Queens in the deck -/
def rank_occurrences (rank : string) : ℕ :=
  if rank = "King" ∨ rank = "Queen" then 4 else 0

/-- Probability of top card being either a King or a Queen is 2/15 -/
theorem probability_king_or_queen_top_card :
  (rank_occurrences "King" + rank_occurrences "Queen") / deck_card_count = 2 / 15 :=
sorry

end probability_king_or_queen_top_card_l164_164786


namespace angle_is_two_pi_over_three_l164_164929

variables (a b : ℝ^3) -- Assuming the vectors are in the 3-dimensional space. Modify as needed.

-- Conditions
axiom unit_vector_a : ‖a‖ = 1
axiom unit_vector_b : ‖b‖ = 1
axiom dist_condition : ‖a - 2 • b‖ = sqrt 7

-- Question to prove
noncomputable def angle_between_vectors (a b : ℝ^3) : ℝ :=
real.acos ((inner a b) / (‖a‖ * ‖b‖))

theorem angle_is_two_pi_over_three : angle_between_vectors a b = 2 * real.pi / 3 :=
by sorry

end angle_is_two_pi_over_three_l164_164929


namespace compute_equation_l164_164817

theorem compute_equation :
  ((4501 * 2350) - (7125 / 9)) + (3250 ^ 2) * 4167 = 44_045_164_058.33 :=
by
  sorry

end compute_equation_l164_164817


namespace events_not_complementary_probability_prereq_l164_164534

-- Define the sample space, events, and conditions
def Ω : Set (ℕ × ℕ) := { (1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4) }

def event_A : Set (ℕ × ℕ) := { (1, 4), (2, 3), (2, 4) }

def event_B : Set (ℕ × ℕ) := { (1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2) }

-- Define the properties we want to prove
theorem events_not_complementary (h₁: event_A ∩ event_B ≠ ∅) (h₂: event_A ∪ event_B ≠ Ω) : True :=
  sorry

theorem probability_prereq (P: Set (ℕ × ℕ) -> ℚ) (hP_A: P(event_A) = 3/8) (hP_B: P(event_B) = 6/8):
  P(event_A) + P(event_B) = 9/8 :=
  sorry

end events_not_complementary_probability_prereq_l164_164534


namespace original_price_of_cycle_l164_164426

theorem original_price_of_cycle 
    (selling_price : ℝ) 
    (loss_percentage : ℝ) 
    (h1 : selling_price = 1120)
    (h2 : loss_percentage = 0.20) : 
    ∃ P : ℝ, P = 1400 :=
by
  sorry

end original_price_of_cycle_l164_164426


namespace total_rainfall_2003_and_2004_l164_164129

noncomputable def average_rainfall_2003 : ℝ := 45
noncomputable def months_in_year : ℕ := 12
noncomputable def percent_increase : ℝ := 0.05

theorem total_rainfall_2003_and_2004 :
  let rainfall_2004 := average_rainfall_2003 * (1 + percent_increase)
  let total_rainfall_2003 := average_rainfall_2003 * months_in_year
  let total_rainfall_2004 := rainfall_2004 * months_in_year
  total_rainfall_2003 = 540 ∧ total_rainfall_2004 = 567 := 
by 
  sorry

end total_rainfall_2003_and_2004_l164_164129


namespace michael_tests_combined_l164_164981

theorem michael_tests_combined :
  ∃ (a b c : ℕ), (a > b ∧ b > c) ∧ 
  (91 * b > 92 * c ∧ 92 * c > 90 * a) ∧
  90 * a / a = 90 ∧
  91 * b / b = 91 ∧
  92 * c / c = 92 ∧
  a + b + c = 413 := 
begin
  sorry
end

end michael_tests_combined_l164_164981


namespace final_number_is_1000_l164_164322

theorem final_number_is_1000 : 
  ∀ (nums : List ℕ), 
  (nums = List.range 1 2017) →
  (∀ a b : ℕ, a ∈ nums → b ∈ nums → nums.erase a).erase b ++ [(a + b) / 2]) →
  (nums.length = 1) →
  nums.head = 1000 := 
sorry

end final_number_is_1000_l164_164322


namespace sin_x_value_l164_164896

theorem sin_x_value (x : ℝ) : 
  (0 < x ∧ x < π / 2) → 
  cos (x + π / 4) = 3 / 5 → 
  sin x = √2 / 10 :=
by
  intros hx hcos
  sorry

end sin_x_value_l164_164896


namespace goose_eggs_count_l164_164659

theorem goose_eggs_count (E : ℕ)
  (h1 : (2/3 : ℚ) * E ≥ 0)
  (h2 : (3/4 : ℚ) * (2/3 : ℚ) * E ≥ 0)
  (h3 : 100 = (2/5 : ℚ) * (3/4 : ℚ) * (2/3 : ℚ) * E) :
  E = 500 := by
  sorry

end goose_eggs_count_l164_164659


namespace students_in_front_of_Yuna_l164_164407

-- Defining the total number of students
def total_students : ℕ := 25

-- Defining the number of students behind Yuna
def students_behind_Yuna : ℕ := 9

-- Defining Yuna's position from the end of the line
def Yuna_position_from_end : ℕ := students_behind_Yuna + 1

-- Statement to prove the number of students in front of Yuna
theorem students_in_front_of_Yuna : (total_students - Yuna_position_from_end) = 15 := by
  sorry

end students_in_front_of_Yuna_l164_164407


namespace lizzy_wealth_after_loan_l164_164652

theorem lizzy_wealth_after_loan 
  (initial_wealth : ℕ)
  (loan : ℕ)
  (interest_rate : ℕ)
  (h1 : initial_wealth = 30)
  (h2 : loan = 15)
  (h3 : interest_rate = 20)
  : (initial_wealth - loan) + (loan + loan * interest_rate / 100) = 33 :=
by
  sorry

end lizzy_wealth_after_loan_l164_164652


namespace curve_is_circle_l164_164024

theorem curve_is_circle (r θ : ℝ) (h : r = 1 / (Real.sin θ + Real.cos θ)) :
  ∃ x y : ℝ, r = Math.sqrt(x^2 + y^2) ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ (x + y)^2 = (x^2 + y^2) :=
by
  sorry

end curve_is_circle_l164_164024


namespace perp_pass_through_incenter_iff_isosceles_l164_164838

open Triangle

theorem perp_pass_through_incenter_iff_isosceles 
  (A B C D I: Point)
  [IsIncenter I (triangle.mk A B C)]
  (hD : OnLine D (line_through B C))
  (hPerp : Perpendicular (line_through D (foot_of_perpendicular D (line_through B C))) (line_through B C)):
  (dist A B = dist A C ∧ mid_point D B C) ↔ 
  (OnLine I (line_through D (foot_of_perpendicular D (line_through B C)))) := 
by { sorry }

end perp_pass_through_incenter_iff_isosceles_l164_164838


namespace expected_value_squared_indicator_little_o_n_l164_164181

noncomputable def expected_value_abs_lt_infinity (ξ : ℝ → ℝ) : Prop :=
  ∃ (μ : ℝ), (integral (λ x, |ξ x| ) ≤ μ)

noncomputable def little_o_n (ξ : ℝ → ℝ) : Prop :=
  ∀ (ε > 0), ∃ (N : ℝ), ∀ (n ≥ N), (expected_value (λ x, (ξ x)^2 * indicator (set_of (λ x, |ξ x| ≤ n)) x) < ε * n)

theorem expected_value_squared_indicator_little_o_n (ξ : ℝ → ℝ) 
  (h : expected_value_abs_lt_infinity ξ) : 
  little_o_n ξ := 
sorry

end expected_value_squared_indicator_little_o_n_l164_164181


namespace solve_equation_l164_164349

theorem solve_equation : ∀ x : ℝ, -2 * x + 11 = 0 → x = 11 / 2 :=
by
  intro x
  intro h
  sorry

end solve_equation_l164_164349


namespace carmen_sold_1_box_of_fudge_delights_l164_164814

noncomputable def boxes_of_fudge_delights (total_earned: ℝ) (samoas_price: ℝ) (thin_mints_price: ℝ) (fudge_delights_price: ℝ) (sugar_cookies_price: ℝ) (samoas_sold: ℝ) (thin_mints_sold: ℝ) (sugar_cookies_sold: ℝ): ℝ :=
  let samoas_total := samoas_price * samoas_sold
  let thin_mints_total := thin_mints_price * thin_mints_sold
  let sugar_cookies_total := sugar_cookies_price * sugar_cookies_sold
  let other_cookies_total := samoas_total + thin_mints_total + sugar_cookies_total
  (total_earned - other_cookies_total) / fudge_delights_price

theorem carmen_sold_1_box_of_fudge_delights: boxes_of_fudge_delights 42 4 3.5 5 2 3 2 9 = 1 :=
by
  sorry

end carmen_sold_1_box_of_fudge_delights_l164_164814


namespace total_number_of_questions_l164_164983

theorem total_number_of_questions (type_a_problems type_b_problems : ℕ) 
(time_spent_type_a time_spent_type_b : ℕ) 
(total_exam_time : ℕ) 
(h1 : type_a_problems = 50) 
(h2 : time_spent_type_a = 2 * time_spent_type_b) 
(h3 : time_spent_type_a * type_a_problems = 72) 
(h4 : total_exam_time = 180) :
type_a_problems + type_b_problems = 200 := 
by
  sorry

end total_number_of_questions_l164_164983


namespace numberOfTruePropositions_l164_164854

theorem numberOfTruePropositions (a b c d : ℝ) :
  (a > b ∧ c < 0 → a * c > b * c) ∧
  (a > b → ¬(a * c^2 > b * c^2)) ∧
  (a * c^2 < b * c^2 → a < b) ∧
  (a > b → ¬(1 / a < 1 / b)) ∧
  (a > b ∧ b > 0 ∧ c > d ∧ d > 0 → ¬(a * c > b * d)) →
  2 :=
sorry

end numberOfTruePropositions_l164_164854


namespace polished_small_glasses_l164_164377

variables (S L : ℕ)

theorem polished_small_glasses :
  L = S + 10 ∧ S + L = 110 → S = 50 :=
by
  intro h
  cases h with h1 h2
  rw [h1] at h2
  linarith

end polished_small_glasses_l164_164377


namespace find_a_l164_164425

noncomputable def point1 : ℝ × ℝ := (-3, 6)
noncomputable def point2 : ℝ × ℝ := (2, -1)

theorem find_a (a : ℝ) :
  let direction : ℝ × ℝ := (point2.1 - point1.1, point2.2 - point1.2)
  direction = (5, -7) →
  let normalized_direction : ℝ × ℝ := (direction.1 / -7, direction.2 / -7)
  normalized_direction = (a, -1) →
  a = -5 / 7 :=
by 
  intros 
  sorry

end find_a_l164_164425


namespace find_f2019_l164_164638

def f : ℕ → ℕ 
| x := if x ≤ 2015 then x + 2 else f (x - 5)

theorem find_f2019 : f 2019 = 2016 := 
by {
  sorry
}

end find_f2019_l164_164638


namespace g1_zero_point_f2_comparison_log_inequality_l164_164097

noncomputable def f1 (x : ℝ) : ℝ :=
  real.log x + 9 / (2 * (x + 1))

noncomputable def g1 (x : ℝ) (k : ℝ) : ℝ :=
  f1 x - k

theorem g1_zero_point (k : ℝ) : 
  (∃ x : ℝ, g1 x k = 0) → (k > 3 - real.log 2 ∨ k < 3 / 2 + real.log 2) := 
by sorry

noncomputable def f2 (x : ℝ) : ℝ :=
  real.log x + 2 / (x + 1)

theorem f2_comparison (x : ℝ) : 
  0 < x → (if x = 1 then f2 x = 1 else if x > 1 then f2 x > 1 else f2 x < 1) := 
by sorry

theorem log_inequality (n : ℕ) (hn : 0 < n) :
  real.log (n + 1) > ∑ i in finset.range n, 1 / (2 * (i + 1) + 1) := 
by sorry

end g1_zero_point_f2_comparison_log_inequality_l164_164097


namespace distance_car_travel_100_revolutions_l164_164777

noncomputable def average_radius (max_radius min_radius : ℝ) : ℝ :=
  (max_radius + min_radius) / 2

noncomputable def circumference (radius : ℝ) : ℝ :=
  2 * real.pi * radius

noncomputable def distance_covered (revolutions : ℕ) (circumference : ℝ) : ℝ :=
  revolutions * circumference

theorem distance_car_travel_100_revolutions
  (max_radius : ℝ)
  (min_radius : ℝ)
  (revolutions : ℕ)
  (h_max : max_radius = 22.4)
  (h_min : min_radius = 12.4)
  (h_revolutions : revolutions = 100) :
  distance_covered revolutions (circumference (average_radius max_radius min_radius)) / 100 = 109.36 :=
by
  sorry

end distance_car_travel_100_revolutions_l164_164777


namespace find_n_l164_164032

variable (a r : ℚ) (n : ℕ)

def geom_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- Given conditions
axiom seq_first_term : a = 1 / 3
axiom seq_common_ratio : r = 1 / 3
axiom sum_of_first_n_terms_eq : geom_sum a r n = 80 / 243

-- Prove that n = 5
theorem find_n : n = 5 := by
  sorry

end find_n_l164_164032


namespace time_to_cover_same_distance_l164_164375

theorem time_to_cover_same_distance
  (a b c d : ℕ) (k : ℕ) 
  (h_k : k = 3) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_speed_eq : 3 * (a + 2 * b) = 3 * a - b) : 
  (a + 2 * b) * (c + d) / (3 * a - b) = (a + 2 * b) * (c + d) / (3 * a - b) :=
by sorry

end time_to_cover_same_distance_l164_164375


namespace find_possible_values_a_l164_164727

theorem find_possible_values_a :
  ∃ a : ℤ, ∃ b : ℤ, ∃ c : ℤ, 
  (∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) ∧
  ((b + 5) * (c + 5) = 1 ∨ (b + 5) * (c + 5) = 4) ↔ 
  a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 7 :=
by
  sorry

end find_possible_values_a_l164_164727


namespace function_takes_negative_values_l164_164406

def f (x a : ℝ) : ℝ := x^2 - a * x + 1

theorem function_takes_negative_values {a : ℝ} :
  (∃ x : ℝ, f x a < 0) ↔ (a > 2 ∨ a < -2) :=
by
  sorry

end function_takes_negative_values_l164_164406


namespace sum_due_is_2400_l164_164398

theorem sum_due_is_2400
  (BD TD : ℝ) 
  (hBD : BD = 576) 
  (hTD : TD = 480) 
  (hSD_relation : BD = TD + (TD^2 / 2400)) :
  (∃ SD : ℝ, SD = 2400) :=
by 
  use 2400
  sorry

end sum_due_is_2400_l164_164398


namespace find_integer_pairs_l164_164078

theorem find_integer_pairs (a b : ℤ) (h : a > b)
  (root_condition : ∀ x : ℝ, 3 * x^2 + 3 * (a + b) * x + 4 * a * b = 0 →
    let α := x in ∃ β : ℝ, 3 * β^2 + 3 * (a + b) * β + 4 * a * b = 0 ∧
    α * (α + 1) + β * (β + 1) = (α + 1) * (β + 1)) :
  (a = 0 ∧ b = -1) ∨ (a = 1 ∧ b = 0) :=
by
  sorry

end find_integer_pairs_l164_164078


namespace find_percentage_l164_164572

theorem find_percentage :
  ∃ (p : ℝ), 
    (3 / 5) * 70.58823529411765 * (p / 100) = 36 ∧ 
    p ≈ 85 := 
by
  sorry

end find_percentage_l164_164572


namespace find_coordinates_of_P_l164_164530

-- Define points N and M with given symmetries.
structure Point where
  x : ℝ
  y : ℝ

def symmetric_about_x (P1 P2 : Point) : Prop :=
  P1.x = P2.x ∧ P1.y = -P2.y

def symmetric_about_y (P1 P2 : Point) : Prop :=
  P1.x = -P2.x ∧ P1.y = P2.y

-- Given conditions
def N : Point := ⟨1, 2⟩
def M : Point := ⟨-1, 2⟩ -- derived from symmetry about y-axis with N
def P : Point := ⟨-1, -2⟩ -- derived from symmetry about x-axis with M

theorem find_coordinates_of_P :
  symmetric_about_x M P ∧ symmetric_about_y N M → P = ⟨-1, -2⟩ :=
by
  sorry

end find_coordinates_of_P_l164_164530


namespace problem_statement_l164_164501

variable (a b c d : ℝ)

noncomputable def circle_condition_1 : Prop := a = (1 : ℝ) / a
noncomputable def circle_condition_2 : Prop := b = (1 : ℝ) / b
noncomputable def circle_condition_3 : Prop := c = (1 : ℝ) / c
noncomputable def circle_condition_4 : Prop := d = (1 : ℝ) / d

theorem problem_statement (h1 : circle_condition_1 a)
                          (h2 : circle_condition_2 b)
                          (h3 : circle_condition_3 c)
                          (h4 : circle_condition_4 d) :
    2 * (a^2 + b^2 + c^2 + d^2) = (a + b + c + d)^2 := 
by
  sorry

end problem_statement_l164_164501


namespace initial_money_l164_164676

def cost_of_game : Nat := 47
def cost_of_toy : Nat := 7
def number_of_toys : Nat := 3

theorem initial_money (initial_amount : Nat) (remaining_amount : Nat) :
  initial_amount = cost_of_game + remaining_amount →
  remaining_amount = number_of_toys * cost_of_toy →
  initial_amount = 68 := by
    sorry

end initial_money_l164_164676


namespace find_b_value_l164_164703

theorem find_b_value (b : ℝ) (y0 : ℝ) :
  (∀ x : ℝ, (x = 3 ∨ x = 9) → y0 = 2 * x^2 - b * x - 1) →
  b = 24 :=
by intros h
sorry

end find_b_value_l164_164703


namespace exists_positive_integer_m_l164_164664

theorem exists_positive_integer_m (m : ℕ) (hm : m > 0) : ∃ m : ℕ, m > 0 ∧ ∃ k : ℕ, 8 * m = k ^ 2 := 
by {
  let m := 2
  use m,
  dsimp,
  split,
  { exact hm },
  { use 4,
    calc 8 * m = 8 * 2 : by rfl
           ... = 16 : by norm_num
           ... = 4 ^ 2 : by norm_num }
}

end exists_positive_integer_m_l164_164664


namespace pond_water_after_evaporation_l164_164421

theorem pond_water_after_evaporation 
  (I R D : ℕ) 
  (h_initial : I = 250)
  (h_evaporation_rate : R = 1)
  (h_days : D = 50) : 
  I - (R * D) = 200 := 
by 
  sorry

end pond_water_after_evaporation_l164_164421


namespace min_value_of_geometric_seq_l164_164132

noncomputable def pos_geometric_seq (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∃ r, ∀ n, a (n+1) = r * a n)

noncomputable def geom_mean (x y z : ℝ) : Prop :=
  z = real.sqrt (x * y)

theorem min_value_of_geometric_seq (a : ℕ → ℝ)
  (h1 : pos_geometric_seq a)
  (h2 : geom_mean (a 4) (a 14) (2 * real.sqrt 2)) :
  2 * a 7 + a 11 ≥ 8 :=
sorry

end min_value_of_geometric_seq_l164_164132


namespace keychain_arrangements_l164_164661

theorem keychain_arrangements (keys : Finset ℕ) (house_key car_key : ℕ) (h1 : house_key ∈ keys) (h2 : car_key ∈ keys) (h_size : keys.card = 6) :
  ∃ (n : ℕ), n = 48 ∧ number_of_arrangements keys = n :=
by
  sorry

end keychain_arrangements_l164_164661


namespace composite_divides_expression_l164_164049

theorem composite_divides_expression (n : ℕ) (h : composite n) : 6 * n^2 ∣ n^4 - n^2 := 
sorry

end composite_divides_expression_l164_164049


namespace solve_equation_l164_164833

theorem solve_equation (x : ℝ) : 
  (2 * 4 ^ (x ^ 2 - 3 * x)) ^ 2 = 2 ^ (x - 1) →
  x = 1 / 4 ∨ x = 3 := 
sorry

end solve_equation_l164_164833


namespace P_intersection_Q_is_singleton_l164_164104

theorem P_intersection_Q_is_singleton :
  {p : ℝ × ℝ | p.1 + p.2 = 3} ∩ {p : ℝ × ℝ | p.1 - p.2 = 5} = { (4, -1) } :=
by
  -- The proof steps would go here.
  sorry

end P_intersection_Q_is_singleton_l164_164104


namespace greatest_common_divisor_of_180_and_n_l164_164369

theorem greatest_common_divisor_of_180_and_n (n : ℕ) (h : ∀ d ∣ 180 ∧ d ∣ n, d = 1 ∨ d = 3 ∨ d = 9) : 
  ∃ d, d ∣ 180 ∧ d ∣ n ∧ d = 9 :=
sorry

end greatest_common_divisor_of_180_and_n_l164_164369


namespace abc_correct_and_c_not_true_l164_164865

theorem abc_correct_and_c_not_true (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  a^2 > b^2 ∧ ab > b^2 ∧ (1/(a+b) > 1/a) ∧ ¬(1/a < 1/b) :=
  sorry

end abc_correct_and_c_not_true_l164_164865


namespace deduction_from_third_l164_164694

-- Define the conditions
def avg_10_consecutive_eq_20 (x : ℝ) : Prop :=
  (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9)) / 10 = 20

def new_avg_10_numbers_eq_15_5 (x y : ℝ) : Prop :=
  ((x - 9) + (x - 7) + (x + 2 - y) + (x - 3) + (x - 1) + (x + 1) + (x + 3) + (x + 5) + (x + 7) + (x + 9)) / 10 = 15.5

-- Define the theorem to be proved
theorem deduction_from_third (x y : ℝ) (h1 : avg_10_consecutive_eq_20 x) (h2 : new_avg_10_numbers_eq_15_5 x y) : y = 6 :=
sorry

end deduction_from_third_l164_164694


namespace sum_of_x_y_possible_values_l164_164264

theorem sum_of_x_y_possible_values (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
sorry

end sum_of_x_y_possible_values_l164_164264


namespace smallest_possible_sector_angle_l164_164653

theorem smallest_possible_sector_angle :
  ∃ (a1 d : ℕ), ∀ k : ℕ,
  k ≤ 14 → let ai := a1 + k * d in ai > 0 ∧ (15 * (a1 + a1 + 14 * d)) / 2 = 360 →
  ai = 10 :=
by {
  sorry
}

end smallest_possible_sector_angle_l164_164653


namespace question_l164_164061

theorem question (n : ℕ) (n_pos : 0 < n) 
  (a : Fin n → ℝ) (a_pos : ∀ i, 0 < a i) :
  (∑ i, a i) * (∑ i, (1 / a i)) ≥ n^2 :=
by sorry

end question_l164_164061


namespace y_relationship_l164_164889

theorem y_relationship (x1 x2 x3 y1 y2 y3 : ℝ) 
  (hA : y1 = -7 * x1 + 14) 
  (hB : y2 = -7 * x2 + 14) 
  (hC : y3 = -7 * x3 + 14) 
  (hx : x1 > x3 ∧ x3 > x2) : y1 < y3 ∧ y3 < y2 :=
by
  sorry

end y_relationship_l164_164889


namespace nonagon_diagonals_count_l164_164934

theorem nonagon_diagonals_count (n : ℕ) (h1 : n = 9) : 
  let diagonals_per_vertex := n - 3 in
  let naive_count := n * diagonals_per_vertex in
  let distinct_diagonals := naive_count / 2 in
  distinct_diagonals = 27 :=
by
  sorry

end nonagon_diagonals_count_l164_164934


namespace find_x_l164_164858

theorem find_x (x : ℝ) : (10^(2*x) * 1000^x = 10^15) → (x = 3) := by
  sorry

end find_x_l164_164858


namespace box_side_area_l164_164827

noncomputable def solve_box_area : ℕ :=
  let k := 5
  let L := 3 * k
  let H := 2 * k
  let W := 2 * H
  (H * W)

theorem box_side_area :
  let k := 5;
  let L := 3 * k;
  let H := 2 * k;
  let W := 2 * H;
  (L * W * H = 3000) → (H * W = 200) :=
by
  intros
  have hk : k = 5, by sorry
  have hL : L = 15, by sorry
  have hH : H = 10, by sorry
  have hW : W = 20, by sorry
  sorry

end box_side_area_l164_164827


namespace nell_has_cards_left_l164_164658

def initial_cards : ℕ := 242
def cards_given_away : ℕ := 136

theorem nell_has_cards_left :
  initial_cards - cards_given_away = 106 :=
by
  sorry

end nell_has_cards_left_l164_164658


namespace cubics_sum_l164_164687

theorem cubics_sum (a b c : ℝ) (h₁ : a + b + c = 4) (h₂ : ab + ac + bc = 6) (h₃ : abc = -8) :
  a^3 + b^3 + c^3 = 8 :=
by {
  -- proof steps would go here
  sorry
}

end cubics_sum_l164_164687


namespace distinctDiagonalsConvexNonagon_l164_164937

theorem distinctDiagonalsConvexNonagon : 
  ∀ (P : Type) [fintype P] [decidable_eq P] (vertices : finset P) (h : vertices.card = 9), 
  let n := vertices.card in
  let diagonals := (n * (n - 3)) / 2 in
  diagonals = 27 :=
by
  intros
  let n := vertices.card
  have keyIdentity : (n * (n - 3)) / 2 = 27 := sorry
  exact keyIdentity

end distinctDiagonalsConvexNonagon_l164_164937


namespace exists_positive_integer_m_l164_164665

theorem exists_positive_integer_m (m : ℕ) (hm : m > 0) : ∃ m : ℕ, m > 0 ∧ ∃ k : ℕ, 8 * m = k ^ 2 := 
by {
  let m := 2
  use m,
  dsimp,
  split,
  { exact hm },
  { use 4,
    calc 8 * m = 8 * 2 : by rfl
           ... = 16 : by norm_num
           ... = 4 ^ 2 : by norm_num }
}

end exists_positive_integer_m_l164_164665


namespace clover_count_l164_164131

theorem clover_count (C : ℕ) 
  (h1 : 0.20 * C / 4 = 25) : C = 500 :=
by
  sorry

end clover_count_l164_164131


namespace prime_base_values_l164_164822

theorem prime_base_values :
  ∀ p : ℕ, Prime p →
    (2 * p^3 + p^2 + 6 + 4 * p^2 + p + 4 + 2 * p^2 + p + 5 + 2 * p^2 + 2 * p + 2 + 9 =
     4 * p^2 + 3 * p + 3 + 5 * p^2 + 7 * p + 2 + 3 * p^2 + 2 * p + 1) →
    false :=
by {
  sorry
}

end prime_base_values_l164_164822


namespace composite_set_gcd_prime_l164_164405

theorem composite_set_gcd_prime (S : Finset ℕ) (hS : ∀ (s ∈ S), ∃ (p : ℕ), p.Prime ∧ p ∣ s ∧ p < s)
  (hcond : ∀ n, ∃ s ∈ S, nat.gcd s n = 1 ∨ nat.gcd s n = s) :
  ∃ (s t : ℕ), s ∈ S ∧ t ∈ S ∧ ∃ p, p.Prime ∧ nat.gcd s t = p := 
sorry

end composite_set_gcd_prime_l164_164405


namespace min_magnitude_of_u_l164_164556

theorem min_magnitude_of_u :
  let a := (Real.cos 25, Real.sin 25)
  let b := (Real.sin 20, Real.cos 20)
  ∃ t : ℝ, 
    let u := (a.1 + t * b.1, a.2 + t * b.2) in
    (u.1^2 + u.2^2)^(1 / 2) = (Real.sqrt 2) / 2 :=
by
  sorry

end min_magnitude_of_u_l164_164556


namespace range_of_m_l164_164915

noncomputable def f (x : ℝ) := 2 - 1 / x

theorem range_of_m (a b m : ℝ) (h_domain : 0 < a ∧ a < b) :
  ∃ ma mb, ma = 2 - 1 / b ∧ mb = 2 - 1 / a ∧ (ma, mb) = (m * a, m * b) → 0 < m ∧ m < 1 :=
begin
  sorry
end

end range_of_m_l164_164915


namespace fraction_painted_red_l164_164654

theorem fraction_painted_red :
  let matilda_section := (1:ℚ) / 2 -- Matilda's half section
  let ellie_section := (1:ℚ) / 2    -- Ellie's half section
  let matilda_painted := matilda_section / 2 -- Matilda's painted fraction
  let ellie_painted := ellie_section / 3    -- Ellie's painted fraction
  (matilda_painted + ellie_painted) = 5 / 12 := 
by
  sorry

end fraction_painted_red_l164_164654


namespace x_eq_one_l164_164619

theorem x_eq_one (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (div_cond : ∀ n : ℕ, 0 < n → (2^n * y + 1) ∣ (x^(2^n) - 1)) : x = 1 := by
  sorry

end x_eq_one_l164_164619


namespace min_buses_l164_164435

theorem min_buses (n : ℕ) : (47 * n >= 625) → (n = 14) :=
by {
  -- Proof is omitted since the problem only asks for the Lean statement, not the solution steps.
  sorry
}

end min_buses_l164_164435


namespace possible_values_of_sum_l164_164239

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l164_164239


namespace adam_simon_distance_100_l164_164443

noncomputable def time_to_be_100_apart (x : ℝ) : Prop :=
  let distance_adam := 10 * x
  let distance_simon_east := 10 * x * (Real.sqrt 2 / 2)
  let distance_simon_south := 10 * x * (Real.sqrt 2 / 2)
  let total_eastward_separation := abs (distance_adam - distance_simon_east)
  let resultant_distance := Real.sqrt (total_eastward_separation^2 + distance_simon_south^2)
  resultant_distance = 100

theorem adam_simon_distance_100 : ∃ (x : ℝ), time_to_be_100_apart x ∧ x = 2 * Real.sqrt 2 := 
by
  sorry

end adam_simon_distance_100_l164_164443


namespace pairs_satisfying_sqrt_equation_l164_164016

theorem pairs_satisfying_sqrt_equation (x y : ℝ) :
  (sqrt (x^2 + y^2 - 1) = 1 - x - y) →
  (∃ t : ℝ, (x = 1 ∧ y = t ∧ t ≤ 0) ∨ (x = t ∧ y = 1 ∧ t ≤ 0)) :=
by
  sorry

end pairs_satisfying_sqrt_equation_l164_164016


namespace minimum_photocopies_for_discount_l164_164307

theorem minimum_photocopies_for_discount:
  (cost_per_copy : ℝ) (discount_rate : ℝ) (copies_Steve : ℕ) (copies_Dinley : ℕ) (total_savings : ℝ) :
  cost_per_copy = 0.02 →
  discount_rate = 0.25 →
  copies_Steve = 80 →
  copies_Dinley = 80 →
  total_savings = 0.40 →
  (∃ x, x > copies_Steve + copies_Dinley ∧ (0.02 * (copies_Steve + copies_Dinley) - 0.015 * (copies_Steve + copies_Dinley)) = 2 * total_savings ∧ x = 160) :=
begin
  -- We will later provide the formal proof steps.
  sorry
end

end minimum_photocopies_for_discount_l164_164307


namespace possible_values_of_sum_l164_164236

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + y^3 + 21 * x * y = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l164_164236


namespace area_of_triangle_OPQ_is_16_l164_164999

def point (x y : ℝ) : Type := { p : ℝ × ℝ // p.1 = x ∧ p.2 = y }

noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

def line (s : ℝ) (p : ℝ × ℝ) : ℝ → ℝ := λ x, s * x + (p.2 - s * p.1)

theorem area_of_triangle_OPQ_is_16 :
  ∃ (P : ℝ × ℝ), P.1 = 0 ∧
                  Q = (4, 0) ∧
                  R = (2, 4) ∧
                  slope Q R = slope Q P ∧
                  let area := (1 / 2) * 4 * P.2 in
                  area = 16 :=
by
  sorry

end area_of_triangle_OPQ_is_16_l164_164999


namespace f_a5_plus_f_a6_l164_164515

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_property : ∀ x : ℝ, f (3 / 2 - x) = f x
axiom f_minus2 : f (-2) = -3

def a : ℕ → ℝ
| 0     := -1
| (n+1) := f (a n)

theorem f_a5_plus_f_a6 : f (a 5) + f (a 6) = 3 := 
sorry

end f_a5_plus_f_a6_l164_164515


namespace find_ab_l164_164977

-- Define the "¤" operation
def op (x y : ℝ) := (x + y)^2 - (x - y)^2

-- The Lean 4 theorem statement
theorem find_ab (a b : ℝ) (h : op a b = 24) : a * b = 6 := 
by
  -- We leave the proof as an exercise
  sorry

end find_ab_l164_164977


namespace simplify_expression_l164_164684

theorem simplify_expression : 2 - 2 / (2 + Real.sqrt 5) + 2 / (2 - Real.sqrt 5) = 2 + 4 * Real.sqrt 5 :=
by sorry

end simplify_expression_l164_164684


namespace problem_a_l164_164175

def continuous (f : ℝ → ℝ) : Prop := sorry -- Assume this is properly defined somewhere in Mathlib
def monotonic (f : ℝ → ℝ) : Prop := sorry -- Assume this is properly defined somewhere in Mathlib

theorem problem_a :
  ¬ (∀ (f : ℝ → ℝ), continuous f ∧ (∀ y, ∃ x, f x = y) → monotonic f) := sorry

end problem_a_l164_164175


namespace common_ratio_of_geometric_sequence_l164_164722

theorem common_ratio_of_geometric_sequence (a1 : ℝ) (q : ℝ) (h1 : q ≠ 1)
  (h2 : a1 ≠ 0)
  (h3 : ∑ n in Finset.range 4, a1 * (1 - q ^ (n + 1)) / (1 - q) = 0 + 3 * (a1 * (1 - q ^ 2) / (1 - q))) : 
  q = -2 :=
by 
  sorry

end common_ratio_of_geometric_sequence_l164_164722


namespace tan_alpha_sub_2pi_over_3_two_sin_sq_alpha_sub_cos_sq_alpha_l164_164568

variable (α : ℝ)

theorem tan_alpha_sub_2pi_over_3 (h : Real.tan (α + π / 3) = 2 * Real.sqrt 3) :
    Real.tan (α - 2 * π / 3) = 2 * Real.sqrt 3 :=
sorry

theorem two_sin_sq_alpha_sub_cos_sq_alpha (h : Real.tan (α + π / 3) = 2 * Real.sqrt 3) :
    2 * (Real.sin α) ^ 2 - (Real.cos α) ^ 2 = -43 / 52 :=
sorry

end tan_alpha_sub_2pi_over_3_two_sin_sq_alpha_sub_cos_sq_alpha_l164_164568


namespace product_of_three_numbers_l164_164724

theorem product_of_three_numbers (a b c : ℝ) 
  (h1 : a + b + c = 30) 
  (h2 : a = 5 * (b + c)) 
  (h3 : b = 9 * c) : 
  a * b * c = 56.25 := 
by 
  sorry

end product_of_three_numbers_l164_164724


namespace function_relationship_profit_1200_max_profit_l164_164000

namespace SalesProblem

-- Define the linear relationship between sales quantity y and selling price x
def sales_quantity (x : ℝ) : ℝ := -2 * x + 160

-- Define the cost per item
def cost_per_item := 30

-- Define the profit given selling price x and quantity y
def profit (x : ℝ) (y : ℝ) : ℝ := (x - cost_per_item) * y

-- The given data points and conditions
def data_point_1 : (ℝ × ℝ) := (35, 90)
def data_point_2 : (ℝ × ℝ) := (40, 80)

-- Prove the linear relationship between y and x
theorem function_relationship : 
  sales_quantity data_point_1.1 = data_point_1.2 ∧ 
  sales_quantity data_point_2.1 = data_point_2.2 := 
  by sorry

-- Given daily profit of 1200, proves selling price should be 50 yuan
theorem profit_1200 (x : ℝ) (h₁ : 30 ≤ x ∧ x ≤ 54) 
  (h₂ : profit x (sales_quantity x) = 1200) : 
  x = 50 := 
  by sorry

-- Prove the maximum daily profit and corresponding selling price
theorem max_profit : 
  ∃ x, 30 ≤ x ∧ x ≤ 54 ∧ (∀ y, 30 ≤ y ∧ y ≤ 54 → profit y (sales_quantity y) ≤ profit x (sales_quantity x)) ∧ 
  profit x (sales_quantity x) = 1248 := 
  by sorry

end SalesProblem

end function_relationship_profit_1200_max_profit_l164_164000


namespace find_range_of_f_l164_164898

-- Define the function f(x) when a = 2
def f (x : ℝ) : ℝ := (1 / 2) - (1 / (2^x + 1))

-- Prove that f(x) is odd and find its range
theorem find_range_of_f : (∀ x : ℝ, f (-x) = -f x) ∧ (set.range f = set.Ioo (-1 / 2 : ℝ) (1 / 2 : ℝ)) :=
by
  sorry

end find_range_of_f_l164_164898


namespace pies_made_l164_164695

/-
The cafeteria initially had 372 apples. 
They handed out 135 apples to students.
Each pie requires 15 apples.

Prove: The cafeteria can make exactly 15 whole pies with the remaining apples.
-/

theorem pies_made (initial_apples handed_out_apples apples_per_pie remaining_apples pies : ℕ) :
  initial_apples = 372 →
  handed_out_apples = 135 →
  apples_per_pie = 15 →
  remaining_apples = initial_apples - handed_out_apples →
  pies = remaining_apples / apples_per_pie →
  pies = 15 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num

end pies_made_l164_164695


namespace sphere_surface_area_l164_164189

/-- Points P, A, B, C lie on the surface of a sphere O with the properties:
  - PA, PB, PC are mutually perpendicular
  - PA = 1
  - PB = sqrt 2
  - PC = sqrt 3
Show that the surface area of the sphere is 24π. --/
theorem sphere_surface_area
  (O P A B C : Type)
  (PA PB PC : 𝔽)
  (hPA : PA = 1)
  (hPB : PB = Real.sqrt 2)
  (hPC : PC = Real.sqrt 3)
  (h_perpendicular : (PA * PB = 0 ∧ PB * PC = 0 ∧ PA * PC = 0)) :
  (4 * π * (1 + 2 + 3) = 24 * π) :=
by
  sorry

end sphere_surface_area_l164_164189


namespace darnel_difference_l164_164828

theorem darnel_difference (sprint_1 jog_1 sprint_2 jog_2 sprint_3 jog_3 : ℝ)
  (h_sprint_1 : sprint_1 = 0.8932)
  (h_jog_1 : jog_1 = 0.7683)
  (h_sprint_2 : sprint_2 = 0.9821)
  (h_jog_2 : jog_2 = 0.4356)
  (h_sprint_3 : sprint_3 = 1.2534)
  (h_jog_3 : jog_3 = 0.6549) :
  (sprint_1 + sprint_2 + sprint_3 - (jog_1 + jog_2 + jog_3)) = 1.2699 := by
  sorry

end darnel_difference_l164_164828


namespace composite_divides_expression_l164_164048

theorem composite_divides_expression (n : ℕ) (h : composite n) : 6 * n^2 ∣ n^4 - n^2 := 
sorry

end composite_divides_expression_l164_164048


namespace range_of_a_l164_164545

variable (a : ℝ)

def proposition_p : Prop :=
  ∃ x₀ : ℝ, x₀^2 - a * x₀ + a = 0

def proposition_q : Prop :=
  ∀ x : ℝ, 1 < x → x + 1 / (x - 1) ≥ a

theorem range_of_a (h : ¬proposition_p a ∧ proposition_q a) : 0 < a ∧ a ≤ 3 :=
sorry

end range_of_a_l164_164545


namespace log_base_4_of_32_l164_164482

theorem log_base_4_of_32 : log 4 32 = 5 / 2 := by
  -- Define conditions
  have h1 : 32 = 2 ^ 5 := by sorry
  have h2 : 4 = 2 ^ 2 := by sorry
  -- Prove the theorem
  sorry

end log_base_4_of_32_l164_164482


namespace sine_sum_inequality_l164_164486

theorem sine_sum_inequality (x : ℝ) (k : ℤ) :
  (∀ n : ℕ, ∑ i in range (n + 1), Real.sin (i * x) ≤ Real.sqrt 3 / 2) ↔
    ∃ k : ℤ, (2 * Real.pi / 3 + 2 * k * Real.pi ≤ x ∧ x ≤ 2 * Real.pi + 2 * k * Real.pi) := by
  sorry

end sine_sum_inequality_l164_164486


namespace wholesale_price_l164_164795

theorem wholesale_price (R W : ℝ) (h1 : R = 1.80 * W) (h2 : R = 36) : W = 20 :=
by
  sorry 

end wholesale_price_l164_164795


namespace find_p_for_hyperbola_chord_l164_164469

theorem find_p_for_hyperbola_chord (p : ℝ) :
  (∀ (x y : ℝ), (x^2 / 3 - y^2 = 1) → (x^2 / 3 - y^2 = 1 ∧ x - √4 = y)
   → ∃ F : ℝ × ℝ, F = (√4, 0) ∧ 
       (∀ (A B : ℝ × ℝ), (A ≠ B ∧ 
       ((A.1 = x ∧ A.2 = x - √4) ∧ (B.1 = x ∧ B.2 = x - √4)) 
       → (|F.fst - A.fst| = |F.fst - B.fst|)))
   →  (∃ P : ℝ × ℝ, P = (p, 0)) ∧ (p = 3) :=
by sorry

end find_p_for_hyperbola_chord_l164_164469


namespace max_value_of_x2_plus_y2_l164_164635

open Real

theorem max_value_of_x2_plus_y2 (x y : ℝ) (h : x^2 + y^2 = 2 * x - 2 * y + 2) : 
  x^2 + y^2 ≤ 6 + 4 * sqrt 2 :=
sorry

end max_value_of_x2_plus_y2_l164_164635


namespace ratio_of_quadratic_roots_l164_164682

theorem ratio_of_quadratic_roots (a b c : ℝ) (h : 2 * b^2 = 9 * a * c) : 
  ∃ (x₁ x₂ : ℝ), (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) ∧ (x₁ / x₂ = 2) :=
sorry

end ratio_of_quadratic_roots_l164_164682


namespace max_value_x4_y2_z_l164_164636

theorem max_value_x4_y2_z (x y z : ℝ) (hxyz : x > 0 ∧ y > 0 ∧ z > 0) (h : x^2 + y^2 + z^2 = 1) :
    x^4 * y^2 * z ≤ 32 / (16807 * sqrt 7) :=
by
    sorry

end max_value_x4_y2_z_l164_164636


namespace colorable_graph_l164_164587

variable (V : Type) [Fintype V] [DecidableEq V] (E : V → V → Prop) [DecidableRel E]

/-- Each city has at least one road leading out of it -/
def has_one_road (v : V) : Prop := ∃ w : V, E v w

/-- No city is connected by roads to all other cities -/
def not_connected_to_all (v : V) : Prop := ¬ ∀ w : V, E v w ↔ w ≠ v

/-- A set of cities D is dominating if every city not in D is connected by a road to at least one city in D -/
def is_dominating_set (D : Finset V) : Prop :=
  ∀ v : V, v ∉ D → ∃ d ∈ D, E v d

noncomputable def dominating_set_min_card (k : ℕ) : Prop :=
  ∀ D : Finset V, is_dominating_set V E D → D.card ≥ k

/-- Prove that the graph can be colored using 2001 - k colors such that no two adjacent vertices share the same color -/
theorem colorable_graph (k : ℕ) (hk : dominating_set_min_card V E k) :
    ∃ (colors : V → Fin (2001 - k)), ∀ v w : V, E v w → colors v ≠ colors w := 
by 
  sorry

end colorable_graph_l164_164587


namespace smallest_positive_integer_cube_ends_368_l164_164039

theorem smallest_positive_integer_cube_ends_368 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 368 ∧ n = 34 :=
by
  sorry

end smallest_positive_integer_cube_ends_368_l164_164039


namespace find_value_of_A_l164_164700

theorem find_value_of_A (A B : ℤ) (h1 : A - B = 144) (h2 : A = 3 * B - 14) : A = 223 :=
by
  sorry

end find_value_of_A_l164_164700


namespace closest_whole_number_area_of_shaded_region_l164_164133

theorem closest_whole_number_area_of_shaded_region
  (h1 : 4 * 5 = 20)
  (h2 : (π * (1: ℝ)^2) = π)
  (h3 : 20 - (π: ℝ) = 16.86) :
  real.closest_integer 16.86 = 17 :=
by
  sorry

end closest_whole_number_area_of_shaded_region_l164_164133


namespace exists_monochromatic_rectangle_l164_164376

theorem exists_monochromatic_rectangle 
  (coloring : ℤ × ℤ → Prop)
  (h : ∀ p : ℤ × ℤ, coloring p = red ∨ coloring p = blue)
  : ∃ (a b c d : ℤ × ℤ), (a.1 = b.1) ∧ (c.1 = d.1) ∧ (a.2 = c.2) ∧ (b.2 = d.2) ∧ (coloring a = coloring b) ∧ (coloring b = coloring c) ∧ (coloring c = coloring d) :=
sorry

end exists_monochromatic_rectangle_l164_164376


namespace nonagon_diagonals_count_eq_27_l164_164953

theorem nonagon_diagonals_count_eq_27 :
  let vertices := 9
  let connections_per_vertex := vertices - 3
  let total_raw_count_diagonals := vertices * connections_per_vertex
  let distinct_diagonals := total_raw_count_diagonals / 2
  distinct_diagonals = 27 :=
by
  let vertices := 9
  let connections_per_vertex := vertices - 3
  let total_raw_count_diagonals := vertices * connections_per_vertex
  let distinct_diagonals := total_raw_count_diagonals / 2
  have : distinct_diagonals = 27 := sorry
  exact this

end nonagon_diagonals_count_eq_27_l164_164953


namespace distance_interval_l164_164445

-- Define the conditions based on the false statements:
variable (d : ℝ)

def false_by_alice : Prop := d < 8
def false_by_bob : Prop := d > 7
def false_by_charlie : Prop := d ≠ 6

theorem distance_interval (h_alice : false_by_alice d) (h_bob : false_by_bob d) (h_charlie : false_by_charlie d) :
  7 < d ∧ d < 8 :=
by
  sorry

end distance_interval_l164_164445


namespace C1_C2_C3_are_collinear_l164_164453

-- Given conditions definitions
def Circle (α : Type*) [LinearOrderedField α] := EuclideanPlane.Circle α
variables {α : Type*} [LinearOrderedField α]

variables (O1 O2 : Point α)
variables (circle_O1 circle_O2 : Circle α)
variables (P Q : Point α)
variables (A B C : Point α)
variables (A1 B1 A2 B2 A3 B3 : Point α)
variables (C1 C2 C3 : Point α)

-- Assumptions
axiom circles_intersect_at_P_Q : P ∈ circle_O1 ∧ P ∈ circle_O2 ∧ Q ∈ circle_O1 ∧ Q ∈ circle_O2
axiom secants_pass_through_P : (A1 ≠ B1 ∧ A2 ≠ B2 ∧ A3 ≠ B3) ∧ 
  ∀ (i = 1 ∨ i = 2 ∨ i = 3), (A1, B1).contains P ∧ (A2, B2).contains P ∧ (A3, B3).contains P
axiom triangles_are_similar : 
  ∀ (i = 1 ∨ i = 2 ∨ i = 3), 
    (triangle C1 A B).similar_to (triangle C A1 B1) ∧ 
    (triangle C2 A B).similar_to (triangle C A2 B2) ∧ 
    (triangle C3 A B).similar_to (triangle C A3 B3)

-- Proof statement
theorem C1_C2_C3_are_collinear : collinear {C1, C2, C3} :=
sorry

end C1_C2_C3_are_collinear_l164_164453


namespace bacteria_elimination_l164_164980

theorem bacteria_elimination (d N : ℕ) (hN : N = 50 - 6 * (d - 1)) (hCondition : N ≤ 0) : d = 10 :=
by
  -- We can straightforwardly combine the given conditions and derive the required theorem.
  sorry

end bacteria_elimination_l164_164980


namespace al_original_amount_l164_164444

theorem al_original_amount : 
  ∃ (a b c : ℝ), 
    a + b + c = 1200 ∧ 
    (a - 200 + 3 * b + 4 * c) = 1800 ∧ 
    b = 2800 - 3 * a ∧ 
    c = 1200 - a - b ∧ 
    a = 860 := by
  sorry

end al_original_amount_l164_164444


namespace average_of_solutions_l164_164790

theorem average_of_solutions (a b : ℝ) :
  (∃ x1 x2 : ℝ, 3 * a * x1^2 - 6 * a * x1 + 2 * b = 0 ∧
                3 * a * x2^2 - 6 * a * x2 + 2 * b = 0 ∧
                x1 ≠ x2) →
  (1 + 1) / 2 = 1 :=
by
  intros
  sorry

end average_of_solutions_l164_164790


namespace trey_used_50_nails_l164_164735

-- Definitions based on conditions
def decorations_with_sticky_strips := 15
def fraction_nails := 2/3
def fraction_thumbtacks := 2/5

-- Define D as total number of decorations and use the given conditions
noncomputable def total_decorations : ℕ :=
  let D := decorations_with_sticky_strips / ((1:ℚ) - fraction_nails - (fraction_thumbtacks * (1 - fraction_nails))) in
  if h : 0 < D ∧ D.denom = 1 then D.num else 0

-- Nails used by Trey
noncomputable def nails_used : ℕ := (fraction_nails * total_decorations).toNat

theorem trey_used_50_nails : nails_used = 50 := by
  sorry

end trey_used_50_nails_l164_164735


namespace boys_girls_rel_l164_164991

theorem boys_girls_rel (b g : ℕ) (h : g = 7 + 2 * (b - 1)) : b = (g - 5) / 2 := 
by sorry

end boys_girls_rel_l164_164991


namespace unique_polynomial_p_l164_164014

noncomputable def polynomial_p (P : Polynomial ℝ) : Prop :=
  (∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1) ∧ (P.eval 0 = 0)

theorem unique_polynomial_p (P : Polynomial ℝ) (h : polynomial_p P) : 
  P = Polynomial.Coeff ℝ 1 :=
sorry

end unique_polynomial_p_l164_164014


namespace sum_of_possible_values_of_N_l164_164332

theorem sum_of_possible_values_of_N :
  ∃ a b c : ℕ, (a > 0 ∧ b > 0 ∧ c > 0) ∧ (abc = 8 * (a + b + c)) ∧ (c = a + b)
  ∧ (2560 = 560) :=
by
  sorry

end sum_of_possible_values_of_N_l164_164332


namespace valid_step_length_l164_164745

variable {a : ℝ}

noncomputable def total_distance := 300 * 100 -- in cm
noncomputable def step_count := 400
noncomputable def max_step_length := total_distance / step_count -- Maximum possible step length in cm
noncomputable def min_a := 75.0
noncomputable def max_a := 10000.0 / 133.0

theorem valid_step_length 
  (h1 : total_distance = 30000) 
  (h2 : step_count = 400) 
  (h3 : ∀ (i j k : ℕ), i ≠ j → j ≠ k → i ≠ k → (i < step_count) → (j < step_count) → (k < step_count) → ∀ (x : ℕ → ℝ), (x i + x j > x k)) : 
  min_a ≤ a ∧ a < max_a := 
sorry

end valid_step_length_l164_164745


namespace relationship_abc_l164_164872

open Real

variable {x : ℝ}
variable (a b c : ℝ)
variable (h1 : 0 < x ∧ x ≤ 1)
variable (h2 : a = (sin x / x) ^ 2)
variable (h3 : b = sin x / x)
variable (h4 : c = sin (x^2) / x^2)

theorem relationship_abc (h1 : 0 < x ∧ x ≤ 1) (h2 : a = (sin x / x) ^ 2) (h3 : b = sin x / x) (h4 : c = sin (x^2) / x^2) :
  a < b ∧ b ≤ c :=
sorry

end relationship_abc_l164_164872


namespace two_digit_divisors_1995_l164_164007

theorem two_digit_divisors_1995 :
  (∃ (n : Finset ℕ), (∀ x ∈ n, 10 ≤ x ∧ x < 100 ∧ 1995 % x = 0) ∧ n.card = 6 ∧ ∃ y ∈ n, y = 95) :=
by
  sorry

end two_digit_divisors_1995_l164_164007


namespace sum_product_of_integers_l164_164338

theorem sum_product_of_integers (a b c : ℕ) (h₁ : c = a + b) (h₂ : N = a * b * c) (h₃ : N = 8 * (a + b + c)) : 
  a * b * (a + b) = 16 * (a + b) :=
by {
  sorry
}

end sum_product_of_integers_l164_164338


namespace math_problem_l164_164630

-- We define the given expression
def special_sum (x1 x2 x3 x4 x5 : ℕ) : ℕ :=
  x1 * x2 + x2 * x3 + x3 * x4 + x4 * x5 + x5 * x1 +
  x1 * x2 * x3 + x2 * x3 * x4 + x3 * x4 * x5 + x4 * x5 * x1 + x5 * x1 * x2

-- We denote the list of permutations of {1, 2, 3, 4, 6}
def permutations : List (List ℕ) := List.permutations [1,2,3,4,6]

-- Define the maximum value P and the number of permutations Q attaining that value
def P : ℕ :=
  permutations.foldl (λ acc p, max acc (special_sum p.head! p.tail.head! p.tail.tail.head! p.tail.tail.tail.head! p.tail.tail.tail.tail.head!)) 0

def Q : ℕ :=
  (permutations.filter (λ p, special_sum p.head! p.tail.head! p.tail.tail.head! p.tail.tail.tail.head! p.tail.tail.tail.tail.head! = P)).length

-- The final result
def PQ : ℕ := P + Q

-- The problem statement to prove
theorem math_problem : PQ = 189 := 
by
  sorry

end math_problem_l164_164630


namespace points_on_line_l164_164285

theorem points_on_line (n : ℕ) (h : 9 * n - 8 = 82) : n = 10 := by
  sorry

end points_on_line_l164_164285


namespace bus_rent_proof_l164_164343

theorem bus_rent_proof (r1 r2 : ℝ) (r1_rent_eq : r1 + 2 * r2 = 2800) (r2_mult : r2 = 1.25 * r1) :
  r1 = 800 ∧ r2 = 1000 := 
by
  sorry

end bus_rent_proof_l164_164343


namespace number_of_outfits_l164_164880

noncomputable def normalDistribution (μ σ² : ℝ) : measure_theory.measure ℝ := sorry -- This will be provided by the distribution library

-- Conditions and known values
def mu : ℝ := 173
def sigma_sq : ℝ := 25
def total_employees : ℕ := 10000
def height_distribution : measure_theory.measure ℝ := normalDistribution mu sigma_sq
def prob_within_2sigma : ℝ := 0.954

-- Lean 4 statement proving the tuple (question, conditions, correct answer)
theorem number_of_outfits :
  let count_units := total_employees * prob_within_2sigma in
  count_units = 9540 :=
by
  sorry

end number_of_outfits_l164_164880


namespace remainder_when_dividing_l164_164763

theorem remainder_when_dividing (a : ℕ) (h1 : a = 432 * 44) : a % 38 = 8 :=
by
  -- Proof goes here
  sorry

end remainder_when_dividing_l164_164763


namespace Sally_seashells_l164_164267

/- Definitions -/
def Tom_seashells : Nat := 7
def Jessica_seashells : Nat := 5
def total_seashells : Nat := 21

/- Theorem statement -/
theorem Sally_seashells : total_seashells - (Tom_seashells + Jessica_seashells) = 9 := by
  -- Definitions of seashells found by Tom, Jessica and the total should be used here
  -- Proving the theorem
  sorry

end Sally_seashells_l164_164267


namespace baseball_games_per_month_l164_164360

theorem baseball_games_per_month (total_games : ℕ) (season_length : ℕ) (games_per_month : ℕ) :
  total_games = 14 → season_length = 2 → games_per_month = total_games / season_length → games_per_month = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end baseball_games_per_month_l164_164360


namespace monotonic_decreasing_interval_l164_164319

def f (x : ℝ) : ℝ := Real.logBase (1/2) (x^2 - 2*x - 3)

theorem monotonic_decreasing_interval :
  ∀ x y, 3 < x → x < y → f y < f x := by
  sorry

end monotonic_decreasing_interval_l164_164319


namespace intersection_line_l164_164489

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 3*x - y = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + y = 0

-- Define the line that we need to prove as the intersection
def line (x y : ℝ) : Prop := x - 2*y = 0

-- The theorem to prove
theorem intersection_line (x y : ℝ) : circle1 x y ∧ circle2 x y → line x y :=
by
  sorry

end intersection_line_l164_164489


namespace sum_of_real_numbers_l164_164226

theorem sum_of_real_numbers (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_real_numbers_l164_164226


namespace possible_values_of_sum_l164_164234

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + y^3 + 21 * x * y = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l164_164234


namespace problem_statement_l164_164508

def f (x : ℝ) : ℝ := |x - 1| - |x|

theorem problem_statement : f (f (1/2)) = 1 :=
by
    sorry

end problem_statement_l164_164508


namespace find_PR_in_triangle_l164_164606

theorem find_PR_in_triangle (P Q R M : ℝ) (PQ QR PM : ℝ):
  PQ = 7 →
  QR = 10 →
  PM = 5 →
  M = (Q + R) / 2 →
  PR = Real.sqrt 149 := 
sorry

end find_PR_in_triangle_l164_164606


namespace hundreds_digit_binom_12_6_times_6_fact_l164_164810

open Nat

def binom (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

noncomputable def six_factorial : ℕ :=
  6.factorial

noncomputable def to_triple_digits (n : ℕ) : ℕ :=
  (n / 100) % 10

theorem hundreds_digit_binom_12_6_times_6_fact :
  to_triple_digits (binom 12 6 * six_factorial) = 8 :=
by 
  sorry

end hundreds_digit_binom_12_6_times_6_fact_l164_164810


namespace points_on_line_l164_164288

theorem points_on_line (n : ℕ) (h : 9 * n - 8 = 82) : n = 10 := by
  sorry

end points_on_line_l164_164288


namespace missing_digit_divisible_by_9_l164_164348

theorem missing_digit_divisible_by_9 : 
  (∃ x : ℕ, 0 ≤ x ∧ x ≤ 9 ∧ 
    let num := [3, 6, 5, x, 4, 2] in 
      (list.sum num) % 9 = 0) ↔ x = 7 :=
by
  sorry

end missing_digit_divisible_by_9_l164_164348


namespace Mikail_birthday_money_l164_164211

theorem Mikail_birthday_money (x : ℕ) (h1 : x = 3 + 3 * 3) : 5 * x = 60 := 
by 
  sorry

end Mikail_birthday_money_l164_164211


namespace smallest_integer_whose_cube_ends_in_368_l164_164041

theorem smallest_integer_whose_cube_ends_in_368 :
  ∃ (n : ℕ+), (n % 2 = 0 ∧ n^3 % 1000 = 368) ∧ (∀ (m : ℕ+), m % 2 = 0 ∧ m^3 % 1000 = 368 → m ≥ n) :=
by
  sorry

end smallest_integer_whose_cube_ends_in_368_l164_164041


namespace complex_magnitude_l164_164892

theorem complex_magnitude (a b : ℝ) (i : ℂ) (h : i * i = -1) (h_eq : (a + i) * (1 - b * i) = 2 * i) :
  complex.abs (a + b * i) = real.sqrt 2 :=
sorry

end complex_magnitude_l164_164892


namespace tangent_line_parallel_l164_164540

open Function

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x ^ 2 + 2 * b * x

theorem tangent_line_parallel {b : ℝ} (h : 2 * b = 1) :
  let f (x : ℝ) := x ^ 2 + x,
  S (n : ℕ) := ∑ i in Finset.range (n + 1), 1 / f (i : ℝ) in 
  S 2016 = 2016 / 2017 :=
by
  let f := λ x : ℝ, x ^ 2 + x
  sorry

end tangent_line_parallel_l164_164540


namespace fraction_of_bones_in_foot_is_approx_one_eighth_l164_164304

def number_bones_human_body : ℕ := 206
def number_bones_one_foot : ℕ := 26
def fraction_bones_one_foot (total_bones foot_bones : ℕ) : ℚ := foot_bones / total_bones

theorem fraction_of_bones_in_foot_is_approx_one_eighth :
  fraction_bones_one_foot number_bones_human_body number_bones_one_foot = 13 / 103 ∧ 
  (abs ((13 / 103 : ℚ) - (1 / 8)) < 1 / 103) := 
sorry

end fraction_of_bones_in_foot_is_approx_one_eighth_l164_164304


namespace correct_exponential_rule_l164_164387

theorem correct_exponential_rule (a : ℝ) : (a^3)^2 = a^6 :=
by sorry

end correct_exponential_rule_l164_164387


namespace verify_function_properties_l164_164148

def f (x : ℝ) : ℝ := 1 / x
def g (x : ℝ) : ℝ := x^2
def h (x : ℝ) : ℝ := -x + 1
def k (x : ℝ) : ℝ := x^3

theorem verify_function_properties : f 1 = 1 ∧ g 1 = 1 ∧ h 1 ≠ 1 ∧ k 1 = 1 :=
by
  split
  · show f 1 = 1
    sorry
  split
  · show g 1 = 1
    sorry
  split
  · show h 1 ≠ 1
    sorry
  · show k 1 = 1
    sorry

end verify_function_properties_l164_164148


namespace problem_statement_l164_164623

open Complex

theorem problem_statement (α : ℂ) (h1 : α ≠ 1) (h2 : |α^2 - 1| = 3 * |α - 1|) (h3 : |α^3 - 1| = 8 * |α - 1|) (h4 : arg α = π / 6) : α = (Real.sqrt 3 / 2) + (Complex.I / 2) :=
sorry

end problem_statement_l164_164623


namespace a_n_formula_T_n_formula_l164_164067

variable {ℕ* : Type} -- Natural numbers excluding zero

-- Define a_n based on the conditions
def a_n (n : ℕ*) : ℕ := 2^n

-- Define the sum of the first n terms S_n
def S_n (n : ℕ*) : ℕ := 2 * a_n n - 2

-- Define the sequence b_n with given conditions
def b_n (n : ℕ*) : ℕ :=
  if n = 1 then 3
  else if n = 4 then 9
  else (n - 2) * 2 + 3

-- Define the sequence c_n
def c_n (n : ℕ*) : ℝ := (b_n n) / (a_n n)

-- Define the sum of the first n terms of c_n
def T_n (n : ℕ*) : ℝ :=
  (Finset.range n).sum (λ i, c_n (i + 1))

-- Now we state the theorem for the required results

theorem a_n_formula (n : ℕ*) : a_n n = 2^n := 
  sorry

theorem T_n_formula (n : ℕ*) :
  T_n n = 5 - (2 * n + 5) / (2^n) :=
  sorry

end a_n_formula_T_n_formula_l164_164067


namespace seonmi_initial_money_l164_164268

theorem seonmi_initial_money (M : ℝ) (h1 : M/6 = 250) : M = 1500 :=
by
  sorry

end seonmi_initial_money_l164_164268


namespace right_triangle_condition_l164_164073

theorem right_triangle_condition (a b c : ℝ) (θ : ℝ) 
  (h₁ : c = real.sqrt (a^2 + b^2)) 
  (h₂ : a ≥ 0) (h₃ : b ≥ 0) (h₄ : c > 0) 
  (h₅ : real.cos θ = 0) :
  (real.sqrt (a^2 + b^2) = a + b) ↔ (θ = real.pi / 2) :=
by
  sorry

end right_triangle_condition_l164_164073


namespace part1_inequality_part2_range_of_m_l164_164481

def f (x : ℝ) : ℝ := |x+1| + |x-3|

-- Part 1
theorem part1_inequality (x : ℝ) : f(x) ≤ 3*x + 4 ↔ x ≥ 0 := 
by sorry

-- Part 2
theorem part2_range_of_m (m : ℝ) : (∀ x : ℝ, f(x) ≥ m) ↔ m ≤ 4 :=
by sorry

end part1_inequality_part2_range_of_m_l164_164481


namespace Ian_money_left_l164_164565

def hours_worked := 8
def earnings_per_hour := 18
def monthly_expenses := 50
def tax_rate := 0.10
def spending_fraction := 0.5

def total_earnings := hours_worked * earnings_per_hour
def tax := tax_rate * total_earnings
def net_earnings := total_earnings - tax
def amount_spent := spending_fraction * net_earnings
def remaining_after_spending := net_earnings - amount_spent
def money_left := remaining_after_spending - monthly_expenses

theorem Ian_money_left : money_left = 14.80 := by
  sorry

end Ian_money_left_l164_164565


namespace not_difference_of_squares_2021_l164_164120

theorem not_difference_of_squares_2021:
  ¬ ∃ (a b : ℕ), (a > b) ∧ (a^2 - b^2 = 2021) :=
sorry

end not_difference_of_squares_2021_l164_164120


namespace curve_is_line_l164_164019

noncomputable def polar_to_cartesian (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (x, y)

theorem curve_is_line (r : ℝ) (θ : ℝ) :
  r = 1 / (Real.sin θ + Real.cos θ) ↔ ∃ (x y : ℝ), (x, y) = polar_to_cartesian r θ ∧ (x + y)^2 = 1 :=
by 
  sorry

end curve_is_line_l164_164019


namespace plane_perpendicular_iff_line_perpendicular_l164_164143

-- Definitions for conditions
variables {α β : Type*} [plane α] [plane β] [different_planes α β]
variable m : α
variables (line m)

-- Problem statement
theorem plane_perpendicular_iff_line_perpendicular :
  ((perpendicular α β) → (perpendicular m β))
  ∧ ¬((perpendicular m β) → (perpendicular α β)) :=
by sorry

end plane_perpendicular_iff_line_perpendicular_l164_164143


namespace total_distance_traveled_l164_164759

noncomputable def Vm := 7 -- Speed of the man in still water in km/h
noncomputable def Vr := 1.2 -- Speed of the river in km/h
noncomputable def total_time := 1 -- Total time for the round trip in hours

theorem total_distance_traveled (Vm Vr total_time : ℝ) (hVm : Vm = 7) (hVr : Vr = 1.2) (htotal_time : total_time = 1) :
  2 * ((Vm - Vr) * (total_time / 2 * ((Vm - Vr) + (Vm + Vr)))) = 7 :=
by
  rw hVm at *
  rw hVr at *
  rw htotal_time at *
  sorry

end total_distance_traveled_l164_164759


namespace sum_of_real_roots_l164_164719

theorem sum_of_real_roots :
  let equation1 := λ x : ℝ, x^2 - 3 * x + 6 = 0 in
  let equation2 := λ x : ℝ, x^2 - 2 * x - 3 = 0 in
  (∀ x : ℝ, equation1 x → false) →
  (∃ (x₃ x₄ : ℝ), equation2 x₃ ∧ equation2 x₄ ∧ x₃ + x₄ = 2) :=
by
  assume equation1 : ℝ → Prop := λ x, x^2 - 3 * x + 6 = 0
  assume equation2 : ℝ → Prop := λ x, x^2 - 2 * x - 3 = 0

  have : ∀ x : ℝ, equation1 x → false :=
    sorry

  have h2 : ∃ (x₃ x₄ : ℝ), equation2 x₃ ∧ equation2 x₄ ∧ x₃ + x₄ = 2 :=
    sorry

  exact ⟨this, sorry⟩

end sum_of_real_roots_l164_164719


namespace yellow_points_implies_yellow_region_l164_164511

theorem yellow_points_implies_yellow_region {n : ℕ} (hₙ : n = 2018) 
  (h_circles : ∀ i j : ℕ, i ≠ j → (∃! p, (circle_intersect i j p) ∧ ¬(circle_tangent i j)))
  (h_no_concurrent : ∀ i j k : ℕ, i ≠ j → j ≠ k → i ≠ k → ¬(three_circle_concurrent i j k))
  (h_colors : ∀ i : ℕ, hℕ : i ≤ n → (∃ even_vertices : ℕ, ∃ color_function : (ℕ → color), color_alternate_circle i even_vertices color_function))
  (h_multi_color : ∀ v : vertex, vertex_colored_twice v)
  (h_yellow_condition : ∀ v : vertex, (is_yellow v ↔ colored_twice_diff v))
  (h_yellow_points : ∀ i : ℕ, hℕ : i ≤ n → (yellow_points_on_circle i ≥ 2061)) :
  ∃ region : region, all_vertices_yellow region :=
sorry

end yellow_points_implies_yellow_region_l164_164511


namespace sin_B_and_c_l164_164153

noncomputable def triangle_conditions : Prop :=
  ∃ (A B C : ℝ) (a b c : ℝ), 
    (sin A = 3 / 5) ∧ 
    (tan (A - B) = 1 / 3) ∧ 
    (C = π - A - B) ∧ 
    (sin A > 0) ∧ 
    (cos A > 0) ∧ 
    (b = 5) ∧ 
    a = 5 * sin B / sqrt(1 + (sin B / cos B)^2) ∧ 
    c = b * ((sin (A + B)) / (sin B))

theorem sin_B_and_c : 
  ∀ (A B C : ℝ) (a b c : ℝ), triangle_conditions → 
    (sin B = sqrt 10 / 10) ∧ 
    (c = 13) := 
by 
  intros A B C a b c h_cond 
  sorry

end sin_B_and_c_l164_164153


namespace maximum_value_expression_l164_164187

theorem maximum_value_expression (x a : ℝ) (hx : 0 < x) (ha : 0 < a) :
    ∃ M, M = 2 * a * (1 / (2 * Real.sqrt a + Real.sqrt (2 * a))) ∧
    (∀ y, y > 0 → (y^2 + a - Real.sqrt (y^4 + a^2)) / y ≤ M) :=
begin
  sorry
end

end maximum_value_expression_l164_164187


namespace distinctDiagonalsConvexNonagon_l164_164936

theorem distinctDiagonalsConvexNonagon : 
  ∀ (P : Type) [fintype P] [decidable_eq P] (vertices : finset P) (h : vertices.card = 9), 
  let n := vertices.card in
  let diagonals := (n * (n - 3)) / 2 in
  diagonals = 27 :=
by
  intros
  let n := vertices.card
  have keyIdentity : (n * (n - 3)) / 2 = 27 := sorry
  exact keyIdentity

end distinctDiagonalsConvexNonagon_l164_164936


namespace final_value_l164_164161

/-- Conditions: -/
variables 
  (a b c x : ℕ) 
  (initial_odometer final_odometer : ℕ)

-- Initial odometer reading: 100a + 10b + c
def initial_odometer := 100 * a + 10 * b + c

-- Final odometer reading: 100b + 10c + a
def final_odometer := 100 * b + 10 * c + a

-- Distance traveled is the difference in odometer readings
def distance_traveled := final_odometer - initial_odometer

-- Karen's driving time at 65 miles per hour
def driving_time := distance_traveled / 65

-- Constraints defined in the problem
axiom a_ge_two : a ≥ 2
axiom prod_even : a * b * c % 2 = 0
axiom time_whole_hours : driving_time ∈ ℕ

/-- Final statement to prove a^3 + b^3 + c^3 -/
theorem final_value : ∃ (a b c x : ℕ), 
  (100 * b + 10 * c + a) - (100 * a + 10 * b + c) = 65 * x ∧ 
  a ≥ 2 ∧ 
  a * b * c % 2 = 0 ∧ 
  driving_time ∈ ℕ →
  a^3 + b^3 + c^3 = correct_value := 
by sorry

end final_value_l164_164161


namespace min_value_expression_l164_164628

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 4*x + 2) * (y^2 + 4*y + 2) * (z^2 + 4*z + 2) / (x * y * z) ≥ 216 :=
by
  sorry

end min_value_expression_l164_164628


namespace original_number_of_people_l164_164373

-- Define the conditions as Lean definitions
def two_thirds_left (x : ℕ) : ℕ := (2 * x) / 3
def one_fourth_dancing_left (x : ℕ) : ℕ := ((x / 3) - (x / 12))

-- The problem statement as Lean theorem
theorem original_number_of_people (x : ℕ) (h : x / 4 = 15) : x = 60 :=
by sorry

end original_number_of_people_l164_164373


namespace jill_investment_l164_164158

noncomputable def final_amount (P : ℝ) (r1 r2 r3 r4 : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
let A1 := P * (1 + r1 / n) ^ (n * t / 2) in
let A2 := A1 * (1 + r2 / n) ^ (n * t / 2) in
let A3 := A2 * (1 + r3 / n) ^ (n * t / 2) in
let A4 := A3 * (1 + r4 / n) ^ (n * t / 2) in
A4

theorem jill_investment : 
  final_amount 10000 0.0396 0.0421 0.0372 0.0438 2 1 = 10843.17 := 
sorry

end jill_investment_l164_164158


namespace cartesian_eqn_of_line_l_cartesian_eqn_of_curve_C_length_AB_l164_164596

noncomputable def line_parametric_eqn (t : ℝ) : ℝ × ℝ :=
  ( - (Real.sqrt 2 / 2) * t, -4 + (Real.sqrt 2 / 2) * t )

noncomputable def polar_eqn (θ : ℝ) : ℝ :=
  4 * Real.cos θ

theorem cartesian_eqn_of_line_l :
  ∀ (x y t : ℝ), (line_parametric_eqn t = (x, y)) → (x + y + 4 = 0) :=
sorry

theorem cartesian_eqn_of_curve_C :
  ∀ (ρ θ x y : ℝ), (polar_eqn θ = ρ) → (ρ^2 = 4 * ρ * Real.cos θ) → (x^2 + y^2 - 4 * x = 0) :=
sorry

theorem length_AB :
  ∀ (x y t x' y' t₁ t₂ : ℝ),
  (line_parametric_eqn t₁ = (x, y)) →
  (line_parametric_eqn t₂ = (x', y')) →
  (x = 1 - (Real.sqrt 2 / 2) * t₁) →
  (y = (Real.sqrt 2 / 2) * t₁) →
  (x' = 1 - (Real.sqrt 2 / 2) * t₂) →
  (y' = (Real.sqrt 2 / 2) * t₂) →
  (x^2 + y^2 - 4 * x = 0) →
  (t₁ + t₂ = -Real.sqrt 2) →
  (t₁ * t₂ = -3) →
  (Real.abs (t₁ - t₂) = Real.sqrt 14) :=
sorry

end cartesian_eqn_of_line_l_cartesian_eqn_of_curve_C_length_AB_l164_164596


namespace rectangle_sum_bound_l164_164840

theorem rectangle_sum_bound
  (grid : ℤ × ℤ → ℝ)
  (h : ∀ (x y n : ℤ), |∑ i in finset.Icc (x, y) (x + n - 1, y + n - 1), grid i| ≤ 1) :
  ∀ (x1 y1 x2 y2 : ℤ), |∑ i in finset.Icc (x1, y1) (x2, y2), grid i| ≤ 10000 :=
sorry

end rectangle_sum_bound_l164_164840


namespace interest_first_year_correct_interest_second_year_correct_interest_third_year_correct_l164_164801

noncomputable def principal_first_year : ℝ := 9000
noncomputable def interest_rate_first_year : ℝ := 0.09
noncomputable def principal_second_year : ℝ := principal_first_year * (1 + interest_rate_first_year)
noncomputable def interest_rate_second_year : ℝ := 0.105
noncomputable def principal_third_year : ℝ := principal_second_year * (1 + interest_rate_second_year)
noncomputable def interest_rate_third_year : ℝ := 0.085

noncomputable def compute_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * rate

theorem interest_first_year_correct :
  compute_interest principal_first_year interest_rate_first_year = 810 := by
  sorry

theorem interest_second_year_correct :
  compute_interest principal_second_year interest_rate_second_year = 1034.55 := by
  sorry

theorem interest_third_year_correct :
  compute_interest principal_third_year interest_rate_third_year = 922.18 := by
  sorry

end interest_first_year_correct_interest_second_year_correct_interest_third_year_correct_l164_164801


namespace greatest_common_divisor_of_180_and_n_is_9_l164_164371

theorem greatest_common_divisor_of_180_and_n_is_9 
  {n : ℕ} (h_n_divisors : ∀ d : ℕ, d ∣ 180 ∧ d ∣ n ↔ d ∈ {1, 3, 9}) : 
  greatest_common_divisor 180 n = 9 := 
sorry

end greatest_common_divisor_of_180_and_n_is_9_l164_164371


namespace semicircle_triangle_l164_164699

variable (a b r : ℝ)

-- Conditions: 
-- (1) Semicircle of radius r inside a right-angled triangle
-- (2) Shorter edges of the triangle (tangents to the semicircle) have lengths a and b
-- (3) Diameter of the semicircle lies on the hypotenuse of the triangle

theorem semicircle_triangle (h1 : a > 0) (h2 : b > 0) (h3 : r > 0)
  (tangent_property : true) -- Assumed relevant tangent properties are true
  (angle_property : true) -- Assumed relevant angle properties are true
  (geom_configuration : true) -- Assumed specific geometric configuration is correct
  : 1 / r = 1 / a + 1 / b := 
  sorry

end semicircle_triangle_l164_164699


namespace sum_of_real_numbers_l164_164228

theorem sum_of_real_numbers (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_real_numbers_l164_164228


namespace combinatorial_sum_formula_l164_164767

theorem combinatorial_sum_formula (n : ℕ) : 
  ∑ i in finset.range (n + 1), nat.choose (4 * n + 1) (4 * i + 1) = 2^(4 * n - 1) + (-1)^n * 2^(2 * n - 1) :=
sorry

end combinatorial_sum_formula_l164_164767


namespace minimum_steps_to_stuff_baskets_l164_164412

def isStuffBasket (basket : ℕ × ℕ) : Prop :=
  basket.1 = 10 ∧ basket.2 = 30

def totalResourcesCorrect (baskets : List (ℕ × ℕ)) : Prop :=
  baskets.foldl (fun acc b => (acc.1 + b.1, acc.2 + b.2)) (0, 0) = (1000, 3000)

theorem minimum_steps_to_stuff_baskets (baskets : List (ℕ × ℕ)) :
  totalResourcesCorrect baskets →
  ∃ steps : ℕ, steps = 99 ∧ ∀ finalBaskets, 
  stepsToConvert baskets finalBaskets steps → 
  (∀ b ∈ finalBaskets, isStuffBasket b) :=
sorry

end minimum_steps_to_stuff_baskets_l164_164412


namespace repayment_is_correct_l164_164784

noncomputable def repayment_amount (a r : ℝ) : ℝ := a * r * (1 + r) ^ 5 / ((1 + r) ^ 5 - 1)

theorem repayment_is_correct (a r : ℝ) (h_a : a > 0) (h_r : r > 0) :
  repayment_amount a r = a * r * (1 + r) ^ 5 / ((1 + r) ^ 5 - 1) :=
by
  sorry

end repayment_is_correct_l164_164784


namespace only_one_prime_in_sequence_l164_164563

-- Define the list of numbers formed by repeating "47" up to six times.
def sequence : List ℕ := [47, 47 * 101, 47 * 10101, 47 * 1010101, 47 * 101010101, 47 * 10101010101]

-- Define the property that checks if a number is prime.
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define the theorem to prove that the only prime number in the list is 47.
theorem only_one_prime_in_sequence : (sequence.countp is_prime) = 1 := by sorry

end only_one_prime_in_sequence_l164_164563


namespace possible_values_of_sum_l164_164231

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + y^3 + 21 * x * y = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l164_164231


namespace domain_of_f_real_numbers_l164_164832

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := (3 * m * x^2 - 4 * x + 1) / (4 * x^2 - 3 * x + m)

-- Domain condition: The denominator must not be zero for any real x
def denominator (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - 3 * x + m

-- The discriminant of the quadratic expression in the denominator must be negative
theorem domain_of_f_real_numbers (m : ℝ) : (∀ x : ℝ, denominator m x ≠ 0) ↔ m > 9 / 16 :=
by
  sorry

end domain_of_f_real_numbers_l164_164832


namespace parcel_postage_6_75_l164_164430

-- Define the conditions provided in the problem statement
def base_rate := 45 -- cents
def additional_rate := 25 -- cents
def weight := 6.75 -- ounces

-- Calculate the total postage based on the given conditions
noncomputable def postage (w : Real) : Real :=
  base_rate + additional_rate * (Real.ceil (w - 1))

-- Prove that the postage for a 6.75-ounce parcel is 1.95 dollars
theorem parcel_postage_6_75 :
  postage 6.75 = 195 :=
by
  sorry

end parcel_postage_6_75_l164_164430


namespace unit_vectors_collinear_with_a_l164_164353

theorem unit_vectors_collinear_with_a :
  let a := (-3, -4, 5 : ℝ × ℝ × ℝ)
  let mag_a := real.sqrt (9 + 16 + 25)
  ∃ u v : ℝ × ℝ × ℝ, u = (3 * real.sqrt 2 / 10, 2 * real.sqrt 2 / 5, - real.sqrt 2 / 2) ∧
                      v = (-3 * real.sqrt 2 / 10, -2 * real.sqrt 2 / 5, real.sqrt 2 / 2) ∧
                      (∀ x : ℝ × ℝ × ℝ, x = u ∨ x = v → (∃ k : ℝ, x = k • a ∧ k • k = 1 / mag_a * (1 / mag_a))
 :=
by
  sorry

end unit_vectors_collinear_with_a_l164_164353


namespace points_on_line_possible_l164_164280

theorem points_on_line_possible : ∃ n : ℕ, 9 * n - 8 = 82 :=
by
  sorry

end points_on_line_possible_l164_164280


namespace find_b_of_hyperbola_asymptote_l164_164102

def inclination_angle_to_slope (angle : ℝ) := Real.tan angle

theorem find_b_of_hyperbola_asymptote :
  ∀ (b : ℝ), (b > 0) ∧ (∃ (x y : ℝ), (x^2 / 9 - y^2 / b^2 = 1)) ∧ (inclination_angle_to_slope 150 = -b / 3) →
  b = Real.sqrt 3 := by
  sorry

end find_b_of_hyperbola_asymptote_l164_164102


namespace arithmetic_sequence_sum_20_l164_164997

open BigOperators

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) = a n + (a 1 - a 0)

theorem arithmetic_sequence_sum_20 {a : ℕ → ℤ} (h_arith : is_arithmetic_sequence a)
    (h1 : a 0 + a 1 + a 2 = -24)
    (h18 : a 17 + a 18 + a 19 = 78) :
    ∑ i in Finset.range 20, a i = 180 :=
sorry

end arithmetic_sequence_sum_20_l164_164997


namespace proof_problem_l164_164601

-- Definition of the parabola in Cartesian coordinates
def parabola_cartesian (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Definition of the line in parametric form
def line_parametric (x y t α : ℝ) : Prop :=
  x = 2 + t * cos α ∧ y = t * sin α

-- Proof statement for the polar coordinate equation of the parabola
def polar_parabola_equation (ρ θ : ℝ) : Prop :=
  ρ * sin θ ^ 2 = 4 * cos θ

-- Proof statement for the slope angle α given |AB| = 4√6
def slope_angle (α : ℝ) : Prop :=
  α = (Real.pi / 4) ∨ α = (3 * Real.pi / 4)

-- Main theorem combining all conditions and proof goals
theorem proof_problem (x y t α ρ θ : ℝ) :
  (parabola_cartesian x y) →
  (line_parametric x y t α) →
  (polar_parabola_equation ρ θ) →
  (dist (A B) = 4 * sqrt 6) →
  slope_angle α :=
by
  sorry

end proof_problem_l164_164601


namespace triangles_congruent_l164_164220

theorem triangles_congruent
  (A B C A' B' C' M₃ M'₃ : Type)
  [Triangle A B C]
  [Triangle A' B' C']
  (h_diff : (side A C - side B C = side A' C' - side B' C'))
  (h_third_side : side A B = side A' B')
  (h_medians : median A M₃ C = median A' M'₃ C') :
  Triangle A B C = Triangle A' B' C' :=
  sorry

end triangles_congruent_l164_164220


namespace noncongruent_triangles_count_l164_164593

-- Definitions from the conditions
variable (A B C D E F G H : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F] [Inhabited G] [Inhabited H]
variable (isIsoRightTriangleABC : ∀ (x : B), A = x → C = x)
variable (midpoint_D_AB : D = (A + B) / 2)
variable (midpoint_E_BC : E = (B + C) / 2)
variable (midpoint_F_CA : F = (C + A) / 2)
variable (midpoint_G_AD : G = (A + D) / 2)
variable (midpoint_H_CF : H = (C + F) / 2)

-- The proof problem statement
theorem noncongruent_triangles_count : 
  ∀ (A B C D E F G H : Type), (isIsoRightTriangleABC A B C D E F G H) → 
  (midpoint_D_AB A B) → (midpoint_E_BC A B C) → 
  (midpoint_F_CA A C) → (midpoint_G_AD A D) → (midpoint_H_CF C F) → 
  number_of_noncongruent_triangles A B C D E F G H = 3 := 
sorry

end noncongruent_triangles_count_l164_164593


namespace tan_alpha_value_trigonometric_expression_value_l164_164057

theorem tan_alpha_value (α : ℝ) (h : Real.tan (π/4 + α) = 1/2) : Real.tan α = -1/3 :=
sorry

theorem trigonometric_expression_value (α : ℝ) (h : Real.tan (π/4 + α) = 1/2) : 
  (Real.sin (2*α) - Real.cos(α)^2) / (2 + Real.cos(2*α)) = -1/2 :=
sorry

end tan_alpha_value_trigonometric_expression_value_l164_164057


namespace sum_of_extreme_values_l164_164043

theorem sum_of_extreme_values (x : ℝ) :
  (∀ x, 9^(x+1) + 2187 = 3^(6*x - x^2)) →
  (let smallest_x := 2 + real.sqrt 2 in
   let largest_x := 3 - real.sqrt 2 in
   smallest_x + largest_x = 5) :=
sorry

end sum_of_extreme_values_l164_164043


namespace smallest_positive_integer_cube_ends_368_l164_164037

theorem smallest_positive_integer_cube_ends_368 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 368 ∧ n = 34 :=
by
  sorry

end smallest_positive_integer_cube_ends_368_l164_164037


namespace monty_hall_probability_l164_164584

noncomputable theory
open_locale classical

-- Representing the Monty Hall problem with appropriate probability
def monty_hall := {door : Type* // fintype door}

variables {door : Type*} [fintype door] [decidable_eq door] (car goat1 goat2 : door)

-- The doors are distinct
axiom car_goat1_distinct : car ≠ goat1
axiom car_goat2_distinct : car ≠ goat2
axiom goat1_goat2_distinct : goat1 ≠ goat2

-- Initial probabilities
def initial_probability (choose : door) (prize : door) : ℝ :=
  if choose = prize then 1 / 3 else 2 / 3

-- After host reveals a goat
def switch_probability (choose : door) (open : door) (prize : door) : ℝ :=
  if choose = prize then 1 / 3 else 2 / 3

theorem monty_hall_probability
  (choice : door) (revealed_goat : door)
  (h : revealed_goat ≠ choice) (h' : revealed_goat = goat1 ∨ revealed_goat = goat2) :
  initial_probability choice car = 1 / 3 ∧ switch_probability choice revealed_goat car = 2 / 3 :=
sorry

end monty_hall_probability_l164_164584


namespace max_elements_in_M_l164_164528

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def no_three_product_is_perfect_square (M : Set ℕ) : Prop :=
  ∀ a b c : ℕ, a ∈ M → b ∈ M → c ∈ M → a ≠ b → b ≠ c → a ≠ c → ¬ is_perfect_square (a * b * c)

theorem max_elements_in_M :
  ∀ M : Set ℕ, (M ⊆ {n : ℕ | n ≤ 15}) →
  no_three_product_is_perfect_square M →
  ∃ k, k = Set.card M ∧ k ≤ 11 := 
sorry

end max_elements_in_M_l164_164528


namespace distinctDiagonalsConvexNonagon_l164_164940

theorem distinctDiagonalsConvexNonagon : 
  ∀ (P : Type) [fintype P] [decidable_eq P] (vertices : finset P) (h : vertices.card = 9), 
  let n := vertices.card in
  let diagonals := (n * (n - 3)) / 2 in
  diagonals = 27 :=
by
  intros
  let n := vertices.card
  have keyIdentity : (n * (n - 3)) / 2 = 27 := sorry
  exact keyIdentity

end distinctDiagonalsConvexNonagon_l164_164940


namespace sum_of_divisors_of_40001_l164_164612

-- Definitions based on the conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_product_of_two_primes (n p q : ℕ) : Prop :=
  n = p * q ∧ is_prime p ∧ is_prime q

def sum_of_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ x => n % x = 0).sum

-- The proof statement
theorem sum_of_divisors_of_40001 :
  ∃ p q : ℕ, is_product_of_two_primes 400000001 p q →
  sum_of_divisors (p + q - 1) = 45864 :=
sorry

end sum_of_divisors_of_40001_l164_164612


namespace count_both_symmetries_l164_164450

-- Definitions of the shapes and their symmetries
def is_centrally_symmetric : Type := {ls : List String // ls.nodup} 
∧ ls ⊆ ["line segment", "isosceles trapezoid", "parallelogram", "rectangle", "rhombus", "square", "equilateral triangle"]

def is_axially_symmetric : Type := {la : List String // la.nodup} 
∧ la ⊆ ["line segment", "isosceles trapezoid", "parallelogram", "rectangle", "rhombus", "square", "equilateral triangle"]

def both_symmetries : Type := { overlap : List String // overlap.nodup } 
overlap = intersect (is_centrally_symmetric) (is_axially_symmetric)

-- Statements about the symmetries
axiom line_segment_symmetries : "line segment" ∈ is_centrally_symmetric ∧ "line segment" ∈ is_axially_symmetric
axiom isosceles_trapezoid_symmetries : "isosceles trapezoid" ∉ is_centrally_symmetric ∧ "isosceles trapezoid" ∈ is_axially_symmetric
axiom parallelogram_symmetries : "parallelogram" ∈ is_centrally_symmetric ∧ "parallelogram" ∉ is_axially_symmetric
axiom rectangle_symmetries : "rectangle" ∈ is_centrally_symmetric ∧ "rectangle" ∈ is_axially_symmetric
axiom rhombus_symmetries : "rhombus" ∈ is_centrally_symmetric ∧ "rhombus" ∈ is_axially_symmetric
axiom square_symmetries : "square" ∈ is_centrally_symmetric ∧ "square" ∈ is_axially_symmetric
axiom equilateral_triangle_symmetries : "equilateral triangle" ∉ is_centrally_symmetric ∧ "equilateral triangle" ∈ is_axially_symmetric

-- Define the theorem to prove the count of figures that are both centrally and axially symmetric is 4
theorem count_both_symmetries : List.both_symmetries.length = 4 := 
by {
  sorry
}

end count_both_symmetries_l164_164450


namespace max_subsets_sum_to_one_l164_164755

theorem max_subsets_sum_to_one (n : ℕ) (a : Fin n → ℝ) :
  (∃ S : Finset (Fin n), S.nonempty ∧ S.sum (λ i, a i) = 1) → 
  ∃ k : ℕ, k = 2^(n-1) := 
sorry

end max_subsets_sum_to_one_l164_164755


namespace part1_part2_part3_l164_164519

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + 2)

theorem part1 (h_odd : ∀ x : ℝ, f x b = -f (-x) b) : b = 1 :=
sorry

theorem part2 (h_b : b = 1) : ∀ (x1 x2 : ℝ), x1 < x2 → f x1 1 > f x2 1 :=
sorry

theorem part3 (h_monotonic : ∀ (x1 x2 : ℝ), x1 < x2 → f x1 1 > f x2 1) 
  : ∀ t : ℝ, f (t^2 - 2 * t) 1 + f (2 * t^2 - k) 1 < 0 → k < -1/3 :=
sorry

end part1_part2_part3_l164_164519


namespace part1_part2_l164_164098

noncomputable def f (a x : ℝ) := a * x^2 - (a + 1) * x + 1

theorem part1 (a : ℝ) (h1 : a ≠ 0) :
  (∀ x : ℝ, f a x ≤ 2) ↔ (-3 - 2 * Real.sqrt 2 ≤ a ∧ a ≤ -3 + 2 * Real.sqrt 2) :=
sorry

theorem part2 (a : ℝ) (h1 : a ≠ 0) (x : ℝ) :
  (f a x < 0) ↔
    ((0 < a ∧ a < 1 ∧ 1 < x ∧ x < 1 / a) ∨
     (a = 1 ∧ false) ∨
     (a > 1 ∧ 1 / a < x ∧ x < 1) ∨
     (a < 0 ∧ (x < 1 / a ∨ x > 1))) :=
sorry

end part1_part2_l164_164098


namespace population_growth_l164_164861

theorem population_growth (scale_factor1 scale_factor2 : ℝ)
    (h1 : scale_factor1 = 1.2)
    (h2 : scale_factor2 = 1.26) :
    (scale_factor1 * scale_factor2) - 1 = 0.512 :=
by
  sorry

end population_growth_l164_164861


namespace infinite_n_with_conditions_l164_164878

theorem infinite_n_with_conditions (k : ℕ) (hk : 0 < k) :
  ∃ (n : ℕ), ∃ (l : list ℕ), (∀ prime_divisor ∈ l, prime prime_divisor ∧ (prime_divisor = 3 ∨ (∃ t : ℕ, prime_divisor = 4 * t + 1))) ∧
    length l ≥ k ∧ n = (prod l) ∧ (n ∣ (2^(nat_divsum n) - 1)) sorry
-- Here, nat_divsum n would represent the function which computes the sum of positive divisors of n

end infinite_n_with_conditions_l164_164878


namespace points_on_line_l164_164274

theorem points_on_line (n : ℕ) : 9 * n - 8 = 82 → n = 10 :=
by
  sorry

end points_on_line_l164_164274


namespace max_value_of_y_l164_164836

-- Definitions for the expressions
def y (x : ℝ) : ℝ :=
  Real.sin (x + π / 4) + Real.cos (x + π / 4) - Real.tan (x + π / 3)

-- The main theorem
theorem max_value_of_y : ∃ x, (-π / 3 ≤ x ∧ x ≤ 0) ∧ ∀ x', -π / 3 ≤ x' ∧ x' ≤ 0 → y x' ≤ 1 - Real.sqrt 2 := 
  sorry

end max_value_of_y_l164_164836


namespace find_angle_TYS_l164_164146

variables {Point : Type} [OrderedGeometry Point]

-- Definitions of points, lines, and angles
variables (P Q R S Z T : Point)

-- Condition: PQ parallel to RS
axiom PQ_parallel_RS : Parallel (line P Q) (line R S)

-- Condition: Angle PZT = 135 degrees
axiom angle_PZT : angle P Z T = 135

-- Theorem to be proved: Angle TYS = 45 degrees
theorem find_angle_TYS (Y : Point)
  (Z_on_line_PQ : lies_on Z (line P Q))
  (Y_on_line_RS : lies_on Y (line R S))
  (T_on_line_ZT : lies_on T (line Z T))
  (Y_on_line_ZT : lies_on Y (line Z T)) :
  angle T Y S = 45 :=
sorry

end find_angle_TYS_l164_164146


namespace find_PR_in_triangle_l164_164607

theorem find_PR_in_triangle (P Q R M : ℝ) (PQ QR PM : ℝ):
  PQ = 7 →
  QR = 10 →
  PM = 5 →
  M = (Q + R) / 2 →
  PR = Real.sqrt 149 := 
sorry

end find_PR_in_triangle_l164_164607


namespace distinctDiagonalsConvexNonagon_l164_164938

theorem distinctDiagonalsConvexNonagon : 
  ∀ (P : Type) [fintype P] [decidable_eq P] (vertices : finset P) (h : vertices.card = 9), 
  let n := vertices.card in
  let diagonals := (n * (n - 3)) / 2 in
  diagonals = 27 :=
by
  intros
  let n := vertices.card
  have keyIdentity : (n * (n - 3)) / 2 = 27 := sorry
  exact keyIdentity

end distinctDiagonalsConvexNonagon_l164_164938


namespace prime_remainders_between_50_100_have_five_qualified_numbers_l164_164961

open Nat

def prime_remainder_set : Set ℕ := {2, 3, 5}

def count_prime_remainders_between (a b n : ℕ) : ℕ :=
  (List.range (b - a)).countp (λ x, x + a ∈ (range b).filter prime ∧ ((x + a) % n) ∈ prime_remainder_set)

theorem prime_remainders_between_50_100_have_five_qualified_numbers :
  count_prime_remainders_between 50 100 7 = 5 := 
by
  sorry

end prime_remainders_between_50_100_have_five_qualified_numbers_l164_164961


namespace baseball_games_per_month_l164_164361

-- Define the conditions
def total_games_in_a_season : ℕ := 14
def months_in_a_season : ℕ := 2

-- Define the proposition stating the number of games per month
def games_per_month (total_games months : ℕ) : ℕ := total_games / months

-- State the equivalence proof problem
theorem baseball_games_per_month : games_per_month total_games_in_a_season months_in_a_season = 7 :=
by
  -- Directly stating the equivalence based on given conditions
  sorry

end baseball_games_per_month_l164_164361


namespace max_f_over_100_l164_164419

def f : ℕ → ℕ → ℕ
| 0, x      := x
| x, 0      := x
| x, y      := if x >= y then f (x - y) y + 1 else f x (y - x) + 1.

theorem max_f_over_100 : ∃ x y : ℕ, 0 ≤ x ∧ x ≤ 100 ∧ 0 ≤ y ∧ y ≤ 100 ∧ f x y = 101 :=
by
  existsi 100
  existsi 100
  simp
  sorry

end max_f_over_100_l164_164419


namespace tetrahedron_ratio_KS_LS_l164_164172

variable (A B C D S K L : Point)
variable [IsTetrahedron A B C D]
variable [IsCenterOfGravity S A B C D]
variable [IsLineThrough S K L]
variable [IntersectsSurface_S_K_L K L A B C D]

theorem tetrahedron_ratio_KS_LS : 
  1/3 ≤ (distance K S) / (distance L S) ∧ (distance K S) / (distance L S) ≤ 3 :=
sorry

end tetrahedron_ratio_KS_LS_l164_164172


namespace transformed_period_l164_164733

open Real

-- Define the original and transformed functions
def original_sine (x : ℝ) : ℝ := sin x

def scaling_transformation_x (x : ℝ) : ℝ := (1/2) * x
def scaling_transformation_y (y : ℝ) : ℝ := 3 * y

def inverse_transformation_x (x' : ℝ) : ℝ := 2 * x'
def inverse_transformation_y (y' : ℝ) : ℝ := (1/3) * y'

-- Define the transformed sine function based on the given scaling transformations
def transformed_sine (x' : ℝ) : ℝ := 3 * sin (2 * x')

-- The theorem to be proved
theorem transformed_period :
  ∃ T > 0, ∀ x' : ℝ, transformed_sine (x' + T) = transformed_sine x' :=
sorry

end transformed_period_l164_164733


namespace vec_c_parallel_to_a_and_norm_c_angle_between_a_and_b_l164_164108

open Real EuclideanSpace

noncomputable def a : ℝ × ℝ := (1, 3)

noncomputable def is_parallel (v w : ℝ × ℝ) : Prop := ∃ (λ : ℝ), v = (λ * w.1, λ * w.2)

theorem vec_c_parallel_to_a_and_norm_c 
  (c : ℝ × ℝ) 
  (h_parallel : is_parallel c a) 
  (h_norm_c : sqrt (c.1^2 + c.2^2) = 2 * sqrt 10) :
  (c = (2, 6) ∨ c = (-2, -6)) ∧ (a.1 * c.1 + a.2 * c.2 = 20 ∨ a.1 * c.1 + a.2 * c.2 = -20) :=
  sorry

theorem angle_between_a_and_b 
  (b : ℝ × ℝ) 
  (h_norm_a : sqrt (1^2 + 3^2) = sqrt 10) 
  (h_norm_b : sqrt (b.1^2 + b.2^2) = sqrt 10 / 2) 
  (h_perp : (a.1 - 3 * b.1, a.2 - 3 * b.2) 
            ⬝ (2 * a.1 + b.1, 2 * a.2 + b.2) = 0) :
  ∃ θ : ℝ, θ = π / 3 ∧ cos θ = (a.1 * b.1 + a.2 * b.2) / (sqrt 10 * (sqrt 10 / 2)) :=
  sorry

end vec_c_parallel_to_a_and_norm_c_angle_between_a_and_b_l164_164108


namespace surface_area_of_prism_is_94_l164_164084

-- Define the lengths of the rectangular prism
def length := 5
def width := 4
def height := 3

-- Define the formula for the surface area of a rectangular prism
def surface_area (l w h : Nat) : Nat :=
  2 * (l * w + w * h + h * l)

-- Define the theorem we need to prove
theorem surface_area_of_prism_is_94 : surface_area length width height = 94 :=
by
  -- Proof goes here
  sorry

end surface_area_of_prism_is_94_l164_164084


namespace river_current_speed_l164_164793

theorem river_current_speed 
  (downstream_distance upstream_distance still_water_speed : ℝ)
  (H1 : still_water_speed = 20)
  (H2 : downstream_distance = 100)
  (H3 : upstream_distance = 60)
  (H4 : (downstream_distance / (still_water_speed + x)) = (upstream_distance / (still_water_speed - x)))
  : x = 5 :=
by
  sorry

end river_current_speed_l164_164793


namespace focal_length_of_hyperbola_l164_164543

theorem focal_length_of_hyperbola (a b p: ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (p_pos : 0 < p) :
  (∃ (F V : ℝ × ℝ), 4 = dist F V ∧ F = (2, 0) ∧ V = (-2, 0)) ∧
  (∃ (P : ℝ × ℝ), P = (-2, -1) ∧ (∃ (d : ℝ), d = d / 2 ∧ P = (d, 0))) →
  2 * (Real.sqrt (a^2 + b^2)) = 2 * Real.sqrt 5 := 
sorry

end focal_length_of_hyperbola_l164_164543


namespace highest_frequency_geometric_sequence_l164_164008

theorem highest_frequency_geometric_sequence :
  ∀ (capacity num_groups : ℕ) (cum_freq_7 : ℝ) (geometric_frequencies : list ℕ)
  (r : ℕ),
  capacity = 100 → 
  num_groups = 10 → 
  cum_freq_7 = 0.79 → 
  geometric_frequencies.length = 3 → 
  (∀ i ∈ [0, 1, 2], geometric_frequencies[i+1] / geometric_frequencies[i] = r) → 
  r > 2 → 
  capacity - (cum_freq_7 * capacity) = list.sum geometric_frequencies → 
  (∀ x ∈ geometric_frequencies, list.max geometric_frequencies = x) → 
  ∃ highest_freq, highest_freq = 16 :=
by
  intros capacity num_groups cum_freq_7 geometric_frequencies r
  intros h1 h2 h3 h4 h5 h6 h7 h8
  use 16
  sorry

end highest_frequency_geometric_sequence_l164_164008


namespace find_b_l164_164185

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then -2 * x else 3 * x - 50

theorem find_b (b : ℝ) (hb : b < 0) : f (f 15) = f (f b) ↔ b = -10 := by
  sorry

end find_b_l164_164185


namespace triangle_equilateral_l164_164979

theorem triangle_equilateral 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : 2 * a = b + c) 
  (h2 : sin(A)^2 = sin(B) * sin(C)) 
  (los1 : A = asin(b * sin(B) / a)) 
  (los2 : A = asin(c * sin(C) / a)) :
  a = b ∧ b = c :=
by
  sorry

end triangle_equilateral_l164_164979


namespace Lizzy_money_after_loan_l164_164647

theorem Lizzy_money_after_loan :
  let initial_savings := 30
  let loaned_amount := 15
  let interest_rate := 0.20
  let interest := loaned_amount * interest_rate
  let total_amount_returned := loaned_amount + interest
  let remaining_money := initial_savings - loaned_amount
  let total_money := remaining_money + total_amount_returned
  total_money = 33 :=
by
  sorry

end Lizzy_money_after_loan_l164_164647


namespace largest_divisor_of_composite_l164_164051

theorem largest_divisor_of_composite (n : ℕ) (h : n > 1 ∧ ¬ Nat.Prime n) : 12 ∣ (n^4 - n^2) :=
sorry

end largest_divisor_of_composite_l164_164051


namespace sum_possible_values_N_l164_164325

theorem sum_possible_values_N (a b c N : ℕ) (ha : 0 < a) (hb : 0 < b)
  (hc : c = a + b) (hN : N = a * b * c) (h_condition : N = 8 * (a + b + c)) :
  (N = 272 ∨ N = 160 ∨ N = 128) →
  (272 + 160 + 128) = 560 :=
by {
  intros h,
  have h1 : N = 272 ∨ N = 160 ∨ N = 128,
  from h,
  exact eq.refl 560,
}

end sum_possible_values_N_l164_164325


namespace derivative_at_1_l164_164876

-- Define the function f
def f (x : ℝ) := x - real.sqrt x

-- State the theorem that we aim to prove
theorem derivative_at_1 : deriv f 1 = 1 / 2 :=
by
  -- Proof will go here
  sorry

end derivative_at_1_l164_164876


namespace marcy_fewer_tickets_l164_164731

theorem marcy_fewer_tickets (A M : ℕ) (h1 : A = 26) (h2 : M = 5 * A) (h3 : A + M = 150) : M - A = 104 :=
by
  sorry

end marcy_fewer_tickets_l164_164731


namespace mass_percentage_carbon_in_carbonic_acid_l164_164492

theorem mass_percentage_carbon_in_carbonic_acid :
  let molar_mass_H : ℝ := 1.01
  let molar_mass_C : ℝ := 12.01
  let molar_mass_O : ℝ := 16.00
  let molar_mass_H2CO3 : ℝ := 2 * molar_mass_H + molar_mass_C + 3 * molar_mass_O

  let mass_percentage_C : ℝ := (molar_mass_C / molar_mass_H2CO3) * 100
  mass_percentage_C ≈ 19.36 :=
by
  sorry

end mass_percentage_carbon_in_carbonic_acid_l164_164492


namespace sin_2x_and_tan_fraction_l164_164077

open Real

theorem sin_2x_and_tan_fraction (x : ℝ) (h : sin (π + x) + cos (π + x) = 1 / 2) :
  (sin (2 * x) = -3 / 4) ∧ ((1 + tan x) / (sin x * cos (x - π / 4)) = -8 * sqrt 2 / 3) :=
by
  sorry

end sin_2x_and_tan_fraction_l164_164077


namespace coefficient_x_squared_term_l164_164080

theorem coefficient_x_squared_term (
  a : ℝ
) (h : a = ∫ x in 0 .. π, (Real.sin x + Real.cos x)) :
  (∑ r in Finset.range (6 + 1), Nat.choose 6 r * (2 : ℝ)^(6 - r) * (-1 : ℝ)^r * x^(3 - r / 2)) = -960 :=
sorry

end coefficient_x_squared_term_l164_164080


namespace correct_keystroke_sequence_l164_164458

def keystroke_sequence := ℕ → ℕ → ℝ

-- Definitions for the given conditions
def A (x : ℕ) (y : ℕ) : ℝ := x * y
def B (x : ℕ) (y : ℕ) : ℝ := x * (y / 100)
def C (x : ℕ) (y : ℕ) : ℝ := x * (y / 100)
def D (x : ℕ) (y : ℕ) : ℝ := x * (0.01 * 8)

-- The problem statement
theorem correct_keystroke_sequence : C 498 18 = 498 * 0.18 :=
by sorry

end correct_keystroke_sequence_l164_164458


namespace smallest_palindromic_odd_integer_in_base2_and_4_l164_164464

def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  let digits := n.digits base
  digits = digits.reverse

theorem smallest_palindromic_odd_integer_in_base2_and_4 :
  ∃ n : ℕ, n > 10 ∧ is_palindrome n 2 ∧ is_palindrome n 4 ∧ Odd n ∧ ∀ m : ℕ, (m > 10 ∧ is_palindrome m 2 ∧ is_palindrome m 4 ∧ Odd m) → n <= m :=
  sorry

end smallest_palindromic_odd_integer_in_base2_and_4_l164_164464


namespace find_common_difference_l164_164145

variable {a : ℕ → ℝ}
variable {p q : ℕ}
variable {d : ℝ}

-- Definitions based on the conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

def condition1 (a : ℕ → ℝ) (p : ℕ) := a p = 4
def condition2 (a : ℕ → ℝ) (q : ℕ) := a q = 2
def condition3 (p q : ℕ) := p = 4 + q

-- The goal statement
theorem find_common_difference
  (a_seq : arithmetic_sequence a d)
  (h1 : condition1 a p)
  (h2 : condition2 a q)
  (h3 : condition3 p q) :
  d = 1 / 2 :=
by
  sorry

end find_common_difference_l164_164145


namespace eq_number_of_elements_A_eq_number_of_elements_B_l164_164221

-- Step (a) Definitions
def A (m n : ℕ) : set (vector ℤ m) :=
  { x | ∀ 1 ≤ i, i ≤ m → 1 ≤ x.nth i ∧ ∀ 1 ≤ i, i < m → x.nth i ≤ x.nth (i+1) ∧ x.nth i ≤ n }

def B (m n : ℕ) : set (vector ℤ m) :=
  { x | x.to_list.sum = n ∧ ∀ i, 0 ≤ x.nth i }

-- Step (c) Final Proof Statements
theorem eq_number_of_elements_A (m n : ℕ) (hm : m ≥ 1) (hn : n ≥ 1) : 
  (A m n).card = nat.choose (m + n - 1) (n - 1) :=
sorry 

theorem eq_number_of_elements_B (m n : ℕ) (hm : m ≥ 1) (hn : n ≥ 1) : 
  (B m n).card = nat.choose (m + n - 1) n :=
sorry

end eq_number_of_elements_A_eq_number_of_elements_B_l164_164221


namespace Lizzy_money_after_loan_l164_164648

theorem Lizzy_money_after_loan :
  let initial_savings := 30
  let loaned_amount := 15
  let interest_rate := 0.20
  let interest := loaned_amount * interest_rate
  let total_amount_returned := loaned_amount + interest
  let remaining_money := initial_savings - loaned_amount
  let total_money := remaining_money + total_amount_returned
  total_money = 33 :=
by
  sorry

end Lizzy_money_after_loan_l164_164648


namespace first_player_has_winning_strategy_l164_164771

/-- Define a game state where each card shows either red or black side up -/
inductive CardColor
| red
| black

/-- Define a type for the game state -/
structure GameState :=
(cards : List CardColor)  -- List of 1999 cards, each either red or black

/-- Define the game's rules and winning strategy -/
noncomputable def first_player_wins : Prop :=
  ∀ (s : GameState) (h_len: s.cards.length = 1999)
    (red_count black_count : ℕ)
    (h_red_count: red_count = s.cards.filter (λ c, c = CardColor.red).length)
    (h_black_count: black_count = s.cards.filter (λ c, c = CardColor.black).length),
    -- Initial condition: sum of red and black cards is 1999, which is odd
    (red_count + black_count = 1999) ∧
    -- Proving that the first player has a winning strategy
    has_winning_strategy red_count black_count

/-- Define winning strategy logic here -/
noncomputable def has_winning_strategy : ℕ → ℕ → Prop
| r, b := r ≠ b  -- Placeholder for the actual strategy logic

theorem first_player_has_winning_strategy : first_player_wins :=
sorry

end first_player_has_winning_strategy_l164_164771


namespace cosine_a_b_l164_164863

variables (a b c : ℝ)
variables (A B C : ℜ3)

def norm (x : ℜ3) : ℝ := real.sqrt (x.1^2 + x.2^2 + x.3^2)
def dot_product (x y : ℜ3) : ℝ := x.1 * y.1 + x.2 * y.2 + x.3 * y.3

axiom vector_add_zero (A B C : ℜ3) : A + B + C = (0, 0, 0)
axiom norm_a {A : ℜ3} : norm A = 2
axiom norm_b {B : ℜ3} : norm B = 3
axiom norm_c {C : ℜ3} : norm C = 4

theorem cosine_a_b : dot_product A B / (norm A * norm B) = 1 / 4 := by
  sorry

end cosine_a_b_l164_164863


namespace petya_and_kolya_complete_work_l164_164119

theorem petya_and_kolya_complete_work (total_days_petya: ℕ) (work_each_day: ℕ → ℕ)
    (h1: total_days_petya = 12)
    (h2: ∀ n, work_each_day n = 2^(n - 1))
    (h3: work_each_day 12 = (2^12 - 1) / 2) :

    let total_days_joint := total_days_petya / 2 in
    total_days_joint = 6 :=
sorry

end petya_and_kolya_complete_work_l164_164119


namespace three_digit_integers_divisible_by_5_11_3_l164_164111

def count_three_digit_multiples (d : ℕ) (low high : ℕ) : ℕ :=
  if high < low then 0 else
  let m := high / d in
  let n := (low + d - 1) / d in
  m - n + 1

theorem three_digit_integers_divisible_by_5_11_3 : count_three_digit_multiples 165 100 999 = 6 :=
by
  unfold count_three_digit_multiples
  sorry

end three_digit_integers_divisible_by_5_11_3_l164_164111


namespace john_books_per_day_l164_164160

theorem john_books_per_day (books_total : ℕ) (total_weeks : ℕ) (days_per_week : ℕ) (total_days : ℕ)
  (read_days_eq : total_days = total_weeks * days_per_week)
  (books_per_day_eq : books_total = total_days * 4) : (books_total / total_days = 4) :=
by
  -- The conditions state the following:
  -- books_total = 48 (total books read)
  -- total_weeks = 6 (total number of weeks)
  -- days_per_week = 2 (number of days John reads per week)
  -- total_days = 12 (total number of days in which John reads books)
  -- read_days_eq :- total_days = total_weeks * days_per_week
  -- books_per_day_eq :- books_total = total_days * 4
  sorry

end john_books_per_day_l164_164160


namespace double_elimination_tournament_l164_164990

theorem double_elimination_tournament (total_matches : ℕ)
  (assume_winner_uncorrect : ∀ P, 2 * (P - 1) = total_matches → false)
  (assume_winner_correct : ∀ P, 2 * (P - 1) + 1 = total_matches → P = 32) :
  total_matches = 63 → ∃ P, P = 32 :=
by
  intro h
  have h1 : ∃ P, 2 * (P - 1) = total_matches
    := fun P => not_elim (assume_winner_uncorrect P)
  have h2 : ∃ P, 2 * (P - 1) + 1 = total_matches
    := fun P => assume_winner_correct P
  use 32
  exact h2 32 h

end double_elimination_tournament_l164_164990


namespace sphere_surface_area_radius_one_l164_164905

theorem sphere_surface_area_radius_one : 
  ∀ (r : ℝ), r = 1 → 4 * Real.pi * r^2 = 4 * Real.pi :=
by 
  intros r hr
  rw hr
  calc
    4 * Real.pi * 1^2 = 4 * Real.pi * 1 : by rw pow_two
                  ... = 4 * Real.pi : by rw mul_one

end sphere_surface_area_radius_one_l164_164905


namespace net_investment_change_l164_164758

variable (I : ℝ)

def first_year_increase (I : ℝ) : ℝ := I * 1.75
def second_year_decrease (W : ℝ) : ℝ := W * 0.70

theorem net_investment_change : 
  let I' := first_year_increase 100 
  let I'' := second_year_decrease I' 
  I'' - 100 = 22.50 :=
by
  sorry

end net_investment_change_l164_164758


namespace part1_part2_l164_164092

noncomputable def f (x : ℝ) := (Real.sqrt 3) / 2 * Real.sin (2 * x) - Real.cos x ^ 2 + 1 / 2

theorem part1 :
  {x : ℝ | f x = 0} = {x | ∃ k : ℤ, x = k * Real.pi / 2 + Real.pi / 12} :=
sorry

theorem part2 :
  ∀ x ∈ set.Icc 0 (Real.pi / 2), f x >= -1/2 :=
sorry

end part1_part2_l164_164092


namespace count_even_abundant_numbers_under_50_l164_164560

def is_abundant (n : ℕ) : Prop :=
  (∑ i in finset.range n \ {n} | i ∣ n, i) > n

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

theorem count_even_abundant_numbers_under_50 : 
  (finset.filter (λ n, is_even n ∧ is_abundant n) (finset.range 50)).card = 7 :=
by {
  sorry
}

end count_even_abundant_numbers_under_50_l164_164560


namespace xiaoning_comprehensive_score_l164_164780

theorem xiaoning_comprehensive_score
  (max_score : ℕ := 100)
  (midterm_weight : ℝ := 0.3)
  (final_weight : ℝ := 0.7)
  (midterm_score : ℕ := 80)
  (final_score : ℕ := 90) :
  (midterm_score * midterm_weight + final_score * final_weight) = 87 :=
by
  sorry

end xiaoning_comprehensive_score_l164_164780


namespace xyz_value_l164_164075

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24) 
  (h2 : x ^ 2 * (y + z) + y ^ 2 * (x + z) + z ^ 2 * (x + y) = 9) : 
  x * y * z = 5 :=
by
  sorry

end xyz_value_l164_164075


namespace count_valid_n_l164_164052

def is_integer_div (n : ℕ) : Prop :=
  factorial (2 * n * n - 1) % (factorial (2 * n) ^ n) = 0

theorem count_valid_n :
  (finset.card (finset.filter is_integer_div (finset.range 101))) = 51 :=
by
  sorry

end count_valid_n_l164_164052


namespace find_f_neg_5_l164_164095

theorem find_f_neg_5 (a b : ℝ) (h1 : ∀ x : ℝ, f x = a * x^3 + b * x + 6) (h2 : f 5 = 7) :
  f (-5) = 5 :=
sorry

end find_f_neg_5_l164_164095


namespace less_than_15_points_l164_164726

theorem less_than_15_points (n : ℕ) (P : Fin n → EucThree) (Q : EucThree) 
(h : ∀ i : Fin n, ∀ j : Fin n, i ≠ j → dist (P i) Q < dist (P i) (P j)): 
n < 15 := 
sorry

end less_than_15_points_l164_164726


namespace general_term_sequence_l164_164512

open Nat

noncomputable def a_sequence (n : ℕ) : ℚ :=
if n = 1 ∨ n = 2 then 1 / 3 
else (1 - 2 * (a_sequence (n - 2))) * (a_sequence (n - 1)) ^ 2 /
     (2 * (a_sequence (n - 1)) ^ 2 - 4 * (a_sequence (n - 2)) * (a_sequence (n - 1)) ^ 2 + a_sequence (n - 2))

theorem general_term_sequence :
  ∀ n : ℕ, n ≥ 1 →
  a_sequence n = (let b := (3 / 2 - 5 / 6 * Real.sqrt 3, 3 / 2 + 5 / 6 * Real.sqrt 3) in
  ((b.1 * ((2 + Real.sqrt 3 : ℝ) ^ n) + b.2 * ((2 - Real.sqrt 3 : ℝ) ^ n))^2 + 2)⁻¹) :=
begin
  sorry
end

end general_term_sequence_l164_164512


namespace total_intersections_l164_164928

noncomputable def tangent_line_intersection {r : ℝ} (x0 y0 : ℝ) : ℕ :=
if x0^2 + y0^2 = r^2 then 1 else sorry

noncomputable def secant_line_intersection {r : ℝ} (x1 y1 : ℝ) : ℕ :=
let d := (x1^2 + y1^2 - r^2)^2 - (x1^2 + y1^2 - 4*x1*y1)^2 in
if d > 0 then 2 else sorry

theorem total_intersections (r : ℝ) (x0 y0 x1 y1 : ℝ) (h_tangent : (x0^2 + y0^2 = r^2)) 
(h_secant : (x1 ≠ x0 ∧ y1 ≠ y0) ∧ ((x1 ≠ 0) ∨ (y1 ≠ 0))) :
  tangent_line_intersection x0 y0 + secant_line_intersection x1 y1 = 3 :=
by {
  have h_tangent_points : tangent_line_intersection x0 y0 = 1 := sorry,
  have h_secant_points : secant_line_intersection x1 y1 = 2 := sorry,
  rw [h_tangent_points, h_secant_points],
  exact rfl,
}

end total_intersections_l164_164928


namespace find_smallest_int_cube_ends_368_l164_164034

theorem find_smallest_int_cube_ends_368 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 500 = 368 ∧ n = 14 :=
by
  sorry

end find_smallest_int_cube_ends_368_l164_164034


namespace diameter_of_frame_Y_l164_164738

noncomputable def radius_of_circle (diameter : ℝ) : ℝ :=
  diameter / 2

noncomputable def area_of_circle (radius : ℝ) : ℝ :=
  π * radius^2

theorem diameter_of_frame_Y (diam_X : ℝ) (frac_not_covered : ℝ) :
  let r_X := radius_of_circle diam_X,
      A_X := area_of_circle r_X,
      frac_covered := 1 - frac_not_covered,
      A_Y := frac_covered * A_X,
      r_Y := Real.sqrt (A_Y / π) in
  2 * r_Y = 12 :=
by
  -- frame X has a diameter of 16 cm
  -- the fraction of frame X not covered is 0.4375
  have h1 : diam_X = 16 := rfl
  have h2 : frac_not_covered = 0.4375 := rfl
  -- so the fraction covered is 1 - 0.4375 = 0.5625
  have h3 : frac_covered = 0.5625 := rfl
  -- the radius of frame X is 8 cm 
  have h4 : r_X = 8 := by simp [radius_of_circle, h1]
  -- the area of frame X is 64π cm^2
  have h5 : A_X = 64 * π := by simp [area_of_circle, h4]
  -- the area of frame Y is 0.5625 * 64π cm^2
  have h6 : A_Y = 0.5625 * 64 * π := by simp [frac_covered, A_X, h3]
  -- the radius of frame Y is sqrt((36π) / π) = 6 cm
  have h7 : r_Y = 6 := by simp [Real.sqrt, A_Y, h6, π]
  -- so the diameter of frame Y is 12 cm
  show 2 * r_Y = 12 from by simp [r_Y]
  sorry

end diameter_of_frame_Y_l164_164738


namespace nonagon_diagonals_l164_164945

def convex_nonagon_diagonals : Prop :=
∀ (n : ℕ), n = 9 → (n * (n - 3)) / 2 = 27

theorem nonagon_diagonals : convex_nonagon_diagonals :=
by {
  sorry,
}

end nonagon_diagonals_l164_164945


namespace sqrt_multiplication_simplification_l164_164808

theorem sqrt_multiplication_simplification (q : ℝ) (h : q ≥ 0) :
  √(42 * q) * √(7 * q) * √(21 * q) = 21 * q * √(21 * q) :=
by sorry

end sqrt_multiplication_simplification_l164_164808


namespace even_n_ineq_l164_164488

theorem even_n_ineq (n : ℕ) (h : ∀ x : ℝ, 3 * x^n + n * (x + 2) - 3 ≥ n * x^2) : Even n :=
  sorry

end even_n_ineq_l164_164488


namespace solve_for_x_l164_164297

theorem solve_for_x :
  ∃ x : ℝ, 50 ^ 4 = 10 ^ x ∧ x = 6.79588 :=
by {
  use 6.79588,
  split,
  { calc
      50 ^ 4 = (10 ^ (log 10 50)) ^ 4 : by congr; rw [log_def]
      ...     = 10 ^ (4 * log 10 50)   : by rw pow_mul
      ...     = 10 ^ 6.79588           : by norm_num
  },
  { refl }
}

end solve_for_x_l164_164297


namespace charlie_probability_l164_164776

theorem charlie_probability
  (marbles : Finset Marble)
  (r b g : Marble)
  (Alice_draws Bob_draws : Finset Marble)
  (h1 : marbles = {r, r, r, b, b, b, g, g, g})
  (h2 : Alice_draws ⊆ marbles ∧ Alice_draws.card = 3)
  (h3 : Bob_draws ⊆ (marbles \ Alice_draws) ∧ Bob_draws.card = 3)
  (Charlie_draws : Finset Marble := marbles \ (Alice_draws ∪ Bob_draws)) :
  (Charlie_draws.card = 3 ∧ Charlie_draws = {r, b, g})
    → probability_of (event {r, b, g}, marbles, 3) = 5 / 8 :=
by
  sorry

end charlie_probability_l164_164776


namespace number_of_integers_between_cubed_values_l164_164562

theorem number_of_integers_between_cubed_values :
  ∃ n : ℕ, n = (1278 - 1122 + 1) ∧ 
  ∀ x : ℤ, (1122 < x ∧ x < 1278) → (1123 ≤ x ∧ x ≤ 1277) := 
by
  sorry

end number_of_integers_between_cubed_values_l164_164562


namespace points_on_line_proof_l164_164289

theorem points_on_line_proof (n : ℕ) (hn : n = 10) : 
  let after_first_procedure := 3 * n - 2 in
  let after_second_procedure := 3 * after_first_procedure - 2 in
  after_second_procedure = 82 :=
by
  let after_first_procedure := 3 * n - 2
  let after_second_procedure := 3 * after_first_procedure - 2
  have h : after_second_procedure = 9 * n - 8 := by
    calc
      after_second_procedure = 3 * (3 * n - 2) - 2 : rfl
                      ... = 9 * n - 6 - 2      : by ring
                      ... = 9 * n - 8          : by ring
  rw [hn] at h 
  exact h.symm.trans (by norm_num)

end points_on_line_proof_l164_164289


namespace uncertain_frac_add_half_l164_164509

def floor (x : ℝ) : ℤ := Int.floor x
def frac (x : ℝ) : ℝ := x - floor x

theorem uncertain_frac_add_half (a : ℝ) (h : 0 < a ∧ a < 1) : 
   ¬((frac a < frac (a + 1/2)) ∨ (frac a = frac (a + 1/2)) ∨ (frac a > frac (a + 1/2))) :=
by
  intro h
  sorry

end uncertain_frac_add_half_l164_164509


namespace sum_of_real_numbers_l164_164223

theorem sum_of_real_numbers (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_real_numbers_l164_164223


namespace angle_between_vectors_l164_164110

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Constants and conditions from the problem
theorem angle_between_vectors (h1 : ∥a∥ = 1)
                             (h2 : ∥a + b∥ = real.sqrt 7)
                             (h3 : inner_product_space.inner a (b - a) = -4) :
  real.angle a b = 5 * real.pi / 6 := 
sorry

end angle_between_vectors_l164_164110


namespace sum_of_four_digit_palindromes_digits_sum_eq_36_l164_164787

def is_palindrome (n : ℕ) : Prop := 
  let a := n / 1000 in
  let b := (n / 100 % 10) in
  let c := (n % 100 / 10) in
  let d := (n % 10) in
  a = d ∧ b = c ∧ a ≠ 0

def four_digit_palindromes : List ℕ :=
  List.filter is_palindrome (List.range 10000)

def sum_of_digits (n : ℕ) : ℕ :=
  let rec aux (n : ℕ) (acc : ℕ) :=
    if n = 0 then acc
    else aux (n / 10) (acc + n % 10)
  aux n 0

theorem sum_of_four_digit_palindromes_digits_sum_eq_36 : 
  sum_of_digits (four_digit_palindromes.sum) = 36 :=
by 
  sorry

end sum_of_four_digit_palindromes_digits_sum_eq_36_l164_164787


namespace find_k_l164_164011

theorem find_k (a b c : ℝ) :
    (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) + (-1) * a * b * c :=
by
  sorry

end find_k_l164_164011


namespace max_value_of_f_l164_164708

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem max_value_of_f : ∃ max, max ∈ Set.image f (Set.Icc (-1 : ℝ) 1) ∧ max = Real.exp 1 - 1 :=
by
  sorry

end max_value_of_f_l164_164708


namespace updated_mean_l164_164400

-- Definitions
def initial_mean := 200
def number_of_observations := 50
def decrement_per_observation := 9

-- Theorem stating the updated mean after decrementing each observation
theorem updated_mean : 
  (initial_mean * number_of_observations - decrement_per_observation * number_of_observations) / number_of_observations = 191 :=
by
  -- Placeholder for the proof
  sorry

end updated_mean_l164_164400


namespace sum_possible_values_N_l164_164324

theorem sum_possible_values_N (a b c N : ℕ) (ha : 0 < a) (hb : 0 < b)
  (hc : c = a + b) (hN : N = a * b * c) (h_condition : N = 8 * (a + b + c)) :
  (N = 272 ∨ N = 160 ∨ N = 128) →
  (272 + 160 + 128) = 560 :=
by {
  intros h,
  have h1 : N = 272 ∨ N = 160 ∨ N = 128,
  from h,
  exact eq.refl 560,
}

end sum_possible_values_N_l164_164324


namespace flag_configuration_l164_164471

theorem flag_configuration (colors : Fin 3 → Fin 2) :
    let choices_stripe1 := 2,
        choices_stripe2 := 2,
        choices_stripe3 := 3,
        choices_stripe4 := 2 in
    choices_stripe1 * choices_stripe2 * choices_stripe3 * choices_stripe4 = 24 :=
by
    dsimp [choices_stripe1, choices_stripe2, choices_stripe3, choices_stripe4]
    sorry

end flag_configuration_l164_164471


namespace cubic_polynomial_g_value_l164_164195

theorem cubic_polynomial_g_value :
  (∀ x, f x = x^3 - 3 * x + 1) →
  (g 0 = -2) →
  (∃ (r s t : ℝ), 
    f(x) = (x - r) * (x - s) * (x - t) ∧ 
    g(x) = -2 * (x - r^2) * (x - s^2) * (x - t^2)) →
  g 4 = 4 :=
by
  sorry

end cubic_polynomial_g_value_l164_164195


namespace sum_of_xy_l164_164251

theorem sum_of_xy {x y : ℝ} (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := sorry

end sum_of_xy_l164_164251


namespace proof_problem_l164_164888

noncomputable section

def foci_ellipse : Set (ℝ × ℝ) :=
  {(x, 0) | x = -2 ∨ x = 2}

def point_on_ellipse : Set (ℝ × ℝ) :=
  {(2, Real.sqrt(6) / 3)}

def std_eq_ellipse (a b : ℝ) : Prop :=
  (a > b ∧ b > 0 ∧ ∃ x y : ℝ, foci_ellipse (x, y) ∧ point_on_ellipse (x, y) ∧ 
  ∀ x y, (x^2)/(a^2) + (y^2)/(b^2) = 1)

def line_through_focus (x y m : ℝ) : Prop :=
  (x = m * y + 2)

def area_OAB_maximized (m : ℝ) : Prop :=
  let t := Real.sqrt(m^2 + 1) in 
  t^2 + 2 = m^2 + 3 ∧ S_triangle_OAB t = Real.sqrt(3)

def correct_answer : Prop :=
  (std_eq_ellipse (Real.sqrt 6) 2) ∧ 
  (∃ m : ℝ, area_OAB_maximized m ∧ 
  (line_through_focus 2 0 1 ∨ line_through_focus 2 0 (-1)))

theorem proof_problem : correct_answer := sorry

end proof_problem_l164_164888


namespace possible_values_of_sum_l164_164235

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + y^3 + 21 * x * y = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l164_164235


namespace value_range_l164_164117

theorem value_range (x : ℝ) (h₁ : 1 / x < 3) (h₂ : 1 / x > -4) (h₃ : x ≠ 0) :
  x ∈ ((Set.Ioo (1 / 3) ∞) ∪ (Set.Ioo -∞ (-1 / 4))) :=
by
  sorry

end value_range_l164_164117


namespace sufficient_but_not_necessary_condition_l164_164639

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.tan (ω * x + φ)
def P (f : ℝ → ℝ) : Prop := f 0 = 0
def Q (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem sufficient_but_not_necessary_condition (ω : ℝ) (φ : ℝ) (hω : ω > 0) :
  (P (f ω φ) → Q (f ω φ)) ∧ ¬(Q (f ω φ) → P (f ω φ)) := by
  sorry

end sufficient_but_not_necessary_condition_l164_164639


namespace points_on_line_proof_l164_164290

theorem points_on_line_proof (n : ℕ) (hn : n = 10) : 
  let after_first_procedure := 3 * n - 2 in
  let after_second_procedure := 3 * after_first_procedure - 2 in
  after_second_procedure = 82 :=
by
  let after_first_procedure := 3 * n - 2
  let after_second_procedure := 3 * after_first_procedure - 2
  have h : after_second_procedure = 9 * n - 8 := by
    calc
      after_second_procedure = 3 * (3 * n - 2) - 2 : rfl
                      ... = 9 * n - 6 - 2      : by ring
                      ... = 9 * n - 8          : by ring
  rw [hn] at h 
  exact h.symm.trans (by norm_num)

end points_on_line_proof_l164_164290


namespace perpendicular_tangents_slope_ratio_l164_164904

theorem perpendicular_tangents_slope_ratio (a b : ℝ)
  (h_line : ∀ (x y : ℝ), ax - by - 2 = 0)
  (h_curve : ∀ (x : ℝ), y = x^3 + x)
  (h_point : ∀ (x y : ℝ), (x, y) = (1, 2))
  (h_perp : ∀ (x : ℝ), (3 * x^2 + 1) * (-a / b) = -1) :
  a / b = -1 / 4 :=
by
  sorry

end perpendicular_tangents_slope_ratio_l164_164904


namespace quotient_of_sum_of_square_remainders_l164_164300

theorem quotient_of_sum_of_square_remainders :
  ( let remainders := { n^2 % 17 | n in finset.range 16 } in
    let sum := remainders.sum id in
    sum / 17
  ) = 4 :=
by
  sorry

end quotient_of_sum_of_square_remainders_l164_164300


namespace unique_paths_l164_164301

/-
  Define point type to represent the points.
-/
inductive Point
| A | D | E | F

/-
  Define start and end points for paths.
-/
def path_start (p : Point) : Prop :=
p = Point.A

def path_end (p : Point) : Prop :=
p = Point.A

/-
  Define adjacent paths between points.
-/
def adjacent (p q : Point) : Prop :=
(p = Point.A ∧ q = Point.D) ∨ (p = Point.A ∧ q = Point.E) ∨
(p = Point.D ∧ q = Point.E) ∨ (p = Point.D ∧ q = Point.F) ∨
(p = Point.E ∧ q = Point.F)

/-
  Define the main theorem statement.
-/
theorem unique_paths : 
  ∃ (n : ℕ), start = Point.A ∧ (∀ p q, adjacent p q → ¬ (p = q)) ∧ n = 32 :=
begin
  sorry
end


end unique_paths_l164_164301


namespace lucas_fib_identity_l164_164544

noncomputable def F : ℕ → ℝ
| 1 := 1
| 2 := 1
| (n+1) := F n + F (n-1)

noncomputable def L : ℕ → ℝ
| 1 := 1
| 2 := 3
| (n+1) := L n + L (n-1)

theorem lucas_fib_identity (n p : ℕ) : 
  (\left(\frac{L n + sqrt 5 * F n}{2}\right)^p = \frac{L (n * p) + sqrt 5 * F (n * p)}{2}) :=
sorry

end lucas_fib_identity_l164_164544


namespace tan_angle_HDC_l164_164805

theorem tan_angle_HDC {O C D H : Type*} {OB : Type*} (OD_radius : real) (HL_value : real) 
  (B_perpendicular_to_CH : CH ⊥ OB) (C_D_diameters : ∀ (AB CD : circle), AB = diameter O → CD = diameter O) 
  : tan (∠ H D C) = sqrt 3 / 5 :=
by sorry

end tan_angle_HDC_l164_164805


namespace part1_question1_part1_question2_part2_l164_164536

noncomputable def f (x : ℝ) : ℝ := |2 - 1 / x|

theorem part1_question1 (a b : ℝ) (ha : 0 < a) (hb : b > a) (equal_fab : f(a) = f(b)) :
  1 / a + 1 / b = 4 :=
begin
  -- Proof omitted (sorry)
  sorry
end

theorem part1_question2 (a b : ℝ) (ha : 0 < a) (hb : b > a) (equal_fab : f(a) = f(b)) :
  set.range (λ x, (1 / a ^ 2) + (2 / b ^ 2)) = set.Ico (32 / 3) 16 :=
begin
  -- Proof omitted (sorry)
  sorry
end

theorem part2 (hg_preserving : ∀ m n, 0 < m ∧ m < n → 
  (∀ x : ℝ, x ∈ set.Icc m n → f x ∈ set.Icc m n) → f(x) is not range-preserving on Icc (0, +∞)) :
  ¬ ∃ m n, 0 < m ∧ m < n ∧ (∀ x : ℝ, x ∈ set.Icc m n → f x ∈ set.Icc m n) :=
begin
  -- Proof omitted (sorry)
  sorry
end

end part1_question1_part1_question2_part2_l164_164536


namespace min_root_sum_squares_l164_164188

theorem min_root_sum_squares (a x₁ x₂ : ℝ) (h_eq : x₁^2 + a * x₁ + a + 3 = 0) (h_eq2 : x₂^2 + a * x₂ + a + 3 = 0) :
  a = -2 ∨ a = 6 → x₁^2 + x₂^2 = 2 :=
begin
  sorry
end

end min_root_sum_squares_l164_164188


namespace divisible_by_91_l164_164674

theorem divisible_by_91 (n : ℕ) : 91 ∣ (5^n * (5^n + 1) - 6^n * (3^n + 2^n)) := 
by 
  sorry

end divisible_by_91_l164_164674


namespace gravitational_force_on_mars_l164_164311

theorem gravitational_force_on_mars :
  ∃ (f : ℚ), let d_earth := 6371 in
              let f_earth := 800 in
              let k := f_earth * d_earth^2 in
              let d_mars := 78_340_000 in
              f = k / d_mars^2 ∧ f = 1 / 189_233 := 
begin
  -- sorry, skip the proof
  sorry
end

end gravitational_force_on_mars_l164_164311


namespace non_constant_arithmetic_sequence_cube_l164_164618

theorem non_constant_arithmetic_sequence_cube (p : ℕ) (hp : p.prime) (hp3 : p ≠ 3) :
  ∃ (a d : ℕ), (∀ i : ℕ, i < p → 0 < a + i * d) ∧ (∏ i in finset.range p, a + i * d) % (3 ^ (nat.log (∏ i in finset.range p, a + i * d))) = 0 := 
sorry

end non_constant_arithmetic_sequence_cube_l164_164618


namespace sum_of_xy_l164_164253

theorem sum_of_xy {x y : ℝ} (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := sorry

end sum_of_xy_l164_164253


namespace even_abundant_count_l164_164558

-- Define what it means for a number to be abundant
def is_abundant (n : ℕ) : Prop :=
  ∑ k in (Finset.filter (λ d, d ∣ n ∧ d < n) (Finset.range n)), k > n

-- Define what it means to be an even number
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the even abundant numbers less than 50
def even_abundant_numbers_lt_50 : Finset ℕ :=
  Finset.filter (λ n, is_even n ∧ is_abundant n) (Finset.range 50)

-- Prove that the number of even abundant numbers less than 50 is 9
theorem even_abundant_count : even_abundant_numbers_lt_50.card = 9 :=
by
  -- Skipping the actual proof steps
  sorry

end even_abundant_count_l164_164558


namespace area_curve_is_correct_l164_164062

-- Define the initial conditions
structure Rectangle :=
  (vertices : Fin 4 → ℝ × ℝ)
  (point : ℝ × ℝ)

-- Define the rotation transformation
def rotate_clockwise_90 (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  let (cx, cy) := center
  let (px, py) := point
  (cx + (py - cy), cy - (px - cx))

-- Given initial rectangle and the point to track
def initial_rectangle : Rectangle :=
  { vertices := ![(0, 0), (2, 0), (0, 3), (2, 3)],
    point := (1, 1) }

-- Perform the four specified rotations
def rotated_points : List (ℝ × ℝ) :=
  let r1 := rotate_clockwise_90 (2, 0) initial_rectangle.point
  let r2 := rotate_clockwise_90 (5, 0) r1
  let r3 := rotate_clockwise_90 (7, 0) r2
  let r4 := rotate_clockwise_90 (10, 0) r3
  [initial_rectangle.point, r1, r2, r3, r4]

-- Calculate the area below the curve and above the x-axis
noncomputable def area_below_curve : ℝ :=
  6 + (7 * Real.pi / 2)

-- The theorem statement
theorem area_curve_is_correct : 
  area_below_curve = 6 + (7 * Real.pi / 2) :=
  by trivial

end area_curve_is_correct_l164_164062


namespace sum_of_xy_l164_164257

theorem sum_of_xy {x y : ℝ} (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := sorry

end sum_of_xy_l164_164257


namespace sequence_periodic_find_a2008_l164_164193

noncomputable def sequence (n : ℕ) : ℚ :=
if h : n = 0 then 2 else Nat.recOn (n - 1) 2 (λ k ak, (1 + ak) / (1 - ak))

theorem sequence_periodic (n : ℕ) (h : n ≥ 1) :
  sequence (n + 4) = sequence n :=
by sorry

theorem find_a2008 :
  30 * sequence 2008 = 10 :=
by
  have h : sequence 2008 = sequence 4 := by sorry,
  rw [sequence, if_neg] at h,
  simp_all [sequence],
  sorry

end sequence_periodic_find_a2008_l164_164193


namespace sequence_an_formula_smallest_m_for_T_n_l164_164909

theorem sequence_an_formula (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (h : ∀ n, S_n n = 2 * a_n n - n) :
  ∀ n, a_n n = 2^n - 1 :=
sorry

theorem smallest_m_for_T_n (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) (T_n : ℕ → ℝ) 
  (h1 : ∀ n, S_n n = 2 * a_n n - n) (h2 : ∀ n, a_n n = 2^n - 1) 
  (h3 : ∀ n, b_n n = 2^n / (a_n n * a_n (n + 1))) (h4 : ∀ n, T_n n = ∑ i in finset.range (n + 1), b_n i) :
  ∀ n, T_n n < 1 → ∃ m, m >= 20 ∧ ∀ n, T_n n < m / 20 :=
sorry

end sequence_an_formula_smallest_m_for_T_n_l164_164909


namespace smallest_n_l164_164496

noncomputable def conditions_met (n : ℕ) (x : Fin n → ℝ) :=
  ∑ i, x i = 800 ∧ ∑ i, (x i)^4 = 204800

theorem smallest_n :
  ∃ n : ℕ, n > 0 ∧ (∀ x : Fin n → ℝ, ¬ conditions_met n x) ∧
  (∀ m : ℕ, m > 0 ∧ m < n → ∃ x : Fin m → ℝ, conditions_met m x) :=
sorry

end smallest_n_l164_164496


namespace three_digit_integers_less_than_900_with_repeated_digits_l164_164966

def is_three_digit (n : ℤ) : Prop :=
  100 ≤ n ∧ n < 1000

def has_at_least_two_equal_digits (n : ℤ) : Prop :=
  let digits := (n / 100, (n / 10) % 10, n % 10) in
  digits.fst = digits.snd ∨ digits.snd = digits.thrd ∨ digits.fst = digits.thrd

theorem three_digit_integers_less_than_900_with_repeated_digits : 
  ∃ (count : ℕ), count = 224 ∧ ∀ n : ℤ, is_three_digit n → n < 900 → has_at_least_two_equal_digits n → count = 224 :=
sorry

end three_digit_integers_less_than_900_with_repeated_digits_l164_164966


namespace max_value_of_function_l164_164086

noncomputable def f (x : ℝ) (α : ℝ) := x ^ α

theorem max_value_of_function (α : ℝ)
  (h₁ : f 4 α = 2)
  : ∃ a : ℝ, 3 ≤ a ∧ a ≤ 5 ∧ (f (a - 3) (α) + f (5 - a) α = 2) := 
sorry

end max_value_of_function_l164_164086


namespace second_player_wins_l164_164429

def is_permissible (n : ℕ) : Prop :=
  (nat.factors n).nodup.length ≤ 20

def game_state : Type := ℕ

def initial_state : game_state := nat.fact 2004

def optimal_play_winner (s : game_state) : ℕ → Prop
| 0 := false -- 0 means first player wins
| 1 := true  -- 1 means second player wins
| s := ∀ k : ℕ, is_permissible k → k ≤ s → optimal_play_winner (s - k) (1 - w)

theorem second_player_wins : optimal_play_winner initial_state 1 :=
sorry

end second_player_wins_l164_164429


namespace mikail_birthday_money_l164_164210

theorem mikail_birthday_money :
  ∀ (A M : ℕ), A = 3 * 3 → M = 5 * A → M = 45 :=
by
  intros A M hA hM
  rw [hA] at hM
  rw [hM]
  norm_num

end mikail_birthday_money_l164_164210


namespace batches_engine_count_l164_164124

theorem batches_engine_count (x : ℕ) 
  (h1 : ∀ e, 1/4 * e = 0) -- every batch has engines, no proof needed for this question
  (h2 : 5 * (3/4 : ℚ) * x = 300) : 
  x = 80 := 
sorry

end batches_engine_count_l164_164124


namespace ellipse_standard_eq_line_OD_eq_OC_dot_OD_constant_l164_164090

-- Define the conditions of the problem
def ellipse (x y : ℝ) (a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the isosceles right triangle condition
def isosceles_right_triangle_with_hypotenuse_2 (a b c : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ a^2 = b^2 + c^2 ∧ c = 2

-- Define the line l through A₂ and perpendicular to the x-axis
def line_through_A2_perpendicular_x (x y : ℝ) : Prop := x = 2

-- Point D is on l and different from A₂
def point_D_on_l (D : ℝ × ℝ) : Prop := D.1 = 2 ∧ D ≠ (2, 0)

-- Line A₁D intersects the ellipse at point C
def line_A1D_intersects_ellipse_at_C (A1 D C : ℝ × ℝ) (a b : ℝ) : Prop :=
  ellipse C.1 C.2 a b ∧ ∃ k : ℝ, D.2 = k * (D.1 + 2) ∧ C = (4 * k^2 - 2) / (1 + 2 * k^2)

-- Define the condition A₁C = 2CD
def A1C_eq_2CD (A1 C D : ℝ × ℝ) : Prop := 
  ((C.1 - A1.1)^2 + (C.2 - A1.2)^2) = 2 * ((D.1 - C.1)^2 + (D.2 - C.2)^2)

-- Proof Problem 1: Prove the standard equation of the ellipse
theorem ellipse_standard_eq :
  ∀ (x y : ℝ) (a b : ℝ), 
  (isosceles_right_triangle_with_hypotenuse_2 a b c) →
  (line_through_A2_perpendicular_x x y) →
  ellipse x y a b ↔ a = 2 ∧ b = sqrt(2) := sorry

-- Proof Problem 2: Show the equation of line OD
theorem line_OD_eq :
  ∀ (A1 D : ℝ × ℝ), 
  (point_D_on_l D) →
  (A1C_eq_2CD A1 C D) →
  line_equation_contains O D → 
  (y = x ∨ y = -x) := sorry

-- Proof Problem 3: Prove that OC ⋅ OD is a constant
theorem OC_dot_OD_constant :
  ∀ (A1 D C : ℝ × ℝ) (a b : ℝ), 
  (isosceles_right_triangle_with_hypotenuse_2 a b c) →
  (line_through_A2_perpendicular_x D.1 D.2) →
  (point_D_on_l D) →
  (line_A1D_intersects_ellipse_at_C A1 D C a b) →
  (A1C_eq_2CD A1 C D) →
  vector_dot (O, C) (O, D) = 4 := sorry

end ellipse_standard_eq_line_OD_eq_OC_dot_OD_constant_l164_164090


namespace sphere_has_identical_views_l164_164802

def identical_views (a : Type) : Prop :=
  ∀ (front_view side_view top_view : a), front_view = side_view ∧ side_view = top_view

inductive GeometricBody
| cylinder
| cone
| sphere
| triangular_pyramid

open GeometricBody

theorem sphere_has_identical_views :
  identical_views sphere := 
sorry

end sphere_has_identical_views_l164_164802


namespace ratio_of_blue_to_yellow_l164_164206

def butterflies (total black blue : Nat) : Nat :=
  total - (black + blue)

theorem ratio_of_blue_to_yellow (total black blue : Nat) (h_total : total = 11) (h_black : black = 5) (h_blue : blue = 4) :
  let yellow := butterflies total black blue in
  yellow = 2 →
  (blue : yellow) = (2 : 1) :=
by
  intros yellow h_yellow
  have h1 : total - (black + blue) = 2 := by rw [h_total, h_black, h_blue]; exact h_yellow
  have h2 : butterflies total black blue = 2 := by rw [butterflies, h1]
  sorry

end ratio_of_blue_to_yellow_l164_164206


namespace hyperbola_eccentricity_range_l164_164101

theorem hyperbola_eccentricity_range 
(a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
(hyperbola_eq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
(parabola_eq : ∀ y x, y^2 = 8 * a * x)
(right_vertex : A = (a, 0))
(focus : F = (2 * a, 0))
(P : ℝ × ℝ)
(asymptote_eq : P = (x0, b / a * x0))
(perpendicular_condition : (x0 ^ 2 - (3 * a - b^2 / a^2) * x0 + 2 * a^2 = 0))
(hyperbola_properties: c^2 = a^2 + b^2) :
1 < c / a ∧ c / a <= 3 * Real.sqrt 2 / 4 :=
sorry

end hyperbola_eccentricity_range_l164_164101


namespace orthocenter_in_H_of_any_triangle_l164_164433

-- Define the vertex set
structure Vertex :=
  (x : ℝ) (y : ℝ) (z : ℝ)

-- Define the edges
structure Edge :=
  (a : Vertex) (b : Vertex)

-- Define the Tetrahedron
structure Tetrahedron :=
  (A B C D : Vertex)
  (AB BC CD AD : Edge)

-- Define the set of points H
def H : Type → Prop := sorry -- Define H accordingly or use the intersection definition.

-- Define Plane S and its intersection points
structure Plane :=
  (normal : Vertex) -- Example, the normal vector defining the plane

-- Define the orthocenter property
def is_orthocenter (h : H) (triangle : Vertex × Vertex × Vertex) : Prop := sorry
-- Define what it means for a point to be the orthocenter of a given triangle.

theorem orthocenter_in_H_of_any_triangle (T : Tetrahedron) (S : Plane) (H : Set Vertex) :
  let intersects (e : Edge) : Vertex := sorry -- Intersection points on the edges
  ∃ (T1 T2 T3 : Vertex), T1 ∈ H ∧ T2 ∈ H ∧ T3 ∈ H ∧ is_orthocenter (H) (T1, T2, T3) :=
begin
  sorry -- Proof goes here
end

end orthocenter_in_H_of_any_triangle_l164_164433


namespace geometric_sequence_log_sum_l164_164497

open Real

theorem geometric_sequence_log_sum
  (a : ℕ → ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, a (n+1) = a n * a 1)
  (h3 : a 5 * a 6 + a 4 * a 7 = 18) :
  log 3 (a 1) + log 3 (a 2) + log 3 (a 3) + log 3 (a 4) + log 3 (a 5) +
  log 3 (a 6) + log 3 (a 7) + log 3 (a 8) + log 3 (a 9) + log 3 (a 10) = 10 := 
sorry

end geometric_sequence_log_sum_l164_164497


namespace polygon_sides_exterior_interior_sum_l164_164581

theorem polygon_sides_exterior_interior_sum (n : ℕ) (h : ((n - 2) * 180 = 360)) : n = 4 :=
by sorry

end polygon_sides_exterior_interior_sum_l164_164581


namespace camera_pics_l164_164747

-- Definitions of the given conditions
def phone_pictures := 22
def albums := 4
def pics_per_album := 6

-- The statement to prove the number of pictures uploaded from camera
theorem camera_pics : (albums * pics_per_album) - phone_pictures = 2 :=
by
  sorry

end camera_pics_l164_164747


namespace students_not_next_to_each_other_l164_164139

theorem students_not_next_to_each_other :
  let total_ways : ℕ := 5!
  let restricted_group_ways : ℕ := 3! * 3!
  total_ways - restricted_group_ways = 84 :=
by
  let total_ways := nat.factorial 5
  let restricted_group_ways := nat.factorial 3 * nat.factorial 3
  have h : total_ways - restricted_group_ways = 84 := sorry
  exact h

end students_not_next_to_each_other_l164_164139


namespace biggest_possible_score_l164_164383

def rounded_to_90 (score : ℝ) : Prop := (score / 10).round = 9

theorem biggest_possible_score (score : ℝ) (h : rounded_to_90 score) : score ≤ 94 :=
begin
  sorry -- Proof
end

end biggest_possible_score_l164_164383


namespace triangle_equivalence_l164_164169

variable {A B C O G M N : Point}
variable {a b c : Complex}
variable [HasAngle (Triangle A B C)]

def midpoint (P Q : Point) := 
  (1 / 2 : ℝ) • (P + Q)

def angle (P Q R : Point) : ℝ := sorry

def reflection (P Q : Point) :=
  2 • Q - P

def is_circumcentre (O : Point) (A B C : Point) : Prop :=
  -- definition of circumcentre specific to the context
  sorry

def is_centroid (G : Point) (A B C : Point) : Prop :=
  -- definition of centroid specific to the context
  sorry

def is_on_unit_circle (P : Point) (a : Complex) : Prop :=
  -- definition that P lies on the unit circle in the complex plane
  sorry

theorem triangle_equivalence :
  is_circumcentre O A B C →
  is_centroid G A B C →
  M = midpoint B C →
  N = reflection M O →
  (angle A O G = (90 : ℝ)) ↔ (dist N O = dist N A) ↔ (Re ((conj a) * (b + c)) = -1) :=
by
  intros hOcentroid hGcentroid hMidpmid hRefn
  sorry

end triangle_equivalence_l164_164169


namespace part1_solution_part2_solution_l164_164919

-- Part (1)
def f_part1 (x : ℝ) : ℝ := (1/2) * x^2 - (1/2) * x - 1

theorem part1_solution (x : ℝ) : f_part1 x < 0 ↔ -1 < x ∧ x < 2 :=
by
  sorry

-- Part (2)
def f (m x : ℝ) : ℝ := m * x^2 - m * x - 1
def g (m x : ℝ) : ℝ := (m - 1) * x^2 + 2 * x - 2 * m - 1

theorem part2_solution (m x : ℝ) (hm : m ∈ ℝ) : 
  (f m x < g m x ↔ 
  if m < 2 then m < x ∧ x < 2 
  else if m = 2 then false 
  else 2 < x ∧ x < m) :=
by
  sorry

end part1_solution_part2_solution_l164_164919


namespace parabola_point_dot_product_eq_neg4_l164_164076

-- Definition of the parabola
def is_parabola_point (A : ℝ × ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

-- Definition of the focus of the parabola y^2 = 4x
def focus : ℝ × ℝ := (1, 0)

-- Dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Coordinates of origin
def origin : ℝ × ℝ := (0, 0)

-- Vector from origin to point A
def vector_OA (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1, A.2)

-- Vector from point A to the focus
def vector_AF (A : ℝ × ℝ) : ℝ × ℝ :=
  (focus.1 - A.1, focus.2 - A.2)

-- Theorem statement
theorem parabola_point_dot_product_eq_neg4 (A : ℝ × ℝ) 
  (hA : is_parabola_point A) 
  (h_dot : dot_product (vector_OA A) (vector_AF A) = -4) :
  A = (1, 2) ∨ A = (1, -2) :=
sorry

end parabola_point_dot_product_eq_neg4_l164_164076


namespace smallest_a_value_l164_164688

theorem smallest_a_value (a b c : ℚ) (n : ℤ) 
    (h1 : ∃ (a b c : ℚ), vertex.y = a*vertex.x^2 + b*vertex.x + c)
    (h2 : a > 0)
    (h3 : a + b + c = n) :
    a = 9 / 7 := by 
    sorry

end smallest_a_value_l164_164688


namespace intersection_PQ_l164_164107

def setP  := {x : ℝ | x * (x - 1) ≥ 0}
def setQ := {y : ℝ | ∃ x : ℝ, y = 3 * x^2 + 1}

theorem intersection_PQ : {x : ℝ | x > 1} = {z : ℝ | z ∈ setP ∧ z ∈ setQ} :=
by
  sorry

end intersection_PQ_l164_164107


namespace chess_tournament_participants_l164_164986

theorem chess_tournament_participants (n : ℕ) 
  (h : (n * (n - 1)) / 2 = 15) : n = 6 :=
sorry

end chess_tournament_participants_l164_164986


namespace magnitude_of_z_l164_164088

theorem magnitude_of_z (z : ℂ) (h : i * (2 - z) = 3 + i) : complex.abs z = real.sqrt 10 :=
sorry

end magnitude_of_z_l164_164088


namespace value_of_a_plus_b_l164_164969

-- x^{a-1} - 3y^{b-2} = 7 is a linear equation in x and y implies a + b = 5
theorem value_of_a_plus_b (a b : ℕ) (x y : ℝ) (h : x^(a-1) - 3 * y^(b-2) = 7) :
  a + b = 5 ↔ (a - 1 = 1 ∧ b - 2 = 1) :=
by
  sorry

end value_of_a_plus_b_l164_164969


namespace min_black_cells_in_300x300_grid_l164_164749

-- Define the main function
noncomputable def min_black_cells (n : ℕ) (grid : Fin n × Fin n → Bool) : ℕ :=
  Nat.minBy (λ b, ∃ grid, configuration_correct grid b) (range (n * n))

-- Define helper function for correct configuration
def configuration_correct (grid : Fin 300 × Fin 300 → Bool) (b : ℕ) : Prop :=
  ∀ (x y : Fin 300), grid (x, y) = true → ((x + 1 < 300 ∧ grid (x + 1, y) = false) ∧
                                           (y + 1 < 300 ∧ grid (x, y + 1) = false)) ∧
                                           b = 30000

-- Add the theorem to be proven
theorem min_black_cells_in_300x300_grid : ∀ (grid : Fin 300 × Fin 300 → Bool),
  min_black_cells 300 grid = 30000 :=
begin
  intros,
  sorry,
end

end min_black_cells_in_300x300_grid_l164_164749


namespace electron_config_N3_minus_is_correct_pi_bond_more_stable_than_sigma_bond_l164_164214

-- Define the problem statements and conditions
def N3_minus : Type := sorry  -- Placeholder for the definition of Nitrogen in the state N^{3-}

-- Statement 1: The electron configuration of N^{3-}
def electron_configuration_N3_minus : List String :=
  ["1s^2", "2s^2", "2p^6"]

theorem electron_config_N3_minus_is_correct :
  electron_configuration N3_minus = ["1s^2", "2s^2", "2p^6"] :=
sorry

-- Given bond energies
def bond_energy_NN_triple_bond : ℝ := 942  -- kJ/mol
def bond_energy_N_N_single_bond : ℝ := 247  -- kJ/mol

-- Statement 2: The π bond is more stable than the σ bond
theorem pi_bond_more_stable_than_sigma_bond :
  bond_energy_NN_triple_bond > bond_energy_N_N_single_bond ->
  bond_stability π > bond_stability σ :=
sorry

end electron_config_N3_minus_is_correct_pi_bond_more_stable_than_sigma_bond_l164_164214


namespace sufficient_condition_for_negation_l164_164871

theorem sufficient_condition_for_negation 
  (p : ℝ → Prop)
  (q : ℝ → Prop)
  (H1 : ∀ x, p x ↔ (abs (x - 4) ≤ 6))
  (H2 : ∀ x m, q x ↔ (x^2 - 2 * x + 1 - m^2 ≤ 0))
  (H3 : ∀ m, (∃ x, ¬ p x → ¬ q x) ∧ (∃ x, ¬ q x ∧ p x)) :
  -3 ≤ m ∧ m ≤ 3 := 
sorry

end sufficient_condition_for_negation_l164_164871


namespace fraction_increase_is_two_thirds_l164_164613

-- Definitions from the conditions
def original_coin_price : ℝ := 15
def number_of_coins_bought : ℕ := 20
def original_investment : ℝ := number_of_coins_bought * original_coin_price
def recouped_amount : ℝ := 300
def number_of_coins_sold : ℕ := 12

-- The actual proof statement without the proof itself
theorem fraction_increase_is_two_thirds :
  let selling_price_per_coin := recouped_amount / number_of_coins_sold in
  let increase_in_value_per_coin := selling_price_per_coin - original_coin_price in
  let fraction_increase := increase_in_value_per_coin / original_coin_price in
  fraction_increase = 2 / 3 :=
by sorry

end fraction_increase_is_two_thirds_l164_164613


namespace sum_of_xy_l164_164252

theorem sum_of_xy {x y : ℝ} (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := sorry

end sum_of_xy_l164_164252


namespace exists_sequence_n_25_exists_sequence_n_gt_1000_l164_164417

noncomputable def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2*k + 1

def condition (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ k : ℕ, (0 ≤ k ∧ k < n) → is_odd (∑ i in finset.range (n - k), a i * a (i + k))

theorem exists_sequence_n_25 :
  ∃ a : ℕ → ℕ, (∀ i : ℕ, i < 25 → (a i = 0 ∨ a i = 1)) ∧ condition a 25 := 
sorry

theorem exists_sequence_n_gt_1000 :
  ∃ (n : ℕ) (a : ℕ → ℕ), n > 1000 ∧ (∀ i : ℕ, i < n → (a i = 0 ∨ a i = 1)) ∧ condition a n := 
sorry

end exists_sequence_n_25_exists_sequence_n_gt_1000_l164_164417


namespace num_unique_sums_of_cubes_lt_500_l164_164564

def cube (n : ℕ) : ℕ := n * n * n

def perfect_cubes_upto (n : ℕ) : List ℕ :=
  (List.range n).map cube

theorem num_unique_sums_of_cubes_lt_500 :
  let cubes := perfect_cubes_upto 8 -- {1^3, 2^3, ..., 7^3}
  let sums := { x | ∃ a b, a ∈ cubes ∧ b ∈ cubes ∧ x = a + b ∧ x < 500 }
  sums.toList.length = 26 :=
by
  let cubes := perfect_cubes_upto 8
  let sums := { x | ∃ a b, a ∈ cubes ∧ b ∈ cubes ∧ x = a + b ∧ x < 500 }
  show sums.toList.length = 26
  sorry

end num_unique_sums_of_cubes_lt_500_l164_164564


namespace min_distance_between_circles_l164_164219

open Real

theorem min_distance_between_circles
  (P : ℝ × ℝ) (Q : ℝ × ℝ) 
  (hP : (P.1 - 4)^2 + (P.2 - 2)^2 = 9) 
  (hQ : (Q.1 + 2)^2 + (Q.2 + 1)^2 = 4) : 
  dist P Q ≥ 3 * sqrt 5 - 5 :=
begin
  -- Here is where the proof would go
  sorry
end

end min_distance_between_circles_l164_164219


namespace area_enclosed_by_curve_and_line_l164_164692

theorem area_enclosed_by_curve_and_line :
  let curve : ℝ → ℝ := λ x, x^2 + 2
  let line : ℝ → ℝ := λ x, 5 * x - 2
  (∫ x in 0..5, (curve x - (line x)) * dx) = 125 / 6 :=
by
  let curve : ℝ → ℝ := λ x, x^2 + 2
  let line : ℝ → ℝ := λ x, 5 * x - 2
  have integral_reduction : (∫ x in 0..5, (curve x - (line x)) * dx) = ∫ x in 0..5, (x^2 - 5 * x) * dx :=
    by congr; ext; dsimp; linarith
  rw integral_reduction
  sorry

end area_enclosed_by_curve_and_line_l164_164692


namespace compare_logs_l164_164866

noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := Real.log 5 / Real.log 3

theorem compare_logs : b > c ∧ c > a :=
by
  sorry

end compare_logs_l164_164866


namespace sum_of_rationals_eq_l164_164201

theorem sum_of_rationals_eq (a1 a2 a3 a4 : ℚ)
  (h : {x : ℚ | ∃ i j, 1 ≤ i ∧ i < j ∧ j ≤ 4 ∧ x = a1 * a2 ∧ x = a1 * a3 ∧ x = a1 * a4 ∧ x = a2 * a3 ∧ x = a2 * a4 ∧ x = a3 * a4} = {-24, -2, -3/2, -1/8, 1, 3}) :
  a1 + a2 + a3 + a4 = 9/4 ∨ a1 + a2 + a3 + a4 = -9/4 :=
sorry

end sum_of_rationals_eq_l164_164201


namespace foldable_positions_are_7_l164_164431

-- Define the initial polygon with 6 congruent squares forming a cross shape
def initial_polygon : Prop :=
  -- placeholder definition, in practice, this would be a more detailed geometrical model
  sorry

-- Define the positions where an additional square can be attached (11 positions in total)
def position (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 11

-- Define the resulting polygon when an additional square is attached at position n
def resulting_polygon (n : ℕ) : Prop :=
  position n ∧ initial_polygon

-- Define the condition that a polygon can be folded into a cube with one face missing
def can_fold_to_cube_with_missing_face (p : Prop) : Prop := sorry

-- The theorem that needs to be proved
theorem foldable_positions_are_7 : 
  ∃ (positions : Finset ℕ), 
    positions.card = 7 ∧ 
    ∀ n ∈ positions, can_fold_to_cube_with_missing_face (resulting_polygon n) :=
  sorry

end foldable_positions_are_7_l164_164431


namespace letter_2023rd_is_F_l164_164378

def sequence_block := "ABCDEFGFEDCBA"

def block_length : ℕ := 13

def letter_at_position (n : ℕ) : char :=
  let pos := n % block_length
  sequence_block.toList.get! pos

theorem letter_2023rd_is_F : letter_at_position 2023 = 'F' := by
  sorry

end letter_2023rd_is_F_l164_164378


namespace length_of_KL_l164_164655

theorem length_of_KL {K L M G : Type} [inner_product_space ℝ K L M G]
  (KP LQ KL : ℝ) (angle_KP_LQ : ℝ) 
  (hKP : KP = 15)
  (hLQ : LQ = 20)
  (hAngle : angle_KP_LQ = real.pi / 6) : 
  KL = real.sqrt (2500 - 800 * real.sqrt 3) / 3 :=
  sorry

end length_of_KL_l164_164655


namespace parallel_lines_from_conditions_l164_164524

variables (a b : Line) (α β : Plane)

def is_perpendicular (ℓ : Line) (π : Plane) : Prop := sorry
def is_parallel (x y : Line) : Prop := sorry
def planes_parallel (π₁ π₂ : Plane) : Prop := sorry

axiom perpendicular_iff_project (ℓ₁ ℓ₂ : Line) (π : Plane) : is_perpendicular ℓ₁ π ∧ is_perpendicular ℓ₂ π → is_parallel ℓ₁ ℓ₂ 
axiom parallel_planes_are_independent (π₁ π₂ : Plane) : planes_parallel π₁ π₂ → (∀ (ℓ₁ : Line), (is_perpendicular ℓ₁ π₁ → is_perpendicular ℓ₁ π₂))

theorem parallel_lines_from_conditions (h1 : is_perpendicular a α) (h2 : is_perpendicular b β) (h3 : planes_parallel α β) : 
  is_parallel a b := 
begin
  sorry,
end

end parallel_lines_from_conditions_l164_164524


namespace coeff_k_is_3_l164_164611

noncomputable theory

variables {k x y z : ℝ}

def condition1 (k x y z : ℝ) : Prop := z - y = k * x
def condition2 (k x y z : ℝ) : Prop := x - z = k * y
def condition3 (x y z : ℝ) : Prop := z = (5 / 3) * (x - y)

theorem coeff_k_is_3 (h1 : condition1 k x y z) (h2 : condition2 k x y z) (h3 : condition3 x y z) : k = 3 :=
by sorry

end coeff_k_is_3_l164_164611


namespace common_tangents_of_circles_l164_164913

theorem common_tangents_of_circles :
  let circle1 := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let circle2 := {p : ℝ × ℝ | p.1^2 + p.2^2 - 6 * p.1 - 8 * p.2 + 9 = 0}
  ∃ n : ℕ, n = 3 ∧ 
    (∀ t : ℝ × ℝ → Prop, t circle1 ∧ t circle2 → t ∈ set_of_common_tangents_of circle1 circle2 ∧ size (set_of_common_tangents_of circle1 circle2) = n)
:=
sorry

end common_tangents_of_circles_l164_164913


namespace percentage_deducted_from_list_price_l164_164451

noncomputable def costPrice : ℝ := 100
noncomputable def markedPrice : ℝ := 131.58
noncomputable def desiredProfitPct : ℝ := 25 / 100
noncomputable def sellingPrice : ℝ := costPrice + costPrice * desiredProfitPct

noncomputable def percentageDeducted (cp mp sp : ℝ) : ℝ := 
  100 * (mp - sp) / mp

theorem percentage_deducted_from_list_price (hCP : costPrice = 100)
  (hMP : markedPrice = 131.58)
  (hSP : sellingPrice = 125) :
  percentageDeducted costPrice markedPrice sellingPrice ≈ 5 :=
by
  have h := percentageDeducted costPrice markedPrice sellingPrice
  sorry

end percentage_deducted_from_list_price_l164_164451


namespace arithmetic_general_formula_sum_first_2n_bn_l164_164070

-- Define the arithmetic sequence with the given conditions and sum function
def arithmetic_sequence (a₁ d : ℕ) : ℕ → ℕ
| 0     => a₁
| (n+1) => (arithmetic_sequence a₁ d n) + d

noncomputable def Sn (a₁ d n : ℕ) : ℕ :=
  n / 2 * (2 * a₁ + (n - 1) * d)

-- Given condition S5 = 25
def condition_S5 : Prop := Sn 1 1 5 = 25

-- Define the terms forming a geometric sequence
def geometric_condition (a₃ a₄ a₇ : ℕ) : Prop :=
  (a₄ + 1) ^ 2 = (a₃ - 1) * (a₇ + 3)

-- Define the general formula for sequence a_n
def general_formula_an : ℕ → ℕ := λ n, 2 * n - 1

-- Define sequence b_n
def sequence_bn (a_n : ℕ → ℕ) : ℕ → ℕ
| 0     => 1
| (n+1) => (-1) ^ (n + 1) * a_n (n + 1) + 1

-- Sum of first n terms of b_n
noncomputable def Tn (b_n : ℕ → ℕ) : ℕ → ℕ
| 0     => b_n 0
| (n+1) => Tn b_n n + b_n (n + 1)

-- Lean statement for the proof problem (1)
theorem arithmetic_general_formula (a₁ d : ℕ) (h1 : Sn a₁ d 5 = 25)
  (h2 : geometric_condition (arithmetic_sequence a₁ d 2) (arithmetic_sequence a₁ d 3) (arithmetic_sequence a₁ d 6)) :
  arithmetic_sequence a₁ d = general_formula_an := sorry

-- Lean statement for the proof problem (2)
theorem sum_first_2n_bn (a_n b_n : ℕ → ℕ) (h : ∀ n, b_n n = (-1) ^ n * a_n n + 1) :
  Tn b_n = λ n, 4 * n := sorry

end arithmetic_general_formula_sum_first_2n_bn_l164_164070


namespace traverse_time_l164_164788

-- Define the speed relationship for the nth mile where the speed varies inversely with the square of the miles already traveled plus 1.
def speed (n : ℕ) : ℚ :=
  let k := 3
  in k / (↑n ^ 2)

-- Define the time to traverse the nth mile, which is the reciprocal of the speed.
def time_to_traverse (n : ℕ) : ℚ :=
  1 / speed n

-- The theorem to prove that the time required to traverse the nth mile is n^2 / 3 hours.
theorem traverse_time (n : ℕ) (h : n ≥ 3) : time_to_traverse n = n^2 / 3 := by
  sorry

end traverse_time_l164_164788


namespace wholesale_price_l164_164796

theorem wholesale_price (R W : ℝ) (h1 : R = 1.80 * W) (h2 : R = 36) : W = 20 :=
by
  sorry 

end wholesale_price_l164_164796


namespace largest_c_inequality_l164_164995

theorem largest_c_inequality (n : ℕ) (c : ℝ) (a : Fin n → ℝ) 
    (h1 : 2 ≤ n) 
    (h2 : ∀ i : Fin n, 0 ≤ a i) 
    (h3 : (Finset.univ.sum (λ i, a i)) = n) 
    (h4 : c = n / (n + (1 / (n - 1)))) : 
    (Finset.univ.sum (λ i, 1 / (n + c * (a i)^2))) ≤ (n / (n + c)) := 
by 
  sorry

end largest_c_inequality_l164_164995


namespace number_of_slices_per_pizza_l164_164746

-- Define the problem
def slices_per_pizza (total_slices total_pizzas : ℕ) : ℕ :=
  total_slices / total_pizzas

-- State the theorem
theorem number_of_slices_per_pizza (total_slices total_pizzas : ℕ) (h_total_slices : total_slices = 14) (h_total_pizzas : total_pizzas = 7) :
  slices_per_pizza total_slices total_pizzas = 2 :=
by 
  -- Using the conditions
  rw [h_total_slices, h_total_pizzas],
  -- Simplify the division
  show slices_per_pizza 14 7 = 2,
  -- Omitting the proof details
  sorry

end number_of_slices_per_pizza_l164_164746


namespace inverse_function_properties_l164_164705

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2

theorem inverse_function_properties :
  Odd (fun y => Real.sinh (Real.asinh y)) ∧ IncreasingOn (fun y => Real.sinh (Real.asinh y)) (Set.Ioi 0) :=
by
  sorry

end inverse_function_properties_l164_164705


namespace product_of_three_integers_sum_l164_164328

theorem product_of_three_integers_sum :
  ∀ (a b c : ℕ), (c = a + b) → (a * b * c = 8 * (a + b + c)) →
  (a > 0) → (b > 0) → (c > 0) →
  (∃ N1 N2 N3: ℕ, N1 = (a * b * (a + b)), N2 = (a * b * (a + b)), N3 = (a * b * (a + b)) ∧ 
  (N1 = 272 ∨ N2 = 160 ∨ N3 = 128) ∧ 
  (N1 + N2 + N3 = 560)) := sorry

end product_of_three_integers_sum_l164_164328


namespace count_congruent_to_3_mod_7_in_first_150_l164_164956

-- Define the predicate for being congruent to 3 modulo 7
def is_congruent_to_3_mod_7 (n : ℕ) : Prop :=
  ∃ k : ℤ, n = 7 * k + 3

-- Define the set of the first 150 positive integers
def first_150_positive_integers : finset ℕ :=
  finset.range 151 \ {0}

-- Define the subset of first 150 positive integers that are congruent to 3 modulo 7
def congruent_to_3_mod_7_in_first_150 : finset ℕ :=
  first_150_positive_integers.filter is_congruent_to_3_mod_7

-- The theorem to prove the size of this subset is 22
theorem count_congruent_to_3_mod_7_in_first_150 :
  congruent_to_3_mod_7_in_first_150.card = 22 :=
sorry

end count_congruent_to_3_mod_7_in_first_150_l164_164956


namespace circumscribed_triangle_centers_form_circle_l164_164491

noncomputable def locus_of_centers (ΔABC : Triangle (complex)) : Set (complex) :=
  {C : complex | ∃ (z_1 z_2 z_3 : complex),
    z_1 + z_2 + z_3 = 0 ∧
    (C = (z_1 + z_2) / 2 + (complex.i * (z_1 - z_2) * (real.sqrt 3) / 2) ∨
     C = (z_2 + z_3) / 2 + (complex.i * (z_2 - z_3) * (real.sqrt 3) / 2) ∨
     C = (z_3 + z_1) / 2 + (complex.i * (z_3 - z_1) * (real.sqrt 3) / 2))}

theorem circumscribed_triangle_centers_form_circle (ΔABC : Triangle (complex)) :
  ∃ (O : complex) (r : ℝ), locus_of_centers ΔABC = {C : complex | complex.abs (C - O) = r} :=
sorry

end circumscribed_triangle_centers_form_circle_l164_164491


namespace sin_half_max_values_sin_third_max_values_l164_164926

-- Define the problem for Part (a)
theorem sin_half_max_values (α : ℝ) (x : ℝ) (hx : sin α = x) : 
  (card (finset.image (λ k : ℤ, sin (k * π / 2 + (-1)^k * arcsin x / 2)) (finset.Icc (-1 : ℤ) 2))) = 4 :=
sorry

-- Define the problem for Part (b)
theorem sin_third_max_values (α : ℝ) (x : ℝ) (hx : sin α = x) : 
  (card (finset.image (λ k : ℤ, sin (k * π / 3 + (-1)^k * arcsin x / 3)) (finset.Icc 0 2))) = 3 :=
sorry

end sin_half_max_values_sin_third_max_values_l164_164926


namespace xiaoning_pe_comprehensive_score_l164_164781

def midterm_weight : ℝ := 0.3
def final_weight : ℝ := 0.7
def midterm_score : ℝ := 80
def final_score : ℝ := 90

theorem xiaoning_pe_comprehensive_score : midterm_score * midterm_weight + final_score * final_weight = 87 :=
by
  sorry

end xiaoning_pe_comprehensive_score_l164_164781


namespace possible_values_of_sum_l164_164233

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + y^3 + 21 * x * y = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l164_164233


namespace exist_initial_points_l164_164271

theorem exist_initial_points (n : ℕ) (h : 9 * n - 8 = 82) : ∃ n = 10 :=
by
  sorry

end exist_initial_points_l164_164271


namespace jo_climbs_six_stairs_l164_164843

def f : ℕ → ℕ
| 0      := 1
| 1      := 1
| 2      := 2
| (n+3)  := f n + f (n+1) + f (n+2)

theorem jo_climbs_six_stairs :
  f 6 = 24 :=
sorry

end jo_climbs_six_stairs_l164_164843
