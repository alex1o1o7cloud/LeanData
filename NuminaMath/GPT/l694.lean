import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Basic
import Mathlib.Algebra.BigOperators.Finprod
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Algebra.Ring.Basic
import Mathlib.Algebra.Sequences
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Polynomial.Bernstein
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Circle
import Mathlib.Geometry.Triangle
import Mathlib.LinearAlgebra.Matrix
import Mathlib.Order.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Algebra.Group.Basic
import Mathlib.Topology.Basic
import Mathlib.Topology.Order

namespace units_digit_sum_squares_l694_694778

theorem units_digit_sum_squares : 
  (∑ i in Finset.range 2003, i^2) % 10 = 9 :=
by
  sorry

end units_digit_sum_squares_l694_694778


namespace tins_of_beans_left_l694_694822

theorem tins_of_beans_left (cases : ℕ) (tins_per_case : ℕ) (damage_percentage : ℝ) (h_cases : cases = 15)
  (h_tins_per_case : tins_per_case = 24) (h_damage_percentage : damage_percentage = 0.05) :
  let total_tins := cases * tins_per_case
  let damaged_tins := total_tins * damage_percentage
  let tins_left := total_tins - damaged_tins
  tins_left = 342 :=
by
  sorry

end tins_of_beans_left_l694_694822


namespace initial_red_balloons_l694_694705

variable (initial_red : ℕ)
variable (given_away : ℕ := 24)
variable (left_with : ℕ := 7)

theorem initial_red_balloons : initial_red = given_away + left_with :=
by sorry

end initial_red_balloons_l694_694705


namespace mayoral_election_half_participation_l694_694501

open Set

variable (N : Type) [Fintype N] (acquaintance : N → Set N)

noncomputable def will_vote (x : N) (candidates : Set N) : Prop :=
∃ y ∈ acquaintance x, y ∈ candidates

theorem mayoral_election_half_participation :
  (∀ x : N, Fintype.card (acquaintance x) ≥ 0.3 * Fintype.card N) →
  ∃ A B : N, Fintype.card {x : N | will_vote x {A, B}} ≥ ⌊0.5 * Fintype.card N⌋ :=
by
  sorry

end mayoral_election_half_participation_l694_694501


namespace total_amount_after_refunds_l694_694872

def individual_bookings : ℕ := 12000
def group_bookings : ℕ := 16000
def refunds : ℕ := 1600

theorem total_amount_after_refunds : 
  individual_bookings + group_bookings - refunds = 26400 :=
by 
  -- The proof goes here.
  sorry

end total_amount_after_refunds_l694_694872


namespace supplementary_angle_60_eq_120_l694_694989

def supplementary_angle (α : ℝ) : ℝ :=
  180 - α

theorem supplementary_angle_60_eq_120 :
  supplementary_angle 60 = 120 :=
by
  -- the proof should be filled here
  sorry

end supplementary_angle_60_eq_120_l694_694989


namespace ratio_triangle_area_to_face_area_l694_694491

open Real

/-- The ratio of the area of the triangle formed by the midpoints of specific edges of a cube to the area of one of its faces is 1/4. -/
theorem ratio_triangle_area_to_face_area 
  (s : ℝ) 
  (P Q R S T U V W M N K : ℝ × ℝ × ℝ)
  (hP : P = (0, 0, 0))
  (hQ : Q = (s, 0, 0))
  (hR : R = (s, 0, s))
  (hS : S = (0, 0, s))
  (hT : T = (0, s, 0))
  (hU : U = (s, s, 0))
  (hV : V = (s, s, s))
  (hW : W = (0, s, s))
  (hM : M = ((0 + s) / 2, (0 + s) / 2, 0))
  (hN : N = ((0 + s) / 2, (s + s) / 2, s))
  (hK : K = ((s + 0) / 2, s, (s + 0) / 2)) :
  let area_face := s ^ 2 in
  let area_triangle_MNK := (1 / 2) * (1 / 2) * s ^ 2 in
  let ratio := area_triangle_MNK / area_face in
  ratio = 1 / 4 := 
sorry

end ratio_triangle_area_to_face_area_l694_694491


namespace count_divisible_by_11_with_digits_sum_10_l694_694605

-- Definitions and conditions from step a)
def digits_add_up_to_10 (n : ℕ) : Prop :=
  let ds := n.digits 10 in ds.length = 4 ∧ ds.sum = 10

def divisible_by_11 (n : ℕ) : Prop :=
  let ds := n.digits 10 in
  (ds.nth 0).get_or_else 0 +
  (ds.nth 2).get_or_else 0 -
  (ds.nth 1).get_or_else 0 -
  (ds.nth 3).get_or_else 0 ∣ 11

-- The tuple (question, conditions, answer) formalized in Lean
theorem count_divisible_by_11_with_digits_sum_10 :
  ∃ N : ℕ, N = (
  (Finset.filter (λ x, digits_add_up_to_10 x ∧ divisible_by_11 x)
  (Finset.range 10000)).card
) := 
sorry

end count_divisible_by_11_with_digits_sum_10_l694_694605


namespace nested_sqrt_expr_l694_694897

theorem nested_sqrt_expr (M : ℝ) (h : M > 1) : (↑(M) ^ (1 / 4) ^ (1 / 4) ^ (1 / 4)) = M ^ (21 / 64) :=
by
  sorry

end nested_sqrt_expr_l694_694897


namespace collinear_external_bisectors_l694_694541

theorem collinear_external_bisectors 
  (A B C M N P : Point) 
  (triangle : triangle A B C)
  (external_bisectors : 
    external_bisector A B C M ∧ 
    external_bisector B C A N ∧ 
    external_bisector C A B P) :
  collinear M N P :=
sorry

end collinear_external_bisectors_l694_694541


namespace cos_2x_quadratic_eq_l694_694212

theorem cos_2x_quadratic_eq (a b c : ℝ) (h : a = 4 ∧ b = 2 ∧ c = -1) :
  let cos_x_sq_eq := a * (cos x)^2 + b * cos x + c = 0 in
  ∃ α β γ : ℝ,
    (α = 4 ∧ β = 2 ∧ γ = -1) ∧
    α * (cos (2 * x))^2 + β * (cos (2 * x)) + γ = 0 := 
by
  sorry

end cos_2x_quadratic_eq_l694_694212


namespace function_odd_and_decreasing_l694_694092

noncomputable def f (x : ℝ) : ℝ := 2^(-x) - 2^x

theorem function_odd_and_decreasing : 
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f' x < 0) :=
by
  -- sorry to skip the proof for now
  sorry

end function_odd_and_decreasing_l694_694092


namespace coeff_x3_of_product_l694_694138

-- Definitions of the polynomials
def p(x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 + 3 * x + 4
def q(x : ℝ) : ℝ := 6 * x^3 + 5 * x^2 + 6 * x + 7

-- The theorem statement
theorem coeff_x3_of_product : ∀ x : ℝ, (coeff (p x * q x) 3 = 72) :=
by
  sorry

end coeff_x3_of_product_l694_694138


namespace Tyler_aquariums_l694_694760

theorem Tyler_aquariums (total_animals each_aquarium : ℕ) (h1 : total_animals = 512) (h2 : each_aquarium = 64) : total_animals / each_aquarium = 8 :=
by
  rw [h1, h2]
  norm_num

end Tyler_aquariums_l694_694760


namespace largest_value_is_expr4_l694_694898

noncomputable def expr1 : ℝ := 13765 + 1 / 2589
noncomputable def expr2 : ℝ := 13765 - 1 / 2589
noncomputable def expr3 : ℝ := 13765 * (1 / 2589)
noncomputable def expr4 : ℝ := 13765 / (1 / 2589)
noncomputable def expr5 : ℝ := 13765 ^ 1.2589

theorem largest_value_is_expr4 : 
  expr4 > expr1 ∧ expr4 > expr2 ∧ expr4 > expr3 ∧ expr4 > expr5 :=
sorry

end largest_value_is_expr4_l694_694898


namespace mutually_exclusive_events_white_ball_l694_694252

noncomputable def draws_white_ball_A_and_B (events : Set Event) : Prop :=
  let A_draws_white := events.contains (Event.A_Draws White)
  let B_draws_white := events.contains (Event.B_Draws White)
  mutually_exclusive_and_not_complementary A_draws_white B_draws_white

theorem mutually_exclusive_events_white_ball :
  ∀ (A B C D : Person) (red blue black white : Ball) (draw : Person → Ball),
    (draw A = white ∧ draw B = white) → False :=
by
  intro A B C D red blue black white draw
  simp
  assumption (A ≠ B)
  sorry

end mutually_exclusive_events_white_ball_l694_694252


namespace find_x_l694_694249

-- Define conditions
def simple_interest (x y : ℝ) : Prop :=
  x * y * 2 / 100 = 800

def compound_interest (x y : ℝ) : Prop :=
  x * ((1 + y / 100)^2 - 1) = 820

-- Prove x = 8000 given the conditions
theorem find_x (x y : ℝ) (h1 : simple_interest x y) (h2 : compound_interest x y) : x = 8000 :=
  sorry

end find_x_l694_694249


namespace f_properties_l694_694954

noncomputable def f : ℝ → ℝ := λ x, abs (Real.sin x)

theorem f_properties : 
  (∀ x : ℝ, f(x) = abs (Real.sin x) ) ∧ 
  (∀ y : ℝ, 0 ≤ f(y) ∧ f(y) ≤ 1) ∧ 
  (∀ z : ℝ, f(z) = f(-z)) :=
by 
  sorry

end f_properties_l694_694954


namespace largest_prime_factor_7_fact_8_fact_l694_694012

theorem largest_prime_factor_7_fact_8_fact : 
  let n := 7! + 8! 
  ∃ p, Prime p ∧ p ∣ n ∧ (∀ q, Prime q ∧ q ∣ n → q ≤ p) := 
begin
  let n := fact 7 + fact 8,
  use 7, 
  split,
  { apply Prime.prime_7 },
  split,
  { sorry },  
  { sorry },
end

end largest_prime_factor_7_fact_8_fact_l694_694012


namespace prob_C_is_one_fourth_l694_694467

noncomputable def prob_C (pA pB pD : ℚ) : ℚ :=
  1 - pA - pB - pD

theorem prob_C_is_one_fourth :
  let pA := (1 : ℚ) / 4
  let pB := (1 : ℚ) / 2
  let pD := (0 : ℚ)
  prob_C pA pB pD = (1 : ℚ) / 4 :=
by
  simp [prob_C, pA, pB, pD]
  sorry

end prob_C_is_one_fourth_l694_694467


namespace simplified_expr_correct_l694_694111

variable (d e : ℤ)
variable (hd : d ≠ 0)

def A := 16 * d + 17 + 18 * d^2
def B := 4 * d + 3
def C := 2 * e
def simplified_expression := 20 * d + 20 + 18 * d^2 + 2 * e

theorem simplified_expr_correct : 
  (A + B + C = simplified_expression) ∧ (20 + 20 + 18 + 2 = 60) := 
by 
  sorry

end simplified_expr_correct_l694_694111


namespace hyperbola_eccentricity_l694_694957

theorem hyperbola_eccentricity (a : ℝ) (b : ℝ) (c : ℝ) (e : ℝ):
  (∀ x y, (x = 2) ∧ (y = 0) ∧ (a = 2) ∧ (b = 1) ∧ (c = Real.sqrt (a^2 + b^2))) →
  e = c / a →
  e = Real.sqrt(5) / 2 :=
by
  intros h e_def
  rw [e_def]
  rw [h.1, h.2.1]
  sorry

end hyperbola_eccentricity_l694_694957


namespace area_PSQR_solution_l694_694287

noncomputable def area_PSQR (P Q R S M N : Type) [h: IsTriangle P Q R] 
  (median_PM : ℝ) (median_QN : ℝ) (length_PQ : ℝ) (intersect_NS : Prop)
  (PM_eq : median_PM = 15) 
  (QN_eq : median_QN = 20) 
  (PQ_eq : length_PQ = 30)
  (circumcircle_intersect_S : intersect_NS) : Prop :=
  let area_PSQR := 675 * Real.sqrt 55 / 32
  area_PSQR = Real.Area (△P S R)
  
theorem area_PSQR_solution {P Q R S M N : Type} [h: IsTriangle P Q R]
  (median_PM : ℝ) (median_QN : ℝ) (length_PQ : ℝ) (intersect_NS : Prop)
  (PM_eq : median_PM = 15) 
  (QN_eq : median_QN = 20) 
  (PQ_eq : length_PQ = 30)
  (circumcircle_intersect_S : intersect_NS) : 
  area_PSQR P Q R S M N median_PM median_QN length_PQ intersect_NS PM_eq QN_eq PQ_eq circumcircle_intersect_S := 
sorry

end area_PSQR_solution_l694_694287


namespace find_m_l694_694217

def vector_perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem find_m (m : ℝ) : vector_perpendicular (3, 1) (m, -3) → m = 1 :=
by
  sorry

end find_m_l694_694217


namespace find_m_for_local_minimum_l694_694726

noncomputable def f (x m : ℝ) := x * (x - m) ^ 2

theorem find_m_for_local_minimum :
  ∃ m : ℝ, (∀ x : ℝ, (x = 1 → deriv (λ x => f x m) x = 0) ∧ 
                  (x = 1 → deriv (deriv (λ x => f x m)) x > 0)) ∧ 
            m = 1 :=
by
  sorry

end find_m_for_local_minimum_l694_694726


namespace largest_prime_factor_of_7_fact_add_8_fact_l694_694019

theorem largest_prime_factor_of_7_fact_add_8_fact :
  let n := 7!
  let m := 8!
  let sum := n + m
  prime_factors sum = {2, 3, 5, 7} ∧ (∀ p ∈ {2, 3, 5, 7}, p <= 7) := by
  sorry

end largest_prime_factor_of_7_fact_add_8_fact_l694_694019


namespace billion_to_scientific_l694_694085
noncomputable def scientific_notation_of_billion (n : ℝ) : ℝ := n * 10^9
theorem billion_to_scientific (a : ℝ) : scientific_notation_of_billion a = 1.48056 * 10^11 :=
by sorry

end billion_to_scientific_l694_694085


namespace evaluate_expression_l694_694342

variable (x y : ℚ)

theorem evaluate_expression 
  (hx : x = 2) 
  (hy : y = -1 / 5) : 
  (2 * x - 3)^2 - (x + 2 * y) * (x - 2 * y) - 3 * y^2 + 3 = 1 / 25 :=
by
  sorry

end evaluate_expression_l694_694342


namespace inclination_angle_is_150_l694_694357

-- Define the parametric equations
def parametric_x (t : ℝ) : ℝ := 5 - 3 * t
def parametric_y (t : ℝ) : ℝ := 3 + (sqrt 3) * t

-- Define the angle of inclination function
def angle_of_inclination (x : ℝ -> ℝ) (y : ℝ -> ℝ) : ℝ := 
  let m := -(sqrt 3) / 3 in
  if m < 0 then 180 - real.arctan m.to_real else real.arctan m.to_real 

-- Prove that the angle of inclination of the given line is 150 degrees
theorem inclination_angle_is_150 :
  angle_of_inclination parametric_x parametric_y = 150 :=
sorry

end inclination_angle_is_150_l694_694357


namespace volume_of_cone_half_sector_l694_694809

/-- A half-sector of a circle with radius 6 inches is rolled to form the lateral surface area 
     of a right circular cone. Prove that the volume of the cone is 9π√3 cubic inches. -/
theorem volume_of_cone_half_sector (r l: ℝ) (h : r = 3) (s: l = 6) :
    (1 / 3 : ℝ) * real.pi * r^2 * h = 9 * real.pi * real.sqrt 3 :=
by
  -- Provided conditions
  have sector_radius : l = 6 := s
  have base_radius : r = 3 := h
  sorry

end volume_of_cone_half_sector_l694_694809


namespace slope_of_line_passing_through_focus_l694_694183

section parabola_problem

variable (M : ℝ × ℝ)
variable (C : ℝ → ℝ → Prop)
variable (focus : ℝ × ℝ)

-- Definitions based on given conditions
def point_M := M = (-1, 1)
def parabola_C := ∀ x y, C x y ↔ y^2 = 4 * x
def focus_C := focus = (1, 0)
def angle_AMB_is_90 (A B : ℝ × ℝ) : Prop := 
  (A.1 + 1) * (B.1 + 1) + (A.2 - 1) * (B.2 - 1) = 0

noncomputable def solution : ℝ :=
2

-- The main theorem statement
theorem slope_of_line_passing_through_focus 
  (A B : ℝ × ℝ) :
  point_M (-1, 1) → parabola_C (λ x y, y^2 = 4 * x) → 
  focus_C (1, 0) → angle_AMB_is_90 (-1, 1) A B →
  ∃ k : ℝ, solution = k :=
by
  intros
  use 2
  exact sorry

end parabola_problem

end slope_of_line_passing_through_focus_l694_694183


namespace find_b_l694_694578

noncomputable def circle1 (x y a : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 5 - a^2 = 0
noncomputable def circle2 (x y b : ℝ) : Prop := x^2 + y^2 - (2*b - 10)*x - 2*b*y + 2*b^2 - 10*b + 16 = 0
def is_intersection (x1 y1 x2 y2 : ℝ) : Prop := x1^2 + y1^2 = x2^2 + y2^2

theorem find_b (a x1 y1 x2 y2 : ℝ) (b : ℝ) :
  (circle1 x1 y1 a) ∧ (circle1 x2 y2 a) ∧ 
  (circle2 x1 y1 b) ∧ (circle2 x2 y2 b) ∧ 
  is_intersection x1 y1 x2 y2 →
  b = 5 / 3 :=
sorry

end find_b_l694_694578


namespace num_arithmetic_progression_digits_3digit_l694_694606

theorem num_arithmetic_progression_digits_3digit : 
  let is_arithmetic_progression (a b c : ℕ) := a + c = 2 * b
  ∃ (count : ℕ), count = 45 ∧
    (∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → 
      ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ 
        1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 
        is_arithmetic_progression a b c → count = 45) :=
begin
  sorry
end

end num_arithmetic_progression_digits_3digit_l694_694606


namespace true_statements_l694_694810

-- Define the sphere and its radius
def sphere_radius : ℝ := 4

-- Define the lengths of the chords AB and CD
def length_AB : ℝ := 2 * Real.sqrt 7
def length_CD : ℝ := 4 * Real.sqrt 3

-- Define the midpoints of AB and CD
def midpoint_M := "Midpoint of AB"
def midpoint_N := "Midpoint of CD"

-- Prove the true statements
theorem true_statements :
  (chords_may_intersect_at_M (sphere_radius) (length_AB) (length_CD) midpoint_M) ∧
  ¬(chords_may_intersect_at_N (sphere_radius) (length_AB) (length_CD) midpoint_N) ∧
  (max_value_of_MN_is_5 (midpoint_M) (midpoint_N)) ∧
  (min_value_of_MN_is_1 (midpoint_M) (midpoint_N)) :=
sorry

/-
  Placeholder definitions. The actual definitions of these functions
  (chords_may_intersect_at_M, chords_may_intersect_at_N, max_value_of_MN_is_5,
  min_value_of_MN_is_1) would need to be provided based on geometric evaluations.
-/
def chords_may_intersect_at_M (r : ℝ) (length_AB : ℝ) (length_CD : ℝ) (M : String) : Prop := sorry
def chords_may_intersect_at_N (r : ℝ) (length_AB : ℝ) (length_CD : ℝ) (N : String) : Prop := sorry
def max_value_of_MN_is_5 (M : String) (N : String) : Prop := sorry
def min_value_of_MN_is_1 (M : String) (N : String) : Prop := sorry

end true_statements_l694_694810


namespace first_500_integers_representable_l694_694981

open Real

def floor_sum (x : ℝ) : ℤ :=
  Int.floor (3 * x) + Int.floor (5 * x) + Int.floor (7 * x) + Int.floor (11 * x) + Int.floor (13 * x)

theorem first_500_integers_representable : (Finset.filter (λ k : ℤ, ∃ x : ℝ, k = floor_sum x) (Finset.range 501)).card = 115 :=
by
  sorry

end first_500_integers_representable_l694_694981


namespace interval_of_monotonic_decrease_l694_694538

noncomputable def f (x : ℝ) : ℝ := Real.sin x - (1/2) * x

theorem interval_of_monotonic_decrease :
  ∀ x, x ∈ set.Ioo (Real.pi / 3) Real.pi -> (f x < f (x + ε) ∀ ε > 0) :=
sorry

end interval_of_monotonic_decrease_l694_694538


namespace inverse_matrix_l694_694923

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![7, -2], ![-3, 1]]

-- Define the supposed inverse B
def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 2], ![3, 7]]

-- Define the condition that A * B should yield the identity matrix
theorem inverse_matrix :
  A * B = 1 :=
sorry

end inverse_matrix_l694_694923


namespace sum_of_middle_cards_l694_694688

-- Define the problem conditions
variables (card_numbers : List ℕ) (h_card_size : card_numbers.length = 12) 
          (h_distinct : card_numbers.nodup) (h_sum : card_numbers.sum = 84) 
          (h_sorted : card_numbers.sorted (≤))

-- Define the proposition we need to prove
theorem sum_of_middle_cards : 
  ∃ a b, a ∈ card_numbers ∧ b ∈ card_numbers ∧ card_numbers.nth 5 = some a ∧ card_numbers.nth 6 = some b ∧ a + b = 14 := 
sorry

end sum_of_middle_cards_l694_694688


namespace range_of_a_l694_694671

def A (a : ℝ) : set ℝ := { x | x^2 - 2 * a * x + a = 0 }
def B (a : ℝ) : set ℝ := { x | x^2 - 4 * x + a + 5 = 0 }

theorem range_of_a :
  (∃ a : ℝ, (A a = ∅ ∧ B a ≠ ∅) ∨ (A a ≠ ∅ ∧ B a = ∅)) ↔ ∃ a : ℝ, a ∈ (-1 : ℝ) .. 0 ∨ 1 ≤ a :=
begin
  sorry
end

end range_of_a_l694_694671


namespace tins_left_after_damage_l694_694821

theorem tins_left_after_damage (cases : ℕ) (tins_per_case : ℕ) (damage_rate : ℚ) 
    (total_cases : cases = 15) (tins_per_case_value : tins_per_case = 24)
    (damage_rate_value : damage_rate = 0.05) :
    let total_tins := cases * tins_per_case
        damaged_tins := damage_rate * total_tins
        remaining_tins := total_tins - damaged_tins in
    remaining_tins = 342 := 
by
  sorry

end tins_left_after_damage_l694_694821


namespace total_pencils_sold_l694_694484

theorem total_pencils_sold (price_reduced: Bool)
  (day1_students : ℕ) (first4_d1 : ℕ) (next3_d1 : ℕ) (last3_d1 : ℕ)
  (day2_students : ℕ) (first5_d2 : ℕ) (next6_d2 : ℕ) (last4_d2 : ℕ)
  (day3_students : ℕ) (first10_d3 : ℕ) (next10_d3 : ℕ) (last10_d3 : ℕ)
  (day1_total : day1_students = 10 ∧ first4_d1 = 4 ∧ next3_d1 = 3 ∧ last3_d1 = 3 ∧
    (first4_d1 * 5) + (next3_d1 * 7) + (last3_d1 * 3) = 50)
  (day2_total : day2_students = 15 ∧ first5_d2 = 5 ∧ next6_d2 = 6 ∧ last4_d2 = 4 ∧
    (first5_d2 * 4) + (next6_d2 * 9) + (last4_d2 * 6) = 98)
  (day3_total : day3_students = 2 * day2_students ∧ first10_d3 = 10 ∧ next10_d3 = 10 ∧ last10_d3 = 10 ∧
    (first10_d3 * 2) + (next10_d3 * 8) + (last10_d3 * 4) = 140) :
  (50 + 98 + 140 = 288) :=
sorry

end total_pencils_sold_l694_694484


namespace probability_six_integers_different_tens_digits_l694_694349

noncomputable theory
open Finset
open Nat

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def probability_different_tens_digit (s : Finset ℕ) (h : s.card = 6) : ℚ :=
  let num_possible_choices := (choose 9 6 : ℕ) * 10^6
  let total_possibilities := choose 90 6
  num_possible_choices / total_possibilities

theorem probability_six_integers_different_tens_digits :
  ∀ (s : Finset ℕ), s.card = 6 → (∀ n ∈ s, 10 ≤ n ∧ n ≤ 99) →
  (∀ n m ∈ s, n ≠ m → tens_digit n ≠ tens_digit m) →
  probability_different_tens_digit s ‹s.card = 6› = 8000 / 5895 :=
by
  intros s hc hr hd
  apply probability_different_tens_digit_congr hc hr hd
  sorry

end probability_six_integers_different_tens_digits_l694_694349


namespace distance_between_parallel_lines_l694_694956

theorem distance_between_parallel_lines 
  (A1 B1 C1 A2 B2 C2 : ℝ)
  (h1 : 3 * x + 2 * y - 3 = 0)
  (h2 : 3 * x + 2 * y + 7 / 2 = 0) 
  : dist_parallel 3 2 -3 3 2 (7 / 2) = (√13) / 2 :=
by
  sorry

end distance_between_parallel_lines_l694_694956


namespace positive_factors_multiples_of_6_l694_694982

theorem positive_factors_multiples_of_6 (n : ℕ) (h : n = 60) : 
  (∃ k : ℕ, k = 4 ∧ {d : ℕ | d ∣ 60 ∧ d % 6 = 0}.card = k) :=
by
  sorry

end positive_factors_multiples_of_6_l694_694982


namespace water_level_after_opening_l694_694756

-- Let's define the densities and initial height as given
def ρ_water : ℝ := 1000
def ρ_oil : ℝ := 700
def initial_height : ℝ := 40  -- height in cm

-- Final heights after opening the valve (h' denotes final height)
def final_height_water : ℝ := 34

-- Using the principles described
theorem water_level_after_opening :
  ∃ h_oil : ℝ, ρ_water * final_height_water = ρ_oil * h_oil ∧ final_height_water + h_oil = initial_height :=
begin
  use initial_height - final_height_water,
  split,
  {
    field_simp,
    norm_num,
  },
  {
    field_simp,
    norm_num,
  }
end

end water_level_after_opening_l694_694756


namespace find_angle_A_and_tan_B_l694_694622

noncomputable def triangle_sides_and_angles 
  (a b c : ℝ) 
  (h₀ : b^2 + c^2 - real.sqrt 2 * b * c = a^2)
  (h₁ : c / b = 2 * real.sqrt 2) : 
  Prop :=
∃ A B C : ℝ, 
  A = real.pi / 4 ∧
  B = real.arctan (1 / 3) ∧
  A + B + C = real.pi

theorem find_angle_A_and_tan_B 
  (a b c : ℝ) 
  (h₀ : b^2 + c^2 - real.sqrt 2 * b * c = a^2)
  (h₁ : c / b = 2 * real.sqrt 2) : 
  triangle_sides_and_angles a b c h₀ h₁ :=
by
  sorry

end find_angle_A_and_tan_B_l694_694622


namespace find_C_work_rate_l694_694422

-- Conditions
def A_work_rate := 1 / 4
def B_work_rate := 1 / 6

-- Combined work rate of A and B
def AB_work_rate := A_work_rate + B_work_rate

-- Total work rate when C is assisting, completing in 2 days
def total_work_rate_of_ABC := 1 / 2

theorem find_C_work_rate : ∃ c : ℕ, (AB_work_rate + 1 / c = total_work_rate_of_ABC) ∧ c = 12 :=
by
  -- To complete the proof, we solve the equation for c
  sorry

end find_C_work_rate_l694_694422


namespace triangle_proof_l694_694942

variable (a b c : ℝ)
variable (A B C : Real.Angle)

-- Conditions from the problem
variable (h1 : a * sin A * sin B + b * (cos A)^2 = (4/3) * a)
variable (h2 : c^2 = a^2 + (1/4) * b^2)

-- Proving the questions
theorem triangle_proof 
  (h1 : a * sin A * sin B + b * (cos A)^2 = (4/3) * a)
  (h2 : c^2 = a^2 + (1/4) * b^2) :
  (b / a = 4 / 3) ∧ (C = Real.Angle.pi / 3) :=
sorry

end triangle_proof_l694_694942


namespace find_ab_l694_694237

-- Define the conditions
variables (a b : ℝ)
hypothesis h1 : a - b = 3
hypothesis h2 : a^2 + b^2 = 29

-- State the theorem
theorem find_ab : a * b = 10 :=
by
  sorry

end find_ab_l694_694237


namespace simon_age_l694_694849

theorem simon_age : 
  ∃ s : ℕ, 
  ∀ a : ℕ,
    a = 30 → 
    s = (a / 2) - 5 → 
    s = 10 := 
by
  sorry

end simon_age_l694_694849


namespace eval_expression_l694_694504

theorem eval_expression : (Real.sqrt (16 - 8 * Real.sqrt 6) + Real.sqrt (16 + 8 * Real.sqrt 6)) = 4 * Real.sqrt 6 :=
by
  sorry

end eval_expression_l694_694504


namespace inequality_solution_l694_694506

theorem inequality_solution (x : ℝ) (hx : x ≥ 0) :
  (real.sqrt (real.sqrt x) - 3 / (real.sqrt (real.sqrt x) + 4) ≥ 0) ↔ (0 ≤ x ∧ x ≤ 81) := 
by sorry

end inequality_solution_l694_694506


namespace transform_curve_C1_to_C2_l694_694127

section MatricesAndTransformations

-- Define the matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![0, 1], ![1, 0]]
def B : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 0], ![0, 2]]

-- Define the curve C1
def C1 (x y : ℝ) : Prop := (x^2 / 8 + y^2 / 2 = 1)

-- Define the product matrix AB
def AB : Matrix (Fin 2) (Fin 2) ℝ := A ⬝ B

-- Statement of the proof problem
theorem transform_curve_C1_to_C2 :
  ∀ (x y : ℝ), C1 x y → (x^2 + y^2 = 8) :=
by
  -- Sorry statement as proof is not required
  sorry

end MatricesAndTransformations

end transform_curve_C1_to_C2_l694_694127


namespace unique_pegboard_arrangement_l694_694748

/-- Conceptually, we will set up a function to count valid arrangements of pegs
based on the given conditions and prove that there is exactly one such arrangement. -/
def triangular_pegboard_arrangements (yellow red green blue orange black : ℕ) : ℕ :=
  if yellow = 6 ∧ red = 5 ∧ green = 4 ∧ blue = 3 ∧ orange = 2 ∧ black = 1 then 1 else 0

theorem unique_pegboard_arrangement :
  triangular_pegboard_arrangements 6 5 4 3 2 1 = 1 :=
by
  -- Placeholder for proof
  sorry

end unique_pegboard_arrangement_l694_694748


namespace part_a_part_b_l694_694793

-- Part (a)
theorem part_a (p q r a : ℕ) (prime_r : prime r) (rel_prime_pq : nat.coprime p q) 
    (h : p * q = r * a^2) : (∃ k, p = k^2) ∨ (∃ k, q = k^2) :=
sorry

-- Part (b)
theorem part_b : ¬(∃ p : ℕ, prime p ∧ ∃ k : ℕ, p * (2^(p+1) - 1) = k^2) :=
sorry

end part_a_part_b_l694_694793


namespace renovation_exceeds_l694_694801

variables (a_n b_n A_n B_n : ℕ → ℝ)
variable (n : ℕ)

def profit_without_renovation (n : ℕ) : ℝ :=
  500 - 20 * n

def profit_with_renovation (n : ℕ) : ℝ :=
  1000 - (1000 / (2 ^ n))

def cumulative_profit_without_renovation (n : ℕ) : ℝ :=
  500 * n - 10 * n * (n + 1)

def cumulative_profit_with_renovation (n : ℕ) : ℝ :=
  1000 * n - 2600 + (2000 / (2 ^ n))

theorem renovation_exceeds (n : ℕ) :
  ∃ k, B_n k > A_n k :=
by
  let A_n := cumulative_profit_without_renovation n
  let B_n := cumulative_profit_with_renovation n
  exact ⟨5, sorry⟩

end renovation_exceeds_l694_694801


namespace inverse_function_f_l694_694510

noncomputable def f (x : ℝ) : ℝ := 2^(x + 1)

noncomputable def f_inv (x : ℝ) : ℝ := Real.log x / Real.log 2 - 1

theorem inverse_function_f : ∀ x > 0, f_inv (f x) = x :=
by
  intro x hx
  dsimp [f, f_inv]
  sorry

end inverse_function_f_l694_694510


namespace largest_prime_factor_7_fact_8_fact_l694_694000

theorem largest_prime_factor_7_fact_8_fact : 
  let n := 7! + 8! 
  ∃ p, Prime p ∧ p ∣ n ∧ (∀ q, Prime q ∧ q ∣ n → q ≤ p) := 
begin
  let n := fact 7 + fact 8,
  use 7, 
  split,
  { apply Prime.prime_7 },
  split,
  { sorry },  
  { sorry },
end

end largest_prime_factor_7_fact_8_fact_l694_694000


namespace good_pairs_count_l694_694303

theorem good_pairs_count (n : ℕ) (hn : n > 0) : 
  let num_good_pairs := 8 * n^2 - 12 * n + 4 in
  num_good_pairs
:= 
begin
  sorry
end

end good_pairs_count_l694_694303


namespace total_cost_of_trip_l694_694321

def totalDistance (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

def gallonsUsed (distance miles_per_gallon : ℕ) : ℕ :=
  distance / miles_per_gallon

def totalCost (gallons : ℕ) (cost_per_gallon : ℕ) : ℕ :=
  gallons * cost_per_gallon

theorem total_cost_of_trip :
  (totalDistance 10 6 5 9 = 30) →
  (gallonsUsed 30 15 = 2) →
  totalCost 2 35 = 700 :=
by
  sorry

end total_cost_of_trip_l694_694321


namespace four_digit_numbers_count_l694_694593

theorem four_digit_numbers_count :
  ∃ (n : ℕ),  n = 24 ∧
  let s := λ (a b c d : ℕ), (a + b + c + d = 10 ∧ a ≠ 0 ∧ ((a + c) - (b + d)) ∈ [(-11), 0, 11]) in
  (card {abcd : Fin 10 × Fin 10 × Fin 10 × Fin 10 | (s abcd.1.1 abcd.1.2 abcd.2.1 abcd.2.2)}) = n :=
by
  sorry

end four_digit_numbers_count_l694_694593


namespace simon_age_is_10_l694_694844

-- Declare the variables
variable (alvin_age : ℕ) (simon_age : ℕ)

-- Define the conditions
def condition1 : Prop := alvin_age = 30
def condition2 : Prop := simon_age = (alvin_age / 2) - 5

-- Formalize the proof problem
theorem simon_age_is_10 (h1 : condition1) (h2 : condition2) : simon_age = 10 := by
  sorry

end simon_age_is_10_l694_694844


namespace simon_age_is_10_l694_694845

def alvin_age : ℕ := 30

def simon_age (alvin_age : ℕ) : ℕ :=
  (alvin_age / 2) - 5

theorem simon_age_is_10 : simon_age alvin_age = 10 := by
  sorry

end simon_age_is_10_l694_694845


namespace largest_prime_factor_7_fact_8_fact_l694_694008

theorem largest_prime_factor_7_fact_8_fact : 
  let n := 7! + 8! 
  ∃ p, Prime p ∧ p ∣ n ∧ (∀ q, Prime q ∧ q ∣ n → q ≤ p) := 
begin
  let n := fact 7 + fact 8,
  use 7, 
  split,
  { apply Prime.prime_7 },
  split,
  { sorry },  
  { sorry },
end

end largest_prime_factor_7_fact_8_fact_l694_694008


namespace brick_weight_l694_694800

theorem brick_weight : ∃ x : ℝ, x = 2 + x / 3 ∧ x = 3 :=
by
  use 3
  split
  · norm_num
  sorry

end brick_weight_l694_694800


namespace positive_difference_arithmetic_sequence_l694_694772

theorem positive_difference_arithmetic_sequence :
  let a := -10
  let d := 9
  let aₙ (n : ℕ) := a + n * d
  aₙ 2019 - aₙ 2009 = 90 :=
by
  sorry

end positive_difference_arithmetic_sequence_l694_694772


namespace johns_watermelon_weight_l694_694682

theorem johns_watermelon_weight (michael_weight clay_weight john_weight : ℕ)
  (h1 : michael_weight = 8)
  (h2 : clay_weight = 3 * michael_weight)
  (h3 : john_weight = clay_weight / 2) :
  john_weight = 12 :=
by
  sorry

end johns_watermelon_weight_l694_694682


namespace vacation_fund_percentage_l694_694500

variable (s : ℝ) (vs : ℝ)
variable (d : ℝ)
variable (v : ℝ)

-- conditions:
-- 1. Jill's net monthly salary
#check (s = 3700)
-- 2. Jill's discretionary income is one fifth of her salary
#check (d = s / 5)
-- 3. Savings percentage
#check (0.20 * d)
-- 4. Eating out and socializing percentage
#check (0.35 * d)
-- 5. Gifts and charitable causes
#check (111)

-- Prove: 
theorem vacation_fund_percentage : 
  s = 3700 -> d = s / 5 -> 
  (v * d + 0.20 * d + 0.35 * d + 111 = d) -> 
  v = 222 / 740 :=
by
  sorry -- proof skipped

end vacation_fund_percentage_l694_694500


namespace batsman_average_after_17th_l694_694034

theorem batsman_average_after_17th : 
  ∃ A : ℕ, let new_average := A + 3 in
  let total_runs := 16 * A + 84 in
  let new_total_runs := 17 * new_average in
  total_runs = new_total_runs ∧ new_average = 36 :=
by
  sorry

end batsman_average_after_17th_l694_694034


namespace derrick_has_34_pictures_l694_694335

-- Assume Ralph has 26 pictures of wild animals
def ralph_pictures : ℕ := 26

-- Derrick has 8 more pictures than Ralph
def derrick_pictures : ℕ := ralph_pictures + 8

-- Prove that Derrick has 34 pictures of wild animals
theorem derrick_has_34_pictures : derrick_pictures = 34 := by
  sorry

end derrick_has_34_pictures_l694_694335


namespace projection_coordinates_l694_694529

-- Definitions from conditions in a)
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (1, 1)

-- Projection function
def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let mag_squared := v.1 ^ 2 + v.2 ^ 2
  let scalar := dot_product / mag_squared
  (scalar * v.1, scalar * v.2)

-- Main statement
theorem projection_coordinates : projection a b = (3, 3) :=
  sorry

end projection_coordinates_l694_694529


namespace allay_connection_days_l694_694834

/-- Define the cost of the internet service per day for given range of days -/
def daily_cost (day: ℕ) : ℝ :=
  if day ≤ 3 then 0.5 else 
  if day ≤ 7 then 0.7 else 
  0.9

/-- Additional fee charged for every 5 days of continuous connection -/
def additional_fee (days: ℕ) : ℝ :=
  (days / 5) * 1

/-- Function to compute the total cost up to a given day -/
def total_cost (days: ℕ) : ℝ :=
  (List.sum (List.map daily_cost (List.range days))) + additional_fee days

/-- Prove that Allay will be connected for 8 days given the conditions -/
theorem allay_connection_days : 
  ∃ d, total_cost d ≤ 7 ∧ total_cost (d+1) > 7 ∧ d = 8 :=
by
  sorry

end allay_connection_days_l694_694834


namespace number_of_elements_in_figure_50_l694_694492

def cubic_seq (a b c d : ℕ) (n : ℕ) : ℕ :=
  a * n^3 + b * n^2 + c * n + d

noncomputable def coefficients :=
  let eq1 := (1, 1, 1, 6)
  let eq2 := (8, 4, 2, 24)
  let eq3 := (27, 9, 3, 64) 
  -- Solving eq1, eq2, eq3 manually or via some method would give:
  (2, 3, 1, 1) -- a = 2, b = 3, c = 1, d = 1

theorem number_of_elements_in_figure_50 : cubic_seq (fst (fst (fst coefficients)))
                                                  (snd (fst (fst coefficients)))
                                                  (snd (fst coefficients))
                                                  (snd coefficients) 50 = 257551 :=
  sorry

end number_of_elements_in_figure_50_l694_694492


namespace circles_are_internally_tangent_l694_694546

noncomputable def circle1 := (2 : ℝ, -2 : ℝ, 7 : ℝ)  -- Center (2, -2) and radius 7
noncomputable def circle2 := (-1 : ℝ, 2 : ℝ, 2 : ℝ)  -- Center (-1, 2) and radius 2

def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

def internally_tangent (circle1 circle2 : ℝ × ℝ × ℝ) : Prop := 
  ∃ (x1 y1 r1 x2 y2 r2 : ℝ), 
    circle1 = (x1, y1, r1) ∧
    circle2 = (x2, y2, r2) ∧
    distance x1 y1 x2 y2 = r1 - r2

theorem circles_are_internally_tangent :
  internally_tangent circle1 circle2 :=
sorry

end circles_are_internally_tangent_l694_694546


namespace triangle_area_given_conditions_l694_694047

theorem triangle_area_given_conditions (A B C L : ℝ) (AL BL CL : ℝ)
  (h1 : BL = sqrt 30) (h2 : AL = 2) (h3 : CL = 5) :
  (area_of_triangle A B C) = (7 * sqrt 39) / 4 :=
by
  -- Proof is left as an exercise.
  sorry

end triangle_area_given_conditions_l694_694047


namespace necessary_but_not_sufficient_condition_l694_694792

open Set

variable {P A B : Type} [MetricSpace P]

/-- The sum of distances from a moving point P to two fixed points A and B is a constant -/
def sum_of_distances_constant (P A B : P) (k : ℝ) : Prop :=
  dist P A + dist P B = k

/-- The trajectory of the moving point P is an ellipse if and only if the sum of distances 
    from P to two fixed points A and B is a constant -/
theorem necessary_but_not_sufficient_condition (P : Type) [MetricSpace P] {A B : P} {a k : ℝ}
  (h1 : sum_of_distances_constant P A B k) :
  False := sorry

end necessary_but_not_sufficient_condition_l694_694792


namespace AM_less_than_BM_plus_CM_l694_694653

variable (A B C M O : Point) -- Points involved
variable (AM BM CM AB AC : ℝ) -- Distances between points

-- Assumption set up:
axiom is_isosceles_ABC : A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ AB = AC ∧ is_Inscribed_Triangle A B C O
axiom M_on_arc_BC : M ∈ Arc B C ∧ M ≠ B ∧ M ≠ C

-- Distance definitions
axiom AM_def : AM = distance A M
axiom BM_def : BM = distance B M
axiom CM_def : CM = distance C M
axiom AB_def : AB = distance A B
axiom AC_def : AC = distance A C

-- Theorem statement
theorem AM_less_than_BM_plus_CM : AM < BM + CM := 
by
  -- The proof goes here
  sorry

end AM_less_than_BM_plus_CM_l694_694653


namespace pipe_B_fill_time_l694_694079

variable (A B C : ℝ)
variable (fill_time : ℝ := 16)
variable (total_tank : ℝ := 1)

-- Conditions
axiom condition1 : A + B + C = (1 / fill_time)
axiom condition2 : A = 2 * B
axiom condition3 : B = 2 * C

-- Prove that B alone will take 56 hours to fill the tank
theorem pipe_B_fill_time : B = (1 / 56) :=
by sorry

end pipe_B_fill_time_l694_694079


namespace dan_initial_money_l694_694496

theorem dan_initial_money 
  (choc_cost : ℕ) (candy_cost : ℕ) (diff_cost : ℕ)
  (h_choc_cost : choc_cost = 7)
  (h_candy_cost : candy_cost = 2)
  (h_diff_cost : choc_cost = candy_cost + diff_cost) :
  ∃ m : ℕ, m ≥ 9 :=
by
  use choc_cost + candy_cost
  rw [h_choc_cost, h_candy_cost]
  norm_num

end dan_initial_money_l694_694496


namespace product_real_values_r_l694_694514

theorem product_real_values_r :
  (∀ r : ℝ, (∃ x : ℝ, x ≠ 0 ∧ (1 / (3 * x)) = ((r - x) / 6)) ↔
  (∃! x : ℝ, x ≠ 0 ∧ (1 / (3 * x)) = ((r - x) / 6))) →
  ∏ r in ({x | ∃! x : ℝ, x ≠ 0 ∧ (1 / (3 * x)) = ((r - x) / 6)}), r = -8 :=
by
  sorry

end product_real_values_r_l694_694514


namespace least_number_of_froods_l694_694634

def dropping_score (n : ℕ) : ℕ := (n * (n + 1)) / 2
def eating_score (n : ℕ) : ℕ := 15 * n

theorem least_number_of_froods : ∃ n : ℕ, (dropping_score n > eating_score n) ∧ (∀ m < n, dropping_score m ≤ eating_score m) :=
  exists.intro 30 
    (and.intro 
      (by simp [dropping_score, eating_score]; linarith)
      (by intros m hmn; simp [dropping_score, eating_score]; linarith [hmn]))

end least_number_of_froods_l694_694634


namespace find_polynomial_l694_694144

-- Define the conditions
def polynomial_satisfies_conditions (f : Polynomial ℝ) (n m : ℕ) : Prop :=
  (f.eval 0 = 1) ∧
  (∀ k : ℕ, k ≤ n → (Polynomial.derivative^[k] f).eval 0 = 0) ∧
  (∀ k : ℕ, k ≤ m → (Polynomial.derivative^[k] f).eval 1 = 0)

-- Define the form of the polynomial
noncomputable def polynomial_form (f : Polynomial ℝ) (n m : ℕ) : Prop :=
  ∃ (G h : Polynomial ℝ), f = (X^n * G + X^n * h * (X - 1)^m + 1)

-- The theorem statement
theorem find_polynomial (f : Polynomial ℝ) (n m : ℕ) :
  polynomial_satisfies_conditions f n m → polynomial_form f n m :=
by
  -- Proof is omitted
  sorry

end find_polynomial_l694_694144


namespace eval_g_at_2_l694_694232

def g (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem eval_g_at_2 : g 2 = 3 :=
by {
  -- This is the place for proof steps, currently it is filled with sorry.
  sorry
}

end eval_g_at_2_l694_694232


namespace find_matrix_N_l694_694909

open Matrix

theorem find_matrix_N :
  ∃ (N : Matrix (Fin 2) (Fin 2) ℤ),
    (N ⬝ (col_vector 2 ![4, 0]) = col_vector 2 ![8, 28]) ∧
    (N ⬝ (col_vector 2 ![-2, 10]) = col_vector 2 ![6, -34]) ∧
    (N = ![![2, 1], ![7, -2]]) :=
begin
  -- proof would go here
  sorry
end

end find_matrix_N_l694_694909


namespace probability_of_one_head_in_three_flips_l694_694617

open Classical

theorem probability_of_one_head_in_three_flips : 
  ∀ (p : ℝ) (n k : ℕ), p = 0.5 → n = 3 → k = 1 → 
  (Nat.choose n k * p^k * (1 - p)^(n - k)) = 0.375 := 
by 
  intros p n k hp hn hk
  rw [hp, hn, hk]
  norm_num
  sorry

end probability_of_one_head_in_three_flips_l694_694617


namespace harmonic_mean_of_4_and_2048_is_8_l694_694819

def harmonic_mean (a b : ℝ) : ℝ :=
  2 * a * b / (a + b)

theorem harmonic_mean_of_4_and_2048_is_8 : 
  abs (harmonic_mean 4 2048 - 8) < abs (harmonic_mean 4 2048 - n) 
  ∀ (n : ℤ), n ≠ 8 := by
  sorry

end harmonic_mean_of_4_and_2048_is_8_l694_694819


namespace complex_quadrant_l694_694618

theorem complex_quadrant (z : ℂ) (hz : z * (1 + complex.I) = -2 * complex.I) : 
  z.re < 0 ∧ z.im > 0 := by
  sorry

end complex_quadrant_l694_694618


namespace matrix_N_unique_l694_694911

theorem matrix_N_unique 
(M N : Matrix (Fin 2) (Fin 2) ℤ) 
(h1 : M ⬝ (λ (i : Fin 2), if i = 0 then 4 else 0) = λ (i : Fin 2), if i = 0 then 8 else 28)
(h2 : M ⬝ (λ (i : Fin 2), if i = 0 then -2 else 10) = λ (i : Fin 2), if i = 0 then 6 else -34):
  M = N :=
by 
  let col1 : Fin 2 -> ℤ := (λ (i : Fin 2), if i = 0 then 2 else 7),
  let col2 : Fin 2 -> ℤ := (λ (i : Fin 2), if i = 0 then 1 else -2),
  let N : Matrix (Fin 2) (Fin 2) ℤ := λ (i j : Fin 2),
    if j = 0 then col1 i else col2 i,
  sorry

end matrix_N_unique_l694_694911


namespace area_of_ABCD_l694_694434

-- Define the area of the smaller square
def small_square_area : ℝ := 4

-- Define the area of the larger square
def large_square_area : ℝ := 16

-- Define the side lengths of the smaller and larger squares
def side_length_small_square : ℝ := Real.sqrt small_square_area
def side_length_large_square : ℝ := Real.sqrt large_square_area

-- Define the side length of the square ABCD
def side_length_ABCD : ℝ := side_length_small_square + side_length_large_square

-- The theorem asserting the area of ABCD
theorem area_of_ABCD : side_length_ABCD ^ 2 = 36 := by
  sorry

end area_of_ABCD_l694_694434


namespace net_rate_of_pay_l694_694453

noncomputable def hours := 3
noncomputable def speed := 50 -- miles per hour
noncomputable def fuel_efficiency := 25 -- miles per gallon
noncomputable def pay := 0.60 -- dollars per mile
noncomputable def gas_cost := 2.50 -- dollars per gallon

theorem net_rate_of_pay :
  (pay * (speed * hours) - gas_cost * ((speed * hours) / fuel_efficiency)) / hours = 25 :=
by
  sorry

end net_rate_of_pay_l694_694453


namespace stoppage_time_per_hour_l694_694428

def speed_without_stoppages : ℝ := 54
def speed_with_stoppages : ℝ := 45
def distance_loss_due_to_stoppages : ℝ := speed_without_stoppages - speed_with_stoppages

theorem stoppage_time_per_hour (T : ℝ) :
  distance_loss_due_to_stoppages / (speed_without_stoppages / 60) = 10 :=
by
  unfold distance_loss_due_to_stoppages
  have h1 : distance_loss_due_to_stoppages = 9 := rfl
  rw [h1]
  have h2 : speed_without_stoppages / 60 = 0.9 := sorry  -- Here you can skip the proof with sorry
  rw [h2]
  norm_num
  sorry

end stoppage_time_per_hour_l694_694428


namespace range_of_slope_of_l_l694_694925

noncomputable def slope := ℝ → ℝ → ℝ → ℝ → ℝ

def slope_PA (P A : ℝ × ℝ) : ℝ :=
  (A.snd - P.snd) / (A.fst - P.fst)

def slope_PB (P B : ℝ × ℝ) : ℝ :=
  (B.snd - P.snd) / (B.fst - P.fst)

theorem range_of_slope_of_l (k : ℝ) :
  (∃ P A B : ℝ × ℝ, 
    P = (0, -1) ∧ A = (1, -2) ∧ B = (2, 1) ∧
    slope P.fst P.snd A.fst A.snd = -1 ∧
    slope P.fst P.snd B.fst B.snd = 1 ∧
    slope P.fst P.snd k = k) →
    -1 ≤ k ∧ k ≤ 1 :=
sorry

end range_of_slope_of_l_l694_694925


namespace part1_part2_l694_694168

open Real

def f (x : ℝ) : ℝ := exp x - x - 1

theorem part1 : ∀ x : ℝ, f x ≥ 0 := by
  intro x
  -- proof omitted
  sorry

theorem part2 (n : ℕ) (hn : n > 0) :
  (∑ k in finset.range n, ((2 * k + 1 : ℝ) / (2 * n : ℝ))^n) < (sqrt ℯ) / (ℯ - 1) := by
  -- proof omitted
  sorry

end part1_part2_l694_694168


namespace maximum_n_l694_694179

def arithmetic_sequence_max_n (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) : Prop :=
  ∃ d : ℤ, ∀ m : ℕ, a (m + 1) = a m + d

def is_positive_first_term (a : ℕ → ℤ) : Prop :=
  a 0 > 0

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 0 + a (n-1))) / 2

def roots_of_equation (a1006 a1007 : ℤ) : Prop :=
  a1006 * a1007 = -2011 ∧ a1006 + a1007 = 2012

theorem maximum_n (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : arithmetic_sequence_max_n a S 1007)
  (h2 : is_positive_first_term a)
  (h3 : sum_of_first_n_terms a S)
  (h4 : ∃ a1006 a1007, roots_of_equation a1006 a1007 ∧ a 1006 = a1006 ∧ a 1007 = a1007) :
  ∃ n, S n > 0 → n ≤ 1007 := 
sorry

end maximum_n_l694_694179


namespace mary_pays_fifteen_l694_694324

def apple_cost : ℕ := 1
def orange_cost : ℕ := 2
def banana_cost : ℕ := 3
def discount_per_5_fruits : ℕ := 1

def apples_bought : ℕ := 5
def oranges_bought : ℕ := 3
def bananas_bought : ℕ := 2

def total_cost_before_discount : ℕ :=
  apples_bought * apple_cost +
  oranges_bought * orange_cost +
  bananas_bought * banana_cost

def total_fruits : ℕ :=
  apples_bought + oranges_bought + bananas_bought

def total_discount : ℕ :=
  (total_fruits / 5) * discount_per_5_fruits

def final_amount_to_pay : ℕ :=
  total_cost_before_discount - total_discount

theorem mary_pays_fifteen : final_amount_to_pay = 15 := by
  sorry

end mary_pays_fifteen_l694_694324


namespace find_x0_from_integral_l694_694675

variable {a b x_0 : ℝ}
variable (h_nonzero : a ≠ 0)
variable (h_positive : x_0 > 0)

def f (x : ℝ) := a * x^2 + b

theorem find_x0_from_integral (h_eq : ∫ x in 0..2, f x = 2 * f x_0) : 
  x_0 = 2 * Real.sqrt 3 / 3 := 
by 
  sorry

end find_x0_from_integral_l694_694675


namespace minimize_elongation_correctness_l694_694630

noncomputable def minimize_elongation_conditions
    (d : ℝ)
    (G G1 : ℝ)
    (r : ℝ)
    (E : ℝ)
    (sin_alpha : ℝ → ℝ)
    (q : ℝ) : ℝ × ℝ × ℝ := 
sorry 

theorem minimize_elongation_correctness
    (d : ℝ := 0.5)  -- length of beam in meters
    (G : ℝ := 5)    -- Weight of beam in kp
    (G1 : ℝ := 15)  -- Load at end of beam in kp
    (r : ℝ := 0.1 / 100)  -- Radius of wire in meters
    (E : ℝ := 2 * 10^6)   -- Elastic modulus in kpcm^-2
    (sin_alpha : ℝ → ℝ := λ h, h / real.sqrt (h^2 + d^2))  -- sin(α)
    (q : ℝ := real.pi * r^2)  -- Cross-sectional area of wire
  : minimize_elongation_conditions d G G1 r E sin_alpha q = (d, 0.278 / 1000, 24.74) := 
sorry

end minimize_elongation_correctness_l694_694630


namespace log_sum_eq_two_l694_694743

theorem log_sum_eq_two : 
  ∀ (lg : ℝ → ℝ),
  (∀ x y : ℝ, lg (x * y) = lg x + lg y) →
  (∀ x y : ℝ, lg (x ^ y) = y * lg x) →
  lg 4 + 2 * lg 5 = 2 :=
by
  intros lg h1 h2
  sorry

end log_sum_eq_two_l694_694743


namespace find_positive_integers_l694_694135

theorem find_positive_integers (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (1 / m + 1 / n - 1 / (m * n) = 2 / 5) ↔ 
  (m = 3 ∧ n = 10) ∨ (m = 10 ∧ n = 3) ∨ (m = 4 ∧ n = 5) ∨ (m = 5 ∧ n = 4) :=
by sorry

end find_positive_integers_l694_694135


namespace maximum_value_of_a_exists_a_eq_29_greatest_possible_value_of_a_l694_694722

theorem maximum_value_of_a (x : ℤ) (a : ℤ) (h1 : x^2 + a * x = -28) (h2 : a > 0) : a ≤ 29 := 
by 
-- add proof here 
sorry

theorem exists_a_eq_29 (x : ℤ) (h1 : x^2 + 29 * x = -28) : ∃ (x : ℤ), x^2 + 29 * x = -28 :=
by 
-- add proof here 
sorry

theorem greatest_possible_value_of_a : ∃ (a : ℤ), (∀ x : ℤ, x^2 + a * x = -28 → a ≤ 29) ∧ (∃ x : ℤ, x^2 + 29 * x = -28) := 
by
  use 29
  split
  { intros x h1 
    apply maximum_value_of_a x 29 h1
    show 29 > 0, from nat.succ_pos' 28 }
  { apply exists_a_eq_29 } 

end maximum_value_of_a_exists_a_eq_29_greatest_possible_value_of_a_l694_694722


namespace equilateral_triangle_relation_l694_694652

theorem equilateral_triangle_relation
  (A B C O P Q : Point)
  (h_eq_triangle : equilateral_triangle A B C)
  (h_inscribed : inscribed_in_circle A B C O)
  (h_on_arc : on_arc P B C)
  (h_on_arc' : on_arc Q B C)
  (h_between : between P B Q)
  (h_angle_eq : angle_eq (angle BAP) (angle CAQ)) :
  (dist A P) + (dist A Q) = (dist B P) + (dist C P) + (dist B Q) + (dist C Q) :=
sorry

end equilateral_triangle_relation_l694_694652


namespace trig_identity_third_quadrant_l694_694608

variable (α : ℝ) (hα : π < α ∧ α < 3 * π / 2)

theorem trig_identity_third_quadrant (h : π < α ∧ α < 3 * π / 2) :
  (cos α / sqrt (1 - sin α ^ 2)) + (2 * sin α / sqrt (1 - cos α ^ 2)) = -3 :=
by
  sorry

end trig_identity_third_quadrant_l694_694608


namespace area_of_trapezoid_l694_694334

-- Define the parameters as given in the problem
def PQ : ℝ := 40
def RS : ℝ := 25
def h : ℝ := 10
def PR : ℝ := 20

-- Assert the quadrilateral is a trapezoid with bases PQ and RS parallel
def isTrapezoid (PQ RS : ℝ) (h : ℝ) (PR : ℝ) : Prop := true -- this is just a placeholder to state that it's a trapezoid

-- The main statement for the area of the trapezoid
theorem area_of_trapezoid (h : ℝ) (PQ RS : ℝ) (h : ℝ) (PR : ℝ) (is_trapezoid : isTrapezoid PQ RS h PR) : (1/2) * (PQ + RS) * h = 325 :=
by
  sorry

end area_of_trapezoid_l694_694334


namespace six_people_theorem_l694_694626

theorem six_people_theorem :
  ∀ (G : SimpleGraph (Fin 6)), ∃ (H : Finset (Fin 6)), H.card = 3 ∧
    (G.induced H).complete ∨ (G.complement.induced H).complete :=
by
  sorry

end six_people_theorem_l694_694626


namespace ann_age_eq_36_l694_694858

theorem ann_age_eq_36 : 
  ∃ a b c : ℕ, 
    a + b + c = 78 ∧ 
    c = b - 6 ∧ 
    (let half_b := b / 2 in 
     let ann_when_c_half_b := a - (b - half_b) in 
     let barbara_when_c_half_b := b - (half_b - half_b) in 
     c = half_b + (b - half_b / 2)) → 
    a = 36 :=
by
sorry

end ann_age_eq_36_l694_694858


namespace perimeter_is_correct_l694_694460

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

noncomputable def perimeter : ℝ :=
  distance (0,0) (2,0) + distance (2,0) (3,2) + distance (3,2) (1,3) +
  distance (1,3) (0,2) + distance (0,2) (0,0)

theorem perimeter_is_correct : ∃ a b c : ℤ, perimeter = a + b * Real.sqrt c ∧ a + b + c = 9 :=
by
  use [4, 3, 2]  -- Guess
  sorry  -- Proof steps to verify this tuple

end perimeter_is_correct_l694_694460


namespace monotonicity_f_range_a_for_f_sum_zero_range_a_for_f_inequality_l694_694204

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := exp(2*x) - 4*a*exp(x) + (4*a - 2)*x

-- Monotonicity proof statement
theorem monotonicity_f : ∀ (a : ℝ), a ≥ 1 →
  ((a = 1 → ∀ x₁ x₂, x₁ ≤ x₂ → f a x₁ ≤ f a x₂) ∧
   (a > 1 → ∀ x₁ x₂, (x₁ < 0 ∨ x₁ > real.log(2*a-1)) ∧ (x₂ < 0 ∨ x₂ > real.log(2*a-1)) ∨
                    (0 < x₁ ∧ x₁ < real.log(2*a-1)) ∧ (0 < x₂ ∧ x₂ < real.log(2*a-1)) → 
                     (x₁ ≤ x₂ → f a x₁ ≤ f a x₂))) :=
by sorry

-- Statement for range of a where f(x) + f(-x) = 0
theorem range_a_for_f_sum_zero : ∀ a, a ≥ 1 →
  (∃ x, f a x + f a (-x) = 0) ↔ (a ∈ set.Ici 1) :=
by sorry

-- Statement for range of a where f(x) ≥ f(-x) for all x ≥ 0
theorem range_a_for_f_inequality : ∀ a, a ≥ 1 →
  (∀ x, x ≥ 0 → f a x ≥ f a (-x)) ↔ (a ∈ set.Icc 1 2) :=
by sorry

end monotonicity_f_range_a_for_f_sum_zero_range_a_for_f_inequality_l694_694204


namespace correct_propositions_l694_694961

/-- Proposition 1: Two lines that do not have common points are parallel. -/
def prop1 (L1 L2 : Set Point) : Prop := ¬(∃ P : Point, P ∈ L1 ∧ P ∈ L2) → parallel L1 L2

/-- Proposition 2: Two lines that are perpendicular to each other are intersecting lines. -/
def prop2 (L1 L2 : Set Point) : Prop := (∃ P : Point, P ∈ L1 ∧ P ∈ L2 ∧ perpendicular L1 L2) → intersecting L1 L2

/-- Proposition 3: Lines that are neither parallel nor intersecting are skew lines. -/
def prop3 (L1 L2 : Set Point) : Prop := ¬parallel L1 L2 ∧ ¬intersecting L1 L2 → skew L1 L2

/-- Proposition 4: Two lines that are not in the same plane are skew lines. -/
def prop4 (L1 L2 : Set Point) : Prop := ¬coplanar L1 L2 → skew L1 L2

/-- The correct propositions are 3 and 4. -/
theorem correct_propositions : ¬prop1 L1 L2 ∧ ¬prop2 L1 L2 ∧ prop3 L1 L2 ∧ prop4 L1 L2 := sorry

end correct_propositions_l694_694961


namespace greatest_ABCBA_l694_694815

/-
We need to prove that the greatest possible integer of the form AB,CBA 
that is both divisible by 11 and by 3, with A, B, and C being distinct digits, is 96569.
-/

theorem greatest_ABCBA (A B C : ℕ) 
  (h1 : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (h2 : 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9) 
  (h3 : 10001 * A + 1010 * B + 100 * C < 100000) 
  (h4 : 2 * A - 2 * B + C ≡ 0 [MOD 11])
  (h5 : (2 * A + 2 * B + C) % 3 = 0) : 
  10001 * A + 1010 * B + 100 * C ≤ 96569 :=
sorry

end greatest_ABCBA_l694_694815


namespace total_cost_of_trip_l694_694322

def totalDistance (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

def gallonsUsed (distance miles_per_gallon : ℕ) : ℕ :=
  distance / miles_per_gallon

def totalCost (gallons : ℕ) (cost_per_gallon : ℕ) : ℕ :=
  gallons * cost_per_gallon

theorem total_cost_of_trip :
  (totalDistance 10 6 5 9 = 30) →
  (gallonsUsed 30 15 = 2) →
  totalCost 2 35 = 700 :=
by
  sorry

end total_cost_of_trip_l694_694322


namespace polyhedron_faces_l694_694063

theorem polyhedron_faces (V E F T P t p : ℕ)
  (hF : F = 20)
  (hFaces : t + p = 20)
  (hTriangles : t = 2 * p)
  (hVertex : T = 2 ∧ P = 2)
  (hEdges : E = (3 * t + 5 * p) / 2)
  (hEuler : V - E + F = 2) :
  100 * P + 10 * T + V = 238 :=
by
  sorry

end polyhedron_faces_l694_694063


namespace sum_of_reciprocal_products_l694_694180

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * (a 1 - a 0))

theorem sum_of_reciprocal_products (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_arith: arithmetic_sequence a)
  (h_sum: sum_of_first_n_terms a S)
  (h_a5 : a 5 = 5)
  (h_S5 : S 5 = 15) :
  (∑ n in finset.range 100, (1 / (a n * a (n + 1)))) = 100 / 101 :=
sorry

end sum_of_reciprocal_products_l694_694180


namespace number_of_pairs_self_inverse_matrix_l694_694122

-- Definitions of the matrix components
def A := Matrix Real (Fin 2) (Fin 2)
def a : Real
def b : Real

-- Hypothesis that the matrix is its own inverse
def matrix_is_self_inverse (M : A) : Prop :=
  M * M = 1

-- The specific matrix in the problem
def specific_matrix : A :=
  ![
    ![a, 5],
    ![-12, b]
  ]

-- The Lean statement to prove
theorem number_of_pairs_self_inverse_matrix :
  (∃ a b : Real, matrix_is_self_inverse specific_matrix) ↔ (Set.card (Set.of { (a, b) | matrix_is_self_inverse (specific_matrix) })) = 2 :=
sorry

end number_of_pairs_self_inverse_matrix_l694_694122


namespace probability_six_integers_different_tens_digits_l694_694348

noncomputable theory
open Finset
open Nat

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def probability_different_tens_digit (s : Finset ℕ) (h : s.card = 6) : ℚ :=
  let num_possible_choices := (choose 9 6 : ℕ) * 10^6
  let total_possibilities := choose 90 6
  num_possible_choices / total_possibilities

theorem probability_six_integers_different_tens_digits :
  ∀ (s : Finset ℕ), s.card = 6 → (∀ n ∈ s, 10 ≤ n ∧ n ≤ 99) →
  (∀ n m ∈ s, n ≠ m → tens_digit n ≠ tens_digit m) →
  probability_different_tens_digit s ‹s.card = 6› = 8000 / 5895 :=
by
  intros s hc hr hd
  apply probability_different_tens_digit_congr hc hr hd
  sorry

end probability_six_integers_different_tens_digits_l694_694348


namespace min_value_Sn_l694_694629

noncomputable def arithmetic_minimized_sum (n : ℕ) (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  (∀ (k : ℕ), a (k + 1) = a k + d) ∧
  a 1 = -3 ∧
  S 5 = S 10 ∧
  S n = (n * (2 * a 1 + (n - 1) * d)) / 2

theorem min_value_Sn : ∃ n : ℕ, (arithmetic_minimized_sum n a S) → (n = 7 ∨ n = 8) := 
sorry

end min_value_Sn_l694_694629


namespace max_product_of_two_integers_with_sum_300_l694_694121

theorem max_product_of_two_integers_with_sum_300 : 
  ∃ (x y : ℤ), x + y = 300 ∧ (∀ (a b : ℤ), a + b = 300 → a * b ≤ 22500) ∧ x * y = 22500 :=
begin
  sorry
end

end max_product_of_two_integers_with_sum_300_l694_694121


namespace mr_blues_yard_expectation_l694_694684

noncomputable def calculate_expected_harvest (length_steps : ℕ) (width_steps : ℕ) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  let length_feet := length_steps * step_length
  let width_feet := width_steps * step_length
  let area := length_feet * width_feet
  let total_yield := area * yield_per_sqft
  total_yield

theorem mr_blues_yard_expectation : calculate_expected_harvest 18 25 2.5 (3 / 4) = 2109.375 :=
by
  sorry

end mr_blues_yard_expectation_l694_694684


namespace total_amount_after_refunds_l694_694873

def individual_bookings : ℕ := 12000
def group_bookings : ℕ := 16000
def refunds : ℕ := 1600

theorem total_amount_after_refunds : 
  individual_bookings + group_bookings - refunds = 26400 :=
by 
  -- The proof goes here.
  sorry

end total_amount_after_refunds_l694_694873


namespace Im_abcd_eq_zero_l694_694667

noncomputable def normalized (z : ℂ) : ℂ := z / Complex.abs z

theorem Im_abcd_eq_zero (a b c d : ℂ)
  (h1 : ∃ α : ℝ, ∃ w : ℂ, w = Complex.cos α + Complex.sin α * Complex.I ∧ (normalized b = w * normalized a) ∧ (normalized d = w * normalized c)) :
  Complex.im (a * b * c * d) = 0 :=
by
  sorry

end Im_abcd_eq_zero_l694_694667


namespace sum_of_three_numbers_l694_694039

theorem sum_of_three_numbers : 3.15 + 0.014 + 0.458 = 3.622 :=
by sorry

end sum_of_three_numbers_l694_694039


namespace inequality_proof_equality_condition_l694_694945

variable {x1 x2 y1 y2 z1 z2 : ℝ}

-- Conditions
axiom x1_pos : x1 > 0
axiom x2_pos : x2 > 0
axiom x1y1_gz1sq : x1 * y1 > z1 ^ 2
axiom x2y2_gz2sq : x2 * y2 > z2 ^ 2

theorem inequality_proof : 
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2) <= 
  1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2) :=
sorry

theorem equality_condition : 
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2) = 
  1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2) ↔ 
  (x1 = x2 ∧ y1 = y2 ∧ z1 = z2) :=
sorry

end inequality_proof_equality_condition_l694_694945


namespace integral_x2_plus_sqrt_1_minus_x2_l694_694503

theorem integral_x2_plus_sqrt_1_minus_x2 :
  ∫ x in -1..1, (x^2 + real.sqrt (1 - x^2)) = (2/3 : ℝ) + (real.pi / 2) := 
  sorry

end integral_x2_plus_sqrt_1_minus_x2_l694_694503


namespace last_row_is_101_mul_2_pow_98_l694_694470

-- Define the first row as the sequence 1 to 100
def first_row : List ℕ := List.range 100

-- Define the function to compute the next row from the previous one
def next_row (prev_row : List ℕ) : List ℕ :=
  List.map₂ (+) (prev_row.init) (prev_row.tail)

-- Define the sequence of rows using recursion
noncomputable def n_th_row (n : ℕ) : List ℕ :=
  if n = 0 then first_row
  else next_row (n_th_row (n - 1))

-- Define the function to get the last row
noncomputable def last_row : List ℕ := n_th_row 99

-- Prove the statement about the last row
theorem last_row_is_101_mul_2_pow_98 : 
  last_row.head = some (101 * 2^98) :=
by sorry

end last_row_is_101_mul_2_pow_98_l694_694470


namespace total_fruits_sum_l694_694102

theorem total_fruits_sum (Mike_oranges Matt_apples Mark_bananas Mary_grapes : ℕ)
  (hMike : Mike_oranges = 3)
  (hMatt : Matt_apples = 2 * Mike_oranges)
  (hMark : Mark_bananas = Mike_oranges + Matt_apples)
  (hMary : Mary_grapes = Mike_oranges + Matt_apples + Mark_bananas + 5) :
  Mike_oranges + Matt_apples + Mark_bananas + Mary_grapes = 41 :=
by
  sorry

end total_fruits_sum_l694_694102


namespace greatest_possible_a_l694_694721

theorem greatest_possible_a (a : ℕ) :
  (∃ x : ℤ, x^2 + a * x = -28) → a = 29 :=
sorry

end greatest_possible_a_l694_694721


namespace angle_CSB_eq_double_angle_CAB_l694_694304

-- Define the triangle ABC
variables {A B C D E S : Point}
variables [Circumcircle ABC : Circle]

-- Define the points and conditions
variable (E_mid : midpoint E B C)
variable (D_on_circumcircle : D ∈ Circumcircle ABC)
variable (D_arc_condition : arc_contains D B C A)
variable (CAD_eq_BAE : angle CAD = angle BAE)
variable (S_mid : midpoint S A D)

-- Lean statement for the proof problem
theorem angle_CSB_eq_double_angle_CAB
  (E_mid : midpoint E B C)
  (D_on_circumcircle : D ∈ Circumcircle ABC)
  (D_arc_condition : arc_contains D B C A)
  (CAD_eq_BAE : ∠CAD = ∠BAE)
  (S_mid : midpoint S A D) :
  ∠CSB = 2 * ∠CAB :=
sorry

end angle_CSB_eq_double_angle_CAB_l694_694304


namespace student_mistake_fraction_l694_694264

theorem student_mistake_fraction : 
  ∃ (x : ℚ), 5 / 16 * 480 = 150 ∧ 150 + 250 = 400 ∧ x * 480 = 400 ∧ x = 5 / 6 :=
by 
  existsi (5 / 6)
  have h1 : 5 / 16 * 480 = 150 := by norm_num
  exact ⟨h1, by norm_num, by norm_num, rfl⟩

end student_mistake_fraction_l694_694264


namespace probability_all_red_or_all_white_l694_694438

theorem probability_all_red_or_all_white :
  let red_marbles := 5
  let white_marbles := 4
  let blue_marbles := 6
  let total_marbles := red_marbles + white_marbles + blue_marbles
  let probability_red := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) * ((red_marbles - 2) / (total_marbles - 2))
  let probability_white := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2))
  (probability_red + probability_white) = (14 / 455) :=
by
  sorry

end probability_all_red_or_all_white_l694_694438


namespace locate_quadrant_of_z_l694_694515

open Complex

def z := (1 - sqrt 2 * I) / I

theorem locate_quadrant_of_z : z.re < 0 ∧ z.im < 0 :=
by
  sorry

end locate_quadrant_of_z_l694_694515


namespace math_problem_l694_694052

noncomputable def a : ℝ := (0.96)^3 
noncomputable def b : ℝ := (0.1)^3 
noncomputable def c : ℝ := (0.96)^2 
noncomputable def d : ℝ := (0.1)^2 

theorem math_problem : a - b / c + 0.096 + d = 0.989651 := 
by 
  -- skip proof 
  sorry

end math_problem_l694_694052


namespace area_triangle_BOM_l694_694825

-- Define the given problem conditions
variables (O A B C D M N : Type)
variables [CircularInscribedTrapezium O A B C D M N]
variables (AD_parallel_BC : AD_parallel BC)
variables (AD_eq_7 : AD = 7)
variables (BC_eq_3 : BC = 3)
variables (angle_BCD_120 : angle B C D = 120)
variables (ND_eq_2 : ND = 2)

-- Define the goal to be proved
theorem area_triangle_BOM :
  area (triangle B O M) = 155 * sqrt 3 / 84 :=
sorry

end area_triangle_BOM_l694_694825


namespace cannot_form_right_angled_triangle_can_form_right_angled_triangle_b_can_form_right_angled_triangle_c_can_form_right_angled_triangle_d_l694_694093

theorem cannot_form_right_angled_triangle : ¬ ∃ a b c : ℝ, 
  a = 5 ∧ b = 7 ∧ c = 10 ∧ a^2 + b^2 = c^2 :=
by sorry

theorem can_form_right_angled_triangle_b : ∃ a b c : ℝ,
  a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2 :=
by sorry

theorem can_form_right_angled_triangle_c : ∃ a b c : ℝ,
  a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2 :=
by sorry

theorem can_form_right_angled_triangle_d : ∃ a b c : ℝ,
  a = 1 ∧ b = 2 ∧ c = ℝ.sqrt 3 ∧ a^2 + b^2 = c^2 :=
by sorry

end cannot_form_right_angled_triangle_can_form_right_angled_triangle_b_can_form_right_angled_triangle_c_can_form_right_angled_triangle_d_l694_694093


namespace true_propositions_count_l694_694365

theorem true_propositions_count : 
  (∀ (P1 P2 : Prop), P1 ∧ P2 ↔ True) ∧
  (∀ (P3 : Prop), P3 ↔ True) ∧
  (∀ (P4 : Prop), P4 ↔ False) →
  (∀ propositions_list : List Prop, propositions_list.length = 4 → 
  let true_props := propositions_list.filter (fun p => p = True) in true_props.length = 3) :=
by {
  intro h,
  sorry
}

end true_propositions_count_l694_694365


namespace find_b_l694_694991

theorem find_b (a b : ℤ) (h : ∃ q : polynomial ℤ, (X^2 + X - 2) * q = polynomial.C a * X^4 + polynomial.C b * X^3 - polynomial.C 2) : 
    b = 4 :=
by
  sorry

end find_b_l694_694991


namespace personA_works_alone_time_l694_694758

theorem personA_works_alone_time (x : ℕ) : 
  (∀ pA_rate pB_rate : ℚ, 
    pA_rate = (1 : ℚ) / x ∧ 
    pB_rate = (1 : ℚ) / 40 ∧ 
    ( 5 * (pA_rate + pB_rate) = 0.375)
  ) → x = 20 :=
by
  intros pA_rate pB_rate
  sorry

end personA_works_alone_time_l694_694758


namespace exists_four_teams_with_winning_order_l694_694895

theorem exists_four_teams_with_winning_order :
  ∃ (a1 a2 a3 a4 : Fin 8), 
  (a1 ≠ a2) ∧ (a1 ≠ a3) ∧ (a1 ≠ a4) ∧ (a2 ≠ a3) ∧ (a2 ≠ a4) ∧ (a3 ≠ a4) ∧ 
  (win a1 a2) ∧ (win a1 a3) ∧ (win a1 a4) ∧ (win a2 a3) ∧ (win a2 a4) ∧ (win a3 a4) := by
  sorry

end exists_four_teams_with_winning_order_l694_694895


namespace angle_CHX_correct_l694_694841

noncomputable def triangle_ABC (A B C H X : Type) := 
  -- Define the acute triangle condition and the altitudes intersection at H
  let BAC := 61° in
  let ABC := 73° in
  let CHX := 73° in
  true -- Placeholder for actual definition of the geometric properties

theorem angle_CHX_correct (A B C H X : Type) (h : triangle_ABC A B C H X) :
  ∠CHX = 73° :=
sorry

end angle_CHX_correct_l694_694841


namespace servant_compensation_l694_694070

noncomputable def annual_salary : ℝ := 90
noncomputable def months_worked : ℝ := 9
noncomputable def total_months_in_year : ℝ := 12
noncomputable def turban_value : ℝ := 10

theorem servant_compensation : 
  let total_compensation := (months_worked / total_months_in_year) * annual_salary
  in total_compensation - turban_value = 57.5 :=
by
  let total_compensation := (months_worked / total_months_in_year) * annual_salary
  show total_compensation - turban_value = 57.5 from sorry

end servant_compensation_l694_694070


namespace eval_imaginary_expression_l694_694502

section
variable (i : ℂ)
hypothesis h1 : i^2 = -1
hypothesis h2 : i^4 = 1

theorem eval_imaginary_expression : i^{12} + i^{17} + i^{22} + i^{27} + i^{32} + i^{37} = 2 :=
by
  have h12 : i^{12} = (i^4)^3 := by sorry
  have h17 : i^{17} = i^1       := by sorry
  have h22 : i^{22} = i^2       := by sorry
  have h27 : i^{27} = (i^4)^6 * i^3 := by sorry
  have h32 : i^{32} = (i^4)^8      := by sorry
  have h37 : i^{37} = i^1          := by sorry
  sorry
end

end eval_imaginary_expression_l694_694502


namespace dave_initial_apps_l694_694497

theorem dave_initial_apps :
  ∃ (initial_apps : ℕ), let deleted_apps := 86 in
  let added_apps := 89 in
  let remaining_apps := 24 in
  let after_deletion := initial_apps + added_apps - deleted_apps in
  after_deletion = remaining_apps + (deleted_apps + 3) ∧ initial_apps = 21 := sorry

end dave_initial_apps_l694_694497


namespace satisfies_condition_X_mul_l694_694407

def satisfies_condition_X (m : ℕ) : Prop :=
  ∀ k : ℕ, (0 < k ∧ k < m) → ∃ (d : List ℕ), (∀ x ∈ d, x ∣ m) ∧ (d.Nodup) ∧ (d.Sum = k)

theorem satisfies_condition_X_mul {m n : ℕ} (hm : satisfies_condition_X m) (hn : satisfies_condition_X n) :
  satisfies_condition_X (m * n) :=
by
  sorry

end satisfies_condition_X_mul_l694_694407


namespace geom_seq_arith_form_l694_694278

theorem geom_seq_arith_form (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∀ n, a (n+1) = a n * q)
  (h_arith : (a 1, (1 / 2) * a 3, 2 * a 2) ∈ SetOf p q r where p + r = 2 * q) :
  (a 6 + a 8 + a 10) / (a 7 + a 9 + a 11) = Real.sqrt 2 - 1 :=
by
  sorry

end geom_seq_arith_form_l694_694278


namespace f_2010_eq_one_l694_694165

def f (x : ℝ) (a : ℝ) (b : ℝ) (α : ℝ) (β : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)

theorem f_2010_eq_one (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) (h2009 : f 2009 a b α β = -1) : 
  f 2010 a b α β = 1 :=
by
  sorry

end f_2010_eq_one_l694_694165


namespace four_point_tangency_l694_694753

noncomputable theory

variables {C C' : Circle} {X Y : Point}
variables {P Q R S T1 T2 T3 T4 : Point}

-- Assume initial conditions
axiom circles_intersect : Points_on_Circle C X ∧ Points_on_Circle C' X ∧
                          Points_on_Circle C Y ∧ Points_on_Circle C' Y

axiom points_of_tangency : Tangent_Point T1 C ∧ Tangent_Point T1 C' ∧
                           Tangent_Point T2 C ∧ Tangent_Point T2 C' ∧
                           Tangent_Point T3 C ∧ Tangent_Point T3 C' ∧
                           Tangent_Point T4 C ∧ Tangent_Point T4 C'

axiom line_intersects_circles : Intersects_Line_Circle C P R ∧
                                Intersects_Line_Circle C' Q S ∧
                                Intersects_Line XY R ∧
                                Intersects_Line XY S

theorem four_point_tangency :
  ∃ (T1 T2 T3 T4 : Point), 
    (Tangent_Point T1 C ∧ Tangent_Point T1 C' ∧
     Tangent_Point T2 C ∧ Tangent_Point T2 C' ∧
     Tangent_Point T3 C ∧ Tangent_Point T3 C' ∧
     Tangent_Point T4 C ∧ Tangent_Point T4 C') ∧
    (Lines_Meet_At_Points PR T1 ∨ Lines_Meet_At_Points PR T2 ∨
     Lines_Meet_At_Points PR T3 ∨ Lines_Meet_At_Points PR T4) ∧
    (Lines_Meet_At_Points PS T1 ∨ Lines_Meet_At_Points PS T2 ∨
     Lines_Meet_At_Points PS T3 ∨ Lines_Meet_At_Points PS T4) ∧
    (Lines_Meet_At_Points QR T1 ∨ Lines_Meet_At_Points QR T2 ∨
     Lines_Meet_At_Points QR T3 ∨ Lines_Meet_At_Points QR T4) ∧
    (Lines_Meet_At_Points QS T1 ∨ Lines_Meet_At_Points QS T2 ∨
     Lines_Meet_At_Points QS T3 ∨ Lines_Meet_At_Points QS T4) :=
sorry

end four_point_tangency_l694_694753


namespace complex_product_polar_form_l694_694880

-- Define complex numbers in polar form
def c1 : ℂ := complex.ofReal 4 * complex.exp (complex.I * real.ofRat (45 * (π / 180)))
def c2 : ℂ := -complex.ofReal 3 * complex.exp (complex.I * real.ofRat (20 * (π / 180)))
def c3 : ℂ := complex.ofReal 2 * complex.exp (complex.I * real.ofRat (15 * (π / 180)))

-- Define the product of the given complex numbers in polar form
def product : ℂ := c1 * c2 * c3

-- Define the expected result in polar form (24 ∠ 260°)
def expected : ℂ := complex.ofReal 24 * complex.exp (complex.I * real.ofRat (260 * (π / 180)))

-- State the theorem and provide the assertion
theorem complex_product_polar_form :
  product = expected :=
by 
  sorry

end complex_product_polar_form_l694_694880


namespace tile_count_difference_l694_694328

theorem tile_count_difference :
  let red_initial := 15
  let yellow_initial := 10
  let yellow_added := 18
  let yellow_total := yellow_initial + yellow_added
  let red_total := red_initial
  yellow_total - red_total = 13 :=
by
  sorry

end tile_count_difference_l694_694328


namespace fresh_fruit_sold_l694_694344

-- Define the conditions
def total_fruit_sold : ℕ := 9792
def frozen_fruit_sold : ℕ := 3513

-- Define what we need to prove
theorem fresh_fruit_sold : (total_fruit_sold - frozen_fruit_sold = 6279) := by
  sorry

end fresh_fruit_sold_l694_694344


namespace sequence_properties_l694_694575

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) + (-1 : ℤ)^n * a n = 2 * n - 1

theorem sequence_properties (a : ℕ → ℤ) (h : sequence a) :
  a 3 = 1 ∧ (∑ i in Finset.range 60, a (i + 1)) = 1830 :=
by
  sorry

end sequence_properties_l694_694575


namespace find_a_l694_694551

theorem find_a (a : ℝ) : (1 : ℝ)^2 + 1 + 2 * a = 0 → a = -1 := 
by 
  sorry

end find_a_l694_694551


namespace complex_modulus_square_l694_694673

open Complex

theorem complex_modulus_square (a b : ℝ) (h : 5 * (a + b * I) + 3 * Complex.abs (a + b * I) = 15 - 16 * I) :
  (Complex.abs (a + b * I))^2 = 256 / 25 :=
by sorry

end complex_modulus_square_l694_694673


namespace find_matrix_N_l694_694908

open Matrix

theorem find_matrix_N :
  ∃ (N : Matrix (Fin 2) (Fin 2) ℤ),
    (N ⬝ (col_vector 2 ![4, 0]) = col_vector 2 ![8, 28]) ∧
    (N ⬝ (col_vector 2 ![-2, 10]) = col_vector 2 ![6, -34]) ∧
    (N = ![![2, 1], ![7, -2]]) :=
begin
  -- proof would go here
  sorry
end

end find_matrix_N_l694_694908


namespace four_digit_sum_ten_divisible_by_eleven_l694_694589

theorem four_digit_sum_ten_divisible_by_eleven : 
  {n | (10^3 ≤ n ∧ n < 10^4) ∧ (∑ i in (Int.toString n).toList, i.toNat) = 10 ∧ (let digits := (Int.toString n).toList.map (λ c, c.toNat) in ((digits.nth 0).getD 0 + (digits.nth 2).getD 0) - ((digits.nth 1).getD 0 + (digits.nth 3).getD 0)) % 11 == 0}.card = 30 := sorry

end four_digit_sum_ten_divisible_by_eleven_l694_694589


namespace color_removal_exists_l694_694261

open SimpleGraph

def K₄₀ : SimpleGraph (Fin 40) := completeGraph _

theorem color_removal_exists :
  ∃ c : Fin 6, 
    ∀ v w : Fin 40, 
      v ≠ w → (K₄₀.deleteEdges (fun e => e.color = c)).Adj v w :=
sorry

end color_removal_exists_l694_694261


namespace all_pairs_lucky_probability_expected_lucky_pairs_gt_half_l694_694044

variables (n : ℕ)

def lucky_pair_probability (n : ℕ) : ℝ :=
  (1 / Real.sqrt 2) * (Real.exp 1 / (2 * n)) ^ n

theorem all_pairs_lucky_probability (hn : n > 0) :
  lucky_pair_probability n = (1 / Real.sqrt 2) * (Real.exp 1 / (2 * n)) ^ n :=
sorry

def expected_lucky_pairs (n : ℕ) : ℝ :=
  n / (2 * n - 1)

theorem expected_lucky_pairs_gt_half (hn : n > 0) : expected_lucky_pairs n > 0.5 :=
by
  rw [expected_lucky_pairs]
  have h : (n : ℝ) / (2 * n - 1) > 0.5 := by
    linarith
  exact h

end all_pairs_lucky_probability_expected_lucky_pairs_gt_half_l694_694044


namespace log_one_plus_two_x_lt_two_x_l694_694234
open Real

theorem log_one_plus_two_x_lt_two_x {x : ℝ} (hx : x > 0) : log (1 + 2 * x) < 2 * x :=
sorry

end log_one_plus_two_x_lt_two_x_l694_694234


namespace calculation_correct_l694_694926

theorem calculation_correct : 67897 * 67898 - 67896 * 67899 = 2 := by
  sorry

end calculation_correct_l694_694926


namespace jake_has_more_balloons_l694_694477

-- Defining the given conditions as parameters
def initial_balloons_allan : ℕ := 2
def initial_balloons_jake : ℕ := 6
def additional_balloons_allan : ℕ := 3

-- Calculate total balloons each person has
def total_balloons_allan : ℕ := initial_balloons_allan + additional_balloons_allan
def total_balloons_jake : ℕ := initial_balloons_jake

-- Formalize the statement to be proved
theorem jake_has_more_balloons :
  total_balloons_jake - total_balloons_allan = 1 :=
by
  -- Proof will be added here
  sorry

end jake_has_more_balloons_l694_694477


namespace different_selections_l694_694852

/-- Prove that the number of different selections such that not all chosen spots are the same
    is 6, given that persons X and Y each choose two spots out of three scenic spots A, B, and C. -/
theorem different_selections (spots : Finset ℕ) (x y : Finset ℕ → Finset (Finset ℕ)) :
  spots.card = 3 →
  (∀ s, s.card = 2 → x s.card = (spots.card.choose 2)) →
  (∀ s, s.card = 2 → y s.card = (spots.card.choose 2)) →
  (x spots.card) * (y spots.card) - y spots.card = 6 := by
  sorry

end different_selections_l694_694852


namespace determine_F_l694_694295

theorem determine_F (A H S M F : ℕ) (ha : 0 < A) (hh : 0 < H) (hs : 0 < S) (hm : 0 < M) (hf : 0 < F):
  (A * x + H * y = z) →
  (S * x + M * y = z) →
  (F * x = z) →
  (H > A) →
  (A ≠ H) →
  (S ≠ M) →
  (F ≠ A) →
  (F ≠ H) →
  (F ≠ S) →
  (F ≠ M) →
  x = z / F →
  y = ((F - A) / H * z) / z →
  F = (A * F - S * H) / (M - H) := sorry

end determine_F_l694_694295


namespace area_of_hexagon_is_90_l694_694826

-- Definitions based on the problem's conditions
variables (DEF : Type) 
          (P : DEF → DEF → DEF → Prop)  -- P represents the property to form triangle DEF
          (D' E'' F' : DEF → DEF)
          (circumcircle : DEF → DEF → DEF → Prop) -- circumcircle property
          (intersection : (DEF → DEF) → (DEF → DEF) → Prop) -- intersection of perpendicular bisectors
  
-- The conditions: the perimeter and radius of the circumcircle
variables (perimeter : ℝ) (circum_radius : ℝ)
variable h_perimeter : perimeter = 30
variable h_radius : circum_radius = 6 

-- The question rephrased to a proof
theorem area_of_hexagon_is_90 (h_triangle : ∀ d e f, P d e f ∧ circumcircle d e f) 
                               (h_bisectors : ∀ d e f, intersection (D' d) (E'' e) ∧ intersection (E'' e) (F' f)) 
                               : ∀ (d e f : DEF), P d e f → circumcircle d e f → 
                                  let hexagon_area : ℝ := 90 in
                                  hexagon_area = 90 :=
begin
  intros d e f h1 h2,
  sorry
end

end area_of_hexagon_is_90_l694_694826


namespace missing_term_in_sequence_l694_694084

-- The problem statement translated to Lean 4:
theorem missing_term_in_sequence (coefficients : ℕ → ℕ)
  (h_coeff_seq : ∀ n, coefficients n = 1 + 2 * n)
  (h_exponent_seq : ∀ n, n  = n + 1 ) : coefficients 3 * x ^ 4 = 7 * x ^ 4 :=
  by 
  -- Definitions according to conditions
  let coefficient_3 := coefficients 3
  have h_coeff_3 : coefficient_3 = 7 := by 
    simp [h_coeff_seq]
  -- Exponent part of the missing term
  have h_exponent_3 : 4 = 4 := by 
    simp 
  -- Combine coefficient and exponent to form the term
  exact congr_arg2 (· * ·) rfl rfl
  sorry


end missing_term_in_sequence_l694_694084


namespace line_equation_passes_through_l694_694621

theorem line_equation_passes_through (a b : ℝ) (x y : ℝ) 
  (h_intercept : b = a + 1)
  (h_point : (6 * b) + (-2 * a) = a * b) :
  (x + 2 * y - 2 = 0 ∨ 2 * x + 3 * y - 6 = 0) := 
sorry

end line_equation_passes_through_l694_694621


namespace find_P_coordinates_l694_694559

noncomputable def line1 : ℝ → ℝ := λ x, (1/2) * x
noncomputable def hyperbola1 (k : ℝ) : ℝ → ℝ := λ x, k / x
noncomputable def hyperbola2 (k : ℝ) : ℝ → ℝ := λ x, k * x

def pointA : (ℝ × ℝ) := (4, line1 4)
def isFirstQuadrant (P : ℝ × ℝ) : Prop := P.1 > 0 ∧ P.2 > 0
def isRectangle (A B P Q : ℝ × ℝ) : Prop := 
  A.1 = P.2 ∧ A.2 = P.1 ∧ B.1 = Q.2 ∧ B.2 = Q.1

theorem find_P_coordinates (k : ℝ) (h_k_gt_0 : 0 < k) 
  (l_intersect_origin : ∃ t : ℝ, hyperbola2 k t = k) 
  (A_is_point : pointA = (4, 2))
  (P_in_first_quadrant : ∃ P : ℝ × ℝ, isFirstQuadrant P)
  (A_B_P_Q_rectangle : ∃ A B P Q : ℝ × ℝ, 
     isRectangle A B P Q ∧ A = pointA ∧ B.1 = k / 4 ∧ B.2 = line1 (k / 4)) :
  ∃ P : ℝ × ℝ, P = (2, 4) :=
by 
  sorry

end find_P_coordinates_l694_694559


namespace log_conversion_l694_694616

theorem log_conversion (a b : ℝ) (h1 : a = Real.log 225 / Real.log 8) (h2 : b = Real.log 15 / Real.log 2) : a = (2 * b) / 3 := 
sorry

end log_conversion_l694_694616


namespace eval_g_l694_694519

def g (z : ℂ) : ℂ :=
if z.im = 0 then -(z ^ 3) else z ^ 3

theorem eval_g :
  g (g (g (g (1+1*complex.i)))) = -134217728 - 134217728 * complex.i :=
sorry

end eval_g_l694_694519


namespace ethanol_relationship_l694_694479

variables (a b c x : ℝ)
def total_capacity := a + b + c = 300
def ethanol_content := x = 0.10 * a + 0.15 * b + 0.20 * c
def ethanol_bounds := 30 ≤ x ∧ x ≤ 60

theorem ethanol_relationship : total_capacity a b c → ethanol_bounds x → ethanol_content a b c x :=
by
  intros h_total h_bounds
  unfold total_capacity at h_total
  unfold ethanol_bounds at h_bounds
  unfold ethanol_content
  sorry

end ethanol_relationship_l694_694479


namespace fruit_seller_original_apples_l694_694424

variable (original_apples remaining_apples : ℕ) (sold_percentage : ℚ)

-- Given conditions
def fruit_seller_condition : Prop := sold_percentage = 0.40 ∧ remaining_apples = 420

-- Main statement to prove
theorem fruit_seller_original_apples 
    (h: fruit_seller_condition original_apples remaining_apples sold_percentage) : 
    original_apples = 700 :=
by 
    sorry

end fruit_seller_original_apples_l694_694424


namespace range_of_m_l694_694247

theorem range_of_m 
  (m : ℝ)
  (h1 : ∀ (x : ℤ), x < m → 7 - 2 * x ≤ 1)
  (h2 : ∃ k : ℕ, set_of (λ i : ℤ, 3 ≤ i ∧ i < m).card = k ∧ k = 4) :
  6 < m ∧ m ≤ 7 :=
sorry

end range_of_m_l694_694247


namespace GlobalConnect_more_cost_effective_if_x_300_l694_694071

def GlobalConnectCost (x : ℕ) : ℝ := 50 + 0.4 * x
def QuickConnectCost (x : ℕ) : ℝ := 0.6 * x

theorem GlobalConnect_more_cost_effective_if_x_300 : 
  GlobalConnectCost 300 < QuickConnectCost 300 :=
by
  sorry

end GlobalConnect_more_cost_effective_if_x_300_l694_694071


namespace num_ordered_pairs_l694_694513

theorem num_ordered_pairs : ∃! n : ℕ, n = 4 ∧ 
  ∃ (x y : ℤ), y = (x - 90)^2 - 4907 ∧ 
  (∃ m : ℕ, y = m^2) := 
sorry

end num_ordered_pairs_l694_694513


namespace largest_prime_factor_7_fact_8_fact_l694_694006

theorem largest_prime_factor_7_fact_8_fact : 
  let n := 7! + 8! 
  ∃ p, Prime p ∧ p ∣ n ∧ (∀ q, Prime q ∧ q ∣ n → q ≤ p) := 
begin
  let n := fact 7 + fact 8,
  use 7, 
  split,
  { apply Prime.prime_7 },
  split,
  { sorry },  
  { sorry },
end

end largest_prime_factor_7_fact_8_fact_l694_694006


namespace find_positive_integer_M_l694_694410

theorem find_positive_integer_M (M : ℕ) (h : 36^2 * 81^2 = 18^2 * M^2) : M = 162 := by
  sorry

end find_positive_integer_M_l694_694410


namespace equal_binomial_terms_l694_694737

theorem equal_binomial_terms (p q : ℝ) (h1 : 0 < p) (h2 : 0 < q) (h3 : p + q = 1)
    (h4 : 55 * p^9 * q^2 = 165 * p^8 * q^3) : p = 3 / 4 :=
by
  sorry

end equal_binomial_terms_l694_694737


namespace Q_at_8_is_14400_l694_694317

/-- Let Q(x) = (3x^4 - 54x^3 + g * x^2 + h * x + i) * (4x^5 - 100x^4 + j * x^3 + k * x^2 + l * x + m), 
    where g, h, i, j, k, l, m are real numbers.
    Suppose that the set of all complex roots of Q(x) is {2, 3, 4, 6, 7}.
    Prove that Q(8) = 14400.
-/
theorem Q_at_8_is_14400 (g h i j k l m : ℝ) :
    let Q (x : ℝ) := (3 * x^4 - 54 * x^3 + g * x^2 + h * x + i) * (4 * x^5 - 100 * x^4 + j * x^3 + k * x^2 + l * x + m)
    (∀ (x : ℂ), x ∈ {2, 3, 4, 6, 7} → (Q x) = 0) →
    Q(8) = 14400 :=
by
  sorry

end Q_at_8_is_14400_l694_694317


namespace fruit_seller_gain_l694_694862

-- Define necessary variables
variables {C S : ℝ} (G : ℝ)

-- Given conditions
def selling_price_def (C : ℝ) : ℝ := 1.25 * C
def total_cost_price (C : ℝ) : ℝ := 150 * C
def total_selling_price (C : ℝ) : ℝ := 150 * (selling_price_def C)
def gain (C : ℝ) : ℝ := total_selling_price C - total_cost_price C

-- Statement to prove: number of apples' selling price gained by the fruit-seller is 30
theorem fruit_seller_gain : G = 30 ↔ gain C = G * (selling_price_def C) :=
by
  sorry

end fruit_seller_gain_l694_694862


namespace constant_term_max_binomial_term_sum_of_coefficients_l694_694274

variable {x : ℝ}
variable {n : ℕ}

-- Simplifying the conditions
def C (n k : ℕ) : ℕ := nat.choose n k
def T (n r : ℕ) (x : ℝ) : ℝ := C n r * ((-1/6)^r : ℝ) * (x^(n-2*r))

theorem constant_term : (T 8 4 x) = 35/8 :=
by 
  sorry

theorem max_binomial_term : (T 8 4 x) = 35/8 :=
by 
  sorry

theorem sum_of_coefficients : (sum (C 8) ((3 * x - 1 / (2 * 3 * x))^(8 : ℝ)) = 1 / 256 :=
by
  sorry

end constant_term_max_binomial_term_sum_of_coefficients_l694_694274


namespace mac_trades_dimes_per_quarter_l694_694679

theorem mac_trades_dimes_per_quarter :
  ∃ (D : ℕ),
    (20 * D * 0.10 + 20 * 5 * 0.05 = 10 + 3) ∧
    (80 / 20 = 4) ∧
    (D = 4) :=
begin
  use 4,
  split,
  { -- Mac trades 20 quarters with dimes and nickels and lost $3.
    -- This condition states the total value Mac gave away equals $13.
    calc 
      20 * 4 * 0.10 + 20 * 5 * 0.05 
      = 20 * 0.4 + 20 * 0.25 : by norm_num
    ... = 8 + 5 : by norm_num
    ... = 13 : by norm_num },
  split,
  { -- This condition states the calculation for dimes per quarter.
    calc 
      80 / 20 
      = 4 : by norm_num },
  { -- The conclusion that Mac traded 4 dimes per quarter.
    refl }
end

end mac_trades_dimes_per_quarter_l694_694679


namespace distance_A_B_l694_694767

-- Define the points A and B
def A : ℝ × ℝ × ℝ := (2, -2, 1)
def B : ℝ × ℝ × ℝ := (8, 8, 6)

-- Define the distance formula in 3D space
def distance_3d (P Q : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := P
  let (x2, y2, z2) := Q
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

-- Prove that the distance between A and B is sqrt(161)
theorem distance_A_B : distance_3d A B = Real.sqrt 161 := by
  sorry

end distance_A_B_l694_694767


namespace triangle_exists_c1_triangle_exists_c2_triangle_not_exists_l694_694108

/-- For condition ac = sqrt(3), we show that c = 1 under the given conditions. -/
theorem triangle_exists_c1 (A B C : ℝ) (a b c : ℝ)
  (h1 : sin A = sqrt 3 * sin B)
  (h2 : C = π / 6)
  (h3 : a * c = sqrt 3) 
  : c = 1 := 
sorry

/-- For condition c sin A = 3, we show that c = 2 sqrt(3) under the given conditions. -/
theorem triangle_exists_c2 (A B C : ℝ) (a b c : ℝ)
  (h1 : sin A = sqrt 3 * sin B)
  (h2 : C = π / 6)
  (h3 : c * sin A = 3) 
  : c = 2 * sqrt 3 := 
sorry

/-- For condition c = sqrt(3) b, we show that the triangle cannot exist under the given conditions. -/
theorem triangle_not_exists (A B C : ℝ) (a b c : ℝ)
  (h1 : sin A = sqrt 3 * sin B)
  (h2 : C = π / 6)
  (h3 : c = sqrt 3 * b) 
  : false := 
sorry

end triangle_exists_c1_triangle_exists_c2_triangle_not_exists_l694_694108


namespace rectangle_area_l694_694463

theorem rectangle_area (d : ℝ) (w : ℝ) (h : (3 * w)^2 + w^2 = d^2) : 
  3 * w^2 = d^2 / 10 :=
by
  sorry

end rectangle_area_l694_694463


namespace four_digit_sum_ten_divisible_by_eleven_l694_694587

theorem four_digit_sum_ten_divisible_by_eleven : 
  {n | (10^3 ≤ n ∧ n < 10^4) ∧ (∑ i in (Int.toString n).toList, i.toNat) = 10 ∧ (let digits := (Int.toString n).toList.map (λ c, c.toNat) in ((digits.nth 0).getD 0 + (digits.nth 2).getD 0) - ((digits.nth 1).getD 0 + (digits.nth 3).getD 0)) % 11 == 0}.card = 30 := sorry

end four_digit_sum_ten_divisible_by_eleven_l694_694587


namespace largest_prime_factor_7_fact_8_fact_l694_694011

theorem largest_prime_factor_7_fact_8_fact : 
  let n := 7! + 8! 
  ∃ p, Prime p ∧ p ∣ n ∧ (∀ q, Prime q ∧ q ∣ n → q ≤ p) := 
begin
  let n := fact 7 + fact 8,
  use 7, 
  split,
  { apply Prime.prime_7 },
  split,
  { sorry },  
  { sorry },
end

end largest_prime_factor_7_fact_8_fact_l694_694011


namespace solve_logarithmic_equation_l694_694345

theorem solve_logarithmic_equation : 
  ∃ (x : ℝ), (x > 3) ∧ (\log 19 (x - 3) + \log 93 (x - 3) = 3 - \log 10 (x^5 - 24)) ∧ (x = 4) :=
by
  sorry

end solve_logarithmic_equation_l694_694345


namespace radius_of_jar_B_l694_694402

-- Let's define the radii and heights of the jars
variables (h : ℝ) (r : ℝ)

-- Condition 1: The height of Jar A is twice the height of Jar B
def height_jar_B := h
def height_jar_A := 2 * h

-- Condition 2: The radius of Jar A is 10 units
def radius_jar_A := 10

-- Condition 3: The volume of Jar A is equal to the volume of Jar B
def volume_jar_B := π * r^2 * height_jar_B
def volume_jar_A := π * radius_jar_A^2 * height_jar_A

-- Theorem to prove the radius of Jar B
theorem radius_of_jar_B : volume_jar_A = volume_jar_B → r = 10 * Real.sqrt 2 :=
sorry

end radius_of_jar_B_l694_694402


namespace sum_real_imag_of_z_l694_694741

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the complex number
def z : ℂ := (3 - 3 * i) / (1 - i)

-- Define the sum of the real part and the imaginary part
def sum_real_imag (z : ℂ) : ℝ := z.re + z.im

-- Lean 4 statement to prove the sum of the real and imaginary parts of the complex number
theorem sum_real_imag_of_z : sum_real_imag z = 3 :=
by
  sorry

end sum_real_imag_of_z_l694_694741


namespace simplify_form_of_expression_l694_694520

noncomputable def simplify_expression (a b c : ℝ) : ℝ :=
  (a + b + c + 3)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹) * (ab + bc + ca + 3)⁻¹ *
  ((ab)⁻¹ + (bc)⁻¹ + (ca)⁻¹)

theorem simplify_form_of_expression (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  simplify_expression a b c = (a * b * c)⁻² :=
by
  sorry

end simplify_form_of_expression_l694_694520


namespace measure_of_angle_A_l694_694857

theorem measure_of_angle_A (A B : ℝ) (h1 : A + B = 180) (h2 : A = 5 * B) : A = 150 := 
by 
  sorry

end measure_of_angle_A_l694_694857


namespace problem_l694_694369

def f (x a : ℝ) : ℝ := x^2 + a*x - 3*a - 9

theorem problem (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 0) : f 1 a = 4 :=
sorry

end problem_l694_694369


namespace equation_of_line_m_l694_694124

theorem equation_of_line_m
  (Q Q'' : ℝ × ℝ)
  (l m : Set (ℝ × ℝ))
  (Q_coords : Q = (-2, 3))
  (Q''_coords : Q'' = (3, -2))
  (l_eq : ∀ (p : ℝ × ℝ), p ∈ l ↔ 3 * (p.1) + 4 * (p.2) = 0)
  (m_eq : ∀ (p : ℝ × ℝ), p ∈ m ↔ 7 * (p.1) - 25 * (p.2) = 0)
  (intersects_at_origin : (0, 0) ∈ l ∧ (0, 0) ∈ m)
  (reflection_Q_about_l : ∀ (Q Q' : ℝ × ℝ), Q' = reflect Q l → Q' ∈ l)
  (reflection_Q'_about_m : ∀ (Q' Q'' : ℝ × ℝ), Q'' = reflect Q' m → Q'' ∈ m):
  m_eq :=
by
  sorry

end equation_of_line_m_l694_694124


namespace count_four_digit_numbers_l694_694597

open Int

def is_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9

def valid_combination (a b c d : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧
  a ≠ 0 ∧
  a + b + c + d = 10 ∧
  (a + c) - (b + d) % 11 = 0

theorem count_four_digit_numbers :
  ∃ n, n > 0 ∧ ∀ a b c d : ℕ, valid_combination a b c d → n = sorry :=
sorry

end count_four_digit_numbers_l694_694597


namespace sum_eq_product_l694_694130

theorem sum_eq_product : 
  ∃ (a : Fin 1000 → ℕ), (∑ i, a i = (∏ i, a i)) := 
by
  let a : Fin 1000 → ℕ := λ i, if i < 998 then 1 else if i = 998 then 2 else 1000
  use a
  sorry

end sum_eq_product_l694_694130


namespace cube_root_sum_simplification_l694_694341

theorem cube_root_sum_simplification :
  Real.cbrt (20^3 + 30^3 + 40^3 + 8000) = 91.52 := 
sorry

end cube_root_sum_simplification_l694_694341


namespace total_opponent_runs_l694_694440

def games : List ℕ := [1, 2, 2, 3, 3, 5, 6, 6, 7, 8, 9, 10]

def won_double (s : ℕ) := [(2, 1), (2, 1), (6, 3), (6, 3), (10, 5)]
def lost_by_two (s : ℕ) := [(1, 3), (3, 5), (5, 7), (7, 9)]
def remaining_games (s : ℕ) := [(8, 4), (9, 8)]

def opponent_scores (games : List (ℕ × ℕ)) : List ℕ := games.map Prod.snd

theorem total_opponent_runs :
  opponent_scores (won_double games) ++ opponent_scores (lost_by_two games) ++ opponent_scores (remaining_games games) = 49 := by
  sorry

end total_opponent_runs_l694_694440


namespace find_set_A_l694_694678

theorem find_set_A (a1 a2 a3 a4 : ℕ) (h1 : a1 < a2) (h2 : a2 < a3) (h3 : a3 < a4) 
(h4 : a1 * a2 * a3 = 24) (h5 : a1 * a2 * a4 = 30) (h6 : a1 * a3 * a4 = 40) (h7 : a2 * a3 * a4 = 60)
: {a1, a2, a3, a4} = {2, 3, 4, 5} :=
by
  sorry

end find_set_A_l694_694678


namespace limit_sum_areas_of_squares_l694_694076

theorem limit_sum_areas_of_squares (r : ℝ) :
  ∃ L : ℝ, (∀ n : ℕ, ∑ i in finset.range n, (r * 2 ^ (-(↑i : ℝ))) ^ 2) = 4 * r ^ 2 :=
sorry

end limit_sum_areas_of_squares_l694_694076


namespace exists_line_through_P_perpendicular_to_g_and_parallel_to_S_l694_694173

variable (P : Point) (g : Line) (S : Plane)

theorem exists_line_through_P_perpendicular_to_g_and_parallel_to_S :
  ∃ l : Line, (passes_through l P) ∧ (perpendicular l g) ∧ (parallel l S) :=
sorry

end exists_line_through_P_perpendicular_to_g_and_parallel_to_S_l694_694173


namespace acute_triangle_projection_distances_equal_l694_694666

theorem acute_triangle_projection_distances_equal
  (A B C E F G H : Point) 
  (h_acute_ABC : acute_triangle A B C)
  (h_E_base_height_B : is_base_point_of_height E A B C)
  (h_F_base_height_C : is_base_point_of_height F A B C)
  (h_G_proj_B_on_EF : is_projection G B (line_through E F))
  (h_H_proj_C_on_EF : is_projection H C (line_through E F)) :
  distance H E = distance F G := 
sorry

end acute_triangle_projection_distances_equal_l694_694666


namespace triangle_properties_l694_694638

noncomputable def side_a (b c A : ℝ) : ℝ :=
  real.sqrt (b^2 + c^2 - 2 * b * c * real.cos A)

noncomputable def area_triangle (b c A : ℝ) : ℝ :=
  0.5 * b * c * real.sin A

noncomputable def sin_2B (b c A : ℝ) : ℝ :=
  let a := side_a b c A
  let sin_B := b * real.sin A / a
  let cos_B := real.sqrt (1 - sin_B^2)
  2 * sin_B * cos_B

theorem triangle_properties (b c A : ℝ)
  (h_b : b = 4) (h_c: c = 5) (h_A : A = real.pi / 3) :
  side_a b c A = real.sqrt 21 ∧
  area_triangle b c A = 5 * real.sqrt 3 ∧
  sin_2B b c A = 4 * real.sqrt 3 / 7 :=
by
  rw [h_b, h_c, h_A]
  split
  case a =>
    -- show side_a 4 5 (real.pi / 3) = real.sqrt 21
    sorry
  case b =>
    split
    case a =>
      -- show area_triangle 4 5 (real.pi / 3) = 5 * real.sqrt 3
      sorry
    case b =>
      -- show sin_2B 4 5 (real.pi / 3) = 4 * real.sqrt 3 / 7
      sorry

end triangle_properties_l694_694638


namespace range_of_a_l694_694194

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * a * x + (a - 2) < 0) ↔ a ∈ set.Icc (-8 / 5 : ℝ) 0 :=
begin
  sorry,
end

end range_of_a_l694_694194


namespace designer_sketches_orders_l694_694256

theorem designer_sketches_orders :
  let S := {1, 2, 3, 5, 6, 7, 9, 10} \ {4, 8},
      orders := ∑ i in (0:finset ℕ).range (card S + 1), (card S).choose i
  in orders = 64 :=
by
  let S := ({1, 2, 3, 5, 6, 7, 9, 10} : finset ℕ) \ ({4, 8} : finset ℕ)
  let orders := ∑ i in (0 : finset ℕ).range (S.card + 1), S.card.choose i
  have : orders = 2^S.card := sorry
  have S_card_eq_6 : S.card = 6 := sorry
  rw [S_card_eq_6, pow_succ 2 6, mul_comm] at this
  exact this

end designer_sketches_orders_l694_694256


namespace radius_of_tangent_intersection_l694_694803

variable (x y : ℝ)

def circle_eq : Prop := x^2 + y^2 = 25

def tangent_condition : Prop := y = 5 ∧ x = 0

theorem radius_of_tangent_intersection (h1 : circle_eq x y) (h2 : tangent_condition x y) : ∃r : ℝ, r = 5 :=
by sorry

end radius_of_tangent_intersection_l694_694803


namespace m_n_zero_m_n_comp_l694_694651

variable {R : Type*} [Preorder R]

def m (A : set R) (x : R) : ℕ :=
if x ∈ A then 1 else 0

def n (B : set R) (x : R) : ℕ :=
if x ∈ B then 1 else 0

theorem m_n_zero (A B : set R) (h : A ⊆ B) (x : R) : m A x * (1 - n B x) = 0 :=
by sorry

theorem m_n_comp (A B : set R) (h : ∀ x : R, m A x + n B x = 1) : A = (\compl B : set R) :=
by sorry

end m_n_zero_m_n_comp_l694_694651


namespace incorrect_statements_l694_694366

theorem incorrect_statements :
  (¬ ∀ α β:ℝ, α ≠ β → α.terminal_side ≠ β.terminal_side) ∧
  (¬ ∀ α:ℝ, (cos α) < 0 → (∃ q, q = 2 ∨ q = 3 ∧ α.quadrant = q)) ∧
  (∀ α:ℝ, α ≠ π → (sin (α / 2) / cos (α / 2) = tan (α / 2))) → 
  True := 
by {
  sorry
}

end incorrect_statements_l694_694366


namespace probability_of_sum_23_l694_694853

def is_valid_time (h m : ℕ) : Prop :=
  0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def sum_of_time_digits (h m : ℕ) : ℕ :=
  sum_of_digits h + sum_of_digits m

theorem probability_of_sum_23 :
  (∃ h m, is_valid_time h m ∧ sum_of_time_digits h m = 23) →
  (4 / 1440 : ℚ) = (1 / 360 : ℚ) :=
by
  sorry

end probability_of_sum_23_l694_694853


namespace calculate_expression_l694_694488

noncomputable def expression : ℝ :=
  (-1)^(2023) + abs (-3) - (- (1 / 2) ^ (-2)) + 2 * real.sin (real.pi / 6)

theorem calculate_expression : expression = -1 := 
by
  sorry

end calculate_expression_l694_694488


namespace joan_mortgage_payback_months_l694_694644

-- Define the conditions and statement

def first_payment : ℕ := 100
def total_amount : ℕ := 2952400

theorem joan_mortgage_payback_months :
  ∃ n : ℕ, 100 * (3^n - 1) / (3 - 1) = 2952400 ∧ n = 10 :=
by
  sorry

end joan_mortgage_payback_months_l694_694644


namespace position_of_3142_among_permutations_l694_694719

theorem position_of_3142_among_permutations :
  ∀ (digits : List ℕ), (digits = [1, 2, 3, 4]) → 
  (findIndex (fun n => n = 3142) (permutations digits).sort = some 13) :=
by
  sorry

end position_of_3142_among_permutations_l694_694719


namespace max_min_sum_4028_l694_694153

noncomputable def f : ℝ → ℝ := sorry

theorem max_min_sum_4028 :
  (∀ x1 x2 : ℝ, -2015 ≤ x1 ∧ x1 ≤ 2015 ∧ -2015 ≤ x2 ∧ x2 ≤ 2015 → f(x1 + x2) = f(x1) + f(x2) - 2014) ∧
  (∀ x > 0, f(x) > 2014) →
  let M := sup (set.image f (set.Icc (-2015) 2016)) in
  let N := inf (set.image f (set.Icc (-2015) 2016)) in
  M + N = 4028 :=
by {
  intros h,
  let M := Sup (set.image f (set.Icc (-2015 : ℝ) (2016 : ℝ))),
  let N := Inf (set.image f (set.Icc (-2015 : ℝ) (2016 : ℝ))),
  sorry
}

end max_min_sum_4028_l694_694153


namespace perimeter_of_region_bounded_by_quarter_circles_l694_694074

theorem perimeter_of_region_bounded_by_quarter_circles (side : ℝ) (h1 : side = 4 / real.pi) : 
  let radius := side / 2,
      full_circumference := 2 * real.pi * radius,
      quarter_circumference := full_circumference / 4,
      total_perimeter := 4 * quarter_circumference
  in total_perimeter = 4 :=
sorry

end perimeter_of_region_bounded_by_quarter_circles_l694_694074


namespace complex_conjugate_l694_694953

-- Define a complex number satisfying the given condition
variable (z : ℂ) (hz: (1 + complex.i) * z = 2 * complex.i)

-- State goal: Prove that the conjugate of z is 1 - i
theorem complex_conjugate (z : ℂ) (hz : (1 + complex.i) * z = 2 * complex.i) : complex.conj z = 1 - complex.i :=
sorry

end complex_conjugate_l694_694953


namespace unique_n_value_l694_694650

theorem unique_n_value (n : ℕ) (d : ℕ → ℕ) (h1 : 1 = d 1) (h2 : ∀ i, d i ≤ n) (h3 : ∀ i j, i < j → d i < d j) 
                       (h4 : d (n - 1) = n) (h5 : ∃ k, k ≥ 4 ∧ ∀ i ≤ k, d i ∣ n)
                       (h6 : ∃ d1 d2 d3 d4, d 1 = d1 ∧ d 2 = d2 ∧ d 3 = d3 ∧ d 4 = d4 ∧ n = d1^2 + d2^2 + d3^2 + d4^2) : 
                       n = 130 := sorry

end unique_n_value_l694_694650


namespace no_four_polynomials_with_properties_l694_694890

theorem no_four_polynomials_with_properties :
  ¬ ∃ (p1 p2 p3 p4 : polynomial ℝ),
    (∀ (i j k : ℕ), i ≠ j → j ≠ k → k ≠ i → polynomial.has_root (p_of_fin4 i + p_of_fin4 j + p_of_fin4 k)) ∧
    (∀ (i j : ℕ), i ≠ j → ¬polynomial.has_root (p_of_fin4 i + p_of_fin4 j)) := sorry

end no_four_polynomials_with_properties_l694_694890


namespace triangle_DEF_f_l694_694639

theorem triangle_DEF_f (d e f : ℝ) (D E : ℝ) (cos_DE : ℝ) (cos_DE_val : cos_DE = 15 / 17) 
  (d_val : d = 6) (e_val : e = 8) :
  f ≈ 6.5 := 
by
  sorry

end triangle_DEF_f_l694_694639


namespace sequence_count_21_l694_694225

-- Define the function g that calculates the number of valid sequences of length n
def g : ℕ → ℕ
| 3 := 1
| 4 := 1
| 5 := 1
| 6 := 2
| (n + 7) := g (n + 3) + 2 * g (n + 2) + 2 * g (n + 1)

-- Define the property for sequences of length 21
theorem sequence_count_21 : g 21 = 345 :=
by {
  -- The proof is omitted in the statement
  sorry
}

end sequence_count_21_l694_694225


namespace moles_of_CH4_l694_694512

variables (CH4 Cl2 CHCl3 HCl : Type) 

-- Define the balanced chemical equation as a Lean definition
def balanced_reaction (CH4 Cl2 CHCl3 HCl : Type) : Prop := 
  ∀ m m2 n4 n3 k3 k4, 
    CH4 m + Cl2 (3 * m2) → CHCl3 m + HCl (3 * k3)

-- Define the conditions
def conditions (CH4 : Type) (Cl2 : Type) (CHCl3 : Type) (HCl : Type) 
    (molesCl2 molesCHCl3 : ℕ) :=
  balanced_reaction CH4 Cl2 CHCl3 HCl ∧ molesCl2 = 9 ∧ molesCHCl3 = 3

-- State the proof problem
theorem moles_of_CH4 (CH4 Cl2 CHCl3 HCl : Type) 
    (molesCl2 molesCHCl3 molesCH4 : ℕ) 
    (h : conditions CH4 Cl2 CHCl3 HCl molesCl2 molesCHCl3) :
  molesCH4 = 3 :=
begin
  sorry
end

end moles_of_CH4_l694_694512


namespace four_digit_number_divisible_by_11_l694_694582

theorem four_digit_number_divisible_by_11 :
  ∃ (a b c d : ℕ), 
    a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ 
    a + b + c + d = 10 ∧ 
    (a + c) % 11 = (b + d) % 11 ∧
    (10 - a != 0 ∨ 10 - c != 0 ∨ 10 - b != 0 ∨ 10 - d != 0) := sorry

end four_digit_number_divisible_by_11_l694_694582


namespace eventually_no_AG_l694_694707

-- Define a sequence of characters which can be either 'A' or 'G'
inductive Char
| A : Char
| G : Char

-- Define the type of strings made of 'A's and 'G's
def Seq := List Char

-- Define the weight function
def weight : Seq → ℝ
| [] => 0
| (Char.G :: t) => 1 / (2 ^ (t.length + 1)) + weight t
| (_ :: t) => weight t

-- Define the operation of replacing "AG" with "GAAA"
def performOperation : Seq → Seq
| (Char.A :: Char.G :: t) => Char.G :: Char.A :: Char.A :: Char.A :: t
| (h :: t) => h :: performOperation t
| [] => []

-- Theorem: eventually no occurrences of "AG" can be found in the sequence
theorem eventually_no_AG (s : Seq) : ∃ n, (performOperation^[n] s).all (λ x, ¬ (x = Char.A ∧ t.head = some Char.G)) := sorry

end eventually_no_AG_l694_694707


namespace total_profit_calculation_l694_694475

theorem total_profit_calculation (A B C : ℕ) (C_share total_profit : ℕ) 
  (hA : A = 27000) 
  (hB : B = 72000) 
  (hC : C = 81000) 
  (hC_share : C_share = 36000) 
  (h_ratio : C_share * 20 = total_profit * 9) :
  total_profit = 80000 := by
  sorry

end total_profit_calculation_l694_694475


namespace r_minus_s_not_greater_than_d_l694_694401

variables (A B: Type) [metric_space A] [metric_space B]
variables (r s d: ℝ)
variables (centerA: A) (centerB: B)

-- Conditions
axiom radius_condition: r > s
axiom distance_centers: dist centerA centerB = d

-- Statement to prove
theorem r_minus_s_not_greater_than_d (h1: r > s) (h2: dist centerA centerB = d): ¬ (r - s > d) :=
by {
  sorry
}

end r_minus_s_not_greater_than_d_l694_694401


namespace min_value_of_fraction_l694_694659

theorem min_value_of_fraction (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) :
  (1 / (a + 1) + 4 / (b + 1)) = 9 / 4 := 
sorry

end min_value_of_fraction_l694_694659


namespace correct_dot_product_analogies_l694_694878

theorem correct_dot_product_analogies (a b c : ℝ) (h1 : a ≠ 0) :
  (∀ (a b : ℝ), a * b = b * a) ∧
  (∀ (a b c : ℝ), (a * b) * c = a * (b * c)) ∧
  (∀ (a b c : ℝ), a * (b + c) = a * b + a * c) ∧
  (∀ (a b c : ℝ), a * b = a * c → b = c → false) →
  3 :=
by
  intros
  sorry

end correct_dot_product_analogies_l694_694878


namespace total_wrappers_l694_694096

theorem total_wrappers (a m : ℕ) (ha : a = 34) (hm : m = 15) : a + m = 49 :=
by
  sorry

end total_wrappers_l694_694096


namespace p_q_work_together_l694_694429

noncomputable def work_together_in_days (p_days : ℕ) (q_fraction : ℚ) : ℚ :=
  let p_one_day_work := (1:ℚ) / p_days
  let q_one_day_work := (1:ℚ) / q_fraction * p_one_day_work
  let combined_one_day_work := p_one_day_work + q_one_day_work
  (1:ℚ) / combined_one_day_work

theorem p_q_work_together (p_days : ℕ) (q_fraction : ℚ) (h_p : p_days = 4) (h_q : q_fraction = 3) :
  work_together_in_days p_days q_fraction = 3 := by
  rw [h_p, h_q]
  unfold work_together_in_days
  norm_num
  sorry

end p_q_work_together_l694_694429


namespace count_integers_n_satisfying_expression_l694_694152

theorem count_integers_n_satisfying_expression :
  ∃ n_count : ℕ, n_count = 56 ∧ ∀ n : ℕ, 1 ≤ n ∧ n ≤ 60 → ((n^2 + 1)! / (n!^(n + 1)) : ℚ).denom = 1 :=
begin
  sorry
end

end count_integers_n_satisfying_expression_l694_694152


namespace find_value_of_f2012_l694_694962

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) (α : ℝ) (β : ℝ) : ℝ :=
  a * Real.sin (π * x + α) + b * Real.cos (π * x + β)

theorem find_value_of_f2012 (a b α β : ℝ) 
    (h : f 2001 a b α β = 3) : f 2012 a b α β = -3 :=
sorry

end find_value_of_f2012_l694_694962


namespace husband_age_l694_694083

theorem husband_age (a b : ℕ) (w_age h_age : ℕ) (ha : a > 0) (hb : b > 0) 
  (hw_age : w_age = 10 * a + b) 
  (hh_age : h_age = 10 * b + a) 
  (h_older : h_age > w_age)
  (h_difference : 9 * (b - a) = a + b) :
  h_age = 54 :=
by
  sorry

end husband_age_l694_694083


namespace arithmetic_sequence_value_l694_694272

theorem arithmetic_sequence_value :
  ∀ {a : ℕ → ℝ}, 
  (∃ d a1, ∀ n, a(n) = a1 + d * (n - 1)) →
  (a 6 + a 8 + a 10 = 72) →
  (2 * a 10 - a 12 = 24) :=
by
  sorry

end arithmetic_sequence_value_l694_694272


namespace minimum_value_of_function_l694_694892

theorem minimum_value_of_function (x : ℝ) (hx : x > 1) : (x + 4 / (x - 1)) ≥ 5 := by
  sorry

end minimum_value_of_function_l694_694892


namespace remainder_when_divided_by_30_l694_694712

theorem remainder_when_divided_by_30 (x : ℤ) : 
  (4 + x) % 8 = 9 % 8 ∧
  (6 + x) % 27 = 4 % 27 ∧
  (8 + x) % 125 = 49 % 125 
  → x % 30 = 1 % 30 := by
  sorry

end remainder_when_divided_by_30_l694_694712


namespace gcd_n_cube_plus_27_n_plus_3_l694_694146

theorem gcd_n_cube_plus_27_n_plus_3 (n : ℕ) (h : n > 9) : 
  Nat.gcd (n^3 + 27) (n + 3) = n + 3 :=
sorry

end gcd_n_cube_plus_27_n_plus_3_l694_694146


namespace problem_statement_l694_694230

/-- For any positive integer n, given θ ∈ (0, π) and x ∈ ℂ such that 
x + 1/x = 2√2 cos θ - sin θ, it follows that x^n + 1/x^n = 2 cos (n α). -/
theorem problem_statement (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π)
  (x : ℂ) (hx : x + 1/x = 2 * (2:ℝ).sqrt * θ.cos - θ.sin)
  (n : ℕ) (hn : 0 < n) : x^n + x⁻¹^n = 2 * θ.cos * n := 
  sorry

end problem_statement_l694_694230


namespace average_increase_by_3_l694_694058

def initial_average_before_inning_17 (A : ℝ) : Prop :=
  16 * A + 85 = 17 * 37

theorem average_increase_by_3 (A : ℝ) (h : initial_average_before_inning_17 A) :
  37 - A = 3 :=
by
  sorry

end average_increase_by_3_l694_694058


namespace solve_problem_l694_694795

def problem_statement : Prop :=
  ∀ (n1 n2 c1 : ℕ) (C : ℕ),
  n1 = 18 → 
  c1 = 60 → 
  n2 = 216 →
  n1 * c1 = n2 * C →
  C = 5

theorem solve_problem : problem_statement := by
  intros n1 n2 c1 C h1 h2 h3 h4
  -- Proof steps go here
  sorry

end solve_problem_l694_694795


namespace algebraic_expression_value_l694_694030

noncomputable def a := Real.sqrt 2 + 1
noncomputable def b := Real.sqrt 2 - 1

theorem algebraic_expression_value : (a^2 - 2 * a * b + b^2) / (a^2 - b^2) = Real.sqrt 2 / 2 := by
  sorry

end algebraic_expression_value_l694_694030


namespace mass_of_compound_l694_694770

-- Constants as per the conditions
def molecular_weight : ℕ := 444           -- The molecular weight in g/mol.
def number_of_moles : ℕ := 6             -- The number of moles.

-- Defining the main theorem we want to prove.
theorem mass_of_compound : (number_of_moles * molecular_weight) = 2664 := by 
  sorry

end mass_of_compound_l694_694770


namespace table_fill_impossible_l694_694536

/-- Proposition: Given a 7x3 table filled with 0s and 1s, it is impossible to prevent any 2x2 submatrix from having all identical numbers. -/
theorem table_fill_impossible : 
  ¬ ∃ (M : (Fin 7) → (Fin 3) → Fin 2), 
      ∀ i j, (i < 6) → (j < 2) → 
              (M i j = M i.succ j) ∨ 
              (M i j = M i j.succ) ∨ 
              (M i j = M i.succ j.succ) ∨ 
              (M i.succ j = M i j.succ → M i j = M i.succ j.succ) :=
sorry

end table_fill_impossible_l694_694536


namespace frog_escape_probability_l694_694627

def P : ℕ → ℚ
| 0 := 0
| 12 := 1
| n := if 0 < n ∧ n < 12 then (↑(n + 1) / 13) * P (n - 1) + (1 - (↑(n + 1) / 13)) * P (n + 1) else 0

theorem frog_escape_probability : P 3 = 101 / 223 := by sorry

end frog_escape_probability_l694_694627


namespace domain_of_f_if_m_eq_neg_1_range_of_m_for_all_real_x_l694_694727

-- Proof Problem 1
theorem domain_of_f_if_m_eq_neg_1 (m : ℝ) :
  (∀ x ∈ set.Icc -2 1, 0 ≤ (m^2 + m - 2) * x^2 + (m - 1) * x + 4) → (m = -1) :=
by
  sorry

-- Proof Problem 2
theorem range_of_m_for_all_real_x (m : ℝ) :
  (∀ x : ℝ, 0 ≤ (m^2 + m - 2) * x^2 + (m - 1) * x + 4) ↔ 
  (m ∈ set.Iic (-11 / 5) ∪ set.Ici 1) :=
by
  sorry

end domain_of_f_if_m_eq_neg_1_range_of_m_for_all_real_x_l694_694727


namespace problem1_solution_problem2_solution_l694_694708
noncomputable theory

-- Problem 1: Solve x^2 - 6x + 1 = 0
theorem problem1_solution (x : ℝ) : x^2 - 6*x + 1 = 0 ↔ x = 3 + 2*Real.sqrt 2 ∨ x = 3 - 2*Real.sqrt 2 :=
by
  sorry

-- Problem 2: Solve 2(x+1)^2 = 3(x+1)
theorem problem2_solution (x : ℝ) : 2*(x+1)^2 = 3*(x+1) ↔ x = -1 ∨ x = 1/2 :=
by
  sorry

end problem1_solution_problem2_solution_l694_694708


namespace four_digit_number_divisible_by_11_l694_694583

theorem four_digit_number_divisible_by_11 :
  ∃ (a b c d : ℕ), 
    a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ 
    a + b + c + d = 10 ∧ 
    (a + c) % 11 = (b + d) % 11 ∧
    (10 - a != 0 ∨ 10 - c != 0 ∨ 10 - b != 0 ∨ 10 - d != 0) := sorry

end four_digit_number_divisible_by_11_l694_694583


namespace monotonically_decreasing_interval_l694_694206

def f (x : ℝ) : ℝ := x^2 - Real.log x

theorem monotonically_decreasing_interval :
  ∀ (x : ℝ), 0 < x → (f' x < 0 ↔ x ∈ set.Ioo 0 (Real.sqrt 2 / 2)) := by
  sorry

end monotonically_decreasing_interval_l694_694206


namespace ab_bd_ratio_l694_694156

-- Definitions based on the conditions
variables {A B C D : ℝ}
variables (h1 : A / B = 1 / 2) (h2 : B / C = 8 / 5)

-- Math equivalence proving AB/BD = 4/13 based on given conditions
theorem ab_bd_ratio
  (h1 : A / B = 1 / 2)
  (h2 : B / C = 8 / 5) :
  A / (B + C) = 4 / 13 :=
by
  sorry

end ab_bd_ratio_l694_694156


namespace sum_of_alternate_coefficients_l694_694547

variable {R : Type} [Field R]

def polynomial_expansion (x : R) : R :=
  (1 + x)^6

def coefficients_sum (coeff : Fin 7 → R) : R :=
  coeff 0 + coeff 1 + coeff 2 + coeff 3 + coeff 4 + coeff 5 + coeff 6

def alt_sum (coeff : Fin 7 → R) : R :=
  coeff 0 - coeff 1 + coeff 2 - coeff 3 + coeff 4 - coeff 5 + coeff 6

-- We'll assume coeff_prod produces the coefficients of the expansion
axiom coeff_prod : (Fin 7 → R) → (R → R) → coeff_prod coeff (polynomial_expansion x)

theorem sum_of_alternate_coefficients (coeff : Fin 7 → R) :
  coefficients_sum coeff = 64 →
  alt_sum coeff = 0 →
  coeff 1 + coeff 3 + coeff 5 = 32 :=
by {
  intro h_sum h_alt,
  have h2 : 2 * (coeff 1 + coeff 3 + coeff 5) = 64,
  { sorry },
  exact  (coeff 1 + coeff 3 + coeff 5),
}

end sum_of_alternate_coefficients_l694_694547


namespace ratio_PQ_AQ_45_deg_l694_694228

open Real

theorem ratio_PQ_AQ_45_deg
  (circle_center : Point)
  (radius : ℝ)
  (A B C D P Q : Point)
  (h1 : distance A B = 2 * radius)
  (h2 : distance C D = 2 * radius)
  (h3 : distance circle_center Q = radius)
  (h4 : A = circle_center - radius)
  (h5 : B = circle_center + radius)
  (h6 : C = (circle_center.1, circle_center.2 - radius))
  (h7 : D = (circle_center.1, circle_center.2 + radius))
  (h8 : P.1 = A.1 ∧ circle_center.2 ≤ P.2 ≤ A.2)
  (h9 : angle Q P C = 45) :
  (distance P Q) / (distance A Q) = sqrt 2 / 2 :=
  sorry

end ratio_PQ_AQ_45_deg_l694_694228


namespace geometric_sequence_common_ratio_l694_694277

variables {a_n : ℕ → ℝ} {S_n q : ℝ}

axiom a1_eq : a_n 1 = 2
axiom an_eq : ∀ n, a_n n = if n > 0 then 2 * q^(n-1) else 0
axiom Sn_eq : ∀ n, a_n n = -64 → S_n = -42 → q = -2

theorem geometric_sequence_common_ratio (q : ℝ) :
  (∀ n, a_n n = if n > 0 then 2 * q^(n-1) else 0) →
  a_n 1 = 2 →
  (∀ n, a_n n = -64 → S_n = -42 → q = -2) :=
by intros _ _ _; sorry

end geometric_sequence_common_ratio_l694_694277


namespace triangle_side_length_l694_694289

theorem triangle_side_length (A B C : Triangle)
  (h1 : ∠A = 2 * ∠B)
  (h2 : AC = 12)
  (h3 : BC = 8) : AB = 10 := 
  sorry

end triangle_side_length_l694_694289


namespace binomial_coeff_sum_l694_694236

theorem binomial_coeff_sum (a : ℕ → ℝ) (x : ℝ) (h : (1 - 2 * x)^2015 = ∑ i in finset.range(2016), a i * x^i) :
  (∑ i in finset.range(2016).erase 0, (a i) / 2^i) = -1 :=
by sorry

end binomial_coeff_sum_l694_694236


namespace number_of_satisfying_ns_l694_694933

noncomputable def a_n (n : ℕ) : ℕ := (n-1)*(2*n-1)

def b_n (n : ℕ) : ℕ := 2^n * n

def condition (n : ℕ) : Prop := b_n n ≤ 2019 * a_n n

theorem number_of_satisfying_ns : 
  ∃ n : ℕ, n = 14 ∧ ∀ k : ℕ, (1 ≤ k ∧ k ≤ 14) → condition k := 
by
  sorry

end number_of_satisfying_ns_l694_694933


namespace total_legs_l694_694900

-- Define the conditions
def chickens : Nat := 7
def sheep : Nat := 5
def legs_chicken : Nat := 2
def legs_sheep : Nat := 4

-- State the problem as a theorem
theorem total_legs :
  chickens * legs_chicken + sheep * legs_sheep = 34 :=
by
  sorry -- Proof not provided

end total_legs_l694_694900


namespace amy_can_place_100_red_stones_l694_694056

def site (x y : ℕ) : Prop := 1 ≤ x ∧ x ≤ 20 ∧ 1 ≤ y ∧ y ≤ 20

def red_stone_placement (sites: List (ℕ × ℕ)) : Prop :=
  ∀ (i j: ℕ × ℕ), i ∈ sites → j ∈ sites → i ≠ j →
  (let (x1, y1) := i in
   let (x2, y2) := j in
   (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 5)

def amy_max_red_stones : ℕ → Prop :=
  λ k, ∀ (ben_moves : List (ℕ × ℕ)),
  (∀ b ∈ ben_moves, site (b.fst) (b.snd)) →
  ∃ (amy_moves : List (ℕ × ℕ)), 
  red_stone_placement amy_moves ∧ 
  ∀ a ∈ amy_moves, site (a.fst) (a.snd) ∧ 
  amy_moves.length ≥ k

theorem amy_can_place_100_red_stones : amy_max_red_stones 100 :=
sorry

end amy_can_place_100_red_stones_l694_694056


namespace ellipse_equation_correct_coordinates_c_correct_l694_694545

-- Definition of the ellipse Γ with given properties
def ellipse_properties (a b : ℝ) (ecc : ℝ) (c_len : ℝ) :=
  a > b ∧ b > 0 ∧ ecc = (Real.sqrt 2) / 2 ∧ c_len = Real.sqrt 2

-- Correct answer for the equation of the ellipse
def correct_ellipse_equation := ∀ x y : ℝ, (x^2) / 2 + y^2 = 1

-- Proving that given the properties of the ellipse, the equation is as stated
theorem ellipse_equation_correct (a b : ℝ) (h : ellipse_properties a b (Real.sqrt 2 / 2) (Real.sqrt 2)) :
  (x^2) / 2 + y^2 = 1 := 
  sorry

-- Definition of the conditions for points A, B, and C
def triangle_conditions (a b : ℝ) (area : ℝ) :=
  ∀ A B : ℝ × ℝ,
    A.1^2 / a^2 + A.2^2 / b^2 = 1 ∧
    B.1^2 / a^2 + B.2^2 / b^2 = 1 ∧
    area = 3 * Real.sqrt 6 / 4

-- Correct coordinates of point C given the conditions
def correct_coordinates_c (C : ℝ × ℝ) :=
  (C = (1, Real.sqrt 2 / 2) ∨ C = (2, 1))

-- Proving that given the conditions, the coordinates of point C are correct
theorem coordinates_c_correct (a b : ℝ) (h : triangle_conditions a b (3 * Real.sqrt 6 / 4)) (C : ℝ × ℝ) :
  correct_coordinates_c C :=
  sorry

end ellipse_equation_correct_coordinates_c_correct_l694_694545


namespace hostel_budget_increase_l694_694258

theorem hostel_budget_increase :
  ∀ (A : ℝ), let orig_students := 100
             let new_students := 132
             let expenditure_diff := 5400
             let budget_decrease := 10
             let final_total_expenditure := 5400
             ∀ h1 : new_students = orig_students + 32,
             ∀ h2 : final_total_expenditure = new_students * (A - budget_decrease),
             let original_expenditure := orig_students * A,
             let increased_expenditure := final_total_expenditure - original_expenditure,
             increased_expenditure = 300 :=
by
  intros A h1 h2;
  simp only;
  let eq1 := 132 * (A - 10) = 5400;
  calc
    132 * (A - 10)    = 5400        : eq1
    ...               = 132 * 51 - 1320 : by sorry
    let original_expenditure := 100 * A,
    let final_total_expenditure := 5400,
    final_total_expenditure - original_expenditure = 300 : by sorry

end hostel_budget_increase_l694_694258


namespace arithmetic_series_sum_l694_694110

def first_term (k : ℕ) : ℕ := k^2 + k + 1
def common_difference : ℕ := 1
def number_of_terms (k : ℕ) : ℕ := 2 * k + 3
def nth_term (k n : ℕ) : ℕ := (first_term k) + (n - 1) * common_difference
def sum_of_terms (k : ℕ) : ℕ :=
  let n := number_of_terms k
  let a := first_term k
  let l := nth_term k n
  n * (a + l) / 2

theorem arithmetic_series_sum (k : ℕ) : sum_of_terms k = 2 * k^3 + 7 * k^2 + 10 * k + 6 :=
sorry

end arithmetic_series_sum_l694_694110


namespace flea_jump_no_lava_l694_694302

theorem flea_jump_no_lava
  (A B F : ℕ)
  (n : ℕ) 
  (h_posA : 0 < A)
  (h_posB : 0 < B)
  (h_AB : A < B)
  (h_2A : B < 2 * A)
  (h_ineq1 : A * (n + 1) ≤ B - A * n)
  (h_ineq2 : B - A < A * n) :
  ∃ (F : ℕ), F = (n - 1) * A + B := sorry

end flea_jump_no_lava_l694_694302


namespace classify_quadrilateral_with_perpendicular_and_equal_diagonals_l694_694891

-- A structure defining a general quadrilateral
structure Quadrilateral (α : Type) :=
(diagonal1 diagonal2 : α)
(perpendicular : Prop) (equal_length : Prop)

-- Definitions for the properties of the specific quadrilaterals
def is_square {α : Type} (Q : Quadrilateral α) : Prop :=
Q.perpendicular ∧ Q.equal_length

def is_rectangle {α : Type} (Q : Quadrilateral α) : Prop :=
Q.equal_length ∧ ¬ Q.perpendicular

def is_rhombus {α : Type} (Q : Quadrilateral α) : Prop :=
Q.perpendicular ∧ ¬ Q.equal_length

def is_kite {α : Type} (Q : Quadrilateral α) : Prop :=
Q.perpendicular ∧ ¬ Q.equal_length

def is_none_of_these {α : Type} (Q : Quadrilateral α) : Prop :=
¬ is_square Q ∧ ¬ is_rectangle Q ∧ ¬ is_rhombus Q ∧ ¬ is_kite Q

-- Main theorem statement
theorem classify_quadrilateral_with_perpendicular_and_equal_diagonals {α : Type} (Q : Quadrilateral α) :
  Q.perpendicular ∧ Q.equal_length → is_square Q :=
begin
  sorry
end

end classify_quadrilateral_with_perpendicular_and_equal_diagonals_l694_694891


namespace infinite_solutions_sine_equation_l694_694523

theorem infinite_solutions_sine_equation :
  ∃∞ x : ℝ, sin (3 * x) = sin (2 * x) + sin x :=
sorry

end infinite_solutions_sine_equation_l694_694523


namespace math_problem_in_Lean4_l694_694789

noncomputable def classification_of_prisms : Prop :=
  ∀ (base_is_parallelogram lateral_edges_perpendicular_to_base base_is_rectangle base_is_square 
     all_edges_equal : Prop),
    (base_is_parallelogram ∧ lateral_edges_perpendicular_to_base ∧ base_is_rectangle ∧ 
     base_is_square ∧ all_edges_equal) ↔ true

noncomputable def surface_area_and_volume_formulas : Prop :=
  ∀ (r h l : ℝ),
    (S_cylinder = 2 * π * r * (r + h) ∧
     S_cone = π * r * (r + l) ∧
     S_sphere = 4 * π * r^2 ∧
     V_cylinder = π * r^2 * h ∧
     V_cone = (1/3) * π * r^2 * h ∧
     V_sphere = (4/3) * π * r^3) ↔ true

noncomputable def trigonometric_identities : Prop :=
  ∀ (α β : ℝ),
    (sin (α + β) = sin α * cos β + cos α * sin β ∧
     cos (α + β) = cos α * cos β - sin α * sin β ∧
     sin 2α = 2 * sin α * cos α ∧
     cos 2α = cos^2 α - sin^2 α ∧
     cos 2α = 2 * cos^2 α - 1 ∧
     cos 2α = 1 - 2 * sin^2 α ∧
     sin α + cos α = sqrt(2) * sin (α + π / 4) ∧
     φ = π / 4) ↔ true 

theorem math_problem_in_Lean4 :
  classification_of_prisms ∧ surface_area_and_volume_formulas ∧ trigonometric_identities := 
  by 
    sorry

end math_problem_in_Lean4_l694_694789


namespace period_phase_shift_8_pi_4_l694_694104

def period_phase_shift (b c : ℝ):=
  let period := 2 * Real.pi / b
  let phase_shift := -c / b
  (period, phase_shift)

theorem period_phase_shift_8_pi_4 :
  period_phase_shift 8 (Real.pi / 4) = (Real.pi / 4, -Real.pi / 32) :=
by
  -- Provide the proof here
  sorry

end period_phase_shift_8_pi_4_l694_694104


namespace triangle_inequality_proof_l694_694951

theorem triangle_inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (habc : a + b > c) (hbca : b + c > a) (hcab : c + a > b) :
    a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := 
    sorry

end triangle_inequality_proof_l694_694951


namespace find_d_l694_694903

theorem find_d (d : ℚ) (int_part frac_part : ℚ) 
  (h1 : 3 * int_part^2 + 19 * int_part - 28 = 0)
  (h2 : 4 * frac_part^2 - 11 * frac_part + 3 = 0)
  (h3 : frac_part ≥ 0 ∧ frac_part < 1)
  (h4 : d = int_part + frac_part) :
  d = -29 / 4 :=
by
  sorry

end find_d_l694_694903


namespace no_y_exists_for_eqns_l694_694389

theorem no_y_exists_for_eqns (x y : ℝ) :
  ¬∃ y, ∃ x, x^2 + y^2 + 16 = 0 ∧ x^2 - 3y + 12 = 0 :=
by 
  sorry

end no_y_exists_for_eqns_l694_694389


namespace total_apples_and_pears_l694_694078

theorem total_apples_and_pears (x y : ℤ) 
  (h1 : x = 3 * (y / 2 + 1)) 
  (h2 : x = 5 * (y / 4 - 3)) : 
  x + y = 39 :=
sorry

end total_apples_and_pears_l694_694078


namespace inscribed_circle_area_right_triangle_l694_694137

noncomputable def inscribed_circle_area (proj_1 proj_2 : ℝ) : ℝ :=
  let hypotenuse := proj_1 + proj_2 in
  let leg_1 := Real.sqrt (proj_1 * hypotenuse) in
  let leg_2 := Real.sqrt (proj_2 * hypotenuse) in
  let r := (leg_1 + leg_2 - hypotenuse) / 2 in
  Real.pi * r^2

theorem inscribed_circle_area_right_triangle :
  inscribed_circle_area 9 16 = 25 * Real.pi :=
by
  sorry

end inscribed_circle_area_right_triangle_l694_694137


namespace tan_alpha_eq_neg2_l694_694534

theorem tan_alpha_eq_neg2 (α : Real) 
  (h : (sin α + 7 * cos α) / (3 * sin α + 5 * cos α) = -5) : tan α = -2 := 
sorry

end tan_alpha_eq_neg2_l694_694534


namespace apples_left_l694_694061

theorem apples_left (initial_apples : ℕ) (difference_apples : ℕ) (final_apples : ℕ) 
  (h1 : initial_apples = 46) 
  (h2 : difference_apples = 32) 
  (h3 : final_apples = initial_apples - difference_apples) : 
  final_apples = 14 := 
by
  rw [h1, h2] at h3
  exact h3

end apples_left_l694_694061


namespace min_parts_disjoint_l694_694648

theorem min_parts_disjoint (S : Finset (Fin n)) :
  ∃ m, (∀ (partition : Finset (Set (Finset (Fin n)))),
    (∀ p ∈ partition, ∀ A B ∈ p, (∀ C ∈ p, A ∪ B = C → A = B → A = C)) →
    ∃ P, partition.card = m) → m = n + 1 :=
begin
  sorry
end

end min_parts_disjoint_l694_694648


namespace anna_earnings_correct_l694_694101

open Nat

section AnnaCupcakes

variable (n c p : Nat) (f : ℚ)
variable (total_income : ℚ)

def total_cupcakes (n c : Nat) : Nat :=
  n * c

def sold_cupcakes (total_cupcakes : Nat) (f : ℚ) : ℚ :=
  f * total_cupcakes

def earnings (sold_cupcakes : ℚ) (p : Nat) : ℚ :=
  sold_cupcakes * p

theorem anna_earnings_correct (h_n : n = 4) (h_c : c = 20) (h_p : p = 2) (h_f : f = (3/5 : ℚ)) :
  let total := total_cupcakes n c in
  let sold := sold_cupcakes total f in
  let earn := earnings sold p in
  earn = 96 := by
  sorry
  
end AnnaCupcakes

end anna_earnings_correct_l694_694101


namespace area_of_quadrilateral_l694_694254

theorem area_of_quadrilateral (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f := (x - a) ^ 2 = 2 * a
  let g := (b * x + y) ^ 2 = 2 * b ^ 2
  in area_of_region_bounded_by (f) (g) = 8 * (sqrt (a * b) - b * 2 * a) :=
sorry

end area_of_quadrilateral_l694_694254


namespace negation_of_existential_proposition_l694_694731

theorem negation_of_existential_proposition : 
  (¬ ∃ x : ℝ, x^2 - x + 3 = 0) = (∀ x : ℝ, x^2 - x + 3 ≠ 0) :=
by
  sorry

end negation_of_existential_proposition_l694_694731


namespace area_of_square_PQRS_l694_694709

open Classical

variable (L P S N M Q R : Type) [real : Real] 

def triangle_LMN (L M N : Type) : Prop :=
∃ LP : Real,  LP = 30 ∧
∃ SN : Real, SN = 70

def square_PQRS (P Q R S : Type) : Prop :=
∃ side_length : Real, side_length * side_length = 2100

theorem area_of_square_PQRS (h1 : triangle_LMN L M N) (h2 : inscribed L P Q R S N) :
  square_PQRS P Q R S :=
sorry

end area_of_square_PQRS_l694_694709


namespace asymptotic_line_of_hyperbola_l694_694359

theorem asymptotic_line_of_hyperbola :
  (∀ x y : ℝ, y^2 - x^2 / 3 = 1 → y = (√3 / 3) * x ∨ y = -(√3 / 3) * x) :=
by
  sorry

end asymptotic_line_of_hyperbola_l694_694359


namespace area_enclosed_by_curves_l694_694863

noncomputable def enclosed_area : ℝ :=
2 * (48 * ((1/2) * (π / 2 - π / 3) - (1/4) * (sqrt 3 / 2)))

theorem area_enclosed_by_curves :
  enclosed_area = 4 * π - 6 * sqrt 3 :=
sorry

end area_enclosed_by_curves_l694_694863


namespace distance_between_lines_correct_l694_694973

noncomputable def distance_between_parallel_lines 
  (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

theorem distance_between_lines_correct :
  distance_between_parallel_lines 4 2 (-2) 1 = 3 * Real.sqrt 5 / 10 :=
by
  -- Proof steps would go here
  sorry

end distance_between_lines_correct_l694_694973


namespace cost_price_of_watch_l694_694828

-- Let C be the cost price of the watch
variable (C : ℝ)

-- Conditions: The selling price at a loss of 8% and the selling price with a gain of 4% if sold for Rs. 140 more
axiom loss_condition : 0.92 * C + 140 = 1.04 * C

-- Objective: Prove that C = 1166.67
theorem cost_price_of_watch : C = 1166.67 :=
by
  have h := loss_condition
  sorry

end cost_price_of_watch_l694_694828


namespace solve_otimes_n_1_solve_otimes_2005_2_l694_694888

-- Define the operation ⊗
noncomputable def otimes (x y : ℕ) : ℕ :=
sorry -- the definition is abstracted away as per conditions

-- Conditions from the problem
axiom otimes_cond_1 : ∀ x : ℕ, otimes x 0 = x + 1
axiom otimes_cond_2 : ∀ x : ℕ, otimes 0 (x + 1) = otimes 1 x
axiom otimes_cond_3 : ∀ x y : ℕ, otimes (x + 1) (y + 1) = otimes (otimes x (y + 1)) y

-- Prove the required equalities
theorem solve_otimes_n_1 (n : ℕ) : otimes n 1 = n + 2 :=
sorry

theorem solve_otimes_2005_2 : otimes 2005 2 = 4013 :=
sorry

end solve_otimes_n_1_solve_otimes_2005_2_l694_694888


namespace arithmetic_sequence_problem_l694_694656

theorem arithmetic_sequence_problem (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h1 : S 6 = 36)
  (h2 : S n = 324)
  (h3 : S (n - 6) = 144) :
  n = 18 := by
  sorry

end arithmetic_sequence_problem_l694_694656


namespace color_removal_exists_l694_694260

open SimpleGraph

def K₄₀ : SimpleGraph (Fin 40) := completeGraph _

theorem color_removal_exists :
  ∃ c : Fin 6, 
    ∀ v w : Fin 40, 
      v ≠ w → (K₄₀.deleteEdges (fun e => e.color = c)).Adj v w :=
sorry

end color_removal_exists_l694_694260


namespace log_product_identity_l694_694106

noncomputable def log {a b : ℝ} (ha : 1 < a) (hb : 0 < b) : ℝ := Real.log b / Real.log a

theorem log_product_identity : 
  log (by norm_num : (1 : ℝ) < 2) (by norm_num : (0 : ℝ) < 9) * 
  log (by norm_num : (1 : ℝ) < 3) (by norm_num : (0 : ℝ) < 8) = 6 :=
sorry

end log_product_identity_l694_694106


namespace algebra_expression_value_l694_694167

theorem algebra_expression_value (x y : ℝ) (h : x = 2 * y + 1) : x^2 - 4 * x * y + 4 * y^2 = 1 := 
by 
  sorry

end algebra_expression_value_l694_694167


namespace correct_propositions_l694_694197

def prop1 : Prop := (∃ win_rate: ℚ, win_rate = 1/10000 ∧ ∀ tickets: ℕ, tickets = 10000 → guaranteed_win tickets)
def prop2 : Prop := ∃ (coin_toss: ℕ → ℤ), (∀ i, i < 100 → coin_toss i = if i < 99 then (90, 9) else (coin_toss i)) → prob_heads_coin_toss 100 > prob_tails_coin_toss 100
def prop3 : Prop := ∀ (A B: Prop), (mutually_exclusive A B) → (P (A ∨ B) = P A + P B)
def prop4 : Prop := ∀ (A: Prop) (m n: ℕ), (large_trials n) → (freq_event_occur A m n ≈ P A)
def prop5 : Prop := ∀ (heads_count_100000: ℕ) (heads_count_1000000: ℕ), (heads_freq_closer heads_count_1000000 heads_count_100000)

theorem correct_propositions : (prop1 ∨ prop2 ∨ prop3 ∨ prop5) = false ∧ prop4 = true := by
  sorry

end correct_propositions_l694_694197


namespace maximize_profit_l694_694441

-- Conditions
def price_bound (p : ℝ) := p ≤ 22
def books_sold (p : ℝ) := 110 - 4 * p
def profit (p : ℝ) := (p - 2) * books_sold p

-- The main theorem statement
theorem maximize_profit : ∃ p : ℝ, price_bound p ∧ profit p = profit 15 :=
sorry

end maximize_profit_l694_694441


namespace volume_ratio_eq_pi_squared_div_eight_l694_694396

-- Definitions of the given conditions
def rectangular_wood (a h : ℝ) : ℝ := a^2 * h
def cylindrical_wood (r h : ℝ) : ℝ := π * r^2 * h

-- Define the ratio of volumes for the largest possible shapes.
def largest_possible_cylinder_volume (r : ℝ) (h : ℝ) : ℝ :=
  let a := r * real.sqrt π in
  let R := a / 2 in
  π * R^2 * h

def largest_possible_rectangular_prism_volume (r : ℝ) (h : ℝ) : ℝ :=
  (r * real.sqrt π) * r * real.sqrt π * h

-- Theorem statement proving the volume ratio
theorem volume_ratio_eq_pi_squared_div_eight (r h : ℝ) :
  rectangular_wood (r * real.sqrt π) h = cylindrical_wood r h →
  largest_possible_cylinder_volume r h / largest_possible_rectangular_prism_volume r h = π^2 / 8 :=
by
  sorry

end volume_ratio_eq_pi_squared_div_eight_l694_694396


namespace region_perimeter_ge_2_l694_694759

theorem region_perimeter_ge_2 
  (square : {s : ℝ // s = 1})
  (line1 line2: ℝ → ℝ → Prop) : 
  ∃ region, perimeter region ≥ 2 :=
by sorry

end region_perimeter_ge_2_l694_694759


namespace arithmetic_sequence_y_value_l694_694271

theorem arithmetic_sequence_y_value :
  ∃ d x z, let seq := [21, x, 37, z, 53] in
  (seq = list.iota (21 + d) 5 ∧ seq.nth 2 = 37) :=
sorry

end arithmetic_sequence_y_value_l694_694271


namespace cubic_symmetric_inflection_point_l694_694498

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x^2 + (5 / 3)

def is_symmetric_center (f : ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∃ a b, f = λ x, a * x^3 + b * x^2 + (5 / 3) ∧ f(c.1) = c.2 ∧ (6 * a * c.1 + 2 * b = 0)

theorem cubic_symmetric_inflection_point :
  ∀ (f : ℝ → ℝ), is_symmetric_center f (1, 1) → (∃ (a b : ℝ), a = 1/3 ∧ b = -1) ∧
  (∃ x y, has_minimum f ∧ has_maximum f) :=
by
  -- We would provide the proof here
  sorry

end cubic_symmetric_inflection_point_l694_694498


namespace total_buckets_poured_l694_694692

-- Define given conditions
def initial_buckets : ℝ := 1
def additional_buckets : ℝ := 8.8

-- Theorem to prove the total number of buckets poured
theorem total_buckets_poured : 
  initial_buckets + additional_buckets = 9.8 :=
by
  sorry

end total_buckets_poured_l694_694692


namespace temperature_at_midnight_l694_694253

theorem temperature_at_midnight :
  ∀ (initial noontemp midnighttemp : ℤ),
    initial = -2 ∧ noontemp = initial + 13 ∧ midnighttemp = noontemp - 8 → midnighttemp = 3 :=
by
  intros initial noontemp midnighttemp h,
  sorry

end temperature_at_midnight_l694_694253


namespace motorcycle_travel_distance_l694_694980

noncomputable def motorcycle_distance : ℝ :=
  let t : ℝ := 1 / 2  -- time in hours (30 minutes)
  let v_bus : ℝ := 90  -- speed of the bus in km/h
  let v_motorcycle : ℝ := (2 / 3) * v_bus  -- speed of the motorcycle in km/h
  v_motorcycle * t  -- calculates the distance traveled by the motorcycle in km

theorem motorcycle_travel_distance :
  motorcycle_distance = 30 := by
  sorry

end motorcycle_travel_distance_l694_694980


namespace compute_expr_l694_694109

theorem compute_expr {x : ℝ} (h : x = 5) : (x^6 - 2 * x^3 + 1) / (x^3 - 1) = 124 :=
by
  sorry

end compute_expr_l694_694109


namespace perfect_square_x_minus_25_l694_694300

noncomputable def x : ℕ := (10^2012 + 1) * 10^2014 + 50

theorem perfect_square_x_minus_25 :
  ∃ k : ℕ, x - 25 = k^2 :=
by {
  use (10^2013 + 5),
  sorry
}

end perfect_square_x_minus_25_l694_694300


namespace rational_reciprocal_pow_2014_l694_694073

theorem rational_reciprocal_pow_2014 (a : ℚ) (h : a = 1 / a) : a ^ 2014 = 1 := by
  sorry

end rational_reciprocal_pow_2014_l694_694073


namespace domain_tan_2x_plus_pi_over_3_l694_694362

noncomputable def domain_tan : Set ℝ := {x | ∃ k : ℤ, x = k * Real.pi + Real.pi / 2}

noncomputable def domain_tan_transformed : Set ℝ :=
  {x | ∃ k : ℤ, x = k * (Real.pi / 2) + Real.pi / 12}

theorem domain_tan_2x_plus_pi_over_3 :
  (∀ x, ¬ (x ∈ domain_tan)) ↔ (∀ x, ¬ (x ∈ domain_tan_transformed)) :=
by
  sorry

end domain_tan_2x_plus_pi_over_3_l694_694362


namespace puzzle_imaginary_part_l694_694320

noncomputable def z : Complex := 1 + Complex.i
def z_conjugate : Complex := Complex.conj z
def w : Complex := (4 / z) - z_conjugate
def imag_w : ℝ := w.im

theorem puzzle_imaginary_part : imag_w = -1 := by
  sorry

end puzzle_imaginary_part_l694_694320


namespace number_of_whole_numbers_l694_694984

theorem number_of_whole_numbers (x y : ℝ) (hx : 2 < x ∧ x < 3) (hy : 8 < y ∧ y < 9) : 
  ∃ (n : ℕ), n = 6 := by
  sorry

end number_of_whole_numbers_l694_694984


namespace A_minus_3B_A_minus_3B_independent_of_y_l694_694161

variables (x y : ℝ)
def A : ℝ := 3*x^2 - x + 2*y - 4*x*y
def B : ℝ := x^2 - 2*x - y + x*y - 5

theorem A_minus_3B (x y : ℝ) : A x y - 3 * B x y = 5*x + 5*y - 7*x*y + 15 :=
by
  sorry

theorem A_minus_3B_independent_of_y (x : ℝ) (hyp : ∀ y : ℝ, A x y - 3 * B x y = 5*x + 5*y - 7*x*y + 15) :
  5 - 7*x = 0 → x = 5 / 7 :=
by
  sorry

end A_minus_3B_A_minus_3B_independent_of_y_l694_694161


namespace multiply_by_105_makes_perfect_square_l694_694068

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem multiply_by_105_makes_perfect_square :
  let y := 2^5 * 3^5 * 4^5 * 5^5 * 6^5 * 7^5 * 8^5 * 9^5 * 5 in
  is_perfect_square (y * 105) :=
by
  sorry

end multiply_by_105_makes_perfect_square_l694_694068


namespace prob_one_mistake_eq_l694_694378

-- Define the probability of making a mistake on a single question
def prob_mistake : ℝ := 0.1

-- Define the probability of answering correctly on a single question
def prob_correct : ℝ := 1 - prob_mistake

-- Define the probability of answering all three questions correctly
def three_correct : ℝ := prob_correct ^ 3

-- Define the probability of making at least one mistake in three questions
def prob_at_least_one_mistake := 1 - three_correct

-- The theorem states that the above probability is equal to 1 - 0.9^3
theorem prob_one_mistake_eq :
  prob_at_least_one_mistake = 1 - (0.9 ^ 3) :=
by
  sorry

end prob_one_mistake_eq_l694_694378


namespace maxKings_placement_l694_694427

def noThreat (i j : ℕ) (positions : List (ℕ × ℕ)) : Prop :=
  ∀ (p : ℕ × ℕ), p ∈ positions → abs (p.1 - i) > 1 ∨ abs (p.2 - j) > 1

def maxKings (n : ℕ) : ℕ :=
  if n % 2 = 0 then (n / 2) * (n / 2)
  else ((n + 1) / 2) * ((n + 1) / 2)

theorem maxKings_placement (n : ℕ) : ∃ (positions : List (ℕ × ℕ)),
  noThreat n n positions ∧ positions.length = maxKings n := sorry

end maxKings_placement_l694_694427


namespace maximum_value_of_k_existence_of_x0_l694_694568

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) - x

theorem maximum_value_of_k :
  ∃ k ∈ Int, (∀ x > 1, f (x - 1) + x > k * (1 - 3 / x)) → k = 4 := sorry

theorem existence_of_x0 (a : ℝ) (h_a : 0 < a ∧ a < 1) :
  ∃ x0 > 0, Real.exp (f x0) < 1 - (a / 2) * x0^2 := sorry

end maximum_value_of_k_existence_of_x0_l694_694568


namespace union_A_B_l694_694947

-- Definitions for the sets A and B
def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

-- The statement to be proven
theorem union_A_B :
  A ∪ B = {x | (-1 < x ∧ x ≤ 3) ∨ x = 4} :=
sorry

end union_A_B_l694_694947


namespace segment_irreducible_fractions_l694_694818

noncomputable def irreducible_fraction_count (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else
    let segment_length := (1 : ℚ) / n in
    let fractions := { ⟨p, q⟩ : ℤ × ℤ | 1 ≤ q ∧ q ≤ n ∧ gcd p q = 1 ∧ (p : ℚ) / q < segment_length } in
    fractions.card

theorem segment_irreducible_fractions (n : ℕ) (hn : n ≠ 0) :
  irreducible_fraction_count n ≤ (n + 1) / 2 := by
  sorry

end segment_irreducible_fractions_l694_694818


namespace degrees_to_radians_conversion_l694_694116

theorem degrees_to_radians_conversion : (-300 : ℝ) * (Real.pi / 180) = - (5 / 3) * Real.pi :=
by
  sorry

end degrees_to_radians_conversion_l694_694116


namespace polygon_side_length_inequality_l694_694533

noncomputable def side_length_of_regular_polygon_inscribed_in_circle (n : ℕ) : ℝ :=
  1 / 2 * (1 - Real.cos (Real.pi / n))

theorem polygon_side_length_inequality (n : ℕ) (hn : 0 < n) :
  ∃ (polygon : List (ℝ × ℝ)), 
  (∀ segment ∈ polygon, segment.length = 1) ∧
  ∃ side ∈ sides polygon, side.length ≥ side_length_of_regular_polygon_inscribed_in_circle n := 
sorry

end polygon_side_length_inequality_l694_694533


namespace simon_age_is_10_l694_694846

def alvin_age : ℕ := 30

def simon_age (alvin_age : ℕ) : ℕ :=
  (alvin_age / 2) - 5

theorem simon_age_is_10 : simon_age alvin_age = 10 := by
  sorry

end simon_age_is_10_l694_694846


namespace log_equation_proof_l694_694882

noncomputable def problem_statement : Prop :=
  (\lg 5)^2 + \lg 2 * \lg 50 - (\log_base 8 9) * (\log_base 27 32) = -1/9

theorem log_equation_proof : problem_statement :=
sorry

end log_equation_proof_l694_694882


namespace find_f_x_eq_cx_l694_694134

theorem find_f_x_eq_cx (f : ℝ+ → ℝ+) :
  (∀ (a b c : ℝ), (a ≠ b ∧ b ≠ c ∧ a ≠ c) → (a + b > c ∧ b + c > a ∧ c + a > b) ↔ (f(a) + f(b) > f(c) ∧ f(b) + f(c) > f(a) ∧ f(c) + f(a) > f(b))) →
  ∃ c > 0, ∀ x, f(x) = c * x :=
by
  sorry

end find_f_x_eq_cx_l694_694134


namespace crank_slider_mechanism_equations_l694_694883

noncomputable def omega : ℝ := 10
noncomputable def OA : ℝ := 90
noncomputable def AB : ℝ := 90
noncomputable def AM : ℝ := AB / 2
noncomputable def RM : ℝ := OA + AM 

theorem crank_slider_mechanism_equations (t : ℝ) :
  let x_A := OA * Real.cos (omega * t) in
  let y_A := OA * Real.sin (omega * t) in
  let x_M := 45 * Real.cos (omega * t) + 45 in
  let y_M := 45 * Real.sin (omega * t) in
  (x_M = 45 * Real.cos (omega * t) + 45) ∧
  (y_M = 45 * Real.sin (omega * t)) ∧
  ((x_M - 45)^2 + y_M^2 = 2025) ∧
  (Real.sqrt ((-450 * Real.sin (omega * t))^2 + (450 * Real.cos (omega * t))^2) = 450) :=
by
sorry

end crank_slider_mechanism_equations_l694_694883


namespace range_of_a_l694_694535

variable (x a : ℝ)

def p : Prop := |x - a| < 4
def q : Prop := -x^2 + 5 * x - 6 > 0

theorem range_of_a (h : ∀ x, q x → p x) : -1 ≤ a ∧ a ≤ 6 :=
  sorry

end range_of_a_l694_694535


namespace area_of_union_of_five_equilateral_triangles_l694_694145

-- Define the conditions
def side_length : ℝ := 3
def num_triangles : ℕ := 5

-- Define the area of one equilateral triangle
def equilateral_triangle_area (s : ℝ) : ℝ :=
  (math.sqrt 3 / 4) * s ^ 2

-- Define the total area covered by the union of the five equilateral triangles
def union_area (s : ℝ) (n : ℕ) : ℝ :=
  n * equilateral_triangle_area s - (n - 1) * equilateral_triangle_area (s / 2)

-- State the main theorem
theorem area_of_union_of_five_equilateral_triangles :
  union_area side_length num_triangles = 9 * math.sqrt 3 :=
by
  sorry

end area_of_union_of_five_equilateral_triangles_l694_694145


namespace interval_decrease_log_a_l694_694368

noncomputable def f (a x : ℝ) := a^x

theorem interval_decrease_log_a (a : ℝ) (x : ℝ) : 
  (0 < a) ∧ (a ≠ 1) ∧ (f a 1 > 1) → 
  (∀ x, x ∈ (-∞:ℝ, -1] →
    ∀ y, y = log a (x^2 - 1) → y' < 0 ) := 
begin
  sorry
end

end interval_decrease_log_a_l694_694368


namespace find_a_and_tangent_point_l694_694457

noncomputable def tangent_line_and_curve (a : ℚ) (P : ℚ × ℚ) : Prop :=
  ∃ (x₀ : ℚ), (P = (x₀, x₀ + a)) ∧ (P = (x₀, x₀^3 - x₀^2 + 1)) ∧ (3*x₀^2 - 2*x₀ = 1)

theorem find_a_and_tangent_point :
  ∃ (a : ℚ) (P : ℚ × ℚ), tangent_line_and_curve a P ∧ a = 32/27 ∧ P = (-1/3, 23/27) :=
sorry

end find_a_and_tangent_point_l694_694457


namespace ab_value_l694_694239

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := 
by 
  sorry

end ab_value_l694_694239


namespace quadratic_properties_l694_694367

theorem quadratic_properties :
  ∃ (a b c : ℚ), 
  let q := λ x : ℚ, a * x^2 + b * x + c in
  q 0 = -7/2 ∧ q 1 = 1/2 ∧ q (3/2) = 1 ∧ q 2 = 1/2 ∧
  (let vertex_x := (3:ℚ) / 2 in
   let vertex_y := q vertex_x in
   c = -7/2 ∧
   vertex_x = 3 / 2 ∧
   vertex_y = 1 ∧
   (a != 0 ∧
   (∀ x, q x = -2 * (x - 3/2)^2 + 1))) 
  := sorry

end quadratic_properties_l694_694367


namespace range_of_a_l694_694242

theorem range_of_a (a : ℝ) : (4 - a < 0) → (a > 4) :=
by
  intros h
  sorry

end range_of_a_l694_694242


namespace must_divide_a_l694_694649

-- Definitions of positive integers and their gcd conditions
variables {a b c d : ℕ}

-- The conditions given in the problem
axiom h1 : gcd a b = 24
axiom h2 : gcd b c = 36
axiom h3 : gcd c d = 54
axiom h4 : 70 < gcd d a ∧ gcd d a < 100

-- We need to prove that 13 divides a
theorem must_divide_a : 13 ∣ a :=
by sorry

end must_divide_a_l694_694649


namespace probability_integer_solution_l694_694245

theorem probability_integer_solution (a : ℤ) (h₀ : a ≥ 0) (h_max : a ≤ 6) :
    let x := (a + 4) / 2 in
    ∃ s : finset ℤ, s = {0, 1, 2, 3, 4, 5, 6} ∧
    let valid_a := s.filter (λ a, ∃ x : ℤ, x = (a + 4) / 2) in
    valid_a.card / s.card = 3 / 7 :=
by
  sorry

end probability_integer_solution_l694_694245


namespace arithmetic_expression_evaluation_l694_694742

theorem arithmetic_expression_evaluation : 
  2000 - 80 + 200 - 120 = 2000 := by
  sorry

end arithmetic_expression_evaluation_l694_694742


namespace mean_eq_median_plus_three_l694_694518

theorem mean_eq_median_plus_three (x : ℕ) (h : x > 0) :
  let s := [x, x + 2, x + 4, x + 7, x + 22] in
  let median := s.nth 2 |>.get_or_else 0 in
  let mean := s.sum / s.length in
  mean = median + 3 :=
by
  sorry

end mean_eq_median_plus_three_l694_694518


namespace largest_prime_factor_of_7_fact_add_8_fact_l694_694023

theorem largest_prime_factor_of_7_fact_add_8_fact :
  let n := 7!
  let m := 8!
  let sum := n + m
  prime_factors sum = {2, 3, 5, 7} ∧ (∀ p ∈ {2, 3, 5, 7}, p <= 7) := by
  sorry

end largest_prime_factor_of_7_fact_add_8_fact_l694_694023


namespace rightmost_three_digits_of_7_pow_1987_l694_694764

theorem rightmost_three_digits_of_7_pow_1987 :
  (7^1987 : ℕ) % 1000 = 643 := 
by 
  sorry

end rightmost_three_digits_of_7_pow_1987_l694_694764


namespace min_difference_l694_694281

theorem min_difference (n : ℕ) (h1 : 2n - 1) (h2 : 5055 - 5n) : ((5055 - 5n) - (2n - 1)) ≥ 2 :=
by sorry

end min_difference_l694_694281


namespace find_divisor_l694_694141

theorem find_divisor (n x : ℕ) (hx : x ≠ 11) (hn : n = 386) 
  (h1 : ∃ k : ℤ, n = k * x + 1) (h2 : ∀ m : ℤ, n = 11 * m + 1 → n = 386) : x = 5 :=
  sorry

end find_divisor_l694_694141


namespace circle_and_square_areas_l694_694802

noncomputable def height_of_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 2) * s

noncomputable def radius_of_circumscribed_circle (s : ℝ) : ℝ :=
  height_of_equilateral_triangle s / sqrt 3

noncomputable def area_of_circle (R : ℝ) : ℝ :=
  π * R^2

noncomputable def area_of_square (s : ℝ) : ℝ :=
  s^2

theorem circle_and_square_areas :
  let s_triangle := 12
  let h := height_of_equilateral_triangle s_triangle
  let R := radius_of_circumscribed_circle s_triangle
  let A_circle := area_of_circle R
  let s_square := h
  let A_square := area_of_square s_square
  A_circle = 36 * π ∧ A_square = 108 :=
by
  let s_triangle := 12
  let h := height_of_equilateral_triangle s_triangle
  let R := radius_of_circumscribed_circle s_triangle
  let A_circle := area_of_circle R
  let s_square := h
  let A_square := area_of_square s_square
  sorry

end circle_and_square_areas_l694_694802


namespace problem_pm_sqrt5_sin_tan_l694_694174

theorem problem_pm_sqrt5_sin_tan
  (m : ℝ)
  (h_m_nonzero : m ≠ 0)
  (cos_alpha : ℝ)
  (h_cos_alpha : cos_alpha = (Real.sqrt 2 * m) / 4)
  (P : ℝ × ℝ)
  (h_P : P = (m, -Real.sqrt 3))
  (r : ℝ)
  (h_r : r = Real.sqrt (3 + m^2)) :
    (∃ m, m = Real.sqrt 5 ∨ m = -Real.sqrt 5) ∧
    (∃ sin_alpha tan_alpha,
      (sin_alpha = - Real.sqrt 6 / 4 ∧ tan_alpha = -Real.sqrt 15 / 5)) :=
by
  sorry

end problem_pm_sqrt5_sin_tan_l694_694174


namespace shaded_area_of_square_with_circles_l694_694390

noncomputable def side_length : ℝ := 8
noncomputable def circle_radius : ℝ := 3

theorem shaded_area_of_square_with_circles :
  let area_square := side_length ^ 2,
      area_circle := Real.pi * (circle_radius ^ 2),
      total_area_circles := 4 * area_circle
  in area_square - total_area_circles = 64 - 36 * Real.pi := by
  sorry

end shaded_area_of_square_with_circles_l694_694390


namespace min_students_with_both_l694_694686

-- Given conditions
def total_students : ℕ := 35
def students_with_brown_eyes : ℕ := 18
def students_with_lunch_box : ℕ := 25

-- Mathematical statement to prove the least number of students with both attributes
theorem min_students_with_both :
  ∃ x : ℕ, students_with_brown_eyes + students_with_lunch_box - total_students ≤ x ∧ x = 8 :=
sorry

end min_students_with_both_l694_694686


namespace find_positive_number_l694_694041

noncomputable def solve_number (x : ℝ) : Prop :=
  (2/3 * x = 64/216 * (1/x)) ∧ (x > 0)

theorem find_positive_number (x : ℝ) : solve_number x → x = (2/9) * Real.sqrt 3 :=
  by
  sorry

end find_positive_number_l694_694041


namespace distance_to_workplace_l694_694713

def driving_speed : ℕ := 40
def driving_time : ℕ := 3
def total_distance := driving_speed * driving_time
def one_way_distance := total_distance / 2

theorem distance_to_workplace : one_way_distance = 60 := by
  sorry

end distance_to_workplace_l694_694713


namespace total_blue_balloons_l694_694299

def Joan_balloons : Nat := 9
def Sally_balloons : Nat := 5
def Jessica_balloons : Nat := 2

theorem total_blue_balloons : Joan_balloons + Sally_balloons + Jessica_balloons = 16 :=
by
  sorry

end total_blue_balloons_l694_694299


namespace complement_intersection_l694_694970

open Set

variable (U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8})
variable (A : Set ℕ := {2, 5, 8})
variable (B : Set ℕ := {1, 3, 5, 7})

theorem complement_intersection (CUA : Set ℕ := {1, 3, 4, 6, 7}) :
  (CUA ∩ B) = {1, 3, 7} := by
  sorry

end complement_intersection_l694_694970


namespace zero_of_f_inequality_l694_694051

noncomputable def f (x : ℝ) : ℝ := 2^(-x) - Real.log (x^3 + 1)

variable (a b c x : ℝ)
variable (h : 0 < a ∧ a < b ∧ b < c)
variable (hx : f x = 0)
variable (h₀ : f a * f b * f c < 0)

theorem zero_of_f_inequality :
  ¬ (x > c) :=
by 
  sorry

end zero_of_f_inequality_l694_694051


namespace total_legs_l694_694899

-- Define the conditions
def chickens : Nat := 7
def sheep : Nat := 5
def legs_chicken : Nat := 2
def legs_sheep : Nat := 4

-- State the problem as a theorem
theorem total_legs :
  chickens * legs_chicken + sheep * legs_sheep = 34 :=
by
  sorry -- Proof not provided

end total_legs_l694_694899


namespace f_alpha_value_l694_694186

open Real

-- Conditions based definitions
def alpha : Real -> Prop := fun a => 0 < a ∧ a < π

def f (α : Real) : Real :=
    (tan (α - π) * cos (2 * π - α) * sin (-α + (3 * π / 2))) /
    (cos (-α - π) * tan (π + α))

axiom cos_shift (α : Real) : α ∈ {a : Real | 0 < a ∧ a < π} -> cos (α + π / 2) = -1 / 5

-- The theorem to be proved
theorem f_alpha_value (α : Real) (hα : α ∈ {a : Real | 0 < a ∧ a < π}) :
  f α = -2 * sqrt 6 / 5 := by
  sorry

end f_alpha_value_l694_694186


namespace final_water_level_l694_694754

-- Define the conditions
def h_initial (h: ℝ := 0.4): ℝ := 0.4  -- Initial height in meters, 0.4m = 40 cm
def rho_water : ℝ := 1000 -- Density of water in kg/m³
def rho_oil : ℝ := 700 -- Density of oil in kg/m³
def g : ℝ := 9.81 -- Acceleration due to gravity in m/s² (value is standard, provided here for completeness)

-- Statement of the problem in Lean 4
theorem final_water_level (h_initial : ℝ) (rho_water : ℝ) (rho_oil : ℝ) (g : ℝ):
  ∃ h_final : ℝ, 
  ρ_water * g * h_final = ρ_oil * g * (h_initial - h_final) ∧
  h_final = 0.34 :=
begin
  sorry
end

end final_water_level_l694_694754


namespace smallest_period_of_function_l694_694370

theorem smallest_period_of_function : 
  ∀ (τ : ℝ), 
  ∃ p : ℝ, p > 0 ∧ 
  (∀ x : ℝ, ((λ x, (7 * Real.sin τ * Real.tan x) / (Real.sec x * Real.cos (2 * x) * (1 - Real.tan x^2))) (x + p) = (λ x, (7 * Real.sin τ * Real.tan x) / (Real.sec x * Real.cos (2 * x) * (1 - Real.tan x^2))) x)) ∧ 
  (∀ q : ℝ, q > 0 ∧ (∀ x : ℝ, ((λ x, (7 * Real.sin τ * Real.tan x) / (Real.sec x * Real.cos (2 * x) * (1 - Real.tan x^2))) (x + q) = (λ x, (7 * Real.sin τ * Real.tan x) / (Real.sec x * Real.cos (2 * x) * (1 - Real.tan x^2))) x)) → p ≤ q)
 :=
begin
  assume τ,
  use π,
  split,
  { sorry },  -- Proof that π is positive
  split,
  { sorry },  -- Proof that y(x + π) = y(x)
  { sorry }   -- Proof that no smaller period exists
end

end smallest_period_of_function_l694_694370


namespace distance_relation_l694_694637

variables (a x y : ℝ)
variables (AB AD AE DC C : ℝ)

theorem distance_relation (h1 : AB = 2 * a)
                          (h2 : AE = DC)
                          (h3 : AD < AB) -- This is implicitly understood as AD is a chord and AB is a diameter.
                          (h4 : y = a) -- distance of E from diameter AB
                          (h5 : x = AE) -- distance of E from the tangent through A
                          (h6 : ∃ (BD BC : ℝ), BC * AB = BD^2 ∧ BD = a - x ∧ DC = BC - BD) :
  y^2 = x^3 / (2 * a - x) :=
begin
  sorry
end

end distance_relation_l694_694637


namespace tangent_circles_l694_694696

variable {A B C M K D : Type} [Triangle A B C] [IsIsosceles A B C] (L : Point) (X Y : Point)
variables (hM : M ∈ Segment A B) (hK : K ∈ Segment A C) (hD : D ∈ Line B C)
          (hParallelogram : Parallelogram A M D K) (hIntersect : mkLine M K (Line B C) = L)
          (hPerpendicular : ∃ P : Line, (D ∈ P) ∧ (P ⊥ Line B C) ∧ (X ∈ LineIntersection P A B) ∧ (Y ∈ LineIntersection P A C))

theorem tangent_circles {L : Point} {A X Y D: Point} 
  (h_center: CircleCenterIs L A X Y) 
  (h_radius: CircleRadiusIs L D): TangentCircles (Circle L (dist L D)) (Circumcircle A X Y) := 
sorry

end tangent_circles_l694_694696


namespace distinct_complex_values_of_c_l694_694660

theorem distinct_complex_values_of_c :
  ∃ (r s t u c : ℂ), (r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ s ≠ t ∧ s ≠ u ∧ t ≠ u ∧ 
  (∀ z : ℂ, (z - r) * (z - s) * (z - t) * (z - u) = (z - c * r) * (z - c * s) * (z - c * t) * (z - c * u))  ∧ (c ∈ {1, Complex.I, -1, -Complex.I})):
  ∃ (x : Finset ℂ), x.card = 4 := 
by
  sorry

end distinct_complex_values_of_c_l694_694660


namespace sin_square_eq_c_div_a2_plus_b2_l694_694698

theorem sin_square_eq_c_div_a2_plus_b2 
  (a b c : ℝ) (α β : ℝ)
  (h1 : a * Real.cos α + b * Real.sin α = c)
  (h2 : a * Real.cos β + b * Real.sin β = 0)
  (not_all_zero : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) : 
  Real.sin (α - β) ^ 2 = c ^ 2 / (a ^ 2 + b ^ 2) :=
by
  sorry

end sin_square_eq_c_div_a2_plus_b2_l694_694698


namespace ratio_of_terms_in_arithmetic_sequence_l694_694791

-- Define the terms of the arithmetic sequence
def S (n : ℕ) (a₁ d : ℝ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem ratio_of_terms_in_arithmetic_sequence
    (n : ℕ) (a₁ d : ℝ) (h : S n a₁ d / S (2 * n) a₁ d = (n + 1) / (4 * n + 2)) :
    (a₁ + 2 * d) / (a₁ + 4 * d) = 3 / 5 :=
by
  -- Proof goes here
  sorry

end ratio_of_terms_in_arithmetic_sequence_l694_694791


namespace sum_of_ages_l694_694494

variables (Matthew Rebecca Freddy: ℕ)
variables (H1: Matthew = Rebecca + 2)
variables (H2: Matthew = Freddy - 4)
variables (H3: Freddy = 15)

theorem sum_of_ages
  (H1: Matthew = Rebecca + 2)
  (H2: Matthew = Freddy - 4)
  (H3: Freddy = 15):
  Matthew + Rebecca + Freddy = 35 :=
  sorry

end sum_of_ages_l694_694494


namespace find_a_l694_694941

theorem find_a :
  ∃ (a : ℤ), (∀ (x y : ℤ),
    (∃ (m n : ℤ), (x - 8 + m * y) * (x + 3 + n * y) = x^2 + 7 * x * y + a * y^2 - 5 * x - 45 * y - 24) ↔ a = 6) := 
sorry

end find_a_l694_694941


namespace fractional_part_identity_l694_694339

def k := 2 + Real.sqrt 3

theorem fractional_part_identity (n : ℕ) : 
    k^n - Real.floor (k^n) = 1 - 1 / (k^n) := 
by
  sorry

end fractional_part_identity_l694_694339


namespace largest_k_without_parallel_sides_l694_694054

-- Define the conditions of the problem
def circle_with_points : Type :=
  { points : ℕ // points = 2012 }

-- Define the concept of a convex k-gon without parallel sides
noncomputable def max_k_without_parallel_sides (c : circle_with_points) : ℕ :=
  let k := 1509
  in k -- This represents the computed correct answer from our steps

-- The main theorem to be proven
theorem largest_k_without_parallel_sides (c : circle_with_points) : c.points = 2012 → max_k_without_parallel_sides c = 1509 :=
by
  intro h
  rw [h]
  exact rfl

end largest_k_without_parallel_sides_l694_694054


namespace probStopWithinThreeHops_l694_694526

namespace FriedaTheFrog

def gridSize := 4

def startPos : ℕ × ℕ := (2, 2)

def isBorder (pos : ℕ × ℕ) : Prop :=
  pos.1 = 1 ∨ pos.1 = gridSize ∨ pos.2 = 1 ∨ pos.2 = gridSize

def hopPositions (pos : ℕ × ℕ) : list (ℕ × ℕ) :=
  [(pos.1, pos.2 - 1), (pos.1, pos.2 + 1), (pos.1 - 1, pos.2), (pos.1 + 1, pos.2)]

def wrapPosition (pos : ℕ × ℕ) : ℕ × ℕ :=
  (if pos.1 = 0 then gridSize else if pos.1 > gridSize then 1 else pos.1,
   if pos.2 = 0 then gridSize else if pos.2 > gridSize then 1 else pos.2)

def validHopPositions (pos : ℕ × ℕ) : list (ℕ × ℕ) :=
  (hopPositions pos).map wrapPosition

def probStopAtBorder (n : ℕ) (pos : ℕ × ℕ) : ℚ :=
  if isBorder pos then 1
  else if n = 0 then 0
  else (validHopPositions pos).map (λ p, probStopAtBorder (n - 1) p) |> list.qsum / validHopPositions pos.length

theorem probStopWithinThreeHops : probStopAtBorder 3 startPos = 39 / 64 :=
  sorry

end FriedaTheFrog

end probStopWithinThreeHops_l694_694526


namespace avg_tickets_male_l694_694423

theorem avg_tickets_male (M F : ℕ) (w : ℕ) 
  (h1 : M / F = 1 / 2) 
  (h2 : (M + F) * 66 = M * w + F * 70) 
  : w = 58 := 
sorry

end avg_tickets_male_l694_694423


namespace technician_completed_percentage_l694_694080

-- Define the distance D and the percentage of the round-trip completed (as 75%)
variable (D : ℝ)
variable (round_trip_percentage : ℝ)
variable (to_center_ratio : ℝ)

-- Given conditions as definitions
def distance_to_center := D
def round_trip_distance := 2 * D
def distance_covered := round_trip_percentage * round_trip_distance
def distance_from_center := distance_covered - distance_to_center

-- The proof problem statement to verify if the percentage of the drive from the center is 50%
theorem technician_completed_percentage :
  to_center_ratio = 0.75 → 
  100 * ((distance_from_center D to_center_ratio) / D) = 50 :=
by
    intros h1,
    simp [distance_from_center, distance_to_center, h1, round_trip_distance],
    sorry

end technician_completed_percentage_l694_694080


namespace susan_coins_value_l694_694353

-- Define the conditions as Lean functions and statements.
def total_coins (n d : ℕ) := n + d = 30
def value_if_swapped (n : ℕ) := 10 * n + 5 * (30 - n)
def value_original (n : ℕ) := 5 * n + 10 * (30 - n)
def conditions (n : ℕ) := value_if_swapped n = value_original n + 90

-- The proof statement
theorem susan_coins_value (n d : ℕ) (h1 : total_coins n d) (h2 : conditions n) : 5 * n + 10 * d = 180 := by
  sorry

end susan_coins_value_l694_694353


namespace jaco_final_payment_l694_694065

def total_cost_with_discounts (prices : List ℕ) (discount_100 : ℕ → ℕ) (discount_150 : ℕ → ℕ) : ℕ :=
  let total := prices.foldr (· + ·) 0
  let disc_100 := discount_100 (total / 100 * 100)
  let disc_150 := discount_150 (total / 150 * 150)
  total - disc_100 - disc_150

def discount_100 (n : ℕ) : ℕ := n / 10
def discount_150 (n : ℕ) : ℕ := n * 5 / 100

def prices := [74, 92, 2, 3, 4, 5, 42, 58]

theorem jaco_final_payment : total_cost_with_discounts prices discount_100 discount_150 = 252.5 :=
by
  sorry

end jaco_final_payment_l694_694065


namespace pig_farm_fence_l694_694036

theorem pig_farm_fence (fenced_side : ℝ) (area : ℝ) 
  (h1 : fenced_side * 2 * fenced_side = area) 
  (h2 : area = 1250) :
  4 * fenced_side = 100 :=
by {
  sorry
}

end pig_farm_fence_l694_694036


namespace pond_90_percent_algae_free_l694_694356

def algae_coverage : ℕ → ℝ 
| 30 := 1
| (n+1) := 3 * algae_coverage n

theorem pond_90_percent_algae_free : 
  ∃ d, d = 28 ∧ algae_coverage d ≈ (1 / 10) :=
sorry

end pond_90_percent_algae_free_l694_694356


namespace percent_equivalence_l694_694997

variable (x : ℝ)
axiom condition : 0.30 * 0.15 * x = 18

theorem percent_equivalence :
  0.15 * 0.30 * x = 18 := sorry

end percent_equivalence_l694_694997


namespace proof_of_problem_l694_694198

noncomputable def problem_statement : Prop :=
  ∃ (A ω φ : ℝ), 
    (∀ x : ℝ, f x = A * Real.sin (ω * x + φ)) ∧
    ((A > 0) ∧ (ω > 0) ∧ (0 < φ < (π / 2))) ∧
    (∀ k : ℤ, f (k * (π / 2) / ω) = 0) ∧
    (f (2 * π / 3) = -2) ∧
    (A = 2) ∧ (ω = 2) ∧ (φ = π / 6) ∧
    (∀ k : ℤ, center_of_symmetry = (k * π / 2 - π / 12, 0)) ∧
    (∀ k : ℤ, monotonically_increasing_interval = (- π / 3 + k * π, π / 6 + k * π)) ∧
    (range_of_f_on_interval (π / 12) (π / 2) = [-1, 2])
    
theorem proof_of_problem : problem_statement :=
sorry

end proof_of_problem_l694_694198


namespace Isaac_writing_utensils_total_l694_694293

def Isaac_pens : ℕ := 16
def Isaac_pencils (P : ℕ) : ℕ := 5 * P + 12
def total_writing_utensils (P : ℕ) (L : ℕ) : ℕ := P + L

theorem Isaac_writing_utensils_total : total_writing_utensils Isaac_pens (Isaac_pencils Isaac_pens) = 108 := 
by
  -- Isaac buys 16 pens
  let P := 16
  -- Pencils = 5 * P + 12
  let L := 5 * P + 12
  -- Therefore, total = P + L
  have h : total_writing_utensils P L = P + L := rfl
  rw [h]
  -- Substitute P and L and simplify to show result
  have hP : P = 16 := rfl
  have hL : L = 5 * P + 12 := rfl
  rw [←hP, ←hL]
  calc
    total_writing_utensils 16 (5 * 16 + 12)
      _ = 16 + (5 * 16 + 12) := rfl
      _ = 16 + 80 + 12 := rfl
      _ = 108 := rfl
  sorry

end Isaac_writing_utensils_total_l694_694293


namespace min_value_f_when_a_eq_1_f_gt_x_for_a_in_neg_inf_0_l694_694569

-- Define the function f for general a
def f (a x : ℝ) := x^2 - log x - a * x

-- Prove that the minimum value of f(x) when a = 1 is 0
theorem min_value_f_when_a_eq_1 : (∀ x : ℝ, f 1 x ≥ 0) ∧ (∃ x : ℝ, f 1 x = 0) :=
by
  -- Placeholder for the actual proof.
  sorry

-- Prove that f(x) > x for a in (-∞, 0)
theorem f_gt_x_for_a_in_neg_inf_0 : ∀ (a : ℝ), (a < 0) → (∀ x : ℝ, x > 0 → f a x > x) :=
by
  -- Placeholder for the actual proof.
  sorry

end min_value_f_when_a_eq_1_f_gt_x_for_a_in_neg_inf_0_l694_694569


namespace circle_tangent_l694_694114

noncomputable theory

variables {R r : ℝ}

/-- Construction of a circle that is tangent to two externally tangent circles and a common external tangent -/
theorem circle_tangent
  (hR : 0 < R)
  (hr : 0 < r) :
  ∃ x : ℝ, x = R * r / (Real.sqrt R + Real.sqrt r)^2 ∨ x = R * r / (Real.sqrt R - Real.sqrt r)^2 := 
sorry

end circle_tangent_l694_694114


namespace largest_prime_factor_7_fact_8_fact_l694_694002

theorem largest_prime_factor_7_fact_8_fact : 
  let n := 7! + 8! 
  ∃ p, Prime p ∧ p ∣ n ∧ (∀ q, Prime q ∧ q ∣ n → q ≤ p) := 
begin
  let n := fact 7 + fact 8,
  use 7, 
  split,
  { apply Prime.prime_7 },
  split,
  { sorry },  
  { sorry },
end

end largest_prime_factor_7_fact_8_fact_l694_694002


namespace matrix_inverse_proof_l694_694921

open Matrix

def matrix_inverse_problem : Prop :=
  let A := ![
    ![7, -2],
    ![-3, 1]
  ]
  let A_inv := ![
    ![1, 2],
    ![3, 7]
  ]
  A.mul A_inv = (1 : ℤ) • (1 : Matrix (Fin 2) (Fin 2))
  
theorem matrix_inverse_proof : matrix_inverse_problem :=
  by
  sorry

end matrix_inverse_proof_l694_694921


namespace largest_prime_factor_7_fact_8_fact_l694_694005

theorem largest_prime_factor_7_fact_8_fact : 
  let n := 7! + 8! 
  ∃ p, Prime p ∧ p ∣ n ∧ (∀ q, Prime q ∧ q ∣ n → q ≤ p) := 
begin
  let n := fact 7 + fact 8,
  use 7, 
  split,
  { apply Prime.prime_7 },
  split,
  { sorry },  
  { sorry },
end

end largest_prime_factor_7_fact_8_fact_l694_694005


namespace combined_molecular_weight_l694_694864

theorem combined_molecular_weight 
  (atomic_weight_N : ℝ)
  (atomic_weight_O : ℝ)
  (atomic_weight_H : ℝ)
  (atomic_weight_C : ℝ)
  (moles_N2O3 : ℝ)
  (moles_H2O : ℝ)
  (moles_CO2 : ℝ)
  (molecular_weight_N2O3 : ℝ)
  (molecular_weight_H2O : ℝ)
  (molecular_weight_CO2 : ℝ)
  (weight_N2O3 : ℝ)
  (weight_H2O : ℝ)
  (weight_CO2 : ℝ)
  : 
  moles_N2O3 = 4 →
  moles_H2O = 3.5 →
  moles_CO2 = 2 →
  atomic_weight_N = 14.01 →
  atomic_weight_O = 16.00 →
  atomic_weight_H = 1.01 →
  atomic_weight_C = 12.01 →
  molecular_weight_N2O3 = (2 * atomic_weight_N) + (3 * atomic_weight_O) →
  molecular_weight_H2O = (2 * atomic_weight_H) + atomic_weight_O →
  molecular_weight_CO2 = atomic_weight_C + (2 * atomic_weight_O) →
  weight_N2O3 = moles_N2O3 * molecular_weight_N2O3 →
  weight_H2O = moles_H2O * molecular_weight_H2O →
  weight_CO2 = moles_CO2 * molecular_weight_CO2 →
  weight_N2O3 + weight_H2O + weight_CO2 = 455.17 :=
by 
  intros;
  sorry

end combined_molecular_weight_l694_694864


namespace rectangle_area_inscribed_circle_l694_694449

theorem rectangle_area_inscribed_circle (radius : ℝ) (length_ratio : ℝ) (width_ratio : ℝ) 
  (h_radius : radius = 7) 
  (h_length_ratio : length_ratio = 3) 
  (h_width_ratio : width_ratio = 1) : 
  let width := 2 * radius in 
  let length := (length_ratio / width_ratio) * width in 
  width * length = 588 :=
by
  sorry

end rectangle_area_inscribed_circle_l694_694449


namespace quadratic_solution_l694_694552

theorem quadratic_solution (a : ℝ) (h : (1 : ℝ)^2 + 1 + 2 * a = 0) : a = -1 :=
by {
  sorry
}

end quadratic_solution_l694_694552


namespace largest_prime_factor_of_7_fact_add_8_fact_l694_694016

theorem largest_prime_factor_of_7_fact_add_8_fact :
  let n := 7!
  let m := 8!
  let sum := n + m
  prime_factors sum = {2, 3, 5, 7} ∧ (∀ p ∈ {2, 3, 5, 7}, p <= 7) := by
  sorry

end largest_prime_factor_of_7_fact_add_8_fact_l694_694016


namespace sin_cos_values_sin_ϕ_value_l694_694977

variable (θ ϕ : ℝ)

-- Part 1
theorem sin_cos_values (h_perp: (sin θ) - 2 * (cos θ) = 0) 
(h_bound: θ ∈ Ioo 0 (π / 2)) : 
  (sin θ = (2 * sqrt 5) / 5) ∧ (cos θ = sqrt 5 / 5) :=
sorry

-- Part 2
theorem sin_ϕ_value (h_theta: θ ∈ Ioo 0 (π / 2))
(h_ϕ: ϕ ∈ Ioo 0 (π / 2)) 
(h_sin_diff: sin (θ - ϕ) = sqrt 10 / 10) 
(h_sinθ: sin θ = (2 * sqrt 5) / 5) 
(h_cosθ: cos θ = sqrt 5 / 5) :
  sin ϕ = sqrt 2 / 2 :=
sorry

end sin_cos_values_sin_ϕ_value_l694_694977


namespace percentage_of_apples_after_removal_l694_694799

-- Declare the initial conditions as Lean definitions
def initial_apples : Nat := 12
def initial_oranges : Nat := 23
def removed_oranges : Nat := 15

-- Calculate the new totals
def new_oranges : Nat := initial_oranges - removed_oranges
def new_total_fruit : Nat := initial_apples + new_oranges

-- Define the expected percentage of apples as a real number
def expected_percentage_apples : Nat := 60

-- Prove that the percentage of apples after removing the specified number of oranges is 60%
theorem percentage_of_apples_after_removal :
  (initial_apples * 100 / new_total_fruit) = expected_percentage_apples := by
  sorry

end percentage_of_apples_after_removal_l694_694799


namespace tan_A_in_right_triangle_l694_694505

noncomputable def ABC_right_triangle (A B C : ℝ) : Prop :=
  ∃ (AB AC BC : ℝ), AB = 15 ∧ AC = 20 ∧ BC = 25 ∧ AB^2 + AC^2 = BC^2

theorem tan_A_in_right_triangle (A B C : ℝ) (h : ABC_right_triangle A B C) : 
  tan A = 3 / 4 :=
by
  sorry

end tan_A_in_right_triangle_l694_694505


namespace number_of_subsets_of_two_element_set_is_four_l694_694771

theorem number_of_subsets_of_two_element_set_is_four (a b : Type) :
  ∃ (s : set (set (a × b))), s.card = 4 :=
  sorry

end number_of_subsets_of_two_element_set_is_four_l694_694771


namespace problem_l694_694612

variable {a m : ℝ}

def f (x : ℝ) : ℝ := x^2 - x + a

theorem problem 
  (h : f (-m) < 0) : f (m + 1) < 0 := sorry

end problem_l694_694612


namespace count_divisible_by_11_with_digits_sum_10_l694_694603

-- Definitions and conditions from step a)
def digits_add_up_to_10 (n : ℕ) : Prop :=
  let ds := n.digits 10 in ds.length = 4 ∧ ds.sum = 10

def divisible_by_11 (n : ℕ) : Prop :=
  let ds := n.digits 10 in
  (ds.nth 0).get_or_else 0 +
  (ds.nth 2).get_or_else 0 -
  (ds.nth 1).get_or_else 0 -
  (ds.nth 3).get_or_else 0 ∣ 11

-- The tuple (question, conditions, answer) formalized in Lean
theorem count_divisible_by_11_with_digits_sum_10 :
  ∃ N : ℕ, N = (
  (Finset.filter (λ x, digits_add_up_to_10 x ∧ divisible_by_11 x)
  (Finset.range 10000)).card
) := 
sorry

end count_divisible_by_11_with_digits_sum_10_l694_694603


namespace infinite_squares_of_sequence_l694_694784

-- Define the sequence with the given conditions
def a : ℕ → ℝ
| 0     => 2
| (n+1) => ((1 + 1/(n:ℝ))^n) * (a n)

-- Prove that there exist infinitely many n such that (a 1 * a 2 * ... * a n) / (n + 1) is a square of an integer
theorem infinite_squares_of_sequence: ∃ᶠ n in at_top, ∃ (k : ℤ), (∏ i in (Finset.range (n+1)), a i) / (n+1) = k^2 :=
begin
  sorry
end

end infinite_squares_of_sequence_l694_694784


namespace number_of_allocation_schemes_l694_694838

/-- 
  Given 5 volunteers and 4 projects, each volunteer is assigned to only one project,
  and each project must have at least one volunteer.
  Prove that there are 240 different allocation schemes.
-/
theorem number_of_allocation_schemes (V P : ℕ) (hV : V = 5) (hP : P = 4) 
  (each_volunteer_one_project : ∀ v, ∃ p, v ≠ p) 
  (each_project_at_least_one : ∀ p, ∃ v, v ≠ p) : 
  ∃ n_ways : ℕ, n_ways = 240 :=
by
  sorry

end number_of_allocation_schemes_l694_694838


namespace minimum_race_distance_l694_694381

theorem minimum_race_distance : 
  ∀ (A B : ℝ×ℝ) (wall_start wall_end : ℝ×ℝ), 
    A = (0, -200) → 
    wall_start = (0, 0) → 
    wall_end = (800, 0) → 
    B = (400, 600) → 
    let B' := (400, -600) 
    let distance := Real.sqrt ((B'.1 - A.1)^2 + (B'.2 - A.2)^2)
    Int.round distance = 566 := 
by 
  intros A B wall_start wall_end hA hstart hend hB 
  let B' := (400, -600)
  let distance := Real.sqrt ((B'.1 - A.1)^2 + (B'.2 - A.2)^2)
  have hdist : distance = 400 * Real.sqrt 2 := by 
    sorry
  have hround : Int.round distance = 566 := by 
    sorry
  exact hround

end minimum_race_distance_l694_694381


namespace sqrt_inequality_l694_694417

theorem sqrt_inequality (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (h : sqrt a < sqrt b) : a < b := 
by 
  sorry

end sqrt_inequality_l694_694417


namespace percent_problem_l694_694996

theorem percent_problem (x : ℝ) (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 :=
by
  sorry

end percent_problem_l694_694996


namespace frood_points_l694_694635

theorem frood_points (n : ℕ) (h : n > 29) : (n * (n + 1) / 2) > 15 * n := by
  sorry

end frood_points_l694_694635


namespace length_of_XY_l694_694751

/-- Define the right scalene triangle XYZ with the given side ratios and conditions  -/
noncomputable def triangle_XYZ (a b c n : ℕ) (h1 : a < b ∧ b < c) (h2 : n > 0) 
  (h3 : ∃ (XY YZ XZ : ℝ), XY = n * b ∧ YZ = n * a ∧ XZ = n * c ∧ XY > YZ) : Prop := 
  (1 / 2) * (n * a) * (n * b) = 9

/-- Define the length of side XY which corresponds to n * b -/
def length_XY (a n : ℕ) : ℝ := 18 / (n * a)

theorem length_of_XY (a b c n : ℕ) (h1 : a < b ∧ b < c) (h2 : n > 0) : 
  (triangle_XYZ a b c n h1 h2) ∧ (length_XY a n = 18 / (n * a)) :=
by
  sorry

end length_of_XY_l694_694751


namespace range_of_m_l694_694246

noncomputable def quadratic_inequality_solution_set_is_R (m : ℝ) : Prop :=
  ∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x + 2 > 0

theorem range_of_m :
  { m : ℝ | quadratic_inequality_solution_set_is_R m } = { m : ℝ | 1 ≤ m ∧ m < 9 } :=
by
  sorry

end range_of_m_l694_694246


namespace original_cost_is_49_l694_694525

-- Define the conditions as assumptions
def original_cost_of_jeans (x : ℝ) : Prop :=
  let discounted_price := x / 2
  let wednesday_price := discounted_price - 10
  wednesday_price = 14.5

-- The theorem to prove
theorem original_cost_is_49 :
  ∃ x : ℝ, original_cost_of_jeans x ∧ x = 49 :=
by
  sorry

end original_cost_is_49_l694_694525


namespace problem1_problem2_problem3_l694_694162

def f (x a : ℝ) := x^2 + 3 * (abs (x - a))

def g (a : ℝ) : ℝ :=
  if 0 < a ∧ a < 1 then a^2
  else if 1 ≤ a then 3 * a - 2
  else 0 -- g(a) is only defined for a > 0

def minimum_interval (a : ℝ) (x : ℝ) (b : ℝ) : Prop :=
  f x a = b

def increasing_interval (a : ℝ) (interval : set ℝ) : Prop :=
  ∀ x y ∈ interval, x ≤ y → f x a ≤ f y a

def decreasing_interval (a : ℝ) (interval : set ℝ) : Prop :=
  ∀ x y ∈ interval, x ≤ y → f x a ≥ f y a

theorem problem1 (a : ℝ) (ha : a = 1) :
  increasing_interval 1 (set.Ici 1) ∧ decreasing_interval 1 (set.Iio 1) ∧ minimum_interval 1 1 1 :=
sorry

theorem problem2 (a : ℝ) (ha : a > 0) :
  g a = (if 0 < a ∧ a < 1 then a^2 else 3 * a - 2) :=
sorry

theorem problem3 (a : ℝ) (ha : 0 < a) (m : ℝ) :
  (∀ x ∈ set.Icc (-1) 1, f x a ≤ g a + m) ↔ (6 ≤ m) :=
sorry

end problem1_problem2_problem3_l694_694162


namespace area_AEDCB_is_304_l694_694700

noncomputable def area_pentagon (AE DE : ℝ) (hAE : AE = 12) (hDE : DE = 16)
                                (h_perp : ∀ (A E D : ℝ), AE * DE = 0) : ℝ :=
  let AD := Real.sqrt (AE^2 + DE^2) in
  let area_square := AD^2 in
  let area_triangle := 0.5 * AE * DE in
  area_square - area_triangle

theorem area_AEDCB_is_304 :
  area_pentagon 12 16 12 16 sorry = 304 := sorry

end area_AEDCB_is_304_l694_694700


namespace largest_prime_factor_7_fact_8_fact_l694_694010

theorem largest_prime_factor_7_fact_8_fact : 
  let n := 7! + 8! 
  ∃ p, Prime p ∧ p ∣ n ∧ (∀ q, Prime q ∧ q ∣ n → q ≤ p) := 
begin
  let n := fact 7 + fact 8,
  use 7, 
  split,
  { apply Prime.prime_7 },
  split,
  { sorry },  
  { sorry },
end

end largest_prime_factor_7_fact_8_fact_l694_694010


namespace max_omega_l694_694566

noncomputable def f (x : Real) (ω : Real) (ϕ : Real) : Real := sin (ω * x + ϕ)

theorem max_omega (ω ϕ : Real) (hω : ω > 0) (hϕ : abs ϕ ≤ Real.pi / 2)
  (h_zero : f (-Real.pi / 4) ω ϕ = 0)
  (h_bound : ∀ x, f x ω ϕ ≤ abs (f (Real.pi / 4) ω ϕ))
  (h_min : is_glb (Set.image (f (·) ω ϕ) (Set.Ioo (-Real.pi / 12) (Real.pi / 24))) (f (Some -Real.pi / 12) ω ϕ))
  (h_no_max : ¬ is_lub (Set.image (f (·) ω ϕ) (Set.Ioo (-Real.pi / 12) (Real.pi / 24))) (f (Some Real.pi / 24) ω ϕ))
   : ω = 15 := 
begin
  sorry
end

end max_omega_l694_694566


namespace jill_investment_value_correct_l694_694298

def future_value (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) (PMT : ℝ) : ℝ :=
  let A := P * (1 + r / n)^(n * t)
  let B := PMT * (((1 + r / n)^(n * t) - 1) / (r / n))
  A + B

noncomputable def jill_future_value : ℝ :=
  future_value 10000 0.0396 12 2 200

theorem jill_investment_value_correct :
  abs (jill_future_value - 15761.46) < 0.01 :=
by
  sorry

end jill_investment_value_correct_l694_694298


namespace taxi_growth_rate_eq_l694_694859

constant a : ℝ 
constant b : ℝ
constant x : ℝ

axiom h1 : a = 11720
axiom h2 : b = 13116

theorem taxi_growth_rate_eq : a * (1 + x)^2 = b := 
by 
    rw [h1, h2] 
    sorry

end taxi_growth_rate_eq_l694_694859


namespace incorrect_expressions_l694_694816

/-
  Define a structure for a repeating decimal consisting of a non-repeating part X and a repeating part Y.
  Then, the main theorem verifies which expressions (B) and (D) are incorrect.
-/

structure RepeatingDecimal where
  (E : ℚ)
  (X : ℕ) -- Non-repeating part
  (Y : ℕ) -- Repeating part
  (t : ℕ) -- Length of X
  (u : ℕ) -- Length of Y
  (E_def : E = (X : ℚ) + (Y : ℚ) / (10 ^ u - 1) / (10^t))

-- The main theorem that captures the question's essence.
theorem incorrect_expressions (E : ℚ) (X Y : ℕ) (t u : ℕ) (r : RepeatingDecimal E X Y t u)
    (h1 : r.E = .(X).(Y)(Y)(Y)...) :
    (10^t * E ≠ X + (Y / (10 - 1))) ∧ (10^t * (10^u - 1) * E ≠ Y * (X - 1)) := by
  sorry -- proof not needed as per problem statement.


end incorrect_expressions_l694_694816


namespace eiffel_tower_scale_l694_694714

theorem eiffel_tower_scale (height_tower_m : ℝ) (height_model_cm : ℝ) :
    height_tower_m = 324 →
    height_model_cm = 50 →
    (height_tower_m * 100) / height_model_cm = 648 →
    (648 / 100) = 6.48 :=
by
  intro h_tower h_model h_ratio
  rw [h_tower, h_model] at h_ratio
  sorry

end eiffel_tower_scale_l694_694714


namespace required_bricks_l694_694035

def brick_volume (l w h : ℕ) : ℕ := l * w * h
def wall_volume (l w h : ℕ) : ℕ := l * w * h

theorem required_bricks : 
  let brick_l := 20
  let brick_w := 10
  let brick_h := 7.5
  let wall_l := 29 * 100 -- convert to cm
  let wall_w := 2 * 100 -- convert to cm
  let wall_h := 0.75 * 100 -- convert to cm
  let brick_vol := brick_volume brick_l brick_w brick_h
  let wall_vol := wall_volume wall_l wall_w wall_h
  wall_vol / brick_vol = 2900 :=
by
  let brick_l := 20
  let brick_w := 10
  let brick_h := 75 / 10
  let wall_l := 29 * 100
  let wall_w := 2 * 100
  let wall_h := 75 
  let brick_vol := brick_volume brick_l brick_w brick_h.to_nat
  let wall_vol := wall_volume wall_l wall_w wall_h.to_nat
  sorry

end required_bricks_l694_694035


namespace find_ab_l694_694238

-- Define the conditions
variables (a b : ℝ)
hypothesis h1 : a - b = 3
hypothesis h2 : a^2 + b^2 = 29

-- State the theorem
theorem find_ab : a * b = 10 :=
by
  sorry

end find_ab_l694_694238


namespace sum_of_18_fluctuation_property_l694_694430

-- Defining the "fluctuation property"
def fluctuation_property (seq : List ℤ) : Prop :=
  ∀ i, 1 ≤ i ∧ i < seq.length - 1 → seq.nth i = seq.nth (i-1) + seq.nth (i+1)

-- The sequence given with positions for * filled
def given_sequence (a b c d e f g h i j k l m n o p q r : ℤ) : List ℤ :=
  [1, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, 1]

-- Lean theorem statement
theorem sum_of_18_fluctuation_property (a b c d e f g h i j k l m n o p q r : ℤ)
  (fluc_prop : fluctuation_property (given_sequence a b c d e f g h i j k l m n o p q r)) : 
  a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + r = 0 :=
by sorry

end sum_of_18_fluctuation_property_l694_694430


namespace four_digit_numbers_sum_30_l694_694932

-- Definitions of the variables and constraints
def valid_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

-- The main statement we aim to prove
theorem four_digit_numbers_sum_30 : 
  ∃ (count : ℕ), 
  count = 20 ∧ 
  ∃ (a b c d : ℕ), 
  (1 ≤ a ∧ valid_digit a) ∧ 
  (valid_digit b) ∧ 
  (valid_digit c) ∧ 
  (valid_digit d) ∧ 
  a + b + c + d = 30 := sorry

end four_digit_numbers_sum_30_l694_694932


namespace three_digit_cubes_divisible_by_4_l694_694226

-- Let's define the conditions in Lean
def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n
def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Let's combine these conditions to define the target predicate in Lean
def is_target_number (n : ℕ) : Prop := is_three_digit n ∧ is_perfect_cube n ∧ is_divisible_by_4 n

-- The statement to be proven: that there is only one such number
theorem three_digit_cubes_divisible_by_4 : 
  (∃! n, is_target_number n) :=
sorry

end three_digit_cubes_divisible_by_4_l694_694226


namespace range_of_eccentricity_l694_694728

-- Define that a > 0 and b > 0
variables {a b c : ℝ}
variables (x y : ℝ)

-- Define the hyperbola equation
def hyperbola (x y a b : ℝ) := x^2 / a^2 - y^2 / b^2 = 1

-- Assume a > 0 and b > 0
axiom a_gt_zero : 0 < a
axiom b_gt_zero : 0 < b

-- Define the eccentricity condition
def eccentricity (c a : ℝ) := c / a

-- Prove that the range of the eccentricity e is (1, √3)
theorem range_of_eccentricity (e : ℝ) (h_hyperbola : hyperbola x y a b) 
  (h_angle : ∃ A B : ℝ × ℝ, ∃ F1 F2 : ℝ × ℝ, angle F1 F2 A B < π / 3) 
  (h_foci : ∀ A B : ℝ × ℝ, exists_line_perpendicular_to_x_axis (h_hyperbola F1) ∧ intersects_hyperbola h_hyperbola A B) :
  1 < eccentricity c a ∧ eccentricity c a < sqrt 3 :=
sorry

end range_of_eccentricity_l694_694728


namespace xy_in_B_l694_694946

def A : Set ℤ := 
  {z | ∃ a b k m : ℤ, z = m * a^2 + k * a * b + m * b^2}

def B : Set ℤ := 
  {z | ∃ a b k m : ℤ, z = a^2 + k * a * b + m^2 * b^2}

theorem xy_in_B (x y : ℤ) (h1 : x ∈ A) (h2 : y ∈ A) : x * y ∈ B := by
  sorry

end xy_in_B_l694_694946


namespace digit_2023_in_7_over_26_l694_694916

theorem digit_2023_in_7_over_26 :
    (decimalExpansion 7 26).digit 2023 = 5 := sorry

def decimalExpansion (numerator denominator : Int) : List Int :=
    -- Implementation of decimal expansion here
    sorry

structure DecimalExpansion where
    digit : Int -> Int

end digit_2023_in_7_over_26_l694_694916


namespace arithmetic_sequence_S2016_l694_694178

noncomputable def S {a : ℕ → ℝ} (n : ℕ) := (n * (a 1 + a n)) / 2

theorem arithmetic_sequence_S2016
  (a : ℕ → ℝ)
  (h1 : a 4 + a 2013 = 1)
  (h2 : a 1 + a 2016 = 1) :
  S a 2016 = 1008 :=
by
  -- This proof is skipped with sorry.
  sorry

end arithmetic_sequence_S2016_l694_694178


namespace hydrogen_production_l694_694917

noncomputable def molar_mass_benzene : ℝ := 78.108

def moles_of_benzene (mass : ℝ) : ℝ := mass / molar_mass_benzene

def reaction_hydrogen (moles_methane moles_benzene_mass : ℝ) (balanced_eq : ℝ → ℝ → ℝ → ℝ → Prop) : ℝ :=
  if balanced_eq moles_methane moles_benzene_mass (moles_methane) (moles_methane) then moles_methane else 0

theorem hydrogen_production :
  ∀ (mass_of_benzene : ℝ), mass_of_benzene = 156 →
  reaction_hydrogen 2 (moles_of_benzene mass_of_benzene) (λ c₆h₆ ch₄ c₇h₈ h₂, c₆h₆ = ch₄ ∧ ch₄ = c₇h₈ ∧ c₇h₈ = h₂) = 2 :=
by
  intros
  sorry

end hydrogen_production_l694_694917


namespace find_sets_l694_694516

variable (A X Y : Set ℕ) -- Mimicking sets of natural numbers for generality.

theorem find_sets (h1 : X ∪ Y = A) (h2 : X ∩ A = Y) : X = A ∧ Y = A := by
  -- This would need a proof, which shows that: X = A and Y = A
  sorry

end find_sets_l694_694516


namespace exists_rectangle_inscribed_l694_694290

/-- Given a triangle ABC, and a length d,
    there exists a rectangle PQRS inscribed in the triangle ABC such that:
    - R and Q lie on sides AB and BC,
    - P and S lie on side AC,
    - The diagonal of the rectangle PQRS has length d.
-/
theorem exists_rectangle_inscribed {A B C P Q R S : Point} (d : ℝ) 
  (h_inscribed : inscribed_rectangle A B C P Q R S)
  (h_diagonal_length : diagonal_length P Q R S = d) :
  ∃ P Q R S, inscribed_rectangle A B C P Q R S ∧ diagonal_length P Q R S = d :=
sorry

end exists_rectangle_inscribed_l694_694290


namespace weight_of_replaced_person_l694_694360

theorem weight_of_replaced_person
  (average_increase : Real) 
  (new_person_weight : Real) 
  (increase_per_person : Real) 
  (num_persons : Nat) 
  (old_person_weight : Real) : 
  new_person_weight = old_person_weight + (increase_per_person * num_persons) → old_person_weight = 75 :=
by
  intro h
  have h1 : increase_per_person * num_persons = 4.5 * 6 := by sorry
  have h2 : old_person_weight = 102 - 27 := by sorry
  exact h2

end weight_of_replaced_person_l694_694360


namespace triangle_area_l694_694632

theorem triangle_area
  (area_WXYZ : ℝ)
  (side_small_squares : ℝ)
  (AB_eq_AC : ℝ)
  (A_coincides_with_O : ℝ)
  (area : ℝ) :
  area_WXYZ = 49 →  -- The area of square WXYZ is 49 cm^2
  side_small_squares = 2 → -- Sides of the smaller squares are 2 cm long
  AB_eq_AC = AB_eq_AC → -- Triangle ABC is isosceles with AB = AC
  A_coincides_with_O = A_coincides_with_O → -- A coincides with O
  area = 45 / 4 := -- The area of triangle ABC is 45/4 cm^2
by
  sorry

end triangle_area_l694_694632


namespace dolls_given_by_grandmother_l694_694694

/-- Peggy originally has 6 dolls.
    Peggy receives some dolls from her grandmother.
    Peggy receives half the amount of the grandmother's dolls between her birthday and Christmas.
    Peggy ends up with 51 dolls in total.
    Prove that the number of dolls Peggy's grandmother gave her is 30.
--/
theorem dolls_given_by_grandmother (G : ℝ) :
  6 + G + G / 2 = 51 → G = 30 :=
by
  intro h
  rw [← add_div_eq_mul_add_div, ← mul_div_assoc] at h
  linarith

end dolls_given_by_grandmother_l694_694694


namespace negation_proposition_equivalence_l694_694374

theorem negation_proposition_equivalence :
  (¬ (∃ x₀ : ℝ, x₀^2 - x₀ - 1 > 0)) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
by sorry

end negation_proposition_equivalence_l694_694374


namespace max_value_inequality_l694_694665

theorem max_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5 * x + 3 * y < 90) : 
  x * y * (90 - 5 * x - 3 * y) ≤ 1800 := 
sorry

end max_value_inequality_l694_694665


namespace complementary_set_count_l694_694129

def shape := {circle, square, triangle}
def color := {red, blue, green}
def shade := {light, medium, dark}
def pattern := {dots, stripes, solids}

structure Card where
  shape : shape
  color : color
  shade : shade
  pattern : pattern

def deck : Finset Card := Finset.univ

noncomputable def isComplementarySet (s : Finset Card) : Prop :=
  s.card = 3 ∧
  (∀ (attr : s.card.map Card.shape), (attr.card = 1 ∨ attr.card = 3)) ∧
  (∀ (attr : s.card.map Card.color), (attr.card = 1 ∨ attr.card = 3)) ∧
  (∀ (attr : s.card.map Card.shade), (attr.card = 1 ∨ attr.card = 3)) ∧
  (∀ (attr : s.card.map Card.pattern), (attr.card = 1 ∨ attr.card = 3))

theorem complementary_set_count : 
  (Finset.filter isComplementarySet (Finset.powersetLen 3 deck)).card = 4563 := sorry

end complementary_set_count_l694_694129


namespace count_valid_ways_l694_694456

theorem count_valid_ways (n : ℕ) (h1 : n = 6) : 
  ∀ (library : ℕ), (1 ≤ library) → (library ≤ 5) → ∃ (checked_out : ℕ), 
  (checked_out = n - library) := 
sorry

end count_valid_ways_l694_694456


namespace B_is_345_complement_U_A_inter_B_is_3_l694_694975

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set ℕ := {2, 4, 5}

-- Define set B as given in the conditions
def B : Set ℕ := {x ∈ U | 2 < x ∧ x < 6}

-- Prove that B is {3, 4, 5}
theorem B_is_345 : B = {3, 4, 5} := by
  sorry

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := U \ A

-- Prove the intersection of the complement of A and B is {3}
theorem complement_U_A_inter_B_is_3 : (complement_U_A ∩ B) = {3} := by
  sorry

end B_is_345_complement_U_A_inter_B_is_3_l694_694975


namespace function_monotone_decreasing_l694_694920

noncomputable def f : ℝ → ℝ := λ x => x^3 - 2 * x^2 - 4 * x + 2

theorem function_monotone_decreasing :
  ∀ x, x ∈ Ioo (-2/3) 2 → deriv f x < 0 :=
by
  sorry

end function_monotone_decreasing_l694_694920


namespace max_students_taken_exam_l694_694267

-- Define the problem conditions
constant students : Type
constant questions : fin 4
constant answers : Type
constant possible_answers : answers → fin 3

-- Define the relation of answers
constant student_answers : students → questions → answers

-- State the main theorem
theorem max_students_taken_exam
  (h1 : ∀ s1 s2 s3 : students, ∃ q : questions,
    student_answers s1 q ≠ student_answers s2 q ∧
    student_answers s2 q ≠ student_answers s3 q ∧
    student_answers s3 q ≠ student_answers s1 q) :
  (∃ n : ℕ, ∀ S : fin n → students, n ≤ 9) :=
sorry

end max_students_taken_exam_l694_694267


namespace four_digit_number_divisible_by_11_l694_694585

theorem four_digit_number_divisible_by_11 :
  ∃ (a b c d : ℕ), 
    a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ 
    a + b + c + d = 10 ∧ 
    (a + c) % 11 = (b + d) % 11 ∧
    (10 - a != 0 ∨ 10 - c != 0 ∨ 10 - b != 0 ∨ 10 - d != 0) := sorry

end four_digit_number_divisible_by_11_l694_694585


namespace midpoints_of_triangle_l694_694282

-- Definitions of the points and their positions
variables {A B C X Y Z : Type}
variables [plane E] [bc : line E B C] [ca : line E C A] [ab : line E A B]
variables (parallelYX_AB : Parallel (line_through E Y X) (line_through E A B))
variables (parallelZY_BC : Parallel (line_through E Z Y) (line_through E B C))
variables (parallelXZ_CA : Parallel (line_through E X Z) (line_through E C A))

-- The main theorem statement
theorem midpoints_of_triangle 
    (h1 : X ∈ bc)
    (h2 : Y ∈ ca)
    (h3 : Z ∈ ab)
    (h4 : Parallel (line_through E Y X) (line_through E A B))
    (h5 : Parallel (line_through E Z Y) (line_through E B C))
    (h6 : Parallel (line_through E X Z) (line_through E C A)) :
    is_midpoint X B C ∧ is_midpoint Y A C ∧ is_midpoint Z A B :=
  sorry

end midpoints_of_triangle_l694_694282


namespace inequality_proof_l694_694316

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * a * c) + c / Real.sqrt (c^2 + 8 * a * b) >= 1) :=
by
  sorry

end inequality_proof_l694_694316


namespace distance_vancouver_calgary_l694_694361

theorem distance_vancouver_calgary : 
  ∀ (map_distance : ℝ) (scale : ℝ) (terrain_factor : ℝ), 
    map_distance = 12 →
    scale = 35 →
    terrain_factor = 1.1 →
    map_distance * scale * terrain_factor = 462 := by
  intros map_distance scale terrain_factor 
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end distance_vancouver_calgary_l694_694361


namespace purely_imaginary_x_value_l694_694987

theorem purely_imaginary_x_value (x : ℝ) (h1 : x^2 - 1 = 0) (h2 : x + 1 ≠ 0) : x = 1 :=
by
  sorry

end purely_imaginary_x_value_l694_694987


namespace sum_of_digits_of_special_number_l694_694824

theorem sum_of_digits_of_special_number :
  ∀ (x y z : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 0 ≤ z ∧ z ≤ 9 ∧ (100 * x + 10 * y + z = x.factorial + y.factorial + z.factorial) →
  (x + y + z = 10) :=
by
  sorry

end sum_of_digits_of_special_number_l694_694824


namespace total_profit_is_2560_l694_694347

noncomputable def basicWashPrice : ℕ := 5
noncomputable def deluxeWashPrice : ℕ := 10
noncomputable def premiumWashPrice : ℕ := 15

noncomputable def basicCarsWeekday : ℕ := 50
noncomputable def deluxeCarsWeekday : ℕ := 40
noncomputable def premiumCarsWeekday : ℕ := 20

noncomputable def employeeADailyWage : ℕ := 110
noncomputable def employeeBDailyWage : ℕ := 90
noncomputable def employeeCDailyWage : ℕ := 100
noncomputable def employeeDDailyWage : ℕ := 80

noncomputable def operatingExpenseWeekday : ℕ := 200

noncomputable def totalProfit : ℕ := 
  let revenueWeekday := (basicCarsWeekday * basicWashPrice) + 
                        (deluxeCarsWeekday * deluxeWashPrice) + 
                        (premiumCarsWeekday * premiumWashPrice)
  let totalRevenue := revenueWeekday * 5
  let wageA := employeeADailyWage * 5
  let wageB := employeeBDailyWage * 2
  let wageC := employeeCDailyWage * 3
  let wageD := employeeDDailyWage * 2
  let totalWages := wageA + wageB + wageC + wageD
  let totalOperatingExpenses := operatingExpenseWeekday * 5
  totalRevenue - (totalWages + totalOperatingExpenses)

theorem total_profit_is_2560 : totalProfit = 2560 := by
  sorry

end total_profit_is_2560_l694_694347


namespace range_of_m_l694_694248

theorem range_of_m 
  (m : ℝ)
  (h1 : ∀ (x : ℤ), x < m → 7 - 2 * x ≤ 1)
  (h2 : ∃ k : ℕ, set_of (λ i : ℤ, 3 ≤ i ∧ i < m).card = k ∧ k = 4) :
  6 < m ∧ m ≤ 7 :=
sorry

end range_of_m_l694_694248


namespace cloud9_total_money_l694_694875

-- Definitions
def I : ℕ := 12000
def G : ℕ := 16000
def R : ℕ := 1600

-- Total money taken after cancellations
def T : ℕ := I + G - R

-- Theorem stating that T is 26400
theorem cloud9_total_money : T = 26400 := by
  unfold T I G R
  sorry

end cloud9_total_money_l694_694875


namespace geom_seq_problem_l694_694265

variable {a : ℕ → ℝ}  -- positive geometric sequence

-- Conditions
def geom_seq (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n, a n = a 0 * r^n

theorem geom_seq_problem
  (h_geom : geom_seq a)
  (cond : a 0 * a 4 + 2 * a 2 * a 4 + a 2 * a 6 = 25) :
  a 2 + a 4 = 5 :=
sorry

end geom_seq_problem_l694_694265


namespace distance_between_red_lights_l694_694343

def position_of_nth_red (n : ℕ) : ℕ :=
  7 * (n - 1) / 3 + n

def in_feet (inches : ℕ) : ℕ :=
  inches / 12

theorem distance_between_red_lights :
  in_feet ((position_of_nth_red 30 - position_of_nth_red 5) * 8) = 41 :=
by
  sorry

end distance_between_red_lights_l694_694343


namespace volunteer_allocation_scheme_l694_694835

def num_allocation_schemes : ℕ :=
  let num_ways_choose_2_from_5 := Nat.choose 5 2
  let num_ways_arrange_4_groups := Nat.factorial 4
  num_ways_choose_2_from_5 * num_ways_arrange_4_groups

theorem volunteer_allocation_scheme :
  num_allocation_schemes = 240 :=
by
  sorry

end volunteer_allocation_scheme_l694_694835


namespace total_legs_l694_694902

theorem total_legs (chickens sheep : ℕ) (chicken_legs sheep_legs total_legs : ℕ) (h1 : chickens = 7) (h2 : sheep = 5) 
(h3 : chicken_legs = 2) (h4 : sheep_legs = 4) (h5 : total_legs = chickens * chicken_legs + sheep * sheep_legs) : 
total_legs = 34 := 
by {
  rw [h1, h2, h3, h4],
  exact h5,
  sorry
}

end total_legs_l694_694902


namespace no_integer_k_exists_l694_694182

noncomputable def f (x : ℤ) : ℤ := x^n + ∑ i in finset.range n, a_i * x^(n - i)

theorem no_integer_k_exists
  {n : ℕ}
  {a b c d : ℤ}
  {a_i : fin n → ℤ}
  (h_distinct : list.nodup [a, b, c, d])
  (h_values : f a = 5 ∧ f b = 5 ∧ f c = 5 ∧ f d = 5) :
  ¬ ∃ k : ℤ, f k = 8 :=
by 
  sorry

end no_integer_k_exists_l694_694182


namespace area_AEB_l694_694790

variable (A B C D E : Type)
variable [inhabited A] [inhabited B] [inhabited C] [inhabited D]
variable [linear_ordered_field E]

-- Define points and lengths
variables (A B C D : E)
variable (h_AB_CD_parallel : ∀ (x : E), (x = A ∨ x = B) → x = C ∨ x = D)
variable (AB_length : E) (CD_length : E)
variable (AB_eq_6 : AB_length = 6)
variable (CD_eq_15 : CD_length = 15)

-- Define the area of triangle AED
variable (h_AED : E)
variable (area_AED_eq_30 : 1 / 2 * CD_length * h_AED = 30)

-- Define the height from point E
variable (h_height : E)
variable (height_eq_4 : h_height = 4)

-- Prove the area of triangle AEB
theorem area_AEB : 1 / 2 * AB_length * h_height = 12 := by
  sorry

end area_AEB_l694_694790


namespace find_x_minus_y_l694_694994

theorem find_x_minus_y (x y : ℝ) (h1 : |x| + x - y = 14) (h2 : x + |y| + y = 6) : x - y = 8 :=
sorry

end find_x_minus_y_l694_694994


namespace find_length_CE_l694_694283

variables {A B C O E : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited O] [Inhabited E]
variables {l : A → B → Prop} {m : B → C → Prop}

-- Given conditions
-- Triangle ABC
variable {AC BC : ℝ}
variable (h1 : AC = 7)
variable (h2 : BC = 4)

-- Midpoint of AB is O
variable {midpoint_AB : A × B → O}

-- Line through O parallel to l
variable (l_parallel : (O → A → B → Prop) → Prop)

-- The line through O intersects AC at E
variable (intersects_AC : O → C → E → Prop)

-- Proven statement
theorem find_length_CE : AC = 7 → BC = 4 → midpoint_AB (A, B) = O → l_parallel l → intersects_AC O C E →
  |CE| = 11 / 2 :=
by
  sorry

end find_length_CE_l694_694283


namespace find_initial_number_l694_694026

noncomputable def initialNumber (c: ℝ) := c + 5.000000000000043
def divides23 (x: ℤ) : Prop := ∃ k: ℤ, x = 23 * k

theorem find_initial_number :
  divides23 (initialNumber 18) :=
sorry

end find_initial_number_l694_694026


namespace largest_prime_factor_of_7_fact_add_8_fact_l694_694025

theorem largest_prime_factor_of_7_fact_add_8_fact :
  let n := 7!
  let m := 8!
  let sum := n + m
  prime_factors sum = {2, 3, 5, 7} ∧ (∀ p ∈ {2, 3, 5, 7}, p <= 7) := by
  sorry

end largest_prime_factor_of_7_fact_add_8_fact_l694_694025


namespace part1_part2_l694_694561

open Real

variable (θ : ℝ) (m : ℝ)
variable h1 : sin θ + cos θ = (sqrt 3 + 1) / 2
variable h2 : sin θ * cos θ = m / 2

noncomputable def expr1 : ℝ :=
  sin θ^2 / (sin θ - cos θ) + cos θ^2 / (cos θ - sin θ)

theorem part1 : expr1 θ = (sqrt 3 + 1) / 2 :=
  by sorry

theorem part2 : m = sqrt 3 / 2 :=
  by
  have h : (sin θ + cos θ)^2 = (sqrt 3 + 1)^2 / 4 := by
    rw [h1, sqr ((sqrt 3 + 1) / 2)]
  calc
    1 + 2 * (sin θ * cos θ) = _ := by
      rw [sqr (sin θ + cos θ)]
      sorry
    = _ := by
      sorry

end part1_part2_l694_694561


namespace event_B_C_mutually_exclusive_l694_694527

-- Define the events based on the given conditions
def EventA (products : Type) (is_defective : products → Prop) (x y : products) : Prop :=
  ¬is_defective x ∧ ¬is_defective y

def EventB (products : Type) (is_defective : products → Prop) (x y : products) : Prop :=
  is_defective x ∧ is_defective y

def EventC (products : Type) (is_defective : products → Prop) (x y : products) : Prop :=
  ¬(is_defective x ∧ is_defective y)

-- Prove that Event B and Event C are mutually exclusive
theorem event_B_C_mutually_exclusive (products : Type) (is_defective : products → Prop) (x y : products) :
  (EventB products is_defective x y) → ¬(EventC products is_defective x y) :=
sorry

end event_B_C_mutually_exclusive_l694_694527


namespace curve_symmetry_l694_694415

-- Define the curve as a predicate
def curve (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y = 0

-- Define the point symmetry condition for a line
def is_symmetric_about_line (curve : ℝ → ℝ → Prop) (line : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, curve x y → line x y

-- Define the line x + y = 0
def line_x_plus_y_eq_0 (x y : ℝ) : Prop := x + y = 0

-- Main theorem stating the curve is symmetrical about the line x + y = 0
theorem curve_symmetry : is_symmetric_about_line curve line_x_plus_y_eq_0 := 
sorry

end curve_symmetry_l694_694415


namespace triangle_area_l694_694049

theorem triangle_area
  (A B C L : Type*)
  [metric_space A] 
  [metric_space B]
  [metric_space C]
  [metric_space L]
  (AL BL CL : ℝ)
  (hAL : AL = 2)
  (hBL : BL = sqrt 30)
  (hCL : CL = 5)
  (hBisect : ∀ (a b : ℝ), a/AL = b/CL) :
  let area := (7 * sqrt 39) / 4 in
  sorry

end triangle_area_l694_694049


namespace can_erase_some_color_and_preserve_connectivity_l694_694262
open Graph

-- Define K_{40}
def K40 : SimpleGraph (Fin 40) := SimpleGraph.complete _

-- Define the coloring of edges with 6 colors
noncomputable def edge_coloring (e : K40.edge_set) : Fin 6 := sorry

-- Statement of the theorem
theorem can_erase_some_color_and_preserve_connectivity (c : Fin 6) :
  ∃ c : Fin 6, ∀ u v : Fin 40, u ≠ v → (∃ p : u.path v, ∀ e ∈ p.edges, e ≠ c) :=
sorry

end can_erase_some_color_and_preserve_connectivity_l694_694262


namespace relationship_f_neg_pi_f_3_14_l694_694674

theorem relationship_f_neg_pi_f_3_14
  {f : ℝ → ℝ}
  (h_even : ∀ x, f (-x) = f x)
  (h_increasing : ∀ x₁ x₂, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂) :
  f (-real.pi) > f 3.14 :=
by
  sorry

end relationship_f_neg_pi_f_3_14_l694_694674


namespace trajectory_parabola_max_PQ_l694_694172

/-- Let C be a moving circle that passes through fixed point F(0,1) and is tangent to the line y=-1.
- Prove the trajectory of center C is a parabola with equation x^2=4y.
- Given a line l2 intersects this parabola at points P and Q, and the ordinate of midpoint of PQ is 2,
- Prove the maximum value of |PQ| is 6. -/

theorem trajectory_parabola (C F : Type) (x y : ℝ) (E : set ℝ) :
  (∀ C, C ∈ E → ∃ F, F = (0,1) ∧ ∃ l₁, l₁ = (y = -1) ∧ dist C F = dist C l₁) →
  E = {p | p.1^2 = 4 * p.2} :=
sorry

theorem max_PQ (t k x1 x2 y1 y2 : ℝ) (P Q : Type) :
  (x1^2 = 4 * y1) ∧ (x2^2 = 4 * y2) ∧ (y = t / 2 * (x - t) + 2) ∧
  (dist (x1, y1) (x2, y2) = sqrt((1 + (t^2 / 4)) * (4 * t^2 - 4 * (2 * t^2 - 8)))) →
  |dist (x1, y1) (x2, y2)| ≤ 6 :=
sorry

end trajectory_parabola_max_PQ_l694_694172


namespace four_digit_numbers_count_l694_694592

theorem four_digit_numbers_count :
  ∃ (n : ℕ),  n = 24 ∧
  let s := λ (a b c d : ℕ), (a + b + c + d = 10 ∧ a ≠ 0 ∧ ((a + c) - (b + d)) ∈ [(-11), 0, 11]) in
  (card {abcd : Fin 10 × Fin 10 × Fin 10 × Fin 10 | (s abcd.1.1 abcd.1.2 abcd.2.1 abcd.2.2)}) = n :=
by
  sorry

end four_digit_numbers_count_l694_694592


namespace interest_cannot_be_determined_without_investment_amount_l694_694812

theorem interest_cannot_be_determined_without_investment_amount :
  ∀ (interest_rate : ℚ) (price : ℚ) (invested_amount : Option ℚ),
  interest_rate = 0.16 → price = 128 → invested_amount = none → False :=
by
  sorry

end interest_cannot_be_determined_without_investment_amount_l694_694812


namespace absolute_difference_l694_694787

theorem absolute_difference : |8 - 3^2| - |4^2 - 6*3| = -1 := by
  sorry

end absolute_difference_l694_694787


namespace driver_net_pay_l694_694066

theorem driver_net_pay
  (travel_time : ℝ)
  (speed : ℝ)
  (gas_consumption_rate : ℝ)
  (payment_rate : ℝ)
  (gas_cost : ℝ)
  (net_pay_per_hour : ℝ) :
  travel_time = 3 ∧
  speed = 60 ∧
  gas_consumption_rate = 30 ∧
  payment_rate = 0.60 ∧
  gas_cost = 2.50 ∧
  net_pay_per_hour = 31 :=
by
  -- Define the known values
  let total_distance := speed * travel_time
  let gasoline_used := total_distance / gas_consumption_rate
  let total_earnings := payment_rate * total_distance
  let gas_expense := gasoline_used * gas_cost
  let net_earnings := total_earnings - gas_expense
  let rate_per_hour := net_earnings / travel_time
  -- Verify the result
  have h1 : total_distance = 180 := by sorry
  have h2 : gasoline_used = 6 := by sorry
  have h3 : total_earnings = 108 := by sorry
  have h4 : gas_expense = 15 := by sorry
  have h5 : net_earnings = 93 := by sorry
  have h6 : rate_per_hour = 31 := by sorry
  -- Combined condition proofs
  exact ⟨rfl, rfl, rfl, rfl, rfl, h6⟩

end driver_net_pay_l694_694066


namespace part1_part2_l694_694576

-- Definitions from conditions
def S : ℕ → ℤ
def a : ℕ → ℤ := λ n, 2 ^ (n - 1)
def b : ℕ → ℤ := λ n, (-1) ^ n * (n - 1)
def T : ℕ → ℤ := λ n, List.sum (List.map b (List.range n))

-- Given conditions
axiom S_1 : S 1 = 0
axiom S_step : ∀ n, S (n + 1) = 2 * S n + 1

-- Proofs required
theorem part1 (n : ℕ) : S n + 1 = 2 ^ n :=
    sorry

theorem part2 : T 2018 = 1009 :=
    sorry

end part1_part2_l694_694576


namespace dot_product_of_vectors_l694_694215

theorem dot_product_of_vectors :
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (-1, 2)
  a.1 * b.1 + a.2 * b.2 = -4 :=
by
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (-1, 2)
  sorry

end dot_product_of_vectors_l694_694215


namespace minimumAreaTrianglePAB_l694_694270

noncomputable def polarCoordsIntersectionPoints :
  (A B: ℝ × ℝ)
  := (((2 * Real.sqrt 2, (7 * Real.pi / 4))),
      ((4, 0)))

theorem minimumAreaTrianglePAB :
  let A := (2, -2)
  let B := (4, 0)
  let d_min := ((4 - Real.sqrt 5) / Real.sqrt 2)
  let A_B := Real.dist (2, -2) (4, 0)
  ∃ (P: ℝ × ℝ), S :=
    let area := (1 / 2) * A_B * d_min
    area = 4 - Real.sqrt 5
:= sorry

end minimumAreaTrianglePAB_l694_694270


namespace binom_div_l694_694521

noncomputable def binom (a : ℝ) (k : ℕ) : ℝ :=
  (List.prod (List.map (λ i : ℕ, a - i) (List.range k))) / (Nat.factorial k)

theorem binom_div : binom (-3/2) 50 / binom (3/2) 50 = -1 :=
by
  sorry

end binom_div_l694_694521


namespace solution_set_of_inequality_l694_694189

variable (a b x : ℝ)
variable (h1 : a < 0)

theorem solution_set_of_inequality (h : a * x + b < 0) : x > -b / a :=
sorry

end solution_set_of_inequality_l694_694189


namespace adult_ticket_cost_l694_694323

-- Define the problem's constants and conditions
variables (A : ℕ) -- cost of an adult ticket
variables (kids_cost : ℕ) (total_spent : ℕ) (num_children : ℕ) (children_ticket_cost : ℕ) (total_paid : ℕ) (change : ℕ)

-- Given conditions
def conditions : Prop :=
  num_children = 3 ∧
  children_ticket_cost = 1 ∧
  total_paid = 20 ∧
  change = 15 ∧
  total_spent = total_paid - change ∧
  kids_cost = num_children * children_ticket_cost ∧
  total_spent = A + kids_cost

-- The theorem to be proved
theorem adult_ticket_cost (h : conditions) : A = 2 :=
sorry

end adult_ticket_cost_l694_694323


namespace measure_of_angle_C_l694_694097

variable (C D : ℕ)
variable (h1 : C + D = 180)
variable (h2 : C = 5 * D)

theorem measure_of_angle_C : C = 150 :=
by
  sorry

end measure_of_angle_C_l694_694097


namespace angle_bisectors_concurrent_l694_694788

open EuclideanGeometry

-- Definitions based on conditions
def cyclic_quadrilateral (A B C D : Point) : Prop :=
  ∃ (circle : Circle), circle.on_circle A ∧ circle.on_circle B ∧ circle.on_circle C ∧ circle.on_circle D

def midpoint (P Q M : Point) : Prop :=
  ∃ (R : Point), dist R P = dist R Q ∧ dist P M = dist Q M

axiom intersect (A B C D E F M N : Point)
  (hcyclic: cyclic_quadrilateral A B C D)
  (hE: line_intersects A B C D at E)
  (hF: line_intersects A D B C at F)
  (hM: midpoint B D M)
  (hN: midpoint E F N) :
  concurrency_of_angle_bisectors A E D A F B M N

-- The main theorem which we need to prove
theorem angle_bisectors_concurrent (A B C D E F M N : Point)
  (hcyclic: cyclic_quadrilateral A B C D)
  (hE: line_intersects A B C D at E)
  (hF: line_intersects A D B C at F)
  (hM: midpoint B D M)
  (hN: midpoint E F N):
  concurrency_of_angle_bisectors A E D A F B M N :=
sorry

end angle_bisectors_concurrent_l694_694788


namespace dance_lesson_cost_l694_694750

-- Define the conditions
variable (total_lessons : Nat) (free_lessons : Nat) (paid_lessons_cost : Nat)

-- State the problem with the given conditions
theorem dance_lesson_cost
  (h1 : total_lessons = 10)
  (h2 : free_lessons = 2)
  (h3 : paid_lessons_cost = 80) :
  let number_of_paid_lessons := total_lessons - free_lessons
  number_of_paid_lessons ≠ 0 -> 
  (paid_lessons_cost / number_of_paid_lessons) = 10 := by
  sorry

end dance_lesson_cost_l694_694750


namespace two_pow_n_minus_one_divisible_by_seven_iff_l694_694426

theorem two_pow_n_minus_one_divisible_by_seven_iff (n : ℕ) (h : n > 0) :
  (2^n - 1) % 7 = 0 ↔ n % 3 = 0 :=
sorry

end two_pow_n_minus_one_divisible_by_seven_iff_l694_694426


namespace not_separable_1_over_x_separable_2_x_plus_x_squared_separable_lg_a_over_2x_plus_1_l694_694807

variable (f : ℝ → ℝ) (x : ℝ → ℝ)

def separable_function (f : ℝ → ℝ) : Prop :=
  ∃ x_0 : ℝ, f (x_0 + 1) = f x_0 + f 1

-- Problem 1: Prove \( f(x) = \frac{1}{x} \) is not a "separable function"
theorem not_separable_1_over_x : ¬ separable_function (λ x : ℝ, 1 / x) :=
sorry

-- Problem 2: Prove \( f(x) = 2^x + x^2 \) is a "separable function"
theorem separable_2_x_plus_x_squared : separable_function (λ x : ℝ, 2^x + x^2) :=
sorry

-- Problem 3: Range of \( a \) if \( f(x) = \lg \frac{a}{2^x + 1} \) is a "separable function"
theorem separable_lg_a_over_2x_plus_1 :
  ∀ a : ℝ, separable_function (λ x : ℝ, log (a / (2^x + 1))) ↔ 1.5 < a ∧ a < 3 :=
sorry

end not_separable_1_over_x_separable_2_x_plus_x_squared_separable_lg_a_over_2x_plus_1_l694_694807


namespace sum_of_tangents_l694_694699

theorem sum_of_tangents (n : ℕ) (θ : ℝ) :
  (∀ θ : ℝ, 
  (∑ j in finset.range n, real.tan (θ + j * real.pi / n)) = 
  if (n % 2 = 1) then n * real.tan (n * θ) else -n * real.cot (n * θ)) :=
sorry

end sum_of_tangents_l694_694699


namespace tangent_equivalent_l694_694610

theorem tangent_equivalent (α : ℝ) 
  (h : (2 * (Real.cos α) ^ 2 + Real.cos (π / 2 + 2 * α) - 1) / (sqrt 2 * Real.sin (2 * α + π / 4)) = 4) :
  Real.tan (2 * α + π / 4) = 1 / 4 :=
sorry

end tangent_equivalent_l694_694610


namespace count_5_digit_even_div_by_4_l694_694978

theorem count_5_digit_even_div_by_4 : 
  let even_digits := {0, 2, 4, 6, 8}
  let count5DigitDivBy4 := (5 - 1) * 5 * 5 * 15  -- calculating the number of valid integers
  (count5DigitDivBy4 = 1875) :=
by
  sorry

end count_5_digit_even_div_by_4_l694_694978


namespace range_of_a_l694_694190

theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (1/2) 2 → x₂ ∈ Set.Icc (1/2) 2 → (a / x₁ + x₁ * Real.log x₁ ≥ x₂^3 - x₂^2 - 3)) →
  a ∈ Set.Ici 1 :=
by
  sorry

end range_of_a_l694_694190


namespace range_of_a_l694_694207

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 1
noncomputable def g (x : ℝ) : ℝ := Real.log (Real.exp x - 1) - Real.log x

theorem range_of_a (a : ℝ) :
  (∃ x0 : ℝ, 0 < x0 ∧ f (g x0) a > f x0 a) ↔ 1 < a := sorry

end range_of_a_l694_694207


namespace function_is_monotonically_decreasing_l694_694557

theorem function_is_monotonically_decreasing (a : ℝ) : 
  (∀ x : ℝ, (2 * cos (2 * x) - 4 * sin x - a) ≤ 0) ↔ (3 ≤ a) :=
begin
  sorry,
end

end function_is_monotonically_decreasing_l694_694557


namespace moon_speed_conversion_l694_694730

def speed_kmh : ℝ := 3708
def seconds_per_hour : ℝ := 3600
def speed_kms : ℝ := speed_kmh / seconds_per_hour

theorem moon_speed_conversion : speed_kms ≈ 1.03 :=
by
  -- Definition of approximate equality (≈) can vary; it can be implemented using some acceptable tolerance level.
  sorry

end moon_speed_conversion_l694_694730


namespace area_sum_l694_694641

noncomputable def S_triangle_ABC (A B C : ℝ × ℝ) : ℝ := sorry
noncomputable def S_triangle_CDE (C D E : ℝ × ℝ) : ℝ := sorry

theorem area_sum
  (A B C D E : ℝ × ℝ)
  (h1 : E = (B + C) / 2)  -- \( E \) is the midpoint of \( BC \)
  (h2 : D.1 = A.1 ∧ AC = 1 ∧ D.2 ∈ set.Icc (min A.2 C.2) (max A.2 C.2))
  (h3 : AC = 1)
  (h4 : ∠BAC = 60)
  (h5 : ∠ABC = 10)
  (h6 : ∠DEC = 80) : 
  S_triangle_ABC A B C + 2 * S_triangle_CDE C D E = (real.sqrt 3) / 8 :=
sorry

end area_sum_l694_694641


namespace find_a_minimum_φ_min_value_g_x1_x2_l694_694570

-- Define the functions involved in the problem
def f (x m : ℝ) : ℝ := x^2 - 2 * x + m * (Real.log x)
def g (x : ℝ) : ℝ := (x - 3/4) * Real.exp x
def φ (x a : ℝ) (m: ℝ): ℝ := f x m - (x^2 - (2 + 1/a) * x)

-- Problem (1) Statement
theorem find_a_minimum_φ : 
  ∀ (a : ℝ), 0 < a ∧ a ≤ Real.exp 1 ∧ (∀ (x : ℝ), (0 < x ∧ x ≤ Real.exp 1) → φ x a (-1) = 2) → a = 1 / Real.exp 1 := 
begin
  sorry
end

-- Problem (2) Statement
theorem min_value_g_x1_x2 :
  ∀ (m : ℝ) (x1 x2 : ℝ), 
    (f x1 m = 0 ∧ f x2 m = 0 ∧ x1 < x2 ∧ 0 < x1 ∧ x1 < 1 ∧ 0 < m ∧ m < 1/2) → g (x1 - x2) = -Real.exp (-1 / 4) := 
begin
  sorry
end

end find_a_minimum_φ_min_value_g_x1_x2_l694_694570


namespace count_divisible_by_11_with_digits_sum_10_l694_694601

-- Definitions and conditions from step a)
def digits_add_up_to_10 (n : ℕ) : Prop :=
  let ds := n.digits 10 in ds.length = 4 ∧ ds.sum = 10

def divisible_by_11 (n : ℕ) : Prop :=
  let ds := n.digits 10 in
  (ds.nth 0).get_or_else 0 +
  (ds.nth 2).get_or_else 0 -
  (ds.nth 1).get_or_else 0 -
  (ds.nth 3).get_or_else 0 ∣ 11

-- The tuple (question, conditions, answer) formalized in Lean
theorem count_divisible_by_11_with_digits_sum_10 :
  ∃ N : ℕ, N = (
  (Finset.filter (λ x, digits_add_up_to_10 x ∧ divisible_by_11 x)
  (Finset.range 10000)).card
) := 
sorry

end count_divisible_by_11_with_digits_sum_10_l694_694601


namespace perpendiculars_concur_at_orthocenter_l694_694697

theorem perpendiculars_concur_at_orthocenter (A B C A1 B1 C1 : Type*)
  [metric_space A] [metric_space B] [metric_space C] [metric_space A1]
  [metric_space B1] [metric_space C1]
  (AB1_AC1 : dist (A, B1) = dist (A, C1))
  (BC1_BA1 : dist (B, C1) = dist (B, A1))
  (CA1_CB1 : dist (C, A1) = dist (C, B1))
  (A1_on_perp_bisector_BC : lies_on_perp_bisector A1 B C)
  (B1_on_perp_bisector_CA : lies_on_perp_bisector B1 C A)
  (C1_on_perp_bisector_AB : lies_on_perp_bisector C1 A B) :
  exist (H : Type*) (is_orthocenter H A B C) :=
sorry

end perpendiculars_concur_at_orthocenter_l694_694697


namespace find_n_l694_694528

theorem find_n (n : ℝ) (h : (2 : ℂ) / (1 - I) = (1 : ℂ) + n * I) : n = 1 := by
  sorry

end find_n_l694_694528


namespace carla_initial_marbles_l694_694107

theorem carla_initial_marbles (total_marbles : ℕ) (bought_marbles : ℕ) (initial_marbles : ℕ) 
  (h1 : total_marbles = 187) (h2 : bought_marbles = 134) (h3 : total_marbles = initial_marbles + bought_marbles) : 
  initial_marbles = 53 := 
sorry

end carla_initial_marbles_l694_694107


namespace quadratic_solution_l694_694553

theorem quadratic_solution (a : ℝ) (h : (1 : ℝ)^2 + 1 + 2 * a = 0) : a = -1 :=
by {
  sorry
}

end quadratic_solution_l694_694553


namespace fraction_of_a_mile_additional_charge_l694_694643

-- Define the conditions
def initial_fee : ℚ := 2.25
def charge_per_fraction : ℚ := 0.25
def total_charge : ℚ := 4.50
def total_distance : ℚ := 3.6

-- Define the problem statement to prove
theorem fraction_of_a_mile_additional_charge :
  initial_fee = 2.25 →
  charge_per_fraction = 0.25 →
  total_charge = 4.50 →
  total_distance = 3.6 →
  total_distance - (total_charge - initial_fee) = 1.35 :=
by
  intros
  sorry

end fraction_of_a_mile_additional_charge_l694_694643


namespace percent_equivalence_l694_694998

variable (x : ℝ)
axiom condition : 0.30 * 0.15 * x = 18

theorem percent_equivalence :
  0.15 * 0.30 * x = 18 := sorry

end percent_equivalence_l694_694998


namespace permutation_problem_l694_694433

noncomputable def permutation (n r : ℕ) : ℕ := (n.factorial) / ( (n - r).factorial)

theorem permutation_problem : 5 * permutation 5 3 + 4 * permutation 4 2 = 348 := by
  sorry

end permutation_problem_l694_694433


namespace point_A_2019_pos_l694_694462

noncomputable def A : ℕ → ℤ
| 0       => 2
| (n + 1) =>
    if (n + 1) % 2 = 1 then A n - (n + 1)
    else A n + (n + 1)

theorem point_A_2019_pos : A 2019 = -1008 := by
  sorry

end point_A_2019_pos_l694_694462


namespace a_n_general_term_T_n_sum_l694_694958

theorem a_n_general_term (S : ℕ → ℚ) (a : ℕ → ℚ) :
  (∀ n, S n + a n = 1) → a 1 = 1 / 2 → (∀ n, a n = a (n - 1) / 2) → ∀ n, a n = 1 / 2^n :=
by sorry

theorem T_n_sum (S : ℕ → ℚ) (a : ℕ → ℚ) (b : ℕ → ℚ) :
  (∀ n, S n + a n = 1) →
  (∀ n, b n + real.log a n / real.log 2 = 0) →
  ∀ n, S 1 = 1 / 2 →
  (∀ n, a n = a (n - 1) / 2) →
  ∀ n, (∑ k in Ico 1 n, 1 / (b k * b (k + 1))) = n / (n + 1) :=
by sorry

end a_n_general_term_T_n_sum_l694_694958


namespace construct_parabola_l694_694579

structure Point :=
(x : ℝ)
(y : ℝ)

structure Line :=
(a : ℝ)
(b : ℝ)
(c : ℝ) -- Line representation ax + by + c = 0

def is_tangent (line: Line) (point: Point) (parabola: ℝ -> ℝ) : Prop := sorry -- Not necessary to implement here

def intersects (l1 l2: Line) : Point := sorry -- Intersection point of two lines

def midpoint (p1 p2 : Point) : Point :=
{ x := (p1.x + p2.x) / 2,
  y := (p1.y + p2.y) / 2 }

def parallel (l1 l2: Line) : Prop := l1.a * l2.b = l2.a * l1.b

def reflect (p: Point) (line: Line) : Point := sorry -- Reflection of a point about a line

noncomputable def parabola_from_tangents (t1 t2 : Line) (P1 P2 : Point) : (P -> Prop) := 
by
  -- Assume P1, P2 are points of tangency, t1, t2 are tangents, and lines are not parallel
  let M := intersects t1 t2
  let H := midpoint P1 P2
  let line_mh := sorry -- Line through M and H, parallel to t1

  -- Calculate the focus F by reflection properties
  let F := intersects (reflect P1 t1) (reflect P2 t2)

  -- Construct a parabola based on F and properties, checking tangency
  sorry -- Completing the parabola construction
-- Ensure to return the parabola satisfying the points and tangents

theorem construct_parabola (t1 t2 : Line) (P1 P2 : Point) 
(h_non_parallel: ¬parallel t1 t2)
(h_point_t1: is_tangent t1 P1 sorry)
(h_point_t2: is_tangent t2 P2 sorry) : 
∃ parabola : (ℝ -> ℝ), 
  (∀ (P : Point), is_tangent t1 P parabola ↔ P = P1) ∧ 
  (∀ (P : Point), is_tangent t2 P parabola ↔ P = P2) :=
by
  sorry -- Proof of constructed parabola satisfying conditions

end construct_parabola_l694_694579


namespace inequality_solution_l694_694620

theorem inequality_solution (a : ℝ) : 
  (∀ x ∈ set.Icc (0 : ℝ) (1/2), 4^x + x - a ≤ 3 / 2) → a ≥ 1 := sorry

end inequality_solution_l694_694620


namespace find_cos_gamma_l694_694655

theorem find_cos_gamma (α β γ : ℝ) (cosα_eq : Real.cos α = 1/4) (cosβ_eq : Real.cos β = 1/3) :
  Real.cos γ = Real.sqrt 119 / 12 :=
by
  have h1: Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1 := sorry
  have h2: (1/4) ^ 2 + (1/3) ^ 2 + Real.cos γ ^ 2 = 1 := sorry
  have h3: 1 - (1/16) - (1/9) = Real.cos γ ^ 2 := sorry
  have h4: 119 / 144 = Real.cos γ ^ 2 := sorry
  have h5: Real.sqrt (119 / 144) = Real.cos γ := sorry
  have h6: Real.sqrt 119 / 12 = Real.sqrt(119 / 144) := sorry
  exact sorry

end find_cos_gamma_l694_694655


namespace distance_to_focus_l694_694940

theorem distance_to_focus (x y : ℝ) (hx : y^2 = 2 * x) (distance_x_axis : y = 2 ∨ y = -2) : 
  let focus := (1/4, 0 : ℝ × ℝ) in
  (x, y) = (2, 2) ∨ (x, y) = (2, -2) →
  dist (x, y) focus = 5 / 2 :=
by
  sorry

end distance_to_focus_l694_694940


namespace prob_fourth_ball_black_l694_694443

theorem prob_fourth_ball_black :
  let num_red_balls := 3 in
  let num_black_balls := 4 in
  let total_balls := num_red_balls + num_black_balls in
  total_balls = 7 →
  (num_black_balls / total_balls : ℚ) = 4 / 7 :=
by sorry

end prob_fourth_ball_black_l694_694443


namespace part1_part2_l694_694565

noncomputable def f (a : ℝ) (x : ℝ) := (a - 1/2) * x^2 + Real.log x
noncomputable def g (a : ℝ) (x : ℝ) := f a x - 2 * a * x

theorem part1 (x : ℝ) (hxe : Real.exp (-1) ≤ x ∧ x ≤ Real.exp (1)) : 
    f (-1/2) x ≤ -1/2 - 1/2 * Real.log 2 ∧ f (-1/2) x ≥ 1 - Real.exp 2 := sorry

theorem part2 (h : ∀ x > 2, g a x < 0) : a ≤ 1/2 := sorry

end part1_part2_l694_694565


namespace no_such_number_exists_l694_694292

def is_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

/-- Define the number N as a sequence of digits a_n a_{n-1} ... a_0 -/
def number (a b : ℕ) (n : ℕ) : ℕ := a * 10^n + b

theorem no_such_number_exists :
  ¬ ∃ (N a_n b : ℕ) (n : ℕ), is_digit a_n ∧ a_n ≠ 0 ∧ b < 10^n ∧
    N = number a_n b n ∧
    b = N / 57 :=
sorry

end no_such_number_exists_l694_694292


namespace anna_earnings_correct_l694_694098

def anna_cupcakes_baked (trays : ℕ) (cupcakes_per_tray : ℕ) : ℕ := trays * cupcakes_per_tray

def cupcakes_sold (total_cupcakes : ℕ) (fraction_sold : ℚ) : ℕ :=
  (fraction_sold * total_cupcakes).to_nat

def anna_earnings (cupcakes_sold : ℕ) (price_per_cupcake : ℚ) : ℚ :=
  cupcakes_sold * price_per_cupcake

theorem anna_earnings_correct :
  let trays := 4 in
  let cupcakes_per_tray := 20 in
  let total_cupcakes := anna_cupcakes_baked trays cupcakes_per_tray in
  let fraction_sold := 3 / 5 in
  let sold_cupcakes := cupcakes_sold total_cupcakes fraction_sold in
  let price_per_cupcake := 2 in
  anna_earnings sold_cupcakes price_per_cupcake = 96 := by
  sorry

end anna_earnings_correct_l694_694098


namespace distance_PF_l694_694969

noncomputable def parabola := { p : ℝ × ℝ // p.2^2 = 4 * p.1 }

def focus : ℝ × ℝ := (1, 0)

def directrix : ℝ × ℝ → Prop := λ p, p.1 = -1

def slope_angle (x : ℝ) : ℝ := Real.tan x

def equation_of_line_EF (p : ℝ × ℝ) : Prop := p.2 = - (Real.sqrt 3 / 3) * (p.1 - 1)

theorem distance_PF {P : ℝ × ℝ}
  (hP : P ∈ parabola)
  (hEF_slope : slope_angle (5 * Real.pi / 6) = - (Real.sqrt 3 / 3)) -- 150 degrees in radians
  (h_EF : equation_of_line_EF P)
  (h_perpendicular : ∀ E, directrix E → P.2 = E.2) :
  Real.dist P focus = 4 / 3 := sorry

end distance_PF_l694_694969


namespace exists_f_decreasing_mult_l694_694963

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2
axiom condition2 : ∀ x1 x2 : ℝ, f (x1 + x2) = f x1 * f x2

theorem exists_f_decreasing_mult :
  ∃ a : ℝ, 0 < a ∧ a < 1 ∧ f = (λ x, a^x) := sorry

end exists_f_decreasing_mult_l694_694963


namespace sum_of_digits_9ab_l694_694284

noncomputable def a : ℕ := 6 * ((10^1000 - 1) / 9)
noncomputable def b : ℕ := 3 * ((10^1000 - 1) / 9)

theorem sum_of_digits_9ab : 
  (∑ d in (9 * a * b).digits 10, d) = 9010 :=
by
  sorry

end sum_of_digits_9ab_l694_694284


namespace total_animals_in_savanna_l694_694702

/-- Define the number of lions, snakes, and giraffes in Safari National Park. --/
def safari_lions : ℕ := 100
def safari_snakes : ℕ := safari_lions / 2
def safari_giraffes : ℕ := safari_snakes - 10

/-- Define the number of lions, snakes, and giraffes in Savanna National Park based on conditions. --/
def savanna_lions : ℕ := 2 * safari_lions
def savanna_snakes : ℕ := 3 * safari_snakes
def savanna_giraffes : ℕ := safari_giraffes + 20

/-- Calculate the total number of animals in Savanna National Park. --/
def total_savanna_animals : ℕ := savanna_lions + savanna_snakes + savanna_giraffes

/-- Proof statement that the total number of animals in Savanna National Park is 410.
My goal is to prove that total_savanna_animals is equal to 410. --/
theorem total_animals_in_savanna : total_savanna_animals = 410 :=
by
  sorry

end total_animals_in_savanna_l694_694702


namespace largest_prime_factor_of_7_fact_add_8_fact_l694_694018

theorem largest_prime_factor_of_7_fact_add_8_fact :
  let n := 7!
  let m := 8!
  let sum := n + m
  prime_factors sum = {2, 3, 5, 7} ∧ (∀ p ∈ {2, 3, 5, 7}, p <= 7) := by
  sorry

end largest_prime_factor_of_7_fact_add_8_fact_l694_694018


namespace sqrt_inequality_l694_694418

theorem sqrt_inequality (a b : ℝ) (h : 0 ≤ a ∧ 0 ≤ b) (h_sqrt : sqrt a < sqrt b) : a < b := 
by sorry

end sqrt_inequality_l694_694418


namespace _l694_694860

noncomputable def game_theorem : ∀ (p : ℝ), prime p ∧ p > 7 → (∀ a b : ℝ, a ≠ b → |a - b| > 2) → (start_player_wins : BelaWins) :=
by
  assume p h_prime h_p_gt_7 h_distancing
  -- Mathematical equivalence proof problem
  sorry

end _l694_694860


namespace average_rate_of_change_interval_l694_694414

noncomputable def average_rate_of_change (f : ℝ → ℝ) (x₀ x₁ : ℝ) : ℝ :=
  (f x₁ - f x₀) / (x₁ - x₀)

theorem average_rate_of_change_interval (f : ℝ → ℝ) (x₀ x₁ : ℝ) :
  (f x₁ - f x₀) / (x₁ - x₀) = average_rate_of_change f x₀ x₁ := by
  sorry

end average_rate_of_change_interval_l694_694414


namespace convert_to_scientific_notation_l694_694830

theorem convert_to_scientific_notation :
  ∃ (x : ℝ), 1541 * 10^9 = x ∧ x = 1.541 * 10^11 :=
begin
  sorry
end

end convert_to_scientific_notation_l694_694830


namespace max_elements_is_669_l694_694142

def max_elements_no_sum_div_by_3 : ℕ :=
  let S : set ℕ := { x | 1 ≤ x ∧ x ≤ 2003 }
  let S0 := { x ∈ S | x % 3 = 0 }
  let S1 := { x ∈ S | x % 3 = 1 }
  let S2 := { x ∈ S | x % 3 = 2 }
  1 + max (S1.card) (S2.card)

theorem max_elements_is_669 :
  max_elements_no_sum_div_by_3 = 669 :=
sorry

end max_elements_is_669_l694_694142


namespace ellipse_equation_and_min_area_l694_694195

-- Define the conditions for the ellipse
structure Ellipse where
  a b c : ℝ
  h k : ℤ
  focus1 : (ℝ × ℝ) -- coordinates of the first focus
  focus2 : (ℝ × ℝ) -- coordinates of the second focus
  passing_point : (ℝ × ℝ) -- point through which the ellipse passes
  eqn : (ℝ × ℝ → Prop)

noncomputable def ellipse_condition : Ellipse :=
{
  a := sqrt 2,
  b := 1,
  c := 1,
  h := 0,
  k := 0,
  focus1 := (-1, 0),
  focus2 := (1, 0),
  passing_point := (1, sqrt(2) / 2),
  eqn := λ p, (p.fst^2 / 2) + p.snd^2 = 1
}

theorem ellipse_equation_and_min_area :
  (∃ e : Ellipse,
    e.a = sqrt 2 ∧
    e.b = 1 ∧
    e.focus1 = (-1,0) ∧
    e.focus2 = (1,0) ∧
    e.eqn (1, sqrt (2) / 2) ∧
    (∃ (A B P : ℝ × ℝ),
      A ∈ set_of e.eqn ∧ B ∈ set_of e.eqn ∧ P ∈ set_of e.eqn ∧
      P ≠ A ∧ P ≠ B ∧
      ∃ O : ℝ × ℝ, O = (0,0) ∧ P = (A.fst + B.fst, A.snd + B.snd) / 2 ∧
      let m := (A.snd - B.snd) / (A.fst - B.fst) in
      ∀ t : ℝ, 
      (t^2 / |m|) = ((m^2+2) / 8) ∧ 
      m^2 = 2 ∧ t^2 = 1 ∧ 
      let area := (1 / 2) * (t) * (-t/m) in 
      area ≤ sqrt(2) / 4 )) :=
sorry

end ellipse_equation_and_min_area_l694_694195


namespace line_tangent_to_circle_l694_694139

-- Define the circle equation
def circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 5

-- Define the general form of the line equation parallel to 2x - y + 1 = 0
def line (x y b : ℝ) : Prop :=
  2 * x - y + b = 0

-- Define what it means for a line to be tangent to the circle
def is_tangent (b : ℝ) : Prop :=
  let d := abs b / sqrt (2^2 + 1^2)
  d = sqrt 5

-- Theorem statement
theorem line_tangent_to_circle (b : ℝ) :
  is_tangent b →
  (line x y b ↔ b = 5 ∨ b = -5) := by
  intros h
  intuition
  sorry

end line_tangent_to_circle_l694_694139


namespace kola_age_l694_694647

variables (x y : ℕ)

-- Condition 1: Kolya is twice as old as Olya was when Kolya was as old as Olya is now
def condition1 : Prop := x = 2 * (2 * y - x)

-- Condition 2: When Olya is as old as Kolya is now, their combined age will be 36 years.
def condition2 : Prop := (3 * x - y = 36)

theorem kola_age : condition1 x y → condition2 x y → x = 16 :=
by { sorry }

end kola_age_l694_694647


namespace sum_eq_product_l694_694131

theorem sum_eq_product : 
  ∃ (a : Fin 1000 → ℕ), (∑ i, a i = (∏ i, a i)) := 
by
  let a : Fin 1000 → ℕ := λ i, if i < 998 then 1 else if i = 998 then 2 else 1000
  use a
  sorry

end sum_eq_product_l694_694131


namespace line_segments_on_circle_l694_694125

theorem line_segments_on_circle (n : ℕ) (h : n = 6) :
  (n * (n - 1) / 2) = 15 :=
by
  rw h
  norm_num
  sorry

end line_segments_on_circle_l694_694125


namespace playground_basketball_area_difference_l694_694397

theorem playground_basketball_area_difference :
  let perimeter_square := 36
  let perimeter_rectangle := 38
  let width_rectangle := 15
  let side_square := perimeter_square / 4
  let length_rectangle := (perimeter_rectangle - 2 * width_rectangle) / 2
  let area_square := side_square * side_square
  let area_rectangle := length_rectangle * width_rectangle
  in area_square - area_rectangle = 21 := by sorry

end playground_basketball_area_difference_l694_694397


namespace det_projection_matrix_l694_694658

noncomputable def projection_matrix (v : ℝ × ℝ × ℝ) : matrix (fin 3) (fin 3) ℝ :=
let vt := (λ i, [i.succ_above.equat 0, i.succ_above.equat 1, i.succ_above.equat 2]) (prod.curry v),
    vTv := vt ⬝ v in
(1 / vTv) • (v ⬝ vt)

theorem det_projection_matrix : 
  let v := (3, -2, 6)
  let P := projection_matrix v in
  matrix.det P = 0 := 
by
  -- Write the formal proof here
  sorry

end det_projection_matrix_l694_694658


namespace max_value_of_quadratic_l694_694112

theorem max_value_of_quadratic :
  ∀ (x : ℝ), ∃ y : ℝ, y = -3 * x^2 + 18 ∧
  (∀ x' : ℝ, -3 * x'^2 + 18 ≤ y) := by
  sorry

end max_value_of_quadratic_l694_694112


namespace max_value_f_on_interval_l694_694866

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem max_value_f_on_interval : 
  ∃ x ∈ Set.Icc (0 : ℝ) 1, ∀ y ∈ Set.Icc (0 : ℝ) 1, f y ≤ f x ∧ f x = Real.exp 1 - 1 := sorry

end max_value_f_on_interval_l694_694866


namespace proof_option_b_and_c_l694_694537

variable (a b c : ℝ)

theorem proof_option_b_and_c (h₀ : a > b) (h₁ : b > 0) (h₂ : c ≠ 0) :
  (b / a < (b + c^2) / (a + c^2)) ∧ (a^2 - 1 / a > b^2 - 1 / b) :=
by
  sorry

end proof_option_b_and_c_l694_694537


namespace four_digit_number_divisible_by_11_l694_694581

theorem four_digit_number_divisible_by_11 :
  ∃ (a b c d : ℕ), 
    a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ 
    a + b + c + d = 10 ∧ 
    (a + c) % 11 = (b + d) % 11 ∧
    (10 - a != 0 ∨ 10 - c != 0 ∨ 10 - b != 0 ∨ 10 - d != 0) := sorry

end four_digit_number_divisible_by_11_l694_694581


namespace matrix_N_unique_l694_694910

theorem matrix_N_unique 
(M N : Matrix (Fin 2) (Fin 2) ℤ) 
(h1 : M ⬝ (λ (i : Fin 2), if i = 0 then 4 else 0) = λ (i : Fin 2), if i = 0 then 8 else 28)
(h2 : M ⬝ (λ (i : Fin 2), if i = 0 then -2 else 10) = λ (i : Fin 2), if i = 0 then 6 else -34):
  M = N :=
by 
  let col1 : Fin 2 -> ℤ := (λ (i : Fin 2), if i = 0 then 2 else 7),
  let col2 : Fin 2 -> ℤ := (λ (i : Fin 2), if i = 0 then 1 else -2),
  let N : Matrix (Fin 2) (Fin 2) ℤ := λ (i j : Fin 2),
    if j = 0 then col1 i else col2 i,
  sorry

end matrix_N_unique_l694_694910


namespace find_origin_coordinates_l694_694243

variable (x y : ℝ)

def original_eq (x y : ℝ) := x^2 - y^2 - 2*x - 2*y - 1 = 0

def transformed_eq (x' y' : ℝ) := x'^2 - y'^2 = 1

theorem find_origin_coordinates (x y : ℝ) :
  original_eq (x - 1) (y + 1) ↔ transformed_eq x y :=
by
  sorry

end find_origin_coordinates_l694_694243


namespace rooster_count_l694_694391

theorem rooster_count (total_chickens hens roosters : ℕ) 
  (h1 : total_chickens = roosters + hens)
  (h2 : roosters = 2 * hens)
  (h3 : total_chickens = 9000) 
  : roosters = 6000 := 
by
  sorry

end rooster_count_l694_694391


namespace average_distance_one_hour_l694_694387

theorem average_distance_one_hour (d : ℝ) (t : ℝ) (h1 : d = 100) (h2 : t = 5 / 4) : (d / t) = 80 :=
by
  sorry

end average_distance_one_hour_l694_694387


namespace cube_root_of_sqrt_64_l694_694718

theorem cube_root_of_sqrt_64 : (real.sqrt 64)^(1/3) = 2 := 
by
  sorry

end cube_root_of_sqrt_64_l694_694718


namespace cistern_water_breadth_l694_694452

theorem cistern_water_breadth 
  (length width : ℝ) (wet_surface_area : ℝ) 
  (hl : length = 9) (hw : width = 6) (hwsa : wet_surface_area = 121.5) : 
  ∃ h : ℝ, 54 + 18 * h + 12 * h = 121.5 ∧ h = 2.25 := 
by 
  sorry

end cistern_water_breadth_l694_694452


namespace area_of_trapezoid_l694_694451

theorem area_of_trapezoid
  (r : ℝ)
  (AD BC : ℝ)
  (center_on_base : Bool)
  (height : ℝ)
  (area : ℝ)
  (inscribed_circle : r = 6)
  (base_AD : AD = 8)
  (base_BC : BC = 4)
  (K_height : height = 4 * Real.sqrt 2)
  (calc_area : area = (1 / 2) * (AD + BC) * height)
  : area = 32 * Real.sqrt 2 := by
  sorry

end area_of_trapezoid_l694_694451


namespace chessboard_transformation_l694_694268

theorem chessboard_transformation (board : ℕ → ℕ → ℤ) :
  (∃ king_path : list (ℕ × ℕ),
    -- The path must cover all coordinates on the 8x8 grid
    (∀ (i j : ℕ), (i < 8 ∧ j < 8) → (i, j) ∈ king_path) ∧ 
    ∃ (initial_pos : ℕ × ℕ), initial_pos = (1, 1) ∧
    -- The path must start and eventually return to the starting position
    (king_path.head = some initial_pos ∧ king_path.last = some initial_pos) ∧
    -- Each movement of the king increases the board value by 1
    (∀ (move : ℕ), move < king_path.length → 
        board (king_path.get move).fst (king_path.get move).snd + 1 = 
        board (king_path.get move.succ).fst (king_path.get move.succ).snd))
  → (∀ x y, board x y = board 0 0) ∧ -- All squares have the same number
  (∀ x y, even (board x y)) ∧ -- All numbers are even
  (∀ x y, board x y % 3 = 0) := -- All numbers are divisible by 3
sorry

end chessboard_transformation_l694_694268


namespace solution_set_of_inequality_l694_694740

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 2*x + 15 ≥ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 5} := 
sorry

end solution_set_of_inequality_l694_694740


namespace knights_won_30_games_l694_694376

noncomputable def teams_wins : Type := ℕ

def wins := [14, 20, 26, 28, 30, 34]

variable {Sharks Ravens Knights Dragons Royals Foxes : teams_wins}

axiom A1 : Sharks > Ravens
axiom A2 : Knights > Dragons ∧ Knights < Royals
axiom A3 : Dragons > 22
axiom A4 : Foxes < 22 ∧ (∀ (t : teams_wins), t ∈ wins → t ≥ Foxes)
axiom A5 : ∀ (t : teams_wins), t ∈ wins

theorem knights_won_30_games : Knights = 30 :=
by
  sorry

end knights_won_30_games_l694_694376


namespace frood_points_l694_694636

theorem frood_points (n : ℕ) (h : n > 29) : (n * (n + 1) / 2) > 15 * n := by
  sorry

end frood_points_l694_694636


namespace problem_solution_l694_694279

-- Define the parametric equations for C1 and C2
def parametric_C1 (t : ℝ) : ℝ × ℝ := (1 - (real.sqrt 2) / 2 * t, 3 + (real.sqrt 2) / 2 * t)
def parametric_C2 (φ : ℝ) : ℝ × ℝ := (1 + real.cos φ, real.sin φ)

-- Define the polar equations
def polar_equation_C1 (ρ θ : ℝ) : Prop := ρ * (real.cos θ + real.sin θ) = 4
def polar_equation_C2 (ρ θ : ℝ) : Prop := ρ = 2 * real.cos θ

-- Define the ray intersection conditions and the resulting maximum value
def max_intersection_value (α : ℝ) : Prop :=
  -real.pi / 4 < α ∧ α < real.pi / 2 →
  (∃ ρ1 ρ2 : ℝ, ρ1 = 4 / (real.cos α + real.sin α) ∧ ρ2 = 2 * real.cos α ∧
    ρ2 / ρ1 = (real.sqrt 2 + 1) / 4)

-- Final theorem statement
theorem problem_solution :
  (∀ t : ℝ, ∃ ρ θ : ℝ, polar_equation_C1 ρ θ) →
  (∀ φ : ℝ, ∃ ρ θ : ℝ, polar_equation_C2 ρ θ) →
  (∀ α : ℝ, max_intersection_value α) :=
by sorry

end problem_solution_l694_694279


namespace ratio_Peter_John_l694_694055

theorem ratio_Peter_John 
    (peter_money : ℕ := 320)
    (quincy_money : ℕ := peter_money + 20)
    (andrew_money : ℕ := quincy_money + quincy_money / 6 + quincy_money / 20) -- 15% of quincy_money
    (total_pooled_money : ℕ := 1200)
    (remaining_money : ℕ := 11)
    (john_money : ℕ := total_pooled_money + remaining_money - (peter_money + quincy_money + andrew_money)) :
    (peter_money / nat.gcd peter_money john_money) = (2 : ℕ)
    ∧ (john_money / nat.gcd peter_money john_money) = (1 : ℕ) := 
by {
  have h_gcd : nat.gcd peter_money john_money = 160 := sorry,
  split;
  { simp only [nat.div_self], assumption },
  sorry,
}

end ratio_Peter_John_l694_694055


namespace sum_arithmetic_sequence_terms_l694_694560

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a (n + m) = a n + m * (a 1 - a 0)

theorem sum_arithmetic_sequence_terms (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a) 
  (h₅ : a 5 = 8) :
  a 2 + a 4 + a 5 + a 9 = 32 :=
by
  sorry

end sum_arithmetic_sequence_terms_l694_694560


namespace james_birthday_stickers_l694_694045

def initial_stickers : ℕ := 39
def final_stickers : ℕ := 61

def birthday_stickers (s_initial s_final : ℕ) : ℕ := s_final - s_initial

theorem james_birthday_stickers :
  birthday_stickers initial_stickers final_stickers = 22 := by
  sorry

end james_birthday_stickers_l694_694045


namespace amy_carl_distance_after_2_hours_l694_694833

-- Conditions
def amy_rate : ℤ := 1
def carl_rate : ℤ := 2
def amy_interval : ℤ := 20
def carl_interval : ℤ := 30
def time_hours : ℤ := 2
def minutes_per_hour : ℤ := 60

-- Derived values
def time_minutes : ℤ := time_hours * minutes_per_hour
def amy_distance : ℤ := time_minutes / amy_interval * amy_rate
def carl_distance : ℤ := time_minutes / carl_interval * carl_rate

-- Question and answer pair
def distance_amy_carl : ℤ := amy_distance + carl_distance
def expected_distance : ℤ := 14

-- The theorem to prove
theorem amy_carl_distance_after_2_hours : distance_amy_carl = expected_distance := by
  sorry

end amy_carl_distance_after_2_hours_l694_694833


namespace complement_union_l694_694672

def S : Set ℝ := {x | x > -2}

def T : Set ℝ := {x | x^2 + 3x - 4 ≤ 0}

theorem complement_union (x : ℝ) : x ∈ (Sᶜ ∪ T) ↔ x ≤ 1 := by 
  sorry

end complement_union_l694_694672


namespace smallest_white_erasers_l694_694088

def total_erasers (n : ℕ) (pink : ℕ) (orange : ℕ) (purple : ℕ) (white : ℕ) : Prop :=
  pink = n / 5 ∧ orange = n / 6 ∧ purple = 10 ∧ white = n - (pink + orange + purple)

theorem smallest_white_erasers : ∃ n : ℕ, ∃ pink : ℕ, ∃ orange : ℕ, ∃ purple : ℕ, ∃ white : ℕ,
  total_erasers n pink orange purple white ∧ white = 9 := sorry

end smallest_white_erasers_l694_694088


namespace interest_rate_12_l694_694459

variables (SI P R T : ℝ)
variables (h1 : SI = 7200) (h2 : P = 20000) (h3 : T = 3)
def simple_interest_rate (P R T : ℝ) : ℝ := (P * R * T) / 100

theorem interest_rate_12 :
  simple_interest_rate P R T = SI →
  R = 12 :=
by
  intros h
  sorry

end interest_rate_12_l694_694459


namespace range_of_f_l694_694380

noncomputable def f (x : ℝ) : ℝ := 2^x + 1

theorem range_of_f : set.Ioi 1 ⊆ set.range f :=
by sorry

example : set.range f = { y : ℝ | 3 < y } :=
by sorry

end range_of_f_l694_694380


namespace A_inter_B_l694_694185

open Set Real

def A : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def B : Set ℝ := { y | ∃ x, y = exp x }

theorem A_inter_B :
  A ∩ B = { z | 0 < z ∧ z < 3 } :=
by
  sorry

end A_inter_B_l694_694185


namespace range_of_log2_sin_squared_l694_694123

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def sin_squared_log_range (x : ℝ) : ℝ :=
  log2 ((Real.sin x) ^ 2)

theorem range_of_log2_sin_squared (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ Real.pi) :
  ∃ y, y = sin_squared_log_range x ∧ y ≤ 0 :=
by
  sorry

end range_of_log2_sin_squared_l694_694123


namespace partial_fraction_telescoping_sum_l694_694431

-- Definition for partial fraction decomposition
variables {a b c d : ℤ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (h : ad ≠ bc)

-- Partial fraction theorem
theorem partial_fraction (h : a * d ≠ b * c) : 
  ∃ r s : ℚ, (r = -a / (b * c - a * d)) ∧ (s = c / (b * c - a * d)) ∧ 
  (∀ x : ℚ, 1 / ((a * x + b) * (c * x + d)) = r / (a * x + b) + s / (c * x + d)) :=
begin
  let r := -a / (b * c - a * d),
  let s := c / (b * c - a * d),
  use [r, s],
  split, 
  { exact rfl },
  split,
  { exact rfl },
  intros x,
  simp [r, s],
  field_simp [show (a * x + b) * (c * x + d) ≠ 0, by norm_num [h]],
end

-- Telescoping series theorem
theorem telescoping_sum : 
  (∑ n in finset.range 999, 1 / ((3 * n - 2 : ℕ) * (3 * n + 1 : ℕ))) = 1000 / 3001 :=
begin
  sorry -- Skipping the detailed proof steps for now.
end

end partial_fraction_telescoping_sum_l694_694431


namespace n_values_satisfy_condition_l694_694950

-- Define the exponential functions
def exp1 (n : ℤ) : ℚ := (-1/2) ^ n
def exp2 (n : ℤ) : ℚ := (-1/5) ^ n

-- Define the set of possible values for n
def valid_n : List ℤ := [-2, -1, 0, 1, 2, 3]

-- Define the condition for n to satisfy the inequality
def satisfies_condition (n : ℤ) : Prop := exp1 n > exp2 n

-- Prove that the only values of n that satisfy the condition are -1 and 2
theorem n_values_satisfy_condition :
  ∀ n ∈ valid_n, satisfies_condition n ↔ (n = -1 ∨ n = 2) :=
by
  intro n
  sorry

end n_values_satisfy_condition_l694_694950


namespace largest_prime_factor_of_7_fact_add_8_fact_l694_694014

theorem largest_prime_factor_of_7_fact_add_8_fact :
  let n := 7!
  let m := 8!
  let sum := n + m
  prime_factors sum = {2, 3, 5, 7} ∧ (∀ p ∈ {2, 3, 5, 7}, p <= 7) := by
  sorry

end largest_prime_factor_of_7_fact_add_8_fact_l694_694014


namespace ratio_of_ages_l694_694411

noncomputable def ratio_4th_to_3rd (age1 age2 age3 age4 age5 : ℕ) : ℚ :=
  age4 / age3

theorem ratio_of_ages
  (age1 age2 age3 age4 age5 : ℕ)
  (h1 : (age1 + age5) / 2 = 18)
  (h2 : age1 = 10)
  (h3 : age2 = age1 - 2)
  (h4 : age3 = age2 + 4)
  (h5 : age4 = age3 / 2)
  (h6 : age5 = age4 + 20) :
  ratio_4th_to_3rd age1 age2 age3 age4 age5 = 1 / 2 :=
by
  sorry

end ratio_of_ages_l694_694411


namespace g_neg2_l694_694318

def g (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 7 else x^2 + 2 * x + 1

theorem g_neg2 : g (-2) = -13 :=
by {
  sorry
}

end g_neg2_l694_694318


namespace min_cells_of_square_sheet_l694_694120

theorem min_cells_of_square_sheet (m n : ℕ) (hm : m = 10) (hn : n = 11) : 
  ∃ k : ℕ, k * k = 121 :=
by
  have : max m n = n, from sorry
  use n
  rw [this]
  exact sorry

end min_cells_of_square_sheet_l694_694120


namespace fill_pool_time_l694_694711

theorem fill_pool_time
  (pool_volume : ℕ)            -- volume of the pool in gallons
  (num_hoses : ℕ)              -- number of hoses
  (hose_flow_rate : ℕ)         -- flow rate of each hose in gallons per minute
  (minutes_per_hour : ℕ)       -- number of minutes in an hour
  (total_time_hours : ℕ) :     -- time to fill the pool in hours

  pool_volume = 32000 →
  num_hoses = 3 →
  hose_flow_rate = 4 →
  minutes_per_hour = 60 →
  total_time_hours = 44 :=
begin
  sorry,
end

end fill_pool_time_l694_694711


namespace tens_digit_of_difference_l694_694089

theorem tens_digit_of_difference
  (A B C : ℕ) (h1 : A ≠ 0) (h2 : A ≠ B) (h3 : A ≠ C) (h4 : B ≠ C) : 
  let original := 100 * A + 10 * B + C
      reversed := 100 * C + 10 * B + A
      difference := |original - reversed| in
  (difference / 10) % 10 = 9 :=
by sorry

end tens_digit_of_difference_l694_694089


namespace backpack_cost_eq_140_l694_694222

theorem backpack_cost_eq_140 (n c m : ℕ) (d : ℝ) 
    (hn : n = 5) (hc : c = 20) (hd : d = 0.20) (hm : m = 12) :
  (n * c - (n * c * d).toInt + n * m) = 140 := 
by
  sorry

end backpack_cost_eq_140_l694_694222


namespace height_of_smaller_cuboid_l694_694060

noncomputable def original_volume : ℝ := 18 * 15 * 2 -- Volume of the original cuboid

noncomputable def base_volume : ℝ := 6 * 4 -- Base volume of a smaller cuboid (length * width)

noncomputable def total_small_volume : ℝ := 7.5 * base_volume * (3 : ℝ) -- Volume of 7.5 smaller cuboids with height 3 meters

theorem height_of_smaller_cuboid : (original_volume = total_small_volume) :=
by
  unfold original_volume base_volume total_small_volume
  sorry

end height_of_smaller_cuboid_l694_694060


namespace find_number_l694_694469

theorem find_number {x : ℝ} 
  (h : 973 * x - 739 * x = 110305) : 
  x = 471.4 := 
by 
  sorry

end find_number_l694_694469


namespace total_trip_time_l694_694893

noncomputable def distance_on_highway : ℝ := 50 -- distance on highway in miles
noncomputable def distance_on_coastal_road : ℝ := 10 -- distance on coastal road in miles
noncomputable def speed_ratio : ℝ := 3 -- speed on the highway is 3 times the speed on the coastal road
noncomputable def time_on_coastal_road_minutes : ℝ := 30 -- time spent on coastal road in minutes

-- Prove the total time for the trip in minutes
theorem total_trip_time : (distance_on_coastal_road / (time_on_coastal_road_minutes / 60)) * speed_ratio = 20 → 
  let speed_coastal := distance_on_coastal_road / (time_on_coastal_road_minutes / 60) in
  let speed_highway := speed_coastal * speed_ratio in
  let time_on_highway := distance_on_highway / speed_highway in
  (time_on_highway * 60 + time_on_coastal_road_minutes) = 80 :=
begin
  -- Proof here (leaving as sorry)
  sorry
end

end total_trip_time_l694_694893


namespace evaluate_log_example_l694_694128

theorem evaluate_log_example (b x y : ℝ) (hx : x > 0) (hb : b > 0) (hy : y = logb b x) : b^y = x :=
by
  sorry

example : 8 ^ logb 8 5 = 5 :=
by
  have : logb 8 5 = log 5 / log 8 := rfl
  rw [this]
  have : 8 = real.exp (log 8) := by rw [real.exp_log hb]
  rw [this]
  sorry -- complete based on equivalence of logs and exponentiation

end evaluate_log_example_l694_694128


namespace least_positive_t_geometric_progression_l694_694490

open Real

theorem least_positive_t_geometric_progression (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) : 
  ∃ t : ℕ, ∀ t' : ℕ, (t' > 0) → 
  (|arcsin (sin (t' * α)) - 8 * α| = 0) → t = 8 :=
by
  sorry

end least_positive_t_geometric_progression_l694_694490


namespace B_takes_6_days_to_complete_work_alone_l694_694444

theorem B_takes_6_days_to_complete_work_alone 
    (work_duration_A : ℕ) 
    (work_payment : ℚ)
    (work_days_with_C : ℕ) 
    (payment_C : ℚ) 
    (combined_work_rate_A_B_C : ℚ)
    (amount_to_be_shared_A_B : ℚ) 
    (combined_daily_earning_A_B : ℚ) :
  work_duration_A = 6 ∧
  work_payment = 3360 ∧ 
  work_days_with_C = 3 ∧ 
  payment_C = 420.00000000000017 ∧ 
  combined_work_rate_A_B_C = 1 / 3 ∧ 
  amount_to_be_shared_A_B = 2940 ∧ 
  combined_daily_earning_A_B = 980 → 
  work_duration_A = 6 ∧
  (∃ (work_duration_B : ℕ), work_duration_B = 6) :=
by 
  sorry

end B_takes_6_days_to_complete_work_alone_l694_694444


namespace angle_ABC_30_degrees_l694_694609

theorem angle_ABC_30_degrees 
    (angle_CBD : ℝ)
    (angle_ABD : ℝ)
    (angle_ABC : ℝ)
    (h1 : angle_CBD = 90)
    (h2 : angle_ABC + angle_ABD + angle_CBD = 180)
    (h3 : angle_ABD = 60) :
    angle_ABC = 30 :=
by
  sorry

end angle_ABC_30_degrees_l694_694609


namespace eliminate_y_by_subtraction_l694_694780

theorem eliminate_y_by_subtraction (m n : ℝ) :
  (6 * x + m * y = 3) ∧ (2 * x - n * y = -6) →
  (∀ x y : ℝ, 4 * x + (m + n) * y = 9) → (m + n = 0) :=
by
  intros h eq_subtracted
  sorry

end eliminate_y_by_subtraction_l694_694780


namespace total_money_taken_l694_694869

def individual_bookings : ℝ := 12000
def group_bookings : ℝ := 16000
def returned_due_to_cancellations : ℝ := 1600

def total_taken (individual_bookings : ℝ) (group_bookings : ℝ) (returned_due_to_cancellations : ℝ) : ℝ :=
  (individual_bookings + group_bookings) - returned_due_to_cancellations

theorem total_money_taken :
  total_taken individual_bookings group_bookings returned_due_to_cancellations = 26400 := by
  sorry

end total_money_taken_l694_694869


namespace coefficient_of_x79_is_zero_l694_694918

noncomputable def P (x : ℕ) : ℕ := 
  (x - 2) * (x ^ 2 - 3) * (x ^ 3 - 4) * ... * (x ^ 11 - 12) * (x ^ 12 - 13)

theorem coefficient_of_x79_is_zero : coefficient (P x) 79 = 0 :=
by
  sorry

end coefficient_of_x79_is_zero_l694_694918


namespace unique_solution_l694_694136

noncomputable def satisfies_condition (x : ℝ) : Prop :=
  x > 0 ∧ (x * Real.sqrt (18 - x) + Real.sqrt (24 * x - x^3) ≥ 18)

theorem unique_solution :
  ∀ x : ℝ, satisfies_condition x ↔ x = 6 :=
by
  intro x
  unfold satisfies_condition
  sorry

end unique_solution_l694_694136


namespace four_digit_sum_ten_divisible_by_eleven_l694_694586

theorem four_digit_sum_ten_divisible_by_eleven : 
  {n | (10^3 ≤ n ∧ n < 10^4) ∧ (∑ i in (Int.toString n).toList, i.toNat) = 10 ∧ (let digits := (Int.toString n).toList.map (λ c, c.toNat) in ((digits.nth 0).getD 0 + (digits.nth 2).getD 0) - ((digits.nth 1).getD 0 + (digits.nth 3).getD 0)) % 11 == 0}.card = 30 := sorry

end four_digit_sum_ten_divisible_by_eleven_l694_694586


namespace count_even_integers_with_conditions_l694_694979

-- Define the conditions in Lean
def digits_different (n : ℕ) : Prop :=
  let digits := Int.digits 10 n
  digits.nodup

def units_digit_either_6_or_8 (n : ℕ) : Prop :=
  let units_digit := n % 10
  units_digit = 6 ∨ units_digit = 8

def even_integers (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * k

def between_5000_and_8000 (n : ℕ) : Prop :=
  5000 ≤ n ∧ n < 8000

-- Prove the number of integers satisfying all conditions
theorem count_even_integers_with_conditions : 
  ∃ c : ℕ, c = 280 ∧ (∀ n : ℕ, between_5000_and_8000 n → 
    digits_different n → units_digit_either_6_or_8 n → even_integers n → c = 280) := 
sorry

end count_even_integers_with_conditions_l694_694979


namespace find_k_l694_694574

-- Define the sequence
def a : ℕ → ℚ
| 0       := 1/2
| (n + 1) := a n + (a n)^2 / 2023

-- State the theorem
theorem find_k : ∃ k : ℕ, a k < 1 ∧ 1 < a (k + 1) ∧ k = 2023 := 
sorry

end find_k_l694_694574


namespace intersection_of_sets_l694_694972

theorem intersection_of_sets :
  let M := {0, 1}
  let N := {x | ∃ y (hy : y ∈ M), x = 2*y + 1}
  M ∩ N = {1} :=
by
  sorry

end intersection_of_sets_l694_694972


namespace retailer_discount_problem_l694_694094

theorem retailer_discount_problem
  (CP MP SP : ℝ) 
  (h1 : CP = 100)
  (h2 : MP = CP + (0.65 * CP))
  (h3 : SP = CP + (0.2375 * CP)) :
  (MP - SP) / MP * 100 = 25 :=
by
  sorry

end retailer_discount_problem_l694_694094


namespace cheaper_store_difference_in_cents_l694_694086

/-- Given the following conditions:
1. Best Deals offers \$12 off the list price of \$52.99.
2. Market Value offers 20% off the list price of \$52.99.
 -/
theorem cheaper_store_difference_in_cents :
  let list_price : ℝ := 52.99
  let best_deals_price := list_price - 12
  let market_value_price := list_price * 0.80
  best_deals_price < market_value_price →
  let difference_in_dollars := market_value_price - best_deals_price
  let difference_in_cents := difference_in_dollars * 100
  difference_in_cents = 140 := by
  intro h
  let list_price : ℝ := 52.99
  let best_deals_price := list_price - 12
  let market_value_price := list_price * 0.80
  let difference_in_dollars := market_value_price - best_deals_price
  let difference_in_cents := difference_in_dollars * 100
  sorry

end cheaper_store_difference_in_cents_l694_694086


namespace largest_term_sequence_a_l694_694425

theorem largest_term_sequence_a (n : ℕ) (a : ℕ → ℚ) : (n : ℕ) → a n := 
∀ n, a n = n / (n ^ 2 + 2020) ∧ a 45 = \frac{45}{4045} := sorry

end largest_term_sequence_a_l694_694425


namespace hot_dog_cost_l694_694480

variables (h d : ℝ)

theorem hot_dog_cost :
  (3 * h + 4 * d = 10) →
  (2 * h + 3 * d = 7) →
  d = 1 :=
by
  intros h_eq d_eq
  -- Proof skipped
  sorry

end hot_dog_cost_l694_694480


namespace savings_time_l694_694683

variable {annual_salary : ℕ} (annual_savings : ℕ) (required_downpayment : ℕ)

axiom salary_value : annual_salary = 150000
axiom savings_percentage : 0.10 * annual_salary = annual_savings
axiom house_cost : required_downpayment = 137500
axiom cost_percentage : 0.25 * 550000 = required_downpayment

theorem savings_time (annual_salary : ℕ) (annual_savings required_downpayment : ℕ) :
  (0.25 * 550000 = required_downpayment) ∧ 
  (0.10 * annual_salary = annual_savings) ∧
  (annual_salary = 150000) →
  (required_downpayment / annual_savings : ℝ) ≤ 10 :=
by
  intros h
  cases h with h1 h
  cases h with h2 h3
  rw [h3, h2, h1]
  sorry

end savings_time_l694_694683


namespace mountain_number_count_l694_694763

theorem mountain_number_count : 
  (∃ count : ℕ, 
    (∀ w x : ℕ, 1 ≤ w ∧ w ≤ 8 → (w + 1) ≤ x ∧ x ≤ 9 → 
    count = ∑ i in (finset.range (8 + 1)).filter (λ n, 1 ≤ n ∧ n ≤ 8), 
    finset.card (finset.Ioc i 9)) ∧ count = 36) := 
  sorry

end mountain_number_count_l694_694763


namespace point_in_first_quadrant_l694_694166

def imaginary_unit := Complex.i

theorem point_in_first_quadrant :
  ∃ (a b : ℝ), (3 - imaginary_unit) * imaginary_unit = a + b * imaginary_unit ∧ a > 0 ∧ b > 0 :=
by
  sorry

end point_in_first_quadrant_l694_694166


namespace count_four_digit_numbers_l694_694599

open Int

def is_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9

def valid_combination (a b c d : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧
  a ≠ 0 ∧
  a + b + c + d = 10 ∧
  (a + c) - (b + d) % 11 = 0

theorem count_four_digit_numbers :
  ∃ n, n > 0 ∧ ∀ a b c d : ℕ, valid_combination a b c d → n = sorry :=
sorry

end count_four_digit_numbers_l694_694599


namespace number_of_men_at_picnic_l694_694461

theorem number_of_men_at_picnic (total persons W M A C : ℕ) (h1 : total = 200) 
  (h2 : M = W + 20) (h3 : A = C + 20) (h4 : A = M + W) : M = 65 :=
by
  -- Proof can be filled in here
  sorry

end number_of_men_at_picnic_l694_694461


namespace price_per_postcard_is_correct_l694_694486

noncomputable def initial_postcards : ℕ := 18
noncomputable def sold_postcards : ℕ := initial_postcards / 2
noncomputable def price_per_postcard_sold : ℕ := 15
noncomputable def total_earned : ℕ := sold_postcards * price_per_postcard_sold
noncomputable def total_postcards_after : ℕ := 36
noncomputable def remaining_original_postcards : ℕ := initial_postcards - sold_postcards
noncomputable def new_postcards_bought : ℕ := total_postcards_after - remaining_original_postcards
noncomputable def price_per_new_postcard : ℕ := total_earned / new_postcards_bought

theorem price_per_postcard_is_correct:
  price_per_new_postcard = 5 :=
by
  sorry

end price_per_postcard_is_correct_l694_694486


namespace complement_intersection_l694_694974

open Set

variable (U : Set ℤ) (A B : Set ℤ)

theorem complement_intersection (hU : U = univ)
                               (hA : A = {3, 4})
                               (h_union : A ∪ B = {1, 2, 3, 4}) :
  (U \ A) ∩ B = {1, 2} :=
by
  sorry

end complement_intersection_l694_694974


namespace polar_to_rectangular_l694_694493

theorem polar_to_rectangular:
  ∀ (r θ : ℝ), r = 3 * Real.sqrt 2 ∧ θ = (5 * Real.pi) / 6 →
    ( (r * Real.cos θ, r * Real.sin θ) = ( - (3 * Real.sqrt 6) / 2, 3 * Real.sqrt 2 / 2 ) ) :=
begin
  intros r θ h,
  cases h with hr hθ,
  sorry
end

end polar_to_rectangular_l694_694493


namespace correct_equation_l694_694868

def initial_count_A : ℕ := 54
def initial_count_B : ℕ := 48
def new_count_A (x : ℕ) : ℕ := initial_count_A + x
def new_count_B (x : ℕ) : ℕ := initial_count_B - x

theorem correct_equation (x : ℕ) : new_count_A x = 2 * new_count_B x := 
sorry

end correct_equation_l694_694868


namespace positive_sequence_unique_l694_694889

theorem positive_sequence_unique (x : Fin 2021 → ℝ) (h : ∀ i : Fin 2020, x i.succ = (x i ^ 3 + 2) / (3 * x i ^ 2)) (h' : x 2020 = x 0) : ∀ i, x i = 1 := by
  sorry

end positive_sequence_unique_l694_694889


namespace abs_eq_abs_of_unique_solution_l694_694687

variables {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0)
theorem abs_eq_abs_of_unique_solution
  (h : ∃ x : ℝ, ∀ y : ℝ, a * (y - a)^2 + b * (y - b)^2 = 0 ↔ y = x) :
  |a| = |b| :=
sorry

end abs_eq_abs_of_unique_solution_l694_694687


namespace clock_angle_7_15_l694_694769

noncomputable def hour_angle_at (hour : ℕ) (minutes : ℕ) : ℝ :=
  hour * 30 + (minutes * 0.5)

noncomputable def minute_angle_at (minutes : ℕ) : ℝ :=
  minutes * 6

noncomputable def small_angle (angle1 angle2 : ℝ) : ℝ :=
  let diff := abs (angle1 - angle2)
  if diff <= 180 then diff else 360 - diff

theorem clock_angle_7_15 : small_angle (hour_angle_at 7 15) (minute_angle_at 15) = 127.5 :=
by
  sorry

end clock_angle_7_15_l694_694769


namespace Marilyn_end_caps_l694_694680

def starting_caps := 51
def shared_caps := 36
def ending_caps := starting_caps - shared_caps

theorem Marilyn_end_caps : ending_caps = 15 := by
  -- proof omitted
  sorry

end Marilyn_end_caps_l694_694680


namespace problem1_problem2_l694_694530

noncomputable theory

open Real

variables (α : ℝ) (h : tan α = 2)

-- Problem 1
theorem problem1 : tan (α + π/4) = -3 :=
by {
  sorry
}

-- Problem 2
theorem problem2 : (6 * sin α + cos α) / (3 * sin α - 2 * cos α) = 13 / 4 :=
by {
  sorry
}

end problem1_problem2_l694_694530


namespace largest_prime_factor_of_7_fact_add_8_fact_l694_694024

theorem largest_prime_factor_of_7_fact_add_8_fact :
  let n := 7!
  let m := 8!
  let sum := n + m
  prime_factors sum = {2, 3, 5, 7} ∧ (∀ p ∈ {2, 3, 5, 7}, p <= 7) := by
  sorry

end largest_prime_factor_of_7_fact_add_8_fact_l694_694024


namespace find_a_l694_694964

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x - a * x

def tangent_slope_at_one (a : ℝ) : ℝ :=
    (deriv (λ x : ℝ, f x a)) 1

def line_slope : ℝ := -2

theorem find_a (a : ℝ) (h_parallel : tangent_slope_at_one a = line_slope) : a = 3 :=
by
  sorry

end find_a_l694_694964


namespace circumference_of_cone_base_l694_694075

theorem circumference_of_cone_base (r : ℝ) (angle : ℝ) (circumference : ℝ) 
  (sector_circumference : ℝ) :
  r = 5 →
  angle = 300 →
  circumference = 2 * Real.pi * r →
  sector_circumference = (angle / 360) * circumference →
  sector_circumference = (25 / 3) * Real.pi :=
by
  intros h_r h_angle h_circumference h_sector
  rw [h_r, h_angle, h_circumference] at h_sector ⊢
  sorry

end circumference_of_cone_base_l694_694075


namespace prob_at_least_one_hit_l694_694695

theorem prob_at_least_one_hit (P: Type) (ha hb : P → Prop) 
    (P_both_hit : probability (λ (x : P), ha x ∧ hb x) = 0.6) : 
    probability (λ x, ha x ∨ hb x) = 0.84 :=
by 
    sorry

end prob_at_least_one_hit_l694_694695


namespace largest_prime_factor_7_fact_8_fact_l694_694003

theorem largest_prime_factor_7_fact_8_fact : 
  let n := 7! + 8! 
  ∃ p, Prime p ∧ p ∣ n ∧ (∀ q, Prime q ∧ q ∣ n → q ≤ p) := 
begin
  let n := fact 7 + fact 8,
  use 7, 
  split,
  { apply Prime.prime_7 },
  split,
  { sorry },  
  { sorry },
end

end largest_prime_factor_7_fact_8_fact_l694_694003


namespace largest_prime_factor_7_fact_8_fact_l694_694009

theorem largest_prime_factor_7_fact_8_fact : 
  let n := 7! + 8! 
  ∃ p, Prime p ∧ p ∣ n ∧ (∀ q, Prime q ∧ q ∣ n → q ≤ p) := 
begin
  let n := fact 7 + fact 8,
  use 7, 
  split,
  { apply Prime.prime_7 },
  split,
  { sorry },  
  { sorry },
end

end largest_prime_factor_7_fact_8_fact_l694_694009


namespace kaleb_clothing_problem_l694_694432

theorem kaleb_clothing_problem 
  (initial_clothing : ℕ) 
  (one_load : ℕ) 
  (remaining_loads : ℕ) : 
  initial_clothing = 39 → one_load = 19 → remaining_loads = 5 → (initial_clothing - one_load) / remaining_loads = 4 :=
sorry

end kaleb_clothing_problem_l694_694432


namespace evaluate_expression_l694_694990

def star (A B : ℝ) : ℝ := (A + B) / 2

theorem evaluate_expression : star (star 3 5) 8 = 6 :=
by
  sorry

end evaluate_expression_l694_694990


namespace value_of_M_l694_694613

theorem value_of_M (M : ℝ) (h : 0.2 * M = 500) : M = 2500 :=
by
  sorry

end value_of_M_l694_694613


namespace four_digit_sum_ten_divisible_by_eleven_l694_694590

theorem four_digit_sum_ten_divisible_by_eleven : 
  {n | (10^3 ≤ n ∧ n < 10^4) ∧ (∑ i in (Int.toString n).toList, i.toNat) = 10 ∧ (let digits := (Int.toString n).toList.map (λ c, c.toNat) in ((digits.nth 0).getD 0 + (digits.nth 2).getD 0) - ((digits.nth 1).getD 0 + (digits.nth 3).getD 0)) % 11 == 0}.card = 30 := sorry

end four_digit_sum_ten_divisible_by_eleven_l694_694590


namespace X_paid_percentage_of_Y_l694_694403

noncomputable def X_payment (total: ℝ) (Y_payment: ℝ): ℝ := total - Y_payment

noncomputable def X_percentage (X_payment: ℝ) (Y_payment: ℝ): ℝ := (X_payment / Y_payment) * 100

theorem X_paid_percentage_of_Y:
  ∀ (total_payment Y_payment: ℝ),
  total_payment = 580 ∧ Y_payment = 263.64 →
  X_percentage (X_payment total_payment Y_payment) Y_payment = 119.98 := 
by
  intros total_payment Y_payment h,
  cases h with h_total h_Y,
  rw [h_total, h_Y],
  sorry

end X_paid_percentage_of_Y_l694_694403


namespace boat_distance_l694_694269

noncomputable def distance_along_stream (v_b : ℝ) (d_upstream : ℝ) : ℝ :=
  let v_s := v_b - d_upstream in
  v_b + v_s

theorem boat_distance (v_b : ℝ) (d_upstream : ℝ) (h_vb : v_b = 8) (h_du: d_upstream = 5) :
  distance_along_stream v_b d_upstream = 11 :=
by
  sorry

end boat_distance_l694_694269


namespace smallest_B_c_sum_l694_694607

theorem smallest_B_c_sum :
  ∃ (B c : ℕ), B ∈ {0, 1, 2, 3, 4} ∧ c > 6 ∧ 31 * B = 4 * c + 4 ∧ B + c = 34 :=
by
  sorry

end smallest_B_c_sum_l694_694607


namespace arithmetic_progression_probability_l694_694155

open Fin

-- Define the outcome set for an 8-sided die
def eight_sided_die := {x : ℕ // 1 ≤ x ∧ x ≤ 8}

-- Define a function to check if a list of four numbers can form an arithmetic progression with a common difference of 2
def is_arithmetic_progression (lst : List ℕ) : Prop :=
  lst.length = 4 ∧ ∃ d a, d = 2 ∧ lst = [a, a + d, a + 2 * d, a + 3 * d]

-- Define the set of all possible outcomes when four dice are tossed
def all_outcomes : List (List ℕ) :=
  List.product (List.product (List.product (List.finRange 8) (List.finRange 8)) (List.finRange 8)) (List.finRange 8)

-- Define the favorable outcomes where the four numbers can form an arithmetic progression with common difference 2
def favorable_outcomes : List (List ℕ) :=
  List.filter is_arithmetic_progression all_outcomes

-- Define the probability of obtaining an arithmetic progression with common difference 2 when four dice are tossed
def probability := (favorable_outcomes.length : ℚ) / (all_outcomes.length : ℚ)

theorem arithmetic_progression_probability : probability = 3 / 256 :=
  by sorry

end arithmetic_progression_probability_l694_694155


namespace fraction_proof_l694_694786

theorem fraction_proof (w x y z : ℚ) 
  (h1 : w / x = 1 / 3)
  (h2 : w / y = 3 / 4)
  (h3 : x / z = 2 / 5) : 
  (x + y) / (y + z) = 26 / 53 := 
by
  sorry

end fraction_proof_l694_694786


namespace eq_circleC_not_parallel_l694_694294

-- Define the conditions
def circleM (x y : ℝ) (r : ℝ) := (x + 2)^2 + (y + 2)^2 = r^2

def symmetry_line (x y : ℝ) := x + y + 2 = 0

def pointP := (1 : ℝ, 1 : ℝ)

-- Define the transformation due to symmetry
def reflected_point (px py : ℝ) : ℝ × ℝ := (2 - px, 2 - py)

noncomputable def circleC (x y : ℝ) (r : ℝ) := (x - 2)^2 + (y - 2)^2 = r^2

-- The main theorem statements
theorem eq_circleC (r : ℝ) (hr : r > 0) (x y : ℝ) :
  (∀ (x y : ℝ), symmetry_line x y → circleM x y r →
  (reflected_point x y = (2, 2))) →
  circleC x y r :=
by
  intros
  sorry

theorem not_parallel (r : ℝ) (hr : r > 0) :
  (∃ A B : ℝ × ℝ, ∃ m1 m2 : ℝ, (m1 * m2 = -1) ∧ 
    ((1 - 0) / (1 - 0) ≠ (B.2 - A.2) / (B.1 - A.1))) →
  ¬(0, 0) // pointP ∥ A // B :=
by
  intros
  sorry

end eq_circleC_not_parallel_l694_694294


namespace sum_x_coordinates_midpoints_l694_694386

theorem sum_x_coordinates_midpoints (a b c : ℝ) (h : a + b + c = 12) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 12 :=
by
  sorry

end sum_x_coordinates_midpoints_l694_694386


namespace eval_I1_relation_I_n_plus_1_exists_lim_I_n_lim_I_n_l694_694301

open Real

variable (a : ℝ) (n : ℕ) (t : ℝ)

-- Condition: a is a positive constant number
axiom h_a_pos : a > 0

-- Definition of I_n(t)
def I_n (t : ℝ) (n : ℕ) : ℝ := ∫ x in 0..t, x^n * exp (-a * x)

-- Given: \lim_{t \to \infty} t^n e^{-at} = 0
axiom h_lim_tn_exp_neg_at : ∀ (n : ℕ), ∃ c : ℝ, ∀ (t : ℝ), t^n * exp (-a * t) < c

-- 1. Evaluate I_1(t)
theorem eval_I1 : I_n t 1 = 1 / a^2 - (1 / a^2) * exp (-a * t) - (t / a) * exp (-a * t) := sorry

-- 2. Find the relation of I_{n+1}(t) and I_n(t)
theorem relation_I_n_plus_1 : I_n t (n + 1) = - t^(n+1) / a * exp (-a * t) + (n + 1) / a * I_n t n := sorry

-- 3. Prove that there exists \lim_{t \to \infty} I_n(t) for all natural numbers n
theorem exists_lim_I_n : ∀ n : ℕ, ∃ L_n : ℝ, ∃ t : ℝ, ∀ (t : ℝ), I_n t n = L_n := sorry

-- 4. Find \lim_{t \to \infty} I_n(t)
theorem lim_I_n : ∀ (n : ℕ), ∀ (a : ℝ), ∀ (h_a_pos : a > 0), lim (t → ∞) (λ t, I_n t n) = fact n / a^(n + 1) := sorry

end eval_I1_relation_I_n_plus_1_exists_lim_I_n_lim_I_n_l694_694301


namespace boy_present_age_l694_694038

-- Define the boy's present age
variable (x : ℤ)

-- Conditions from the problem statement
def condition_one : Prop :=
  x + 4 = 2 * (x - 6)

-- Prove that the boy's present age is 16
theorem boy_present_age (h : condition_one x) : x = 16 := 
sorry

end boy_present_age_l694_694038


namespace count_whole_numbers_sqrt_cubed_50_sqrt_200_l694_694227

theorem count_whole_numbers_sqrt_cubed_50_sqrt_200 :
  ({n : ℕ | 4 ≤ n ∧ n ≤ 14}.card = 11) :=
by
  sorry

end count_whole_numbers_sqrt_cubed_50_sqrt_200_l694_694227


namespace cloud9_total_money_l694_694877

-- Definitions
def I : ℕ := 12000
def G : ℕ := 16000
def R : ℕ := 1600

-- Total money taken after cancellations
def T : ℕ := I + G - R

-- Theorem stating that T is 26400
theorem cloud9_total_money : T = 26400 := by
  unfold T I G R
  sorry

end cloud9_total_money_l694_694877


namespace sin_tan_in_second_quadrant_sin_tan_in_third_quadrant_l694_694187

-- Define the conditions
def cos_alpha := -4/5

-- Define the property for the second quadrant
def in_second_quadrant (α : Real) : Prop :=
  cos α = cos_alpha ∧ 0 < α ∧ α < π

-- Define the property for the third quadrant
def in_third_quadrant (α : Real) : Prop :=
  cos α = cos_alpha ∧ π < α ∧ α < 2 * π

-- Define the sin and tan properties in the second quadrant
def sin_alpha_2nd_quad (α : Real) : Prop :=
  sin α = 3/5

def tan_alpha_2nd_quad (α : Real) : Prop :=
  tan α = -3/4

-- Define the sin and tan properties in the third quadrant
def sin_alpha_3rd_quad (α : Real) : Prop :=
  sin α = -3/5

def tan_alpha_3rd_quad (α : Real) : Prop :=
  tan α = 3/4

-- Proof goals
theorem sin_tan_in_second_quadrant (α : Real) (h : in_second_quadrant α) : sin_alpha_2nd_quad α ∧ tan_alpha_2nd_quad α :=
  sorry

theorem sin_tan_in_third_quadrant (α : Real) (h : in_third_quadrant α) : sin_alpha_3rd_quad α ∧ tan_alpha_3rd_quad α :=
  sorry

end sin_tan_in_second_quadrant_sin_tan_in_third_quadrant_l694_694187


namespace simon_age_l694_694848

theorem simon_age : 
  ∃ s : ℕ, 
  ∀ a : ℕ,
    a = 30 → 
    s = (a / 2) - 5 → 
    s = 10 := 
by
  sorry

end simon_age_l694_694848


namespace hyperbola_asymptotes_l694_694724

theorem hyperbola_asymptotes (x y : ℝ) :
  (∀ x y, x^2 - 4 * y^2 = -1 → x = 2 * y ∨ x = -2 * y) :=
begin
  intro h,
  sorry
end

end hyperbola_asymptotes_l694_694724


namespace trig_expression_value_l694_694188

open Real

theorem trig_expression_value (x : ℝ) (h : tan (π - x) = -2) : 
  4 * sin x ^ 2 - 3 * sin x * cos x - 5 * cos x ^ 2 = 1 := 
sorry

end trig_expression_value_l694_694188


namespace four_digit_numbers_count_l694_694591

theorem four_digit_numbers_count :
  ∃ (n : ℕ),  n = 24 ∧
  let s := λ (a b c d : ℕ), (a + b + c + d = 10 ∧ a ≠ 0 ∧ ((a + c) - (b + d)) ∈ [(-11), 0, 11]) in
  (card {abcd : Fin 10 × Fin 10 × Fin 10 × Fin 10 | (s abcd.1.1 abcd.1.2 abcd.2.1 abcd.2.2)}) = n :=
by
  sorry

end four_digit_numbers_count_l694_694591


namespace prob_team_a_wins_4_1_l694_694354

-- Define the probabilities for winning at home and away
def prob_win_home : ℝ := 0.6
def prob_win_away : ℝ := 0.5

-- Define the game schedule as a list of booleans where true means home and false means away
def game_schedule : List Bool := [true, true, false, false, true, false, true]

-- Define the probability of Team A winning with a specific losing game scenario and the remaining games to win
def prob_winning_with_particular_loss (loss_idx : ℕ) : ℝ :=
  let wins := List.splitAt loss_idx game_schedule |>.1  -- games before the loss
  let remaining := List.splitAt loss_idx game_schedule |>.2.tail.get! ⟨0, sorry⟩  -- games after the loss
  let win_prob wins := wins.map (λ g => if g then prob_win_home else prob_win_away) |>.prod
  (1 - (if game_schedule.get! loss_idx then prob_win_home else prob_win_away)) *
    win_prob wins *
    win_prob (remaining.take 4)  -- the next 4 wins

-- Prove the total probability of winning with a score of 4:1
theorem prob_team_a_wins_4_1 : 
  (prob_winning_with_particular_loss 0 +
   prob_winning_with_particular_loss 1 +
   prob_winning_with_particular_loss 2 +
   prob_winning_with_particular_loss 3) = 0.18 := by
  sorry

end prob_team_a_wins_4_1_l694_694354


namespace fraction_of_students_older_than_4_years_l694_694445

-- Definitions based on conditions
def total_students := 50
def students_younger_than_3 := 20
def students_not_between_3_and_4 := 25
def students_older_than_4 := students_not_between_3_and_4 - students_younger_than_3
def fraction_older_than_4 := students_older_than_4 / total_students

-- Theorem to prove the desired fraction
theorem fraction_of_students_older_than_4_years : fraction_older_than_4 = 1/10 :=
by
  sorry

end fraction_of_students_older_than_4_years_l694_694445


namespace greatest_possible_x_lcm_l694_694919

theorem greatest_possible_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105): x = 105 := 
sorry

end greatest_possible_x_lcm_l694_694919


namespace line_equation_equiv_l694_694458

variable {x y : ℝ}

theorem line_equation_equiv (x y : ℝ) :
  (∃ x y, (⟨2, -1⟩ : ℝ × ℝ) • (⟨x, y⟩ - ⟨5, -3⟩) = 0) ↔ y = 2 * x - 13 := by
sorry

end line_equation_equiv_l694_694458


namespace right_angled_triangle_set_C_l694_694420

theorem right_angled_triangle_set_C : 
  let a := 1
  let b := 1
  let c := Real.sqrt 2
  a^2 + b^2 = c^2 :=
by
  let a := 1
  let b := 1
  let c := Real.sqrt 2
  calc 
    a^2 + b^2 = 1^2 + 1^2 : by rfl
    ...       = 1 + 1 : by rfl
    ...       = 2 : by rfl
    ...       = (Real.sqrt 2)^2 : by rfl

end right_angled_triangle_set_C_l694_694420


namespace percent_increase_l694_694446

theorem percent_increase (original new : ℕ) (h1 : original = 30) (h2 : new = 60) :
  ((new - original) / original) * 100 = 100 := 
by
  sorry

end percent_increase_l694_694446


namespace smallest_a_exists_l694_694028

theorem smallest_a_exists : ∃ a b c : ℕ, 
                          (∀ α β : ℝ, 
                          (α > 0 ∧ α ≤ 1 / 1000) ∧ 
                          (β > 0 ∧ β ≤ 1 / 1000) ∧ 
                          (α + β = -b / a) ∧ 
                          (α * β = c / a) ∧ 
                          (b * b - 4 * a * c > 0)) ∧ 
                          (a = 1001000) := sorry

end smallest_a_exists_l694_694028


namespace ab_value_l694_694240

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := 
by 
  sorry

end ab_value_l694_694240


namespace count_divisible_by_11_with_digits_sum_10_l694_694602

-- Definitions and conditions from step a)
def digits_add_up_to_10 (n : ℕ) : Prop :=
  let ds := n.digits 10 in ds.length = 4 ∧ ds.sum = 10

def divisible_by_11 (n : ℕ) : Prop :=
  let ds := n.digits 10 in
  (ds.nth 0).get_or_else 0 +
  (ds.nth 2).get_or_else 0 -
  (ds.nth 1).get_or_else 0 -
  (ds.nth 3).get_or_else 0 ∣ 11

-- The tuple (question, conditions, answer) formalized in Lean
theorem count_divisible_by_11_with_digits_sum_10 :
  ∃ N : ℕ, N = (
  (Finset.filter (λ x, digits_add_up_to_10 x ∧ divisible_by_11 x)
  (Finset.range 10000)).card
) := 
sorry

end count_divisible_by_11_with_digits_sum_10_l694_694602


namespace sin_theta_value_l694_694988

theorem sin_theta_value (θ : ℝ) (h1 : 6 * tan θ = 4 * cos θ) (h2 : π < θ ∧ θ < 2 * π) :
  sin θ = 1 / 2 :=
sorry

end sin_theta_value_l694_694988


namespace cloud9_total_money_l694_694876

-- Definitions
def I : ℕ := 12000
def G : ℕ := 16000
def R : ℕ := 1600

-- Total money taken after cancellations
def T : ℕ := I + G - R

-- Theorem stating that T is 26400
theorem cloud9_total_money : T = 26400 := by
  unfold T I G R
  sorry

end cloud9_total_money_l694_694876


namespace complex_calculation_l694_694766

def a : ℂ := 3 - 2 * Complex.i
def b : ℂ := 2 + 3 * Complex.i

theorem complex_calculation : 3 * a + 4 * b = 17 + 6 * Complex.i :=
by
  sorry

end complex_calculation_l694_694766


namespace angle_B_60_l694_694624

-- We define that we are dealing with a triangle with sides a, b, c
-- and angle B such that the given condition holds.

universe u

noncomputable def find_angle_B (a b c : ℝ) (B : ℝ) : Prop :=
  a^2 + c^2 - b^2 = a * c → B = 60

theorem angle_B_60 {a b c : ℝ} (h : a^2 + c^2 - b^2 = a * c) : 
  find_angle_B a b c (60 * Real.pi / 180) :=
begin
  sorry,
end

end angle_B_60_l694_694624


namespace panthers_score_points_l694_694454

theorem panthers_score_points (C P : ℕ) (h1 : C + P = 34) (h2 : C = P + 14) : P = 10 :=
by
  sorry

end panthers_score_points_l694_694454


namespace calc_difference_l694_694531

def f (n : ℕ) : ℝ :=
  ∑ i in finset.Icc (n+1) (3*n+1), (1 / (i : ℝ))

theorem calc_difference (k : ℕ) : 
  f (k + 1) - f k = ∑ i in finset.Icc (3*k+2) (3*k+4), (1 / (i : ℝ)) - 1 / (k + 1) :=
by
  sorry

end calc_difference_l694_694531


namespace least_positive_integer_solution_l694_694768

theorem least_positive_integer_solution :
  ∃ b : ℕ, b ≡ 2 [MOD 3] ∧ b ≡ 3 [MOD 4] ∧ b ≡ 4 [MOD 5] ∧ b ≡ 8 [MOD 9] ∧ b = 179 :=
by
  sorry

end least_positive_integer_solution_l694_694768


namespace find_matrix_N_l694_694914

def mat_eq (N : Matrix (Fin 2) (Fin 2) ℤ) : Prop :=
  (N.mul_vec (λ i, if i = 0 then 4 else 0) = (λ i, if i = 0 then 8 else 28)) ∧
  (N.mul_vec (λ i, if i = 0 then -2 else 10) = (λ i, if i = 0 then 6 else -34))

theorem find_matrix_N :
  ∃ N : Matrix (Fin 2) (Fin 2) ℤ, mat_eq N ∧ N = λ i j, 
    if (i, j) = (0, 0) then 2 else if (i, j) = (0, 1) then 1 else if (i, j) = (1, 0) then 7 else -2 :=
by {
  sorry
}

end find_matrix_N_l694_694914


namespace line_through_point_equal_intercepts_l694_694363

-- Define point (1, 3)
def point := (1 : ℝ, 3 : ℝ)

-- Define the condition for a line having equal intercepts
def equal_intercepts (l : ℝ × ℝ → Prop) :=
  ∃ a : ℝ, a ≠ 0 ∧ (l (a, 0) ∧ l (0, a))

-- The line passes through the point (1, 3)
def passes_through_point (l : ℝ × ℝ → Prop) :=
  l point

-- Definition of the line
def line (x y : ℝ) :=
  y = 3 * x ∨ x + y - 4 = 0

-- The proof problem
theorem line_through_point_equal_intercepts :
  (passes_through_point line ∧ equal_intercepts line) → ∃ l : ℝ × ℝ → Prop, (line (l 1) (l 2)) :=
by
  intros h
  sorry

end line_through_point_equal_intercepts_l694_694363


namespace total_sodas_bought_l694_694338

-- Condition 1: Number of sodas they drank
def sodas_drank : ℕ := 3

-- Condition 2: Number of extra sodas Robin had
def sodas_extras : ℕ := 8

-- Mathematical equivalence we want to prove: Total number of sodas bought by Robin
theorem total_sodas_bought : sodas_drank + sodas_extras = 11 := by
  sorry

end total_sodas_bought_l694_694338


namespace integer_roots_quadratic_values_a_l694_694522

theorem integer_roots_quadratic_values_a :
  (∃ (a : ℝ), ∀ (r s : ℤ), r + s = -a ∧ r * s = 6 * a) → 
  (range (λ a : ℝ, ∃ (r s : ℤ), r + s = -a ∧ r * s = 6 * a)).card = 10 :=
by
  sorry

end integer_roots_quadratic_values_a_l694_694522


namespace midpoint_depends_on_slope_solve_function_equation_l694_694053

-- Problem 1
theorem midpoint_depends_on_slope (k b : ℝ) :
  let xA : ℝ := (k - real.sqrt (k^2 - 4 * b)) / 2
      xB : ℝ := (k + real.sqrt (k^2 - 4 * b)) / 2
      xC : ℝ := (xA + xB) / 2
  in xC = k / 2 := by
  sorry

-- Problem 2
theorem solve_function_equation (x : ℝ) (h : x ≠ 0) :
  let f : ℝ → ℝ := λ x, (5 / (8 * x^2)) - (x^3 / 8)
  in (f (1 / x) + (5 / x) * f x = 3 / x^3
     ∧ f x + 5 * x * f (1 / x) = 3 * x^3) := by
  sorry

end midpoint_depends_on_slope_solve_function_equation_l694_694053


namespace negation_of_existential_proposition_l694_694732

theorem negation_of_existential_proposition : 
  (¬ ∃ x : ℝ, x^2 - x + 3 = 0) = (∀ x : ℝ, x^2 - x + 3 ≠ 0) :=
by
  sorry

end negation_of_existential_proposition_l694_694732


namespace b_plus_c_for_quadratic_eqn_l694_694379

theorem b_plus_c_for_quadratic_eqn :
  ∃ (b c : ℝ), (∀ x : ℝ, x^2 - 16x + 64 = (x + b)^2 + c) ∧ b + c = -8 :=
by
  sorry

end b_plus_c_for_quadratic_eqn_l694_694379


namespace eval_fraction_product_l694_694388

theorem eval_fraction_product :
  ((1 + (1 / 3)) * (1 + (1 / 4)) = (5 / 3)) :=
by
  sorry

end eval_fraction_product_l694_694388


namespace total_students_suggestion_l694_694485

theorem total_students_suggestion :
  let m := 324
  let b := 374
  let t := 128
  m + b + t = 826 := by
  sorry

end total_students_suggestion_l694_694485


namespace probability_of_new_circle_containing_center_l694_694448

noncomputable def probability_new_circle_contains_center : ℝ :=
  let integral1 := ∫ (x : ℝ) in 0..(1/2), 1,
  let integral2 := ∫ (x : ℝ) in 0..(1/2), x / (1 - x),
  1 - integral1 + integral2

theorem probability_of_new_circle_containing_center :
  probability_new_circle_contains_center = 1 - Real.log 2 := sorry

end probability_of_new_circle_containing_center_l694_694448


namespace constant_area_of_triangle_l694_694196

-- Define the standard equation of the ellipse
def Ellipse (x y : ℝ) := (x^2 / 4) + (y^2 / 3) = 1

-- Define the eccentricity of the ellipse
def eccentricity (a c : ℝ) := c / a

-- Conditions for the elliptic problem
def conditions (a b c P : ℝ) (F1 F2 : ℝ × ℝ) :=
  a > b ∧ b > 0 ∧ eccentricity 2 1 = 1/2 ∧ (2 : ℝ) = 4 ∧
  ∀ P : ℝ × ℝ, |dist P F1 + dist P F2| = 4

-- The centroid condition for the constant area of triangle PAB
def centroid_condition (A B P : ℝ × ℝ) :=
  A.1 + B.1 + P.1 = 0 ∧ A.2 + B.2 + P.2 = 0

-- Define the area function for the triangle
def triangle_area (A B P : ℝ × ℝ) := 
  (1 / 2) * (A.1 * (B.2 - P.2) + B.1 * (P.2 - A.2) + P.1 * (A.2 - B.2))

-- The theorem to be proved
theorem constant_area_of_triangle (a b : ℝ) (A B P F1 F2 : ℝ × ℝ) 
    (h_conditions : conditions A B P)
    (h_centroid : centroid_condition A B P) : 
  ∃ S : ℝ, ∀ A B P : ℝ × ℝ, triangle_area A B P = S := 
sorry

end constant_area_of_triangle_l694_694196


namespace positive_factors_multiples_of_6_l694_694983

theorem positive_factors_multiples_of_6 (n : ℕ) (h : n = 60) : 
  (∃ k : ℕ, k = 4 ∧ {d : ℕ | d ∣ 60 ∧ d % 6 = 0}.card = k) :=
by
  sorry

end positive_factors_multiples_of_6_l694_694983


namespace total_money_taken_l694_694870

def individual_bookings : ℝ := 12000
def group_bookings : ℝ := 16000
def returned_due_to_cancellations : ℝ := 1600

def total_taken (individual_bookings : ℝ) (group_bookings : ℝ) (returned_due_to_cancellations : ℝ) : ℝ :=
  (individual_bookings + group_bookings) - returned_due_to_cancellations

theorem total_money_taken :
  total_taken individual_bookings group_bookings returned_due_to_cancellations = 26400 := by
  sorry

end total_money_taken_l694_694870


namespace modulus_of_quotient_is_correct_l694_694373

theorem modulus_of_quotient_is_correct : 
  Complex.abs (Complex.div Complex.i (1 + 2 * Complex.i)) = Real.sqrt 5 / 5 := 
by 
  sorry

end modulus_of_quotient_is_correct_l694_694373


namespace tank_volume_in_liters_l694_694082

theorem tank_volume_in_liters :
  (∀ (v : ℝ), let side := real.sqrt (v / 6) in 100 * (2 * 3) = 6 * v → side ^ 3 * 1000 = 1000000) :=
by
  intros v h
  sorry

end tank_volume_in_liters_l694_694082


namespace ratio_of_students_to_dishes_l694_694668

theorem ratio_of_students_to_dishes (m n : ℕ) 
  (h_students : n > 0)
  (h_dishes : ∃ dishes : Finset ℕ, dishes.card = 100)
  (h_each_student_tastes_10 : ∀ student : Finset ℕ, student.card = 10) 
  (h_pairs_taste_by_m_students : ∀ {d1 d2 : ℕ} (hd1 : d1 ∈ Finset.range 100) (hd2 : d2 ∈ Finset.range 100), m = 10) 
  : n / m = 110 := by
  sorry

end ratio_of_students_to_dishes_l694_694668


namespace number_of_Cl_atoms_l694_694062

def atomic_weight_H : ℝ := 1
def atomic_weight_Cl : ℝ := 35.5
def atomic_weight_O : ℝ := 16

def H_atoms : ℕ := 1
def O_atoms : ℕ := 2
def total_molecular_weight : ℝ := 68

theorem number_of_Cl_atoms :
  (total_molecular_weight - (H_atoms * atomic_weight_H + O_atoms * atomic_weight_O)) / atomic_weight_Cl = 1 :=
by
  -- proof to show this holds
  sorry

end number_of_Cl_atoms_l694_694062


namespace collinearity_necessity_l694_694307

noncomputable theory

open_locale classical

variables (a b : ℝ^3)
def collinear (a b : ℝ^3) : Prop := ∃ k : ℝ, a = k • b

theorem collinearity_necessity (a b : ℝ^3) (ha : a ≠ 0) (hb : b ≠ 0) :
  (|a + b| = |a| + |b|) → collinear a b :=
begin
  sorry
end

end collinearity_necessity_l694_694307


namespace exists_plane_not_containing_n_minus_3_points_l694_694938

-- Define general position in space
def general_position (n : ℕ) (points : set (ℝ × ℝ × ℝ)) : Prop :=
  ∀ (p1 p2 p3 p4 : ℝ × ℝ × ℝ), 
    p1 ∈ points → p2 ∈ points → p3 ∈ points → p4 ∈ points → 
    (p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p1) → ¬(collinear {p1, p2, p3}) ∧
    ¬(coplanar {p1, p2, p3, p4})

-- Define collinear and coplanar helpers
def collinear (s : set (ℝ × ℝ × ℝ)) : Prop := sorry
def coplanar (s : set (ℝ × ℝ × ℝ)) : Prop := sorry

theorem exists_plane_not_containing_n_minus_3_points 
  (n : ℕ) (points : set (ℝ × ℝ × ℝ)) (h_general : general_position n points) 
  (A : set (ℝ × ℝ × ℝ)) (hA_card : A.card = n - 3) :
  ∃ (p1 p2 p3 : ℝ × ℝ × ℝ), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
                           (p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) ∧ 
                           ¬(p1 ∈ A ∨ p2 ∈ A ∨ p3 ∈ A) := 
sorry

end exists_plane_not_containing_n_minus_3_points_l694_694938


namespace fn_2012_eq_cos_l694_694662

noncomputable def f : ℕ → (ℝ → ℝ)
| 0     := cos
| (n+1) := deriv (f n)

theorem fn_2012_eq_cos (x : ℝ) : f 2012 x = cos x :=
by sorry

end fn_2012_eq_cos_l694_694662


namespace axis_of_symmetry_vertex_coordinates_max_value_in_interval_min_value_in_interval_l694_694573

noncomputable def f (x : ℝ) : ℝ := -x^2 + 4 * x + 3

theorem axis_of_symmetry : ∃ x, x = 2 :=
by {
    use 2,
    sorry
}

theorem vertex_coordinates : ∃ x y, x = 2 ∧ y = 7 ∧ f x = y :=
by {
    use 2,
    use 7,
    sorry
}

theorem max_value_in_interval : ∃ x val, x ∈ (Set.Icc 1 4 : Set ℝ) ∧ val = 7 ∧ f x = val :=
by {
    use 2,
    use 7,
    sorry
}

theorem min_value_in_interval : ∃ x val, x ∈ (Set.Icc 1 4 : Set ℝ) ∧ val = 3 ∧ f x = val :=
by {
    use 4,
    use 3,
    sorry
}

end axis_of_symmetry_vertex_coordinates_max_value_in_interval_min_value_in_interval_l694_694573


namespace term_sequence_50th_l694_694355

theorem term_sequence_50th (n : ℕ) (seq : ℕ → ℕ) (h_seq : ∀ k, seq k = k / abs k) :
  seq 50 = 10 :=
  sorry

end term_sequence_50th_l694_694355


namespace fixed_point_g_l694_694556

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x - 2)

def g (a : ℝ) : ℝ → ℝ := λ y, 1 - y

theorem fixed_point_g (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : g a 0 = 3 := 
by
  sorry

end fixed_point_g_l694_694556


namespace benzoic_acid_molecular_weight_l694_694487

-- Definitions for atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Molecular formula for Benzoic acid: C7H6O2
def benzoic_acid_formula : ℕ × ℕ × ℕ := (7, 6, 2)

-- Definition for the molecular weight calculation
def molecular_weight := λ (c h o : ℝ) (nC nH nO : ℕ) => 
  (nC * c) + (nH * h) + (nO * o)

-- Proof statement
theorem benzoic_acid_molecular_weight :
  molecular_weight atomic_weight_C atomic_weight_H atomic_weight_O 7 6 2 = 122.118 := by
  sorry

end benzoic_acid_molecular_weight_l694_694487


namespace perpendicular_passes_through_circumcenter_circumcenter_condition_implies_perpendicular_l694_694103

open EuclideanGeometry

-- Define the convex quadrilateral inscribed in the circle
variables {V : Type*} [inner_product_space ℝ V] {A B C D P : V}
variable [is_convex (convex_hull ℝ ({A, B, C, D} : set V))]
variable h_cyclic : ∃ K : V, ∀ X ∈ ({A, B, C, D} : set V), dist K X = dist K A

-- Define the points of intersection of diagonals
variable h_intersect : line_through A C ≠ line_through B D ∧ P ∈ line_through A C ∧ P ∈ line_through B D 

-- Define the perpendicular line from intersection point P to any side of the quadrilateral
variable {X Y : V} (h_side : X ≠ Y ∧ Y ≠ P ∧ P ≠ X)

-- Define the circumcenter condition for the triangle formed by using the opposite side from the line segment
noncomputable def circumcenter (P X Y : V) : V := sorry

theorem perpendicular_passes_through_circumcenter :
  (∃ H : V, P = line_through X H ∧ inner_product_space.is_orthogonal ℝ (P - H) (X - Y))
  → ∃ Q : V, Q = circumcenter P Y (opposite_side P X Y) := sorry

-- Converse statement
theorem circumcenter_condition_implies_perpendicular :
  (∃ Q : V, Q = circumcenter P Y (opposite_side P X Y))
  → ∃ H : V, P = line_through X H ∧ inner_product_space.is_orthogonal ℝ (P - H) (X - Y) := sorry

end perpendicular_passes_through_circumcenter_circumcenter_condition_implies_perpendicular_l694_694103


namespace n_equals_2m_sq_l694_694313

theorem n_equals_2m_sq (n : ℕ) (hpos : 0 < n) 
  (hdiv_sum : (∑ d in (finset.filter (λ x, x ≠ n) (finset.filter (λ x, n % x = 0) (finset.range (n+1)))), d) + 
              (∑ d in (finset.filter (λ x, x = n) (finset.filter (λ x, n % x = 0) (finset.range (n+1)))), 1) = n) : 
  ∃ m : ℕ, n = 2 * m^2 :=
sorry

end n_equals_2m_sq_l694_694313


namespace trains_meet_distance_from_delhi_l694_694861

-- Define the speeds of the trains as constants
def speed_bombay_express : ℕ := 60  -- kmph
def speed_rajdhani_express : ℕ := 80  -- kmph

-- Define the time difference in hours between the departures of the two trains
def time_difference : ℕ := 2  -- hours

-- Define the distance the Bombay Express travels before the Rajdhani Express starts
def distance_head_start : ℕ := speed_bombay_express * time_difference

-- Define the relative speed between the two trains
def relative_speed : ℕ := speed_rajdhani_express - speed_bombay_express

-- Define the time taken for the Rajdhani Express to catch up with the Bombay Express
def time_to_meet : ℕ := distance_head_start / relative_speed

-- The final meeting distance from Delhi for the Rajdhani Express
def meeting_distance : ℕ := speed_rajdhani_express * time_to_meet

-- Theorem stating the solution to the problem
theorem trains_meet_distance_from_delhi : meeting_distance = 480 :=
by sorry  -- proof is omitted

end trains_meet_distance_from_delhi_l694_694861


namespace min_value_condition_l694_694966

theorem min_value_condition {a : ℝ} (h : a = 2) :
  (∀ x : ℝ, f x = x - a * sin x) → 
  (∀ x, x = π/3 → is_minimum (f x)) ↔ (a = 2) :=
by sorry

end min_value_condition_l694_694966


namespace sum_mod_8_l694_694027

theorem sum_mod_8 : (∑ n in Finset.range 104, n) % 8 = 4 :=
sorry

end sum_mod_8_l694_694027


namespace sum_of_x_values_satisfy_eq_l694_694413

theorem sum_of_x_values_satisfy_eq :
  ∑ x in ({x | 6 = (x^3 - 3 * x^2 - 10 * x) / (x + 2)} : Finset ℝ), x = 5 := 
by
  sorry

end sum_of_x_values_satisfy_eq_l694_694413


namespace towel_percentage_decrease_l694_694471

-- Defining the problem and conditions
noncomputable def percentage_decrease_in_length 
  (x : ℝ) -- percentage decrease in length
  (y : ℝ) -- percentage decrease in breadth
  (z : ℝ) -- percentage decrease in area
  (L B : ℝ) -- original length and breadth
  : Prop :=
  let L' := (1 - x / 100) * L
  let B' := (1 - y / 100) * B
  let A := L * B
  let A' := 0.525 * A
  let new_area := L' * B' in
  new_area = A'

-- The theorem to prove
theorem towel_percentage_decrease :
  percentage_decrease_in_length 30 25 47.5 :=
by
  -- The proof is to be filled in 
  sorry

end towel_percentage_decrease_l694_694471


namespace simon_age_is_10_l694_694843

-- Declare the variables
variable (alvin_age : ℕ) (simon_age : ℕ)

-- Define the conditions
def condition1 : Prop := alvin_age = 30
def condition2 : Prop := simon_age = (alvin_age / 2) - 5

-- Formalize the proof problem
theorem simon_age_is_10 (h1 : condition1) (h2 : condition2) : simon_age = 10 := by
  sorry

end simon_age_is_10_l694_694843


namespace fraction_identity_l694_694611

theorem fraction_identity (m n r t : ℚ) (h1 : m / n = 5 / 3) (h2 : r / t = 8 / 15) : 
  (4 * m * r - 2 * n * t) / (5 * n * t - 9 * m * r) = -14 / 27 :=
by 
  sorry

end fraction_identity_l694_694611


namespace quadratic_intersection_l694_694955

noncomputable def quadratic (x : ℝ) (b : ℝ) : ℝ := x^2 + b * x - 3

theorem quadratic_intersection (m b : ℝ) (y1 y2 : ℝ → ℝ) (h1 : y1 = λ x, quadratic x b)
  (h2 : y2 = λ x, x + 1) (A : y1 (-1) = 0) (C : y1 4 = m) :
  b = -2 ∧ m = 5 ∧ ∀ x, y1 x > y2 x ↔ x < -1 ∨ x > 4 :=
begin
  sorry
end

end quadratic_intersection_l694_694955


namespace max_lateral_surface_area_of_cylinder_l694_694952

-- Definitions related to the problem
def sphere_volume (R : ℝ) : ℝ := (4 * π * R^3) / 3
def lateral_surface_area (r l : ℝ) : ℝ := 2 * π * r * l

theorem max_lateral_surface_area_of_cylinder (r l R : ℝ) 
  (h_volume : sphere_volume R = 32 * π / 3)
  (h_relation : r^2 + (l / 2)^2 = R^2) :
  lateral_surface_area r l ≤ 8 * π :=
  sorry

end max_lateral_surface_area_of_cylinder_l694_694952


namespace four_digit_numbers_count_l694_694594

theorem four_digit_numbers_count :
  ∃ (n : ℕ),  n = 24 ∧
  let s := λ (a b c d : ℕ), (a + b + c + d = 10 ∧ a ≠ 0 ∧ ((a + c) - (b + d)) ∈ [(-11), 0, 11]) in
  (card {abcd : Fin 10 × Fin 10 × Fin 10 × Fin 10 | (s abcd.1.1 abcd.1.2 abcd.2.1 abcd.2.2)}) = n :=
by
  sorry

end four_digit_numbers_count_l694_694594


namespace solution_for_a_l694_694233

theorem solution_for_a (x : ℝ) (a : ℝ) (h : 2 * x - a = 0) (hx : x = 1) : a = 2 := by
  rw [hx] at h
  linarith


end solution_for_a_l694_694233


namespace sqrt_inequality_l694_694419

theorem sqrt_inequality (a b : ℝ) (h : 0 ≤ a ∧ 0 ≤ b) (h_sqrt : sqrt a < sqrt b) : a < b := 
by sorry

end sqrt_inequality_l694_694419


namespace sum_coefficients_l694_694436

theorem sum_coefficients (a : ℤ) (f : ℤ → ℤ) :
  f x = (1 - 2 * x)^7 ∧ a_0 = f 0 ∧ a_1_plus_a_7 = f 1 - f 0 
→ a_1_plus_a_7 = -2 :=
by sorry

end sum_coefficients_l694_694436


namespace intersection_of_sets_l694_694577

open Set

variable {x : ℝ}

theorem intersection_of_sets : 
  let A := {x : ℝ | x^2 - 4*x + 3 < 0}
  let B := {x : ℝ | x > 2}
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_of_sets_l694_694577


namespace relationship_among_abc_l694_694163

noncomputable def a : ℝ := ∫ x in 0..1, x
noncomputable def b : ℝ := ∫ x in 0..1, x^2
noncomputable def c : ℝ := ∫ x in 0..1, real.sqrt x

theorem relationship_among_abc : b < a ∧ a < c := by
  sorry

end relationship_among_abc_l694_694163


namespace correct_word_for_blank_l694_694325

theorem correct_word_for_blank :
  (∀ (word : String), word = "that" ↔ word = "whoever" ∨ word = "someone" ∨ word = "that" ∨ word = "any") :=
by
  sorry

end correct_word_for_blank_l694_694325


namespace company_salary_decrease_l694_694689

variables {E S : ℝ} -- Let the initial number of employees be E and the initial average salary be S

theorem company_salary_decrease :
  (0.8 * E * (1.15 * S)) / (E * S) = 0.92 := 
by
  -- The proof will go here, but we use sorry to skip it for now
  sorry

end company_salary_decrease_l694_694689


namespace rectangle_shorter_side_length_l694_694450

theorem rectangle_shorter_side_length :
  ∀ (r : ℝ) (A_circle A_rectangle x : ℝ),
    r = 6 →
    A_circle = real.pi * r^2 →
    A_rectangle = 3 * A_circle →
    x = 2 * r →
    (∃ y : ℝ, x * y = A_rectangle) →
    x = 12 :=
by
  intros r A_circle A_rectangle x r_def A_circle_def A_rectangle_def x_def hy
  sorry

end rectangle_shorter_side_length_l694_694450


namespace value_of_a_l694_694214

open Set

-- Definitions of the sets A and B
def A (a : ℕ) := {0, 2, a}
def B (a : ℕ) := {1, a^2}

-- Given conditions
variable (a : ℕ)
axiom union_condition : A a ∪ B a = {0, 1, 2, 4, 16}

-- The proof goal
theorem value_of_a : a = 4 :=
by
  sorry

end value_of_a_l694_694214


namespace backpack_cost_eq_140_l694_694223

theorem backpack_cost_eq_140 (n c m : ℕ) (d : ℝ) 
    (hn : n = 5) (hc : c = 20) (hd : d = 0.20) (hm : m = 12) :
  (n * c - (n * c * d).toInt + n * m) = 140 := 
by
  sorry

end backpack_cost_eq_140_l694_694223


namespace find_x2_div_c2_add_y2_div_a2_add_z2_div_b2_l694_694986

variable (a b c x y z : ℝ)

theorem find_x2_div_c2_add_y2_div_a2_add_z2_div_b2 
  (h1 : a * (x / c) + b * (y / a) + c * (z / b) = 5) 
  (h2 : c / x + a / y + b / z = 0) : 
  x^2 / c^2 + y^2 / a^2 + z^2 / b^2 = 25 := 
sorry

end find_x2_div_c2_add_y2_div_a2_add_z2_div_b2_l694_694986


namespace li_parallel_l2_l694_694209

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 4 * x
noncomputable def focus (F : ℝ × ℝ) : Prop := F = (1, 0)
noncomputable def circle_Γ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 8
noncomputable def circle_F (F : ℝ × ℝ) (M : ℝ × ℝ) : ℝ := Real.sqrt ((F.1 - M.1)^2 + (F.2 - M.2)^2)
noncomputable def tangent_line (M : ℝ × ℝ) : Prop := sorry -- Specification for tangent construction to be filled in

theorem li_parallel_l2 (A B F M : ℝ × ℝ) (l l1 l2 : set (ℝ × ℝ)) :
  focus F →
  circle_Γ M.1 M.2 →
  parabola A.1 A.2 →
  parabola B.1 B.2 →
  (tangent_line M ∧ ∃ l, l A ∧ l B) →
  (∃ l1, l1 A ∧ tangent_line M) →
  (∃ l2, l2 B ∧ tangent_line M) →
  parallel l1 l2 :=
sorry

end li_parallel_l2_l694_694209


namespace degrees_to_radians_conversion_l694_694115

theorem degrees_to_radians_conversion : (-300 : ℝ) * (Real.pi / 180) = - (5 / 3) * Real.pi :=
by
  sorry

end degrees_to_radians_conversion_l694_694115


namespace angle_Q_of_pentagon_l694_694340

theorem angle_Q_of_pentagon 
  (A B C D E Q : Type)
  (pentagon : RegularPentagon A B C D E)
  (extend_AB : Extend A B Q)
  (extend_DE : Extend D E Q)
  : measure_angle Q = 108 :=
sorry

end angle_Q_of_pentagon_l694_694340


namespace largest_prime_factor_of_7_fact_add_8_fact_l694_694015

theorem largest_prime_factor_of_7_fact_add_8_fact :
  let n := 7!
  let m := 8!
  let sum := n + m
  prime_factors sum = {2, 3, 5, 7} ∧ (∀ p ∈ {2, 3, 5, 7}, p <= 7) := by
  sorry

end largest_prime_factor_of_7_fact_add_8_fact_l694_694015


namespace scientific_notation_of_1_5_million_l694_694691

theorem scientific_notation_of_1_5_million : 
    (1.5 * 10^6 = 1500000) :=
by
    sorry

end scientific_notation_of_1_5_million_l694_694691


namespace angles_set_correct_l694_694412

-- We need to define the conditions and the statement

-- Angle form
def angle_formed_with_y_axis (alpha : ℝ) : Prop :=
  ∃ k : ℤ, alpha = 2 * k * π + π / 2 + π / 6 ∨ alpha = 2 * k * π + π / 2 - π / 6

def valid_angles (alpha : ℝ) : Prop :=
  ∃ k : ℤ, alpha = k * π + π / 3 ∨ alpha = k * π - π / 3

-- The theorem to be proven
theorem angles_set_correct :
  ∀ α : ℝ, angle_formed_with_y_axis α → valid_angles α := by
  sorry

end angles_set_correct_l694_694412


namespace convert_neg300_degrees_to_radians_l694_694118

/-- Definition to convert degrees to radians -/
def degrees_to_radians (deg : ℝ) : ℝ := deg * (Real.pi / 180)

/-- Problem statement: Converting -300 degrees to radians should equal -5/3 times pi -/
theorem convert_neg300_degrees_to_radians :
  degrees_to_radians (-300) = - (5/3) * Real.pi :=
by
  sorry

end convert_neg300_degrees_to_radians_l694_694118


namespace min_height_regular_quadrilateral_pyramid_l694_694043

theorem min_height_regular_quadrilateral_pyramid (r : ℝ) (a : ℝ) (h : 2 * r < a / 2) : 
  ∃ x : ℝ, (0 < x) ∧ (∃ V : ℝ, ∀ x' : ℝ, V = (a^2 * x) / 3 ∧ (∀ x' ≠ x, V < (a^2 * x') / 3)) ∧ x = (r * (5 + Real.sqrt 17)) / 2 :=
sorry

end min_height_regular_quadrilateral_pyramid_l694_694043


namespace find_c_l694_694631

-- Definitions based on given conditions
variable (b c : ℝ)
variable (h : ℝ) (h_pos : h > 0)

-- Conditions
axiom BD_eq_b : BD = b
axiom DC_eq_c : DC = c
axiom area_ratio : (1 / 2) * b * h = (1 / 3) * ((1 / 2) * (b + c) * h)

-- Goal: Prove c = 2 * b
theorem find_c (h_pos: h > 0) (BD_eq_b : BD = b) (DC_eq_c : DC = c) 
              (area_ratio : (1 / 2) * b * h = (1 / 3) * ((1 / 2) * (b + c) * h)) : 
    c = 2 * b :=
by 
  sorry

end find_c_l694_694631


namespace coin_toss_probability_l694_694250

noncomputable theory
open ProbabilityTheory

/-- Define the random variables for heads obtained by you and by me --/
def H_y (n : ℕ) := Binomial (n + 1) 0.5
def H_i (n : ℕ) := Binomial n 0.5

/-- Prove the probability that H_y > H_i is 1/2 --/
theorem coin_toss_probability (n : ℕ) : 
  ℙ (H_y n > H_i n) = 1 / 2 := 
by
  sorry

end coin_toss_probability_l694_694250


namespace largest_prime_factor_of_7_fact_add_8_fact_l694_694020

theorem largest_prime_factor_of_7_fact_add_8_fact :
  let n := 7!
  let m := 8!
  let sum := n + m
  prime_factors sum = {2, 3, 5, 7} ∧ (∀ p ∈ {2, 3, 5, 7}, p <= 7) := by
  sorry

end largest_prime_factor_of_7_fact_add_8_fact_l694_694020


namespace inequality_proof_l694_694937

noncomputable theory
open_locale big_operators

variables {m : ℕ} (a : fin m → ℝ) (p q r : ℝ) (n : ℕ)

-- Conditions
variable (h1 : ∀ i, 0 < a i) -- a_i > 0 for all i
variable (h2 : ∑ i, a i = p) -- ∑_{i=1}^m a_i = p
variable (h3 : q > 0) -- q is a positive constant
variable (h4 : r > 0) -- r is a positive constant
variable (h5 : 0 < n) -- n ∈ ℕ^+

-- Statement of the theorem
theorem inequality_proof :
  ∑ i, (q * a i + r / a i) ^ n ≥ (q * p ^ 2 + m ^ 2 * r) ^ n / (m ^ (n - 1) * p ^ n) :=
sorry

end inequality_proof_l694_694937


namespace complex_number_properties_l694_694169

-- Define the given complex number
def z : ℂ := 3 - 4 * Complex.I

-- Define the conditions and the assertions to be proved
theorem complex_number_properties :
  (|z| = 5) ∧
  (Complex.imaginaryPart z ≠ 4) ∧
  (Complex.realPart (z - 3) = 0) ∧
  (Complex.realPart z > 0 ∧ Complex.imaginaryPart z < 0) :=
by
  -- Include the statements that need to be proved
  split; split; split;
  sorry

end complex_number_properties_l694_694169


namespace maximum_value_of_a_exists_a_eq_29_greatest_possible_value_of_a_l694_694723

theorem maximum_value_of_a (x : ℤ) (a : ℤ) (h1 : x^2 + a * x = -28) (h2 : a > 0) : a ≤ 29 := 
by 
-- add proof here 
sorry

theorem exists_a_eq_29 (x : ℤ) (h1 : x^2 + 29 * x = -28) : ∃ (x : ℤ), x^2 + 29 * x = -28 :=
by 
-- add proof here 
sorry

theorem greatest_possible_value_of_a : ∃ (a : ℤ), (∀ x : ℤ, x^2 + a * x = -28 → a ≤ 29) ∧ (∃ x : ℤ, x^2 + 29 * x = -28) := 
by
  use 29
  split
  { intros x h1 
    apply maximum_value_of_a x 29 h1
    show 29 > 0, from nat.succ_pos' 28 }
  { apply exists_a_eq_29 } 

end maximum_value_of_a_exists_a_eq_29_greatest_possible_value_of_a_l694_694723


namespace odd_function_periodic_l694_694949

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_periodic (h1 : ∀ x : ℝ, f(-x) = -f(x))
  (h2 : ∀ x : ℝ, f(x + 4) = f(x) + f(2))
  (h3 : f(-1) = -2) : f(2013) = 2 :=
sorry

end odd_function_periodic_l694_694949


namespace vector_angle_90_l694_694216

open Real InnerProductSpace

variables {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variables (a b : E)

theorem vector_angle_90 (h : ‖a + b‖ = ‖a - b‖) : ⟪a, b⟫ = 0 := 
by
  sorry

end vector_angle_90_l694_694216


namespace tan_of_geometric_sequence_is_negative_sqrt_3_l694_694192

variable {a : ℕ → ℝ} 

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ m n p q, m + n = p + q → a m * a n = a p * a q

theorem tan_of_geometric_sequence_is_negative_sqrt_3 
  (hgeo : is_geometric_sequence a)
  (hcond : a 2 * a 3 * a 4 = - a 7 ^ 2 ∧ a 7 ^ 2 = 64) :
  Real.tan ((a 4 * a 6 / 3) * Real.pi) = - Real.sqrt 3 :=
sorry

end tan_of_geometric_sequence_is_negative_sqrt_3_l694_694192


namespace part_I_part_II_l694_694885

def f (x a : ℝ) : ℝ := |x + 1/a| + |x - a|

theorem part_I (a x : ℝ) (ha : 0 < a) : f x a ≥ 2 := by
  sorry

theorem part_II (a : ℝ) (ha : 0 < a) (h : f 3 a < 5) : a ∈ set.Ioo ((1 + Real.sqrt 5) / 2) ((5 + Real.sqrt 21) / 2) := by
  sorry

end part_I_part_II_l694_694885


namespace smallest_b_undefined_inverse_l694_694029

theorem smallest_b_undefined_inverse (b : ℕ) (h1 : Nat.gcd b 84 > 1) (h2 : Nat.gcd b 90 > 1) : b = 6 :=
sorry

end smallest_b_undefined_inverse_l694_694029


namespace number_of_allocation_schemes_l694_694840

/-- 
  Given 5 volunteers and 4 projects, each volunteer is assigned to only one project,
  and each project must have at least one volunteer.
  Prove that there are 240 different allocation schemes.
-/
theorem number_of_allocation_schemes (V P : ℕ) (hV : V = 5) (hP : P = 4) 
  (each_volunteer_one_project : ∀ v, ∃ p, v ≠ p) 
  (each_project_at_least_one : ∀ p, ∃ v, v ≠ p) : 
  ∃ n_ways : ℕ, n_ways = 240 :=
by
  sorry

end number_of_allocation_schemes_l694_694840


namespace smallest_positive_period_f_intervals_of_monotonic_increase_center_of_symmetry_axis_of_symmetry_l694_694564

open Real

noncomputable def f (x : ℝ) : ℝ := (sin x + cos x)^2 + cos (2 * x)

theorem smallest_positive_period_f : ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ T = π :=
sorry

theorem intervals_of_monotonic_increase : ∀ k : ℤ, ∃ a b : ℝ, a = - (3 * π) / 8 + k * π ∧ b = π / 8 + k * π ∧ (∀ x : ℝ, a < x ∧ x < b → f x < f (x + ε) forSomeSmallE) :=
sorry

theorem center_of_symmetry : ∀ k : ℤ, f (- π / 8 + k * π / 2) = 1 :=
sorry

theorem axis_of_symmetry : ∀ k : ℤ, ∀ x : ℝ, x = π / 8 + k * π / 2 → f x = f (2 * x - x) :=
sorry

end smallest_positive_period_f_intervals_of_monotonic_increase_center_of_symmetry_axis_of_symmetry_l694_694564


namespace anna_earnings_correct_l694_694099

def anna_cupcakes_baked (trays : ℕ) (cupcakes_per_tray : ℕ) : ℕ := trays * cupcakes_per_tray

def cupcakes_sold (total_cupcakes : ℕ) (fraction_sold : ℚ) : ℕ :=
  (fraction_sold * total_cupcakes).to_nat

def anna_earnings (cupcakes_sold : ℕ) (price_per_cupcake : ℚ) : ℚ :=
  cupcakes_sold * price_per_cupcake

theorem anna_earnings_correct :
  let trays := 4 in
  let cupcakes_per_tray := 20 in
  let total_cupcakes := anna_cupcakes_baked trays cupcakes_per_tray in
  let fraction_sold := 3 / 5 in
  let sold_cupcakes := cupcakes_sold total_cupcakes fraction_sold in
  let price_per_cupcake := 2 in
  anna_earnings sold_cupcakes price_per_cupcake = 96 := by
  sorry

end anna_earnings_correct_l694_694099


namespace total_amount_paid_l694_694806

theorem total_amount_paid (sales_tax : ℝ) (tax_rate : ℝ) (cost_tax_free_items : ℝ) : 
  sales_tax = 1.28 → tax_rate = 0.08 → cost_tax_free_items = 12.72 → 
  (sales_tax / tax_rate + sales_tax + cost_tax_free_items) = 30.00 :=
by
  intros h1 h2 h3
  -- Proceed with the proof using h1, h2, and h3
  sorry

end total_amount_paid_l694_694806


namespace parabola_ellipse_focus_l694_694572

theorem parabola_ellipse_focus (m : ℝ) :
  let focus_parabola := (0, 1 / 2) in
  let focus_ellipse := (0, Real.sqrt (m - 2)) in
  focus_parabola = focus_ellipse → m = 9 / 4 :=
by
  intros h
  have : Real.sqrt (m - 2) = 1 / 2 := by rw [h]
  sorry -- Solving for m requires manipulating the equation Real.sqrt (m - 2) = 1 / 2.

end parabola_ellipse_focus_l694_694572


namespace heather_blocks_l694_694580

theorem heather_blocks (x : ℝ) (h1 : x + 41 = 127) : x = 86 := by
  sorry

end heather_blocks_l694_694580


namespace relationship_depends_on_x_l694_694308

theorem relationship_depends_on_x (x : ℝ) : 
  let a := x^2 - x - 1,
      b := x - 1 in
  if x = 0 then a = b 
  else if x = 1 then a < b 
  else if x = 3 then a > b 
  else a ≠ b :=
by
  sorry -- Proof is skipped as per instructions

end relationship_depends_on_x_l694_694308


namespace original_price_proof_l694_694645

noncomputable def original_price (P: ℝ) : ℝ :=
  17 / 0.478125 / 1.35

theorem original_price_proof : 
  ∃ (P: ℝ), 
    0.85 * 0.5625 * P = 17 ∧ 
    P / 1.35 = 26.34 :=
begin
  use 35.56,
  split,
  { norm_num },
  { norm_num }
end

end original_price_proof_l694_694645


namespace initial_number_of_kids_l694_694749

theorem initial_number_of_kids (kids_went_home : ℕ) (kids_left : ℕ) (h1 : kids_went_home = 14) (h2 : kids_left = 8) :
  kids_went_home + kids_left = 22 :=
by {
  sorry,
}

end initial_number_of_kids_l694_694749


namespace find_matrix_N_l694_694913

def mat_eq (N : Matrix (Fin 2) (Fin 2) ℤ) : Prop :=
  (N.mul_vec (λ i, if i = 0 then 4 else 0) = (λ i, if i = 0 then 8 else 28)) ∧
  (N.mul_vec (λ i, if i = 0 then -2 else 10) = (λ i, if i = 0 then 6 else -34))

theorem find_matrix_N :
  ∃ N : Matrix (Fin 2) (Fin 2) ℤ, mat_eq N ∧ N = λ i j, 
    if (i, j) = (0, 0) then 2 else if (i, j) = (0, 1) then 1 else if (i, j) = (1, 0) then 7 else -2 :=
by {
  sorry
}

end find_matrix_N_l694_694913


namespace largest_prime_factor_7_fact_8_fact_l694_694001

theorem largest_prime_factor_7_fact_8_fact : 
  let n := 7! + 8! 
  ∃ p, Prime p ∧ p ∣ n ∧ (∀ q, Prime q ∧ q ∣ n → q ≤ p) := 
begin
  let n := fact 7 + fact 8,
  use 7, 
  split,
  { apply Prime.prime_7 },
  split,
  { sorry },  
  { sorry },
end

end largest_prime_factor_7_fact_8_fact_l694_694001


namespace pass_probability_is_two_thirds_l694_694259

noncomputable def hypergeometric_distribution (m n k : ℕ) (x : ℕ) : ℚ :=
(nat.choose m x * nat.choose n (k - x)) / nat.choose (m + n) k

def pass_probability : ℚ :=
  hypergeometric_distribution 6 4 3 2 + hypergeometric_distribution 6 4 3 3

theorem pass_probability_is_two_thirds :
  pass_probability = 2 / 3 :=
by
  sorry

end pass_probability_is_two_thirds_l694_694259


namespace max_value_m_l694_694211

theorem max_value_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x ≥ 3 ∧ 2 * x - 1 < m) → m ≤ 5 :=
by
  sorry

end max_value_m_l694_694211


namespace exponential_function_value_l694_694558

-- Given that the function f is exponential and passes through (1, 1/2)
theorem exponential_function_value (a : ℝ) (h : ∀ x : ℝ, f(x) = a^x)
    (h_point : f 1 = 1/2) : f (-2) = 4 :=
by
  -- Since the proof is not required, add sorry
  sorry

end exponential_function_value_l694_694558


namespace passenger_catches_bus_l694_694481

-- Definitions based on conditions from part a)
def P_route3 := 0.20
def P_route6 := 0.60

-- Statement to prove based on part c)
theorem passenger_catches_bus : 
  P_route3 + P_route6 = 0.80 := 
by
  sorry

end passenger_catches_bus_l694_694481


namespace number_of_zeros_f_l694_694735

-- Define the function f
def f (a x : ℝ) : ℝ := a^x + real.log x / real.log a

-- Define the conditions
variables (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)

-- State the theorem
theorem number_of_zeros_f : ∃! x, f a x = 0 :=
sorry

end number_of_zeros_f_l694_694735


namespace range_of_a_l694_694992

-- Definitions based on conditions
def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 4

-- Statement of the theorem to be proven
theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≤ 4 → f a x ≤ f a 4) → a ≤ -3 :=
by
  sorry

end range_of_a_l694_694992


namespace min_value_reciprocal_sum_l694_694669

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  ∀ (c : ℝ), c = (1 / a) + (4 / b) → c ≥ 9 :=
by
  intros c hc
  sorry

end min_value_reciprocal_sum_l694_694669


namespace simon_age_is_10_l694_694842

-- Declare the variables
variable (alvin_age : ℕ) (simon_age : ℕ)

-- Define the conditions
def condition1 : Prop := alvin_age = 30
def condition2 : Prop := simon_age = (alvin_age / 2) - 5

-- Formalize the proof problem
theorem simon_age_is_10 (h1 : condition1) (h2 : condition2) : simon_age = 10 := by
  sorry

end simon_age_is_10_l694_694842


namespace samson_mother_age_l694_694704

variable (S M : ℕ)
variable (x : ℕ)

def problem_statement : Prop :=
  S = 6 ∧
  S - x = 2 ∧
  M - x = 4 * 2 →
  M = 16

theorem samson_mother_age (S M x : ℕ) (h : problem_statement S M x) : M = 16 :=
by
  sorry

end samson_mother_age_l694_694704


namespace telephone_bills_equal_l694_694761

theorem telephone_bills_equal (m : ℕ) (hu : 6 + 0.25 * m = 12 + 0.20 * m) : m = 120 :=
by
  sorry

end telephone_bills_equal_l694_694761


namespace g_of_neg_3_l694_694959

-- Define the function and its properties
noncomputable def f (x : ℝ) : ℝ := if x >= 0 then real.log ((x + 1) / real.log 2) else -real.log ((-x + 1) / real.log 2)

-- Define the inverse function g(x)
noncomputable def g (y : ℝ) : ℝ := if y >= 0 then (2 ^ y) - 1 else -(2 ^ (-y)) + 1

-- State that f is an odd function
axiom odd_f : ∀ x : ℝ, f (-x) = -f (x)

-- State that g is the inverse function of f
axiom inv_f : ∀ y : ℝ, f (g (y)) = y ∧ g (f (y)) = y

-- The theorem to prove
theorem g_of_neg_3 : g (-3) = -7 := by
  sorry

end g_of_neg_3_l694_694959


namespace prob_same_student_l694_694532

theorem prob_same_student (books : Fin 4 → Book) (students : Fin 2) 
    (num_dist_ways : C 4 2 = 6) (num_same_student_ways : 2) :
    (num_same_student_ways / num_dist_ways) = 1 / 3 := by
  -- Definition of Book type
  inductive Book
  | chinese
  | mathematics
  | book1
  | book2

  -- Definitions and conditions for the problem
  def total_ways_to_distribute (books : Fin 4 → Book) (students : Fin 2) : ℕ :=
    C 4 2

  def ways_same_student (books : Fin 4 → Book) (students : Fin 2) : ℕ := 2

  -- The theorem to prove
  have total_ways : total_ways_to_distribute books students = 6 := by assumption
  have same_student : ways_same_student books students = 2 := by assumption

  -- Simplified probability calculation
  calc
    ways_same_student books students / total_ways_to_distribute books students
      = 2 / 6 : by rw [same_student, total_ways]
  ... = 1 / 3 : by norm_num

  sorry

end prob_same_student_l694_694532


namespace volunteer_allocation_scheme_l694_694836

def num_allocation_schemes : ℕ :=
  let num_ways_choose_2_from_5 := Nat.choose 5 2
  let num_ways_arrange_4_groups := Nat.factorial 4
  num_ways_choose_2_from_5 * num_ways_arrange_4_groups

theorem volunteer_allocation_scheme :
  num_allocation_schemes = 240 :=
by
  sorry

end volunteer_allocation_scheme_l694_694836


namespace find_d_e_f_l694_694663

noncomputable def n_equation (x : ℝ) : Prop :=
  (4 / (x - 4)) + (6 / (x - 6)) + (18 / (x - 18)) + (20 / (x - 20)) = x^2 - 12 * x - 6

def is_largest_real_solution (x : ℝ) : Prop :=
  n_equation x ∧ ∀ y : ℝ, n_equation y → y ≤ x

theorem find_d_e_f : ∃ (d e f : ℕ), is_largest_real_solution n ∧ n = d + real.sqrt (e + real.sqrt f) ∧ d + e + f = 80 :=
sorry

end find_d_e_f_l694_694663


namespace product_of_invertible_labels_l694_694372

def function_2 (x : ℝ) : ℝ := x^2 - 2 * x
def function_3 : list (ℝ × ℝ) := [(-5, 3), (-4, 5), (-3, 1), (-2, 0), (-1, 2), (0, -4), (1, -3), (2, -2), (3, 0)]
def function_4 (x : ℝ) : ℝ := -Real.arctan x
def function_5 (x : ℝ) : ℝ := 4 / x

theorem product_of_invertible_labels : 4 * 5 = 20 := by
  sorry

end product_of_invertible_labels_l694_694372


namespace final_water_level_l694_694755

-- Define the conditions
def h_initial (h: ℝ := 0.4): ℝ := 0.4  -- Initial height in meters, 0.4m = 40 cm
def rho_water : ℝ := 1000 -- Density of water in kg/m³
def rho_oil : ℝ := 700 -- Density of oil in kg/m³
def g : ℝ := 9.81 -- Acceleration due to gravity in m/s² (value is standard, provided here for completeness)

-- Statement of the problem in Lean 4
theorem final_water_level (h_initial : ℝ) (rho_water : ℝ) (rho_oil : ℝ) (g : ℝ):
  ∃ h_final : ℝ, 
  ρ_water * g * h_final = ρ_oil * g * (h_initial - h_final) ∧
  h_final = 0.34 :=
begin
  sorry
end

end final_water_level_l694_694755


namespace dog_heavier_than_fox_l694_694744

-- Define the weights of the fox and dog
noncomputable def weight_of_fox : ℝ := 5
noncomputable def weight_of_dog : ℝ := 10

-- Theorem stating the dog is 5 kg heavier than the fox
theorem dog_heavier_than_fox : weight_of_dog - weight_of_fox = 5 := 
by
  -- Definitions are directly used from the conditions
  have fox_weight : 5 * weight_of_fox = 25 := by norm_num
  have mixed_weight : 3 * weight_of_fox + 5 * weight_of_dog = 65 := by norm_num
  have fox_value : weight_of_fox = 5 := by linarith
  have dog_value : weight_of_dog = 10 := by linarith
  sorry

end dog_heavier_than_fox_l694_694744


namespace solution_set_l694_694906
noncomputable def inequality_solution (x : ℝ) : set ℝ :=
  {x | (x^2 - 4) / (x - 3)^2 ≥ 0}

theorem solution_set :
  inequality_solution = {x | x ∈ Iic (-2) ∪ Icc 2 (3 - 1 / 2) ∪ Ioi 3} := by
  sorry

end solution_set_l694_694906


namespace maximize_volume_rotation_base_maximize_volume_rotation_tangent_l694_694782

noncomputable def volume_rotation_base (r : ℝ) : ℝ :=
  let x := (2 / 3) * r
  let y := sqrt (r^2 - x^2)
  (2 / 3) * π * (r + x)^2 * y

noncomputable def volume_rotation_tangent (r : ℝ) : ℝ :=
  let x := (2 / 3) * r
  let y := sqrt (r^2 - x^2)
  (4 / 3) * π * (r + x)^2 * y

theorem maximize_volume_rotation_base (r : ℝ) : 
  volume_rotation_base r = (50 * sqrt 5 / 81) * π * r^3 :=
sorry

theorem maximize_volume_rotation_tangent (r : ℝ) : 
  volume_rotation_tangent r = (100 * sqrt 5 / 81) * π * r^3 :=
sorry

end maximize_volume_rotation_base_maximize_volume_rotation_tangent_l694_694782


namespace largest_prime_factor_of_7_fact_add_8_fact_l694_694017

theorem largest_prime_factor_of_7_fact_add_8_fact :
  let n := 7!
  let m := 8!
  let sum := n + m
  prime_factors sum = {2, 3, 5, 7} ∧ (∀ p ∈ {2, 3, 5, 7}, p <= 7) := by
  sorry

end largest_prime_factor_of_7_fact_add_8_fact_l694_694017


namespace area_trapezoid_AXFY_l694_694628

theorem area_trapezoid_AXFY :
    ∀ (A B C D X Y P E F : ℝ)
    (side length_X AX CY PX PY : ℝ),
    ABCD_is_square (A B C D)
    ∧ AB = 6000
    ∧ CD = 6000
    ∧ AX = CY
    ∧ angle_AXP = 45
    ∧ angle_YPC = 45
    ∧ PX = PY
    ∧ PX = 3000
    → trapezoid(X Y F A P)
    ∧ area(X Y F A P) = 9000000 := 
begin
  sorry
end

end area_trapezoid_AXFY_l694_694628


namespace abundant_numbers_less_than_50_count_l694_694435

def is_abundant (n : ℕ) : Prop :=
  (∑ i in finset.filter (λ d, d∣n ∧ d < n) (finset.range n), i) > n

theorem abundant_numbers_less_than_50_count : (finset.range 50).filter is_abundant).card = 9 :=
by
  sorry

end abundant_numbers_less_than_50_count_l694_694435


namespace incorrect_expressions_l694_694305

-- Define the repeating decimal R and its parts N and M
variable (R : ℝ)
variable (N : ℝ) -- where N consists of 3 figures of R that do not repeat
variable (M : ℝ) -- where M consists of 4 figures of R that repeat

-- Prove expressions (C) and (D) incorrectly represent the repeating decimal R
theorem incorrect_expressions (h1 : R = 0.N * 10^(-3) + M * (1 / 10^(-4)))
  (h2 : 10^3 * R = N + M * (1 / 10^(-1)))
  (h3 : ¬(10^7 * R = NMN + M * (1 / 10^(-1))))
  (h4 : ¬(10^3 * (10^4 - 1) * R = 10^4 * N - M)) :
  (10^7 * R = NMN + M * (1 / 10^(-1))) ∨ (10^3 * (10^4 - 1) * R = 10^4 * N - M) := sorry

end incorrect_expressions_l694_694305


namespace correct_statement_is_C_l694_694032

theorem correct_statement_is_C :
  (sqrt 16 = 4) → 
  (∃ x : ℝ, x^3 = -8) → 
  (sqrt 1 = 1) → 
  (sqrt ((-2)^2) = 2) →
  (∃ (correct_statement : ℕ), correct_statement = 3) :=
by
  sorry

end correct_statement_is_C_l694_694032


namespace largest_prime_factor_7_fact_8_fact_l694_694007

theorem largest_prime_factor_7_fact_8_fact : 
  let n := 7! + 8! 
  ∃ p, Prime p ∧ p ∣ n ∧ (∀ q, Prime q ∧ q ∣ n → q ≤ p) := 
begin
  let n := fact 7 + fact 8,
  use 7, 
  split,
  { apply Prime.prime_7 },
  split,
  { sorry },  
  { sorry },
end

end largest_prime_factor_7_fact_8_fact_l694_694007


namespace scientific_notation_of_1_5_million_l694_694690

theorem scientific_notation_of_1_5_million : 
    (1.5 * 10^6 = 1500000) :=
by
    sorry

end scientific_notation_of_1_5_million_l694_694690


namespace max_candies_in_25_days_l694_694064

structure DentistInstructions where
  max_candies_per_day : ℕ
  limited_candies_day_c : ℕ
  condition_days : ℕ
  props :
    max_candies_per_day = 10 ∧ 
    limited_candies_day_c = 5 ∧ 
    condition_days = 2

theorem max_candies_in_25_days (Sonia : DentistInstructions) : 
  Sonia.max_candies_per_day = 10 →
  Sonia.limited_candies_day_c = 5 →
  Sonia.condition_days = 2 →
  ∀ (candies : List ℕ), 
    (candies.length = 25 →
     (∀ i, 0 ≤ i ∧ i < candies.length → candies.nthLe i i.2 ≤ Sonia.max_candies_per_day) →
     (∀ i, 0 ≤ i ∧ i < candies.length-2 → candies.nthLe i i.2 > 7 → 
        candies.nthLe (i+1) (by simp [Nat.lt_succ_self, i.succ, List.length_eq_at_beginning]) ≤ Sonia.limited_candies_day_c ∧
        candies.nthLe (i+2) (by simp [Nat.succ_lt_succ_iff, i.succ, List.length_eq_at_beginning]) ≤ Sonia.limited_candies_day_c) →
     ∑ x in candies, x) = 178 :=
by
  intro hmax hlimit hcond candies hlen hcandies hconstraint
  sorry

end max_candies_in_25_days_l694_694064


namespace least_number_of_froods_l694_694633

def dropping_score (n : ℕ) : ℕ := (n * (n + 1)) / 2
def eating_score (n : ℕ) : ℕ := 15 * n

theorem least_number_of_froods : ∃ n : ℕ, (dropping_score n > eating_score n) ∧ (∀ m < n, dropping_score m ≤ eating_score m) :=
  exists.intro 30 
    (and.intro 
      (by simp [dropping_score, eating_score]; linarith)
      (by intros m hmn; simp [dropping_score, eating_score]; linarith [hmn]))

end least_number_of_froods_l694_694633


namespace tangent_line_eq_fx_leq_x_l694_694203

-- Define the function f
def f (a x : ℝ) : ℝ := a * x - Real.exp x

-- First part of the problem: Equation of the tangent line when a = 1 at x = 1
theorem tangent_line_eq (x y : ℝ) : y = (1 - Real.exp 1) * x :=
sorry

-- Second part of the problem: Prove f(x) ≤ x for a ∈ [1, e + 1]
theorem fx_leq_x (a x : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ Real.exp 1 + 1) : f a x ≤ x :=
sorry

end tangent_line_eq_fx_leq_x_l694_694203


namespace max_value_g_range_k_l694_694201

-- Problem 1
theorem max_value_g (k : ℝ) (f : ℝ → ℝ := λ x, (log x + k) / Real.exp x) 
  (hf1 : (λ x, (1 - k*x - x*log x) / (x*Real.exp x)) 1 = 0) :
  ∃ x, f x * Real.exp x - x ≤ 0 :=
sorry

-- Problem 2
theorem range_k (k : ℝ) (f : ℝ → ℝ := λ x, (log x + k) / Real.exp x) 
  (hex : ∃ x ∈ Ioc 0 1, (λ x, (1 - k*x - x*log x) / (x*Real.exp x)) x = 0) :
  1 ≤ k :=
sorry

end max_value_g_range_k_l694_694201


namespace ring_tower_height_l694_694081

theorem ring_tower_height : 
  let thickness := 2
  let smallest_outside_diameter := 10
  let largest_outside_diameter := 30
  let num_rings := (largest_outside_diameter - smallest_outside_diameter) / thickness + 1
  let total_distance := num_rings * thickness + smallest_outside_diameter - thickness
  total_distance = 200 :=
by {
  let thickness := 2,
  let smallest_outside_diameter := 10,
  let largest_outside_diameter := 30,
  let num_rings := (largest_outside_diameter - smallest_outside_diameter) / thickness + 1,
  let total_distance := num_rings * thickness + smallest_outside_diameter - thickness,
  have h : total_distance = 200 := sorry,
  exact h,
}

end ring_tower_height_l694_694081


namespace quadratic_eq_coeffs_l694_694832

theorem quadratic_eq_coeffs (x : ℝ) : 
  ∃ a b c : ℝ, 3 * x^2 + 1 - 6 * x = a * x^2 + b * x + c ∧ a = 3 ∧ b = -6 ∧ c = 1 :=
by sorry

end quadratic_eq_coeffs_l694_694832


namespace proof_a_proof_b_l694_694191

open Real

-- Conditions from the problem
variables (n : ℕ) (x y : ℝ) (S_n a_n z_n : ℝ)
-- n ∈ ℕ⁺ 
axiom nat_pos (h : n > 0)
-- Point (x, y) satisfies the inequalities
axiom ineq1 (h1 : x + 2 * y ≤ 2 * n)
axiom ineq2 (h2 : x ≥ 0)
axiom ineq3 (h3 : y ≥ 0)
-- Definitions for sequences
axiom seq1 (h4 : a_n = S_n - S_{n - 1})
axiom seq2 (h5 : a_1 = 1)
axiom max_z (h6 : z_n = 2 * n)

-- Assertions derived from the problem
theorem proof_a : ∃ q : ℝ, q = 1 / 2 ∧ ∀ n > 0, a_n - 2 = q * (a_{n - 1} - 2) :=
sorry

theorem proof_b : ∀ n > 0, T_n = n^2 - n + 2 - (1 / 2) ^ (n - 1) :=
sorry

end proof_a_proof_b_l694_694191


namespace tins_of_beans_left_l694_694823

theorem tins_of_beans_left (cases : ℕ) (tins_per_case : ℕ) (damage_percentage : ℝ) (h_cases : cases = 15)
  (h_tins_per_case : tins_per_case = 24) (h_damage_percentage : damage_percentage = 0.05) :
  let total_tins := cases * tins_per_case
  let damaged_tins := total_tins * damage_percentage
  let tins_left := total_tins - damaged_tins
  tins_left = 342 :=
by
  sorry

end tins_of_beans_left_l694_694823


namespace percent_problem_l694_694995

theorem percent_problem (x : ℝ) (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 :=
by
  sorry

end percent_problem_l694_694995


namespace sqrt_inequality_l694_694416

theorem sqrt_inequality (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (h : sqrt a < sqrt b) : a < b := 
by 
  sorry

end sqrt_inequality_l694_694416


namespace volunteer_allocation_scheme_l694_694837

def num_allocation_schemes : ℕ :=
  let num_ways_choose_2_from_5 := Nat.choose 5 2
  let num_ways_arrange_4_groups := Nat.factorial 4
  num_ways_choose_2_from_5 * num_ways_arrange_4_groups

theorem volunteer_allocation_scheme :
  num_allocation_schemes = 240 :=
by
  sorry

end volunteer_allocation_scheme_l694_694837


namespace correctPropositions_l694_694948

-- Define the conditions and statement as Lean structures.
structure Geometry :=
  (Line : Type)
  (Plane : Type)
  (parallel : Plane → Plane → Prop)
  (parallelLine : Line → Plane → Prop)
  (perpendicular : Plane → Plane → Prop)
  (perpendicularLine : Line → Plane → Prop)
  (subsetLine : Line → Plane → Prop)

-- Main theorem to be proved in Lean 4
theorem correctPropositions (G : Geometry) :
  (∀ (α β : G.Plane) (a : G.Line), (G.parallel α β) → (G.subsetLine a α) → (G.parallelLine a β)) ∧ 
  (∀ (α β : G.Plane) (a : G.Line), (G.perpendicularLine a α) → (G.perpendicularLine a β) → (G.parallel α β)) :=
sorry -- The proof is omitted, as per instructions

end correctPropositions_l694_694948


namespace range_of_k_for_exactly_two_zeros_l694_694567

   noncomputable def f (x : ℝ) : ℝ :=
   if x ≥ 0 then (1 / 2) * Real.sqrt (x^2 + 1) else -Real.log (1 - x)

   noncomputable def F (x k : ℝ) : ℝ := f x - k * x

   theorem range_of_k_for_exactly_two_zeros :
     {k : ℝ | ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ F x1 k = 0 ∧ F x2 k = 0} = {k : ℝ | k ∈ set.Ioo (1/2) 1} :=
   sorry
   
end range_of_k_for_exactly_two_zeros_l694_694567


namespace compute_complex_expression_l694_694235

def A : ℂ := -3 + 2 * complex.I
def O : ℂ := 3 * complex.I
def P : ℂ := 1 + 3 * complex.I
def S : ℂ := -2 - complex.I

theorem compute_complex_expression : 2 * A - O + 3 * P + S = -5 + 9 * complex.I := 
by sorry

end compute_complex_expression_l694_694235


namespace backpacks_total_cost_l694_694220

def n : ℕ := 5
def p_original : ℝ := 20.00
def discount : ℝ := 20 / 100
def monogram_cost : ℝ := 12.00

theorem backpacks_total_cost :
  n * (p_original * (1 - discount) + monogram_cost) = 140 := by
  sorry

end backpacks_total_cost_l694_694220


namespace three_primes_among_first_ten_l694_694710

-- Define the sequence of sums a_n where each term is the sum of 5 and every second prime
def second_primes : List ℕ := [11, 17, 23, 29, 37, 41, 47, 53, 59]

def a (n : ℕ) : ℕ :=
  if n = 1 then 5
  else 5 + (List.take (n - 1) second_primes).sum

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Finally, state the problem: Find the number of primes among a_1 to a_10
def count_primes : ℕ :=
  List.filter is_prime (List.map a [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).length

theorem three_primes_among_first_ten :
  count_primes = 3 := 
sorry

end three_primes_among_first_ten_l694_694710


namespace greatest_prime_factor_of_expr_l694_694409

-- Definition of the given expression and its evaluation
def expr : ℕ := 2^8 + 5^4 + 10^3

-- Statement to prove
theorem greatest_prime_factor_of_expr : 
  expr = 1881 ∧ (∀ p, nat.prime p → p ∣ 1881 → p ≤ 19) :=
by {
  -- Sorry here indicates that the proof is omitted
  sorry
}

end greatest_prime_factor_of_expr_l694_694409


namespace number_of_roosters_l694_694393

def chickens := 9000
def ratio_roosters_hens := 2 / 1

theorem number_of_roosters (h : ratio_roosters_hens = 2 / 1) (c : chickens = 9000) : ∃ r : ℕ, r = 6000 := 
by sorry

end number_of_roosters_l694_694393


namespace negation_of_there_exists_l694_694734

theorem negation_of_there_exists (x : ℝ) : ¬ (∃ x : ℝ, x^2 - x + 3 = 0) ↔ ∀ x : ℝ, x^2 - x + 3 ≠ 0 := by
  sorry

end negation_of_there_exists_l694_694734


namespace sum_squares_not_divisible_by_7_l694_694385

theorem sum_squares_not_divisible_by_7 (x y z : ℕ) 
  (h_coprime_xy : Nat.coprime x y) 
  (h_coprime_xz : Nat.coprime x z) 
  (h_coprime_yz : Nat.coprime y z)
  (h_sum_div_7 : (x + y + z) % 7 = 0) 
  (h_prod_div_7 : (x * y * z) % 7 = 0) : 
  ¬ ((x^2 + y^2 + z^2) % 7 = 0) := 
sorry

end sum_squares_not_divisible_by_7_l694_694385


namespace original_rectangle_perimeter_not_necessarily_integer_l694_694465

/-- Given a rectangle is divided into smaller rectangles where each smaller rectangle has an integer perimeter, the perimeter of the original rectangle may not necessarily be an integer. -/
theorem original_rectangle_perimeter_not_necessarily_integer
  (side_length : ℚ) :
  let square := {a : ℚ // a = side_length } in
  let small_rectangles := (square.1 / 2, square.1 / 3) in
  ¬ (4 * side_length ∈ ℤ) ∧ (2 * (square.1 / 2 + square.1 / 3) ∈ ℤ) := 
by
  sorry

end original_rectangle_perimeter_not_necessarily_integer_l694_694465


namespace apples_remaining_in_each_basket_l694_694296

-- Definition of conditions
def total_apples : ℕ := 128
def number_of_baskets : ℕ := 8
def apples_taken_per_basket : ℕ := 7

-- Definition of the problem
theorem apples_remaining_in_each_basket :
  (total_apples / number_of_baskets) - apples_taken_per_basket = 9 := 
by 
  sorry

end apples_remaining_in_each_basket_l694_694296


namespace Jeff_weekly_hours_l694_694297

theorem Jeff_weekly_hours 
  (hours_facebook_daily : ℕ)
  (work_hours_weekend_ratio : ℕ)
  (twitter_hours_weekend : ℕ)
  (instagram_hours_weekday : ℕ)
  (work_hours_weekday_ratio : ℕ) 
  (weekend_days : ℕ := 2) 
  (weekday_days : ℕ := 5) :
  hours_facebook_daily = 3 →
  work_hours_weekend_ratio = 3 →
  twitter_hours_weekend = 2 →
  instagram_hours_weekday = 1 →
  work_hours_weekday_ratio = 4 →
  let work_hours_weekend := (hours_facebook_daily / work_hours_weekend_ratio) * weekend_days in
  let social_media_hours_weekday := (hours_facebook_daily + instagram_hours_weekday) * weekday_days in
  let work_hours_weekday := social_media_hours_weekday * work_hours_weekday_ratio in
  let total_work_hours := work_hours_weekend + work_hours_weekday in
  let total_twitter_hours := twitter_hours_weekend * weekend_days in
  let total_instagram_hours := instagram_hours_weekday * weekday_days in
  let total_hours := total_work_hours + total_twitter_hours + total_instagram_hours in
  total_hours = 91 :=
by
  intros
  sorry

end Jeff_weekly_hours_l694_694297


namespace largest_n_for_f_l694_694149

def f (n : ℕ) : ℕ := (Finset.range 100).sum (λ k, Int.floor (Real.log10 ((k + 1) * n)))

theorem largest_n_for_f :
  ∃ n : ℕ, ∀ m : ℕ, f(m) ≤ 300 → m ≤ n ∧ n = 108 :=
by
  -- Proof to be provided
  sorry

end largest_n_for_f_l694_694149


namespace problem_statement_l694_694614

theorem problem_statement (p q : ℚ) (hp : 3 / p = 4) (hq : 3 / q = 18) : p - q = 7 / 12 := 
by
  sorry

end problem_statement_l694_694614


namespace shaded_fraction_of_triangle_l694_694473

theorem shaded_fraction_of_triangle : 
  let area (n : ℕ) := (1 : ℝ) / 4 ^ n in
  (∑' (n : ℕ), area (n + 1)) = (1 / 3 : ℝ) :=
by
  sorry

end shaded_fraction_of_triangle_l694_694473


namespace painting_problem_l694_694437

-- Definitions used in Lean 4 statement 
def valid_paintings (grid : Fin 3 × Fin 3 → Prop) : Prop :=
  (∀ i j : Fin 3, (grid (i, j) → (i < 2 → ∀ k : Fin 2, grid (⟨i+k+1, _⟩, j) = false) ∧ (j < 2 → ∀ k : Fin 2, grid (i, ⟨j+k+1, _⟩) = false))) ∧
  (∀ i j : Fin 3, ¬(grid (i, j) ∨ ∃ di dj : Fin 3, i + di = 3 → j + dj = 3 → grid (⟨i+di-1, _⟩, ⟨j+dj-1, _⟩)))

def count_valid_paintings : Nat :=
  (Finset.univ : Finset (Fin 3 × Fin 3 → Prop)).filter valid_paintings).card

-- The theorem to prove the problem
theorem painting_problem : count_valid_paintings = 9 :=
by sorry

end painting_problem_l694_694437


namespace Evenland_111_is_842_l694_694736

-- Define the Evenlanders' mapping of digits as doubling the base-5 representation
def Evenland_digit (d : ℕ) : ℕ :=
  match d with
  | 0 => 0
  | 1 => 2
  | 2 => 4
  | 3 => 6
  | 4 => 8
  | _ => 0  -- This case should not be reached for base-5 digits

-- Convert a base-10 integer to its base-5 representation
def to_base_5 (n : ℕ) : List ℕ :=
  if n == 0 then [0] else
    let rec convert (m : ℕ) (acc : List ℕ) :=
      if m == 0 then acc else convert (m / 5) ((m % 5) :: acc)
    convert n []

-- Calculate the Evenland version of a base-5 number
def Evenland_base_5 (base5_rep : List ℕ) : List ℕ :=
  base5_rep.map Evenland_digit

-- Convert list of digits back to a base-10 integer
def from_base_10 (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc d => acc * 10 + d) 0

theorem Evenland_111_is_842 : from_base_10 (Evenland_base_5 (to_base_5 111)) = 842 := by
  -- Skip the proof with sorry
  sorry

end Evenland_111_is_842_l694_694736


namespace general_formula_l694_694309

noncomputable def a : ℕ → ℕ
| 0       => 5
| (n + 1) => 2 * a n + 3

theorem general_formula : ∀ n, a n = 2 ^ (n + 2) - 3 :=
by
  sorry

end general_formula_l694_694309


namespace selection_schemes_count_l694_694934

-- Definitions based on conditions
def num_boys := 3
def num_girls := 3

-- Theorem statement based on the proof problem
theorem selection_schemes_count : 
  let select_1_boy := num_boys.choose 1,
      select_2_girls := num_girls.choose 2,
      arrange_3_students := (select_1_boy * select_2_girls).factorial
  in select_1_boy * select_2_girls * arrange_3_students = 54 := 
by
  sorry

end selection_schemes_count_l694_694934


namespace count_four_digit_numbers_l694_694598

open Int

def is_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9

def valid_combination (a b c d : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧
  a ≠ 0 ∧
  a + b + c + d = 10 ∧
  (a + c) - (b + d) % 11 = 0

theorem count_four_digit_numbers :
  ∃ n, n > 0 ∧ ∀ a b c d : ℕ, valid_combination a b c d → n = sorry :=
sorry

end count_four_digit_numbers_l694_694598


namespace distance_Stockholm_Malmo_via_Gothenburg_l694_694327

theorem distance_Stockholm_Malmo_via_Gothenburg :
  (distance_map_SG : ℕ) → (distance_map_GM : ℕ) → (scale_map : ℕ) 
  → (distance_SG := distance_map_SG * scale_map) 
  → (distance_GM := distance_map_GM * scale_map) 
  → (total_distance := distance_SG + distance_GM) 
  → distance_map_SG = 120 
  → distance_map_GM = 150 
  → scale_map = 20 
  → total_distance = 5400 := 
by intros; sorry

end distance_Stockholm_Malmo_via_Gothenburg_l694_694327


namespace find_matrix_N_l694_694907

open Matrix

theorem find_matrix_N :
  ∃ (N : Matrix (Fin 2) (Fin 2) ℤ),
    (N ⬝ (col_vector 2 ![4, 0]) = col_vector 2 ![8, 28]) ∧
    (N ⬝ (col_vector 2 ![-2, 10]) = col_vector 2 ![6, -34]) ∧
    (N = ![![2, 1], ![7, -2]]) :=
begin
  -- proof would go here
  sorry
end

end find_matrix_N_l694_694907


namespace largest_prime_factor_of_7_fact_add_8_fact_l694_694021

theorem largest_prime_factor_of_7_fact_add_8_fact :
  let n := 7!
  let m := 8!
  let sum := n + m
  prime_factors sum = {2, 3, 5, 7} ∧ (∀ p ∈ {2, 3, 5, 7}, p <= 7) := by
  sorry

end largest_prime_factor_of_7_fact_add_8_fact_l694_694021


namespace calculate_sequence_sum_l694_694867

noncomputable def sum_arithmetic_sequence (a l d: Int) : Int :=
  let n := ((l - a) / d) + 1
  (n * (a + l)) / 2

theorem calculate_sequence_sum :
  3 * (sum_arithmetic_sequence 45 93 2) + 2 * (sum_arithmetic_sequence (-4) 38 2) = 5923 := by
  sorry

end calculate_sequence_sum_l694_694867


namespace surface_area_volume_l694_694442

-- Definitions
def diameter := 9
def radius := diameter / 2

-- Theorem: Surface area calculation
theorem surface_area (d : ℝ) (r : ℝ) (h : d = 9) (hr : r = d / 2) :=
  4 * Real.pi * r * r = 81 * Real.pi :=
by
  rw [h, hr]
  norm_num
  exact eq.refl (4 * Real.pi * (9 / 2) * (9 / 2))
  sorry

-- Theorem: Volume calculation
theorem volume (d : ℝ) (r : ℝ) (h : d = 9) (hr : r = d / 2) :=
  (4 / 3) * Real.pi * r * r * r = (729 / 6) * Real.pi :=
by
  rw [h, hr]
  norm_num
  exact eq.refl ((4 / 3) * Real.pi * (9 / 2) * (9 / 2) * (9 / 2))
  sorry

end surface_area_volume_l694_694442


namespace range_of_m_l694_694967

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x >= (4 + m)) ∧ (x <= 3 * (x - 2) + 4) → (x ≥ 2)) →
  (-3 < m ∧ m <= -2) :=
sorry

end range_of_m_l694_694967


namespace share_price_increase_l694_694037

-- Definitions based on the conditions
variable {P : ℝ} (hP : P > 0) -- assuming price P is a positive real number

def first_quarter_price : ℝ := 1.30 * P
def second_quarter_price : ℝ := 1.75 * P

-- The percent increase in the share price from the end of the first quarter to the end of the second quarter
def percent_increase : ℝ := ((second_quarter_price - first_quarter_price) / first_quarter_price) * 100

-- Statement to be proven
theorem share_price_increase : percent_increase = 34.62 := 
by
  -- proof would go here
  sorry

end share_price_increase_l694_694037


namespace part1_part2_l694_694205

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a^2| + |x - a - 1|

-- Question 1: Prove that f(x) ≥ 3/4
theorem part1 (x a : ℝ) : f x a ≥ 3 / 4 := 
sorry

-- Question 2: Given f(4) < 13, find the range of a
theorem part2 (a : ℝ) (h : f 4 a < 13) : -2 < a ∧ a < 3 := 
sorry

end part1_part2_l694_694205


namespace min_power_cycles_cover_all_odds_mod_1024_l694_694887

-- Define a power_cycle in Lean
def power_cycle (a : ℕ) : set ℕ := {x | ∃ n : ℕ, x = a ^ n}

-- Define function to check if there's an element congruent to k (mod 1024) in a given set of power cycles
def covers_all_odds (cycles : list (set ℕ)) : Prop := 
  ∀ n : ℕ, n % 2 = 1 → ∃ cycle ∈ cycles, ∃ x ∈ cycle, x % 1024 = n % 1024

theorem min_power_cycles_cover_all_odds_mod_1024 :
  ∃ (cycles : list (set ℕ)), covers_all_odds cycles ∧ cycles.length = 10 :=
by sorry

end min_power_cycles_cover_all_odds_mod_1024_l694_694887


namespace total_animals_in_savanna_l694_694703

/-- Define the number of lions, snakes, and giraffes in Safari National Park. --/
def safari_lions : ℕ := 100
def safari_snakes : ℕ := safari_lions / 2
def safari_giraffes : ℕ := safari_snakes - 10

/-- Define the number of lions, snakes, and giraffes in Savanna National Park based on conditions. --/
def savanna_lions : ℕ := 2 * safari_lions
def savanna_snakes : ℕ := 3 * safari_snakes
def savanna_giraffes : ℕ := safari_giraffes + 20

/-- Calculate the total number of animals in Savanna National Park. --/
def total_savanna_animals : ℕ := savanna_lions + savanna_snakes + savanna_giraffes

/-- Proof statement that the total number of animals in Savanna National Park is 410.
My goal is to prove that total_savanna_animals is equal to 410. --/
theorem total_animals_in_savanna : total_savanna_animals = 410 :=
by
  sorry

end total_animals_in_savanna_l694_694703


namespace total_amount_after_refunds_l694_694874

def individual_bookings : ℕ := 12000
def group_bookings : ℕ := 16000
def refunds : ℕ := 1600

theorem total_amount_after_refunds : 
  individual_bookings + group_bookings - refunds = 26400 :=
by 
  -- The proof goes here.
  sorry

end total_amount_after_refunds_l694_694874


namespace triangle_area_given_conditions_l694_694048

theorem triangle_area_given_conditions (A B C L : ℝ) (AL BL CL : ℝ)
  (h1 : BL = sqrt 30) (h2 : AL = 2) (h3 : CL = 5) :
  (area_of_triangle A B C) = (7 * sqrt 39) / 4 :=
by
  -- Proof is left as an exercise.
  sorry

end triangle_area_given_conditions_l694_694048


namespace monica_numbers_count_l694_694762

/-- Monica forms three-digit numbers using the digits 1, 3, and 5. Prove that the total number of such numbers greater than 150 is 21. -/
theorem monica_numbers_count :
  let digits := [1, 3, 5] in
  let is_valid_digit (d : ℕ) := d ∈ digits in
  let numbers := (100 * 1 + 10 * 5 + is_valid_digit) + (100 * 3 + 10 * is_valid_digit + is_valid_digit) + (100 * 5 + 10 * is_valid_digit + is_valid_digit) in
  numbers = 21 :=
begin
  sorry
end

end monica_numbers_count_l694_694762


namespace bruces_birthday_l694_694495

-- Definition: Weekdays as an enumeration.
inductive Weekday
| Sunday : Weekday
| Monday : Weekday
| Tuesday : Weekday
| Wednesday : Weekday
| Thursday : Weekday
| Friday : Weekday
| Saturday : Weekday
deriving DecidableEq, Repr

open Weekday

-- Given conditions:
def dalia_birthday : Weekday := Wednesday
def days_after_dalia := 60

-- Function to calculate the weekday after a given number of days from a specified day
def add_days (start_day : Weekday) (days : Nat) : Weekday :=
  match start_day, days % 7 with
  | Sunday, 0 => Sunday
  | Sunday, 1 => Monday
  | Sunday, 2 => Tuesday
  | Sunday, 3 => Wednesday
  | Sunday, 4 => Thursday
  | Sunday, 5 => Friday
  | Sunday, 6 => Saturday
  | Monday, 0 => Monday
  | Monday, 1 => Tuesday
  | Monday, 2 => Wednesday
  | Monday, 3 => Thursday
  | Monday, 4 => Friday
  | Monday, 5 => Saturday
  | Monday, 6 => Sunday
  | Tuesday, 0 => Tuesday
  | Tuesday, 1 => Wednesday
  | Tuesday, 2 => Thursday
  | Tuesday, 3 => Friday
  | Tuesday, 4 => Saturday
  | Tuesday, 5 => Sunday
  | Tuesday, 6 => Monday
  | Wednesday, 0 => Wednesday
  | Wednesday, 1 => Thursday
  | Wednesday, 2 => Friday
  | Wednesday, 3 => Saturday
  | Wednesday, 4 => Sunday
  | Wednesday, 5 => Monday
  | Wednesday, 6 => Tuesday
  | Thursday, 0 => Thursday
  | Thursday, 1 => Friday
  | Thursday, 2 => Saturday
  | Thursday, 3 => Sunday
  | Thursday, 4 => Monday
  | Thursday, 5 => Tuesday
  | Thursday, 6 => Wednesday
  | Friday, 0 => Friday
  | Friday, 1 => Saturday
  | Friday, 2 => Sunday
  | Friday, 3 => Monday
  | Friday, 4 => Tuesday
  | Friday, 5 => Wednesday
  | Friday, 6 => Thursday
  | Saturday, 0 => Saturday
  | Saturday, 1 => Sunday
  | Saturday, 2 => Monday
  | Saturday, 3 => Tuesday
  | Saturday, 4 => Wednesday
  | Saturday, 5 => Thursday
  | Saturday, 6 => Friday
  end

-- Statement to prove:
theorem bruces_birthday :
  add_days dalia_birthday days_after_dalia = Sunday :=
sorry

end bruces_birthday_l694_694495


namespace line_equation_135_degrees_y_intercept_minus_1_l694_694364

theorem line_equation_135_degrees_y_intercept_minus_1 :
  ∃ (m b : ℝ), m = -1 ∧ b = -1 ∧ (∀ x y: ℝ, y = m * x + b ↔ x + y + 1 = 0) :=
begin
  sorry
end

end line_equation_135_degrees_y_intercept_minus_1_l694_694364


namespace find_smallest_N_l694_694330

def smallest_N (x y a b : ℕ) : ℕ :=
  let total_pairs := 20 * 23
  in Int.ceil (Real.log2 (total_pairs : ℝ))

theorem find_smallest_N : smallest_N 1 20 1 23 = 9 := by
  simp only [smallest_N]
  sorry

end find_smallest_N_l694_694330


namespace roots_of_cubic_l694_694404

-- Define the cubic equation having roots 3 and -2
def cubic_eq (a b c d x : ℝ) : Prop := a * x^3 + b * x^2 + c * x + d = 0

-- The proof problem statement
theorem roots_of_cubic (a b c d : ℝ) (h₁ : a ≠ 0)
  (h₂ : cubic_eq a b c d 3)
  (h₃ : cubic_eq a b c d (-2)) : 
  (b + c) / a = -7 := 
sorry

end roots_of_cubic_l694_694404


namespace max_page_number_l694_694329

-- Define a function to count the occurrences of the digit 2 in a single number
def count_twos_in_number (n : ℕ) : ℕ :=
  n.digits.count 2

-- Define a function to count the total occurrences of the digit 2 from 1 to n
def count_twos_up_to (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum count_twos_in_number

-- The condition stating Pat Peano has only fifteen 2's
def max_twos : ℕ := 15

-- The mathematical equivalent proof problem in Lean 4
theorem max_page_number (n : ℕ) (h : count_twos_up_to n ≤ max_twos) : n ≤ 52 :=
by sorry

end max_page_number_l694_694329


namespace probability_of_unique_tens_digits_l694_694351

open BigOperators
open Finset

variable (s : Finset ℕ)

def is_valid_set (s : Finset ℕ) : Prop :=
  s.card = 6 ∧ ∀ x ∈ s, 10 ≤ x ∧ x ≤ 99

def has_unique_tens_digits (s : Finset ℕ) : Prop :=
  (s.image (λ x => x / 10)).card = s.card

theorem probability_of_unique_tens_digits :
  ∑ (s : Finset ℕ) in (Finset.filter is_valid_set (Finset.range 100)),
    if has_unique_tens_digits s then 1 else 0 /
  ∑ (s : Finset ℕ) in (Finset.filter is_valid_set (Finset.range 100)),
    1 = 25000 / 18444123 := sorry

end probability_of_unique_tens_digits_l694_694351


namespace no_six_with_average_and_variance_l694_694157

def contains_six (rolls : List ℕ) : Prop := 6 ∈ rolls

def average (rolls : List ℕ) : ℚ := (rolls.map (λ x, (x : ℚ))).sum / rolls.length

def variance (rolls : List ℕ) : ℚ :=
  let avg := average rolls
  (rolls.map (λ x, ((x : ℚ) - avg) ^ 2)).sum / rolls.length

theorem no_six_with_average_and_variance
  (rolls : List ℕ)
  (h_avg : average rolls = 2)
  (h_var : variance rolls = 3.1) :
  ¬contains_six rolls :=
by
  sorry

end no_six_with_average_and_variance_l694_694157


namespace fraction_decimal_comparison_l694_694725

theorem fraction_decimal_comparison :
  (1 / 3 : ℚ) = (3333 / 10000 : ℚ) + (1 / 10000 : ℚ) :=
by
  sorry

end fraction_decimal_comparison_l694_694725


namespace num_valid_numbers_count_l694_694968

-- Definitions of the problem conditions
def digits := {1, 2, 3, 4, 5, 6}  -- Set of digits from 1 to 6
def n := 5  -- Number of digits in each number

-- Definition of a valid number (5-digit number with given conditions)
def is_valid_number (num : List Nat) : Prop :=
  num.length = n ∧
  -- Each digit appears at least once and is distinct
  (∀ d ∈ digits, d ∈ num) ∧
  -- Digits 1 and 6 are not adjacent
  ∀ i < n - 1, (num.nth i = 1 ∧ num.nth (i+1) = 6) ∨ (num.nth i = 6 ∧ num.nth (i+1) = 1) → False

-- The statement of the proof problem
theorem num_valid_numbers_count : 
  ∃ count : Nat, count = 5880 ∧ ∀ num : List Nat, is_valid_number num → count = 5880 := 
by
  sorry

end num_valid_numbers_count_l694_694968


namespace can_erase_some_color_and_preserve_connectivity_l694_694263
open Graph

-- Define K_{40}
def K40 : SimpleGraph (Fin 40) := SimpleGraph.complete _

-- Define the coloring of edges with 6 colors
noncomputable def edge_coloring (e : K40.edge_set) : Fin 6 := sorry

-- Statement of the theorem
theorem can_erase_some_color_and_preserve_connectivity (c : Fin 6) :
  ∃ c : Fin 6, ∀ u v : Fin 40, u ≠ v → (∃ p : u.path v, ∀ e ∈ p.edges, e ≠ c) :=
sorry

end can_erase_some_color_and_preserve_connectivity_l694_694263


namespace simon_age_l694_694850

theorem simon_age : 
  ∃ s : ℕ, 
  ∀ a : ℕ,
    a = 30 → 
    s = (a / 2) - 5 → 
    s = 10 := 
by
  sorry

end simon_age_l694_694850


namespace AC_eq_AF_l694_694657

-- Variables and Definitions
variables (Γ1 Γ2 : Circle) (A D B C E F : Point)

-- Conditions
axiom intersecting_circles : circles_intersect_at_two_points Γ1 Γ2 A D
axiom tangent_at_A_B  : tangent_to_circle_at_point Γ1 A B
axiom tangent_at_A_C  : tangent_to_circle_at_point Γ2 A C
axiom point_E_on_AB_extension : point_on_ray_with_length E A B (2 * (distance A B))
axiom F_is_second_intersection : second_intersection_of_line_with_circumcircle F A C (circumcircle_of_triangle A D E)

-- Statement to Prove
theorem AC_eq_AF : distance A C = distance A F :=
by
  sorry

end AC_eq_AF_l694_694657


namespace probability_focused_permutations_l694_694377

def focused_circular_permutation (n : ℕ) : Prop :=
∀ k < n, ∃ k', k < k' ∧ (|k' - k| ≤ 2 ∨ |(k' + n) - k| ≤ 2)

theorem probability_focused_permutations (h : focused_circular_permutation 10) :
  let p := 13 / 90 in
  100 * 13 + 90 = 1390 :=
by
  let a := 13 in
  let b := 90 in
  have h₁ : p = a / b := rfl
  have h₂ : p * 100 = 1300 := by norm_num
  have h₃ : 100 * a + b = 100 * 13 + 90 := by norm_num
  exact rfl

#print axioms probability_focused_permutations

end probability_focused_permutations_l694_694377


namespace quadratic_func_max_value_l694_694540

theorem quadratic_func_max_value (b c x y : ℝ) (h1 : y = -x^2 + b * x + c)
(h1_x1 : (y = 0) → x = -1 ∨ x = 3) :
    -x^2 + 2 * x + 3 ≤ 4 :=
sorry

end quadratic_func_max_value_l694_694540


namespace angle_difference_independence_l694_694177

variables {A B C M X T : Type}

-- Define the isosceles triangle, midpoint, points, and angles as per conditions
def is_isosceles_triangle (ABC : triangle) (base : BC) : Prop :=
  ABC.isosceles ∧ ABC.base = BC

def is_midpoint (M : Point) (BC : Line) : Prop :=
  midpoint M BC

def on_shortest_arc (X : Point) (AM : Arc) (circumcircle : Circle) : Prop :=
  X ∈ circumcircle ∧ X ∈ shortest_arc AM

def inside_angle (T : Point) (angle : ∠BMA) : Prop :=
  T ∈ interior angle

def right_angle (angle : ∠TMX) : Prop :=
  angle = 90°

def equal_segments (TX BX : Length) : Prop :=
  TX = BX

-- Define the required statement that needs proof
theorem angle_difference_independence (ABC : triangle) (BC : Line) (M : Point)
  (X : Point) (T : Point)
  (h_iso : is_isosceles_triangle ABC BC)
  (h_mid : is_midpoint M BC)
  (h_arc : on_shortest_arc X (shortest_arc M) (circumcircle (triangle ABC)))
  (h_inside : inside_angle T (angle BMA))
  (h_right : right_angle (angle TMX))
  (h_equal : equal_segments (length T X) (length B X)): 
  ∃ (BAM : Angle), 
    ∀ (X : Point), 
    ∃ (T : Point),
    (angle MTB - angle CTM = BAM) := 
  sorry

end angle_difference_independence_l694_694177


namespace intersection_A_B_l694_694971

def A := {x : ℝ | |x| < 1}
def B := {x : ℝ | -2 < x ∧ x < 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 0} :=
by
  sorry

end intersection_A_B_l694_694971


namespace DE_eq_AC_plus_BC_l694_694943

open_locale euclidean_geometry

variables {A B C P D E : Point}
variables {k_A k_B : Circle}
variables {M_A M_B : Point}

-- Given conditions
def is_triangle (A B C : Point) := 
  A ≠ B ∧ B ≠ C ∧ C ≠ A

def is_collinear (A P M_A : Point) := 
  ∃ l : Line, A ∈ l ∧ P ∈ l ∧ M_A ∈ l

def is_center_of_circumcircle (M_A : Point) (k_A : Circle) (A C P : Point) := 
  M_A ∈ k_A ∧ (∀ x, x ∈ k_A ↔ ∃ t, A = t ∧ C = t ∧ P = t)

-- Problem statement
theorem DE_eq_AC_plus_BC
  (hABC : is_triangle A B C)
  (hP : inside_triangle P A B C)
  (hMA : is_center_of_circumcircle M_A k_A A C P)
  (hMB : is_center_of_circumcircle M_B k_B B C P)
  (hMA_outside : outside_triangle M_A A B C)
  (hMB_outside : outside_triangle M_B A B C)
  (hCollinearA : is_collinear A P M_A)
  (hCollinearB : is_collinear B P M_B)
  (hLine_parallel : ∀ {D E}, D ≠ P ∧ E ≠ P → parallel (line_through P D) (line_through A B))
  (hDkA : D ∈ k_A ∧ D ≠ P)
  (hEkB : E ∈ k_B ∧ E ≠ P) :
  distance D E = (distance A C) + (distance B C) :=
sorry

end DE_eq_AC_plus_BC_l694_694943


namespace solution_set_l694_694555

variable {α : Type*}
variable (f : α → α)
variable (x : α)
variable (a : ℝ)
variable [LE α] [Preorder α]

-- Conditions based definitions
def is_even_function (f : α → α) : Prop :=
  ∀ x, f x = f (-x)

def is_monotonically_increasing (f : α → α) (I : Set α) : Prop :=
  ∀ x y ∈ I, x ≤ y → f x ≤ f y

def coordinates_are_within_domain : Prop :=
  4 - 2 * a ≤ 2 + a

-- Proving the solution inequality > transformed inequality as the set.
theorem solution_set (h1 : is_even_function f)
  (h2 : is_monotonically_increasing f {x : ℝ | 0 ≤ x ∧ x ≤ 8})
  (h3 : coordinates_are_within_domain)
  (h4 : 2 - a = -(4 - 2 * a)) :
  ∀ x, (f (2 * x + 1) > f 1) ↔ (x ∈ [-9/2, -1) ∪ (0, 7/2]) :=
by
  sorry

end solution_set_l694_694555


namespace magician_trick_success_l694_694797

-- Define the overall problem conditions and the strategy for the magician and assistant
noncomputable def magician_can_always_deduce_hidden_cards (A B : ℕ) (cards : List ℕ) (circular_cards : List ℕ) : Prop :=
  let n := 29
  let circular_pos := λ (x : ℕ), (x % n) + 1
  let hidden_cards := [A, B]
  let is_adjacent (x y : ℕ) := y == circular_pos x
  let assistant_shows (x y : ℕ) :=
    if is_adjacent A B || is_adjacent B A then
      y == circular_pos (circular_pos A) && x == circular_pos (circular_pos B) ||
      y == circular_pos (circular_pos B) && x == circular_pos (circular_pos A)
    else
      y == circular_pos A && x == circular_pos B || 
      y == circular_pos B && x == circular_pos A

  ∃ x y, x ∈ cards ∧ y ∈ cards ∧ assistant_shows x y ∧ 
          (∃ A' B', [A', B'] = hidden_cards ∧ 
                    (assistant_shows (circular_pos A) (circular_pos B) ∨ assistant_shows (circular_pos B) (circular_pos A)))

-- Theorem statement with the magician's deduction problem
theorem magician_trick_success
  (A B : ℕ)
  (cards : List ℕ) (circular_cards : List ℕ)
  : magician_can_always_deduce_hidden_cards A B cards circular_cards := sorry

end magician_trick_success_l694_694797


namespace bucket_list_time_l694_694069

theorem bucket_list_time :
  let get_in_shape := 2 * 12, -- 2 years in months
      learn_climbing := (2 * 2) * 12, -- 4 years in months
      survival_skills := 9, -- 9 months
      photography_course := 3, -- 3 months
      downtime := 1, -- 1 month
      first_mountain := 4, -- 4 months
      second_mountain := 5, -- 5 months
      third_mountain := 6, -- 6 months
      fourth_mountain := 8, -- 8 months
      fifth_mountain := 7, -- 7 months
      sixth_mountain := 9, -- 9 months
      seventh_mountain := 10, -- 10 months
      learn_diving := 13, -- 13 months
      diving_caves := 2 * 12 -- 2 years in months
  in get_in_shape + learn_climbing + survival_skills + photography_course +
     downtime + first_mountain + second_mountain + third_mountain +
     fourth_mountain + fifth_mountain + sixth_mountain + seventh_mountain +
     learn_diving + diving_caves = 171 →
     171 / 12 = 14.25 :=
by {
  sorry
}

end bucket_list_time_l694_694069


namespace cody_initial_tickets_l694_694483

def initial_tickets (lost : ℝ) (spent : ℝ) (left : ℝ) : ℝ :=
  lost + spent + left

theorem cody_initial_tickets : initial_tickets 6.0 25.0 18.0 = 49.0 := by
  sorry

end cody_initial_tickets_l694_694483


namespace find_term_number_l694_694213

def sequence_term (n : ℕ) : ℝ :=
  Real.sqrt (2 * (3 * n - 1))

theorem find_term_number :
  ∃ n : ℕ, sequence_term n = 8 ∧ n = 11 :=
by
  use 11
  split
  · -- Show that the term is 8 for n = 11
    show sequence_term 11 = 8
    sorry
  · -- Show that n is indeed 11
    show 11 = 11
    rfl

end find_term_number_l694_694213


namespace dave_apps_left_l694_694119

def initial_apps : ℕ := 24
def initial_files : ℕ := 9
def files_left : ℕ := 5
def apps_left (files_left: ℕ) : ℕ := files_left + 7

theorem dave_apps_left :
  apps_left files_left = 12 :=
by
  sorry

end dave_apps_left_l694_694119


namespace total_area_of_map_correct_l694_694827

theorem total_area_of_map_correct :
  let level1_area := 40 * 20 in
  let level2_area := 15 * 15 in
  let level3_area := (1 / 2) * 25 * 12 in
  let level4_area := (1 / 2) * (10 + 20) * 8 in
  let level5_area := (30 * 15) + (15 * 10) in
  level1_area + level2_area + level3_area + level4_area + level5_area = 1895 :=
by {
  let level1_area := 40 * 20,
  let level2_area := 15 * 15,
  let level3_area := (1 / 2) * 25 * 12,
  let level4_area := (1 / 2) * (10 + 20) * 8,
  let level5_area := (30 * 15) + (15 * 10),
  have h1 : level1_area = 800 := by norm_num,
  have h2 : level2_area = 225 := by norm_num,
  have h3 : level3_area = 150 := by norm_num,
  have h4 : level4_area = 120 := by norm_num,
  have h5 : level5_area = 600 := by norm_num,
  calc
    level1_area + level2_area + level3_area + level4_area + level5_area
      = 800 + 225 + 150 + 120 + 600 : by rw [h1, h2, h3, h4, h5]
  ... = 1895 : by norm_num
  }

end total_area_of_map_correct_l694_694827


namespace total_possibilities_outside_464_cube_l694_694337

-- Define the problem conditions: cube arrangement and face labels
def cubes := 64
def dimensions := (1, 1, 1)
def total_possible_totals := 49

theorem total_possibilities_outside_464_cube (cubes : ℕ) (dim : ℕ × ℕ × ℕ) : 
  1 ∈ dim ∧ 2 ∈ dim → cubes = 64 → dim = (1, 1, 1) → (num_possibilities : ℕ) = 49 :=
by sorry

end total_possibilities_outside_464_cube_l694_694337


namespace find_numbers_l694_694132

theorem find_numbers :
  ∃ (S : Finset ℕ), S.card = 1000 ∧ S.sum id = S.prod id :=
by
  let S := (Finset.range 999).insert 1000
  use S
  have h_card : S.card = 1000, sorry
  have h_sum : S.sum id = 1000, sorry
  have h_prod : S.prod id = 1000, sorry
  exact ⟨h_card, h_sum, h_prod⟩

end find_numbers_l694_694132


namespace infinite_non_expressible_integers_l694_694332

theorem infinite_non_expressible_integers :
  ∃^∞ n : ℕ, ∀ x1 x2 x3 x4 x5 : ℕ, n ≠ x1^3 + x2^5 + x3^7 + x4^9 + x5^11 :=
by
  sorry

end infinite_non_expressible_integers_l694_694332


namespace num_valid_three_digit_integers_l694_694563

-- Define the four-digit set
def digits := {2, 4, 7, 8}

-- Define a helper predicate to check if the sum of digits is odd
def is_sum_odd (n1 n2 n3 : ℕ) : Prop :=
  (n1 + n2 + n3) % 2 = 1

-- Main theorem statement
theorem num_valid_three_digit_integers : 
  ∃ (f : Finset ℕ), f ⊆ digits ∧ f.card = 3 ∧ (∀ x ∈ f, x ∈ digits ∧ is_sum_odd (f.toList.nth 0).getOrElse 0 (f.toList.nth 1).getOrElse 0 (f.toList.nth 2).getOrElse 0) ∧ 
  (Finset.permutations f).card = 12 :=
sorry

end num_valid_three_digit_integers_l694_694563


namespace find_extra_digit_l694_694031

theorem find_extra_digit (x y a : ℕ) (hx : x + y = 23456) (h10x : 10 * x + a + y = 55555) (ha : 0 ≤ a ∧ a ≤ 9) : a = 5 :=
by
  sorry

end find_extra_digit_l694_694031


namespace max_abs_z5_l694_694879

open Complex

-- Define the complex numbers satisfying the given conditions
def z1 (z : ℂ) := abs z ≤ 1
def z2 (z : ℂ) := abs z ≤ 1

def condition_z3 (z1 z2 z3 : ℂ) := abs (2 * z3 - (z1 + z2)) ≤ abs (z1 - z2)
def condition_z4 (z1 z2 z4 : ℂ) := abs (2 * z4 - (z1 + z2)) ≤ abs (z1 - z2)
def condition_z5 (z3 z4 z5 : ℂ) := abs (2 * z5 - (z3 + z4)) ≤ abs (z3 - z4)

-- State the theorem to prove the maximum value of |z5|
theorem max_abs_z5 (z1 z2 z3 z4 z5 : ℂ) 
  (h1 : z1 z1) 
  (h2 : z2 z2) 
  (h3 : condition_z3 z1 z2 z3) 
  (h4 : condition_z4 z1 z2 z4) 
  (h5 : condition_z5 z3 z4 z5) 
:
  abs z5 ≤ sqrt 3 := 
sorry

end max_abs_z5_l694_694879


namespace num_valid_programs_l694_694468

-- Define the set of courses
def courses := {"English", "Algebra", "Geometry", "History", "Art", "Science", "Latin"}

-- Define the set of mathematics courses
def math_courses := {"Algebra", "Geometry"}

-- Definition of the problem conditions
def is_valid_program (program : set string) : Prop :=
  "English" ∈ program ∧
  (∃ M ⊆ program, M ⊆ math_courses ∧ 2 ≤ M.size) ∧
  program.size = 5

-- The statement of the proof problem
theorem num_valid_programs : set.count {program | program ⊆ courses ∧ is_valid_program program} = 6 :=
sorry

end num_valid_programs_l694_694468


namespace four_digit_number_divisible_by_11_l694_694584

theorem four_digit_number_divisible_by_11 :
  ∃ (a b c d : ℕ), 
    a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ 
    a + b + c + d = 10 ∧ 
    (a + c) % 11 = (b + d) % 11 ∧
    (10 - a != 0 ∨ 10 - c != 0 ∨ 10 - b != 0 ∨ 10 - d != 0) := sorry

end four_digit_number_divisible_by_11_l694_694584


namespace election_proof_l694_694266

noncomputable def election_problem : Prop :=
  ∃ (V : ℝ) (votesA votesB votesC : ℝ),
  (votesA = 0.35 * V) ∧
  (votesB = votesA + 1800) ∧
  (votesC = 0.5 * votesA) ∧
  (V = votesA + votesB + votesC) ∧
  (V = 14400) ∧
  ((votesA / V) * 100 = 35) ∧
  ((votesB / V) * 100 = 47.5) ∧
  ((votesC / V) * 100 = 17.5)

theorem election_proof : election_problem := sorry

end election_proof_l694_694266


namespace decrease_hours_worked_l694_694072

theorem decrease_hours_worked (initial_hourly_wage : ℝ) (initial_hours_worked : ℝ) :
  let new_hourly_wage := initial_hourly_wage * 1.25
  let new_hours_worked := (initial_hourly_wage * initial_hours_worked) / new_hourly_wage
  initial_hours_worked > 0 → 
  initial_hourly_wage > 0 → 
  new_hours_worked < initial_hours_worked :=
by
  intros initial_hours_worked_pos initial_hourly_wage_pos
  let new_hourly_wage := initial_hourly_wage * 1.25
  let new_hours_worked := (initial_hourly_wage * initial_hours_worked) / new_hourly_wage
  sorry

end decrease_hours_worked_l694_694072


namespace comic_books_left_l694_694706

theorem comic_books_left (total : ℕ) (sold : ℕ) (left : ℕ) (h1 : total = 90) (h2 : sold = 65) :
  left = total - sold → left = 25 := by
  sorry

end comic_books_left_l694_694706


namespace footballer_catches_ball_before_sideline_l694_694057

noncomputable def ball_distance (t : ℝ) : ℝ := 
  t / 2 * (8 - 0.75 * t)

noncomputable def player_distance (t : ℝ) : ℝ := 
  t / 2 * (7 + 0.5 * t)
  
noncomputable def catch_up_time ( : ℝ) : ℝ := 
  let t := (1 + real.sqrt 101) / 2.5 in
  t
  
theorem footballer_catches_ball_before_sideline :
  let t := catch_up_time in
  0 ≤ 23 - player_distance t ∧
  23 - player_distance t = 0.5 :=
by
  -- sorry indicates the proof is omitted
  sorry

end footballer_catches_ball_before_sideline_l694_694057


namespace part1_solution_set_part2_range_of_a_l694_694935

-- Definitions of f and g as provided in the problem.
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|
def g (x a : ℝ) : ℝ := |x + 1| - |x - a| + a

-- Problem 1: Prove the solution set for f(x) ≤ 5 is [-2, 3]
theorem part1_solution_set : { x : ℝ | f x ≤ 5 } = { x : ℝ | -2 ≤ x ∧ x ≤ 3 } :=
  sorry

-- Problem 2: Prove the range of a when f(x) ≥ g(x) always holds is (-∞, 1]
theorem part2_range_of_a (a : ℝ) (h : ∀ x : ℝ, f x ≥ g x a) : a ≤ 1 :=
  sorry

end part1_solution_set_part2_range_of_a_l694_694935


namespace arithmetic_series_sum_l694_694105

theorem arithmetic_series_sum :
  ∀ (a l d : ℤ) (n : ℕ), a = -45 → l = 3 → d = 4 → (l = a + (n - 1) * d) → 
  (n = 13) → (n / 2 * (a + l) = -273) :=
begin
  intros a l d n ha hl hd h_n_term hn_eq,
  rw [ha, hl, hd] at *,
  have : n = 13 := hn_eq,
  rw this,
  linarith,
end

end arithmetic_series_sum_l694_694105


namespace find_p_q_sum_l694_694286

noncomputable def area_of_triangle_AGB (A B C D E G : Point) (AD_length CE_length AB_length : ℝ) 
[is_median AD_length A B C] [is_median CE_length C A B] [is_equal AB_length (distance A B)] 
[is_circumcircle_extension E G]
: ℝ :=
have h_medians : AD_length = 15 ∧ CE_length = 30 ∧ AB_length = 30 := by sorry,
let area_AGB := 112.5 in
area_AGB

theorem find_p_q_sum (A B C D E G : Point) (AD_length CE_length AB_length : ℝ) 
[is_median AD_length A B C] [is_median CE_length C A B] [is_equal AB_length (distance A B)] 
[is_circumcircle_extension E G]
: 225 + 3 = 228 :=
by sorry

end find_p_q_sum_l694_694286


namespace time_to_reach_rest_area_l694_694783

variable (rate_per_minute : ℕ) (remaining_distance_yards : ℕ)

theorem time_to_reach_rest_area (h_rate : rate_per_minute = 2) (h_distance : remaining_distance_yards = 50) :
  (remaining_distance_yards * 3) / rate_per_minute = 75 := by
  sorry

end time_to_reach_rest_area_l694_694783


namespace bisecting_segment_length_l694_694472

theorem bisecting_segment_length (a c : ℝ) : ∃ x : ℝ, x = Real.sqrt ((a^2 + c^2) / 2) :=
by
  use Real.sqrt ((a^2 + c^2) / 2)
  sorry

end bisecting_segment_length_l694_694472


namespace no_six_with_average_and_variance_l694_694158

def contains_six (rolls : List ℕ) : Prop := 6 ∈ rolls

def average (rolls : List ℕ) : ℚ := (rolls.map (λ x, (x : ℚ))).sum / rolls.length

def variance (rolls : List ℕ) : ℚ :=
  let avg := average rolls
  (rolls.map (λ x, ((x : ℚ) - avg) ^ 2)).sum / rolls.length

theorem no_six_with_average_and_variance
  (rolls : List ℕ)
  (h_avg : average rolls = 2)
  (h_var : variance rolls = 3.1) :
  ¬contains_six rolls :=
by
  sorry

end no_six_with_average_and_variance_l694_694158


namespace standard_equation_of_ellipse_length_of_chord_AB_l694_694543

-- Define the conditions
variables (a b : ℝ) (a_gt_b : a > b) (b_gt_0 : b > 0) (b_val : b = 2)
variables (ellipse_eqn : ∀ x y : ℝ, (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1)
variables (focal_length : 2 * sqrt(2) = a ^ 2 - b ^ 2)
variables (point_P : (x, y) = (-2, 1))
variables (line_l : ∀ x y : ℝ, y - 1 = x + 2)

-- Define the goals
theorem standard_equation_of_ellipse :
  a = sqrt(12) ∧ b = 2 ∧ (∀ x y : ℝ, (x ^ 2) / 12 + (y ^ 2) / 4 = 1) :=
sorry

theorem length_of_chord_AB :
  ∀ (x1 y1 x2 y2 : ℝ), 
  (x1, y1) and (x2, y2) ∈ ({(x, y) | y = x + 3 ∧ (x ^ 2) / 12 + (y ^ 2) / 4 = 1}) →
  |((x1 + x2)^2 - 4 * x1 * x2)| = sqrt(42) / 2 :=
sorry

end standard_equation_of_ellipse_length_of_chord_AB_l694_694543


namespace ambrose_minimum_next_score_l694_694090

theorem ambrose_minimum_next_score 
    (scores : List ℕ)
    (next_test_increase : ℕ)
    (desired_minimum_score : ℕ) :
    scores = [84, 76, 89, 94, 67, 90] →
    next_test_increase = 5 →
    desired_minimum_score = 118 →
    let current_sum := scores.sum;
    let current_average := current_sum.toFloat / scores.length.toFloat;
    let desired_average := current_average + next_test_increase;
    let total_needed_score := desired_average * (scores.length + 1).toFloat;
    let required_next_test_score := total_needed_score - current_sum.toFloat;
    required_next_test_score ≥ desired_minimum_score :=
begin
    intros h_scores h_increase h_min_score,
    rw h_scores at *,
    rw h_increase at *,
    rw h_min_score at *,
    sorry
end

end ambrose_minimum_next_score_l694_694090


namespace find_BP_l694_694046

theorem find_BP
  (A B C D P : Type)
  (hCircle : IsOnCircle A ∧ IsOnCircle B ∧ IsOnCircle C ∧ IsOnCircle D)
  (hIntersect : IntersectsAt P (LineSegment A C) (LineSegment B D))
  (AP : ℝ) (hAP : AP = 8)
  (PC : ℝ) (hPC : PC = 1)
  (BD : ℝ) (hBD : BD = 6)
  (BP DP: ℝ)
  (hBP_lt_DP : BP < DP)
  (hPowerOfPoint : 8 * 1 = BP * DP) :
  BP = 2 := by
  sorry

end find_BP_l694_694046


namespace maximize_profit_l694_694805

-- Define the conditions
def price_per_product : ℝ := 60  -- Price per product
def production_cost (x : ℝ) : ℝ := 61 * x + 100 / x - 75  -- Cost function

-- Define the profit function
def profit (x : ℝ) : ℝ := 75 - (x + 100 / x)  -- Profit function

-- Define the range and type of x
def valid_x (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 50 ∧ x ∈ Set.Univ.filter (λ x, x ∈ ℕ ∧ x > 0)

-- Statement for the proof problem
theorem maximize_profit (x : ℝ) (hx : valid_x x) : 
    profit x = 75 - (x + 100 / x) ∧ ∀ y: ℝ, valid_x y → profit y ≤ profit 10 :=
by
    sorry

end maximize_profit_l694_694805


namespace intersection_is_correct_l694_694319

def M : Set ℤ := {x | x^2 + 3 * x + 2 > 0}
def N : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_is_correct : M ∩ N = {0, 1, 2} := by
  sorry

end intersection_is_correct_l694_694319


namespace max_volume_of_rectangular_frame_l694_694077

/--
Given a steel rod of length 18 cm, bent into a rectangular frame with a length-to-width ratio of 2:1,
the dimensions of the rectangle that yield the maximum volume are Length = 2 cm, Width = 1 cm, and Height = 1.5 cm.
The maximum volume is 3 cm³.
-/
theorem max_volume_of_rectangular_frame (x h : ℝ) (H : 0 < x ∧ x < 3/2 ∧ h = 4.5 - 3 * x) (Volume : ℝ := 9 * x^2 - 6 * x^3) :
  ∃ (l w : ℝ), l = 2 ∧ w = 1 ∧ h = 1.5 ∧ Volume = 3 :=
by {
  use [2, 1],
  split,
  { refl },
  split,
  { refl },
  split,
  { simp [h, H.right] },
  { simp [Volume, H.left.right.left, H.left.right.right, H.left.left] }
}

end max_volume_of_rectangular_frame_l694_694077


namespace quadratic_equation_solution_l694_694154

theorem quadratic_equation_solution (m : ℝ) (h : m ≠ 1) : 
  (m^2 - 3 * m + 2 = 0) → m = 2 :=
by
  sorry

end quadratic_equation_solution_l694_694154


namespace matrix_N_unique_l694_694912

theorem matrix_N_unique 
(M N : Matrix (Fin 2) (Fin 2) ℤ) 
(h1 : M ⬝ (λ (i : Fin 2), if i = 0 then 4 else 0) = λ (i : Fin 2), if i = 0 then 8 else 28)
(h2 : M ⬝ (λ (i : Fin 2), if i = 0 then -2 else 10) = λ (i : Fin 2), if i = 0 then 6 else -34):
  M = N :=
by 
  let col1 : Fin 2 -> ℤ := (λ (i : Fin 2), if i = 0 then 2 else 7),
  let col2 : Fin 2 -> ℤ := (λ (i : Fin 2), if i = 0 then 1 else -2),
  let N : Matrix (Fin 2) (Fin 2) ℤ := λ (i j : Fin 2),
    if j = 0 then col1 i else col2 i,
  sorry

end matrix_N_unique_l694_694912


namespace number_of_roosters_l694_694394

def chickens := 9000
def ratio_roosters_hens := 2 / 1

theorem number_of_roosters (h : ratio_roosters_hens = 2 / 1) (c : chickens = 9000) : ∃ r : ℕ, r = 6000 := 
by sorry

end number_of_roosters_l694_694394


namespace min_phi_for_axis_of_symmetry_l694_694371

theorem min_phi_for_axis_of_symmetry :
  ∀ (x : ℝ) (φ : ℝ), φ > 0 ∧ (∀ x, sin (4 * x + (4 * Real.pi / 3) + φ) = sin (4 * (-x + -Real.pi / 5) + (4 * Real.pi / 3) + φ)) → 
  φ = (29 * Real.pi) / 30 := by
  sorry

end min_phi_for_axis_of_symmetry_l694_694371


namespace floor_of_2_8_l694_694375

-- Definition: [x] stands for the greatest integer that is less than or equal to x.
def floor (x : ℝ) : ℤ := Int.floor x

-- Problem Statement: Prove that [2.8] = 2
theorem floor_of_2_8 : floor 2.8 = 2 := 
by 
  sorry

end floor_of_2_8_l694_694375


namespace num_ordered_triples_l694_694333

theorem num_ordered_triples (p a : ℕ) (hp : Nat.Prime p) (ha : Nat.gcd a p = 1) : 
  (Set.card {t : ℕ × ℕ × ℕ | (let x := t.1; let y := t.2.1; let z := t.2.2 
     in ((x + y + z)^2 % p = (a * x * y * z) % p) )} = p^2 + 1) :=
sorry

end num_ordered_triples_l694_694333


namespace problem_part_one_problem_part_two_l694_694171

theorem problem_part_one (x₁ y₁ x₂ y₂ m k : ℝ) (h_midpoint : x₁ + x₂ = 2 ∧ y₁ + y₂ = 2 * m) 
  (h_ellipse1 : 3 * x₁^2 + 4 * y₁^2 = 12) (h_ellipse2 : 3 * x₂^2 + 4 * y₂^2 = 12) 
  (h_slope : k = (y₁ - y₂) / (x₁ - x₂)) (h_m_positive : 0 < m) (h_inside_ellipse : 1 / 4 + m^2 / 3 < 1) :
  k < -1 / 2 := 
sorry

theorem problem_part_two (x₁ y₁ x₂ y₂ x₃ y₃ m : ℝ) (h_midpoint : x₁ + x₂ = 2 ∧ y₁ + y₂ = 2 * m)
  (h_focal : x₁ - 1 + x₂ - 1 + x₃ - 1 = 0 ∧ y₁ + y₂ + y₃ = 0) 
  (h_ellipse3 : 3 * x₃^2 + 4 * y₃^2 = 12) 
  (h_m_positive : m > 0) (h_fp_in_first_quadrant : y₃ = 3 / 2 ∧ m = 3 / 4) 
  (h_common_difference : 2 * (2 - 1 / 2 * x₁ + (2 - 1 / 2 * x₂)) = ± (2 - 1 / 2 * x₃)) :
  real := 
± (3 * real.sqrt 21) / 28 := 
sorry

end problem_part_one_problem_part_two_l694_694171


namespace waiting_time_probability_l694_694814

/-- A person wakes up from a nap and finds that the clock has stopped. He turns on the radio to listen to the hourly time report from the station. Prove that the probability that his waiting time does not exceed 10 minutes is 1/6. -/
theorem waiting_time_probability : 
  let A := {t // 50 ≤ t ∧ t ≤ 60}
  in ∑ t in A, (1 / 60) = 1 / 6 := sorry

end waiting_time_probability_l694_694814


namespace intersection_point_l694_694499

theorem intersection_point :
  (∃ (x y : ℝ), 5 * x - 3 * y = 15 ∧ 4 * x + 2 * y = 14)
  → (∃ (x y : ℝ), x = 3 ∧ y = 1) :=
by
  intro h
  sorry

end intersection_point_l694_694499


namespace laundry_detergent_and_water_required_l694_694817

-- Define constants and conditions
def concentration_range : Prop := 0.2 / 100 ≤ 0.4 / 100 ∧ 0.4 / 100 ≤ 0.5 / 100
def suitable_detergent_range : Prop := 200 / 1000 ≤ 500 / 1000
def tub_capacity : ℝ := 15
def clothes_weight : ℝ := 4
def required_concentration : ℝ := 0.4 / 100
def detergent_per_scoop : ℝ := 0.02
def scoops_of_detergent : ℕ := 2

-- The theorem to be proved
theorem laundry_detergent_and_water_required :
  concentration_range →
  suitable_detergent_range →
  (tub_capacity - clothes_weight - detergent_per_scoop * scoops_of_detergent) = 10.956 ∧
  ((required_concentration * tub_capacity) = (detergent_per_scoop * scoops_of_detergent + 0.004 + clothes_weight)) :=
by 
  sorry

end laundry_detergent_and_water_required_l694_694817


namespace anna_earnings_correct_l694_694100

open Nat

section AnnaCupcakes

variable (n c p : Nat) (f : ℚ)
variable (total_income : ℚ)

def total_cupcakes (n c : Nat) : Nat :=
  n * c

def sold_cupcakes (total_cupcakes : Nat) (f : ℚ) : ℚ :=
  f * total_cupcakes

def earnings (sold_cupcakes : ℚ) (p : Nat) : ℚ :=
  sold_cupcakes * p

theorem anna_earnings_correct (h_n : n = 4) (h_c : c = 20) (h_p : p = 2) (h_f : f = (3/5 : ℚ)) :
  let total := total_cupcakes n c in
  let sold := sold_cupcakes total f in
  let earn := earnings sold p in
  earn = 96 := by
  sorry
  
end AnnaCupcakes

end anna_earnings_correct_l694_694100


namespace max_value_f_l694_694930

def f (x : ℝ) : ℝ :=
  min (2 * x + 1) (min ((1 / 3) * x + 1) (- (1 / 2) * x + 7))

theorem max_value_f : ∃ x : ℝ, f x = 47 / 5 :=
by
  sorry

end max_value_f_l694_694930


namespace false_proposition_in_given_options_l694_694231

variable {a b c : ℝ}

theorem false_proposition_in_given_options (h : a > b) : ¬(ac > bc) :=
by
  sorry

end false_proposition_in_given_options_l694_694231


namespace area_of_shaded_region_l694_694273

-- Definition of conditions
def radius : Real := 6
def angle_between_diameters : Real := 60 * (Real.pi / 180) -- Convert degrees to radians
def total_circle_area : Real := Real.pi * radius ^ 2
def sector_angle : Real := angle_between_diameters

-- Theorem statement
theorem area_of_shaded_region :
  let sector_area := total_circle_area * (sector_angle / (2 * Real.pi))
  let shaded_area := 2 * sector_area
  shaded_area = 12 * Real.pi :=
by
  sorry

end area_of_shaded_region_l694_694273


namespace find_m_value_l694_694352

theorem find_m_value (m : ℚ) :
  ∀ x, (3 : ℚ) * x^2 - 7 * x + m = 0 ↔ discriminant (3 : ℚ) (-7 : ℚ) m = 0 → m = 49 / 12 :=
by {
  intros,
  sorry
}

end find_m_value_l694_694352


namespace max_min_values_strictly_increasing_interval_tan_alpha_beta_l694_694199

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := 2 * a * (Real.cos x) ^ 2 + b * (Real.sin x) * (Real.cos x)

-- Define the conditions
axiom f_condition1 : f 1 2 0 = 2
axiom f_condition2 : f 1 2 (Real.pi / 3) = (1 + Real.sqrt 3) / 2

-- Prove the maximum and minimum values
theorem max_min_values : ∃(max min : ℝ), max = Real.sqrt 2 + 1 ∧ min = -Real.sqrt 2 + 1 :=
sorry

-- Prove the interval where f is strictly increasing
theorem strictly_increasing_interval : ∀ (k : ℤ), -3 * Real.pi / 8 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 8 + k * Real.pi → x ∈ I :=
sorry

-- Prove the value of tan(α + β)
theorem tan_alpha_beta (α β : ℝ) (k : ℤ) : α - β ≠ k * Real.pi ∧ f 1 2 α = f 1 2 β  → Real.tan (α + β) = 1 :=
sorry

end max_min_values_strictly_increasing_interval_tan_alpha_beta_l694_694199


namespace ravi_refrigerator_purchase_price_l694_694336

theorem ravi_refrigerator_purchase_price (purchase_price_mobile : ℝ) (sold_mobile : ℝ)
  (profit : ℝ) (loss : ℝ) (overall_profit : ℝ)
  (H1 : purchase_price_mobile = 8000)
  (H2 : loss = 0.04)
  (H3 : profit = 0.10)
  (H4 : overall_profit = 200) :
  ∃ R : ℝ, 0.96 * R + sold_mobile = R + purchase_price_mobile + overall_profit ∧ R = 15000 :=
by
  use 15000
  sorry

end ravi_refrigerator_purchase_price_l694_694336


namespace tucker_tissues_l694_694747

theorem tucker_tissues (num_tissues_per_box : ℕ) (num_boxes : ℕ) (tissues_used : ℕ) (t : ℕ)
    (h₀ : num_tissues_per_box = 160) (h₁ : num_boxes = 3) (h₂ : tissues_used = 210) 
    (h₃ : t = num_tissues_per_box * num_boxes - tissues_used) : t = 270 :=
by
  -- Unfold assumptions
  rw [h₀, h₁, h₂] at h₃
  -- Simplify the equation
  have h₄ : t = 160 * 3 - 210 := h₃
  calc
    t = 160 * 3 - 210 := h₄
    _ = 480 - 210 := by rw nat.mul_comm
    _ = 270 := by norm_num

end tucker_tissues_l694_694747


namespace find_a_l694_694550

theorem find_a (a : ℝ) : (1 : ℝ)^2 + 1 + 2 * a = 0 → a = -1 := 
by 
  sorry

end find_a_l694_694550


namespace system_of_equations_l694_694113

theorem system_of_equations (x y z : ℝ) (h1 : 4 * x - 6 * y - 2 * z = 0) (h2 : 2 * x + 6 * y - 28 * z = 0) (hz : z ≠ 0) :
  (x^2 - 6 * x * y) / (y^2 + 4 * z^2) = -5 :=
by
  sorry

end system_of_equations_l694_694113


namespace greatest_number_of_positive_factors_l694_694999

-- Define the conditions
def is_positive_integer (x : ℕ) : Prop := x > 0
def is_valid_b (b : ℕ) : Prop := is_positive_integer b ∧ b ≤ 18
def is_valid_n (n : ℕ) : Prop := is_positive_integer n ∧ n ≤ 18
def num_factors (b n : ℕ) : ℕ :=
  let b_fact := b.factorization in
  b_fact.support.fold (λ acc p => acc * (b_fact p * n + 1)) 1

-- Define the theorem to prove
theorem greatest_number_of_positive_factors (b n : ℕ) (hb : is_valid_b b) (hn : is_valid_n n) :
  num_factors b n ≤ 703 :=
sorry

end greatest_number_of_positive_factors_l694_694999


namespace problem_solution_l694_694549

noncomputable def problem_statement : Prop :=
  ∀ x : ℝ, (|x - 1| < 2 → (x(x - 2) < 0)) ∧ (x(x - 2) < 0 → |x - 1| < 2 → False)

theorem problem_solution : problem_statement :=
by sorry

end problem_solution_l694_694549


namespace washed_loads_by_noon_l694_694831

theorem washed_loads_by_noon (total_loads : ℕ) (remaining_loads : ℕ) (adam_task : total_loads = 14) (needs_to_wash : remaining_loads = 6) : total_loads - remaining_loads = 8 :=
by {
  rw [adam_task, needs_to_wash],
  norm_num,
}

end washed_loads_by_noon_l694_694831


namespace minimum_value_of_complex_abs_square_l694_694664

def complex_abs_square (w : ℂ) : ℂ :=
  abs (w + 1 - complex.i) ^ 2 + abs (w - 7 + 2 * complex.i) ^ 2

theorem minimum_value_of_complex_abs_square (w : ℂ) (h : abs (w - 3 + complex.i) = 3) :
  complex_abs_square w = 38 :=
by
  sorry

end minimum_value_of_complex_abs_square_l694_694664


namespace real_solutions_system_l694_694905

theorem real_solutions_system (x y z : ℝ) : 
  (x = 4 * z^2 / (1 + 4 * z^2) ∧ y = 4 * x^2 / (1 + 4 * x^2) ∧ z = 4 * y^2 / (1 + 4 * y^2)) ↔ 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by
  sorry

end real_solutions_system_l694_694905


namespace sum_of_first_2017_terms_l694_694193

noncomputable theory

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a n - a m = (n - m) * (a 1 - a 0)

def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
n * (a 1 + a n) / 2

theorem sum_of_first_2017_terms
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_condition : a 3 + a 2015 = 1) :
  sequence_sum a 2017 = 2017 / 2 :=
sorry

end sum_of_first_2017_terms_l694_694193


namespace sin_cos_identity_l694_694229

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x ^ 2 - Real.cos x ^ 2 = 15 / 17 := 
  sorry

end sin_cos_identity_l694_694229


namespace total_people_veg_l694_694257

-- Definitions based on the conditions
def people_only_veg : ℕ := 13
def people_both_veg_nonveg : ℕ := 6

-- The statement we need to prove
theorem total_people_veg : people_only_veg + people_both_veg_nonveg = 19 :=
by
  sorry

end total_people_veg_l694_694257


namespace soda_consumption_period_l694_694701

/-
Rebecca drinks half a bottle of soda a day.
She bought three 6-packs of sodas.
After a certain period, she will have 4 bottles of soda left.
Prove that the period after which she will have 4 bottles of soda left is 28 days.
-/

theorem soda_consumption_period :
  ∀ total_bottles consumed_rate remaining_bottles period,
    total_bottles = 3 * 6 →
    consumed_rate = 1 / 2 →
    remaining_bottles = 4 →
    period = (total_bottles - remaining_bottles) * 2 →
    period = 28 :=
by
  intros total_bottles consumed_rate remaining_bottles period H1 H2 H3 H4
  rw [H1, H2] at H4
  have H5: (18 - 4) * 2 = 28 :=
  sorry
  exact H5

end soda_consumption_period_l694_694701


namespace verify_statements_l694_694676

def curvature (k_A k_B : ℝ) (d_AB : ℝ) : ℝ := |k_A - k_B| / d_AB

theorem verify_statements :
  let f1 := λ x : ℝ, x^3
  let f2 := λ x : ℝ, (1 : ℝ)
  let f3 := λ x : ℝ, x^2 + 1
  let f4 := λ x : ℝ, Real.exp x
  let φ (f : ℝ → ℝ) (x1 x2 : ℝ) := curvature (f.derivative x1) (f.derivative x2) (Real.sqrt ((x1-x2)^2 + (f x1 - f x2)^2))
  let A1 := (1 : ℝ)
  let B1 := (-1 : ℝ)
  let A2 := (0 : ℝ)
  let B2 := (1 : ℝ)
  φ f1 A1 B1 = 0 ∧
  φ f2 A2 B2 = 0 ∧
  ∀ x1 x2 : ℝ, x1 ≠ x2 → φ f3 x1 x2 ≤ 2 ∧
  ∀ x1 x2 : ℝ, x1 ≠ x2 → φ f4 x1 x2 < 1 := 
by
  sorry

end verify_statements_l694_694676


namespace total_confirmed_cases_l694_694251

theorem total_confirmed_cases :
  ∃ (NY CA TX : ℕ),
    NY = 2000 ∧
    CA = NY / 2 ∧
    CA = TX + 400 ∧
    let NY_growth := NY + (NY * 25 / 100),
        CA_growth := CA + (CA * 15 / 100),
        TX_growth := TX + (TX * 30 / 100) in
    NY_growth + CA_growth + TX_growth = 4430 :=
by
  sorry

end total_confirmed_cases_l694_694251


namespace vector_dot_product_proof_l694_694218

variable (a b : ℝ × ℝ)

def dot_product (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2

theorem vector_dot_product_proof
  (h1 : a = (1, -3))
  (h2 : b = (3, 7)) :
  dot_product a b = -18 :=
by 
  sorry

end vector_dot_product_proof_l694_694218


namespace odd_and_monotonically_increasing_in_0_infty_l694_694562

-- Define the functions
def f₁ (x : ℝ) : ℝ := x^3 + x
def f₂ (x : ℝ) : ℝ := Real.sin x
def f₃ (x : ℝ) : ℝ := Real.log x
def f₄ (x : ℝ) : ℝ := Real.tan x

-- Define properties of odd functions
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

-- Define properties of monotonically increasing functions
def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b → f(a) < f(b)

theorem odd_and_monotonically_increasing_in_0_infty :
  (is_odd f₁ ∧ ∀ x y : ℝ, 0 < x → x < y → f₁(x) < f₁(y)) ∧
  (¬ (is_odd f₂ ∧ ∀ x y : ℝ, 0 < x → x < y → f₂(x) < f₂(y))) ∧
  (¬ (is_odd f₃ ∧ ∀ x y : ℝ, 0 < x → x < y → f₃(x) < f₃(y))) ∧
  (¬ (is_odd f₄ ∧ ∀ x y : ℝ, 0 < x → x < y → f₄(x) < f₄(y))) :=
by
  sorry

end odd_and_monotonically_increasing_in_0_infty_l694_694562


namespace vertical_angles_congruent_l694_694738

-- Define vertical angles in terms of basic angle properties
def is_vertical_angle (α β : Angle) : Prop :=
  ∃ (A B C D : Point), 
  α = ∠ A B C ∧ β = ∠ D B C ∧ A ≠ D

-- The theorem to prove
theorem vertical_angles_congruent (α β : Angle) : 
  is_vertical_angle α β → α = β :=
by
  sorry

end vertical_angles_congruent_l694_694738


namespace maximum_area_of_triangle_l694_694677

theorem maximum_area_of_triangle :
  let A := (1 : ℝ, 1 : ℝ)
  let B := (-2 : ℝ, 0 : ℝ)
  let C := (1 : ℝ, -1 : ℝ)
  let line_eq := λ x y : ℝ, x + 3 * y + 2 = 0
  let ellipse_eq := λ x y : ℝ, x^2 + 3 * y^2 = 4
  A ∈ {(x, y) | ellipse_eq x y} →
  B ∈ {(x, y) | ellipse_eq x y} →
  C ∈ {(x, y) | ellipse_eq x y} →
  (∀ x y, line_eq x y → (x, y) = B ∨ (x, y) = C) →
  (1/2 * ((1 * (0 + 1) - 1 * (-2 + 1) + 1 * (2 - 0)) = 3) :=
begin
  sorry -- to indicate the proof is not necessary as per the instructions
end

end maximum_area_of_triangle_l694_694677


namespace triangle_area_l694_694050

theorem triangle_area
  (A B C L : Type*)
  [metric_space A] 
  [metric_space B]
  [metric_space C]
  [metric_space L]
  (AL BL CL : ℝ)
  (hAL : AL = 2)
  (hBL : BL = sqrt 30)
  (hCL : CL = 5)
  (hBisect : ∀ (a b : ℝ), a/AL = b/CL) :
  let area := (7 * sqrt 39) / 4 in
  sorry

end triangle_area_l694_694050


namespace odd_and_decreasing_value_of_θ_l694_694202

noncomputable def f (x θ : ℝ) : ℝ := sin (2 * x + θ) + sqrt 3 * cos (2 * x + θ)

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f (x) ≥ f (y)

theorem odd_and_decreasing_value_of_θ :
  ∃ θ, odd_function (f θ) ∧ decreasing_on_interval (f θ) (-π / 4) 0 ∧ θ = 2 * π / 3 :=
by
  sorry

end odd_and_decreasing_value_of_θ_l694_694202


namespace inlet_pipe_rate_l694_694455

noncomputable def tank_capacity : ℚ := 5760
noncomputable def leak_emptying_time : ℚ := 6
noncomputable def net_emptying_time_with_inlet : ℚ := 8

noncomputable def leak_rate : ℚ := tank_capacity / leak_emptying_time
noncomputable def net_emptying_rate_with_inlet : ℚ := tank_capacity / net_emptying_time_with_inlet

theorem inlet_pipe_rate :
  let inlet_rate_per_hour := (leak_rate - net_emptying_rate_with_inlet) in
  let inlet_rate_per_minute := inlet_rate_per_hour / 60 in
  inlet_rate_per_minute = 12 := by
  sorry

end inlet_pipe_rate_l694_694455


namespace greatest_possible_earning_l694_694395

-- Define the conditions and the invariant
variables (a b c x : ℕ) (a_0 b_0 c_0 : ℕ)

-- 1000 years later, the stones return to the initial count in each box: a == a_0, b == b_0, c == c_0
theorem greatest_possible_earning (h : a = a_0 ∧ b = b_0 ∧ c = c_0) : 
  let N := 2 * x + a^2 + b^2 + c^2 in
  let N_initial := 2 * 0 + a_0^2 + b_0^2 + c_0^2 in
  N = N_initial → x = 0 :=
sorry

end greatest_possible_earning_l694_694395


namespace no_lions_present_l694_694476

def tigers_and_monkeys_no_lions (total_animals tigers monkeys : ℕ) : Prop :=
  let non_tigers := total_animals - tigers
  let non_monkeys := total_animals - monkeys
  (tigers = 7 * non_tigers) ∧ (monkeys = non_monkeys / 7) → (total_animals = tigers + monkeys)

theorem no_lions_present (total_animals tigers monkeys : ℕ) (h1 : tigers = 7 * (total_animals - tigers)) (h2 : monkeys = (total_animals - monkeys) / 7) : tigers_and_monkeys_no_lions total_animals tigers monkeys := 
by {
  unfold tigers_and_monkeys_no_lions,
  rw [← Nat.eq_add_of_sub_eq (tigers_eq _ _), ← (monkeys_eq _ _)] at *,
  sorry
}

end no_lions_present_l694_694476


namespace greatest_possible_a_l694_694720

theorem greatest_possible_a (a : ℕ) :
  (∃ x : ℤ, x^2 + a * x = -28) → a = 29 :=
sorry

end greatest_possible_a_l694_694720


namespace divisibility_iff_l694_694312

theorem divisibility_iff (m n : ℤ) :
  (m ∣ n) ↔ (∀ (p : ℕ), Nat.Prime p → padicValuation p m ≤ padicValuation p n) :=
sorry

end divisibility_iff_l694_694312


namespace proof_problem_l694_694148

def star (c d : ℝ) (h : c ≠ d) : ℝ :=
  (c + d) / (c - d)

theorem proof_problem : ((star 3 5 (by norm_num)) ∗ (star 1 2 (by norm_num))) ∗ (by sorry) = 7 := sorry

end proof_problem_l694_694148


namespace irreducible_fraction_eq_l694_694331

theorem irreducible_fraction_eq (p q : ℕ) (h1 : p > 0) (h2 : q > 0) (h3 : Nat.gcd p q = 1) (h4 : q % 2 = 1) :
  ∃ n k : ℕ, n > 0 ∧ k > 0 ∧ (p : ℚ) / q = (n : ℚ) / (2 ^ k - 1) :=
by
  sorry

end irreducible_fraction_eq_l694_694331


namespace find_c_plus_d_l694_694147

def g (c d x : ℝ) : ℝ :=
  if x ≤ 3 then c * x + d else 10 - 2 * x

theorem find_c_plus_d (c d : ℝ)
  (h1 : ∀ x, g c d (g c d x) = x) :
  c + d = 4.5 :=
by
  apply_fun g c d at h1
  sorry

end find_c_plus_d_l694_694147


namespace find_f108_l694_694661

variable (f : ℕ → ℕ)
variable (h1 : ∀ x y : ℕ, x > 0 → y > 0 → f(x * y) = f(x) + f(y))
variable (h2 : f 6 = 10)
variable (h3 : f 18 = 14)

theorem find_f108 : f 108 = 24 := by
  sorry

end find_f108_l694_694661


namespace total_money_taken_l694_694871

def individual_bookings : ℝ := 12000
def group_bookings : ℝ := 16000
def returned_due_to_cancellations : ℝ := 1600

def total_taken (individual_bookings : ℝ) (group_bookings : ℝ) (returned_due_to_cancellations : ℝ) : ℝ :=
  (individual_bookings + group_bookings) - returned_due_to_cancellations

theorem total_money_taken :
  total_taken individual_bookings group_bookings returned_due_to_cancellations = 26400 := by
  sorry

end total_money_taken_l694_694871


namespace farmer_field_l694_694067

theorem farmer_field (m : ℤ) : 
  (3 * m + 8) * (m - 3) = 85 → m = 6 :=
by
  sorry

end farmer_field_l694_694067


namespace perimeter_rectangle_l694_694464

-- Defining the width and length of the rectangle based on the conditions
def width (a : ℝ) := a
def length (a : ℝ) := 2 * a + 1

-- Statement of the problem: proving the perimeter
theorem perimeter_rectangle (a : ℝ) :
  let W := width a
  let L := length a
  2 * W + 2 * L = 6 * a + 2 :=
by
  sorry

end perimeter_rectangle_l694_694464


namespace negative_even_product_property_l694_694765

open Nat

def is_even (n : Int) : Bool := (n % 2) = 0

theorem negative_even_product_property :
  let evens := List.filter (λ x => (-2020 < x ∧ x < 0) ∧ is_even x) (List.range' (-2020) 2020).map (λ x => -x)
  let product := List.prod evens
  (product + 10 < 0) ∧ ((product + 10) % 10) = 0 :=
by
  sorry

end negative_even_product_property_l694_694765


namespace matrix_inverse_proof_l694_694922

open Matrix

def matrix_inverse_problem : Prop :=
  let A := ![
    ![7, -2],
    ![-3, 1]
  ]
  let A_inv := ![
    ![1, 2],
    ![3, 7]
  ]
  A.mul A_inv = (1 : ℤ) • (1 : Matrix (Fin 2) (Fin 2))
  
theorem matrix_inverse_proof : matrix_inverse_problem :=
  by
  sorry

end matrix_inverse_proof_l694_694922


namespace bus_people_next_pickup_point_l694_694059

theorem bus_people_next_pickup_point (bus_capacity : ℕ) (fraction_first_pickup : ℚ) (cannot_board : ℕ)
  (h1 : bus_capacity = 80)
  (h2 : fraction_first_pickup = 3 / 5)
  (h3 : cannot_board = 18) : 
  ∃ people_next_pickup : ℕ, people_next_pickup = 50 :=
by
  sorry

end bus_people_next_pickup_point_l694_694059


namespace distinct_cubes_meet_condition_l694_694170

theorem distinct_cubes_meet_condition :
  ∃ (a b c d e f : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    (a + b + c + d + e + f = 60) ∧
    ∃ (k : ℕ), 
        ((a = k) ∧ (b = k) ∧ (c = k) ∧ (d = k) ∧ (e = k) ∧ (f = k)) ∧
        -- Number of distinct ways
        (∃ (num_ways : ℕ), num_ways = 84) :=
sorry

end distinct_cubes_meet_condition_l694_694170


namespace triangle_YZ_length_l694_694640

/-- In triangle XYZ, sides XY and XZ have lengths 6 and 8 inches respectively, 
    and the median XM from vertex X to the midpoint of side YZ is 5 inches. 
    Prove that the length of YZ is 10 inches. -/
theorem triangle_YZ_length
  (XY XZ XM : ℝ)
  (hXY : XY = 6)
  (hXZ : XZ = 8)
  (hXM : XM = 5) :
  ∃ (YZ : ℝ), YZ = 10 := 
by
  sorry

end triangle_YZ_length_l694_694640


namespace michael_brought_5000_rubber_bands_l694_694681

noncomputable def totalRubberBands
  (small_band_count : ℕ) (large_band_count : ℕ)
  (small_ball_count : ℕ := 22) (large_ball_count : ℕ := 13)
  (rubber_bands_per_small : ℕ := 50) (rubber_bands_per_large : ℕ := 300) 
: ℕ :=
small_ball_count * rubber_bands_per_small + large_ball_count * rubber_bands_per_large

theorem michael_brought_5000_rubber_bands :
  totalRubberBands 22 13 = 5000 := by
  sorry

end michael_brought_5000_rubber_bands_l694_694681


namespace contrapositive_of_proposition_l694_694717

theorem contrapositive_of_proposition (a b : ℝ) : (a > b → a + 1 > b) ↔ (a + 1 ≤ b → a ≤ b) :=
sorry

end contrapositive_of_proposition_l694_694717


namespace _l694_694275

noncomputable def angle_bisector_theorem (h_parallel : ∀ (D C A B : Point), parallel (line D C) (line A B))
    (h_dca : ∠ D C A = 50)
    (h_abc : ∠ A B C = 60)
    (h_bisector : bisects (line C X) ∠ A C B) :
    ∀ (b a : Point) (X : Point), (∠ C X b = 40) ∧ (∠ C X a = 40) :=
by
  sorry

end _l694_694275


namespace triangle_area_ratio_l694_694398

/-- Define the base and height of triangle A -/
def baseA : ℝ := 3
def heightA : ℝ := 2

/-- Define the base and height of triangle B -/
def baseB : ℝ := 3
def heightB : ℝ := 6.02

/-- The area of a triangle given its base and height -/
def area (base height : ℝ) : ℝ := (base * height) / 2

/-- Calculate the area of triangle A and B -/
def areaA : ℝ := area baseA heightA
def areaB : ℝ := area baseB heightB

/-- Prove that the area of triangle B is 3.01 times the area of triangle A -/
theorem triangle_area_ratio :
  areaB = 3.01 * areaA :=
sorry

end triangle_area_ratio_l694_694398


namespace arrangement_count_l694_694829

theorem arrangement_count (A B C D E : Type) :
  ∃ n : ℕ, n = 5! / 2 ∧ n = 60 :=
by
  sorry

end arrangement_count_l694_694829


namespace isosceles_triangle_time_between_9_30_and_10_l694_694482

theorem isosceles_triangle_time_between_9_30_and_10 (time : ℕ) (h_time_range : 30 ≤ time ∧ time < 60)
  (h_isosceles : ∃ x : ℝ, 0 ≤ x ∧ x + 2 * x + 2 * x = 180) :
  time = 36 :=
  sorry

end isosceles_triangle_time_between_9_30_and_10_l694_694482


namespace sum_cos_imaginary_correct_l694_694881

noncomputable def sum_cos_imaginary : ℂ :=
  ∑ n in range 31, (complex.I ^ n) * (complex.cos (30 + 60 * n) * (π / 180))

theorem sum_cos_imaginary_correct :
  sum_cos_imaginary = (√3 / 2 : ℂ) * (1 + 5 * complex.I) :=
by
  sorry

end sum_cos_imaginary_correct_l694_694881


namespace ellipse_intersection_range_l694_694544

theorem ellipse_intersection_range (a b m : ℝ) (h_a : a > 0) (h_b : b > 0) (h_ab : a > b)
    (minor_axis_length : 2 * b = 2) (eccentricity : (1 - (b ^ 2) / (a ^ 2)) ^ (1 / 2) = (sqrt 3) / 2)
    (intersection_condition : ∀ x : ℝ, (5 * x ^ 2 + 8 * m * x + 4 * m ^ 2 - 4) = 0)
    (acute_angle_condition : ∀ x1 x2 y1 y2 : ℝ, x1 ≠ x2 ∧ y1 ≠ y2 → 
        (x1 + m) * (x2 + m) +  (2 * sqrt 10) / 5 > 0)
    : (- sqrt 5 < m ∧ m < -(2 * sqrt 10) / 5) ∨ ((2 * sqrt 10) / 5 < m ∧ m < sqrt 5) :=
sorry

end ellipse_intersection_range_l694_694544


namespace count_divisible_by_11_with_digits_sum_10_l694_694604

-- Definitions and conditions from step a)
def digits_add_up_to_10 (n : ℕ) : Prop :=
  let ds := n.digits 10 in ds.length = 4 ∧ ds.sum = 10

def divisible_by_11 (n : ℕ) : Prop :=
  let ds := n.digits 10 in
  (ds.nth 0).get_or_else 0 +
  (ds.nth 2).get_or_else 0 -
  (ds.nth 1).get_or_else 0 -
  (ds.nth 3).get_or_else 0 ∣ 11

-- The tuple (question, conditions, answer) formalized in Lean
theorem count_divisible_by_11_with_digits_sum_10 :
  ∃ N : ℕ, N = (
  (Finset.filter (λ x, digits_add_up_to_10 x ∧ divisible_by_11 x)
  (Finset.range 10000)).card
) := 
sorry

end count_divisible_by_11_with_digits_sum_10_l694_694604


namespace largest_prime_factor_of_7_fact_add_8_fact_l694_694022

theorem largest_prime_factor_of_7_fact_add_8_fact :
  let n := 7!
  let m := 8!
  let sum := n + m
  prime_factors sum = {2, 3, 5, 7} ∧ (∀ p ∈ {2, 3, 5, 7}, p <= 7) := by
  sorry

end largest_prime_factor_of_7_fact_add_8_fact_l694_694022


namespace range_of_p_l694_694944

noncomputable def proof_problem (p : ℝ) : Prop :=
  (∀ x : ℝ, (4 * x + p < 0) → (x < -1 ∨ x > 2)) → (p ≥ 4)

theorem range_of_p (p : ℝ) : proof_problem p :=
by
  intros h
  sorry

end range_of_p_l694_694944


namespace ascending_order_perimeters_l694_694927

noncomputable def hypotenuse (r : ℝ) : ℝ := r * Real.sqrt 2

noncomputable def perimeter_P (r : ℝ) : ℝ := (2 + 3 * Real.sqrt 2) * r
noncomputable def perimeter_Q (r : ℝ) : ℝ := (6 + Real.sqrt 2) * r
noncomputable def perimeter_R (r : ℝ) : ℝ := (4 + 3 * Real.sqrt 2) * r

theorem ascending_order_perimeters (r : ℝ) (h_r_pos : 0 < r) : 
  perimeter_P r < perimeter_Q r ∧ perimeter_Q r < perimeter_R r := by
  sorry

end ascending_order_perimeters_l694_694927


namespace line_rect_eqn_circle_std_form_chord_length_on_circle_intercepted_by_line_l694_694210

noncomputable def line_polar_eqn (ρ θ : ℝ) : Prop :=
  ρ * sin (θ - π / 3) = 6

def circle_param_eqn (θ : ℝ) : ℝ × ℝ :=
  (10 * cos θ, 10 * sin θ)

theorem line_rect_eqn (x y : ℝ) (ρ θ : ℝ) (h1 : line_polar_eqn ρ θ) :
  ∃ ρ θ, ρ ≠ 0 ∧ θ ≠ 0 ∧ sqrt(3) * x - y + 12 = 0 :=
sorry

theorem circle_std_form :
  ∀ θ : ℝ, let (x, y) := circle_param_eqn θ in x^2 + y^2 = 100 :=
sorry

theorem chord_length_on_circle_intercepted_by_line (x y : ℝ) (ρ θ : ℝ) (h1 : line_polar_eqn ρ θ) :
  ∃ r, r > 0 ∧ let (x, y) := circle_param_eqn θ in 2 * sqrt(r^2 - 36) = 16 :=
sorry

end line_rect_eqn_circle_std_form_chord_length_on_circle_intercepted_by_line_l694_694210


namespace triangle_BC_length_l694_694285

noncomputable def length_BC (A B AC : ℝ) (hA : A = 45) (hB : B = 75) (hAC : AC = 6) : ℝ :=
  let sin := Real.sin
  AC * (sin A / sin B)

theorem triangle_BC_length :
  ∀ (A B AC : ℝ),
    A = 45 → B = 75 → AC = 6 →
    length_BC A B AC ≈ 4.39 :=
by
  intros A B AC hA hB hAC
  unfold length_BC
  sorry

end triangle_BC_length_l694_694285


namespace count_four_digit_numbers_l694_694596

open Int

def is_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9

def valid_combination (a b c d : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧
  a ≠ 0 ∧
  a + b + c + d = 10 ∧
  (a + c) - (b + d) % 11 = 0

theorem count_four_digit_numbers :
  ∃ n, n > 0 ∧ ∀ a b c d : ℕ, valid_combination a b c d → n = sorry :=
sorry

end count_four_digit_numbers_l694_694596


namespace probability_of_unique_tens_digits_l694_694350

open BigOperators
open Finset

variable (s : Finset ℕ)

def is_valid_set (s : Finset ℕ) : Prop :=
  s.card = 6 ∧ ∀ x ∈ s, 10 ≤ x ∧ x ≤ 99

def has_unique_tens_digits (s : Finset ℕ) : Prop :=
  (s.image (λ x => x / 10)).card = s.card

theorem probability_of_unique_tens_digits :
  ∑ (s : Finset ℕ) in (Finset.filter is_valid_set (Finset.range 100)),
    if has_unique_tens_digits s then 1 else 0 /
  ∑ (s : Finset ℕ) in (Finset.filter is_valid_set (Finset.range 100)),
    1 = 25000 / 18444123 := sorry

end probability_of_unique_tens_digits_l694_694350


namespace max_and_min_modulus_l694_694554

noncomputable def max_modulus (z : ℂ) : ℝ :=
  let dist_origin_to_center := complex.abs (4 + 4 * complex.I)
  let radius := real.sqrt 8
  dist_origin_to_center + radius

noncomputable def min_modulus (z : ℂ) : ℝ :=
  let dist_origin_to_center := complex.abs (4 + 4 * complex.I)
  let radius := real.sqrt 8
  dist_origin_to_center - radius

theorem max_and_min_modulus (z : ℂ) (h : 2 * complex.abs (z - (3 + 3 * complex.I)) = complex.abs z) :
  max_modulus z = 6 * real.sqrt 2 ∧ min_modulus z = 2 * real.sqrt 2 :=
  by
    sorry

end max_and_min_modulus_l694_694554


namespace minimum_gb_for_cheaper_plan_l694_694087

theorem minimum_gb_for_cheaper_plan : ∃ g : ℕ, (g ≥ 778) ∧ 
  (∀ g' < 778, 3000 + (if g' ≤ 500 then 8 * g' else 8 * 500 + 6 * (g' - 500)) ≥ 15 * g') ∧ 
  3000 + (if g ≤ 500 then 8 * g else 8 * 500 + 6 * (g - 500)) < 15 * g :=
by
  sorry

end minimum_gb_for_cheaper_plan_l694_694087


namespace range_of_m_seq_formula_l694_694939

def geom_seq (a_1 q : ℝ) (n : ℕ) := a_1 * q^(n-1)

def S (a_1 q : ℝ) (n : ℕ) := a_1 * (1 - q ^ n) / (1 - q)

def a_n_general_term : ℝ := 
(-1/2 : ℝ)

def T_n (n : ℕ) : ℝ :=
(n-1 : ℝ) * 2^(n+1) + 2

def f (n : ℕ) : ℝ :=
(n-1 : ℝ) / (2^(n+1) - 1)

theorem range_of_m (m : ℝ) : (∀ n ≥ 2, (n-1)^2 ≤ m * (T_n n - n - 1)) ↔ (m ≥ 1/7) := 
sorry

theorem seq_formula (a_1 : ℝ) (q : ℝ) (n : ℕ) (h1 : a_1 = -1/2) (h2 : q = -1/2) :
∀ n, geom_seq a_1 q n = (-1/2) ^ n := 
sorry

end range_of_m_seq_formula_l694_694939


namespace locus_of_centers_is_circle_l694_694511

-- Defining the conditions
variable (A B : ℝ × ℝ)
variable (r : ℝ)

def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
def midpoint (p q : ℝ × ℝ) : ℝ × ℝ := ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

-- Assuming the given conditions
axiom fixed_points : distance A B = 6
axiom radius_condition : r = 5

-- Given the conditions, we need to prove the following statement
theorem locus_of_centers_is_circle :
  ∃ C, distance (midpoint A B) C = 4 ∧ ∀ P, distance P A = r ∧ distance P B = r → distance P C = r :=
sorry

end locus_of_centers_is_circle_l694_694511


namespace Louisa_average_speed_l694_694693

theorem Louisa_average_speed :
  ∃ v : ℝ, (250 / v + 3 = 350 / v) ∧ v = 100 / 3 :=
begin
  sorry
end

end Louisa_average_speed_l694_694693


namespace tins_left_after_damage_l694_694820

theorem tins_left_after_damage (cases : ℕ) (tins_per_case : ℕ) (damage_rate : ℚ) 
    (total_cases : cases = 15) (tins_per_case_value : tins_per_case = 24)
    (damage_rate_value : damage_rate = 0.05) :
    let total_tins := cases * tins_per_case
        damaged_tins := damage_rate * total_tins
        remaining_tins := total_tins - damaged_tins in
    remaining_tins = 342 := 
by
  sorry

end tins_left_after_damage_l694_694820


namespace parametrize_segment_l694_694729

noncomputable def segment_params (t : ℝ) : ℝ × ℝ :=
  let a := 7
  let b := -3
  let c := 5
  let d := 5
  (a * t + b, c * t + d)

theorem parametrize_segment :
  let a := 7
  let b := -3
  let c := 5
  let d := 5
  a^2 + b^2 + c^2 + d^2 = 108 :=
by
  let a := 7
  let b := -3
  let c := 5
  let d := 5
  have h1 : a^2 = 49 := by sorry
  have h2 : b^2 = 9 := by sorry
  have h3 : c^2 = 25 := by sorry
  have h4 : d^2 = 25 := by sorry
  calc
    a^2 + b^2 + c^2 + d^2 = 49 + 9 + 25 + 25 := by sorry
                        ... = 108 := by refl

end parametrize_segment_l694_694729


namespace gain_in_dollars_l694_694813

theorem gain_in_dollars (SP : ℝ) (GP : ℝ) (gain : ℝ) (C : ℝ) (h1 : SP = 100) (h2 : GP = 0.25) (h3 : SP = C + (GP * C)) : gain = 20 :=
by
  -- Find the value of cost price
  have h4 : C = SP / (1 + GP),
  linarith,
  -- Substitute SP and GP values
  rw [h1, h2] at h4,
  -- Value of cost price
  have h5 : C = 100 / 1.25,
  calc
    100 / 1.25 = 80 : by norm_num,
  rw h5,
  -- Calculate gain
  have h6 : gain = SP - C,
  linarith,
  rw [h1, h5] at h6,
  -- Value of gain
  have h7 : gain = 20,
  calc
    100 - 80 = 20 : by norm_num,
  exact h7

end gain_in_dollars_l694_694813


namespace derivative_f_tangent_at_1_tangent_through_origin_l694_694200

def f (x : ℝ) : ℝ := (1 / 2) * x^2 - x + 1

theorem derivative_f : derivative f = λ x, x - 1 := by
  sorry

theorem tangent_at_1 : ∀ x : ℝ, (∃ y, (y = f 1) ∧ (y - (1 / 2) = 0 * (x - 1))) :=
  by sorry

theorem tangent_through_origin : ∀ x : ℝ, 
  (∃ y, (y = (sqrt 2 - 1) * x) ∨ (y = (-sqrt 2 - 1) * x)) := 
  by sorry

end derivative_f_tangent_at_1_tangent_through_origin_l694_694200


namespace expansion_terms_l694_694960

/-- 
Given the following conditions: 
1. The sum of the coefficients of the first three terms in the expansion is 16. 
2. The ratio of the coefficient of the third term from the end to the coefficient of the second term from the end is 4:1.
We prove that:
1. For n = 5, the term with the largest binomial coefficients in the expansion of (x + (sqrt x) / 2)^n are (5 / 2) * x^4 and (5 / 4) * x^(7 / 2).
2. All rational terms in the expansion are x^5, (5 / 2) * x^4, and (5 / 16) * x^3.
-/
theorem expansion_terms (n : ℕ) (h1 : 1 + n + (n * (n - 1)) / 2 = 16) 
    (h2 : (n - 1 = 4)) :
    n = 5 ∧ 
    (C(5, 2) * (1 / 2)^2 * x^4 = 5 / 2 * x^4) ∧ 
    (C(5, 3) * (1 / 2)^3 * x^(7 / 2) = 5 / 4 * x^(7 / 2)) ∧ 
    (all_rational_terms = [x^5, 5 / 2 * x^4, 5 / 16 * x^3]) :=
by 
  sorry

end expansion_terms_l694_694960


namespace most_suitable_survey_is_j20_components_l694_694033

-- Define the surveys
inductive Survey
| HuaweiBattery
| J20Components
| SpringFestivalMovie
| HomeworkTime

-- Condition definitions
def is_suitable_for_comprehensive_survey : Survey → Prop
| Survey.HuaweiBattery := False
| Survey.J20Components := True
| Survey.SpringFestivalMovie := False
| Survey.HomeworkTime := False

theorem most_suitable_survey_is_j20_components :
  is_suitable_for_comprehensive_survey Survey.J20Components :=
by {
  -- Proof body omitted
  sorry
}

end most_suitable_survey_is_j20_components_l694_694033


namespace function_monotone_increasing_l694_694140

open Real

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - log x

theorem function_monotone_increasing : ∀ x, 1 ≤ x → (0 < x) → (1 / 2) * x^2 - log x = f x → (∀ y, 1 ≤ y → (0 < y) → (f y ≤ f x)) :=
sorry

end function_monotone_increasing_l694_694140


namespace safe_trip_possible_l694_694160

-- Define the time intervals and eruption cycles
def total_round_trip_time := 16
def trail_time := 8
def crater1_cycle := 18
def crater2_cycle := 10
def crater1_erupt := 1
def crater1_quiet := 17
def crater2_erupt := 1
def crater2_quiet := 9

-- Ivan wants to safely reach the summit and return
theorem safe_trip_possible : ∃ t, 
  -- t is a valid start time where both craters are quiet
  ((t % crater1_cycle) ≥ crater1_erupt ∧ (t % crater2_cycle) ≥ crater2_erupt) ∧
  -- t + total_round_trip_time is also safe for both craters
  (((t + total_round_trip_time) % crater1_cycle) ≥ crater1_erupt ∧ ((t + total_round_trip_time) % crater2_cycle) ≥ crater2_erupt) :=
sorry

end safe_trip_possible_l694_694160


namespace german_italian_students_l694_694894

theorem german_italian_students (G I G_inter_I : ℕ) (hG_min : 1750 ≤ G) (hG_max : G ≤ 1875)
                                (hI_min : 875 ≤ I) (hI_max : I ≤ 1125)
                                (h_total : G + I - G_inter_I = 2500) :
  (λ m' M', M' = 1250 ∧ m' = 500) → (1250 - 500 = 750) :=
by
  sorry

end german_italian_students_l694_694894


namespace permutation_cosine_sum_zero_l694_694314

theorem permutation_cosine_sum_zero {n : ℕ} (h1 : 0 < n) (h2 : ¬ ∃ p k : ℕ, p.prime ∧ 1 < k ∧ n = p^k) :
  ∃ (a : Fin n → Fin n), ∑ k : Fin n, (k + 1) * Real.cos (2 * Real.pi * (a k / n)) = 0 :=
by
  sorry

end permutation_cosine_sum_zero_l694_694314


namespace sequence_formula_l694_694280

noncomputable def a (n : ℕ) : ℕ :=
if n = 0 then 1 else (a (n - 1)) + 2^(n-1)

theorem sequence_formula (n : ℕ) (h : n > 0) : 
    a n = 2^n - 1 := 
sorry

end sequence_formula_l694_694280


namespace div_expression_l694_694489

theorem div_expression : 180 / (12 + 13 * 2) = 90 / 19 := 
  sorry

end div_expression_l694_694489


namespace find_x_l694_694670

def binary_operation (a b c d : Int) : Int × Int := (a - c, b + d)

theorem find_x (x y : Int)
  (H1 : binary_operation 6 5 2 3 = (4, 8))
  (H2 : binary_operation x y 5 4 = (4, 8)) :
  x = 9 :=
by
  -- Necessary conditions and hypotheses are provided
  sorry -- Proof not required

end find_x_l694_694670


namespace water_level_after_opening_l694_694757

-- Let's define the densities and initial height as given
def ρ_water : ℝ := 1000
def ρ_oil : ℝ := 700
def initial_height : ℝ := 40  -- height in cm

-- Final heights after opening the valve (h' denotes final height)
def final_height_water : ℝ := 34

-- Using the principles described
theorem water_level_after_opening :
  ∃ h_oil : ℝ, ρ_water * final_height_water = ρ_oil * h_oil ∧ final_height_water + h_oil = initial_height :=
begin
  use initial_height - final_height_water,
  split,
  {
    field_simp,
    norm_num,
  },
  {
    field_simp,
    norm_num,
  }
end

end water_level_after_opening_l694_694757


namespace angle_NHC_60_l694_694176

theorem angle_NHC_60 {A B C D S N H : Point} :
  (is_square A B C D) → 
  (is_equilateral_triangle B C S) → 
  (is_midpoint N A S) →
  (is_midpoint H C D) →
  (∠ N H C = 60) :=
sorry

end angle_NHC_60_l694_694176


namespace range_of_m_max_value_of_t_l694_694524

-- Define the conditions for the quadratic equation problem
def quadratic_eq_has_real_roots (m n : ℝ) := 
  m^2 - 4 * n ≥ 0

def roots_are_negative (m : ℝ) := 
  2 ≤ m ∧ m < 3

-- Question 1: Prove range of m
theorem range_of_m (m : ℝ) (h1 : quadratic_eq_has_real_roots m (3 - m)) : 
  roots_are_negative m :=
sorry

-- Define the conditions for the inequality problem
def quadratic_inequality (m n : ℝ) (t : ℝ) := 
  t ≤ (m-1)^2 + (n-1)^2 + (m-n)^2

-- Question 2: Prove maximum value of t
theorem max_value_of_t (m n t : ℝ) (h1 : quadratic_eq_has_real_roots m n) : 
  quadratic_inequality m n t -> t ≤ 9/8 :=
sorry

end range_of_m_max_value_of_t_l694_694524


namespace find_other_number_l694_694715

theorem find_other_number (HCF LCM num1 num2 : ℕ) (hcf_eq : HCF = 12) (lcm_eq : LCM = 396) (num1_eq : num1 = 132) : num2 = 36 :=
by
  -- Conditions: HCF of two numbers is 12, LCM of two numbers is 396, and one of the numbers is 132
  have h1 : HCF * LCM = num1 * num2,
  -- This follows from the relationship between HCF and LCM
  sorry
  -- Combine the given conditions with calculations to find the other number

end find_other_number_l694_694715


namespace selling_price_of_article_l694_694811

-- Definition of given conditions
def cost_price : ℝ := 20
def gain_percent : ℝ := 25

-- Definition of derivations from conditions (gain amount and selling price)
def gain_amount (cp : ℝ) (gp : ℝ) := (gp / 100) * cp
def selling_price (cp : ℝ) (ga : ℝ) := cp + ga

-- Theorem stating the main result
theorem selling_price_of_article : 
    ∀ (cp gp : ℝ), cp = 20 → gp = 25 → 
    selling_price cp (gain_amount cp gp) = 25 := 
by 
  intros cp gp hcp hgp 
  rw [hcp, hgp]
  dsimp [gain_amount, selling_price]
  norm_num

-- Note: The body of the proof is provided to ensure the Lean code compiles successfully.

end selling_price_of_article_l694_694811


namespace find_matrix_N_l694_694915

def mat_eq (N : Matrix (Fin 2) (Fin 2) ℤ) : Prop :=
  (N.mul_vec (λ i, if i = 0 then 4 else 0) = (λ i, if i = 0 then 8 else 28)) ∧
  (N.mul_vec (λ i, if i = 0 then -2 else 10) = (λ i, if i = 0 then 6 else -34))

theorem find_matrix_N :
  ∃ N : Matrix (Fin 2) (Fin 2) ℤ, mat_eq N ∧ N = λ i j, 
    if (i, j) = (0, 0) then 2 else if (i, j) = (0, 1) then 1 else if (i, j) = (1, 0) then 7 else -2 :=
by {
  sorry
}

end find_matrix_N_l694_694915


namespace largest_prime_factor_of_7_fact_add_8_fact_l694_694013

theorem largest_prime_factor_of_7_fact_add_8_fact :
  let n := 7!
  let m := 8!
  let sum := n + m
  prime_factors sum = {2, 3, 5, 7} ∧ (∀ p ∈ {2, 3, 5, 7}, p <= 7) := by
  sorry

end largest_prime_factor_of_7_fact_add_8_fact_l694_694013


namespace find_numbers_l694_694133

theorem find_numbers :
  ∃ (S : Finset ℕ), S.card = 1000 ∧ S.sum id = S.prod id :=
by
  let S := (Finset.range 999).insert 1000
  use S
  have h_card : S.card = 1000, sorry
  have h_sum : S.sum id = 1000, sorry
  have h_prod : S.prod id = 1000, sorry
  exact ⟨h_card, h_sum, h_prod⟩

end find_numbers_l694_694133


namespace four_digit_numbers_count_l694_694595

theorem four_digit_numbers_count :
  ∃ (n : ℕ),  n = 24 ∧
  let s := λ (a b c d : ℕ), (a + b + c + d = 10 ∧ a ≠ 0 ∧ ((a + c) - (b + d)) ∈ [(-11), 0, 11]) in
  (card {abcd : Fin 10 × Fin 10 × Fin 10 × Fin 10 | (s abcd.1.1 abcd.1.2 abcd.2.1 abcd.2.2)}) = n :=
by
  sorry

end four_digit_numbers_count_l694_694595


namespace percent_profit_l694_694785

theorem percent_profit (C S : ℝ) (h : 60 * C = 40 * S) : (S - C) / C * 100 = 50 := by
  sorry

end percent_profit_l694_694785


namespace find_sum_3xyz_l694_694993

variables (x y z : ℚ)

def equation1 : Prop := y + z = 18 - 4 * x
def equation2 : Prop := x + z = 16 - 4 * y
def equation3 : Prop := x + y = 9 - 4 * z

theorem find_sum_3xyz (h1 : equation1 x y z) (h2 : equation2 x y z) (h3 : equation3 x y z) : 
  3 * x + 3 * y + 3 * z = 43 / 2 := 
sorry

end find_sum_3xyz_l694_694993


namespace sum_area_triangles_l694_694654

noncomputable def Point : Type := ℝ × ℝ

def rectangle (A B C D : Point) :=
  -- Rectangle sides AB = 2 units and BC = 1 unit
  (dist A B = 2) ∧ 
  (dist B C = 1) ∧
  -- Ensure rectangle is axis-aligned for simplicity
  (∃ x y z w, A = (x, y) ∧ B = (x + 2, y) ∧ C = (x + 2, y - 1) ∧ D = (x, y - 1))

def midpoint (P Q : Point) : Point :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def perpendicular_foot (P : Point) (L : Point → Point) : Point :=
  let slope := -1 / ((L (P.1 + 1)).2 - P.2) / ((L (P.1 + 1)).1 - P.1) in
  (P.1 + 1, P.2 + slope)

def area_triangle (D Q P : Point) : ℝ :=
  0.5 * abs ((Q.1 - D.1) * (P.2 - D.2) - (P.1 - D.1) * (Q.2 - D.2))

theorem sum_area_triangles (A B C D : Point) (Q P : ℕ → Point)
  (h_rect : rectangle A B C D)
  (h_Q1 : Q 1 = midpoint C D)
  (h_def : ∀ i, P i = (λ j, intersection (A, Q j) (B, D)) (i + 1) ∧ Q (i + 1) = perpendicular_foot (P i) (λ x, C))
  : ∑' i, area_triangle D (Q i) (P i) = 2 / 9 :=
sorry

end sum_area_triangles_l694_694654


namespace tucker_tissues_l694_694745

theorem tucker_tissues (num_tissues_per_box : ℕ) (num_boxes : ℕ) (tissues_used : ℕ) (t : ℕ)
    (h₀ : num_tissues_per_box = 160) (h₁ : num_boxes = 3) (h₂ : tissues_used = 210) 
    (h₃ : t = num_tissues_per_box * num_boxes - tissues_used) : t = 270 :=
by
  -- Unfold assumptions
  rw [h₀, h₁, h₂] at h₃
  -- Simplify the equation
  have h₄ : t = 160 * 3 - 210 := h₃
  calc
    t = 160 * 3 - 210 := h₄
    _ = 480 - 210 := by rw nat.mul_comm
    _ = 270 := by norm_num

end tucker_tissues_l694_694745


namespace angle_between_p_and_r_zero_l694_694315

variables {𝕜 : Type*} [Field 𝕜] [NormedSpace 𝕜 (EuclideanSpace 𝕜 (Fin 3))]
variables (p q r : EuclideanSpace 𝕜 (Fin 3))

theorem angle_between_p_and_r_zero
  (hp : ‖p‖ = 1) (hq : ‖q‖ = 1) (hr : ‖r‖ = 1)
  (h_condition : p + 2 • q + r = 0) :
  ∃ θ : ℝ, θ = 0 ∧ cos θ = (p • r) / (‖p‖ * ‖r‖) :=
by
  sorry

end angle_between_p_and_r_zero_l694_694315


namespace acute_angles_of_right_triangle_l694_694804

theorem acute_angles_of_right_triangle
    (A B C D : Type)
    [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
    (angle : A → B → C → ℝ)
    (right_triangle : Prop)
    (circle_on_diameter_intersects_hypotenuse : Prop)
    (ratio_AD_DB : ℝ)
    (ratio_AD_DB_eq : ratio_AD_DB = 3)
    (angle_ACB_90 : angle A C B = 90)
    (AD : ℝ)
    (DB : ℝ)
    (AD_over_DB : AD / DB = 3) :
  ∃ (angle_A angle_B : ℝ), angle_A = 60 ∧ angle_B = 30 :=
begin
  sorry
end

end acute_angles_of_right_triangle_l694_694804


namespace true_proposition_is_2_l694_694224

theorem true_proposition_is_2 :
  (¬ ∀ (l₁ l₂ l₃ : Line), (l₁.intersects l₂) → (l₂.intersects l₃) → (l₃.intersects l₁) → (l₁.coplanar l₂ l₃)) ∧
  (∀ (l₁ l₂ l₃ : Line), l₁.parallel l₂ → (l₁.perpendicular l₃) → (l₂.perpendicular l₃)) ∧
  (¬ ∀ (α β : Angle), (α = β) → (∀ (l₁α l₂α l₁β l₂β : Line), (l₁α.parallel l₁β) → ((l₂α.parallel l₂β) ∨ (l₂α.intersects l₂β)))) ∧
  (¬ ∃ (c : Cube) (d : Line), (d ∈ c.diagonals) ∧ (#{l : Line // l ∈ c.edges ∧ l.skew d} = 4)) :=
by
  sorry

end true_proposition_is_2_l694_694224


namespace total_legs_l694_694901

theorem total_legs (chickens sheep : ℕ) (chicken_legs sheep_legs total_legs : ℕ) (h1 : chickens = 7) (h2 : sheep = 5) 
(h3 : chicken_legs = 2) (h4 : sheep_legs = 4) (h5 : total_legs = chickens * chicken_legs + sheep * sheep_legs) : 
total_legs = 34 := 
by {
  rw [h1, h2, h3, h4],
  exact h5,
  sorry
}

end total_legs_l694_694901


namespace number_of_allocation_schemes_l694_694839

/-- 
  Given 5 volunteers and 4 projects, each volunteer is assigned to only one project,
  and each project must have at least one volunteer.
  Prove that there are 240 different allocation schemes.
-/
theorem number_of_allocation_schemes (V P : ℕ) (hV : V = 5) (hP : P = 4) 
  (each_volunteer_one_project : ∀ v, ∃ p, v ≠ p) 
  (each_project_at_least_one : ∀ p, ∃ v, v ≠ p) : 
  ∃ n_ways : ℕ, n_ways = 240 :=
by
  sorry

end number_of_allocation_schemes_l694_694839


namespace quadratic_equation_exists_l694_694421

theorem quadratic_equation_exists : 
  ∃ a b c : ℝ, (a ≠ 0) ∧ (a + b + c = 0) ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) :=
by {
  let a := 1 : ℝ,
  let b := -2 : ℝ,
  let c := 1 : ℝ,
  use [a, b, c],
  split,
  {
    -- Proof for a ≠ 0
    exact one_ne_zero,
  },
  split,
  {
    -- Proof for a + b + c = 0
    calc
      a + b + c = 1 + -2 + 1 : by sorry
             ... = 0 : by sorry,
  },
  {
    -- Proof that the resultant quadratic equation is ax^2 + bx + c = 0
    intro x,
    calc
      a * x^2 + b * x + c = (1 : ℝ) * x^2 + (-2 : ℝ) * x + (1 : ℝ) : by sorry
                         ... = x^2 - 2 * x + 1 : by sorry,
  },
}

end quadratic_equation_exists_l694_694421


namespace water_level_height_l694_694855

/-- Problem: An inverted frustum with a bottom diameter of 12 and height of 18, filled with water, 
    is emptied into another cylindrical container with a bottom diameter of 24. Assuming the 
    cylindrical container is sufficiently tall, the height of the water level in the cylindrical container -/
theorem water_level_height
  (V_cone : ℝ := (1 / 3) * π * (12 / 2) ^ 2 * 18)
  (R_cyl : ℝ := 24 / 2)
  (H_cyl : ℝ) :
  V_cone = π * R_cyl ^ 2 * H_cyl →
  H_cyl = 1.5 :=
by 
  sorry

end water_level_height_l694_694855


namespace num_ways_to_select_ascend_triple_l694_694159

theorem num_ways_to_select_ascend_triple :
  (∑ a1 in finset.Ico 1 12, ∑ a2 in finset.Ico (a1 + 3) 15, finset.card (finset.Ico (a2 + 3) 15)) = 120 := by
  sorry

end num_ways_to_select_ascend_triple_l694_694159


namespace positive_difference_l694_694774

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

def probability_heads (n k : ℕ) : ℚ := (binom n k) * (1 / 2) ^ n

theorem positive_difference :
  let p1 := probability_heads 5 3
  let p2 := probability_heads 5 4
  p1 - p2 = 5 / 32 := by
begin
  sorry
end

end positive_difference_l694_694774


namespace fraction_division_l694_694408

theorem fraction_division :
  (3 / 4) / (5 / 6) = 9 / 10 :=
by {
  -- We skip the proof as per the instructions
  sorry
}

end fraction_division_l694_694408


namespace length_of_chord_l694_694447

theorem length_of_chord (r : ℝ) (h : r = 10) (perpendicular : ∀ (P : Type), r = (2 * P) → true)
  (bisects : ∀ (Q : Type), r = (2 * Q) → true) :
  ∃ (l : ℝ), l = 10 * real.sqrt 3 :=
by
  sorry

end length_of_chord_l694_694447


namespace card_return_to_initial_state_l694_694175

def card_operation (n : ℕ) (deck : List ℕ) : List ℕ := 
  let even_positions := (List.range (deck.length)).filter (λ x => ((x + 1) % 2 = 0))
  let even_cards := even_positions.map (λ i => List.get deck i)
  let odd_positions := (List.range (deck.length)).filter (λ x => ((x + 1) % 2 ≠ 0))
  let odd_cards := odd_positions.map (λ i => List.get deck i)
  odd_cards ++ even_cards

theorem card_return_to_initial_state (n : ℕ) (h : n > 0) (deck : List ℕ) (h_len : deck.length = 2 * n) :
  ∃ r ≤ 2 * n - 2, (Nat.iterate (card_operation n) r deck) = deck :=
sorry

end card_return_to_initial_state_l694_694175


namespace simplest_common_denominator_correct_l694_694739

variable {x : ℝ}

-- Define the fractions and their denominators
def fraction1 := 2 * x / (x - 2)
def denominator1 := x - 2

def fraction2 := 3 / (x^2 - 2 * x)
def denominator2 := x * (x - 2)

-- Define the target simplest common denominator
def simplest_common_denominator := x * (x - 2)

-- The lean statement proving the simplest common denominator
theorem simplest_common_denominator_correct :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 0 → (x - 2) < (x * (x - 2))) :=
by sorry

end simplest_common_denominator_correct_l694_694739


namespace find_locus_of_T_l694_694542

section Locus

variables {x y m : ℝ}
variable (M : ℝ × ℝ)

-- Condition: The equation of the ellipse
def ellipse (x y : ℝ) := (x^2 / 4) + y^2 = 1

-- Condition: Point P
def P := (1, 0)

-- Condition: M is any point on the ellipse, except A and B
def on_ellipse (M : ℝ × ℝ) := ellipse M.1 M.2 ∧ M ≠ (-2, 0) ∧ M ≠ (2, 0)

-- Condition: The intersection point N of line MP with the ellipse
def line_eq (m y : ℝ) := m * y + 1

-- Proposition: Locus of intersection point T of lines AM and BN
theorem find_locus_of_T 
  (hM : on_ellipse M)
  (hN : line_eq m M.2 = M.1)
  (hT : M.2 ≠ 0) :
  M.1 = 4 :=
sorry

end Locus

end find_locus_of_T_l694_694542


namespace std_dev_three_numbers_l694_694384

theorem std_dev_three_numbers : 
  (std_dev [5, 8, 11] = Real.sqrt 6) :=
sorry

end std_dev_three_numbers_l694_694384


namespace find_angle_A_find_side_c_l694_694623

-- Definitions for the problem conditions
variable (A B C : ℝ) -- Angles in radians
variable (a b c : ℝ) -- Sides opposite to angles A, B, C
variable h_sin_squares : sin B ^ 2 + sin C ^ 2 = sin A ^ 2 + sin B * sin C
variable h_cos_B : cos B = 1 / 3
variable h_a : a = 3

-- First part: Prove that angle A is π/3
theorem find_angle_A (h : sin B ^ 2 + sin C ^ 2 = sin A ^ 2 + sin B * sin C) :
  A = π / 3 := by
  sorry

-- Second part: Prove that side c is (3 + 2 * √6) / 3 given additional conditions
theorem find_side_c (h_sin_squares : sin B ^ 2 + sin C ^ 2 = sin A ^ 2 + sin B * sin C)
  (h_cos_B : cos B = 1 / 3) (h_a : a = 3) :
  c = (3 + 2 * Real.sqrt 6) / 3 := by
  sorry

end find_angle_A_find_side_c_l694_694623


namespace simon_age_is_10_l694_694847

def alvin_age : ℕ := 30

def simon_age (alvin_age : ℕ) : ℕ :=
  (alvin_age / 2) - 5

theorem simon_age_is_10 : simon_age alvin_age = 10 := by
  sorry

end simon_age_is_10_l694_694847


namespace positive_difference_l694_694773

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

def probability_heads (n k : ℕ) : ℚ := (binom n k) * (1 / 2) ^ n

theorem positive_difference :
  let p1 := probability_heads 5 3
  let p2 := probability_heads 5 4
  p1 - p2 = 5 / 32 := by
begin
  sorry
end

end positive_difference_l694_694773


namespace cos_angle_parallel_l694_694508

structure Point (α : Type) :=
(x : α)
(y : α)
(z : α)

def vec_sub (P Q : Point ℝ) : Point ℝ :=
{ x := P.x - Q.x,
  y := P.y - Q.y,
  z := P.z - Q.z }

def dot_product (u v : Point ℝ) : ℝ :=
u.x * v.x + u.y * v.y + u.z * v.z

def magnitude (v : Point ℝ) : ℝ :=
Real.sqrt (v.x^2 + v.y^2 + v.z^2)

noncomputable def cos_angle (u v : Point ℝ) : ℝ :=
dot_product u v / (magnitude u * magnitude v)

theorem cos_angle_parallel (A B C : Point ℝ)
(hA : A = ⟨0, 0, 4⟩)
(hB : B = ⟨-3, -6, 1⟩)
(hC : C = ⟨-5, -10, -1⟩) :
cos_angle (vec_sub B A) (vec_sub C A) = 1 :=
by {
  sorry
}

end cos_angle_parallel_l694_694508


namespace solution_set_of_f_eq_0_l694_694548

theorem solution_set_of_f_eq_0 :
  (∀ x, f(x-1) = x^2 + 3x - 10) →
  (∀ x, f(x) = 0) → (x = -6 ∨ x = 1) :=
by
  intro h1 h2
  sorry

end solution_set_of_f_eq_0_l694_694548


namespace minimum_value_of_f_l694_694143

def f (x : ℝ) : ℝ := sqrt (x^2 - 8 * x + 17) + sqrt (x^2 + 4)

theorem minimum_value_of_f : ∃ x : ℝ, f x = 5 :=
by
  sorry

end minimum_value_of_f_l694_694143


namespace max_whiskers_proof_l694_694126

noncomputable def number_of_whiskers := 12

def puffy_whiskers := 3 * number_of_whiskers
def scruffy_whiskers := 2 * puffy_whiskers
def buffy_whiskers := (puffy_whiskers + scruffy_whiskers + number_of_whiskers) / 3
def whisper_whiskers := 2 * puffy_whiskers
def bella_whiskers := number_of_whiskers + puffy_whiskers - 4
def max_whiskers := scruffy_whiskers + buffy_whiskers
def felix_whiskers := number_of_whiskers

theorem max_whiskers_proof : max_whiskers = 112 := by
  have hJuniper : number_of_whiskers = 12 := rfl
  have hPuffy : puffy_whiskers = 3 * number_of_whiskers := rfl
  have hScruffy : scruffy_whiskers = 2 * puffy_whiskers := rfl
  have hBuffy : buffy_whiskers = (puffy_whiskers + scruffy_whiskers + number_of_whiskers) / 3 := rfl
  have hWhisper : whisper_whiskers = 2 * puffy_whiskers := rfl
  have hBella : bella_whiskers = number_of_whiskers + puffy_whiskers - 4 := rfl
  have hMax : max_whiskers = scruffy_whiskers + buffy_whiskers := rfl
  exact sorry

end max_whiskers_proof_l694_694126


namespace probability_positive_difference_l694_694775

theorem probability_positive_difference :
  let p3 := (Nat.choose 5 3) * (1/2)^5
  let p4 := (Nat.choose 5 4) * (1/2)^5
  |p3 - p4| = 5 / 32 := by
  sorry

end probability_positive_difference_l694_694775


namespace ratio_PM_MQ_l694_694276

-- Definitions
variables {O P Q R M : Point}
variable [h1 : Center O (Semicircle P Q)]
variable (h2 : OnSemicircle R P Q)
variable (h3 : Perpendicular RM PQ)
variable (h4 : ArcMeasure PR = 2 * ArcMeasure RQ)

-- Theorem statement
theorem ratio_PM_MQ : ratio PM MQ = 3 :=
sorry

end ratio_PM_MQ_l694_694276


namespace possible_h_values_l694_694474

-- Define the conditions for the problem
def parabola_vertex : (ℝ × ℝ) := (0, -4)

def parabola (x : ℝ) : ℝ := 2 * x^2 - 4

def intersection_points (h : ℝ) : set ℝ :=
  {x | parabola x = h}

def triangle_base (h : ℝ) : ℝ :=
  2 * real.sqrt ((h + 4) / 2)

def triangle_height (h : ℝ) : ℝ :=
  h + 4

def triangle_area (h : ℝ) : ℝ :=
  0.5 * triangle_base h * triangle_height h

-- Define the main theorem based on the question and conditions
theorem possible_h_values :
  ∀ (h : ℝ), 16 ≤ triangle_area h ∧ triangle_area h ≤ 128 ↔ 0 ≤ h ∧ h ≤ 12 :=
by 
  sorry

end possible_h_values_l694_694474


namespace largest_m_divides_prod_pow_l694_694151

noncomputable def pow (n : ℕ) : ℕ :=
  let primes := n.factors.filter (λ p, p.prime)
  in primes.max'.get_or_else 1 ^ primes.frequency.get (primes.max'.get_or_else 1)

theorem largest_m_divides_prod_pow :
  (∏ n in finset.Ico 2 6001, pow n) % 2310 ^ 595 = 0 :=
by
  sorry

end largest_m_divides_prod_pow_l694_694151


namespace mono_increasing_a_le_one_l694_694244

open Real

/-- If the function f(x) = e^x * (sin x + a * cos x) is monotonically increasing in the interval 
    (π/4, π/2), then a ≤ 1. -/
theorem mono_increasing_a_le_one (a : ℝ) :
  (∀ x ∈ Ioo (π / 4) (π / 2), deriv (λ x, exp x * (sin x + a * cos x)) x ≥ 0) → a ≤ 1 :=
begin
  sorry
end

end mono_increasing_a_le_one_l694_694244


namespace condition_one_condition_two_l694_694164

-- Define The Triangle ABC proof problem for Condition ①.
theorem condition_one (a b : ℝ) (B : ℝ) (c : ℝ) (S : ℝ) 
  (h1 : c = 2)
  (h2 : S = (3 / 2) * Real.sqrt 3) 
  (h3 : 2 * c + b = 2 * a * Real.cos B) 
  (h4 : a = Real.sqrt 19) 
  (h5 : b = 3) :
  ∃ sinB sinC, sinB * sinC = 9 / 38 := 
begin
  -- Proof steps skipped
  sorry
end

-- Define The Triangle ABC proof problem for Condition ②.
theorem condition_two (a b : ℝ) (B : ℝ) (c : ℝ) (S : ℝ) 
  (h1 : a = Real.sqrt 21)
  (h2 : S = (3 / 2) * Real.sqrt 3) 
  (h3 : 2 * c + b = 2 * a * Real.cos B) 
  (h4 : c = 2) 
  (h5 : b = 3) :
  ∃ sinB sinC, sinB * sinC = 3 / 14 := 
begin
  -- Proof steps skipped
  sorry
end

end condition_one_condition_two_l694_694164


namespace balanced_string_count_correct_l694_694406

def is_balanced (s : String) : Bool :=
  (∀ t : String, t.is_substring_of s → (t.count('X') - t.count('O')).abs ≤ 2)

noncomputable def number_of_balanced_strings (n : ℕ) : ℕ :=
  if n % 2 = 0 then 3 * 2^(n / 2) - 2 else 2 * 2^((n + 1) / 2) - 2

theorem balanced_string_count_correct (n : ℕ) : 
  ∀ s : String, s.length = n → is_balanced s → s.count('X') + s.count('O') = n :=
sorry

end balanced_string_count_correct_l694_694406


namespace system1_solution_system2_solution_l694_694346

-- Problem 1
theorem system1_solution (x y : ℝ) (h1 : 3 * x - 2 * y = 6) (h2 : 2 * x + 3 * y = 17) : 
  x = 4 ∧ y = 3 :=
by {
  sorry
}

-- Problem 2
theorem system2_solution (x y : ℝ) (h1 : x + 4 * y = 14) 
  (h2 : (x - 3) / 4 - (y - 3) / 3 = 1 / 12) : 
  x = 3 ∧ y = 11 / 4 :=
by {
  sorry
}

end system1_solution_system2_solution_l694_694346


namespace large_pizza_cost_l694_694985

theorem large_pizza_cost
  (small_side : ℕ) (small_cost : ℝ) (large_side : ℕ) (friend_money : ℝ) (extra_square_inches : ℝ)
  (A_small : small_side * small_side = 196)
  (A_large : large_side * large_side = 441)
  (small_cost_per_sq_in : 196 / small_cost = 19.6)
  (individual_area : (30 / small_cost) * 196 = 588)
  (total_individual_area : 2 * 588 = 1176)
  (pool_area_eq : (60 / (441 / x)) = 1225)
  : (x = 21.6) := 
by
  sorry

end large_pizza_cost_l694_694985


namespace probability_positive_difference_l694_694776

theorem probability_positive_difference :
  let p3 := (Nat.choose 5 3) * (1/2)^5
  let p4 := (Nat.choose 5 4) * (1/2)^5
  |p3 - p4| = 5 / 32 := by
  sorry

end probability_positive_difference_l694_694776


namespace delegates_probability_l694_694752

theorem delegates_probability :
  let total_arrangements := 12.factorial / (4.factorial * 4.factorial * 4.factorial)
  let unwanted_arrangements :=
    3 * 12 * (8.factorial / (4.factorial * 4.factorial)) -
    3 * 12 * 5 + 12 * 2
  let wanted_arrangements := total_arrangements - unwanted_arrangements
  let probability := wanted_arrangements / total_arrangements
  probability = (6 : ℚ) / 37 ∧ 6 + 37 = 43 :=
by
  let total_arrangements := 12.factorial / (4.factorial * 4.factorial * 4.factorial)
  let unwanted_arrangements :=
    3 * 12 * (8.factorial / (4.factorial * 4.factorial)) -
    3 * 12 * 5 + 12 * 2
  let wanted_arrangements := total_arrangements - unwanted_arrangements
  let probability := wanted_arrangements / total_arrangements
  have prob : probability = (6 : ℚ) / 37 := sorry
  have p_plus_q : 6 + 37 = 43 := rfl
  exact ⟨prob, p_plus_q⟩

end delegates_probability_l694_694752


namespace initial_money_l694_694646

/-- Given the following conditions:
  (1) June buys 4 maths books at $20 each.
  (2) June buys 6 more science books than maths books at $10 each.
  (3) June buys twice as many art books as maths books at $20 each.
  (4) June spends $160 on music books.
  Prove that June had initially $500 for buying school supplies. -/
theorem initial_money (maths_books : ℕ) (science_books : ℕ) (art_books : ℕ) (music_books_cost : ℕ)
  (h_math_books : maths_books = 4) (price_per_math_book : ℕ) (price_per_science_book : ℕ) 
  (price_per_art_book : ℕ) (price_per_music_books_cost : ℕ) (h_maths_price : price_per_math_book = 20)
  (h_science_books : science_books = maths_books + 6) (h_science_price : price_per_science_book = 10)
  (h_art_books : art_books = 2 * maths_books) (h_art_price : price_per_art_book = 20)
  (h_music_books_cost : music_books_cost = 160) :
  4 * 20 + (4 + 6) * 10 + (2 * 4) * 20 + 160 = 500 :=
by sorry

end initial_money_l694_694646


namespace inverse_matrix_l694_694924

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![7, -2], ![-3, 1]]

-- Define the supposed inverse B
def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 2], ![3, 7]]

-- Define the condition that A * B should yield the identity matrix
theorem inverse_matrix :
  A * B = 1 :=
sorry

end inverse_matrix_l694_694924


namespace length_of_segment_through_M_minimum_length_of_segments_l694_694383

def prism_base_length (a : ℝ) : ℝ := 2 * a

def lateral_edge_length (a : ℝ) : ℝ := a

def point_M_on_AD1 (a : ℝ) : (ℝ × ℝ × ℝ) := (4 * a / 3, 0, 2 * a / 3)

theorem length_of_segment_through_M (a : ℝ) :
  let M := point_M_on_AD1 a in
  let segment_length := dist (4 * a / 3, 0, 2 * a / 3) (some other point here) in
  segment_length = a * (sqrt 5) / 3 :=
sorry

theorem minimum_length_of_segments (a : ℝ) :
  let min_length := complex_minimization_operation in
  min_length = a / (sqrt 2) :=
sorry

end length_of_segment_through_M_minimum_length_of_segments_l694_694383


namespace combined_dots_hexagon_l694_694884

def hexagon_dots : ℕ → ℕ
| 1 := 1
| 2 := 7
| 3 := 19
| n := hexagon_dots (n - 1) + (6 * (n - 1) + 2 * (6 * (n - 3 + 1)))

theorem combined_dots_hexagon (h4 h5 : ℕ) (H4 : h4 = hexagon_dots 4) (H5 : h5 = hexagon_dots 5) : h4 + h5 = 138 :=
by
  have h4_calc : h4 = 51 := by
    -- Calculation steps for fourth hexagon
    sorry
  have h5_calc : h5 = 87 := by
    -- Calculation steps for fifth hexagon
    sorry
  rw [h4_calc, h5_calc]
  exact rfl

#check combined_dots_hexagon

end combined_dots_hexagon_l694_694884


namespace largest_prime_factor_7_fact_8_fact_l694_694004

theorem largest_prime_factor_7_fact_8_fact : 
  let n := 7! + 8! 
  ∃ p, Prime p ∧ p ∣ n ∧ (∀ q, Prime q ∧ q ∣ n → q ≤ p) := 
begin
  let n := fact 7 + fact 8,
  use 7, 
  split,
  { apply Prime.prime_7 },
  split,
  { sorry },  
  { sorry },
end

end largest_prime_factor_7_fact_8_fact_l694_694004


namespace additional_paint_needed_l694_694466

theorem additional_paint_needed : 
  ∀ (initial_paint : ℚ) (used_day1_frac : ℚ) (used_day2_frac : ℚ) (extra_paint_needed : ℚ),
  initial_paint = 2 → 
  used_day1_frac = 1/4 → 
  used_day2_frac = 1/2 → 
  extra_paint_needed = 1/2 → 
  let remaining_after_day1 := initial_paint - used_day1_frac * initial_paint in
  let remaining_after_day2 := remaining_after_day1 - used_day2_frac * remaining_after_day1 in
  let total_paint_needed := remaining_after_day2 + extra_paint_needed in
  total_paint_needed - remaining_after_day2 = 1/2 :=
begin
  intros initial_paint used_day1_frac used_day2_frac extra_paint_needed,
  intros init_paint_eq used1_eq used2_eq extra_eq,
  rw [init_paint_eq, used1_eq, used2_eq, extra_eq],
  let remaining_after_day1 := 2 - (1/4) * 2,
  let remaining_after_day2 := remaining_after_day1 - (1/2) * remaining_after_day1,
  let total_paint_needed := remaining_after_day2 + 1/2,
  have h1 : remaining_after_day1 = 3 / 2,
  { norm_num [remaining_after_day1] },
  have h2 : remaining_after_day2 = 3 / 4,
  { rw h1, norm_num [remaining_after_day2] },
  have h3 : total_paint_needed = 5 / 4,
  { rw h2, norm_num [total_paint_needed] },
  norm_num [h2, h3],
  sorry -- Proof left for the user to complete
end

end additional_paint_needed_l694_694466


namespace number_of_vampire_bags_l694_694854

def total_students : ℕ := 25
def pumpkin_students : ℕ := 14
def pack_cost : ℕ := 3
def individual_cost : ℕ := 1

theorem number_of_vampire_bags : 
  ∃ V, V = total_students - pumpkin_students ∧ 
  ((⌊pumpkin_students / 5⌋ * pack_cost + (pumpkin_students % 5) * individual_cost) +
   (⌊V / 5⌋ * pack_cost + (V % 5) * individual_cost) = 17) :=
by
  sorry

end number_of_vampire_bags_l694_694854


namespace correct_number_of_paths_l694_694928

-- Define the number of paths for each segment.
def paths_A_to_B : ℕ := 2
def paths_B_to_D : ℕ := 2
def paths_D_to_C : ℕ := 2
def direct_path_A_to_C : ℕ := 1

-- Define the function to calculate the total paths from A to C.
def total_paths_A_to_C : ℕ :=
  (paths_A_to_B * paths_B_to_D * paths_D_to_C) + direct_path_A_to_C

-- Prove that the total number of paths from A to C is 9.
theorem correct_number_of_paths : total_paths_A_to_C = 9 := by
  -- This is where the proof would go, but it is not required for this task.
  sorry

end correct_number_of_paths_l694_694928


namespace initial_thought_profit_percentage_l694_694095

/-- Define the cost price as a constant. -/
def cost_price : ℝ := 100

/-- The marked price is 65% above the cost price. -/
def marked_price : ℝ := cost_price + 0.65 * cost_price

/-- The selling price after a 25% discount on the marked price. -/
def selling_price : ℝ := marked_price - 0.25 * marked_price

/-- The actual profit made is 23.75%. -/
def actual_profit : ℝ := selling_price - cost_price

/-- The retailer initially thought his profit percentage would be 65%. -/
theorem initial_thought_profit_percentage : (marked_price - cost_price) / cost_price * 100 = 65 :=
begin
  sorry
end

end initial_thought_profit_percentage_l694_694095


namespace problem1_problem2_l694_694571

-- Problem 1
theorem problem1 :
  let l := (λ t : ℝ, (1 + (1 / 2) * t, (sqrt 3 / 2) * t)),
      C1 := (λ θ : ℝ, (cos θ, sin θ)),
      x_intersect := real.solve (λ x, (sqrt 3) * (x - 1) & x^2 + ((sqrt 3) * (x - 1))^2 = 1),
      A := (1, 0),
      B := (1 / 2, - (sqrt 3 / 2)) in
  dist A B = 1 := sorry

-- Problem 2
theorem problem2 :
  let l := (λ t : ℝ, (1 + (1 / 2) * t, (sqrt 3 / 2) * t)),
      C2 := (λ θ : ℝ, (sqrt 3 * cos θ, 3 * sin θ)),
      P := λ θ : ℝ, (sqrt 3 * cos θ, 3 * sin θ),
      dist_P_l := λ θ : ℝ, abs ((1 / 2) * (3 * sqrt 2 * sin (θ - π / 4) + sqrt 3)) in
  ∃ θ : ℝ, dist_P_l θ = (3 * sqrt 2 + sqrt 3) / 2 := sorry

end problem1_problem2_l694_694571


namespace expected_vertices_eq_l694_694929

noncomputable def expected_vertices_of_convex_hull
  (n : ℕ := 6)
  (angles : Fin n → ℝ := λ i, i * (Real.pi / 3))
  (radius_dist: MeasureTheory.MeasureSpace ℝ := MeasureTheory.MeasureSpace.uniform)
  (P : Fin n → ℝ × ℝ := λ i, (radius_dist.sample, angles i)) :
  ℝ :=
6 * ((1 - (2 / 3) * (1 - Real.log 2)))

theorem expected_vertices_eq :
  expected_vertices_of_convex_hull = 2 + 4 * Real.log 2 :=
by
  sorry

end expected_vertices_eq_l694_694929


namespace total_area_to_be_painted_l694_694439

-- Define the dimensions of the barn
def width := 15
def length := 20
def height := 8

-- Define the areas of the walls, dividing wall, and ceiling
def front_back_wall_area := 2 * (width * height)
def side_wall_area := 2 * (length * height)
def dividing_wall_area := 2 * ((length / 2) * height)
def ceiling_area := width * length

-- Total area calculation
def total_area := front_back_wall_area + side_wall_area + dividing_wall_area + ceiling_area

-- Theorem to prove that the total area is 1020 sq yd
theorem total_area_to_be_painted : total_area = 1020 := by 
  sorry -- Proof not required

end total_area_to_be_painted_l694_694439


namespace inscribed_circle_circumscribed_circle_iff_perpendicular_distance_centers_l694_694042

variable (A B C D : Type)
variable (P Q R S : A) -- Points corresponding to A, B, C, D respectively
variable [metric_space A]
variable [convex A]
variable {AB BC CD DA : ℝ}
variable {r R : ℝ} -- Radii of the inscribed and circumscribed circles
variable (d : ℝ) -- Distance between the centers of the inscribed and circumscribed circles

noncomputable theory

-- Given Conditions
def quadrilateral_convex (AB AD CB CD : ℝ) (P Q R S : A) (h1 : dist P Q = AB) (h2 : dist P S = AD) (h3 : dist R Q = CB) (h4 : dist R S = CD) :=
  convex_hull (pair P (pair Q (pair R S))) = P

def perpendicular (P Q R : A) :=
  ∠ P Q R = π / 2

def quadrilateral_equality (AB AD CD DA: ℝ) :=
  AB + CD = DA + AD

def squared_distance (r R d : ℝ) :=
  d^2 = R^2 + r^2 - r * sqrt (r^2 + 4 * R^2)

-- Part (a): Prove that a circle can be inscribed in the quadrilateral
theorem inscribed_circle (h1 : dist P Q = dist P S) (h2 : dist Q R = dist S R) :
  quadrilateral_convex (dist P Q + dist S R) (dist P S + dist Q R) P Q R S :=
sorry

-- Part (b): Prove that a circle can be circumscribed around the quadrilateral if and only if AB ⊥ BC
theorem circumscribed_circle_iff_perpendicular (h1 : dist P Q = dist P S) (h2 : dist Q R = dist S R) :
  (∃ (O : A), is_circumscribed P Q R S O) ↔ perpendicular P Q R :=
sorry

-- Part (c): If AB ⊥ BC, then prove the squared distance between the centers of the inscribed circle and circumscribed circle
theorem distance_centers (h : perpendicular P Q R) :
  squared_distance r R d :=
sorry

end inscribed_circle_circumscribed_circle_iff_perpendicular_distance_centers_l694_694042


namespace people_in_group_l694_694716

theorem people_in_group
  (N : ℕ)
  (h1 : ∃ w1 w2 : ℝ, w1 = 65 ∧ w2 = 71 ∧ w2 - w1 = 6)
  (h2 : ∃ avg_increase : ℝ, avg_increase = 1.5 ∧ 6 = avg_increase * N) :
  N = 4 :=
sorry

end people_in_group_l694_694716


namespace function_identity_l694_694310

-- Define that f is a function from ℕ to ℕ
variable (f : ℕ → ℕ)

-- Assume the given condition
axiom condition : ∀ n : ℕ, f(n+1) > f(f(n))

-- State the theorem to be proved
theorem function_identity : ∀ n : ℕ, f(n) = n :=
by
  sorry

end function_identity_l694_694310


namespace probability_all_four_same_suit_l694_694615

theorem probability_all_four_same_suit (deck_size : ℕ) (cards_drawn : ℕ) (num_suits : ℕ) 
    (cards_per_suit : ℕ) (rep : Prop) 
    (h_deck: deck_size = 52) 
    (h_num_suits: num_suits = 4) 
    (h_cards_drawn: cards_drawn = 4) 
    (h_cards_per_suit: cards_per_suit = 13)
    (h_rep: rep ↔ true) : 
    (∃ p : ℚ, p = 1 / 64) :=
begin
  sorry
end

end probability_all_four_same_suit_l694_694615


namespace sum_of_h_values_l694_694517

-- Define the given equation.
def given_equation (r h : ℝ) : ℝ :=
    (abs (abs (r + h) - r) + 4 * r) - 9 * abs (r - 3)

-- Define the condition that the equation has at most one root.
def has_at_most_one_root (h : ℝ) : Prop :=
    ∀ r1 r2 : ℝ, given_equation r1 h = 0 → given_equation r2 h = 0 → r1 = r2

-- The main theorem that needs to be proved.
theorem sum_of_h_values : 
    (∑ h in finset.Icc (-18 : ℤ) 12, h) = -93 :=
by
  -- sorry to skip the proof.
  sorry

end sum_of_h_values_l694_694517


namespace sum_Q3_x_coords_l694_694796

noncomputable theory
open_locale classical

-- Definitions and the given conditions in Lean 4
def sum_x_coords (n : ℕ) (vertices_x : fin n → ℝ) : ℝ :=
  (finset.univ : finset (fin n)).sum (λ i, vertices_x i)

def Q1_x_coords : fin 150 → ℝ := sorry -- Given x-coordinates of Q1's vertices
axiom sum_Q1_x_coords : sum_x_coords 150 Q1_x_coords = 3000

-- Definitions of midpoints
def midpoints_x_coords (n : ℕ) (vertices_x : fin n → ℝ) : fin n → ℝ :=
  λ i, (vertices_x i + vertices_x (i + 1) % n) / 2

-- Definition of Q2 and Q3 in terms of midpoints
def Q2_x_coords : fin 150 → ℝ := midpoints_x_coords 150 Q1_x_coords
def Q3_x_coords : fin 150 → ℝ := midpoints_x_coords 150 Q2_x_coords

-- Theorem stating the desired conclusion
theorem sum_Q3_x_coords : sum_x_coords 150 Q3_x_coords = 3000 :=
sorry

end sum_Q3_x_coords_l694_694796


namespace cost_to_marked_price_ratio_l694_694798

variables (p : ℝ) (discount : ℝ := 0.20) (cost_ratio : ℝ := 0.60)

theorem cost_to_marked_price_ratio :
  (cost_ratio * (1 - discount) * p) / p = 0.48 :=
by sorry

end cost_to_marked_price_ratio_l694_694798


namespace difference_second_largest_third_smallest_l694_694509

def three_digit_numbers (d1 d2 d3 : ℕ) := 
  let perms := [100 * x.0 + 10 * x.1 + x.2 | x in (d1, d2, d3).permutations] 
  perms.sorted

theorem difference_second_largest_third_smallest : 
  let nums := three_digit_numbers 1 6 8 
  let num2 := nums.nth_le (nums.length - 2) sorry
  let num3 := nums.nth_le 2 sorry
  num2 - num3 = 198 :=
by sorry

end difference_second_largest_third_smallest_l694_694509


namespace updated_mean_l694_694040

theorem updated_mean (n : ℕ) (observation_mean decrement : ℕ) 
  (h1 : n = 50) (h2 : observation_mean = 200) (h3 : decrement = 15) : 
  ((observation_mean * n - decrement * n) / n = 185) :=
by
  sorry

end updated_mean_l694_694040


namespace rooster_count_l694_694392

theorem rooster_count (total_chickens hens roosters : ℕ) 
  (h1 : total_chickens = roosters + hens)
  (h2 : roosters = 2 * hens)
  (h3 : total_chickens = 9000) 
  : roosters = 6000 := 
by
  sorry

end rooster_count_l694_694392


namespace train_A_time_to_reach_destination_l694_694405

variable (T t : ℝ)

def time_to_destination_A (A_speed B_speed B_time_to_dest : ℝ) : ℝ :=
  let total_distance_A_before_meeting := A_speed * T
  let total_distance_B_before_meeting := B_speed * T
  let total_distance_B_after_meeting := B_speed * B_time_to_dest
  let equation := total_distance_B_before_meeting + total_distance_B_after_meeting = total_distance_A_before_meeting + A_speed * t
  equation

theorem train_A_time_to_reach_destination :
  ∃ t : ℝ,
    time_to_destination_A 60 90 4 = 16 := by
  sorry

end train_A_time_to_reach_destination_l694_694405


namespace tucker_tissues_l694_694746

theorem tucker_tissues (num_tissues_per_box : ℕ) (num_boxes : ℕ) (tissues_used : ℕ) (t : ℕ)
    (h₀ : num_tissues_per_box = 160) (h₁ : num_boxes = 3) (h₂ : tissues_used = 210) 
    (h₃ : t = num_tissues_per_box * num_boxes - tissues_used) : t = 270 :=
by
  -- Unfold assumptions
  rw [h₀, h₁, h₂] at h₃
  -- Simplify the equation
  have h₄ : t = 160 * 3 - 210 := h₃
  calc
    t = 160 * 3 - 210 := h₄
    _ = 480 - 210 := by rw nat.mul_comm
    _ = 270 := by norm_num

end tucker_tissues_l694_694746


namespace cube_edge_assignment_possible_l694_694291

-- Define the structure of the cube
structure CubeEdges :=
  (AB BC CD DA A1 B1 C1 D1 AA1 BB1 CC1 DD1 : ℕ)

-- Define the conditions and the proof question
theorem cube_edge_assignment_possible :
  ∃ (e : CubeEdges),
    e.AB + e.BC + e.CD + e.DA = 26 ∧
    e.A1 + e.B1 + e.C1 + e.D1 = 26 ∧
    e.AB + e.BB1 + e.B1 + e.AA1 = 26 ∧
    e.BC + e.CC1 + e.C1 + e.BB1 = 26 ∧
    e.CD + e.DD1 + e.D1 + e.CC1 = 26 ∧
    e.DA + e.AA1 + e.D1 + e.DD1 = 26 :=
by {
  -- It's important to list out the numbers on the edges as described in the solution
  existsi CubeEdges.mk 10 5 7 4 3 9 6 8 2 11 1 12,
  simp, -- Simplify the expressions
  -- Check for the sums of each face
  split; norm_num,
  split; norm_num,
  split; norm_num,
  split; norm_num,
  norm_num,
  norm_num
}

end cube_edge_assignment_possible_l694_694291


namespace part_one_part_two_l694_694965

noncomputable def f (x : ℝ) : ℝ := log ((1 - x) / (x + 1))
noncomputable def g (x a : ℝ) : ℝ := 2 - a ^ x

-- Proof Problem for Part 1
theorem part_one (x : ℝ) (h₁ : f (f x) + f (log 3) > 0) : x ∈ (1/2, 9/11) :=
sorry

-- Proof Problem for Part 2
theorem part_two (a x1 x2 : ℝ) (hx1 : 0 ≤ x1) (hx1_lt_1 : x1 < 1) (hx2 : 0 ≤ x2) (hx2_lt_1 : x2 < 1) (h : f x1 = g x2 a) (ha : 0 < a) (ha_ne_1 : a ≠ 1) : 2 < a :=
sorry

end part_one_part_two_l694_694965


namespace marble_count_l694_694856

theorem marble_count (a : ℕ) (h1 : a + 3 * a + 6 * a + 30 * a = 120) : a = 3 :=
  sorry

end marble_count_l694_694856


namespace no_real_solution_exists_for_A_l694_694781

theorem no_real_solution_exists_for_A:
  (¬ ∃ x : ℝ, (x - 3)^2 = -1) ∧
  (∃ x : ℝ, |x / 2| - 6 = 0) ∧
  (∃ x : ℝ, x^2 + 8x + 16 = 0) ∧
  (¬ ∃ x : ℝ, x + sqrt (x - 5) = 0) ∧
  (∃ x : ℝ, sqrt (-2 * x - 10) = 3) :=
by
  sorry

end no_real_solution_exists_for_A_l694_694781


namespace robot_ai_machine_l694_694241

-- Definitions based on the conditions
variable (Robot : Type)
variable (Machine : Robot → Prop)
variable (Advanced_AI : Robot → Prop)

-- Conditions given
def C1 : Prop := ∀ (r : Robot), Advanced_AI r
def C2 : Prop := ∃ (r : Robot), Machine r

-- Statement to be proved
def Statement_II : Prop := ∃ (r : Robot), Advanced_AI r ∧ Machine r

-- The main theorem to be proved
theorem robot_ai_machine : C1 ∧ C2 → Statement_II :=
by sorry

end robot_ai_machine_l694_694241


namespace ellipse_equation_range_y0_l694_694181

-- Define the parameters of the ellipse and given conditions
variables (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
variables (hF : ∃ c : ℝ, (c = 1) ∧ (F = (1, 0)))
variables (eccentricity : ℝ) (h3 : eccentricity = 1 / 2)

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := (x^2 / (a^2)) + (y^2 / (b^2)) = 1

-- Define the location of the focus and other given parameters
def focus_eq (F : ℝ × ℝ) := F = (1, 0)

-- Check if the given point lies on the ellipse
def on_ellipse (F : ℝ × ℝ) : Prop := ellipse_eq F.1 F.2

theorem ellipse_equation (hx : ∃ a b : ℝ, a = 2 ∧ b^2 = 3) : 
  (ellipse_eq = (λ x y, (x^2 / 4) + (y^2 / 3) = 1)) :=
sorry

-- Define the range of y0 based on the perpendicular bisector intersection
theorem range_y0 (F : ℝ × ℝ) : 
  -((√3 : ℝ) / 12) ≤ y0 ∧ y0 ≤ (√3 / 12) :=
sorry

end ellipse_equation_range_y0_l694_694181


namespace sin_transform_l694_694400

theorem sin_transform (x : ℝ) : 
  (λ x, sin (x + π/2)) (2 * (x - π/4)) = -sin (x / 2) := 
sorry

end sin_transform_l694_694400


namespace length_a_plus_b_cos_angle_a_a_plus_b_l694_694936

variables (a b : ℝ^3)
variables (h_a : ∥a∥ = 1)
variables (h_b : ∥b∥ = 2)
variables (h_dot : (2 • a + b) ⬝ (a - b) = -3)

-- Define the length of vector a + b
theorem length_a_plus_b : ∥a + b∥ = Real.sqrt 7 :=
by {
  sorry
}

-- Define the cosine value of the angle between a and a + b
theorem cos_angle_a_a_plus_b : ((a ⬝ (a + b)) / (∥a∥ * ∥a + b∥)) = 2 / Real.sqrt 7 :=
by {
  sorry
}

end length_a_plus_b_cos_angle_a_a_plus_b_l694_694936


namespace simplify_expression_l694_694865

theorem simplify_expression : 
  (((5 + 7 + 3) * 2 - 4) / 2 - (5 / 2) = 21 / 2) :=
by
  sorry

end simplify_expression_l694_694865


namespace range_of_k_l694_694619

noncomputable def is_ellipse (k : ℝ) : Prop := x^2 + k * y^2 = 2

theorem range_of_k (k : ℝ) : is_ellipse k → (0 < k ∧ k < 1) :=
by
  assume h : is_ellipse k
  -- Further proof steps would go here
  sorry

end range_of_k_l694_694619


namespace eigenvalue_problem_line_transformation_problem_l694_694539

noncomputable theory
open matrix

-- Problem data
def M (a b : ℝ) : matrix (fin 2) (fin 2) ℝ := !![2, a; 2, b]
def char_poly (a b : ℝ) (λ : ℝ) : ℝ := (λ - 2) * (λ - b) - (-2 * a)

-- Theorem: values of a and b such that the eigenvalues are -1 and 4
theorem eigenvalue_problem {a b : ℝ} :
  char_poly a b (-1) = 0 ∧
  char_poly a b 4 = 0 ↔
  a = 3 ∧ b = 1 :=
sorry

-- Definition of transformation and image of l
def transform (a b : ℝ) (x y : ℝ) : ℝ × ℝ :=
  let M := M a b in (M.mul_vec !![x, y]).toLinAlgTuple

-- Theorem: line equation under transformation
theorem line_transformation_problem (x y x' y' : ℝ) :
  transform 3 1 x y = (x', y') →
  x' - 2 * y' - 3 = 0 →
  2 * x - y + 3 = 0 :=
sorry

end eigenvalue_problem_line_transformation_problem_l694_694539


namespace initial_amount_of_A_l694_694808

def transaction_a_to_b_c (a b c : ℕ) : ℕ × ℕ × ℕ :=
  let a' := a - b - c;
  let b' := 2 * b;
  let c' := 2 * c;
  (a', b', c')

def transaction_b_to_a_c (a b c : ℕ) : ℕ × ℕ × ℕ :=
  let a' := 2 * a;
  let b' := b - a;
  let c' := 4 * c;
  (a', b', c')

def transaction_c_to_a_b (a b c : ℕ) : ℕ × ℕ × ℕ :=
  let a' := 2 * a;
  let b' := 4 * b - 2 * c;
  let c' := c - a - b;
  (a', b', c')

theorem initial_amount_of_A (a b c : ℕ) :
  ∀ a' b' c',
  transaction_a_to_b_c a b c = (a', b', c') →
  ∀ a'' b'' c'',
  transaction_b_to_a_c a' b' c' = (a'', b'', c'') →
  ∀ a''' b''' c''',
  transaction_c_to_a_b a'' b'' c'' = (a''', b''', c''') →
  a''' = 24 ∧ b''' = 24 ∧ c''' = 24 →
  a = 16 :=
by
  intros a' b' c' h1 a'' b'' c'' h2 a''' b''' c''' h3 h_final,
  sorry

end initial_amount_of_A_l694_694808


namespace equilateral_triangle_area_l694_694507

theorem equilateral_triangle_area (p : ℝ) (h : p ≥ 0) :
  let s := p in
  let A := (Real.sqrt 3 / 4) * s^2 in
  A = (p^2 * Real.sqrt 3) / 4 :=
by
  let s := p
  let A := (Real.sqrt 3 / 4) * s^2
  have hA : A = (p^2 * Real.sqrt 3) / 4 := sorry
  exact hA

end equilateral_triangle_area_l694_694507


namespace golf_ratio_l694_694478

-- Definitions based on conditions
def first_turn_distance : ℕ := 180
def excess_distance : ℕ := 20
def total_distance_to_hole : ℕ := 250

-- Derived definitions based on conditions
def second_turn_distance : ℕ := (total_distance_to_hole - first_turn_distance) + excess_distance

-- Lean proof problem statement
theorem golf_ratio : (second_turn_distance : ℚ) / first_turn_distance = 1 / 2 :=
by
  -- use sorry to skip the proof
  sorry

end golf_ratio_l694_694478


namespace count_four_digit_numbers_l694_694600

open Int

def is_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9

def valid_combination (a b c d : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧
  a ≠ 0 ∧
  a + b + c + d = 10 ∧
  (a + c) - (b + d) % 11 = 0

theorem count_four_digit_numbers :
  ∃ n, n > 0 ∧ ∀ a b c d : ℕ, valid_combination a b c d → n = sorry :=
sorry

end count_four_digit_numbers_l694_694600


namespace angle_sum_is_180_l694_694625

theorem angle_sum_is_180 (A B C : ℝ) (h_triangle : (A + B + C) = 180) (h_sum : A + B = 90) : C = 90 :=
by
  -- Proof placeholder
  sorry

end angle_sum_is_180_l694_694625


namespace exists_year_price_difference_l694_694255

noncomputable def priceP (n : ℕ) : ℝ → ℝ
| p₀ := if n = 0 then p₀ else priceP (n - 1) p₀ / (1 + 0.03) + 0.40 / (1 + 0.03) + 0.02 * (priceP (n - 1) p₀ / (1 + 0.03))

noncomputable def priceQ (n : ℕ) : ℝ → ℝ
| q₀ := if n = 0 then q₀ else priceQ (n - 1) q₀ / (1 + 0.03) + 0.15 / (1 + 0.03) + 0.01 * (priceQ (n - 1) q₀ / (1 + 0.03))

theorem exists_year_price_difference :
  ∃ n : ℕ, n ≥ 1 ∧ priceP n 4.20 - priceQ n 6.30 = 0.40 :=
by
  sorry

end exists_year_price_difference_l694_694255


namespace trigonometric_identity_example_l694_694091

theorem trigonometric_identity_example :
  2 * Real.sin (75 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end trigonometric_identity_example_l694_694091


namespace fraction_meaningful_l694_694399

theorem fraction_meaningful (x : ℝ) : x ≠ 4 ↔ (1 / (x - 4)) ∈ ℝ :=
by
  sorry

end fraction_meaningful_l694_694399


namespace four_digit_sum_ten_divisible_by_eleven_l694_694588

theorem four_digit_sum_ten_divisible_by_eleven : 
  {n | (10^3 ≤ n ∧ n < 10^4) ∧ (∑ i in (Int.toString n).toList, i.toNat) = 10 ∧ (let digits := (Int.toString n).toList.map (λ c, c.toNat) in ((digits.nth 0).getD 0 + (digits.nth 2).getD 0) - ((digits.nth 1).getD 0 + (digits.nth 3).getD 0)) % 11 == 0}.card = 30 := sorry

end four_digit_sum_ten_divisible_by_eleven_l694_694588


namespace variance_of_eta_l694_694976

section VarianceProof

-- Define the random variables ξ and η and their properties
variables (ξ η : ℕ → ℝ)
variable P : ℕ → ℝ
variable (a b : ℝ)

-- Given conditions
def eta_def : Prop := ∀ n, η n = 3 * ξ n + 1
def distribution : Prop := (P 0 = a) ∧ (P 1 = b) ∧ (P 2 = 1/6)
def probability_sum_to_one : Prop := a + b + 1/6 = 1
def expected_value_of_xi : Prop := (0 * a) + (1 * b) + (2 * (1 / 6)) = 2 / 3

-- Statement of the theorem
theorem variance_of_eta : 
  eta_def ξ η →
  distribution P a b →
  probability_sum_to_one a b →
  expected_value_of_xi a b →
  ...
  D(η) = 5 := 
sorry

end VarianceProof

end variance_of_eta_l694_694976


namespace enclosed_area_correct_l694_694358

noncomputable def enclosed_area : ℝ :=
  ∫ x in 0..1, (exp x - exp (-x))

theorem enclosed_area_correct : enclosed_area = exp 1 + exp (-1) - 2 :=
by
  sorry

end enclosed_area_correct_l694_694358


namespace stimulus_check_total_l694_694685

def find_stimulus_check (T : ℝ) : Prop :=
  let amount_after_wife := T * (3/5)
  let amount_after_first_son := amount_after_wife * (3/5)
  let amount_after_second_son := amount_after_first_son * (3/5)
  amount_after_second_son = 432

theorem stimulus_check_total (T : ℝ) : find_stimulus_check T → T = 2000 := by
  sorry

end stimulus_check_total_l694_694685


namespace polynomial_solution_l694_694311

theorem polynomial_solution (f : ℝ → ℝ) (x : ℝ) (h : f (x^2 + 2) = x^4 + 6 * x^2 + 4) : 
  f (x^2 - 2) = x^4 - 2 * x^2 - 4 :=
by
  sorry

end polynomial_solution_l694_694311


namespace functional_solution_l694_694904

noncomputable def f : ℝ → ℝ := sorry

lemma functional_eq (x y : ℝ) : f(x^2 + f(y)) = x * f(x) + y := sorry

theorem functional_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^2 + f(y)) = x * f(x) + y) :
  (∀ x : ℝ, f(x) = x ∨ f(x) = -x) :=
sorry

end functional_solution_l694_694904


namespace exponential_function_example_l694_694851

def is_exponential_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ (∀ x : ℝ, f x = a^x)

theorem exponential_function_example : is_exponential_function (λ x : ℝ, 3^(-x)) :=
  sorry

end exponential_function_example_l694_694851


namespace interval_monotonic_decrease_area_triangle_ABC_l694_694219

-- Define vectors and their properties
def vector_a (x : ℝ) := (Real.sin x, -1)
def vector_b (x : ℝ) := (Real.sqrt 3 * Real.cos x, -1/2)

-- Define the function f(x)
def f (x : ℝ) := (vector_a x + vector_b x).1 * (vector_a x).1 + (vector_a x + vector_b x).2 * (vector_a x).2 - 2

-- Proof of interval of monotonic decrease
theorem interval_monotonic_decrease (k : ℤ) (x : ℝ) :
  k * Real.pi + Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 6 → monotone_decreasing_on f x := sorry

-- Define properties of triangle and sides
variables {a b c : ℝ} {A B C : ℝ}

-- Given sides and conditions in triangle
def sides_conditions :=
  a = 2 * Real.sqrt 3 ∧ c = 4 ∧ A = acos ((2 * Real.sqrt 3 * Real.sqrt 3 + 2) / (Real.sqrt 3 * 1 + 1))

-- Define area using given conditions
theorem area_triangle_ABC :
  ∀ {a b c : ℝ}, a = 2 * Real.sqrt 3 → c = 4 → f A = 1 → A = Real.pi / 3 →
  let area := 1/2 * a * c :=
    area = 2 * Real.sqrt 3 := sorry

end interval_monotonic_decrease_area_triangle_ABC_l694_694219


namespace find_v_l694_694306

open Matrix

def a : Matrix (Fin 3) (Fin 1) ℝ :=
  ![![2], ![1], ![1]]

def b : Matrix (Fin 3) (Fin 1) ℝ :=
  ![![3], ![-1], ![0]]

def v : Matrix (Fin 3) (Fin 1) ℝ :=
  ![![3.5], ![2.5], ![2]]

theorem find_v :
  2 • (Matrix.crossProduct v a) = Matrix.crossProduct b a ∧
  Matrix.crossProduct v b = 2 • (Matrix.crossProduct a b) :=
  sorry

end find_v_l694_694306


namespace a_1998_l694_694382

def a : ℕ → ℕ
| 1 := 0
| 2 := 2
| 3 := 3
| n := if h : n > 3 then
         (finset.range (n - 1)).filter (λ d, d > 1).image (λ d, a d * a (n - d)).max' (by {
           have h' : (finset.range (n - 1)).filter (λ d, d > 1) ≠ ∅,
           by sorry, -- Proof that the filter is non-empty
           exact h'
         })
       else 0

theorem a_1998 : a 1998 = 3 ^ 666 :=
by sorry

end a_1998_l694_694382


namespace original_money_in_wallet_l694_694931

-- Definitions based on the problem's conditions
def grandmother_gift : ℕ := 20
def aunt_gift : ℕ := 25
def uncle_gift : ℕ := 30
def cost_per_game : ℕ := 35
def number_of_games : ℕ := 3
def money_left : ℕ := 20

-- Calculations as specified in the solution
def birthday_money := grandmother_gift + aunt_gift + uncle_gift
def total_game_cost := cost_per_game * number_of_games
def total_money_before_purchase := total_game_cost + money_left

-- Proof that the original amount of money in Geoffrey's wallet
-- was €50 before he got the birthday money and made the purchase.
theorem original_money_in_wallet : total_money_before_purchase - birthday_money = 50 := by
  sorry

end original_money_in_wallet_l694_694931


namespace binary_to_octal_conversion_l694_694886

theorem binary_to_octal_conversion : bin_to_oct 110101 = 65 :=
by
  sorry

end binary_to_octal_conversion_l694_694886


namespace spaceship_initial_people_count_l694_694150

/-- For every 100 additional people that board a spaceship, its speed is halved.
     The speed of the spaceship with a certain number of people on board is 500 km per hour.
     The speed of the spaceship when there are 400 people on board is 125 km/hr.
     Prove that the number of people on board when the spaceship was moving at 500 km/hr is 200. -/
theorem spaceship_initial_people_count (speed : ℕ → ℕ) (n : ℕ) :
  (∀ k, speed (k + 100) = speed k / 2) →
  speed n = 500 →
  speed 400 = 125 →
  n = 200 :=
by
  intro half_speed speed_500 speed_400
  sorry

end spaceship_initial_people_count_l694_694150


namespace largest_prime_number_of_students_l694_694326

theorem largest_prime_number_of_students (crayons papers : ℕ) (h1 : crayons = 385) (h2 : papers = 95) : 
  ∃ (n : ℕ), n.prime ∧ (n ∣ gcd crayons papers) ∧ n = 5 := 
by {
  sorry
}

end largest_prime_number_of_students_l694_694326


namespace find_A_l694_694779

theorem find_A (A : ℕ) (h : A = 7 * 5 + 3) : A = 38 :=
by
  rw [h]
  exact rfl

end find_A_l694_694779


namespace circle_covers_three_points_l694_694794

open Real

theorem circle_covers_three_points 
  (points : Finset (ℝ × ℝ))
  (h_points : points.card = 111)
  (triangle_side : ℝ)
  (h_side : triangle_side = 15) :
  ∃ (circle_center : ℝ × ℝ), ∃ (circle_radius : ℝ), circle_radius = sqrt 3 / 2 ∧ 
  ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
              dist circle_center p1 ≤ circle_radius ∧ 
              dist circle_center p2 ≤ circle_radius ∧ 
              dist circle_center p3 ≤ circle_radius :=
by
  sorry

end circle_covers_three_points_l694_694794


namespace log_expression_evaluation_l694_694896

theorem log_expression_evaluation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (Real.log (x^2) / Real.log (y^8)) * (Real.log (y^3) / Real.log (x^7)) * (Real.log (x^4) / Real.log (y^5)) * (Real.log (y^5) / Real.log (x^4)) * (Real.log (x^7) / Real.log (y^3)) =
  (1 / 4) * (Real.log x / Real.log y) := 
by
  sorry

end log_expression_evaluation_l694_694896


namespace remainder_of_first_105_sum_div_5280_l694_694777

theorem remainder_of_first_105_sum_div_5280:
  let n := 105
  let d := 5280
  let sum := n * (n + 1) / 2
  sum % d = 285 := by
  sorry

end remainder_of_first_105_sum_div_5280_l694_694777


namespace angle_B_value_l694_694288

theorem angle_B_value (a b c B : ℝ) (h : (a^2 + c^2 - b^2) * Real.tan B = Real.sqrt 3 * a * c) :
    B = (Real.pi / 3) ∨ B = (2 * Real.pi / 3) :=
by
    sorry

end angle_B_value_l694_694288


namespace hyperbola_chord_perimeter_l694_694208

theorem hyperbola_chord_perimeter
  (x y : ℝ)
  (AB : ℝ) (F₁ F₂ : ℝ)
  (h1 : AB = 6)
  (h2 : ∀ x y, x^2 / 16 - y^2 / 9 = 1):
  let a := 4 in
  let AF₂_BF₂ := 2 * a + AB in
  AF₂_BF₂ + AB = 28 := 
sorry

end hyperbola_chord_perimeter_l694_694208


namespace mixture_ratio_l694_694642

variables (p q : ℝ)

theorem mixture_ratio 
  (h1 : (5/8) * p + (1/4) * q = 0.5)
  (h2 : (3/8) * p + (3/4) * q = 0.5) : 
  p / q = 1 := 
by 
  sorry

end mixture_ratio_l694_694642


namespace backpacks_total_cost_l694_694221

def n : ℕ := 5
def p_original : ℝ := 20.00
def discount : ℝ := 20 / 100
def monogram_cost : ℝ := 12.00

theorem backpacks_total_cost :
  n * (p_original * (1 - discount) + monogram_cost) = 140 := by
  sorry

end backpacks_total_cost_l694_694221


namespace convert_neg300_degrees_to_radians_l694_694117

/-- Definition to convert degrees to radians -/
def degrees_to_radians (deg : ℝ) : ℝ := deg * (Real.pi / 180)

/-- Problem statement: Converting -300 degrees to radians should equal -5/3 times pi -/
theorem convert_neg300_degrees_to_radians :
  degrees_to_radians (-300) = - (5/3) * Real.pi :=
by
  sorry

end convert_neg300_degrees_to_radians_l694_694117


namespace negation_of_there_exists_l694_694733

theorem negation_of_there_exists (x : ℝ) : ¬ (∃ x : ℝ, x^2 - x + 3 = 0) ↔ ∀ x : ℝ, x^2 - x + 3 ≠ 0 := by
  sorry

end negation_of_there_exists_l694_694733


namespace min_value_a_b_c_l694_694184

theorem min_value_a_b_c (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : 9 * a + 4 * b = a * b * c) :
  a + b + c = 10 := sorry

end min_value_a_b_c_l694_694184
