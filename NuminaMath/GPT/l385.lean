import Mathlib
import Mathlib.Algebra.EuclideanDomain
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.Field
import Mathlib.Analysis
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.Gcd
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Dynamics.OrdinaryDiffEq
import Mathlib.Geometry
import Mathlib.Integration
import Mathlib.NumberTheory.Prime
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Simp
import Mathlib.Topology.Basic

namespace cos_210_eq_neg_sqrt_3_div_2_l385_385831

theorem cos_210_eq_neg_sqrt_3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by
  -- leave proof as sorry
  sorry

end cos_210_eq_neg_sqrt_3_div_2_l385_385831


namespace invalid_external_diagonals_l385_385378

def external_diagonals (a b c : ℝ) : set ℝ :=
  {sqrt (a^2 + b^2), sqrt (b^2 + c^2), sqrt (a^2 + c^2)}

theorem invalid_external_diagonals :
  ¬(∃ a b c : ℝ, external_diagonals a b c = {3, 4, 6} ∨ external_diagonals a b c = {4, 5, 8} ∨ external_diagonals a b c = {6, 6, 9}) :=
by
  sorry

end invalid_external_diagonals_l385_385378


namespace correct_statements_are_ACD_l385_385728

theorem correct_statements_are_ACD :
  (∀ (width : ℝ), narrower_width_implies_better_fit width) ∧
  (rA rB : ℝ) (h_rA : rA = 0.97) (h_rB : rB = -0.99)
    → ¬ (stronger_A_implies_correlation rA rB) ∧
  (∀ (R2 : ℝ), smaller_R2_implies_worse_fit R2) ∧
  (num_products : ℕ) (defective_products : ℕ) (selected_products : ℕ)
    (h_num : num_products = 10) (h_def : defective_products = 3) (h_sel : selected_products = 2),
    probability_exactly_one_defective num_products defective_products selected_products = 7 / 15 :=
by
  sorry  -- Proof is not required

end correct_statements_are_ACD_l385_385728


namespace stratified_sampling_second_class_l385_385041

theorem stratified_sampling_second_class (total_products : ℕ) (first_class : ℕ) (second_class : ℕ) (third_class : ℕ) (sample_size : ℕ) (h_total : total_products = 200) (h_first : first_class = 40) (h_second : second_class = 60) (h_third : third_class = 100) (h_sample : sample_size = 40) :
  (second_class * sample_size) / total_products = 12 :=
by
  sorry

end stratified_sampling_second_class_l385_385041


namespace cos_210_eq_neg_sqrt3_div2_l385_385840

theorem cos_210_eq_neg_sqrt3_div2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div2_l385_385840


namespace product_of_distances_l385_385946

structure Ellipse :=
(a b : ℝ)
(focus_dist : ℝ)

structure PointOnEllipse :=
(x y : ℝ)
(ellipse : Ellipse)
(on_ellipse : (x ^ 2) / (ellipse.a ^ 2) + (y ^ 2) / (ellipse.b ^ 2) = 1)

structure Foci :=
(P F1 F2: PointOnEllipse)
(dot_product: (P.x - F1.x) * (P.y - F1.y) + (P.y - F2.y) * (P.x - F2.x) = 9)

noncomputable def find_product_of_distances (P F1 F2 : PointOnEllipse) [cond : Foci] : ℝ :=
  sorry

theorem product_of_distances {P F1 F2 : PointOnEllipse} [h : Foci P F1 F2] :
  find_product_of_distances P F1 F2 = 15 :=
  sorry

end product_of_distances_l385_385946


namespace count_positive_area_triangles_l385_385983

/-
 Problem Statement:
 How many triangles with positive area have all their vertices at points (i,j) on the coordinate plane,
 where i and j are integers between 1 and 6?
 -/

theorem count_positive_area_triangles : 
    let points := {p : ℤ × ℤ | 1 ≤ p.1 ∧ p.1 ≤ 6 ∧ 1 ≤ p.2 ∧ p.2 ≤ 6}
    let count_triangles_with_positive_area := 
      ∑ p1 in points, ∑ p2 in points, ∑ p3 in points,
      if p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p1 ∧ ¬collinear p1 p2 p3 then 1 else 0
    in count_triangles_with_positive_area = 6796 :=
begin
    sorry
end

/--
 Helper definition to check collinearity
 --/
def collinear (p1 p2 p3 : ℤ × ℤ) : Prop :=
    (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

end count_positive_area_triangles_l385_385983


namespace combined_weight_of_emma_and_henry_l385_385489

variables (e f g h : ℕ)

theorem combined_weight_of_emma_and_henry 
  (h1 : e + f = 310)
  (h2 : f + g = 265)
  (h3 : g + h = 280) : e + h = 325 :=
by
  sorry

end combined_weight_of_emma_and_henry_l385_385489


namespace brocard_point_congruence_max_eight_points_l385_385390

-- Problem (a): Prove that triangle ABC is congruent to triangle B1C1A1.
theorem brocard_point_congruence
  (ABC : Triangle)
  (P : Point)
  (hP : P = brocard_point ABC)
  (A1 B1 C1 : Point)
  (hA1 : line_through P (ABC.A) ∩ circumcircle ABC = A1)
  (hB1 : line_through P (ABC.B) ∩ circumcircle ABC = B1)
  (hC1 : line_through P (ABC.C) ∩ circumcircle ABC = C1) :
  Triangle_congruent ABC (Triangle.mk B1 C1 A1) :=
sorry

-- Problem (b): Prove that no more than eight different points P can make the intersection triangles equal to triangle ABC.
theorem max_eight_points
  (ABC : Triangle)
  (S : Circle)
  (h_inscribed : inscribed ABC S)
  (A1 B1 C1 : Point)
  (h_distinct : A1 ≠ ABC.A ∧ B1 ≠ ABC.B ∧ C1 ≠ ABC.C) :
  ∃ up_to_eight_Ps : Fin 8 (P : Point), same_triangle_by_intersection ABC S P A1 B1 C1 :=
sorry

end brocard_point_congruence_max_eight_points_l385_385390


namespace max_length_of_third_side_of_triangle_l385_385660

noncomputable def max_third_side_length (D E F : ℝ) (a b : ℝ) : ℝ :=
  let c_square := a^2 + b^2 - 2 * a * b * Real.cos (90 * Real.pi / 180)
  Real.sqrt c_square

theorem max_length_of_third_side_of_triangle (D E F : ℝ) (a b : ℝ) (h₁ : Real.cos (2 * D) + Real.cos (2 * E) + Real.cos (2 * F) = 1)
    (h₂ : a = 8) (h₃ : b = 15) : 
    max_third_side_length D E F a b = 17 := 
by
  sorry

end max_length_of_third_side_of_triangle_l385_385660


namespace minimum_distance_sum_squared_l385_385931

variable (P : ℝ × ℝ)
variable (F₁ F₂ : ℝ × ℝ)

def on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x^2 / 4) + y^2 = 1

def distance_squared (A B : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2

theorem minimum_distance_sum_squared
  (hP : on_ellipse P)
  (hF1 : F₁ = (2, 0) ∨ F₁ = (-2, 0)) -- Assuming standard position of foci
  (hF2 : F₂ = (2, 0) ∨ F₂ = (-2, 0)) :
  ∃ P : ℝ × ℝ, on_ellipse P ∧ F₁ ≠ F₂ → distance_squared P F₁ + distance_squared P F₂ = 8 :=
by
  sorry

end minimum_distance_sum_squared_l385_385931


namespace largest_period_polynomial_2020_l385_385118

theorem largest_period_polynomial_2020 :
  ∃ (N : ℕ), N ∈ { n : ℕ | 1 ≤ n ∧ n ≤ 2019 } ∧ 
  (∀ (P : ℕ → ℕ) (h : ∀ k, P k = P (P k) mod 2020),
    (∀ k, k % N = 0 ↔ P 0 % 2020 = 0)) ∧ 
  N = 1980 := 
begin
  sorry,
end

end largest_period_polynomial_2020_l385_385118


namespace cosine_210_l385_385842

noncomputable def Q := rotate 210 (1, 0)
noncomputable def E := foot Q (1, 0) (-1, 0)

theorem cosine_210 : cos 210 = - (sqrt 3/2) :=
sorry

end cosine_210_l385_385842


namespace letters_typed_by_additional_typists_l385_385215

def typists := 20
def letters := 42
def minutes := 20
def additional_typists := 30
def additional_minutes := 60

theorem letters_typed_by_additional_typists : 
  (let letters_per_typist_per_20_minutes := letters / typists in
   let letters_per_20_minutes := additional_typists * letters_per_typist_per_20_minutes in
   let total_letters_in_60_minutes := (additional_minutes / minutes) * letters_per_20_minutes in
   total_letters_in_60_minutes) = 189 :=
by
  sorry

end letters_typed_by_additional_typists_l385_385215


namespace cone_lateral_area_and_sector_area_l385_385405

theorem cone_lateral_area_and_sector_area 
  (slant_height : ℝ) 
  (height : ℝ) 
  (r : ℝ) 
  (h_slant : slant_height = 1) 
  (h_height : height = 0.8) 
  (h_r : r = Real.sqrt (slant_height^2 - height^2)) :
  (1 / 2 * 2 * Real.pi * r * slant_height = 3 / 5 * Real.pi) ∧
  (1 / 2 * 2 * Real.pi * r * slant_height = 3 / 5 * Real.pi) :=
by
  sorry

end cone_lateral_area_and_sector_area_l385_385405


namespace regular_hexagon_area_l385_385646

open Real

-- Auxiliary functions and definitions
def distance (p q : ℝ × ℝ) : ℝ := sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

def hexagon_area (A C : ℝ × ℝ) : ℝ :=
  let s := distance A C / 2
  let triangle_area := (sqrt 3 / 4) * s^2
  2 * triangle_area

-- Conditions
def A : ℝ × ℝ := (0,0)
def C : ℝ × ℝ := (6,2)

-- Theorem statement
theorem regular_hexagon_area : hexagon_area A C = 20 * sqrt 3 := by
  sorry

end regular_hexagon_area_l385_385646


namespace number_and_sum_of_factors_72_l385_385400

theorem number_and_sum_of_factors_72 : 
  let n := 72 in
  let p_factors := (2, 3) in
  let e_factors := (3, 2) in
  let num_factors := (e_factors.1 + 1) * (e_factors.2 + 1) in
  let sum_factors := ((2^(3 + 1) - 1) / (2 - 1)) * ((3^(2 + 1) - 1) / (3 - 1)) in
  num_factors = 12 ∧ sum_factors = 195 := by
    sorry

end number_and_sum_of_factors_72_l385_385400


namespace find_f_x_minus_1_l385_385132

theorem find_f_x_minus_1 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = x ^ 2 + 2 * x) :
  ∀ x : ℤ, f (x - 1) = x ^ 2 - 2 * x :=
by
  sorry

end find_f_x_minus_1_l385_385132


namespace at_least_one_nonnegative_l385_385144

theorem at_least_one_nonnegative (a b c d e f g h : ℝ) :
  ac + bd ≥ 0 ∨ ae + bf ≥ 0 ∨ ag + bh ≥ 0 ∨ ce + df ≥ 0 ∨ cg + dh ≥ 0 ∨ eg + fh ≥ 0 :=
sorry

end at_least_one_nonnegative_l385_385144


namespace johns_consumption_reduction_l385_385579

-- Define the initial conditions
def initial_petrol_price (P : ℝ) : ℝ := P
def price_increase (P : ℝ) : ℝ := 1.3 * P
def miles_per_week : ℝ := 100
def fuel_efficiency : ℝ := 20

-- Initial fuel consumption per week in gallons
def initial_consumption (miles_per_week : ℝ) (fuel_efficiency : ℝ) : ℝ :=
  miles_per_week / fuel_efficiency

-- Initial weekly expenditure in terms of P
def initial_expenditure (initial_consumption : ℝ) (initial_petrol_price : ℝ) : ℝ :=
  initial_consumption * initial_petrol_price

-- New weekly fuel consumption in gallons to keep expenditure constant
def new_consumption (initial_expenditure : ℝ) (price_increase : ℝ) : ℝ :=
  initial_expenditure / price_increase

-- Required reduction in consumption to maintain the same expenditure
def reduction_in_consumption (initial_consumption : ℝ) (new_consumption : ℝ) : ℝ :=
  initial_consumption - new_consumption

-- The main theorem to be proven
theorem johns_consumption_reduction :
  ∀ (P : ℝ),
    reduction_in_consumption 
    (initial_consumption miles_per_week fuel_efficiency)
    (new_consumption 
      (initial_expenditure 
        (initial_consumption miles_per_week fuel_efficiency) 
        (initial_petrol_price P)
      ) 
      (price_increase P)
    ) ≈ 1.154 :=
by sorry

end johns_consumption_reduction_l385_385579


namespace cos_210_eq_neg_sqrt3_div_2_l385_385874

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385874


namespace cos_210_eq_neg_sqrt3_div_2_l385_385855

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385855


namespace change_received_correct_l385_385629

-- Define the prices of items and the amount paid
def price_hamburger : ℕ := 4
def price_onion_rings : ℕ := 2
def price_smoothie : ℕ := 3
def amount_paid : ℕ := 20

-- Define the total cost and the change received
def total_cost : ℕ := price_hamburger + price_onion_rings + price_smoothie
def change_received : ℕ := amount_paid - total_cost

-- Theorem stating the change received
theorem change_received_correct : change_received = 11 := by
  sorry

end change_received_correct_l385_385629


namespace hostel_provisions_l385_385413

theorem hostel_provisions (x : ℕ) (h1 : 250 * x = 200 * 40) : x = 32 :=
by
  sorry

end hostel_provisions_l385_385413


namespace wrongly_entered_mark_l385_385422

theorem wrongly_entered_mark (x : ℕ) 
    (h1 : x - 33 = 52) : x = 85 :=
by
  sorry

end wrongly_entered_mark_l385_385422


namespace zero_points_of_y_l385_385546

def f (a : ℝ) (x : ℝ) : ℝ :=
if x > 0 then log 2 x else a * x + 1

noncomputable def y (a : ℝ) (x : ℝ) : ℝ :=
f a (f a x) - 1

theorem zero_points_of_y (a : ℝ) (h : a > 0) :
  ∃ x1 x2 x3 : ℝ, y a x1 = 0 ∧ y a x2 = 0 ∧ y a x3 = 0 := sorry

end zero_points_of_y_l385_385546


namespace vector_subtraction_l385_385972

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

-- Define the expression to be proven
def expression : ℝ × ℝ := ((2 * a.1 - b.1), (2 * a.2 - b.2))

-- The theorem statement
theorem vector_subtraction : expression = (5, 7) :=
by
  -- The proof will be provided here
  sorry

end vector_subtraction_l385_385972


namespace cos_210_eq_neg_sqrt3_div_2_l385_385865

theorem cos_210_eq_neg_sqrt3_div_2 : real.cos (210 * real.pi / 180) = - (real.sqrt 3 / 2) :=
by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385865


namespace find_constants_l385_385902

theorem find_constants (P Q R : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x^2 + 2 ≠ 0 → 
  \frac{-2*x^2 + 5*x - 6}{x^3 + 2*x} = 
  \frac{P}{x} + \frac{Q*x + R}{x^2 + 2}) → 
  P = -3 ∧ Q = 1 ∧ R = 5 :=
by sorry

end find_constants_l385_385902


namespace find_side_length_find_angle_C_l385_385748

-- Definition for Problem 1
def triangle_problem_1 (A B C : ℝ) (a b c : ℝ) := 
  B = (30 * Real.pi / 180) ∧ 
  C = (135 * Real.pi / 180) ∧ 
  b = 2

-- Lean theorem for Problem 1
theorem find_side_length (a b c : ℝ) : 
  triangle_problem_1 a b c →
  a = Real.sqrt 6 - Real.sqrt 2 :=
begin
  sorry,
end

-- Definition for Problem 2
def triangle_area_problem (a b c S : ℝ) := 
  S = (1 / 4) * (a^2 + b^2 - c^2)

-- Lean theorem for Problem 2
theorem find_angle_C (a b c S : ℝ) : 
  triangle_area_problem a b c S →
  ∠C = Real.pi / 4 :=
begin
  sorry,
end

end find_side_length_find_angle_C_l385_385748


namespace greatest_perfect_power_sum_l385_385812

def sum_c_d_less_500 : ℕ :=
  let c := 22
  let d := 2
  c + d

theorem greatest_perfect_power_sum :
  ∃ c d : ℕ, 0 < c ∧ 1 < d ∧ c^d < 500 ∧
  ∀ x y : ℕ, 0 < x ∧ 1 < y ∧ x^y < 500 → x^y ≤ c^d ∧ (c + d = 24) :=
by
  sorry

end greatest_perfect_power_sum_l385_385812


namespace cos_210_eq_neg_sqrt3_div_2_l385_385862

theorem cos_210_eq_neg_sqrt3_div_2 : real.cos (210 * real.pi / 180) = - (real.sqrt 3 / 2) :=
by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385862


namespace floor_S_proof_l385_385277

noncomputable def floor_S (a b c d: ℝ) : ℝ :=
⌊a + b + c + d⌋

theorem floor_S_proof (a b c d : ℝ)
  (h1 : a ^ 2 + 2 * b ^ 2 = 2016)
  (h2 : c ^ 2 + 2 * d ^ 2 = 2016)
  (h3 : a * c = 1024)
  (h4 : b * d = 1024)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) : floor_S a b c d = 129 := 
sorry

end floor_S_proof_l385_385277


namespace number_of_noncongruent_triangles_is_five_l385_385981

def points : list (ℝ × ℝ) := [(0,0), (1,0), (2,0), (0,0), (1,1), (2,2), (0,0), (0.5,1), (1,2)]

def unique_triangles (pts : list (ℝ × ℝ)) : ℕ :=
  -- definition for counting unique (non-congruent) triangles from given points
  sorry

theorem number_of_noncongruent_triangles_is_five :
  unique_triangles points = 5 :=
sorry

end number_of_noncongruent_triangles_is_five_l385_385981


namespace general_term_formula_l385_385186

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then 1 else
  let a₀ := 1 in
  let a_n := λ n, (sequence (n - 1) / (1 + 2 * sequence (n - 1))) in
  a_n n

theorem general_term_formula (n : ℕ) : sequence n = 1 / (2 * n - 1) :=
sorry

end general_term_formula_l385_385186


namespace train_seat_count_l385_385036

theorem train_seat_count (t : ℝ)
  (h1 : ∃ (t : ℝ), t = 36 + 0.2 * t + 0.5 * t) :
  t = 120 :=
by
  sorry

end train_seat_count_l385_385036


namespace sum_of_digits_of_even_numbers_600_to_800_l385_385124

theorem sum_of_digits_of_even_numbers_600_to_800 
  (S : Finset ℕ) 
  (h : ∀ n ∈ S, 600 ≤ n ∧ n ≤ 800 ∧ Even n)
: S.sum (λ n, (n / 100) + (n % 100 / 10) + (n % 10)) = 1508 :=
sorry

end sum_of_digits_of_even_numbers_600_to_800_l385_385124


namespace ten_digit_numbers_multiple_of_11111_l385_385982

open Nat

theorem ten_digit_numbers_multiple_of_11111 : 
  let distinct_digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let ten_digit_numbers := Finset.filter 
                            (λ n : ℕ => (∃ (d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 : ℕ), 
                                         distinct_digits d1 ∧ 
                                         distinct_digits d2 ∧ 
                                         distinct_digits d3 ∧ 
                                         distinct_digits d4 ∧ 
                                         distinct_digits d5 ∧ 
                                         distinct_digits d6 ∧ 
                                         distinct_digits d7 ∧ 
                                         distinct_digits d8 ∧ 
                                         distinct_digits d9 ∧ 
                                         distinct_digits d10 ∧ 
                                         distinct {d1, d2, d3, d4, d5, d6, d7, d8, d9, d10} = 10 ∧ 
                                         n = 1000000000 * d1 + 100000000 * d2 + 10000000 * d3 + 
                                              1000000 * d4 + 100000 * d5 + 10000 * d6 + 
                                              1000 * d7 + 100 * d8 + 10 * d9 + d10 ∧
                                         n % 11111 = 0))
                            (Finset.range (10^10))
  ten_digit_numbers.card = 7560 :=
begin
  sorry
end

end ten_digit_numbers_multiple_of_11111_l385_385982


namespace systematic_sampling_correct_l385_385519

-- Define the total number of people and sample size
def total_people : ℕ := 60
def sample_size : ℕ := 6

-- Calculate the sampling interval
def sampling_interval (N : ℕ) (n : ℕ) : ℕ := N / n

noncomputable def sampled_people (start : ℕ) (k : ℕ) : list ℕ :=
  (list.range sample_size).map (λ i, start + i * k)

-- The statement of the problem
theorem systematic_sampling_correct :
  sampling_interval total_people sample_size = 10 ∧
  sampled_people 7 10 = [7, 17, 27, 37, 47, 57] :=
by
  sorry

end systematic_sampling_correct_l385_385519


namespace dodecahedron_probability_endpoints_edge_l385_385424

theorem dodecahedron_probability_endpoints_edge :
  let V := 20
  let E := 30
  let pairs := (V * (V - 1)) / 2
  probability (endpoints_of_edge E pairs) = 3 / 19 := 
sorry

end dodecahedron_probability_endpoints_edge_l385_385424


namespace cos_210_eq_neg_sqrt3_div2_l385_385836

theorem cos_210_eq_neg_sqrt3_div2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div2_l385_385836


namespace part1_part2_max_part2_max_at_0_part2_min_part2_min_at_5pi6_l385_385565

def a (x : ℝ) : ℝ × ℝ := ⟨Real.cos x, Real.sin x⟩
def b : ℝ × ℝ := ⟨3, -Real.sqrt 3⟩
def f (x : ℝ) : ℝ := 3 * Real.cos x - Real.sqrt 3 * Real.sin x

theorem part1 (x : ℝ) (hx : x ∈ Set.Icc 0 Real.pi) (h : a x = b) : x = 5 * Real.pi / 6 :=
sorry

theorem part2_max (x : ℝ) (hx : x ∈ Set.Icc 0 Real.pi) : f x ≤ 3 :=
sorry

theorem part2_max_at_0 : f 0 = 3 :=
sorry

theorem part2_min (x : ℝ) (hx : x ∈ Set.Icc 0 Real.pi) : f x ≥ -2 * Real.sqrt 3 :=
sorry

theorem part2_min_at_5pi6 : f (5 * Real.pi / 6) = -2 * Real.sqrt 3 :=
sorry

end part1_part2_max_part2_max_at_0_part2_min_part2_min_at_5pi6_l385_385565


namespace intersection_of_lines_l385_385117

theorem intersection_of_lines :
  ∃ (x y : ℝ), (8 * x + 5 * y = 40) ∧ (3 * x - 10 * y = 15) ∧ (x = 5) ∧ (y = 0) := 
by 
  sorry

end intersection_of_lines_l385_385117


namespace johns_average_speed_l385_385257

-- Conditions
def biking_time_minutes : ℝ := 45
def biking_speed_mph : ℝ := 20
def walking_time_minutes : ℝ := 120
def walking_speed_mph : ℝ := 3

-- Proof statement
theorem johns_average_speed :
  let biking_time_hours := biking_time_minutes / 60
  let biking_distance := biking_speed_mph * biking_time_hours
  let walking_time_hours := walking_time_minutes / 60
  let walking_distance := walking_speed_mph * walking_time_hours
  let total_distance := biking_distance + walking_distance
  let total_time := biking_time_hours + walking_time_hours
  let average_speed := total_distance / total_time
  average_speed = 7.64 :=
by
  sorry

end johns_average_speed_l385_385257


namespace total_pages_in_book_l385_385703

theorem total_pages_in_book : 
  ∀ (n : ℕ), (∑ i in finset.range n, if i < 10 then 1 else if i < 100 then 2 else 3) = 990 → n = 366 := by
  intro n h
  sorry

end total_pages_in_book_l385_385703


namespace min_roots_g_equals_zero_l385_385474

-- Given conditions as functions and properties
def g : ℝ → ℝ := sorry
axiom g_symmetry1 : ∀ x : ℝ, g (3 + x) = g (3 - x)
axiom g_symmetry2 : ∀ x : ℝ, g (8 + x) = g (8 - x)
axiom g_at_zero : g 0 = 0

-- Statement of the problem
theorem min_roots_g_equals_zero : 
  ∃ s : Set ℝ, (∀ x : ℝ, x ∈ s → g x = 0 ∧ -1500 ≤ x ∧ x ≤ 1500) ∧ s.card ≥ 690 :=
sorry

end min_roots_g_equals_zero_l385_385474


namespace sum_c_d_eq_24_l385_385818

theorem sum_c_d_eq_24 (c d : ℕ) (h_pos_c : c > 0) (h_pos_d : d > 1) (h_max_power : c^d < 500 ∧ ∀ ⦃x y : ℕ⦄, x^y < 500 → x^y ≤ c^d) : c + d = 24 :=
sorry

end sum_c_d_eq_24_l385_385818


namespace probability_less_than_8000_l385_385333

def air_distance : matrix (fin 4) (fin 4) ℕ :=
  ![![0, 9700, 10870, 7800], 
    ![9700, 0, 5826, 16900],
    ![10870, 5826, 0, 16000],
    ![7800, 16900, 16000, 0]]

def num_pairs : ℕ := 6
def pairs_below_8000 : ℕ := 2

theorem probability_less_than_8000 : (pairs_below_8000 : ℚ) / (num_pairs : ℚ) = 1 / 3 :=
by
  sorry

end probability_less_than_8000_l385_385333


namespace exists_polygonal_chain_l385_385235

theorem exists_polygonal_chain (n : ℕ) (lines : Fin n → Set (ℝ × ℝ))
  (h_non_parallel : ∀ i j, i ≠ j → ¬ ∃ k, lines i = lines j ∧ (lines i ≠ lines k))
  (h_no_triplet_intersection : ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬ ∃ p, p ∈ lines i ∧ p ∈ lines j ∧ p ∈ lines k) :
  ∃ (chain : Fin (n + 1) → (ℝ × ℝ)), (∀ i, (chain i, chain (i + 1 % (n+1))) ∈ lines i) ∧ (∀ i j, i ≠ j → chain i ≠ chain j) := 
sorry

end exists_polygonal_chain_l385_385235


namespace arrangement_combination_l385_385012

open Nat

noncomputable def A (n r : ℕ) : ℕ :=
  factorial n / factorial (n - r)

noncomputable def C (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem arrangement_combination (hA : A 7 2 = 7 * 6)
                                (hC : C 10 2 = (10 * 9) / 2) :
  (A 7 2 : ℚ) / C 10 2 = 14 / 15 := 
by
  rw [hA, hC]
  norm_num
  sorry

end arrangement_combination_l385_385012


namespace problem_statement_l385_385283

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 0 then 1 - real.sqrt x else 2^x

theorem problem_statement :
  f (f (-2)) = 1 / 2 := by
  sorry

end problem_statement_l385_385283


namespace solution_set_of_inequalities_l385_385695

-- Define the conditions of the inequality system
def inequality1 (x : ℝ) : Prop := x - 2 ≥ -5
def inequality2 (x : ℝ) : Prop := 3 * x < x + 2

-- The statement to prove the solution set of the inequalities
theorem solution_set_of_inequalities :
  { x : ℝ | inequality1 x ∧ inequality2 x } = { x : ℝ | -3 ≤ x ∧ x < 1 } :=
  sorry

end solution_set_of_inequalities_l385_385695


namespace broken_line_length_l385_385070

theorem broken_line_length (A B C D S K L X Y : Point) 
  (h1 : is_square A B C D) 
  (h2 : distance A B = 6 ∧ distance B C = 6 ∧ distance C D = 6 ∧ distance D A = 6)
  (h3 : diagonals_intersect A C B D S) 
  (h4 : is_square B K C S) 
  (h5 : is_square A S D L) 
  (h6 : line_intersects_segment K L A D X) 
  (h7 : line_intersects_segment K L B C Y) :
  length_broken_line K Y B A X L = 18 :=
sorry

end broken_line_length_l385_385070


namespace range_of_alpha_l385_385700

theorem range_of_alpha (α θ : ℝ) 
  (h1 : ∃ θ, (sin (α - π / 3) = sin (α - π / 3)) ∧ (sqrt 3 = sqrt 3))
  (h2 : sin (2 * θ) ≤ 0) :
  -2 * π / 3 ≤ α ∧ α ≤ π / 3 :=
by
  sorry

end range_of_alpha_l385_385700


namespace cos_210_eq_neg_sqrt3_div2_l385_385835

theorem cos_210_eq_neg_sqrt3_div2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div2_l385_385835


namespace man_walking_time_l385_385890

section TrainProblem

variables {T W : ℕ}

/-- Each day a man meets his wife at the train station after work,
    and then she drives him home. She always arrives exactly on time to pick him up.
    One day he catches an earlier train and arrives at the station an hour early.
    He immediately begins walking home along the same route the wife drives.
    Eventually, his wife sees him on her way to the station and drives him the rest of the way home.
    When they arrive home, the man notices that they arrived 30 minutes earlier than usual.
    How much time did the man spend walking? -/
theorem man_walking_time : 
    (∃ (T : ℕ), T > 30 ∧ (W = T - 30) ∧ (W + 30 = T)) → W = 30 :=
sorry

end TrainProblem

end man_walking_time_l385_385890


namespace factorize_difference_of_squares_l385_385107

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_difference_of_squares_l385_385107


namespace estimated_weight_l385_385361

def sum_x := 1600
def sum_y := 460
def b_hat := 0.85
def n := 10

def x_bar := sum_x / n
def y_bar := sum_y / n
def a_hat := y_bar - b_hat * x_bar

theorem estimated_weight (height : ℝ) : height = 170 → b_hat * height + a_hat = 54.5 := 
by 
  let x_bar_val := x_bar
  let y_bar_val := y_bar
  let a_hat_val := a_hat
  intro h170
  have : x_bar = 160 := by 
    unfold x_bar
    simp [sum_x, n]
  have : y_bar = 46 := by 
    unfold y_bar
    simp [sum_y, n]
  unfold a_hat at a_hat_val
  rw [this] at a_hat_val
  rw [this_1] at a_hat_val
  have : a_hat = -90 := by 
    unfold a_hat
    rw [this, this_1]
    simp [b_hat]
  rw [this] at a_hat_val
  simp [h170, b_hat, a_hat_val]
  sorry

end estimated_weight_l385_385361


namespace num_valid_m_l385_385508

theorem num_valid_m (m : ℕ) : (∃ n : ℕ, n * (m^2 - 3) = 1722) → ∃ p : ℕ, p = 3 := 
  by
  sorry

end num_valid_m_l385_385508


namespace composite_function_properties_l385_385171

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem composite_function_properties
  (a b c : ℝ)
  (h_a_nonzero : a ≠ 0)
  (h_no_real_roots : ∀ x : ℝ, f a b c x ≠ x) :
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) ∧
  (a > 0 → ∀ x : ℝ, f a b c (f a b c x) > x) :=
by sorry

end composite_function_properties_l385_385171


namespace main_theorem_l385_385345

noncomputable def parabola_focus : Point := ⟨1, 0⟩

def line_through_focus (x : Real) : Prop := x = 1

def hyperbola_asymptote_1 (x y : Real) : Prop := y = √3 * x

def hyperbola_asymptote_2 (x y : Real) : Prop := y = -√3 * x

def distance (A B : Point) : Real :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem main_theorem : distance ⟨1, √3⟩ ⟨1, -√3⟩ = 2 * √3 := 
sorry

end main_theorem_l385_385345


namespace harriet_speed_l385_385385

theorem harriet_speed (v_AB : ℝ) (t_total : ℝ) (t_AB_min : ℝ) (v_BA : ℝ) : 
  v_AB = 95 → 
  t_total = 5 → 
  t_AB_min = 186 →
  v_BA = 294.5 / ((5 - t_AB_min / 60)) :=
begin
  intros h1 h2 h3,
  have t_AB : ℝ := t_AB_min / 60,
  have distance : ℝ := v_AB * t_AB,
  have t_BA : ℝ := t_total - t_AB,
  have h4 : v_BA = distance / t_BA,
  simp [h1, h2, h3] at *,
  rw [h4],
  exact h4,
  sorry
end

end harriet_speed_l385_385385


namespace Q_on_circumcircle_AIC_l385_385760

-- Defining the conditions in the problem
variables {A B C D I P Q : Type} -- Points
variables [has_center I (inscribed_circle A B C D I)] -- Quadrilateral ABCD has an inscribed circle centered at I
variables [intersection P (ray BA) (ray CD)] -- P is the intersection of rays BA and CD
variables [intersection Q (ray AD) (ray BC)] -- Q is the intersection of rays AD and BC
variables [P_on_circumcircle (circumcircle_of_triangle A I C) P] -- P lies on the circumcircle of triangle AIC

-- Defining the proof problem
theorem Q_on_circumcircle_AIC :
  Q_on_circumcircle (circumcircle_of_triangle A I C) Q :=
sorry

end Q_on_circumcircle_AIC_l385_385760


namespace problem_l385_385045

-- Defining the conditions as per the problem statement
variables (p q : ℝ) (A B C : { x y : ℝ // x^2 / p^2 + y^2 / q^2 = 1 })
          (F1 F2 : { x y : ℝ // x^2 / p^2 + y^2 / q^2 = 1 })
          (s : ℝ)
          (F1_F2 : ℝ)

-- Stating the given conditions
axiom is_equilateral_triangle : (B.1, B.2) = (0, q)
axiom AC_parallel_to_x : ∃ M : { x y : ℝ // x^2 / p^2 + y^2 / q^2 = 1 }, 
  (C.1, C.2) = (M.1 + s/2, M.2 - sqrt 3 / 2 * s)
axiom foci_on_sides : (F1.1, F1.2) ∈ set_of (λ F, F ∈ segment (interval (B.1, B.2) (C.1, C.2)))
                    ∧ (F2.1, F2.2) ∈ set_of (λ F, F ∈ segment (interval (A.1, A.2) (B.1, B.2)))
axiom F1_F2_distance : F1_F2 = 4

-- Defining the problem to be proven
theorem problem : p = 4 ∧ q = 2 * sqrt 3 → 
  ∃ (s : ℝ), (AB F1 F2 : ℝ) ≥ 0 ∧ s = 8 * sqrt 3 / 5 → AB / F1_F2 = 2 * sqrt 3 / 5 :=
sorry

end problem_l385_385045


namespace factorial_fraction_l385_385060

theorem factorial_fraction :
  (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_fraction_l385_385060


namespace binom_coeff_sum_l385_385465

theorem binom_coeff_sum :
  (Nat.choose 510 0) + (Nat.choose 510 510) = 2 :=
by
  have h₁ : Nat.choose 510 0 = 1 := Nat.choose_zero_right 510
  have h₂ : Nat.choose 510 510 = 1 := Nat.choose_self 510
  rw [h₁, h₂]
  norm_num
  sorry

end binom_coeff_sum_l385_385465


namespace problem_part1_problem_part2_l385_385301

open Real

-- Definitions for the conditions
def vec_a (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin x, sin x)
def vec_b (x : ℝ) : ℝ × ℝ := (cos x, sin x)
def domain_x (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ π / 2

-- Definitions for the magnitudes of vectors
def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Definition for the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Given the conditions, prove that x = π / 6
theorem problem_part1 (x : ℝ) (h₀ : domain_x x)
  (h₁ : magnitude (vec_a x) = magnitude (vec_b x)) :
  x = π / 6 :=
  sorry

-- Given the conditions, find the range of the function f(x)
theorem problem_part2 : set.range (λ x, dot_product (vec_a x) (vec_b x))
  = set.Icc 0 (3 / 2) :=
  sorry

end problem_part1_problem_part2_l385_385301


namespace integral_sqrt_9_minus_x_squared_l385_385705

theorem integral_sqrt_9_minus_x_squared :
  ∫ x in 0..3, real.sqrt (9 - x^2) = (9 / 4) * real.pi :=
by
  sorry

end integral_sqrt_9_minus_x_squared_l385_385705


namespace find_daisy_sale_first_day_l385_385878

def daisy_sale_first_day (D : ℕ) (D2 : ℕ) (D3 : ℕ) (D4 : ℕ) : Prop :=
  D2 = D + 20 ∧ 
  D3 = (2 * D2) - 10 ∧ 
  D + D2 + D3 + D4 = 350 ∧ 
  D4 = 120 → 
  D = 45

theorem find_daisy_sale_first_day (D : ℕ) (D2 : ℕ) (D3 : ℕ) (D4 : ℕ) : 
  daisy_sale_first_day D D2 D3 D4 :=
by
  -- conditions
  assume h: D2 = D + 20 ∧ D3 = (2 * D2) - 10 ∧ D + D2 + D3 + D4 = 350 ∧ D4 = 120,
  sorry

end find_daisy_sale_first_day_l385_385878


namespace irrationals_count_correct_l385_385802

noncomputable def countIrrationals (l : List ℝ) : Nat :=
  l.countp (λ x, ¬ ∃ (a b : ℚ), b ≠ 0 ∧ x = a / b)

theorem irrationals_count_correct :
  countIrrationals [22 / 7, 3.14159, real.sqrt 7, -8, real.cbrt 2, 0.6, 0, real.sqrt 36, real.pi / 3] = 3 := 
by 
  sorry

end irrationals_count_correct_l385_385802


namespace hexadecagon_area_l385_385425

noncomputable def area_of_regular_hexadecagon (r : ℝ) : ℝ :=
  16 * (1 / 2 * r * r * real.sin (real.pi / 8))

theorem hexadecagon_area (r : ℝ) : area_of_regular_hexadecagon r = 3 * r * r :=
by
  sorry

end hexadecagon_area_l385_385425


namespace cos_210_eq_neg_sqrt3_div_2_l385_385871

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385871


namespace ellipse_equation_chord_length_l385_385947

-- Define the conditions using Lean definitions
variables {a b : ℝ} (h_a_gt_b : a > b) (h_b_gt_0 : b > 0)
variables (e : ℝ) (h_e : e = (Real.sqrt 6) / 3)
variables (c : ℝ) (h_c_eq_ea : c = e * a)
variables (d : ℝ) (h_d : d = Real.sqrt 3) (h_distance : d = a - c)

-- Define the first proof goal
theorem ellipse_equation (h1 : a = Real.sqrt 3) (h2 : c = Real.sqrt 2) 
  (h3 : b^2 = a^2 - c^2) : 
  (ellipse_eq : ∀ x y : ℝ, (x^2 / 3 + y^2 = 1)) :=
sorry

-- Parameters and definitions for the second proof goal
variables (x1 x2 : ℝ) (y1 y2 : ℝ)
variables (h_focus : x1 + x2 = -(3 * Real.sqrt 2) / 2)
variables (h_product : x1 * x2 = 3 / 4)
variables (h_line : y1 = x1 + Real.sqrt 2) (h_line2 : y2 = x2 + Real.sqrt 2)

-- Define the second proof goal
theorem chord_length : 
  ∀ x1 x2 : ℝ, (x1 + x2 = -(3 * Real.sqrt 2) / 2) →
  (x1 * x2 = 3 / 4) →
  (abs (x1 - x2) = 3 / 2) :=
sorry 

end ellipse_equation_chord_length_l385_385947


namespace factorize_x_squared_minus_one_l385_385101

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_x_squared_minus_one_l385_385101


namespace intersection_when_a_eq_2_subset_range_l385_385967

-- Definitions for the sets A and B
def setA (a : ℝ) : set ℝ := {x | x^2 - 3 * (a + 1) * x + 2 * (3 * a + 1) < 0}
def setB (a : ℝ) : set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

-- Problem 1: Prove that when a = 2, A ∩ B = (4, 5)
theorem intersection_when_a_eq_2 : setA 2 ∩ setB 2 = set.Ioo 4 5 :=
by sorry

-- Problem 2: Prove that the range of real numbers a such that B ⊆ A is [-1, 3]
theorem subset_range : {a : ℝ | setB a ⊆ setA a} = {a | -1 ≤ a ∧ a ≤ 3} :=
by sorry

end intersection_when_a_eq_2_subset_range_l385_385967


namespace roots_sum_l385_385155

theorem roots_sum (a b : ℝ) 
  (h₁ : 3^(a-1) = 6 - a)
  (h₂ : 3^(6-b) = b - 1) : 
  a + b = 7 := 
by sorry

end roots_sum_l385_385155


namespace asymptote_sum_l385_385341

theorem asymptote_sum:
  (∀ (x : ℝ), (y : ℝ) = (6 * x^2 - 8) / (4 * x^2 + 6 * x + 3)) →
  let roots := {x : ℝ | 4 * x^2 + 6 * x + 3 = 0} in
  ∃ p q : ℝ, (p ∈ roots ∧ q ∈ roots ∧ p + q = -2) :=
by
  sorry

end asymptote_sum_l385_385341


namespace find_interest_rate_l385_385999

-- Definitions corresponding to the conditions
def doubilium (r : ℝ) : ℝ := 70 / r

def previous_amount : ℝ := 5000
def final_amount : ℝ := 20000
def years : ℝ := 18

-- Derived condition based on the quadrupling in 18 years.
def double_times (rate_of_doubling : ℝ) := final_amount = previous_amount * 2^2
def years_per_doubling (rate_of_doubling : ℝ) := years / 2 = doubilium rate_of_doubling

-- The main proof statement
theorem find_interest_rate : 
  ∃ r : ℝ, final_amount = previous_amount * 2^2 ∧ years / 2 = doubilium r := 
  sorry

end find_interest_rate_l385_385999


namespace cos_210_eq_neg_sqrt3_div_2_l385_385869

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385869


namespace dataset_25th_percentile_l385_385661

noncomputable def dataset : List ℝ := [2.8, 3.6, 4.0, 3.0, 4.8, 5.2, 4.8, 5.7, 5.8, 3.3]

noncomputable def percentile_25 (data : List ℝ) : ℝ :=
  let sorted_data := data.qsort (· < ·)
  if sorted_data.length = 0 then 0 else
    let pos := (sorted_data.length * 25) / 100
    if pos.ceil.toNat < sorted_data.length then
      sorted_data.get! pos.ceil.toNat
    else 0

theorem dataset_25th_percentile : percentile_25 dataset = 3.3 :=
  by
    -- We will sort the dataset and find the 25th percentile position.
    sorry

end dataset_25th_percentile_l385_385661


namespace spinners_even_product_probability_l385_385326

def even_product_probability {A B : Type} [Fintype A] [Fintype B] (pA : A → ℕ) (pB : B → ℕ) (P : Finset A) (Q : Finset B) : ℚ :=
  (P.card * Q.card) / ((Fintype.card A) * (Fintype.card B))

theorem spinners_even_product_probability :
  let A := {1, 2, 3, 4, 5, 6}
  let B := {1, 2, 3, 4}
  let pA : Finset ℕ := {n : ℕ | n ∈ A ∧ n % 2 = 0}
  let pB : Finset ℕ := {n : ℕ | n ∈ B ∧ n % 2 = 0}
  let oddsA : Finset ℕ := {n : ℕ | n ∈ A ∧ ¬(n % 2 = 0)}
  let oddsB : Finset ℕ := {n : ℕ | n ∈ B ∧ ¬(n % 2 = 0)}
  even_product_probability (λ x => x) (λ y => y) pA B + even_product_probability (λ x => x) (λ y => y) A pB - even_product_probability (λ x => x) (λ y => y) pA pB = 1 / 2 := by
  sorry
 
end spinners_even_product_probability_l385_385326


namespace john_finds_train_l385_385433

noncomputable def probability_train_present (x y : ℝ) : ℝ :=
if (0 ≤ y ∧ y ≤ 60 ∧ y ≤ x ∧ x ≤ y + 30 ∧ 0 ≤ x ∧ x ≤ 60)
then 1 else 0

noncomputable def total_probability : ℝ :=
∫ x in 0..60, ∫ y in 0..60, probability_train_present x y / (60 * 60)

theorem john_finds_train :
  total_probability = 1 / 2 :=
by
  sorry

end john_finds_train_l385_385433


namespace cos_210_eq_neg_sqrt3_div_2_l385_385863

theorem cos_210_eq_neg_sqrt3_div_2 : real.cos (210 * real.pi / 180) = - (real.sqrt 3 / 2) :=
by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385863


namespace solve_inequalities_system_l385_385693

theorem solve_inequalities_system (x : ℝ) :
  (x - 2 ≥ -5) → (3x < x + 2) → (-3 ≤ x ∧ x < 1) :=
by
  intros h1 h2
  sorry

end solve_inequalities_system_l385_385693


namespace log_eq_one_third_l385_385567

theorem log_eq_one_third (x : ℝ) (h : log 4 (x - 3) = 2 / 3) : log 16 x = 1 / 3 :=
sorry

end log_eq_one_third_l385_385567


namespace range_of_m_l385_385536

def f (x : ℝ) : ℝ := Real.log (2 + x) - Real.log (2 - x)
def A : Set ℝ := Set.Ioo (-2) 2

def g (x : ℝ) : ℝ := x^2 + 2 * x + m
def B (m : ℝ) : Set ℝ := Set.Ici (m - 1)

theorem range_of_m (m : ℝ) : A ⊆ B m ↔ m ≤ -1 := sorry

end range_of_m_l385_385536


namespace solve_for_x_l385_385325

theorem solve_for_x (x : ℝ) : 5 * x - 3 * x = 405 - 9 * (x + 4) → x = 369 / 11 := by
  sorry

end solve_for_x_l385_385325


namespace sin_neg_pi_over_6_eq_neg_half_l385_385704

theorem sin_neg_pi_over_6_eq_neg_half : sin (-π / 6) = -1 / 2 :=
by
  -- Using the identity sin(-x) = -sin(x)
  have h1 : sin (-π / 6) = -sin (π / 6) := by sorry
  -- Knowing sin(π / 6) = 1/2
  have h2 : sin (π / 6) = 1 / 2 := by sorry
  -- Combining the results
  rw [h1, h2]
  -- Showing the final result
  exact rfl

end sin_neg_pi_over_6_eq_neg_half_l385_385704


namespace find_g_neg2_l385_385298

def f (x : ℝ) : ℝ := x^3 - 2 * x^2 + 5

def g_polynomial_roots_quadratic (g f : polynomial ℝ) :=
∀ (x : ℝ), g.eval x = 1 → (∃ u v w : ℝ,
  f.evals [u, v, w] = 0 ∧
  g = polynomial.monomial 0 (1/5) *
    (X - C u^2) *
    (X - C v^2) *
    (X - C w^2))

theorem find_g_neg2 :
  (∃ (g : polynomial ℝ), g.eval 0 = 1 ∧
  g_polynomial_roots_quadratic g f) →
  g.eval (-2) = 24.2 :=
by
  sorry

end find_g_neg2_l385_385298


namespace jims_investment_l385_385392

theorem jims_investment (total_investment : ℝ) (john_ratio : ℝ) (james_ratio : ℝ) (jim_ratio : ℝ) 
                        (h_total_investment : total_investment = 80000)
                        (h_ratio_john : john_ratio = 4)
                        (h_ratio_james : james_ratio = 7)
                        (h_ratio_jim : jim_ratio = 9) : 
    jim_ratio * (total_investment / (john_ratio + james_ratio + jim_ratio)) = 36000 :=
by 
  sorry

end jims_investment_l385_385392


namespace factorize_x_squared_minus_one_l385_385094

theorem factorize_x_squared_minus_one (x : Real) : (x^2 - 1) = (x + 1) * (x - 1) :=
sorry

end factorize_x_squared_minus_one_l385_385094


namespace instantaneous_velocity_at_t_2_l385_385758

def y (t : ℝ) : ℝ := 3 * t^2 + 4

theorem instantaneous_velocity_at_t_2 :
  deriv y 2 = 12 :=
by
  sorry

end instantaneous_velocity_at_t_2_l385_385758


namespace greatest_perfect_power_sum_l385_385814

def sum_c_d_less_500 : ℕ :=
  let c := 22
  let d := 2
  c + d

theorem greatest_perfect_power_sum :
  ∃ c d : ℕ, 0 < c ∧ 1 < d ∧ c^d < 500 ∧
  ∀ x y : ℕ, 0 < x ∧ 1 < y ∧ x^y < 500 → x^y ≤ c^d ∧ (c + d = 24) :=
by
  sorry

end greatest_perfect_power_sum_l385_385814


namespace opposite_of_neg_three_l385_385684

def opposite (x : Int) : Int := -x

theorem opposite_of_neg_three : opposite (-3) = 3 := by
  -- To be proven using Lean
  sorry

end opposite_of_neg_three_l385_385684


namespace ashley_family_spent_30_l385_385808

def cost_of_child_ticket : ℝ := 4.25
def cost_of_adult_ticket : ℝ := cost_of_child_ticket + 3.25
def discount : ℝ := 2.00
def num_adult_tickets : ℕ := 2
def num_child_tickets : ℕ := 4

def total_cost : ℝ := num_adult_tickets * cost_of_adult_ticket + num_child_tickets * cost_of_child_ticket - discount

theorem ashley_family_spent_30 :
  total_cost = 30.00 :=
sorry

end ashley_family_spent_30_l385_385808


namespace larger_R2_smaller_residual_l385_385727

-- Define the correlation coefficient R^2 and its properties
variable {R2 : Type} [Preorder R2] [TopologicalSpace R2] [OrderTopology R2]

/-- R^2 is a measure of goodness of fit -/
def goodness_of_fit (R2 : Type) [Preorder R2] : Prop :=
∀ ε > 0, ∃ R2, R2 > 1 - ε

/-- The closer R^2 is to 1, the better the fit -/
axiom closer_to_one_better_fit : ∀ {R2}, goodness_of_fit R2 → 
  (∀ ε > 0, |R2 - 1| < ε → ResidualSumOfSquares R2 < ResidualSumOfSquares (1 - ε))

-- The theorem to prove
theorem larger_R2_smaller_residual (R2 : Type) [(r : Preorder R2)] [TopologicalSpace R2]
  [OrderTopology R2] (h : goodness_of_fit R2) : 
  ∀ ε > 0, ∃ R2, R2 > 1 - ε → ResidualSumOfSquares R2 < ResidualSumOfSquares (1 - ε) := 
by 
  sorry

end larger_R2_smaller_residual_l385_385727


namespace morgan_change_l385_385626

-- Define the costs of the items and the amount paid
def hamburger_cost : ℕ := 4
def onion_rings_cost : ℕ := 2
def smoothie_cost : ℕ := 3
def amount_paid : ℕ := 20

-- Define total cost
def total_cost := hamburger_cost + onion_rings_cost + smoothie_cost

-- Define the change received
def change_received := amount_paid - total_cost

-- Statement of the problem in Lean 4
theorem morgan_change : change_received = 11 := by
  -- include proof steps here
  sorry

end morgan_change_l385_385626


namespace correct_option_D_l385_385726

theorem correct_option_D : -2 = -|-2| := 
by 
  sorry

end correct_option_D_l385_385726


namespace train_length_l385_385037

theorem train_length :
  ∀ (speed_train speed_man : ℝ) (time : ℝ),
  speed_train = 63 → speed_man = 3 →
  time = 35.99712023038157 →
  let relative_speed := (speed_train - speed_man) * (1000 / 3600) in
  let length_train := relative_speed * time in
  length_train = 599.95200371166 :=
begin
  intros speed_train speed_man time h_speed_train h_speed_man h_time,
  let relative_speed := (speed_train - speed_man) * (1000 / 3600),
  let length_train := relative_speed * time,
  have h := by {
                   rw [h_speed_train, h_speed_man, h_time],
                   norm_num,
                 },
  assumption,
  sorry
end

end train_length_l385_385037


namespace parabola_intersects_negative_x_axis_l385_385129

noncomputable def count_valid_parabolas : ℕ :=
  -- Define the set of coefficients
  let coefficients := {-1, 0, 1, 2, 3} in
  -- Ensure coefficients are distinct
  let is_distinct (a b c : ℤ) := a ≠ b ∧ a ≠ c ∧ b ≠ c in
  -- Check if the quadratic equation intersects the negative x-axis
  let intersects_negative_x_axis (a b c : ℤ) := 
    let Δ := b * b - 4 * a * c in
    Δ ≥ 0 ∧ (b < 0 ∨ (b * b > 4 * a * c)) in
  -- Count all valid (a, b, c) combinations
  Finset.filter (λ (abc : ℤ × ℤ × ℤ), is_distinct abc.1 abc.2.1 abc.2.2 ∧ intersects_negative_x_axis abc.1 abc.2.1 abc.2.2)
    (Finset.product (Finset.product coefficients coefficients) coefficients).card

theorem parabola_intersects_negative_x_axis :
  count_valid_parabolas = 26 := sorry

end parabola_intersects_negative_x_axis_l385_385129


namespace PartI_PartII_l385_385180

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Problem statement for (Ⅰ)
theorem PartI (x : ℝ) : (f x < |x| + 1) ↔ (0 < x ∧ x < 2) :=
by sorry

-- Define conditions for PartII
variables (x y : ℝ)
def condition1 : Prop := |x - y - 1| ≤ 1 / 3
def condition2 : Prop := |2 * y + 1| ≤ 1 / 6

-- Problem statement for (Ⅱ)
theorem PartII (h1 : condition1 x y) (h2 : condition2 y) : f x < 1 :=
by sorry

end PartI_PartII_l385_385180


namespace knights_round_table_l385_385003

theorem knights_round_table (n : ℕ) (h : ∃ (f e : ℕ), f = e ∧ f + e = n) : n % 4 = 0 :=
sorry

end knights_round_table_l385_385003


namespace max_sum_cd_l385_385822

theorem max_sum_cd (c d : ℕ) (hc : c > 0) (hd : d > 1) (hcd : c^d < 500) 
  (hmax : ∀ (c' d': ℕ), c' > 0 → d' > 1 → c'^d' < 500 → c'^d' ≤ c^d) : c + d = 24 := 
by
  have h1 : 22^2 = 484 := rfl
  have h2 : c = 22 ∧ d = 2 := by sorry
  exact by sorry

end max_sum_cd_l385_385822


namespace range_of_a_l385_385956

theorem range_of_a (a : ℝ) :
  (∃ x ∈ set.Ioo 4 6, deriv (λ x, (1 / 2) * x^2 - (a + 2) * x + 2 * a * real.log x + 1) x = 0) →
  4 < a ∧ a < 6 :=
begin
  -- Proof will be filled in here
  sorry
end

end range_of_a_l385_385956


namespace calculate_cakes_left_l385_385788

-- Define the conditions
def b_lunch : ℕ := 5
def s_dinner : ℕ := 6
def b_yesterday : ℕ := 3

-- Define the calculation of the total cakes baked and cakes left
def total_baked : ℕ := b_lunch + b_yesterday
def cakes_left : ℕ := total_baked - s_dinner

-- The theorem we want to prove
theorem calculate_cakes_left : cakes_left = 2 := 
by
  sorry

end calculate_cakes_left_l385_385788


namespace problem_f_2016_eq_l385_385179

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin x + b * x^3 + 4
noncomputable def f' (a b : ℝ) (x : ℝ) : ℝ := a * Real.cos x + 3 * b * x^2

theorem problem_f_2016_eq :
  ∀ (a b : ℝ),
  f a b 2016 + f a b (-2016) + f' a b 2017 - f' a b (-2017) = 8 + 2 * b * 2016^3 :=
by
  intro a b
  sorry

end problem_f_2016_eq_l385_385179


namespace range_of_a_l385_385954

noncomputable def f (x a : ℝ) : ℝ :=
  (1/2) * x^2 - (a + 2) * x + 2 * a * Real.log x + 1

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ set.Ioo 4 6, deriv (λ x, f x a) x = 0) ↔ a ∈ set.Ioo 4 6 :=
by
  sorry

end range_of_a_l385_385954


namespace polynomial_expansion_coefficient_a8_l385_385965

theorem polynomial_expansion_coefficient_a8 :
  let a := 1
  let a_1 := 10
  let a_2 := 45
  let a_3 := 120
  let a_4 := 210
  let a_5 := 252
  let a_6 := 210
  let a_7 := 120
  let a_8 := 45
  let a_9 := 10
  let a_10 := 1
  a_8 = 45 :=
by {
  sorry
}

end polynomial_expansion_coefficient_a8_l385_385965


namespace sum_second_largest_and_third_smallest_l385_385501

theorem sum_second_largest_and_third_smallest :
  let digits := [1, 6, 8] in
  let numbers := list.filter (λ n, 100 ≤ n ∧ n < 1000) (list.permutations digits).map (λ l, 100 * l.head! + 10 * l[1]! + l[2]!) in
  let sorted_numbers := list.sorted numbers in
  sorted_numbers[numbers.length - 2] + sorted_numbers[2] = 1434 :=
begin
  sorry,
end

end sum_second_largest_and_third_smallest_l385_385501


namespace intersection_of_A_and_B_l385_385926

variable (A : Set ℝ)
variable (B : Set ℝ)
variable (C : Set ℝ)

theorem intersection_of_A_and_B (hA : A = { x | -1 < x ∧ x < 3 })
                                (hB : B = { -1, 1, 2 })
                                (hC : C = { 1, 2 }) :
  A ∩ B = C := by
  sorry

end intersection_of_A_and_B_l385_385926


namespace find_x_from_triangle_area_l385_385909

theorem find_x_from_triangle_area :
  ∀ (x : ℝ), x > 0 ∧ (1 / 2) * x * 3 * x = 96 → x = 8 :=
by
  intros x hx
  -- The proof goes here
  sorry

end find_x_from_triangle_area_l385_385909


namespace pizzas_bought_l385_385630

def slices_per_pizza := 8
def total_slices := 16

theorem pizzas_bought : total_slices / slices_per_pizza = 2 := by
  sorry

end pizzas_bought_l385_385630


namespace sin_alpha_through_point_l385_385540

theorem sin_alpha_through_point (α : ℝ) (P : ℝ × ℝ) (hP : P = (-3, -Real.sqrt 3)) :
    Real.sin α = -1 / 2 :=
by
  sorry

end sin_alpha_through_point_l385_385540


namespace solve_z_l385_385577

noncomputable def complex_equation (z : ℂ) := (1 + 3 * Complex.I) * z = Complex.I - 3

theorem solve_z (z : ℂ) (h : complex_equation z) : z = Complex.I :=
by
  sorry

end solve_z_l385_385577


namespace jasons_down_payment_l385_385253

theorem jasons_down_payment :
  (∃ (down_payment loan_payment month_payment: ℝ),
    (car_cost : ℝ) (down_payment_total : ℝ) (interest_rate : ℝ) (months : ℕ),
    car_cost = 32000 ∧ 
    loan_payment = 525 * 48 ∧ 
    month_payment = 525 ∧ 
    interest_rate = 0.05 ∧
    months = 48 ∧
    down_payment_total = 525 * 48 + 48 * (0.05 * 525) ∧
    down_payment = car_cost - down_payment_total ∧
    down_payment = 5540) :=
begin
  sorry
end

end jasons_down_payment_l385_385253


namespace gcd_digits_l385_385223

theorem gcd_digits (a b : ℕ) (h₁ : a < 10^6) (h₂ : b < 10^6) (h₃ : 10^9 ≤ Nat.lcm a b) : Nat.log10 (Nat.gcd a b) < 3 :=
by
  sorry

end gcd_digits_l385_385223


namespace lines_perpendicular_to_same_plane_are_parallel_l385_385317

theorem lines_perpendicular_to_same_plane_are_parallel
  (a b α : Type)
  [Line a] [Line b] [Plane α]
  (h₁ : Perpendicular a α)
  (h₂ : Perpendicular b α) :
  Parallel a b :=
sorry

end lines_perpendicular_to_same_plane_are_parallel_l385_385317


namespace incorrect_statements_c_d_l385_385732

theorem incorrect_statements_c_d
  (a b c m : ℝ)
  (h1 : ¬ (∃ x : ℝ, x^2 + 4 * x + m = 0)) :
  ¬ ((a * b^2 > c * b^2) ↔ (a > c)) ∧ (m ≤ 4 → false) := 
begin
  sorry
end

end incorrect_statements_c_d_l385_385732


namespace max_value_3x_plus_4y_l385_385149

theorem max_value_3x_plus_4y (x y : ℝ) : x^2 + y^2 = 14 * x + 6 * y + 6 → 3 * x + 4 * y ≤ 73 :=
sorry

end max_value_3x_plus_4y_l385_385149


namespace sum_of_non_palindrome_six_steps_to_palindrome_l385_385911

noncomputable def reverse_num (n : ℕ) : ℕ :=
  let digits := n.digits 10
  (digits.reverse.zipWith (λ x k, x * 10^k) (List.range digits.length)).sum

def steps_to_palindrome (n : ℕ) : ℕ :=
  let rec helper (m : ℕ) (steps : ℕ) : ℕ :=
    if m = reverse_num m then steps else
      helper (m + reverse_num m) (steps + 1)
  helper n 0

def is_palindrome (n : ℕ) : Prop := n = reverse_num n

def two_digit_non_palindrome (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ ¬ is_palindrome n

theorem sum_of_non_palindrome_six_steps_to_palindrome :
  (∑ n in Finset.filter two_digit_non_palindrome (Finset.range 100), if steps_to_palindrome n = 6 then n else 0) = 176 :=
by sorry

end sum_of_non_palindrome_six_steps_to_palindrome_l385_385911


namespace gift_distribution_count_l385_385889

theorem gift_distribution_count :
  ∃ (n : ℕ), n = (25 * 24 * 23 * 22) :=
by 
  existsi (25 * 24 * 23 * 22)
  simp only [mul_assoc]
  sorry

end gift_distribution_count_l385_385889


namespace convex_polygon_sides_l385_385225

theorem convex_polygon_sides (n : ℕ) (h1 : ∑ i in range n, a i = 2970 + a k) (h2 : a k < 180) :
  n = 19 :=
sorry

end convex_polygon_sides_l385_385225


namespace integer_points_count_l385_385421

theorem integer_points_count :
  ∃ (n : ℤ), n = 9 ∧
  ∀ a b : ℝ, (1 < a) → (1 < b) → (ab + a - b - 10 = 0) →
  (a + b = 6) → 
  ∃ (x y : ℤ), (3 * x^2 + 2 * y^2 ≤ 6) :=
by
  sorry

end integer_points_count_l385_385421


namespace range_of_a_extremum_ratio_l385_385918

noncomputable def has_extremum (f : ℝ → ℝ) : Prop := 
  ∃ a : ℝ, f = λ x, x^2 + a * Real.log (x + 1) ∧ 
            (∃ x : ℝ, ∃ y : ℝ, x ≠ y ∧ x ∈ (-1 : ℝ, ∞) ∧ y ∈ (-1 : ℝ, ∞) ∧ 
              (∀ z : ℝ, z ∈ (-1 : ℝ, ∞) → x ≠ z → y ≠ z → (f x > f z ∧ f y > f z)))

theorem range_of_a {f : ℝ → ℝ} (h : has_extremum f) : 
  ∃ a : ℝ, 0 < a ∧ a < 1/2 :=
sorry

theorem extremum_ratio {a x1 x2 : ℝ} 
  (hx1 : x1 = 1/2 - Real.sqrt (1 - 2 * a) / 2) 
  (hx2 : x2 = -1/2 + Real.sqrt (1 - 2 * a) / 2)
  (hx2_cond : -1/2 < x2 ∧ x2 < 0)
  (h : f = λ x, x^2 + a * Real.log (x + 1)) : 
  0 < (f x2 / x1) ∧ (f x2 / x1) < -1/2 + Real.log 2 :=
sorry

end range_of_a_extremum_ratio_l385_385918


namespace arithmetic_sequence_third_term_l385_385227

theorem arithmetic_sequence_third_term {a d : ℝ} (h : 2 * a + 4 * d = 10) : a + 2 * d = 5 :=
sorry

end arithmetic_sequence_third_term_l385_385227


namespace sum_first_10_terms_arithmetic_sequence_l385_385266

-- Define the first term and the sum of the second and sixth terms as given conditions
def a1 : ℤ := -2
def condition_a2_a6 (a2 a6 : ℤ) : Prop := a2 + a6 = 2

-- Define the general term 'a_n' of the arithmetic sequence
def a_n (a1 d : ℤ) (n : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum 'S_n' of the first 'n' terms of the arithmetic sequence
def S_n (a1 d : ℤ) (n : ℤ) : ℤ := n * (a1 + ((n - 1) * d) / 2)

-- The theorem statement to prove that S_10 = 25 given the conditions
theorem sum_first_10_terms_arithmetic_sequence 
  (d a2 a6 : ℤ) 
  (h1 : a2 = a_n a1 d 2) 
  (h2 : a6 = a_n a1 d 6)
  (h3 : condition_a2_a6 a2 a6) : 
  S_n a1 d 10 = 25 := 
by
  sorry

end sum_first_10_terms_arithmetic_sequence_l385_385266


namespace grinder_price_l385_385256

variable (G : ℝ) (PurchasedMobile : ℝ) (SoldMobile : ℝ) (overallProfit : ℝ)

theorem grinder_price (h1 : PurchasedMobile = 10000)
                      (h2 : SoldMobile = 11000)
                      (h3 : overallProfit = 400)
                      (h4 : 0.96 * G + SoldMobile = G + PurchasedMobile + overallProfit) :
                      G = 15000 := by
  sorry

end grinder_price_l385_385256


namespace find_base_l385_385237

def distinct_three_digit_numbers (b : ℕ) : ℕ :=
    (b - 2) * (b - 3) + (b - 1) * (b - 3) + (b - 1) * (b - 2)

theorem find_base (b : ℕ) (h : distinct_three_digit_numbers b = 144) : b = 9 :=
by 
  sorry

end find_base_l385_385237


namespace least_distance_fly_crawls_on_cone_l385_385031

theorem least_distance_fly_crawls_on_cone
  (r : ℝ) (h : ℝ) (dist_start : ℝ) (dist_end : ℝ)
  (distance_fly : ℝ) :
  r = 800 →
  h = 300 * real.sqrt 7 →
  dist_start = 200 →
  dist_end = 450 * real.sqrt 2 →
  distance_fly ≈ 790.57 :=
by
  sorry

end least_distance_fly_crawls_on_cone_l385_385031


namespace power_sum_mod_l385_385722

/-- The remainder of the sum of powers of 5 from 0 to 15 divided by 7 is 2. -/
theorem power_sum_mod (n : ℕ) (h : n = 15) : 
  (∑ i in finset.range (n + 1), 5^i) % 7 = 2 :=
by 
  sorry

end power_sum_mod_l385_385722


namespace B_plus_C_is_330_l385_385440

-- Definitions
def A : ℕ := 170
def B : ℕ := 300
def C : ℕ := 30

axiom h1 : A + B + C = 500
axiom h2 : A + C = 200
axiom h3 : C = 30

-- Theorem statement
theorem B_plus_C_is_330 : B + C = 330 :=
by
  sorry

end B_plus_C_is_330_l385_385440


namespace range_of_a_l385_385286

variable {a : ℝ}

def p (a : ℝ) : Prop := ∀ x : ℝ, x < a → 2 * (x - 2) ≤ 0
def q (a : ℝ) : Prop := 7 - 2 * a > 1

theorem range_of_a (h₁ : p a ∨ q a) (h₂ : ¬(p a ∧ q a)) : 2 < a ∧ a < 3 :=
by
  split
  · sorry
  · sorry

end range_of_a_l385_385286


namespace polygon_sides_l385_385699

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 + 360 = 1260 → n = 7 :=
by 
  intro h,
  sorry -- The proof steps are skipped for now

end polygon_sides_l385_385699


namespace vasya_max_earning_l385_385356

theorem vasya_max_earning (k : ℕ) (h₀: k ≤ 2013) (h₁: 2013 - 2*k % 11 = 0) : k % 11 = 0 → (k ≤ 5) := 
by
  sorry

end vasya_max_earning_l385_385356


namespace total_copies_to_save_40_each_l385_385666

-- Definitions for the conditions.
def cost_per_copy : ℝ := 0.02
def discount_rate : ℝ := 0.25
def min_copies_for_discount : ℕ := 100
def savings_required : ℝ := 0.40
def steve_copies : ℕ := 80
def dinley_copies : ℕ := 80

-- Lean 4 statement to prove the total number of copies 
-- to save $0.40 each.
theorem total_copies_to_save_40_each : 
  (steve_copies + dinley_copies) + 
  (savings_required / (cost_per_copy * discount_rate)) * 2 = 320 :=
by 
  sorry

end total_copies_to_save_40_each_l385_385666


namespace max_value_l385_385120

noncomputable def max_value_cos_2x_plus_2sin_x (x : ℝ) : ℝ :=
  cos (2 * x) + 2 * sin x

theorem max_value (h : - (Real.pi / 2) ≤ x ∧ x ≤ Real.pi / 2) :
  max_value_cos_2x_plus_2sin_x x ≤ 3 / 2 :=
sorry

end max_value_l385_385120


namespace find_x_given_sin_interval_l385_385209

open Real

theorem find_x_given_sin_interval (x : ℝ) (h1 : sin x = -3 / 5) (h2 : π < x ∧ x < 3 / 2 * π) :
  x = π + arcsin (3 / 5) :=
sorry

end find_x_given_sin_interval_l385_385209


namespace minimum_distance_value_l385_385162

-- Definitions of the points and the parabola
def F : ℝ × ℝ := (0, 4)
def P : ℝ × ℝ := (2, 3)
def parabola (p : ℝ) (h : p > 0) : set (ℝ × ℝ) := {M | M.1^2 = 2 * p * M.2}

-- The conditions
axiom hF : ∃ p : ℝ, p > 0 ∧ parabola p hF.contains F
axiom hP : P = (2, 3)

-- The theorem to be proven
theorem minimum_distance_value (M : ℝ × ℝ) :
  (∃ M ∈ parabola (classical.some hF.fst) (classical.some_spec hF.fst).1, 
  M.1^2 = 16 * M.2) →
    |dist M F| + |dist M P| = 7 :=
sorry

end minimum_distance_value_l385_385162


namespace sum_first_10_terms_arithmetic_sequence_l385_385262

-- Define the first term and the sum of the second and sixth terms as given conditions
def a1 : ℤ := -2
def condition_a2_a6 (a2 a6 : ℤ) : Prop := a2 + a6 = 2

-- Define the general term 'a_n' of the arithmetic sequence
def a_n (a1 d : ℤ) (n : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum 'S_n' of the first 'n' terms of the arithmetic sequence
def S_n (a1 d : ℤ) (n : ℤ) : ℤ := n * (a1 + ((n - 1) * d) / 2)

-- The theorem statement to prove that S_10 = 25 given the conditions
theorem sum_first_10_terms_arithmetic_sequence 
  (d a2 a6 : ℤ) 
  (h1 : a2 = a_n a1 d 2) 
  (h2 : a6 = a_n a1 d 6)
  (h3 : condition_a2_a6 a2 a6) : 
  S_n a1 d 10 = 25 := 
by
  sorry

end sum_first_10_terms_arithmetic_sequence_l385_385262


namespace monotonicity_intervals_and_min_m_l385_385549

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * real.log x + (2 * x + 1) / x

noncomputable def tangent_slope (a : ℝ) : ℝ := (2 * a - 2) / 4

noncomputable def g (x : ℝ) : ℝ := (x * real.log x + 2 * x - 2) / (x - 1)

theorem monotonicity_intervals_and_min_m (a : ℝ) (m : ℤ) :
  (tangent_slope a = 1/4) →
  (∀ x, (x > 0 ∧ x ≤ 2) → deriv (f 1) x ≤ 0) ∧ 
  (∀ x, (x ≥ 2) → deriv (f 1) x ≥ 0) ∧ 
  (∃ x, x ∈ set.Ioi 1 ∧ (f 1 x < (m * (x - 1) + 2) / x)) →
  m ≥ 5 :=
begin
  intros h1 h2,
  sorry
end

end monotonicity_intervals_and_min_m_l385_385549


namespace geometric_sequence_S_n_general_formula_a_n_maximum_value_k_l385_385922

-- Define the sequence and the conditions
def a : ℕ → ℝ
def S : ℕ → ℝ
axiom h1 : a 1 = 1
axiom h2 : ∀ n : ℕ, (S (n + 1) + 3 * S n) / 2 = -3 / 2

-- Proof problem 1: Arithmetic mean condition implies geometric sequence
theorem geometric_sequence_S_n : ∀ n : ℕ, S (n + 1) - 3 / 2 = (S n - 3 / 2) * 1 / 3 :=
by sorry

-- Proof problem 2: General formula for sequence {a_n}
theorem general_formula_a_n : ∀ n : ℕ, a n = 1 / 3^(n - 1) :=
by sorry

-- Proof problem 3: Maximum value of k such that k ≤ S_n for all n
theorem maximum_value_k : ∀ k : ℝ, (∀ n : ℕ, k ≤ S n) ↔ k ≤ 1 :=
by sorry

end geometric_sequence_S_n_general_formula_a_n_maximum_value_k_l385_385922


namespace find_b_and_c_l385_385576

variable (f : ℝ → ℝ)
variable (a b c : ℝ)

axiom f_def : ∀ x : ℝ, x ≠ 0 → f x = -1 / x
axiom f_a : f a = -1 / 3
axiom f_ab : f (a * b) = 1 / 6
axiom f_c : f c = sin c / cos c
axiom a_val : a = 3

theorem find_b_and_c : b = -2 ∧ tan c = -1 / c :=
by
  have b_val : b = -2 := 
    by
      have h1 : f (3 * b) = 1 / 6 := 
        by rw [← a_val]; exact f_ab
      have h2 : f (3 * b) = -1 / (3 * b) := by 
        apply f_def; linarith
      rw [h2] at h1
      sorry
  have tan_c_val : tan c = -1 / c := 
    by
      sorry
  exact ⟨b_val, tan_c_val⟩

end find_b_and_c_l385_385576


namespace Marty_paint_combinations_l385_385303

theorem Marty_paint_combinations 
  (colors : Fin 5) 
  (methods : Fin 4) : 
  (Fin 5).card * (Fin 4).card = 20 := 
by
  sorry

end Marty_paint_combinations_l385_385303


namespace probability_of_matching_correctly_l385_385447

-- Define the number of plants and seedlings.
def num_plants : ℕ := 4

-- Define the number of total arrangements.
def total_arrangements : ℕ := Nat.factorial num_plants

-- Define the number of correct arrangements.
def correct_arrangements : ℕ := 1

-- Define the probability of a correct guess.
def probability_of_correct_guess : ℚ := correct_arrangements / total_arrangements

-- The problem requires to prove that the probability of correct guess is 1/24
theorem probability_of_matching_correctly :
  probability_of_correct_guess = 1 / 24 :=
  by
    sorry

end probability_of_matching_correctly_l385_385447


namespace shaded_region_area_l385_385590

theorem shaded_region_area (radius : ℝ) (P Q R S T U O : Point)
  (h_circle : Circle O radius)
  (h_diameters : PQ = RS ∧ RS = TU)
  (h_perpendicular : perp PQ RS)
  (h_angle : ∠POQ = 90° ∧ ∠POT = 45°) : 
  area_shaded_region O PQ RS TU (radius := 6) = 36 + 18 * Real.pi := by
  sorry

end shaded_region_area_l385_385590


namespace count_irrational_numbers_l385_385804

def is_irrational (x : ℝ) : Prop := ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

def set_of_real_numbers : Set ℝ := 
  {π / 2, Real.sqrt 3, Real.sqrt 4, 22 / 7, - (1 + ((0.01 + 0.001 + 0.0001) / 3))}

theorem count_irrational_numbers : Finset.card 
  {x ∈ set_of_real_numbers | is_irrational x} = 3 :=
sorry

end count_irrational_numbers_l385_385804


namespace arithmetic_sequence_third_term_l385_385228

theorem arithmetic_sequence_third_term {a d : ℝ} (h : 2 * a + 4 * d = 10) : a + 2 * d = 5 :=
sorry

end arithmetic_sequence_third_term_l385_385228


namespace values_of_a_l385_385912

noncomputable def quadratic_eq (a x : ℝ) : ℝ :=
(a - 1) * x^2 - 2 * (a + 1) * x + 2 * (a + 1)

theorem values_of_a (a : ℝ) :
  (∀ x : ℝ, quadratic_eq a x = 0 → x ≥ 0) ↔ (a = 3 ∨ (-1 ≤ a ∧ a ≤ 1)) :=
sorry

end values_of_a_l385_385912


namespace f_2016_is_1_l385_385524

noncomputable def f : ℤ → ℤ := sorry

axiom h1 : f 1 = 1
axiom h2 : f 2015 ≠ 1
axiom h3 : ∀ a b : ℤ, f (a + b) ≤ max (f a) (f b)
axiom h4 : ∀ x : ℤ, f x = f (-x)

theorem f_2016_is_1 : f 2016 = 1 := 
by 
  sorry

end f_2016_is_1_l385_385524


namespace transformation_sin_function_l385_385552

theorem transformation_sin_function :
  ∀ x : ℝ, (λ x, (sin (2 * (x - π / 3)))) x = (λ x, (sin (2 * x - 2 * π / 3))) x := 
by 
  sorry

end transformation_sin_function_l385_385552


namespace other_continent_passengers_l385_385992

noncomputable def totalPassengers := 240
noncomputable def northAmericaFraction := (1 / 3 : ℝ)
noncomputable def europeFraction := (1 / 8 : ℝ)
noncomputable def africaFraction := (1 / 5 : ℝ)
noncomputable def asiaFraction := (1 / 6 : ℝ)

theorem other_continent_passengers :
  (totalPassengers : ℝ) - (totalPassengers * northAmericaFraction +
                           totalPassengers * europeFraction +
                           totalPassengers * africaFraction +
                           totalPassengers * asiaFraction) = 42 :=
by
  sorry

end other_continent_passengers_l385_385992


namespace tangent_line_eqn_max_interval_monotonic_increase_l385_385177

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x + a * x - x^2

theorem tangent_line_eqn (a : ℝ) (h_a : 0 < a ∧ a ≤ 1) :
  tangent_line (f a) 1 = -1 / 2 * x := sorry

theorem max_interval_monotonic_increase (a : ℝ) (h_a : 0 < a ∧ a ≤ 1) :
  let t := (a + sqrt (a^2 + 8)) / 4
  in 0 < t ∧ t = 1 ↔ a = 1 := sorry

end tangent_line_eqn_max_interval_monotonic_increase_l385_385177


namespace good_number_2013_51st_l385_385226

-- Definition of a good number
def is_good_number (n : ℕ) : Prop :=
  (n.digits.sum = 6)

-- Ascending sequence of good numbers
def good_numbers : List ℕ :=
  List.filter is_good_number (List.range 10000) -- considering numbers up to 9999 for completeness

-- The n-th good number in the sequence (1-based index)
def an (n : ℕ) : ℕ :=
  good_numbers.get! (n - 1) -- convert 1-based to 0-based index

-- Theorem to prove
theorem good_number_2013_51st : ∃ n, an n = 2013 ∧ n = 51 := by
  sorry

end good_number_2013_51st_l385_385226


namespace cross_section_volume_ratio_l385_385807

-- Define the structure of the cube
structure Cube :=
  (A B C D E F G H: Point)

-- Define the condition of the cross section
structure CrossSection :=
  (K : Point)
  (on_edge_EF : K ∈ segment(E, F))

-- Define the volume ratio condition
def VolumeRatioCondition (cube : Cube) (cs : CrossSection) : Prop :=
  let vol1 := volume (cross_section_through_A_C_K cube cs.K)
  let vol2 := volume (cube) - vol1
  vol1 = 3 * vol2

-- Assume existence of functions to define point, segment, and volume calculations
axiom Point : Type
axiom E F : Point
axiom segment : Point → Point → Set Point
axiom cross_section_through_A_C_K : Cube → Point → Set Point
axiom volume : Set Point → Real

theorem cross_section_volume_ratio (cube : Cube) (cs : CrossSection) 
    (vol_ratio_cond : VolumeRatioCondition cube cs) :
    ∃ EK KF, (EK / KF) = sqrt(3) := sorry

end cross_section_volume_ratio_l385_385807


namespace five_digit_wave_number_count_l385_385419

-- Define what it means to be a "wave number"
def is_wave_number (n : ℕ) : Prop :=
  let d1 := (n / 10000) % 10 in
  let d2 := (n / 1000) % 10 in
  let d3 := (n / 100) % 10 in
  let d4 := (n / 10) % 10 in
  let d5 := n % 10 in
  (d2 > d1 ∧ d2 > d3 ∧ d4 > d3 ∧ d4 > d5)

-- Define the allowed digits
def valid_digits (n : ℕ) : Prop :=
  let digits := [0, 1, 2, 3, 4, 5, 6, 7] in
  let d1 := (n / 10000) % 10 in
  let d2 := (n / 1000) % 10 in
  let d3 := (n / 100) % 10 in
  let d4 := (n / 10) % 10 in
  let d5 := n % 10 in
  d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧ d4 ∈ digits ∧ d5 ∈ digits

-- Define the range for five-digit numbers
def in_five_digit_range (n : ℕ) : Prop :=
  n >= 10000 ∧ n < 100000

-- Define the primary condition for five-digit wave numbers
def is_valid_five_digit_wave_number (n : ℕ) : Prop :=
  is_wave_number n ∧ valid_digits n ∧ in_five_digit_range n

-- State the primary problem
theorem five_digit_wave_number_count :
  {n : ℕ | is_valid_five_digit_wave_number n}.card = 721 :=
sorry

end five_digit_wave_number_count_l385_385419


namespace root_count_sqrt_eq_l385_385477

open Real

theorem root_count_sqrt_eq (x : ℝ) :
  (∀ y, (y = sqrt (7 - 2 * x)) → y = x * y → (∃ x, x = 7 / 2 ∨ x = 1)) ∧
  (7 - 2 * x ≥ 0) →
  ∃ s, s = 1 ∧ (7 - 2 * s = 0) → x = 1 ∨ x = 7 / 2 :=
sorry

end root_count_sqrt_eq_l385_385477


namespace sum_first_10_terms_arithmetic_sequence_l385_385263

-- Define the first term and the sum of the second and sixth terms as given conditions
def a1 : ℤ := -2
def condition_a2_a6 (a2 a6 : ℤ) : Prop := a2 + a6 = 2

-- Define the general term 'a_n' of the arithmetic sequence
def a_n (a1 d : ℤ) (n : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum 'S_n' of the first 'n' terms of the arithmetic sequence
def S_n (a1 d : ℤ) (n : ℤ) : ℤ := n * (a1 + ((n - 1) * d) / 2)

-- The theorem statement to prove that S_10 = 25 given the conditions
theorem sum_first_10_terms_arithmetic_sequence 
  (d a2 a6 : ℤ) 
  (h1 : a2 = a_n a1 d 2) 
  (h2 : a6 = a_n a1 d 6)
  (h3 : condition_a2_a6 a2 a6) : 
  S_n a1 d 10 = 25 := 
by
  sorry

end sum_first_10_terms_arithmetic_sequence_l385_385263


namespace points_can_move_on_same_line_l385_385623

variable {A B C x y x' y' : ℝ}

def transform_x (x y : ℝ) : ℝ := 3 * x + 2 * y + 1
def transform_y (x y : ℝ) : ℝ := x + 4 * y - 3

noncomputable def points_on_same_line (A B C : ℝ) (x y : ℝ) : Prop :=
  A*x + B*y + C = 0 ∧
  A*(transform_x x y) + B*(transform_y x y) + C = 0

theorem points_can_move_on_same_line :
  ∃ (A B C : ℝ), ∀ (x y : ℝ), points_on_same_line A B C x y :=
sorry

end points_can_move_on_same_line_l385_385623


namespace sin_double_angle_l385_385530

theorem sin_double_angle
  (α : ℝ) (h1 : Real.sin (3 * Real.pi / 2 - α) = 3 / 5) (h2 : α ∈ Set.Ioo Real.pi (3 * Real.pi / 2)) :
  Real.sin (2 * α) = 24 / 25 :=
sorry

end sin_double_angle_l385_385530


namespace no_solutions_abs_eq_quadratic_l385_385114

theorem no_solutions_abs_eq_quadratic (x : ℝ) : ¬ (|x - 3| = x^2 + 2 * x + 4) := 
by
  sorry

end no_solutions_abs_eq_quadratic_l385_385114


namespace integer_solutions_count_l385_385346

theorem integer_solutions_count :
  let S := {x : ℝ | (x - 1) < (x - 1)^2 ∧ (x - 1)^2 < 3 * x + 7} in
  (S ∩ (Set.Icc 0.0 0.0) ∪ S ∩ (Set.Icc 3.0 5.0)).card = 4 :=
by
  sorry

end integer_solutions_count_l385_385346


namespace find_p_q_r_l385_385927

theorem find_p_q_r  (t : ℝ) (p q r : ℕ) (h1 : (1 + Real.sin t) * (1 + Real.cos t) = 9 / 4) 
                    (h2 : (1 - Real.sin t) * (1 - Real.cos t) = (p / q) - Real.sqrt r)
                    (pos_p : 0 < p) (pos_q : 0 < q) (pos_r : 0 < r) 
                    (rel_prime : Nat.gcd p q = 1) : 
                    p + q + r = 5 := 
by
  sorry

end find_p_q_r_l385_385927


namespace number_of_valid_stacking_sequences_l385_385320

-- Definitions based on the conditions
def red_cards : list ℕ := [1, 2, 3, 4]
def blue_cards : list ℕ := [2, 3, 4]
def green_cards : list ℕ := [5, 6, 7]

def alternating_colors (stack : list (ℕ × string)) : Prop :=
  ∀ i, 
    i < stack.length - 1 → (stack[i].2 ≠ stack[i + 1].2)

def divides_neighbouring (stack : list (ℕ × string)) : Prop :=
  ∀ i, 
    i < stack.length - 1 → (stack[i].1 % stack[i + 1].1 = 0 ∨ stack[i + 1].1 % stack[i].1 = 0)

-- We use length to check the final count of the valid sequences
theorem number_of_valid_stacking_sequences :
  ∃ (stack : list (ℕ × string)), 
    list.length stack = 10 ∧ alternating_colors stack ∧ divides_neighbouring stack := sorry

end number_of_valid_stacking_sequences_l385_385320


namespace smallest_prime_20_less_than_square_l385_385373

open Nat

theorem smallest_prime_20_less_than_square : ∃ (p : ℕ), Prime p ∧ (∃ (n : ℕ), p = n^2 - 20) ∧ p = 5 := by
  sorry

end smallest_prime_20_less_than_square_l385_385373


namespace determine_n_l385_385478

noncomputable def f (n : ℝ) (x : ℝ) : ℝ := x^n + 3^x + 2 * x

noncomputable def f_prime (n : ℝ) (x : ℝ) : ℝ := 
  n * x^(n-1) + 3^x * Real.log 3 + 2

theorem determine_n :
  (f_prime n 1 = 3 + 3 * Real.log 3) → n = 1 :=
by
  intros h
  -- equation simplification
  have : n + 3 * Real.log 3 + 2 = 3 + 3 * Real.log 3
  exact h
  -- solve for n
  sorry

end determine_n_l385_385478


namespace factorial_fraction_l385_385063

theorem factorial_fraction : (fact 8 + fact 9) / fact 7 = 80 :=
by
  sorry

end factorial_fraction_l385_385063


namespace largest_value_l385_385995

theorem largest_value (x : ℝ) (h : x = 1/4) :
  (∀ y ∈ {(x, x^2, 1/2 * x, 1/x, real.sqrt x)}, y ≤ 1/x) :=
by {
  intro y,
  intro h_y,
  cases h_y; simp [h],
  sorry
}

end largest_value_l385_385995


namespace problem_statement_l385_385243

noncomputable def polar_equation_C1 (θ : ℝ) : Prop :=
  ∃ ρ : ℝ, ρ * sin (θ + π/4) = (sqrt 2) / 2

noncomputable def polar_equation_C2 (θ : ℝ) : Prop :=
  ∃ ρ : ℝ, ρ = 4 * cos θ

theorem problem_statement (α : ℝ) (hα : 0 ≤ α ∧ α ≤ π/2) :
  let ρA := 1 / (cos α + sin α),
      ρB := 4 * cos α,
      ratio := ρB / ρA
  in ratio ≤ 2 + 2 * sqrt 2 :=
begin
  sorry,
end

end problem_statement_l385_385243


namespace difference_of_squares_is_39_l385_385670

theorem difference_of_squares_is_39 (L S : ℕ) (h1 : L = 8) (h2 : L - S = 3) : L^2 - S^2 = 39 :=
by
  sorry

end difference_of_squares_is_39_l385_385670


namespace AJ_stamps_l385_385459

theorem AJ_stamps (A : ℕ)
  (KJ := A / 2)
  (CJ := 2 * KJ + 5)
  (BJ := 3 * A - 3)
  (total_stamps := A + KJ + CJ + BJ)
  (h : total_stamps = 1472) :
  A = 267 :=
  sorry

end AJ_stamps_l385_385459


namespace line_intersects_circle_shortest_chord_line_eq_l385_385170

-- Define the line l and circle C conditions
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1 - 2 * k
def circle (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 2 * y - 7 = 0

-- 1. Prove line l and circle C intersect for all k
theorem line_intersects_circle (k : ℝ) : 
  ∃ (x y : ℝ), line k x y ∧ circle x y := 
sorry

-- 2. Prove the equation of line l when the chord length is the shortest
theorem shortest_chord_line_eq : 
  ∃ (k : ℝ), k = -1/2 ∧ (∀ x y : ℝ, line k x y ↔ x + 2 * y - 4 = 0) :=
begin
  use -1 / 2,
  split,
  { refl },
  { intros x y,
    split,
    { intro h,
      have : y = (-1 / 2) * x + 1 + 1,
      { rw [line, h] },
      simp at this,
      linarith, },
    { intro h,
      simp,
      norm_num at h,
      linarith, } }
end

end line_intersects_circle_shortest_chord_line_eq_l385_385170


namespace binary_to_octal_101110_l385_385071

theorem binary_to_octal_101110 :
  binary_to_octal "101110" = "56" :=
sorry

end binary_to_octal_101110_l385_385071


namespace check_conditions_l385_385182

noncomputable def f (x a b : ℝ) : ℝ := |x^2 - 2 * a * x + b|

theorem check_conditions (a b : ℝ) :
  ¬ (∀ x : ℝ, f x a b = f (-x) a b) ∧         -- f(x) is not necessarily an even function
  ¬ (∀ x : ℝ, (f 0 a b = f 2 a b → (f x a b = f (2 - x) a b))) ∧ -- No guaranteed symmetry about x=1
  (a^2 - b^2 ≤ 0 → ∀ x : ℝ, x ≥ a → ∀ y : ℝ, y ≥ x → f y a b ≥ f x a b) ∧ -- f(x) is increasing on [a, +∞) if a^2 - b^2 ≤ 0
  ¬ (∀ x : ℝ, f x a b ≤ |a^2 - b|)         -- f(x) does not necessarily have a max value of |a^2 - b|
:= sorry

end check_conditions_l385_385182


namespace min_value_of_sum_squares_l385_385291

theorem min_value_of_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) :
    x^2 + y^2 + z^2 ≥ 10 :=
sorry

end min_value_of_sum_squares_l385_385291


namespace atm_withdrawal_cost_l385_385667

theorem atm_withdrawal_cost (x y : ℝ)
  (h1 : 221 = x + 40000 * y)
  (h2 : 485 = x + 100000 * y) :
  (x + 85000 * y) = 419 := by
  sorry

end atm_withdrawal_cost_l385_385667


namespace other_friends_meal_distribution_l385_385647

-- Conditions
def total_pizza_slices : ℕ := 5 * 8
def ron_pizza_slices : ℕ := 5
def scott_pizza_slices : ℕ := 5
def mark_pizza_slices : ℕ := 2
def sam_pizza_slices : ℕ := 4

def pasta_total_percent : ℕ := 100
def pasta_ron_scott_percent : ℕ := 40
def friends_count_sharing_pasta : ℕ := 6

def garlic_bread_total : ℕ := 12
def garlic_bread_ron_scott_mark_percent : ℕ := 25
def friends_count_sharing_garlic_bread : ℕ := 5

-- Proof problem
theorem other_friends_meal_distribution :
  let remaining_pizza_slices := total_pizza_slices - (ron_pizza_slices + scott_pizza_slices + mark_pizza_slices + sam_pizza_slices),
      other_friends_pizza_slices := remaining_pizza_slices / 4,
      each_friend_pasta_percent := (pasta_total_percent - pasta_ron_scott_percent) / friends_count_sharing_pasta,
      remaining_garlic_breads := garlic_bread_total * (100 - garlic_bread_ron_scott_mark_percent) / 100,
      each_friend_garlic_breads := remaining_garlic_breads / friends_count_sharing_garlic_bread
  in 
  other_friends_pizza_slices = 6 ∧ each_friend_pasta_percent = 10 ∧ each_friend_garlic_breads = 1.8 :=
by
  let remaining_pizza_slices := 40 - (5 + 5 + 2 + 4),
      other_friends_pizza_slices := remaining_pizza_slices / 4,
      each_friend_pasta_percent := (100 - 40) / 6,
      remaining_garlic_breads := 12 * (100 - 25) / 100,
      each_friend_garlic_breads := remaining_garlic_breads / 5
  show other_friends_pizza_slices = 6 ∧ each_friend_pasta_percent = 10 ∧ each_friend_garlic_breads = 1.8 from sorry

end other_friends_meal_distribution_l385_385647


namespace thirteen_members_divisible_by_13_l385_385360

theorem thirteen_members_divisible_by_13 (B : ℕ) (hB : B < 10) : 
  (∃ B, (2000 + B * 100 + 34) % 13 = 0) ↔ B = 6 :=
by
  sorry

end thirteen_members_divisible_by_13_l385_385360


namespace workerA_time_to_complete_job_l385_385387

theorem workerA_time_to_complete_job :
  ∃ A : ℝ, A = 12 ∧ (1 / A + 1 / 15 = 3 / 20) :=
by
  construct unknown A that satisfies the above, according to the conditions
  -- Assuming from conditions, including the combined work rate calculation.
  -- We know 1/A + 1/15 is supposed to equal to the combined work rate
  sorry

end workerA_time_to_complete_job_l385_385387


namespace sequence_a_n_general_formula_max_m_such_that_T_n_gt_m_div_21_l385_385131

noncomputable def f (x : ℝ) : ℝ := (3 * x^2) - (2 * x)

noncomputable def S (n : ℕ) : ℝ := 3 * (n^2 : ℝ) - 2 * (n : ℝ)

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 1
  else 6 * (n : ℝ) - 5

noncomputable def b (n : ℕ) : ℝ :=
  3 / (a n * a (n + 1))

noncomputable def T (n : ℕ) : ℝ :=
  1 / 2 - 1 / (2 * (6 * n + 1 : ℝ))

theorem sequence_a_n_general_formula (n : ℕ) (n_pos : n > 0) : 
  a n = if n = 1 then 1 else 6 * (n : ℝ) - 5 := 
by
  sorry

theorem max_m_such_that_T_n_gt_m_div_21 (∀ (n : ℕ), n > 0 → T n > (8 / 21)) : 
  ∃ (m : ℤ), m = 8 := 
by
  sorry

end sequence_a_n_general_formula_max_m_such_that_T_n_gt_m_div_21_l385_385131


namespace find_income_l385_385679

noncomputable def income_expenditure_proof : Prop := 
  ∃ (x : ℕ), (5 * x - 4 * x = 3600) ∧ (5 * x = 18000)

theorem find_income : income_expenditure_proof :=
  sorry

end find_income_l385_385679


namespace sum_first_10_terms_arithmetic_seq_l385_385274

theorem sum_first_10_terms_arithmetic_seq :
  ∀ (a : ℕ → ℤ), (a 1 = -2) → (a 2 + a 6 = 2) →
  (∃ S₁₀, S₁₀ = 10 * a 1 + (10 * (10 - 1)) / 2 * (a 2 - a 1) / (2 - 1) ∧ S₁₀ = 25) :=
begin
  assume a,
  assume h1 : a 1 = -2,
  assume h2 : a 2 + a 6 = 2,
  sorry
end

end sum_first_10_terms_arithmetic_seq_l385_385274


namespace dodecahedron_diagonals_l385_385055

-- Define a structure representing a dodecahedron with its properties
structure Dodecahedron where
  faces : Nat
  vertices : Nat
  faces_meeting_at_each_vertex : Nat

-- Concretely define a dodecahedron based on the given problem properties
def dodecahedron_example : Dodecahedron :=
  { faces := 12,
    vertices := 20,
    faces_meeting_at_each_vertex := 3 }

-- Lean statement to prove the number of interior diagonals in a dodecahedron
theorem dodecahedron_diagonals (d : Dodecahedron) (h : d = dodecahedron_example) : 
  (d.vertices * (d.vertices - d.faces_meeting_at_each_vertex) / 2) = 160 := by
  rw [h]
  -- Even though we skip the proof, Lean should recognize the transformation
  sorry

end dodecahedron_diagonals_l385_385055


namespace total_math_and_biology_homework_l385_385644

-- Definitions
def math_homework_pages : ℕ := 8
def biology_homework_pages : ℕ := 3

-- Theorem stating the problem to prove
theorem total_math_and_biology_homework :
  math_homework_pages + biology_homework_pages = 11 :=
by
  sorry

end total_math_and_biology_homework_l385_385644


namespace rotation_preserves_line_equation_l385_385783

noncomputable theory
open_locale classical

-- Defining the given conditions
def original_line : set (ℝ × ℝ) := {p | p.1 - p.2 + 1 = 0}
def P : ℝ × ℝ := (3, 4)

-- Define the line resulting from a 90° counterclockwise rotation of the original line around point P
def rotated_line : set (ℝ × ℝ) := {p | p.1 + p.2 - 7 = 0}

-- State the theorem to prove
theorem rotation_preserves_line_equation :
  (∀ p ∈ original_line, p.1 = 3 → p = P) →
  (∀ p, p ∈ rotated_line ↔ p.1 + p.2 - 7 = 0) :=
begin
  sorry
end

end rotation_preserves_line_equation_l385_385783


namespace triangle_side_a_eq_sqrt_13_sequence_general_term_formula_geometric_sequence_sum_obtuse_triangle_side_range_l385_385744

-- Equivalent Problem 1
theorem triangle_side_a_eq_sqrt_13 (A : ℝ) (b : ℝ) (area : ℝ) (a : ℝ) :
  A = real.pi / 3 ∧ b = 1 ∧ area = real.sqrt 3 → a = real.sqrt 13 := by
  sorry

-- Equivalent Problem 2
theorem sequence_general_term_formula (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) :
  (∀ n, S_n n = 3 * n ^ 2 - 2 * n) →
  (∀ n, a_n n = S_n n - S_n (n - 1)) ∧ a_n 1 = S_n 1 → 
  ∀ n, a_n n = 6 * n - 5 := by
  sorry

-- Equivalent Problem 3
theorem geometric_sequence_sum (S_n S_2n : ℝ) (S_3n : ℝ) :
  S_n = 48 ∧ S_2n = 60 →
  S_3n = 63 := by
  sorry

-- Equivalent Problem 4
theorem obtuse_triangle_side_range (a b c : ℝ) :
  a = 1 ∧ b = 2 ∧ c > real.sqrt 5 ∧ c < 3 → 
  true := by
  sorry

end triangle_side_a_eq_sqrt_13_sequence_general_term_formula_geometric_sequence_sum_obtuse_triangle_side_range_l385_385744


namespace find_p_value_l385_385948

noncomputable def solve_p (m p : ℕ) :=
  (1^m / 5^m) * (1^16 / 4^16) = 1 / (2 * p^31)

theorem find_p_value (m p : ℕ) (hm : m = 31) :
  solve_p m p ↔ p = 10 :=
by
  sorry

end find_p_value_l385_385948


namespace cos_210_eq_neg_sqrt3_div_2_l385_385872

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385872


namespace C_pays_correct_amount_l385_385791

variable (P_A : ℝ) (r_A : ℝ) (r_B : ℝ)

def selling_price_A_to_B (P_A r_A : ℝ) : ℝ := P_A * (1 + r_A)

def selling_price_B_to_C (P_B r_B : ℝ) : ℝ := P_B * (1 + r_B)

theorem C_pays_correct_amount (P_A : ℝ) (r_A : ℝ) (r_B : ℝ) (P_C : ℝ) 
    (h1 : P_A = 150) (h2 : r_A = 0.20) (h3 : r_B = 0.25) : 
    selling_price_B_to_C (selling_price_A_to_B P_A r_A) r_B = P_C := 
begin
    sorry
end

#check C_pays_correct_amount 150 0.20 0.25 225 sorry sorry sorry

end C_pays_correct_amount_l385_385791


namespace range_of_a_l385_385547

open Real

-- Define the function as given in the problem
def f (a x : ℝ) : ℝ := -a * x^2 + ln x

-- Define the statement to be proven
theorem range_of_a (a : ℝ) : (∃ x > 1, f a x > -a) ↔ a < 1/2 :=
sorry

end range_of_a_l385_385547


namespace transformed_data_stats_l385_385934

open Real BigOperators

variables {n : ℕ} (x : Fin n → ℝ)

-- Given conditions
axiom avg_given : (∑ i, x i) / n = 2
axiom var_given : (∑ i, (x i - 2)^2) / n = 5

-- Definitions for transformed data
noncomputable def transformed_data (i : Fin n) : ℝ := 2 * x i + 1

-- Prove that the new average and new variance match the given correct answer
theorem transformed_data_stats :
  (∑ i, transformed_data x i) / n = 5 ∧
  (∑ i, (transformed_data x i - 5)^2) / n = 20 :=
by
  sorry

end transformed_data_stats_l385_385934


namespace sequence_count_l385_385467

theorem sequence_count :
  ∃ N : ℕ,
    (∀ i : fin 200, 
      ∃ j : fin 200, 
        ((a i + a j - N) % 203 = 0)) →
        fintype.card { a : fin 203 ^ 200 // 
        ∀ i j, (i < j → a i < a j) } = 20503 :=
begin
  sorry
end

end sequence_count_l385_385467


namespace max_colors_4x4_grid_l385_385369

def cell := (Fin 4) × (Fin 4)
def color := Fin 8

def valid_coloring (f : cell → color) : Prop :=
∀ c1 c2 : color, (c1 ≠ c2) →
(∃ i : Fin 4, ∃ j1 j2 : Fin 4, j1 ≠ j2 ∧ f (i, j1) = c1 ∧ f (i, j2) = c2) ∨ 
(∃ j : Fin 4, ∃ i1 i2 : Fin 4, i1 ≠ i2 ∧ f (i1, j) = c1 ∧ f (i2, j) = c2)

theorem max_colors_4x4_grid : ∃ (f : cell → color), valid_coloring f :=
sorry

end max_colors_4x4_grid_l385_385369


namespace soda_ratio_l385_385824

theorem soda_ratio (v p : ℝ) (hv : v > 0) (hp : p > 0) : 
  let v_z := 1.3 * v
  let p_z := 0.85 * p
  (p_z / v_z) / (p / v) = 17 / 26 :=
by sorry

end soda_ratio_l385_385824


namespace tom_reads_50_pages_per_hour_l385_385713

theorem tom_reads_50_pages_per_hour :
  (∃ (hours_per_day : ℕ), hours_per_day = 2) ∧
  (∃ (total_hours : ℕ), total_hours = 14) ∧
  (∃ (pages_per_hour : ℕ), pages_per_hour = 50) :=
by
  let hours_per_day := 10 / 5
  let total_hours := hours_per_day * 7
  let pages_per_hour := 700 / total_hours
  exact ⟨
    ⟨hours_per_day, rfl⟩,
    ⟨total_hours, rfl⟩,
    ⟨pages_per_hour, rfl⟩
  ⟩

end tom_reads_50_pages_per_hour_l385_385713


namespace solution_to_inequality_l385_385622

noncomputable def f : ℝ → ℝ := sorry

axiom f_cond1 : ∀ (x1 x2 : ℝ), x1 ≠ 0 → x2 ≠ 0 → f (x1 * x2) = f x1 + f x2
axiom f_cond2 : ∀ (x y : ℝ), 0 < x → 0 < y → x ≤ y → f x ≤ f y

theorem solution_to_inequality :
  {x : ℝ | f x + f (x - 1 / 2) ≤ 0} = 
  (set.Ico ((1 - real.sqrt 17) / 4) 0) ∪ 
  (set.Ioo 0 (1 / 2)) ∪ 
  (set.Ioo (1 / 2) ((1 + real.sqrt 17) / 4)) :=
sorry

end solution_to_inequality_l385_385622


namespace part_I_part_II_l385_385544

noncomputable def tangent_line_equation (C : ℝ × ℝ → ℝ → Prop) (A : ℝ × ℝ) : Prop :=
∀ l₁ : ℝ × ℝ → Prop,
  (l₁ A) →
  (∀ x y : ℝ, (C (x, y) 2) → (l₁ (x, y)) → x = 1 ∨ 3 * x - 4 * y - 3 = 0)

noncomputable def max_area_triangle (C : ℝ × ℝ → ℝ → Prop) (A : ℝ × ℝ) : Prop :=
∃ l₁ : ℝ × ℝ → Prop, 
  (l₁ A) ∧
  (∀ x y : ℝ, (C (x, y) 2) → (l₁ (x, y)) → d (x, y) = √2) ∧ 
  (∃ S : ℝ, S = 2) ∧
  (∀ k : ℝ, ((k = 1 ∨ k = 7) → 
  (∀ x y : ℝ, (l₁ (x, y) ↔ y = k * x - k)))

-- Definitions for the specific circle and the point A
def circle (p : ℝ × ℝ) (r : ℝ) : Prop := (p.1 - 3)^2 + (p.2 - 4)^2 = r^2
def point_A : ℝ × ℝ := (1, 0)

-- Statements for Lean
theorem part_I : tangent_line_equation circle point_A :=
by sorry

theorem part_II : max_area_triangle circle point_A :=
by sorry

end part_I_part_II_l385_385544


namespace math_homework_pages_l385_385645

-- Define Rachel's total pages, math homework pages, and reading homework pages
def total_pages : ℕ := 13
def reading_homework : ℕ := sorry
def math_homework (r : ℕ) : ℕ := r + 3

-- State the main theorem that needs to be proved
theorem math_homework_pages :
  ∃ r : ℕ, r + (math_homework r) = total_pages ∧ (math_homework r) = 8 :=
by {
  sorry
}

end math_homework_pages_l385_385645


namespace seating_arrangement_5_persons_rect_table_l385_385498

noncomputable def num_seating_arrangements (persons : ℕ) (seats : ℕ) (longer_side_seats : ℕ) (shorter_side_seats : ℕ) : ℕ :=
  if persons + 1 = seats
  then 3 * ((seats - 1)!)
  else 0

theorem seating_arrangement_5_persons_rect_table :
  num_seating_arrangements 5 6 2 1 = 360 := 
by
  sorry

end seating_arrangement_5_persons_rect_table_l385_385498


namespace distance_from_E_to_AC_correct_l1_relationship_l385_385334

-- Given conditions
variables {E F : Type}  -- Centers E and F of the circles respectively.
variables {BD AC AB BC : Segment}  -- Segments involved
variables {r1 r2 r l1 l2 : ℝ}  -- Radii of semicircles and circles

-- Distance from E to segment AC
def distance_from_E_to_AC (E AC : Point) (r1 l1 : ℝ) : ℝ :=
  sqrt((r1 + l1)^2 - (r1 - l1)^2)

-- Goal: Prove distance from E to AC
theorem distance_from_E_to_AC_correct 
  (h_tangent_E_AC : is_tangent E AC)
  (h_r1 : radius AB = r1) 
  (h_r2 : radius BC = r2) 
  (h_r : radius AC = r) 
  (h_l1 : radius E = l1) :
  d(E, AC) = sqrt((r1 + l1)^2 - (r1 - l1)^2) :=
sorry

-- Goal: Prove relationship of l1
theorem l1_relationship 
  (h_tangent_E_AB : is_tangent E AB) 
  (h_tangent_F_BC : is_tangent F BC) 
  (h_r1 : radius AB = r1) 
  (h_r2 : radius BC = r2) 
  (h_r : radius AC = r) 
  (h_l1 : radius E = l1) 
  (h_l2 : radius F = l2) :
  l1 = (r1 * r2) / (r1 + r2) :=
sorry

end distance_from_E_to_AC_correct_l1_relationship_l385_385334


namespace butcher_loses_l385_385738

theorem butcher_loses :
  let bill_was_fake := true,
      banker_initial_debt := 5,
      butcher_debt_to_farmer := 5,
      net_loss := 5
  in bill_was_fake ∧ banker_initial_debt = butcher_debt_to_farmer → net_loss = 5 :=
by
  intros
  -- proof goes here
  sorry

end butcher_loses_l385_385738


namespace seq_length_after_10_h_expansions_l385_385582

-- Definitions based on the conditions
def initial_seq := [1, 2]

def h_expansion (seq : List ℕ) : List ℕ :=
  List.bind seq (λ x => [x] ++ match seq with
    | [] => []
    | y :: ys => [x + y] ++ (ys.bind h_expansion_tail) )

-- Define the length of the sequence after n H expansions
def seq_length (n : ℕ) : ℕ :=
  2^n + 1

-- Prove that the sequence length after 10 H expansions is 1025
theorem seq_length_after_10_h_expansions : seq_length 10 = 1025 :=
by {
  sorry
}

end seq_length_after_10_h_expansions_l385_385582


namespace circle_radius_tangent_to_semicircles_l385_385635

theorem circle_radius_tangent_to_semicircles (A B C O : ℝ) (r : ℝ) :
  let AB := 4
  let BC := 2
  let AC := AB + BC
  let r_AB := AB / 2
  let r_BC := BC / 2
  let r_AC := AC / 2
  let d_O1 := r + r_AB
  let d_O2 := r + r_BC
  let d_O3 := r + r_AC
  d_O1 + d_O2 = d_O3 → r = 6 / 7 :=
by intros AB BC AC r_AB r_BC r_AC d_O1 d_O2 d_O3 h
   calc r = 6 / 7 : sorry

end circle_radius_tangent_to_semicircles_l385_385635


namespace g_3_value_l385_385675

noncomputable def g : ℝ → ℝ :=
λ x, if x = 3 then 10.5 else if x = 1 / 3 then 4.5 else sorry

theorem g_3_value :
  (∀ (x : ℝ), x ≠ 0 → 3 * g x - g (1 / x) = 3 ^ x) →
  g 3 = 10.5 :=
begin
  intros h,
  -- At this point, we would show g 3 = 10.5 by using the provided conditions, 
  -- but we skip the proof as per the instructions with a 'sorry'.
  sorry
end

end g_3_value_l385_385675


namespace car_rental_cost_l385_385194

variable (x : ℝ)

theorem car_rental_cost (h : 65 + 0.40 * 325 = x * 325) : x = 0.60 :=
by 
  sorry

end car_rental_cost_l385_385194


namespace isosceles_triangle_proof_l385_385240

open EuclideanGeometry

variables {A B C D M E : Point}

-- Definitions of points and properties
def is_isosceles_triangle (A B C : Point) : Prop := dist A C = dist B C
def is_midpoint (M A B : Point) : Prop := dist A M = dist M B
def line_intersects_at (L1 L2 : Line) (E : Point) : Prop := E ∈ L1 ∧ E ∈ L2

-- Given conditions
variables (h1 : is_isosceles_triangle A B C)
variables (h2 : altitude_foot C A B D)
variables (h3 : is_midpoint M C D)
variables (h4 : exists L, ∃ L', line_intersects_at L (line_through B M) E ∧ line_intersects_at L' (line_through A C) E)

-- Equivalent proof problem
theorem isosceles_triangle_proof (h1 : is_isosceles_triangle A B C)
                                (h2 : altitude_foot C A B D)
                                (h3 : is_midpoint M C D)
                                (h4 : ∃ L, ∃ L', line_intersects_at L (line_through B M) E ∧ line_intersects_at L' (line_through A C) E) :
  dist A C = 3 * dist C E :=
begin
  sorry
end

end isosceles_triangle_proof_l385_385240


namespace find_a5_a7_l385_385943

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Define the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions
axiom h1 : a 1 + a 3 = 2
axiom h2 : a 3 + a 5 = 4

theorem find_a5_a7 (a : ℕ → ℤ) (d : ℤ) (h_seq : is_arithmetic_sequence a d)
  (h1 : a 1 + a 3 = 2) (h2 : a 3 + a 5 = 4) : a 5 + a 7 = 6 :=
sorry

end find_a5_a7_l385_385943


namespace smallest_positive_root_in_interval_l385_385259

def f (x : ℝ) : ℝ := Real.sin x + 2 * Real.cos x + 3 * Real.tan x

theorem smallest_positive_root_in_interval :
  ∃ x ∈ set.Ioo (3 : ℝ) 4, f x = 0 :=
sorry

end smallest_positive_root_in_interval_l385_385259


namespace even_pair_probability_l385_385779

open Finset

theorem even_pair_probability : 
  let S := (range 5).image (λ x, x + 1),
      even_numbers := S.filter (λ x, x % 2 = 0),
      total_pairs := S.product S \ (S.diag),
      even_pairs := even_numbers.product even_numbers \ (even_numbers.diag) in
  ((even_pairs.card : ℚ) / total_pairs.card) = (1 / 10) :=
by 
  let S := (range 5).image (λ x, x + 1),
  let even_numbers := S.filter (λ x, x % 2 = 0),
  let total_pairs := S.product S \ (S.diag),
  let even_pairs := even_numbers.product even_numbers \ (even_numbers.diag),
  have h1 : (even_pairs.card : ℚ) = 1 := by sorry,
  have h2 : total_pairs.card = 10 := by sorry,
  rw [h1, h2],
  norm_num

end even_pair_probability_l385_385779


namespace problem_solution_l385_385533

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

def quadratic_increasing_then_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ c : ℝ, a < c ∧ c < b ∧ (∀ x : ℝ, a <= x ∧ x < c → f x < f (x + 1)) ∧ (∀ x : ℝ, c <= x ∧ x < b → f x > f (x + 1))

theorem problem_solution (m : ℝ) :
  is_even (λ x : ℝ, (m - 1) * x^2 + 3 * m * x + 3) →
  quadratic_increasing_then_decreasing (λ x : ℝ, (m - 1) * x^2 + 3 * m * x + 3) (-4) 2 :=
by
  intro h_even
  sorry

end problem_solution_l385_385533


namespace cube_plane_intersection_distance_l385_385407

theorem cube_plane_intersection_distance :
  let vertices := [(0, 0, 0), (0, 0, 6), (0, 6, 0), (0, 6, 6), (6, 0, 0), (6, 0, 6), (6, 6, 0), (6, 6, 6)]
  let P := (0, 3, 0)
  let Q := (2, 0, 0)
  let R := (2, 6, 6)
  let plane_equation := 3 * x - 2 * y - 2 * z + 6 = 0
  let S := (2, 0, 6)
  let T := (0, 6, 3)
  dist S T = 7 := sorry

end cube_plane_intersection_distance_l385_385407


namespace factorize_difference_of_squares_l385_385088

variable (x : ℝ)

theorem factorize_difference_of_squares :
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorize_difference_of_squares_l385_385088


namespace power_function_value_l385_385940

noncomputable def power_function (f : ℝ → ℝ) := ∃ α : ℝ, ∀ x : ℝ, f(x) = x^α

theorem power_function_value (f : ℝ → ℝ) (h1 : power_function f) (h2 : f 2 = 4) : f 3 = 9 :=
sorry

end power_function_value_l385_385940


namespace set_contains_all_integers_l385_385033

open Int

variables (S : Set ℤ)

-- Conditions
axiom condition1 : ∀ a b : ℤ, a ∈ S → b ∈ S → (a^2 - a) ∈ S ∧ (a^2 - b) ∈ S
axiom condition2 : ∃ a b : ℤ, a ∈ S ∧ b ∈ S ∧ gcd a b = 1 ∧ gcd (a - 2) (b - 2) = 1

-- Statement to prove
theorem set_contains_all_integers (S : Set ℤ) :
  (∀ a b : ℤ, a ∈ S → b ∈ S → (a^2 - a) ∈ S ∧ (a^2 - b) ∈ S) →
  (∃ a b : ℤ, a ∈ S ∧ b ∈ S ∧ gcd a b = 1 ∧ gcd (a - 2) (b - 2) = 1) →
  ∀ n : ℤ, n ∈ S :=
by
  assume condition1 condition2
  sorry

end set_contains_all_integers_l385_385033


namespace handshake_count_l385_385993

theorem handshake_count (n : ℕ) (h : n = 5) : (n * (n - 1)) / 2 = 10 :=
by
  rw [h]
  norm_num
  sorry

end handshake_count_l385_385993


namespace integral_converges_l385_385315

open Real

theorem integral_converges :
  ∃ (I : ℝ), tendsto (λ (b : ℝ), ∫ x in 1..b, (x * cos x) / ((1 + x^2) * sqrt (4 + x^2))) at_top (nhds I) :=
  sorry

end integral_converges_l385_385315


namespace third_term_of_arithmetic_sequence_l385_385229

theorem third_term_of_arithmetic_sequence (a d : ℝ) (h : a + (a + 4 * d) = 10) : a + 2 * d = 5 :=
by {
  sorry
}

end third_term_of_arithmetic_sequence_l385_385229


namespace positive_difference_l385_385663

def average (a b : ℕ) : ℕ := (a + b) / 2

theorem positive_difference : 
  ∃ x : ℕ, average 35 x = 45 → | x - 35 | = 20 :=
by
  sorry

end positive_difference_l385_385663


namespace measure_six_liters_l385_385412

-- Given conditions as constants
def container_capacity : ℕ := 40
def ten_liter_bucket_capacity : ℕ := 10
def nine_liter_jug_capacity : ℕ := 9
def five_liter_jug_capacity : ℕ := 5

-- Goal: Measure out exactly 6 liters of milk using the above containers
theorem measure_six_liters (container : ℕ) (ten_bucket : ℕ) (nine_jug : ℕ) (five_jug : ℕ) :
  container = 40 →
  ten_bucket ≤ 10 →
  nine_jug ≤ 9 →
  five_jug ≤ 5 →
  ∃ (sequence_of_steps : ℕ → ℕ) (final_ten_bucket : ℕ),
    final_ten_bucket = 6 ∧ final_ten_bucket ≤ ten_bucket :=
by
  intro hcontainer hten_bucket hnine_jug hfive_jug
  sorry

end measure_six_liters_l385_385412


namespace smallest_prime_less_than_square_l385_385371

theorem smallest_prime_less_than_square : ∃ p n : ℕ, Prime p ∧ p = n^2 - 20 ∧ p = 5 :=
by 
  sorry

end smallest_prime_less_than_square_l385_385371


namespace binary_difference_l385_385877

theorem binary_difference (x y : ℕ) (b : ℕ := nat.toDigits 2 315) 
  (hx : x = b.count 0) (hy : y = b.count 1) : y - x = 5 :=
by
  sorry

end binary_difference_l385_385877


namespace value_of_a5_l385_385665

variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a : ℕ → α) (r : α) : Prop :=
∀ n, a (n + 1) = r * a n

theorem value_of_a5
  (a : ℕ → α)
  (r : α)
  (h1 : geometric_sequence a r)
  (h2 : r = 2)
  (h3 : 0 < ∀ n, a n)
  (h4 : a 3 * a 11 = 16) :
  a 5 = 1 :=
sorry

end value_of_a5_l385_385665


namespace sequence_v18_eq_l385_385470

noncomputable def sequence : ℕ → ℝ
| 0     := b  -- Note: In Lean we start indexing from 0
| (n+1) := -1 / (sequence n + 2)

variable (b: ℝ) (hb : 0 < b)

theorem sequence_v18_eq : sequence b 17 = -1 / (b + 2) := sorry

end sequence_v18_eq_l385_385470


namespace find_B_squared_l385_385111

noncomputable def g (x : ℝ) : ℝ := Real.sqrt 31 + 85 / x

theorem find_B_squared :
  let x1 := (Real.sqrt 31 + Real.sqrt 371) / 2
  let x2 := (Real.sqrt 31 - Real.sqrt 371) / 2
  let B := |x1| + |x2|
  B^2 = 371 :=
by
  sorry

end find_B_squared_l385_385111


namespace cone_volume_is_correct_l385_385332

theorem cone_volume_is_correct (r l h : ℝ) 
  (h1 : 2 * r = Real.sqrt 2 * l)
  (h2 : π * r * l = 16 * Real.sqrt 2 * π)
  (h3 : h = r) : 
  (1 / 3) * π * r ^ 2 * h = (64 / 3) * π :=
by sorry

end cone_volume_is_correct_l385_385332


namespace four_non_collinear_points_do_not_determine_plane_l385_385380

theorem four_non_collinear_points_do_not_determine_plane
  (h1 : ∀ (l₁ l₂ : Set ℝ^3), (∃ x ∈ l₁, x ∈ l₂) → ∃ P : Set ℝ^3, ∀ y ∈ l₁ ∪ l₂, y ∈ P)
  (h2 : ∀ (l₁ l₂ : Set ℝ^3), (∃ y ∈ l₁, ∀ z ∈ l₂, y ≠ z ∧ l₁ ∩ l₂ = ∅) → ∃ P : Set ℝ^3, ∀ w ∈ l₁ ∪ l₂, w ∈ P)
  (h3 : ∀ (l : Set ℝ^3) (p : ℝ^3), (∃ x ∈ l, x ≠ p) → ∃ P : Set ℝ^3, ∀ y ∈ ({p} ∪ l), y ∈ P)
  (h4 : ∀ (p₁ p₂ p₃ : ℝ^3), (p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₃ ≠ p₁) → ∃ P : Set ℝ^3, ∀ x ∈ ({p₁, p₂, p₃} : Set ℝ^3), x ∈ P) :
  ∃ (a b c d : ℝ^3), a ≠ b ∧ c ≠ d ∧ d ≠ a ∧ ∀ P : Set ℝ^3, ¬∀ x ∈ ({a, b, c, d} : Set ℝ^3), x ∈ P :=
sorry

end four_non_collinear_points_do_not_determine_plane_l385_385380


namespace gradient_at_A_directional_derivative_at_A_l385_385553

noncomputable def z : ℝ × ℝ → ℝ := λ p, Real.arcsin (p.2 / (p.1 ^ 2))
def A : ℝ × ℝ := (-2, -1)
def a : ℝ × ℝ := (3, -4)

theorem gradient_at_A : 
    let grad_z := 
        (λ p, (-2 * p.2 / (p.1 * Real.sqrt (p.1^4 - p.2^2))), 
               1 / Real.sqrt (p.1^4 - p.2^2))
    in grad_z A = (-1 / Real.sqrt 15, 1 / Real.sqrt 15) :=
sorry

theorem directional_derivative_at_A :
    let grad_z := 
        (λ p, (-2 * p.2 / (p.1 * Real.sqrt (p.1^4 - p.2^2))), 
               1 / Real.sqrt (p.1^4 - p.2^2))
        in (grad_z A).1 * (a.1 / 5) + (grad_z A).2 * (a.2 / 5) = -7 / (5 * Real.sqrt 15) :=
sorry

end gradient_at_A_directional_derivative_at_A_l385_385553


namespace average_speed_calculation_l385_385005

-- Define constants and conditions
def speed_swimming : ℝ := 1
def speed_running : ℝ := 6
def distance : ℝ := 1  -- We use a generic distance d = 1 (assuming normalized unit distance)

-- Proof statement
theorem average_speed_calculation :
  (2 * distance) / ((distance / speed_swimming) + (distance / speed_running)) = 12 / 7 :=
by
  sorry

end average_speed_calculation_l385_385005


namespace two_digit_number_l385_385798

theorem two_digit_number (a : ℕ) (h : 3 ≤ a) : 
  ∃ N : ℕ, N = 11 * a - 30 :=
by
  use 11 * a - 30
  sorry

end two_digit_number_l385_385798


namespace jill_food_spending_l385_385308

theorem jill_food_spending :
  ∀ (T : ℝ) (c f o : ℝ),
    c = 0.5 * T →
    o = 0.3 * T →
    (0.04 * c + 0 + 0.1 * o) = 0.05 * T →
    f = 0.2 * T :=
by
  intros T c f o h_c h_o h_tax
  sorry

end jill_food_spending_l385_385308


namespace sum_possible_values_of_d_l385_385757

theorem sum_possible_values_of_d :
  let n := 7^4 -- smallest integer with 5 digits in base 7
  let m := 7^5 - 1 -- largest integer with 5 digits in base 7
  (log_base 3 n).nat + (log_base 3 m).nat = 17 :=
by
  sorry

end sum_possible_values_of_d_l385_385757


namespace factorize_x_squared_minus_1_l385_385097

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by
  -- Proof goes here
  sorry

end factorize_x_squared_minus_1_l385_385097


namespace jennifer_fifth_score_l385_385599

theorem jennifer_fifth_score :
  ∀ (x : ℝ), (85 + 90 + 87 + 92 + x) / 5 = 89 → x = 91 :=
by
  sorry

end jennifer_fifth_score_l385_385599


namespace determine_pairs_l385_385884

theorem determine_pairs (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔
  (∃ k : ℕ, k > 0 ∧ (a = 7 * k^2 ∧ b = 7 * k) ∨ (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1)) :=
by
  sorry

end determine_pairs_l385_385884


namespace common_ratio_is_three_l385_385222

-- Define the first term of the geometric sequence
def a₁ : ℝ := 2 / 3

-- Define the fourth term of the geometric sequence using the integral
def a₄ : ℝ := ∫ x in 1..4, (1 + 2 * x)

-- Define the statement that we need to prove: the common ratio q is 3
theorem common_ratio_is_three (q : ℝ) (hq : a₄ = a₁ * q^3) : q = 3 :=
by
  sorry

end common_ratio_is_three_l385_385222


namespace diametrically_opposite_exists_l385_385158

theorem diametrically_opposite_exists (k : ℕ) (hk : k ≥ 1)
  (points : Finset ℝ) 
  (hcard : points.card = 3 * k)
  (h1 : points.sum (λ x, by if h : x ∈ points then 1 else 0) = k)
  (h2 : points.sum (λ x, by if h : x ∈ points then 2 else 0) = k)
  (h3 : points.sum (λ x, by if h : x ∈ points then 3 else 0) = k) :
  ∃ (a b : ℝ), a ∈ points ∧ b ∈ points ∧ (a + b) % (2 * pi) = pi :=
by
  sorry

end diametrically_opposite_exists_l385_385158


namespace new_area_of_card_l385_385319

-- Conditions from the problem
def original_length : ℕ := 5
def original_width : ℕ := 7
def shortened_length := original_length - 2
def shortened_width := original_width - 1

-- Statement of the proof problem
theorem new_area_of_card : shortened_length * shortened_width = 18 :=
by
  sorry

end new_area_of_card_l385_385319


namespace collinear_O_P_Q_l385_385618

variables {A B C D O P M N Q : Type}
variables [AddGroup A] [AddGroup B]
variables [AddGroup C] [AddGroup D]
variables [AddGroup O] [AddGroup P] [AddGroup M] [AddGroup N] [AddGroup Q]

-- We need some basic assumptions about our geometric points and operations
variables (center_parallelogram : ∀ (A B C D O : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup O], bool)
variables (is_midpoint : ∀ (X Y Z : Type) [AddGroup X] [AddGroup Y] [AddGroup Z], bool)
variables (intersection : ∀ (X Y : Type) [AddGroup X] [AddGroup Y], X)

-- Definitions based on the conditions in the problem
def in_parallelogram_center (ABCD_O : center_parallelogram A B C D O) : Prop := 
    center_parallelogram A B C D O

def midpoint_AP (AP_M : is_midpoint A P M) : Prop := 
    is_midpoint A P M

def midpoint_BP (BP_N : is_midpoint B P N) : Prop := 
    is_midpoint B P N

def intersection_MC_ND (MC_ND_Q : Q = intersection M N) : Prop := 
    Q = intersection M N

-- Theorem statement to be proven
theorem collinear_O_P_Q (h1 : in_parallelogram_center (center_parallelogram A B C D O))
                       (h2 : midpoint_AP (is_midpoint A P M))
                       (h3 : midpoint_BP (is_midpoint B P N))
                       (h4 : intersection_MC_ND (Q = intersection M N)) :
  collinear O P Q := 
    sorry

end collinear_O_P_Q_l385_385618


namespace ball_radius_and_surface_area_l385_385014

theorem ball_radius_and_surface_area (d h : ℝ) (r : ℝ) :
  d = 12 ∧ h = 2 ∧ (6^2 + (r - h)^2 = r^2) → (r = 10 ∧ 4 * Real.pi * r^2 = 400 * Real.pi) := by
  sorry

end ball_radius_and_surface_area_l385_385014


namespace sufficient_and_necessary_condition_l385_385522

variable (a_n : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ)
variable (h_arith_seq : ∀ n : ℕ, a_n (n + 1) = a_n n + d)
variable (h_sum : ∀ n : ℕ, S n = n * (a_n 1 + (n - 1) / 2 * d))

theorem sufficient_and_necessary_condition (d : ℚ) (h_arith_seq : ∀ n : ℕ, a_n (n + 1) = a_n n + d)
  (h_sum : ∀ n : ℕ, S n = n * (a_n 1 + (n - 1) / 2 * d)) :
  (d > 0) ↔ (S 4 + S 6 > 2 * S 5) := by
  sorry

end sufficient_and_necessary_condition_l385_385522


namespace solve_for_x_l385_385067

theorem solve_for_x (x y : ℝ) (h₁ : y = (x^2 - 9) / (x - 3)) (h₂ : y = 3 * x - 4) : x = 7 / 2 :=
by sorry

end solve_for_x_l385_385067


namespace find_constants_and_extrema_l385_385175

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + 1
noncomputable def f' (x : ℝ) (a b : ℝ) : ℝ := 6 * x^2 + 2 * a * x + b

theorem find_constants_and_extrema :
  (∀ x: ℝ, f' x 3 (-12) = 6 * (x - 1) * (x + 2)) →
  f' 1 3 (-12) = 0 →
  ∀ (a b : ℝ), -a / 6 = -1 / 2 → f' 1 a b = 0 → 
  a = 3 ∧ b = -12 ∧ 
  (∀ (x : ℝ), x = -2 → f x 3 (-12) = 21) ∧ 
  (∀ (x : ℝ), x = 1 → f x 3 (-12) = -6) :=
by
  intros h1 h2 a b ha hb
  have ha : a = 3,
  { linarith [ha] },
  have hb : b = -12,
  { linarith [hb, ha] },
  rw [ha, hb] at *,
  split,
  { exact ha },
  split,
  { exact hb },
  split,
  { rw h1 (-2), ring },
  { rw h1 1, ring },
  sorry

end find_constants_and_extrema_l385_385175


namespace arithmetic_seq_no_geometric_l385_385142

variable {α : Type*} [LinearOrderedField α]

theorem arithmetic_seq_no_geometric 
  (a : ℕ → α) 
  (r : α) 
  (h_arith : ∀ n, a (n + 1) = a n + r)
  (k l : ℕ)
  (h_k : a k = 1)
  (h_l : a l = Real.sqrt 2) :
  ∀ {m n p : ℕ}, m ≠ n → n ≠ p → m ≠ p → ¬ (∃ q, a n^2 = a m * a p) :=
by
  sorry

end arithmetic_seq_no_geometric_l385_385142


namespace hyperbola_segment_equality_l385_385137

theorem hyperbola_segment_equality
  (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (h : ∀ l : set (ℝ × ℝ), line l → 
    (∃ A B C D : ℝ × ℝ, distinct_points [A, B, C, D] ∧ intersect l (asymptotes l) = {A, B, C, D})) :
  ∀ l, line l → 
    (|segment_length (proj_point A B)| = |segment_length (proj_point C D)|) :=
begin
  sorry
end

noncomputable def line : Type := ℝ × ℝ → Prop

def hyperbola (a b : ℝ) : set (ℝ × ℝ) := 
  {p | let x := p.1 in let y := p.2 in (x^2 / a^2) - (y^2 / b^2) = 1}

def asymptotes (a b : ℝ) : set (ℝ × ℝ) := 
  {p | let x := p.1 in let y := p.2 in (b^2 * x^2) = (a^2 * y^2)}

noncomputable def segment_length (p1 p2 : ℝ × ℝ) : ℝ :=
  real.dist p1 p2

noncomputable def proj_point {p1 p2 : set (ℝ × ℝ)} : set (ℝ × ℝ) :=
  {p | ∃ p1 p2, p ∈ segment_length (p1 p2)}

end hyperbola_segment_equality_l385_385137


namespace count_permutations_divisible_by_2_l385_385204

-- Definition of the original set of digits
def original_digits : Multiset ℕ := {2, 3, 1, 1, 5, 7, 1, 5, 2}

-- Condition defining even criteria (necessary for divisibility by 2)
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Theorem statement
theorem count_permutations_divisible_by_2 :
  (original_digits.count 2 = 2) →
  (original_digits.count 1 = 3) →
  (original_digits.count 5 = 2) →
  (original_digits.count 3 = 1) →
  (original_digits.count 7 = 1) →
  let even_digits := {x // Multiset.mem x original_digits ∧ is_even x} in
  even_digits = {⟨2, _⟩, ⟨2, _⟩} →
  let remaining_digits := {2, 3, 1, 1, 5, 7, 1, 5} in
  Multiset.card remaining_digits = 8 →
  (Multiset.card (Multiset.of_list [⟨2, sorry⟩, ⟨3, sorry⟩, ⟨1, sorry⟩, ⟨1, sorry⟩, ⟨5, sorry⟩, ⟨7, sorry⟩, ⟨1, sorry⟩, ⟨5, sorry⟩])) = 8 →
  Multiset.count 1 remaining_digits = 3 →
  Multiset.count 2 remaining_digits = 1 →
  Multiset.count 5 remaining_digits = 2 →
  Multiset.count 3 remaining_digits = 1 →
  Multiset.count 7 remaining_digits = 1 →
  multichoose remaining_digits ∅ 8 3 2 1 1 1 = 3360 :=
by sorry

end count_permutations_divisible_by_2_l385_385204


namespace negation_of_p_l385_385555

def p : Prop := ∀ x : ℝ, x > Real.sin x

theorem negation_of_p : ¬p ↔ ∃ x : ℝ, x ≤ Real.sin x :=
by sorry

end negation_of_p_l385_385555


namespace range_m_l385_385682

-- Definitions
def line (k : ℝ) : ℝ → ℝ := λ x, k * x + 1
def ellipse (m : ℝ) (x y : ℝ) : Prop := (x^2 / 5) + (y^2 / m) = 1

-- Mathematical statement
theorem range_m (k : ℝ) :
  (∀ k : ℝ, ∀ x : ℝ, ∃ y : ℝ, ellipse m x y ∧ y = line k x) ↔
  (m ∈ [1, 5) ∪ (5, +∞)) :=
sorry

end range_m_l385_385682


namespace quadratic_roots_l385_385674

theorem quadratic_roots (a b c : ℝ) :
  (∀ (x y : ℝ), ((x, y) = (-2, 12) ∨ (x, y) = (0, -8) ∨ (x, y) = (1, -12) ∨ (x, y) = (3, -8)) → y = a * x^2 + b * x + c) →
  (a * 0^2 + b * 0 + c + 8 = 0) ∧ (a * 3^2 + b * 3 + c + 8 = 0) :=
by sorry

end quadratic_roots_l385_385674


namespace find_breadth_of_rectangle_l385_385737

noncomputable def breadth_of_rectangle (A : ℝ) (length_to_breadth_ratio : ℝ) (breadth : ℝ) : Prop :=
  A = length_to_breadth_ratio * breadth * breadth → breadth = 20

-- Now we can state the theorem.
theorem find_breadth_of_rectangle (A : ℝ) (length_to_breadth_ratio : ℝ) : breadth_of_rectangle A length_to_breadth_ratio 20 :=
by
  intros h
  sorry

end find_breadth_of_rectangle_l385_385737


namespace pyramid_volume_is_1000_l385_385471

-- Define the vertices of the triangle
def A := (0 : ℝ, 0 : ℝ)
def B := (30 : ℝ, 0 : ℝ)
def C := (15 : ℝ, 20 : ℝ)

-- Define the centroid of the triangle
def G := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

-- Define the point above the centroid
def P := (G.1, G.2, (10 : ℝ))

-- Define the volume of the pyramid
def volumePyramid (A B C : (ℝ × ℝ)) (P : ℝ × ℝ × ℝ) : ℝ :=
  (1 / 3) * (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) * P.3

-- Lean proof statement to check the volume
theorem pyramid_volume_is_1000 :
  volumePyramid A B C P = 1000 := by
  -- This is where the proof would be written, but it is omitted here.
  sorry

end pyramid_volume_is_1000_l385_385471


namespace reading_schedule_l385_385697

-- Definitions of reading speeds and conditions
def total_pages := 910
def alice_speed := 30  -- seconds per page
def bob_speed := 60    -- seconds per page
def chandra_speed := 45  -- seconds per page

-- Mathematical problem statement
theorem reading_schedule :
  ∃ (x y : ℕ), 
    (x < y) ∧ 
    (y ≤ total_pages) ∧ 
    (30 * x = 45 * (y - x) ∧ 45 * (y - x) = 60 * (total_pages - y)) ∧ 
    x = 420 ∧ 
    y = 700 :=
  sorry

end reading_schedule_l385_385697


namespace odd_number_divisibility_l385_385314

theorem odd_number_divisibility (a : ℤ) (h : a % 2 = 1) : ∃ (k : ℤ), a^4 + 9 * (9 - 2 * a^2) = 16 * k :=
by
  sorry

end odd_number_divisibility_l385_385314


namespace calculate_expression_divisible_by_2_l385_385009

theorem calculate_expression : 999^2 + 999 = 999000 := 
begin
  -- Proof required
  sorry
end

theorem divisible_by_2 (a : ℤ) : (a^2 + a) % 2 = 0 := 
begin
  -- Proof required
  sorry
end

end calculate_expression_divisible_by_2_l385_385009


namespace sum_of_cot_squared_l385_385609

theorem sum_of_cot_squared (T : set ℝ) (h₁ : ∀ x ∈ T, 0 < x ∧ x < (π / 2)) (h₂ : ∀ x ∈ T, ∃ a b c : ℝ, (a = csc x ∨ a = sec x ∨ a = cot x) ∧ (b = csc x ∨ b = sec x ∨ b = cot x) ∧ (c = csc x ∨ c = sec x ∨ c = cot x) ∧ a^2 = b^2 + c^2) :
  (∑ x in T, (cot x) ^ 2) = 0 :=
by
  sorry

end sum_of_cot_squared_l385_385609


namespace factorize_difference_of_squares_l385_385087

variable (x : ℝ)

theorem factorize_difference_of_squares :
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorize_difference_of_squares_l385_385087


namespace complement_of_M_with_respect_to_U_l385_385560

open Set

def U : Set ℤ := {-1, -2, -3, -4}
def M : Set ℤ := {-2, -3}

theorem complement_of_M_with_respect_to_U :
  (U \ M) = {-1, -4} :=
by
  sorry

end complement_of_M_with_respect_to_U_l385_385560


namespace algebraic_expression_1_algebraic_expression_2_l385_385125

-- Problem 1
theorem algebraic_expression_1 (a : ℚ) (h : a = 4 / 5) : -24.7 * a + 1.3 * a - (33 / 5) * a = -24 := 
by 
  sorry

-- Problem 2
theorem algebraic_expression_2 (a b : ℕ) (ha : a = 899) (hb : b = 101) : a^2 + 2 * a * b + b^2 = 1000000 := 
by 
  sorry

end algebraic_expression_1_algebraic_expression_2_l385_385125


namespace similar_but_not_identical_cuts_impossible_specific_equilateral_cuts_l385_385913

-- Definitions based on the given conditions:
def triangular_prism : Type := sorry -- Assume we have a type representing a triangular prism
def cut_off_triangle (prism : triangular_prism) (cut : Type) : Prop := sorry -- A predicate defining cutting off a triangle from the prism

-- The main proof statements based on the questions and answers:
theorem similar_but_not_identical_cuts (prism : triangular_prism) :
  (∃ T1 T2 : Type, cut_off_triangle prism T1 ∧ cut_off_triangle prism T2 ∧ T1 ≠ T2 ∧ (T1.similar T2) ∧ (¬ T1.identical T2)) := 
  sorry

theorem impossible_specific_equilateral_cuts (prism : triangular_prism) :
  ¬ (cut_off_triangle prism (equilateral_triangle 1) ∧ cut_off_triangle prism (equilateral_triangle 2)) :=
  sorry

end similar_but_not_identical_cuts_impossible_specific_equilateral_cuts_l385_385913


namespace max_sum_cd_l385_385820

theorem max_sum_cd (c d : ℕ) (hc : c > 0) (hd : d > 1) (hcd : c^d < 500) 
  (hmax : ∀ (c' d': ℕ), c' > 0 → d' > 1 → c'^d' < 500 → c'^d' ≤ c^d) : c + d = 24 := 
by
  have h1 : 22^2 = 484 := rfl
  have h2 : c = 22 ∧ d = 2 := by sorry
  exact by sorry

end max_sum_cd_l385_385820


namespace longest_side_obtuse_triangle_l385_385588

theorem longest_side_obtuse_triangle (a b c : ℝ) (h₀ : a = 2) (h₁ : b = 4) 
  (h₂ : a^2 + b^2 < c^2) : 
  2 * Real.sqrt 5 < c ∧ c < 6 :=
by 
  sorry

end longest_side_obtuse_triangle_l385_385588


namespace AP_squared_plus_BQ_squared_eq_PQ_squared_l385_385453

theorem AP_squared_plus_BQ_squared_eq_PQ_squared
  (A B C P Q : Point)
  (h1 : AC = BC)
  (h2 : ∠ACB = 90°)
  (h3 : P ∈ AB)
  (h4 : Q ∈ AB)
  (h5 : ∠PCQ = 45°) : 
  AP^2 + BQ^2 = PQ^2 := 
sorry

end AP_squared_plus_BQ_squared_eq_PQ_squared_l385_385453


namespace mix_solutions_l385_385656

-- Definitions based on conditions
def solution_x_percentage : ℝ := 0.10
def solution_y_percentage : ℝ := 0.30
def volume_y : ℝ := 100
def desired_percentage : ℝ := 0.15

-- Problem statement rewrite with equivalent proof goal
theorem mix_solutions :
  ∃ Vx : ℝ, (Vx * solution_x_percentage + volume_y * solution_y_percentage) = (Vx + volume_y) * desired_percentage ∧ Vx = 300 :=
by
  sorry

end mix_solutions_l385_385656


namespace geometric_sequence_sum_six_l385_385541

theorem geometric_sequence_sum_six (a : ℕ → ℝ) (q : ℝ) 
  (h0 : ∀ n, 0 < a n)
  (h1 : a 2 = 2)
  (h2 : a 4 = 8)
  (q_pos : 0 < q)
  (h_geometric : ∀ n, a (n+1) = a n * q) :
  let S6 := ∑ i in finset.range 6, a i 
  in S6 = 63 := 
by sorry

end geometric_sequence_sum_six_l385_385541


namespace car_rental_cost_l385_385193

variable (x : ℝ)

theorem car_rental_cost (h : 65 + 0.40 * 325 = x * 325) : x = 0.60 :=
by 
  sorry

end car_rental_cost_l385_385193


namespace woman_finishes_work_in_225_days_l385_385749

theorem woman_finishes_work_in_225_days
  (M W : ℝ)
  (h1 : (10 * M + 15 * W) * 6 = 1)
  (h2 : M * 100 = 1) :
  1 / W = 225 :=
by
  sorry

end woman_finishes_work_in_225_days_l385_385749


namespace smallest_n_not_divisible_by_10_smallest_n_correct_l385_385906

theorem smallest_n_not_divisible_by_10 :
  ∃ n ≥ 2017, n % 4 = 0 ∧ (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 :=
by
  -- Existence proof of such n is omitted
  sorry

def smallest_n : Nat :=
  Nat.find $ smallest_n_not_divisible_by_10

theorem smallest_n_correct : smallest_n = 2020 :=
by
  -- Correctness proof of smallest_n is omitted
  sorry

end smallest_n_not_divisible_by_10_smallest_n_correct_l385_385906


namespace factorize_difference_of_squares_l385_385109

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_difference_of_squares_l385_385109


namespace max_product_xy_l385_385147

theorem max_product_xy (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 2*x + y = 1) : 
  xy ≤ 1/8 :=
by sorry

end max_product_xy_l385_385147


namespace area_triangle_XUV_l385_385292

variable {α : Type*} [InnerProductSpace.Real α]

theorem area_triangle_XUV
  (A B C I M N P E F U V X : α)
  (h_mid_M : M = midpoint B C)
  (h_mid_N : N = midpoint C A)
  (h_mid_P : P = midpoint A B)
  (h_tang_E : is_tangency_point γ E)
  (h_tang_F : is_tangency_point γ F)
  (h_inter_U : U = line_intersection (line E F) (line M N))
  (h_inter_V : V = line_intersection (line E F) (line M P))
  (h_mid_X : is_arc_midpoint X A B C)
  (h_AB : dist A B = 5)
  (h_AC : dist A C = 8)
  (h_angle_A : angle B A C = real.pi / 3) :
  area (triangle X U V) = (21 * real.sqrt 3) / 8 :=
sorry

end area_triangle_XUV_l385_385292


namespace exist_unique_point_iff_parallel_l385_385642

variables {A B C D M : Type} [ConvexQuadrilateral A B C D]
variables (S_ABM S_CDM S_ABCD : ℝ)

theorem exist_unique_point_iff_parallel (h1 : M ∈ line_segment A D)
    (h2 : S_ABM = area A B M) 
    (h3 : S_CDM = area C D M) 
    (h4 : S_ABCD = area A B C D) : 
    (AB_parallel_CD : parallel A B C D) ↔ 
    ∃! (M : Type), M ∈ line_segment A D ∧ √S_ABM + √S_CDM = √S_ABCD := 
sorry

end exist_unique_point_iff_parallel_l385_385642


namespace divisibility_problem_l385_385881

theorem divisibility_problem (a b k : ℕ) :
  (a = 7 * k^2 ∧ b = 7 * k) ∨ (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) →
  a * b^2 + b + 7 ∣ a^2 * b + a + b := by
  intro h
  cases h
  case inl h1 =>
    rw [h1.1, h1.2]
    sorry
  case inr h2 =>
    cases h2
    case inl h21 =>
      rw [h21.1, h21.2]
      sorry
    case inr h22 =>
      rw [h22.1, h22.2]
      sorry

end divisibility_problem_l385_385881


namespace sum_first_10_terms_arithmetic_seq_l385_385273

theorem sum_first_10_terms_arithmetic_seq :
  ∀ (a : ℕ → ℤ), (a 1 = -2) → (a 2 + a 6 = 2) →
  (∃ S₁₀, S₁₀ = 10 * a 1 + (10 * (10 - 1)) / 2 * (a 2 - a 1) / (2 - 1) ∧ S₁₀ = 25) :=
begin
  assume a,
  assume h1 : a 1 = -2,
  assume h2 : a 2 + a 6 = 2,
  sorry
end

end sum_first_10_terms_arithmetic_seq_l385_385273


namespace percent_increase_perimeter_third_triangle_l385_385806

noncomputable def side_length_first : ℝ := 4
noncomputable def side_length_second : ℝ := 2 * side_length_first
noncomputable def side_length_third : ℝ := 2 * side_length_second

noncomputable def perimeter (s : ℝ) : ℝ := 3 * s

noncomputable def percent_increase (initial_perimeter final_perimeter : ℝ) : ℝ := 
  ((final_perimeter - initial_perimeter) / initial_perimeter) * 100

theorem percent_increase_perimeter_third_triangle :
  percent_increase (perimeter side_length_first) (perimeter side_length_third) = 300 := 
sorry

end percent_increase_perimeter_third_triangle_l385_385806


namespace find_value_l385_385619

noncomputable def S2013 (x y : ℂ) (h : x ≠ 0 ∧ y ≠ 0) (h_eq : x^2 + x * y + y^2 = 0) : ℂ :=
  (x / (x + y))^2013 + (y / (x + y))^2013

theorem find_value (x y : ℂ) (h : x ≠ 0 ∧ y ≠ 0) (h_eq : x^2 + x * y + y^2 = 0) :
  S2013 x y h h_eq = -2 :=
sorry

end find_value_l385_385619


namespace max_sum_cd_l385_385821

theorem max_sum_cd (c d : ℕ) (hc : c > 0) (hd : d > 1) (hcd : c^d < 500) 
  (hmax : ∀ (c' d': ℕ), c' > 0 → d' > 1 → c'^d' < 500 → c'^d' ≤ c^d) : c + d = 24 := 
by
  have h1 : 22^2 = 484 := rfl
  have h2 : c = 22 ∧ d = 2 := by sorry
  exact by sorry

end max_sum_cd_l385_385821


namespace general_eq_curveC2_polar_eq_curveC1_min_distance_l385_385688

noncomputable def curveC1_param (t : ℝ) : ℝ × ℝ :=
  (1 + t * Real.cos (π / 4), 5 + t * Real.sin (π / 4))

noncomputable def curveC2_param (φ : ℝ) : ℝ × ℝ :=
  (Real.cos φ, Real.sqrt 3 * Real.sin φ)

theorem general_eq_curveC2 :
  ∀ φ, let P := curveC2_param φ in (P.1^2 + (P.2^2) / 3) = 1 :=
by sorry

theorem polar_eq_curveC1 :
  ∀ t, let P := curveC1_param t in ((P.1 * Real.cos (π / 4) - P.2 * Real.sin (π / 4)) + 4) = 0 :=
by sorry

theorem min_distance :
  ∀ φ, let P := curveC2_param φ in 
    (|P.1 - Real.sqrt 3 * P.2 + 4| / Real.sqrt 2) = Real.sqrt 2 :=
by sorry

end general_eq_curveC2_polar_eq_curveC1_min_distance_l385_385688


namespace solve_inequalities_system_l385_385694

theorem solve_inequalities_system (x : ℝ) :
  (x - 2 ≥ -5) → (3x < x + 2) → (-3 ≤ x ∧ x < 1) :=
by
  intros h1 h2
  sorry

end solve_inequalities_system_l385_385694


namespace incorrect_propositions_l385_385475

theorem incorrect_propositions :
  ¬ (∀ P : Prop, P → P) ∨
  (¬ (∀ x : ℝ, x^2 - x ≤ 0) ↔ (∃ x : ℝ, x^2 - x > 0)) ∨
  (∀ (R : Type) (f : R → Prop), (∀ r, f r → ∃ r', f r') = ∃ r, f r ∧ ∃ r', f r') ∨
  (∀ (x : ℝ), x ≠ 3 → abs x = 3 → x = 3) :=
by sorry

end incorrect_propositions_l385_385475


namespace total_length_of_wire_l385_385591

-- Definitions based on conditions
def num_squares : ℕ := 15
def length_of_grid : ℕ := 10
def width_of_grid : ℕ := 5
def height_of_grid : ℕ := 3
def side_length : ℕ := length_of_grid / width_of_grid -- 2 units
def num_horizontal_wires : ℕ := height_of_grid + 1    -- 4 wires
def num_vertical_wires : ℕ := width_of_grid + 1      -- 6 wires
def total_length_horizontal_wires : ℕ := num_horizontal_wires * length_of_grid -- 40 units
def total_length_vertical_wires : ℕ := num_vertical_wires * (height_of_grid * side_length) -- 36 units

-- The theorem to prove the total length of wire needed
theorem total_length_of_wire : total_length_horizontal_wires + total_length_vertical_wires = 76 :=
by
  sorry

end total_length_of_wire_l385_385591


namespace largest_n_sum_pos_l385_385923

section
variables {a : ℕ → ℤ}
variables {d : ℤ}
variables {n : ℕ}

axiom a_seq (n : ℕ) : a n = a 1 + (n - 1) * d
axiom a1_pos : a 1 > 0
axiom a2013_2014_pos : a 2013 + a 2014 > 0
axiom a2013_2014_neg : a 2013 * a 2014 < 0

theorem largest_n_sum_pos :
  ∃ n : ℕ, (∀ k ≤ n, (k * (2 * a 1 + (k - 1) * d) / 2) > 0) → n = 4026 := sorry

end

end largest_n_sum_pos_l385_385923


namespace evaluate_custom_operation_l385_385212

def custom_operation (x y : ℕ) : ℕ := 2 * x - 4 * y

theorem evaluate_custom_operation :
  custom_operation 7 3 = 2 :=
by
  sorry

end evaluate_custom_operation_l385_385212


namespace question_1_conjecture_question_2_proof_l385_385691

def a_seq : (ℕ → ℝ) := sorry

axiom a_seq_def : ∀ n : ℕ, a_seq (n + 1) = (a_seq n)^2 - n * (a_seq n) + 1

theorem question_1_conjecture (a1 : ℝ) (h : a1 = 2) :
  a_seq 2 = 3 ∧ a_seq 3 = 4 ∧ a_seq 4 = 5 ∧ (∀ n : ℕ, a_seq n = n + 1) := by
  have h₁ : a_seq 1 = 2 := sorry
  have h₂ : a_seq 2 = a_seq 1 ^ 2 - 1 * (a_seq 1) + 1 := by rw [a_seq_def 1]; sorry
  have h₃ : a_seq 3 = a_seq 2 ^ 2 - 2 * (a_seq 2) + 1 := by rw [a_seq_def 2]; sorry
  have h₄ : a_seq 4 = a_seq 3 ^ 2 - 3 * (a_seq 3) + 1 := by rw [a_seq_def 3]; sorry
  sorry  -- complete the remaining parts of the theorem

theorem question_2_proof (a1 : ℝ) (h : a1 ≥ 2) :
  ∀ n : ℕ, a_seq n ≥ n + 1 := by
  intros n 
  induction n with k hk
  case zero =>
    sorry  -- base case
  case succ =>
    sorry  -- inductive step

end question_1_conjecture_question_2_proof_l385_385691


namespace third_term_of_arithmetic_sequence_l385_385230

theorem third_term_of_arithmetic_sequence (a d : ℝ) (h : a + (a + 4 * d) = 10) : a + 2 * d = 5 :=
by {
  sorry
}

end third_term_of_arithmetic_sequence_l385_385230


namespace expression1_expression2_l385_385056

noncomputable theory

variable {a b x : ℝ}

theorem expression1 : (2 + 1/4)^(1/2) - 0.3^0 - 16^(-3/4) = 3/8 := by
  sorry

theorem expression2 : 4^(Real.log 5 / Real.log 4) - Real.log (Real.exp 5) + Real.log10 500 + Real.log10 2 = 3 := by
  sorry

end expression1_expression2_l385_385056


namespace problem1_problem2_l385_385949

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x + 1 / x

theorem problem1 (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) (h3 : x2 ≤ 1) :
  a = 1/2 → f a x1 - f a x2 > 0 :=
by
  assume ha : a = 1/2
  have hx1 : 0 < x1 ∧ x1 < x2 ∧ x2 ≤ 1 := ⟨h1, h2, h3⟩
  sorry

theorem problem2 (a : ℝ) (h : ∀ x ∈ Icc 0 1, f a x ≥ 6) : a ≥ 9/2 :=
by
  assume h : ∀ x ∈ Icc 0 1, f a x ≥ 6
  sorry

end problem1_problem2_l385_385949


namespace solve_expression_l385_385657

theorem solve_expression :
  let a := 0.76
  let numerator := a * a * a - 0.008
  let denominator := a * a + a * 0.2 + 0.04
  numerator / denominator ≈ 0.56 :=
by
  sorry

end solve_expression_l385_385657


namespace length_MN_l385_385246

noncomputable def triangle (A B C : Type)
  [Inhabited A] [Inhabited B] [Inhabited C] :=
classical.some sorry
-- Definitions of points, lines, perpendiculars, and angle bisectors
-- Using sorry to handle complex constructions of triangle and points

variables (A B C L K M N : Type)
  [Inhabited A] [Inhabited B] [Inhabited C]
  [Inhabited L] [Inhabited K] [Inhabited M] [Inhabited N]

-- Conditions and given lengths
def AB : ℝ := 130
def AC : ℝ := 144
def BC : ℝ := 150

-- Definitions of points L and K
def L := sorry
def K := sorry

-- Definitions of points M and N
def M := sorry
def N := sorry

-- Proof of the length segment MN is 82
theorem length_MN : MN A B C L K M N = 82 := 
sorry

end length_MN_l385_385246


namespace extremum_when_a_one_monotonicity_of_f_l385_385551

-- Define the function f(x) with variable a
def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x + a) - x

-- Define the first part of the proof problem
theorem extremum_when_a_one : 
  ∃ x_min : ℝ, (f 1 x_min = 2) ∧ (∀ x : ℝ, f 1 x ≥ f 1 x_min) :=
sorry

-- Define the second part of the proof problem
theorem monotonicity_of_f (a : ℝ) : 
  (a ≤ 0 → ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f a x₁ ≥ f a x₂) ∧
  (a > 0 → 
    (∀ x₁ x₂ : ℝ, x₁ < -Real.log a → x₁ ≤ x₂ → f a x₁ ≥ f a x₂) ∧ 
    (∀ x₁ x₂ : ℝ, x₁ > -Real.log a → x₁ ≤ x₂ → f a x₁ ≤ f a x₂)) :=
sorry

end extremum_when_a_one_monotonicity_of_f_l385_385551


namespace factorize_difference_of_squares_l385_385090

variable (x : ℝ)

theorem factorize_difference_of_squares :
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorize_difference_of_squares_l385_385090


namespace equal_binomial_coefficients_binomial_coefficient_max_binomial_expansion_sum_squares_l385_385399

-- Proof problem statement for Problem 1
theorem equal_binomial_coefficients (n : ℕ) (h : choose n 5 * 2^5 = choose n 6 * 2^6) :
  n = 8 := sorry

theorem binomial_coefficient_max (n : ℕ) (h : n = 8) :
  nat.choose 8 4 * 2^4 = 1120 := sorry

-- Proof problem statement for Problem 2
theorem binomial_expansion_sum_squares (a : ℕ → ℤ) (h : ∀ x : ℝ, (2 - real.sqrt 3 * x)^50 = ∑ i in finset.range 51, a i * x^i) :
  (∑ i in finset.range 51, if even i then a i else 0)^2 - (∑ i in finset.range 51, if ¬even i then a i else 0)^2 = 1 := sorry

end equal_binomial_coefficients_binomial_coefficient_max_binomial_expansion_sum_squares_l385_385399


namespace find_b3_plus_b11_l385_385239

noncomputable def a (n: ℕ) : ℚ := (3/4) * (2 ^ (n - 1))

def b (n: ℕ) : ℚ := 3 + (n - 7) * d -- placeholder for the arithmetic sequence difference

theorem find_b3_plus_b11 :
  let q := 2
  let product := a 2 * a 3 * a 4
  let b7_eq_a5 := b 7 = a 5
  let d := b 8 - b 7 -- derive difference d from two consecutive terms (b8 and b7)
  product = 27 / 64
→ b 3 + b 11 = 6 :=
by
  intros q product b7_eq_a5 d
  sorry

end find_b3_plus_b11_l385_385239


namespace total_roses_l385_385767

theorem total_roses (a : ℕ) (x y k : ℕ) (h1 : 300 ≤ a) (h2 : a ≤ 400)
  (h3 : a = 21 * x + 13) (h4 : a = 15 * y - 8) (h5 : a + 8 = 105 * k) :
  a = 307 :=
sorry

end total_roses_l385_385767


namespace cards_red_side_up_count_l385_385764

theorem cards_red_side_up_count : 
  ∀ (deck : Finset ℕ), 
  (∀ n, n ∈ deck ↔ n ∈ (Finset.range 100).map (λ x, x + 1)) →
  let seq1 := deck.filter (λ n, n % 2 = 0) in
  let seq2 := (deck \ seq1).filter (λ n, n % 3 = 0) in
  let seq3 := seq1.filter (λ n, n % 3 = 0) in
  (deck.card - seq1.card - seq2.card + seq3.card) = 49 :=
by
  sorry

end cards_red_side_up_count_l385_385764


namespace additional_fee_per_minute_for_second_plan_l385_385420

theorem additional_fee_per_minute_for_second_plan :
  (∃ x : ℝ, (22 + 0.13 * 280 = 8 + x * 280) ∧ x = 0.18) :=
sorry

end additional_fee_per_minute_for_second_plan_l385_385420


namespace common_focus_hyperbola_ellipse_l385_385941

theorem common_focus_hyperbola_ellipse (p : ℝ) (c : ℝ) :
  (0 < p ∧ p < 8) →
  (c = Real.sqrt (3 + 1)) →
  (c = Real.sqrt (8 - p)) →
  p = 4 := by
sorry

end common_focus_hyperbola_ellipse_l385_385941


namespace neg_p_eq_exist_l385_385313

theorem neg_p_eq_exist:
  (¬ ∀ a b : ℝ, a^2 + b^2 ≥ 2 * a * b) ↔ ∃ a b : ℝ, a^2 + b^2 < 2 * a * b := by
  sorry

end neg_p_eq_exist_l385_385313


namespace Janos_Shortest_Route_l385_385601

-- Defining points and reflections
variables (T1 T2 T1' T2' M N : Type) [MetricSpace T1] [MetricSpace T2] [MetricSpace T1'] [MetricSpace T2'] [MetricSpace M] [MetricSpace N]
variables (P Q : Type) [MetricSpace P] [MetricSpace Q]

-- Defining the constraints from the problem
def shortest_path : Prop :=
  ∀ (T1 T2 T1' T2' M N P Q : Type) [MetricSpace T1] [MetricSpace T2] [MetricSpace T1'] [MetricSpace T2'] [MetricSpace M] [MetricSpace N] [MetricSpace P] [MetricSpace Q],
  T1 ≠ T2 → T1' ≠ T2' → M ≠ N →
  (dist T1' P + dist P Q + dist Q T2' ≥ dist T1' T2') →
  dist T1 M + dist M N + dist N T2 = dist T1' P + dist P Q + dist Q T2'

-- Lean statement of the problem
theorem Janos_Shortest_Route : shortest_path :=
by sorry

end Janos_Shortest_Route_l385_385601


namespace cosine_210_l385_385847

noncomputable def Q := rotate 210 (1, 0)
noncomputable def E := foot Q (1, 0) (-1, 0)

theorem cosine_210 : cos 210 = - (sqrt 3/2) :=
sorry

end cosine_210_l385_385847


namespace centroids_form_parallelogram_l385_385785

-- Definition and conditions for the problem
variables (A B C D O G1 G2 G3 G4 : Type)
variables [plane_geometry A B C D O] [intersection_of_diagonals A B C D O]
variables [centroids_of_triangles A B C D O G1 G2 G3 G4]

-- Lean 4 statement of the proof problem
theorem centroids_form_parallelogram (h1 : is_quadrilateral A B C D)
  (h2 : intersection_point A C B D O)
  (h3 : centroid_of_triangle A O B G1)
  (h4 : centroid_of_triangle B O C G2)
  (h5 : centroid_of_triangle C O D G3)
  (h6 : centroid_of_triangle D O A G4) :
  is_parallelogram G1 G2 G3 G4 :=
sorry

end centroids_form_parallelogram_l385_385785


namespace infinite_perfect_squares_in_sequence_l385_385469

def sequence (n : ℕ) : ℕ := ⌊ n * real.sqrt 2 ⌋

theorem infinite_perfect_squares_in_sequence :
  ∃ (infinitely_many : ∀ N : ℕ, ∃ n ≥ N, (∃ m : ℕ, sequence n = m^2)) :=
sorry

end infinite_perfect_squares_in_sequence_l385_385469


namespace two_digit_number_difference_l385_385438

theorem two_digit_number_difference (a : ℤ) : 
  let tens_digit := 3 * a - 1 in
  let number := 10 * tens_digit + a in
  let neg_number := -(number) in
  neg_number - (-12) = -31 * a + 22 :=
by
  let tens_digit := 3 * a - 1
  let number := 10 * tens_digit + a
  let neg_number := -(number)
  sorry

end two_digit_number_difference_l385_385438


namespace shaded_region_area_l385_385786

structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A B C : Point

def area (T : Triangle) : ℝ :=
  0.5 * (T.B.x - T.A.x) * (T.C.y - T.A.y)

theorem shaded_region_area :
  let A := Point.mk 0 0
  let B := Point.mk 0 8
  let C := Point.mk 12 8
  let D := Point.mk 12 0
  let E := Point.mk 24 0
  let F := Point.mk 18 8
  let G := Point.mk 18 2
  let T := Triangle.mk D E G
  area T = 12 := sorry

end shaded_region_area_l385_385786


namespace cos_210_eq_neg_sqrt3_div_2_l385_385859

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385859


namespace diameter_of_figure_l385_385797

-- Define the sides of the triangle.
variables (a b c : ℝ)

-- Define the semi-perimeter of the triangle.
def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

-- Prove that the diameter of the figure F is the semi-perimeter of the triangle.
theorem diameter_of_figure (a b c : ℝ) : 
  diameter (triangle_with_semicircles a b c) = semi_perimeter a b c := 
sorry

end diameter_of_figure_l385_385797


namespace evaluate_ceiling_sum_l385_385082

theorem evaluate_ceiling_sum : 
  let a := (16 : ℝ) / 9
  in (⌈real.sqrt a⌉ + ⌈a⁻¹⌉ + ⌈a ^ 3⌉) = 9 := 
by 
  let a := (16 : ℝ) / 9
  sorry

end evaluate_ceiling_sum_l385_385082


namespace find_total_roses_l385_385768

open Nat

theorem find_total_roses 
  (a : ℕ)
  (h1 : 300 ≤ a)
  (h2 : a ≤ 400)
  (h3 : a % 21 = 13)
  (h4 : a % 15 = 7) : 
  a = 307 := 
sorry

end find_total_roses_l385_385768


namespace no_power_of_2_digit_permutation_l385_385481

theorem no_power_of_2_digit_permutation :
    (∀ n m : ℕ, n ≠ m → (∃ k : ℕ, 2 ^ k = n) → (∃ l : ℕ, 2 ^ l = m) → ¬perm n m) :=
by
  sorry

end no_power_of_2_digit_permutation_l385_385481


namespace explicit_formula_for_f_l385_385568

def f : ℝ → ℝ := λ x, 2 * x - x^2

theorem explicit_formula_for_f :
  ∀ x ∈ set.Icc (0 : ℝ) 2, f(x) = 2 * x - x^2 := 
by 
  sorry

end explicit_formula_for_f_l385_385568


namespace frac_p_over_q_l385_385285

noncomputable def p : ℝ := sorry
noncomputable def q : ℝ := sorry

-- Assume p and q are nonzero real numbers and (3 - 4i) * (p + qi) is purely imaginary
axiom h1 : p ≠ 0
axiom h2 : q ≠ 0
axiom h3 : ∃ r : ℝ, (3 - 4 * Complex.i) * (p + q * Complex.i) = r * Complex.i

-- Prove that p/q = -4/3
theorem frac_p_over_q : (p / q) = -4 / 3 := by
  sorry

end frac_p_over_q_l385_385285


namespace g_at_10_is_300_l385_385613

-- Define the function g and the given condition about g
def g: ℕ → ℤ := sorry

axiom g_cond (m n: ℕ) (h: m ≥ n): g (m + n) + g (m - n) = 2 * g m + 3 * g n
axiom g_1: g 1 = 3

-- Statement to be proved
theorem g_at_10_is_300 : g 10 = 300 := by
  sorry

end g_at_10_is_300_l385_385613


namespace cos_210_eq_neg_sqrt3_div_2_l385_385873

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385873


namespace tangent_line_eq_f_non_positive_l385_385958

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.log x - m * x + 1

-- Given condition that f(x) has an extreme value at x = 1
axiom has_extreme_value_at_one (m : ℝ) : (∇ (Real.log x - m * x + 1) 1) = (0 : ℝ)

-- Question (Ⅰ): Equation of the tangent line at x = 1/e
theorem tangent_line_eq (m : ℝ) (h : has_extreme_value_at_one m) :
  m = 1 →
  ∀ x₀, x₀ = 1/Real.exp 1 →
  (y : ℝ) = -(1/Real.exp 1) + (Real.exp 1 - 1) * (x - 1/Real.exp 1)
    :=
sorry

-- Question (Ⅱ): Prove that f(x) ≤ 0
theorem f_non_positive (m : ℝ) (h : has_extreme_value_at_one m) :
  ∀ x, (f x 1) ≤ 0
    :=
sorry

end tangent_line_eq_f_non_positive_l385_385958


namespace contractor_paint_cans_l385_385406

/-- Given the following conditions:
1. Initially, there was enough paint for 50 small rooms.
2. Five cans of paint were lost, leaving enough paint for only 38 rooms.
3. Each larger room requires twice as much paint as a small room.
4. The contractor decides to paint 35 small rooms and 5 larger rooms.

Prove that the contractor used 19 cans of paint for these rooms. --/
theorem contractor_paint_cans :
  (initial_rooms : ℕ) (cans_lost : ℕ) (final_rooms : ℕ) (small_rooms : ℕ) (large_rooms : ℕ)
  (covers_rooms_per_can : ℕ → ℝ) 
  (H1 : initial_rooms = 50)
  (H2 : cans_lost = 5)
  (H3 : final_rooms = 38)
  (H4 : small_rooms = 35)
  (H5 : large_rooms = 5)
  (H6 : covers_rooms_per_can 1 = 12 / 5)
  
  : (⌈(small_rooms + 2 * large_rooms) * (1 / covers_rooms_per_can 1)⌉.to_nat = 19) := 
by
  sorry

end contractor_paint_cans_l385_385406


namespace length_increase_147_days_l385_385634

theorem length_increase_147_days (x : ℝ) : 
  (∏ k in finset.range 147, (k + 4) / (k + 3)) * x = 50 * x := 
by 
  sorry

end length_increase_147_days_l385_385634


namespace isosceles_triangle_y_value_l385_385025

theorem isosceles_triangle_y_value :
  ∃ y : ℝ, (y = 1 + Real.sqrt 51 ∨ y = 1 - Real.sqrt 51) ∧ 
  (Real.sqrt ((y - 1)^2 + (4 - (-3))^2) = 10) :=
by sorry

end isosceles_triangle_y_value_l385_385025


namespace range_of_a_l385_385953

noncomputable def f (x a : ℝ) : ℝ :=
  (1/2) * x^2 - (a + 2) * x + 2 * a * Real.log x + 1

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ set.Ioo 4 6, deriv (λ x, f x a) x = 0) ↔ a ∈ set.Ioo 4 6 :=
by
  sorry

end range_of_a_l385_385953


namespace g_symmetric_about_pi_over_3_l385_385178

variable (a : ℝ)

def f (x : ℝ) : ℝ := a * sin x + cos x
def g (x : ℝ) : ℝ := sin x + a * cos x

theorem g_symmetric_about_pi_over_3
  (h : ∀ x, f (x) = f (π/3 - x)) : ∀ x, g (x) = g (π/3 - x) :=
by
  sorry

end g_symmetric_about_pi_over_3_l385_385178


namespace tan_phi_solution_l385_385236

noncomputable def tan_phi 
  (β : ℝ) (φ : ℝ) 
  (hβ : β > 0 ∧ β < π / 2)
  (h_right_triangle : ∀ (β : ℝ), is_right_triangle (triangle.mk β (π / 2))) 
  (h_tan_half_beta : tan (β / 2) = 2 / real.cbrt(3))
  (median : ℝ) 
: ℝ := 
  tan (φ)

theorem tan_phi_solution 
  (β φ : ℝ) 
  (hβ : β > 0 ∧ β < π / 2)
  (h_right_triangle : ∀ (β : ℝ), is_right_triangle (triangle.mk β (π / 2))) 
  (h_tan_half_beta : tan (β / 2) = 2 / real.cbrt(3))
  (median : ℝ) 
  (angle_bisector : ℝ) 
: tan (φ) = 8 / (real.cbrt(27) - 4) :=
sorry

end tan_phi_solution_l385_385236


namespace min_value_of_sum_of_squares_l385_385288

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : x^2 + y^2 + z^2 ≥ 4.8 :=
sorry

end min_value_of_sum_of_squares_l385_385288


namespace range_of_a_l385_385300

def A (a : ℝ) : set ℝ := {x | x^2 - 2 * a * x + a^2 - 1 < 0}
def B : set ℝ := {x | x^2 - 6 * x + 5 < 0}

theorem range_of_a (a : ℝ) (h : A a ∩ B = ∅) : a ≥ 6 ∨ a ≤ 0 :=
sorry

end range_of_a_l385_385300


namespace min_value_of_sum_squares_l385_385290

theorem min_value_of_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) :
    x^2 + y^2 + z^2 ≥ 10 :=
sorry

end min_value_of_sum_squares_l385_385290


namespace sanghyeon_questions_l385_385347

variable (S : ℕ)

theorem sanghyeon_questions (h1 : S + (S + 5) = 43) : S = 19 :=
by
    sorry

end sanghyeon_questions_l385_385347


namespace emily_spent_20_dollars_l385_385487

/-- Let X be the amount Emily spent on Friday. --/
variables (X : ℝ)

/-- Emily spent twice the amount on Saturday. --/
def saturday_spent := 2 * X

/-- Emily spent three times the amount on Sunday. --/
def sunday_spent := 3 * X

/-- The total amount spent over the three days is $120. --/
axiom total_spent : X + saturday_spent X + sunday_spent X = 120

/-- Prove that X = 20. --/
theorem emily_spent_20_dollars : X = 20 :=
sorry

end emily_spent_20_dollars_l385_385487


namespace third_box_number_l385_385778

def N : ℕ := 301

theorem third_box_number (N : ℕ) (h1 : N % 3 = 1) (h2 : N % 4 = 1) (h3 : N % 7 = 0) :
  ∃ x : ℕ, x > 4 ∧ x ≠ 7 ∧ N % x = 1 ∧ (∀ y > 4, y ≠ 7 → y < x → N % y ≠ 1) ∧ x = 6 :=
by
  sorry

end third_box_number_l385_385778


namespace cosine_210_l385_385845

noncomputable def Q := rotate 210 (1, 0)
noncomputable def E := foot Q (1, 0) (-1, 0)

theorem cosine_210 : cos 210 = - (sqrt 3/2) :=
sorry

end cosine_210_l385_385845


namespace range_of_a_l385_385960

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x > 1 then (a / x) + 2 else -x^2 + 2 * x

theorem range_of_a {a : ℝ} :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > 0) →
  a ∈ set.Ico (-1 : ℝ) 0 :=
by
  sorry

end range_of_a_l385_385960


namespace men_apples_l385_385717

theorem men_apples (M W : ℕ) (h1 : M = W - 20) (h2 : 2 * M + 3 * W = 210) : M = 30 :=
by
  -- skipping the proof
  sorry

end men_apples_l385_385717


namespace awards_distribution_l385_385650

theorem awards_distribution :
  ∃ (ways : ℕ), ways = 6300 ∧
    (∀ (awards students : ℕ), awards = 7 → students = 4 → 
      (∃ (distribution : list (list ℕ)), 
        (∀ d ∈ distribution, 1 ≤ length d ∧ length d ≤ 3) ∧
        (∀ s ∈ distribution, ∃ a ∈ s, a ∈ list.range 1 8) ∧
        list.sum (list.map list.length distribution) = awards)) := 
begin
  use 6300,
  split,
  { reflexivity, },
  { intros awards students h_awards h_students,
    use [[1,2,3], [4,5], [6], [7]],
    split,
    { intros d h_d,
      split;
      linarith [h_awards, h_students], },
    { intros s h_s,
      repeat { assumption <|> linarith, },
    exact sorry },
end

end awards_distribution_l385_385650


namespace smallest_prime_less_than_square_l385_385372

theorem smallest_prime_less_than_square : ∃ p n : ℕ, Prime p ∧ p = n^2 - 20 ∧ p = 5 :=
by 
  sorry

end smallest_prime_less_than_square_l385_385372


namespace chord_divided_into_three_equal_parts_l385_385039

-- Definitions of the circles and the point
variables (radius_outer radius_inner : ℝ) [h_pos_outer : radius_outer > 0] [h_pos_inner : radius_inner > 0]
variables (center : ℝ × ℝ)
variables (point_A : ℝ × ℝ) (h_on_outer : dist center point_A = radius_outer)

-- Statement to prove
theorem chord_divided_into_three_equal_parts 
  (h_concentric : center = center)
  (h_radius_ineq : radius_inner < radius_outer)
  : ∃ chord, is_chord_of circle_outer circle_inner (point_A) chord ∧ 
             chord_is_divided_into_three_equal_parts_by_inner_circle chord circle_inner := 
sorry

end chord_divided_into_three_equal_parts_l385_385039


namespace problem_1_problem_2a_problem_2b_problem_3_l385_385959

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^2 + b * Real.log (x + 1)

theorem problem_1 (x : ℝ) : 
  (∀ (x : ℝ), x > -1 → (2*x + 1) / (2*(x + 1)) ≥ 0 ) 
  -> monotone (f x 0.5) sorry

theorem problem_2a (x : ℝ) (b : ℝ) (hb : b < 0) :
  (∃ x2 : ℝ, x2 = ( -1 + Real.sqrt(1 - 2*b))/2 ) → 
  ∀ x > -1 , f x b = x2

theorem problem_2b (x : ℝ) (b : ℝ) (hb0 : 0 < b) (hb1 : b < 0.5) :
  (∃ x1 x2 : ℝ, x1 = ( -1 - Real.sqrt(1 - 2*b))/2) →
  ( ∃ x2 = ( -1 + Real.sqrt(1 - 2*b))/2 ) →  
  ( ∀ x > -1 ,f x b = (x1, x2))

theorem problem_3 (n : ℕ) (hn : n >0 ):
  Real.log (1 / n + 1) > (1 / n^2)  - (1 / n^3) 

end problem_1_problem_2a_problem_2b_problem_3_l385_385959


namespace count_valid_three_digit_numbers_l385_385521

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def are_digits (a b c : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10

def not_zero (a : ℕ) : Prop :=
  a ≠ 0

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧ (a + b > c ∧ b + c > a ∧ c + a > b)

theorem count_valid_three_digit_numbers : 
  ∃ (count : ℕ), count = 165 ∧
  ∀ n, is_three_digit_number n → 
  let a := n / 100,
      b := (n / 10) % 10,
      c := n % 10 in
  are_digits a b c →
  not_zero a →
  is_isosceles_triangle a b c →
  (∃ (L : List ℕ), n ∈ L ∧ L.length = count) :=
sorry

end count_valid_three_digit_numbers_l385_385521


namespace pumpkin_seeds_per_row_l385_385410

-- Conditions
def beans_seedlings := 64
def beans_per_row := 8
def total_pumpkin_seeds := 84
def radishes := 48
def radishes_per_row := 6
def rows_per_bed := 2
def total_beds := 14

-- Proof statement
theorem pumpkin_seeds_per_row : 
  let total_rows := total_beds * rows_per_bed,
      beans_rows := beans_seedlings / beans_per_row,
      radishes_rows := radishes / radishes_per_row,
      pumpkin_rows := total_rows - beans_rows - radishes_rows
  in total_pumpkin_seeds / pumpkin_rows = 7 :=
by
  sorry

end pumpkin_seeds_per_row_l385_385410


namespace C_pays_correct_amount_l385_385792

variable (P_A : ℝ) (r_A : ℝ) (r_B : ℝ)

def selling_price_A_to_B (P_A r_A : ℝ) : ℝ := P_A * (1 + r_A)

def selling_price_B_to_C (P_B r_B : ℝ) : ℝ := P_B * (1 + r_B)

theorem C_pays_correct_amount (P_A : ℝ) (r_A : ℝ) (r_B : ℝ) (P_C : ℝ) 
    (h1 : P_A = 150) (h2 : r_A = 0.20) (h3 : r_B = 0.25) : 
    selling_price_B_to_C (selling_price_A_to_B P_A r_A) r_B = P_C := 
begin
    sorry
end

#check C_pays_correct_amount 150 0.20 0.25 225 sorry sorry sorry

end C_pays_correct_amount_l385_385792


namespace S7_eq_14_l385_385066

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

variables (a : ℕ → ℤ) (S : ℕ → ℤ) (h_arith_seq : arithmetic_sequence a)
variables (h_a3 : a 3 = 0) (h_a6_plus_a7 : a 6 + a 7 = 14)

theorem S7_eq_14 : S 7 = 14 := sorry

end S7_eq_14_l385_385066


namespace cos_210_eq_neg_sqrt3_div_2_l385_385857

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385857


namespace remainder_8354_11_l385_385723

theorem remainder_8354_11 : 8354 % 11 = 6 := sorry

end remainder_8354_11_l385_385723


namespace fold_length_is_correct_l385_385022

variables (x y : ℝ)

-- Given conditions
-- sides of the rectangle
def side_AB := x
def side_BC := y

-- Hypothesis: x >= y (implicitly assumed through steps in the solution)
axiom xy_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ x ≥ y

-- Define the length of the diagonal when opposite vertices are brought together
def diagonal : ℝ := Real.sqrt (x^2 + y^2)

-- Define the length of the fold line PQ
def fold_length : ℝ := (y * diagonal) / x

-- Prove the fold length is as calculated
theorem fold_length_is_correct :
  fold_length x y = (y * Real.sqrt (x^2 + y^2)) / x :=
by sorry

end fold_length_is_correct_l385_385022


namespace Nancy_weighs_90_pounds_l385_385504

theorem Nancy_weighs_90_pounds (W : ℝ) (h1 : 0.60 * W = 54) : W = 90 :=
by
  sorry

end Nancy_weighs_90_pounds_l385_385504


namespace percent_increase_share_price_l385_385001

theorem percent_increase_share_price (P : ℝ) 
  (h1 : ∃ P₁ : ℝ, P₁ = P + 0.25 * P)
  (h2 : ∃ P₂ : ℝ, P₂ = P + 0.80 * P)
  : ∃ percent_increase : ℝ, percent_increase = 44 := by
  sorry

end percent_increase_share_price_l385_385001


namespace probability_of_distinct_multiple_of_five_l385_385448

-- Conditions
def isFourDigitInteger (n : ℕ) : Prop := n >= 1000 ∧ n <= 9999
def isMultipleOfFive (n : ℕ) : Prop := n % 5 = 0
def digitsAreDistinct (n : ℕ) : Prop := (n.digits).nodup

-- Question
def satisfiesConditions (n : ℕ) : Prop :=
  isFourDigitInteger n ∧ isMultipleOfFive n ∧ digitsAreDistinct n

-- Answer
theorem probability_of_distinct_multiple_of_five :
  (∃ nn : ℕ, nn > 0 ∧ nn ≤ 9000 ∧ satisfiesConditions nn) ∧
  ∀ n, satisfiesConditions n → (1 : ℚ) * ⟨119, 1125⟩ := 
sorry

end probability_of_distinct_multiple_of_five_l385_385448


namespace cos_210_eq_neg_sqrt3_div2_l385_385834

theorem cos_210_eq_neg_sqrt3_div2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div2_l385_385834


namespace sum_of_arguments_eq_14pi_l385_385358

noncomputable def sum_of_arguments (n : ℕ) : ℝ :=
  ∑ k in finset.range (2 * n), θ k

variable (n : ℕ)

theorem sum_of_arguments_eq_14pi
  (h : ∀ z : ℂ, z ^ 28 - z ^ 8 - 1 = 0 →
     ∃ θ : ℝ, (z = complex.exp (θ * complex.I)) ∧
     (0 ≤ θ) ∧ (θ < 2 * real.pi) ∧
     ∀ m : ℕ, m < 2 * n → z = complex.exp (θ * complex.I) →
     (∑ k in finset.range (2 * n), θ k) = 14 * real.pi
  ) : sum_of_arguments n = 14 * real.pi := sorry

end sum_of_arguments_eq_14pi_l385_385358


namespace sum_first_10_terms_arithmetic_seq_l385_385276

theorem sum_first_10_terms_arithmetic_seq :
  ∀ (a : ℕ → ℤ), (a 1 = -2) → (a 2 + a 6 = 2) →
  (∃ S₁₀, S₁₀ = 10 * a 1 + (10 * (10 - 1)) / 2 * (a 2 - a 1) / (2 - 1) ∧ S₁₀ = 25) :=
begin
  assume a,
  assume h1 : a 1 = -2,
  assume h2 : a 2 + a 6 = 2,
  sorry
end

end sum_first_10_terms_arithmetic_seq_l385_385276


namespace angle_bisector_on_perpendicular_bisector_l385_385355

theorem angle_bisector_on_perpendicular_bisector (A B C M : Point) (circumcircle : Circle Point) (angle_bisector : Line Point) 
  (h1 : A ∈ circumcircle) (h2 : B ∈ circumcircle) (h3 : C ∈ circumcircle) 
  (h4 : is_angle_bisector angle_bisector ∠A) (h5 : (angle_bisector ∩ circumcircle = {M})) :
  is_on_perpendicular_bisector M B C :=
sorry

end angle_bisector_on_perpendicular_bisector_l385_385355


namespace no_square_in_equilateral_triangle_lattice_l385_385689

noncomputable theory

open_locale classical

-- Define the setup of the plane divided into equilateral triangles
def equilateral_triangle_lattice :=
  {p : ℝ × ℝ // ∃ i j k : ℤ, p = (i + j * 0.5, j * (√3 / 2)) ∨ p = (i + j * 0.5 - 1, j * (√3 / 2))}

-- The main statement asserting it's impossible to find four such vertices forming a square
theorem no_square_in_equilateral_triangle_lattice : ¬∃ (O A B C : equilateral_triangle_lattice),
  O ≠ A ∧ O ≠ B ∧ O ≠ C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
  dist O.1 A.1 = dist A.1 B.1 ∧ dist B.1 C.1 = dist C.1 O.1 :=
sorry

end no_square_in_equilateral_triangle_lattice_l385_385689


namespace fg_mul_eq_x_l385_385621

noncomputable def f (x : ℝ) : ℝ := x^2 / real.sqrt (x - 1)
noncomputable def g (x : ℝ) : ℝ := real.sqrt (x - 1) / x

theorem fg_mul_eq_x {x : ℝ} (h : 1 < x) : f x * g x = x :=
by
  sorry

end fg_mul_eq_x_l385_385621


namespace triangle_side_length_l385_385924

noncomputable def find_side_length (a b C : ℝ) : ℝ :=
real.sqrt (a^2 + b^2 - 2 * a * b * real.cos (C * real.pi / 180))

theorem triangle_side_length :
  find_side_length 2 1 60 = real.sqrt 3 :=
by
  sorry

end triangle_side_length_l385_385924


namespace probability_of_condition_l385_385784

noncomputable def probability_roots_condition := 
  let interval : Set ℝ := { k | 6 ≤ k ∧ k ≤ 10 }
  let quadratic (k : ℝ) (x : ℝ) : ℝ := (k^2 - 3*k - 10)*x^2 + (3*k - 8)*x + 2
  let condition (k : ℝ) (x1 x2 : ℝ) : Prop := x1 + x2 = (8 - 3*k) / (k^2 - 3*k - 10) 
                                  ∧ x1 * x2 = 2 / (k^2 - 3*k - 10)
                                  ∧ x1 ≤ 2*x2
  ∃ (k : ℝ) (x1 x2 : ℝ), k ∈ interval ∧ condition k x1 x2 ∧ k ≤ 22/3

theorem probability_of_condition :
  probability_roots_condition = 1 / 3 :=
sorry

end probability_of_condition_l385_385784


namespace factorize_difference_of_squares_l385_385086

variable (x : ℝ)

theorem factorize_difference_of_squares :
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorize_difference_of_squares_l385_385086


namespace find_sum_of_digits_of_n_l385_385123

theorem find_sum_of_digits_of_n (n : ℕ) (h_pos : n > 0) (h_eq : Real.logBase 2 (Real.logBase 8 n) = Real.logBase 2 (Real.logBase 2 n)) :
  n = 1 ∧ (Nat.digitSum n = 1) :=
by
  sorry

end find_sum_of_digits_of_n_l385_385123


namespace min_f_value_l385_385513

noncomputable def f (x y : ℝ) : ℝ := 
  real.sqrt (x^2 - 3 * x + 3) + 
  real.sqrt (y^2 - 3 * y + 3) + 
  real.sqrt (x^2 - real.sqrt 3 * x * y + y^2)

theorem min_f_value :
  ∀ x y : ℝ, 0 < x → 0 < y → 
  f x y ≥ real.sqrt 6 ∧ ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ f x y = real.sqrt 6 :=
by
  sorry

end min_f_value_l385_385513


namespace birds_more_than_half_sunflower_seeds_l385_385306

theorem birds_more_than_half_sunflower_seeds :
  ∃ (n : ℕ), n = 3 ∧ ((4 / 5)^n * (2 / 5) + (2 / 5) > 1 / 2) :=
by
  sorry

end birds_more_than_half_sunflower_seeds_l385_385306


namespace evaluate_f_neg2010_plus_f_2011_l385_385532

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 ≤ x ∧ x < 2 then log (x + 1) / log 2 else 0 -- Temporary definition for the statement

theorem evaluate_f_neg2010_plus_f_2011 :
  (∀ x, f (-x) = f x) ∧
  (∀ x ≥ 0, f (x + 2) = f x) ∧
  (∀ x, 0 ≤ x ∧ x < 2 → f x = log (x + 1) / log 2) →
  f (-2010) + f (2011) = 1 :=
begin
  sorry
end

end evaluate_f_neg2010_plus_f_2011_l385_385532


namespace first_four_match_last_four_l385_385411

-- Defining a finite sequence as a list of booleans (representing the digits 0 and 1)
def finite_seq := List Bool

-- Condition 1: Any five consecutive digits are unique across the sequence
def unique_quintuplets (S : finite_seq) : Prop :=
  ∀ i j : ℕ, i ≠ j → i + 4 < S.length → j + 4 < S.length → 
  (S.slice i (i + 5)) ≠ (S.slice j (j + 5))

-- Condition 2: If any digit is appended, the unique quintuplets property no longer holds
def not_unique_if_appended (S : finite_seq) : Prop :=
  ∀ b : Bool, ¬ unique_quintuplets (S ++ [b])

-- Main theorem statement: The first four digits match the last four digits
theorem first_four_match_last_four (S : finite_seq) :
  unique_quintuplets S → not_unique_if_appended S →
  (S.length ≥ 4) → (S.take 4 = S.drop (S.length - 4)) :=
by
  intros h_unique h_not_unique h_length_ge_4
  sorry


end first_four_match_last_four_l385_385411


namespace cos_210_eq_neg_sqrt3_div2_l385_385839

theorem cos_210_eq_neg_sqrt3_div2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div2_l385_385839


namespace pears_value_l385_385327

-- Condition: 3/4 of 12 apples is equivalent to 6 pears
def apples_to_pears (a p : ℕ) : Prop := (3 / 4) * a = 6 * p

-- Target: 1/3 of 9 apples is equivalent to 2 pears
def target_equiv : Prop := (1 / 3) * 9 = 2

theorem pears_value (a p : ℕ) (h : apples_to_pears 12 6) : target_equiv := by
  sorry

end pears_value_l385_385327


namespace determine_pairs_l385_385883

theorem determine_pairs (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔
  (∃ k : ℕ, k > 0 ∧ (a = 7 * k^2 ∧ b = 7 * k) ∨ (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1)) :=
by
  sorry

end determine_pairs_l385_385883


namespace factorize_x_squared_minus_1_l385_385099

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by
  -- Proof goes here
  sorry

end factorize_x_squared_minus_1_l385_385099


namespace max_sum_cd_l385_385823

theorem max_sum_cd (c d : ℕ) (hc : c > 0) (hd : d > 1) (hcd : c^d < 500) 
  (hmax : ∀ (c' d': ℕ), c' > 0 → d' > 1 → c'^d' < 500 → c'^d' ≤ c^d) : c + d = 24 := 
by
  have h1 : 22^2 = 484 := rfl
  have h2 : c = 22 ∧ d = 2 := by sorry
  exact by sorry

end max_sum_cd_l385_385823


namespace explicit_formula_inequality_condition_l385_385166

noncomputable def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ :=
  Real.exp x + x^2 / 2 + Real.log (x + m) + n

theorem explicit_formula (m n : ℝ) (hTangent : (e + 1) * 0 - e * (f 0 m n) + 3 * e = 0)
  (hDeriv : deriv (f x m n) x = Real.exp x + x + 1 / (x + m)) :
  f x e 1 = Real.exp x + x^2 / 2 + Real.log (x + e) + 1 :=
sorry

theorem inequality_condition (a : ℝ) (h : ∀ x ≥ 0, 
  f x e 1 ≥ x^2 / 2 + a * x + 3) :
  a ≤ 1 + 1 / e :=
sorry

end explicit_formula_inequality_condition_l385_385166


namespace slope_of_tangent_line_l385_385930

theorem slope_of_tangent_line (e : ℝ) (h : e = Real.exp 1) :
  ∀ (x : ℝ), (deriv (λ x, 2 * Real.exp x) 1) = 2 * e :=
by
  intro x
  have h_deriv : (deriv (λ x, 2 * Real.exp x) 1) = 2 * Real.exp 1 := by
    simp [Real.exp]
  rw [h] at h_deriv
  exact h_deriv

end slope_of_tangent_line_l385_385930


namespace probability_all_genuine_given_equal_weights_l385_385715

variables (coins : Finset ℕ) (c_genuine : Finset ℕ) (c_counterfeit : Finset ℕ)
variable {coin_weight : ℕ → ℝ}

def areWeightsEqual (x y : ℕ) (a b : ℕ) : Prop :=
  coin_weight x + coin_weight y = coin_weight a + coin_weight b

noncomputable def probAllGenuineGivenEqualWeights : ℚ :=
  (conditionedProbability (allSelectedAreGenuine coins c_genuine c_counterfeit) 
    (weightsOfPairsAreEqual coins coin_weight))

axiom allSelectedAreGenuine : ProbEvent coins
axiom weightsOfPairsAreEqual : ProbEvent coins

theorem probability_all_genuine_given_equal_weights :
  probAllGenuineGivenEqualWeights = Rat.ofInt 15 / 19 := sorry

end probability_all_genuine_given_equal_weights_l385_385715


namespace nina_shoe_payment_l385_385632

theorem nina_shoe_payment :
  let first_pair_original := 22
  let first_pair_discount := 0.10 * first_pair_original
  let first_pair_discounted := first_pair_original - first_pair_discount
  let first_pair_tax := 0.05 * first_pair_discounted
  let first_pair_final := first_pair_discounted + first_pair_tax

  let second_pair_original := first_pair_original * 1.50
  let second_pair_discount := 0.15 * second_pair_original
  let second_pair_discounted := second_pair_original - second_pair_discount
  let second_pair_tax := 0.07 * second_pair_discounted
  let second_pair_final := second_pair_discounted + second_pair_tax

  let total_payment := first_pair_final + second_pair_final
  total_payment = 50.80 :=
by 
  sorry

end nina_shoe_payment_l385_385632


namespace find_q_sum_l385_385516

variable (q : ℕ → ℕ)

def conditions :=
  q 3 = 2 ∧ 
  q 8 = 20 ∧ 
  q 16 = 12 ∧ 
  q 21 = 30

theorem find_q_sum (h : conditions q) : 
  (q 1 + q 2 + q 3 + q 4 + q 5 + q 6 + q 7 + q 8 + q 9 + q 10 + q 11 + 
   q 12 + q 13 + q 14 + q 15 + q 16 + q 17 + q 18 + q 19 + q 20 + q 21 + q 22) = 352 := 
  sorry

end find_q_sum_l385_385516


namespace range_of_k_correct_line_equation_correct_l385_385563

noncomputable def F1 := (-Real.sqrt 2, 0)
noncomputable def F2 := (Real.sqrt 2, 0)
def hyperbola (P : ℝ × ℝ) : Prop := ((P.1)^2 - (P.2)^2 = 1) ∧ (P.1 > 0)

def line (k : ℝ) (P : ℝ × ℝ) : Prop := P.2 = k * P.1 - 1

def intersects (k : ℝ) : Prop := ∃ A B : ℝ × ℝ, hyperbola A ∧ hyperbola B ∧ line k A ∧ line k B ∧ A ≠ B

def range_of_k : Set ℝ := {k : ℝ | 1 < k ∧ k < Real.sqrt 2 }

theorem range_of_k_correct :
  ∀ k, intersects k ↔ k ∈ range_of_k :=
sorry

theorem line_equation_correct :
  ∀ k, (intersects k ∧ ∃ A B, |AB| = 2 * Real.sqrt 5) →
  (k = Real.sqrt 6 / 2 ∧ ∀ P : ℝ × ℝ, line k P ↔ P.2 = (Real.sqrt 6 / 2) * P.1 - 1) :=
sorry

end range_of_k_correct_line_equation_correct_l385_385563


namespace factorize_expr_l385_385896

theorem factorize_expr (a b : ℝ) : a^2 - 2 * a * b = a * (a - 2 * b) := 
by 
  sorry

end factorize_expr_l385_385896


namespace courier_cannot_achieve_average_speed_l385_385668

theorem courier_cannot_achieve_average_speed :
  ∀ (total_distance : ℝ) (initial_distance_fraction initial_speed desired_avg_speed : ℝ),
  total_distance = 24 →
  initial_distance_fraction = 2/3 →
  initial_speed = 8 →
  desired_avg_speed = 12 →
  (let time_needed := total_distance / desired_avg_speed in
   let initial_distance := initial_distance_fraction * total_distance in
   let initial_time := initial_distance / initial_speed in
   let remaining_time := time_needed - initial_time in
   remaining_time ≠ 0 ∨ remaining_time * 0 = 0) := by
  sorry

end courier_cannot_achieve_average_speed_l385_385668


namespace find_n_l385_385917

-- We need a definition for permutations counting A_n^2 = n(n-1)
def permutations_squared (n : ℕ) : ℕ := n * (n - 1)

theorem find_n (n : ℕ) (h : permutations_squared n = 56) : n = 8 :=
by {
  sorry -- proof omitted as instructed
}

end find_n_l385_385917


namespace winner_more_than_third_l385_385456

theorem winner_more_than_third (W S T F : ℕ) (h1 : F = 199) 
(h2 : W = F + 105) (h3 : W = S + 53) (h4 : W + S + T + F = 979) : 
W - T = 79 :=
by
  -- Here, the proof steps would go, but they are not required as per instructions.
  sorry

end winner_more_than_third_l385_385456


namespace factorize_x_squared_minus_one_l385_385103

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_x_squared_minus_one_l385_385103


namespace cos_210_eq_neg_sqrt3_div_2_l385_385868

theorem cos_210_eq_neg_sqrt3_div_2 : real.cos (210 * real.pi / 180) = - (real.sqrt 3 / 2) :=
by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385868


namespace max_distance_between_bus_stops_l385_385782

theorem max_distance_between_bus_stops 
  (v_m : ℝ) (v_b : ℝ) (dist : ℝ) 
  (h1 : v_m = v_b / 3) (h2 : dist = 2) : 
  ∀ d : ℝ, d = 1.5 := sorry

end max_distance_between_bus_stops_l385_385782


namespace remainder_when_divided_by_9_l385_385127

open Nat

theorem remainder_when_divided_by_9 (A B : ℕ) (h : A = B * 9 + 13) : A % 9 = 4 :=
by
  sorry

end remainder_when_divided_by_9_l385_385127


namespace ratio_of_areas_is_one_l385_385809

def sq_side_length : ℝ := 2
def triangle_area (side_length: ℝ) : ℝ := 1/2 * (side_length / 2) * (side_length / 2)
def corner_square_area (side_length: ℝ) : ℝ := side_length * side_length / 4 - triangle_area side_length

theorem ratio_of_areas_is_one :
  let t := triangle_area sq_side_length
  let q := corner_square_area sq_side_length
  q / t = 1 := sorry

end ratio_of_areas_is_one_l385_385809


namespace problem_lean_l385_385616

variables (x y : ℝ)
variables (a b c : ℝ × ℝ)

def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0
def parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

theorem problem_lean :
  let a := (x, 2) in
  let b := (4, y) in
  let c := (1, -2) in
  (perpendicular a c) ∧ (parallel b c) →
  x = 4 ∧ y = -8 ∧ (|a.1 + b.1, a.2 + b.2| = 10) :=
by
  intros a b c h,
  unfold perpendicular at h,
  unfold parallel at h,
  sorry

end problem_lean_l385_385616


namespace reflection_problem_l385_385787

def midpoint (u v : ℝ × ℝ) := ((u.1 + v.1) / 2, (u.2 + v.2) / 2)
def projection (u v : ℝ × ℝ) :=
    let dot_product := (u.1 * v.1 + u.2 * v.2)
    let norm_sq := (v.1 * v.1 + v.2 * v.2)
    (dot_product / norm_sq * v.1, dot_product / norm_sq * v.2)

noncomputable def reflection (u v : ℝ × ℝ) :=
    let proj := projection u v
    let scalar := 2 - 1
    (scalar * proj.1 - u.1, scalar * proj.2 - u.2)

theorem reflection_problem :
  reflection (6, -3) (2, 9) = (3, -4) :=
by {
  sorry
}

end reflection_problem_l385_385787


namespace sarah_shaded_area_l385_385649

theorem sarah_shaded_area (r : ℝ) (A_square : ℝ) (A_circle : ℝ) (A_circles : ℝ) (A_shaded : ℝ) :
  let side_length := 27
  let radius := side_length / (3 * 2)
  let area_square := side_length * side_length
  let area_one_circle := Real.pi * (radius * radius)
  let total_area_circles := 9 * area_one_circle
  let shaded_area := area_square - total_area_circles
  shaded_area = 729 - 182.25 * Real.pi := 
by
  sorry

end sarah_shaded_area_l385_385649


namespace value_of_f_2017_l385_385143

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then log 2018 (1 - x) + 2 * cos (Real.pi * x / 2) + m
  else -f m (-x)

theorem value_of_f_2017 (m : ℝ) (h : f m 0 = 0) : f m 2017 = 1 := by
  -- The detailed proof can be left as an exercise to the reader
  sorry

end value_of_f_2017_l385_385143


namespace rectangle_to_square_area_ratio_l385_385344

theorem rectangle_to_square_area_ratio (s : ℝ) :
  let longer_side := 1.2 * s
  let shorter_side := 0.8 * s
  let area_R := longer_side * shorter_side
  let area_S := s^2
  (area_R / area_S) = (24 / 25) :=
by
  let longer_side := 1.2 * s
  let shorter_side := 0.8 * s
  let area_R := longer_side * shorter_side
  let area_S := s^2
  have area_R_is : area_R = 0.96 * s^2 := by
    sorry
  have area_S_is : area_S = s^2 := by
    sorry
  have ratio_is : (area_R / area_S) = (0.96 * s^2) / s^2 := by
    sorry
  have final_ratio_is : (0.96 * s^2) / s^2 = 0.96 := by
    sorry
  have equivalence : 0.96 = 24 / 25 := by
    sorry
  show (area_R / area_S) = 24 / 25
  by
    rw [ratio_is, final_ratio_is, equivalence]

end rectangle_to_square_area_ratio_l385_385344


namespace expected_total_babies_l385_385414

def young_couples := 0.20 * 100
def adult_couples := 0.60 * 100
def old_couples := 0.20 * 100 

def pregnancy_rate_young := 0.40
def pregnancy_rate_adult := 0.25
def pregnancy_rate_old := 0.10

def avg_babies_per_pregnancy := 1.5

def expected_pregnancies (couples : ℕ) (rate : ℝ) : ℝ := couples * rate
def expected_babies (pregnancies : ℝ) (avg_babies : ℝ) : ℝ := pregnancies * avg_babies

def young_babies := expected_babies (expected_pregnancies young_couples pregnancy_rate_young) avg_babies_per_pregnancy
def adult_babies := expected_babies (expected_pregnancies adult_couples pregnancy_rate_adult) avg_babies_per_pregnancy
def old_babies := expected_babies (expected_pregnancies old_couples pregnancy_rate_old) avg_babies_per_pregnancy

def total_babies := (young_babies + ⌊adult_babies⌋ + old_babies : ℝ)

theorem expected_total_babies : total_babies = 37 := by
  sorry

end expected_total_babies_l385_385414


namespace silvia_wins_with_800_l385_385233

theorem silvia_wins_with_800 :
  ∃ (N : ℕ), (0 ≤ N ∧ N ≤ 1999) ∧
  (∀ k, 2 * N + 100 * k ≤ 1999 → ∃ m, 2 * N + 100 * k + 100 = m ∧ m > 1999) :=
by
  use 800
  -- assumption exists that 0 ≤ N ∧ N ≤ 1999
  split
  -- proof of N in range [0, 1999]
  sorry
  -- proof Silvia wins with initial number 800
  sorry

end silvia_wins_with_800_l385_385233


namespace coupon1_greater_l385_385408

variable (x : ℝ)

def coupon1_discount (x : ℝ) : ℝ := 0.15 * x
def coupon2_discount : ℝ := 50
def coupon3_discount (x : ℝ) : ℝ := 0.25 * x - 62.5

theorem coupon1_greater (x : ℝ) (hx1 : 333.33 < x ∧ x < 625) : 
  coupon1_discount x > coupon2_discount ∧ coupon1_discount x > coupon3_discount x := by
  sorry

end coupon1_greater_l385_385408


namespace unique_root_of_f_l385_385950

def f (x a : ℝ) : ℝ := x^2 + 2 * a * log (x^2 + 2) / log 2 + a^2 - 3

theorem unique_root_of_f (a : ℝ) : (∀ x : ℝ, f x a = 0 ↔ x = 0) ↔ a = 1 :=
by {
  sorry
}

end unique_root_of_f_l385_385950


namespace average_pencils_per_box_l385_385488

theorem average_pencils_per_box :
  let pencil_counts := [12, 14, 14, 15, 15, 15, 16, 16, 17, 18] in
  (List.sum pencil_counts : ℚ) / (List.length pencil_counts : ℚ) = 15.2 :=
by {
  let pencil_counts := [12, 14, 14, 15, 15, 15, 16, 16, 17, 18],
  sorry
}

end average_pencils_per_box_l385_385488


namespace value_of_k_odd_function_range_of_m_value_of_k_even_function_value_of_n_l385_385133

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := exp x + (k - 2) * exp (-x)

-- Part (1) Condition (1.1)
theorem value_of_k_odd_function :
  (∀ x : ℝ, f (-x) 1 = -f x 1) → (k = 1)


-- Part (1) Condition (1.2)
theorem range_of_m (m : ℝ) :
  ( ∀ x > 1, m * f x 1 - f (2 * x) 1 - 2 * exp (-2 * x) - 10 < 0) → m < 4 * real.sqrt 3


-- Part (2) Condition (2.1)
theorem value_of_k_even_function :
  (∀ x : ℝ, f (-x) 3 = f x 3) → (k = 3)


-- Part (2) Condition (2.2)
noncomputable def g (x : ℝ) : ℝ := real.log (f x 3) / real.log 2 

noncomputable def h (x n : ℝ) : ℝ := (g x - 1 + n) * (2 * n + 1 - g x) + n ^ 2 - n

theorem value_of_n (n : ℝ) :
  (∀ x : ℝ, h x n = 0 → (n ≤ 0 ∨ n ≥ 4 / 13)) 

end value_of_k_odd_function_range_of_m_value_of_k_even_function_value_of_n_l385_385133


namespace car_rental_cost_l385_385388

def day1_cost (base_rate : ℝ) (miles_driven : ℝ) (cost_per_mile : ℝ) : ℝ :=
  base_rate + miles_driven * cost_per_mile

def day2_cost (base_rate : ℝ) (miles_driven : ℝ) (cost_per_mile : ℝ) : ℝ :=
  base_rate + miles_driven * cost_per_mile

def day3_cost (base_rate : ℝ) (miles_driven : ℝ) (cost_per_mile : ℝ) : ℝ :=
  base_rate + miles_driven * cost_per_mile

def total_cost (day1 : ℝ) (day2 : ℝ) (day3 : ℝ) : ℝ :=
  day1 + day2 + day3

theorem car_rental_cost :
  let day1_base_rate := 150
  let day2_base_rate := 100
  let day3_base_rate := 75
  let day1_miles_driven := 620
  let day2_miles_driven := 744
  let day3_miles_driven := 510
  let day1_cost_per_mile := 0.50
  let day2_cost_per_mile := 0.40
  let day3_cost_per_mile := 0.30
  day1_cost day1_base_rate day1_miles_driven day1_cost_per_mile +
  day2_cost day2_base_rate day2_miles_driven day2_cost_per_mile +
  day3_cost day3_base_rate day3_miles_driven day3_cost_per_mile = 1085.60 :=
by
  let day1 := day1_cost 150 620 0.50
  let day2 := day2_cost 100 744 0.40
  let day3 := day3_cost 75 510 0.30
  let total := total_cost day1 day2 day3
  show total = 1085.60
  sorry

end car_rental_cost_l385_385388


namespace intersection_points_l385_385068

theorem intersection_points (A : ℝ) (hA : 0 < A) :
  let y := λ x : ℝ, A * x^2 in
  let curve1 := λ (x y : ℝ), y = A * x^2 in
  let curve2 := λ (x y : ℝ), y^2 + 2 = x^2 + 6 * y in
  (∃ (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ),
    (curve1 x1 y1 ∧ curve2 x1 y1 ∧
     curve1 x2 y2 ∧ curve2 x2 y2 ∧
     curve1 x3 y3 ∧ curve2 x3 y3 ∧
     curve1 x4 y4 ∧ curve2 x4 y4)) ∧
  ¬(∃ (x5 x6 y5 y6 : ℝ),
    (curve1 x5 y5 ∧ curve2 x5 y5 ∧
     curve1 x6 y6 ∧ curve2 x6 y6 ∧ (x5, x6) ≠ (x1, x2) ∧ (x5, x6) ≠ (x3, x4)))
:=
by {
  sorry
}

end intersection_points_l385_385068


namespace common_number_is_eight_l385_385710

theorem common_number_is_eight (a b c d e f g : ℝ) 
  (h1 : (a + b + c + d) / 4 = 7)
  (h2 : (d + e + f + g) / 4 = 9)
  (h3 : (a + b + c + d + e + f + g) / 7 = 8) :
  d = 8 :=
by
sorry

end common_number_is_eight_l385_385710


namespace nearest_integer_to_U_is_375_l385_385476

noncomputable def sum_term (n : ℕ) : ℝ :=
  have h : n ≥ 2 := by linarith
  1 / (n^2 - 1 : ℝ)

noncomputable def U := 500 * (∑ n in finset.range(4999).filter (λ n, n ≥ 2), sum_term (n + 2))

theorem nearest_integer_to_U_is_375 : Int.nearest U = 375 := 
  by 
  sorry

end nearest_integer_to_U_is_375_l385_385476


namespace perimeter_square_C_l385_385490

theorem perimeter_square_C (pA pB pC : ℕ) (hA : pA = 16) (hB : pB = 32) (hC : pC = (pA + pB) / 2) : pC = 24 := by
  sorry

end perimeter_square_C_l385_385490


namespace evaluate_g_at_neg3_l385_385340

def g (x : ℤ) : ℤ := x^2 - x + 2 * x^3

theorem evaluate_g_at_neg3 : g (-3) = -42 := by
  sorry

end evaluate_g_at_neg3_l385_385340


namespace circle_equation1_circle_equation2_l385_385886

-- Definitions for the first question
def center1 : (ℝ × ℝ) := (2, -2)
def pointP : (ℝ × ℝ) := (6, 3)

-- Definitions for the second question
def pointA : (ℝ × ℝ) := (-4, -5)
def pointB : (ℝ × ℝ) := (6, -1)

-- Theorems we need to prove
theorem circle_equation1 : (x - 2)^2 + (y + 2)^2 = 41 :=
sorry

theorem circle_equation2 : (x - 1)^2 + (y + 3)^2 = 29 :=
sorry

end circle_equation1_circle_equation2_l385_385886


namespace diagonal_length_of_courtyard_l385_385423

noncomputable theory

-- Definitions from conditions
def ratio := 4 / 3
def paving_cost := 600 -- Rs.
def paving_rate := 0.50 -- Rs. per m^2

-- The main statement to be proven
theorem diagonal_length_of_courtyard (x : ℝ) : 
  (12 * x^2 * paving_rate = paving_cost) → 
  (4 * x)^2 + (3 * x)^2 = 2500 := 
by
  sorry

end diagonal_length_of_courtyard_l385_385423


namespace monthly_growth_rate_l385_385020

theorem monthly_growth_rate (x : ℝ)
  (turnover_may : ℝ := 1)
  (turnover_july : ℝ := 1.21)
  (growth_rate_condition : (1 + x) ^ 2 = 1.21) :
  x = 0.1 :=
sorry

end monthly_growth_rate_l385_385020


namespace trigonometric_conditions_nec_not_sufficient_l385_385397

variable (α β : Real)

theorem trigonometric_conditions_nec_not_sufficient :
  (sin α + cos β = 0) → (sin α ≠ cos β) → (sin α ≠ -cos β) → (sin^2 α + sin^2 β = 1) ∧ ((sin α ≠ cos β) ∨ (sin α ≠ -cos β)) :=
by
  sorry

end trigonometric_conditions_nec_not_sufficient_l385_385397


namespace determine_a_b_l385_385570

theorem determine_a_b (a b : ℤ) :
  (∀ x : ℤ, x^2 + a * x + b = (x - 1) * (x + 4)) → (a = 3 ∧ b = -4) :=
by
  intro h
  sorry

end determine_a_b_l385_385570


namespace average_speed_correct_l385_385733

noncomputable def initial_odometer := 12321
noncomputable def final_odometer := 12421
noncomputable def time_hours := 4
noncomputable def distance := final_odometer - initial_odometer
noncomputable def avg_speed := distance / time_hours

theorem average_speed_correct : avg_speed = 25 := by
  sorry

end average_speed_correct_l385_385733


namespace find_total_roses_l385_385770

open Nat

theorem find_total_roses 
  (a : ℕ)
  (h1 : 300 ≤ a)
  (h2 : a ≤ 400)
  (h3 : a % 21 = 13)
  (h4 : a % 15 = 7) : 
  a = 307 := 
sorry

end find_total_roses_l385_385770


namespace sum_c_d_eq_24_l385_385819

theorem sum_c_d_eq_24 (c d : ℕ) (h_pos_c : c > 0) (h_pos_d : d > 1) (h_max_power : c^d < 500 ∧ ∀ ⦃x y : ℕ⦄, x^y < 500 → x^y ≤ c^d) : c + d = 24 :=
sorry

end sum_c_d_eq_24_l385_385819


namespace cos_210_eq_neg_sqrt3_div_2_l385_385848

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385848


namespace perimeter_of_excircle_opposite_leg_l385_385963

noncomputable def perimeter_of_right_triangle (a varrho_a : ℝ) : ℝ :=
  2 * varrho_a * a / (2 * varrho_a - a)

theorem perimeter_of_excircle_opposite_leg
  (a varrho_a : ℝ) (h_a_pos : 0 < a) (h_varrho_a_pos : 0 < varrho_a) :
  (perimeter_of_right_triangle a varrho_a = 2 * varrho_a * a / (2 * varrho_a - a)) :=
by
  sorry

end perimeter_of_excircle_opposite_leg_l385_385963


namespace ginger_total_water_l385_385915

def hours_worked : Nat := 8
def cups_per_bottle : Nat := 2
def bottles_drank_per_hour : Nat := 1
def bottles_for_plants : Nat := 5

theorem ginger_total_water : 
  (hours_worked * cups_per_bottle * bottles_drank_per_hour) + (bottles_for_plants * cups_per_bottle) = 26 :=
by
  sorry

end ginger_total_water_l385_385915


namespace converse_of_P_inverse_of_P_contrapositive_of_P_negation_of_P_l385_385658

theorem converse_of_P (a b : ℤ) : (a - 2 > b - 2) → (a > b) :=
by
  intro h
  exact sorry

theorem inverse_of_P (a b : ℤ) : (a ≤ b) → (a - 2 ≤ b - 2) :=
by
  intro h
  exact sorry

theorem contrapositive_of_P (a b : ℤ) : (a - 2 ≤ b - 2) → (a ≤ b) :=
by
  intro h
  exact sorry

theorem negation_of_P (a b : ℤ) : (a > b) → ¬ (a - 2 ≤ b - 2) :=
by
  intro h
  exact sorry

end converse_of_P_inverse_of_P_contrapositive_of_P_negation_of_P_l385_385658


namespace range_of_a_l385_385955

theorem range_of_a (a : ℝ) :
  (∃ x ∈ set.Ioo 4 6, deriv (λ x, (1 / 2) * x^2 - (a + 2) * x + 2 * a * real.log x + 1) x = 0) →
  4 < a ∧ a < 6 :=
begin
  -- Proof will be filled in here
  sorry
end

end range_of_a_l385_385955


namespace inverse1_inverse2_inverse3_inverse4_l385_385494

theorem inverse1 (x : ℝ) (y : ℝ) : y = 3 * x - 5 → x = (y + 5) / 3 := 
sorry

theorem inverse2 (x : ℝ) (y : ℝ) : y = sqrt(1 - x^3) → x = (1 - y^2)^(1/3) :=
sorry

theorem inverse3 (x : ℝ) (y : ℝ) : y = Real.arcsin(3 * x) → x = sin(y) / 3 := 
sorry

theorem inverse4 (x : ℝ) (y : ℝ) : y = x^2 + 2 → (x = sqrt(y - 2) ∨ x = -sqrt(y - 2)) := 
sorry

end inverse1_inverse2_inverse3_inverse4_l385_385494


namespace vector_calc_l385_385977

-- Definitions of the vectors a and b
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

-- Statement to prove that 2a - b = (5, 7)
theorem vector_calc : 2 • a - b = (5, 7) :=
by {
  -- Proof will be filled here
  sorry
}

end vector_calc_l385_385977


namespace trigonometric_identity_l385_385153

theorem trigonometric_identity (α : ℝ)
  (h1 : Real.sin (π + α) = 3 / 5)
  (h2 : π < α ∧ α < 3 * π / 2) :
  (Real.sin ((π + α) / 2) - Real.cos ((π + α) / 2)) / 
  (Real.sin ((π - α) / 2) - Real.cos ((π - α) / 2)) = -1 / 2 :=
by
  sorry

end trigonometric_identity_l385_385153


namespace problem_proof_l385_385126

def f (x : ℝ) : ℝ := x - Real.log x

def f_k (k x : ℝ) : ℝ := 
if f x ≥ k then f x else k

theorem problem_proof : 
  f_k 3 (f_k 2 Real.exp 1) = 3 :=
by
  sorry

end problem_proof_l385_385126


namespace count_irrationals_in_list_l385_385800

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

theorem count_irrationals_in_list :
  let lst := [22 / 7, 3.14159, real.sqrt 7, -8, real.cbrt 2, 0.6, 0, real.sqrt 36, real.pi / 3] in
  list.countp is_irrational lst = 3 :=
by sorry

end count_irrationals_in_list_l385_385800


namespace geometric_sequence_increasing_condition_l385_385933

-- Define the geometric sequence and sum notation
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a₁ q : ℝ, (∀ n : ℕ, a (n + 1) = a₁ * q ^ n) ∧ (∀ n : ℕ, a n > 0)

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum a

-- The main statement
theorem geometric_sequence_increasing_condition
  (a : ℕ → ℝ)
  (h : is_geometric_sequence a)
  (h_sum_pos : ∀ n : ℕ, sum_of_first_n_terms a n > 0) :
  (sum_of_first_n_terms a 19 + sum_of_first_n_terms a 21 > 2 * sum_of_first_n_terms a 20)
  ↔ (∀ n : ℕ, a (n + 1) > a n) :=
sorry

end geometric_sequence_increasing_condition_l385_385933


namespace factorize_difference_of_squares_l385_385106

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_difference_of_squares_l385_385106


namespace minimum_value_of_f_l385_385121

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |3 - x|

theorem minimum_value_of_f : ∀ x : ℝ, f x ≥ 1 ∧ (∃ x₀ : ℝ, f x₀ = 1) := by
  sorry

end minimum_value_of_f_l385_385121


namespace distance_between_cyclists_l385_385716

def cyclist_distance (v1 : ℝ) (v2 : ℝ) (t : ℝ) : ℝ :=
  (v1 + v2) * t

theorem distance_between_cyclists :
  cyclist_distance 10 25 1.4285714285714286 = 50 := by
  sorry

end distance_between_cyclists_l385_385716


namespace alice_has_ball_after_two_turns_l385_385443

noncomputable def prob_alice_has_ball_after_two_turns : ℚ :=
  let p_A_B := (3 : ℚ) / 5 -- Probability Alice tosses to Bob
  let p_B_A := (1 : ℚ) / 3 -- Probability Bob tosses to Alice
  let p_A_A := (2 : ℚ) / 5 -- Probability Alice keeps the ball
  (p_A_B * p_B_A) + (p_A_A * p_A_A)

theorem alice_has_ball_after_two_turns :
  prob_alice_has_ball_after_two_turns = 9 / 25 :=
by
  -- skipping the proof
  sorry

end alice_has_ball_after_two_turns_l385_385443


namespace orthocenter_on_line_l385_385331

variable {A B C H D E : Point}
variable {circABC : Circle}

-- Let H be the orthocenter of acute-angled triangle ABC.
axiom orthocenter (acute_triangle : Triangle) : Point

-- AH^2 = BH^2 + CH^2
axiom altitude_square_relation (A B C H : Point) [acute_angled_triangle A B C]
  (h_orthocenter : H = orthocenter (Triangle.mk A B C)) :
  dist A H ^ 2 = dist B H ^ 2 + dist C H ^ 2

-- Points D and E are on the circumcircle of triangle ABC with specific parallel conditions
axiom points_on_circumcircle (A B C D E : Point) (circ : Circle) :
  on_circle A circ ∧ on_circle B circ ∧ on_circle C circ ∧ on_circle D circ ∧ on_circle E circ

axiom parallel_conditions (A B C E D : Point) :
  parallel (Line.mk C E) (Line.mk A B) ∧ parallel (Line.mk B D) (Line.mk A C)

-- Given the conditions, prove H lies on the line DE
theorem orthocenter_on_line (A B C H D E : Point) 
  [acute_angled_triangle A B C]
  (h_orthocenter : H = orthocenter (Triangle.mk A B C))
  (h_square_rel : dist A H ^ 2 = dist B H ^ 2 + dist C H ^ 2)
  (h_points_circ : points_on_circumcircle A B C D E circABC)
  (h_parallel : parallel_conditions A B C E D) :
  on_line H (Line.mk D E) :=
sorry

end orthocenter_on_line_l385_385331


namespace inclination_angle_l385_385903

def f (x : ℝ) : ℝ := 1 / x + 2 * x

def f' (x : ℝ) : ℝ := -1 / x^2 + 2

theorem inclination_angle :
  (∃ (θ : ℝ), θ = Real.atan (f' 1) ∧ θ = Real.pi / 4) :=
by
  sorry

end inclination_angle_l385_385903


namespace calculate_paintable_area_l385_385598

noncomputable def bedroom_length : ℝ := 15
noncomputable def bedroom_width : ℝ := 11
noncomputable def bedroom_height : ℝ := 9
noncomputable def door_window_area : ℝ := 70
noncomputable def num_bedrooms : ℝ := 3

theorem calculate_paintable_area :
  (num_bedrooms * ((2 * bedroom_length * bedroom_height) + (2 * bedroom_width * bedroom_height) - door_window_area)) = 1194 := 
by
  -- conditions as definitions
  let total_wall_area := (2 * bedroom_length * bedroom_height) + (2 * bedroom_width * bedroom_height)
  let paintable_wall_in_bedroom := total_wall_area - door_window_area
  let total_paintable_area := num_bedrooms * paintable_wall_in_bedroom
  show total_paintable_area = 1194
  sorry

end calculate_paintable_area_l385_385598


namespace ratio_of_cards_bought_l385_385305

variable (Buddy_cards_on_Monday : ℕ)
variable (Buddy_loss_fraction : ℚ)
variable (Buddy_cards_bought_on_Wednesday : ℕ)
variable (Buddy_total_cards_on_Thursday : ℕ)

def Buddy_cards_on_Tuesday := Buddy_cards_on_Monday * (1 - Buddy_loss_fraction)
def Buddy_cards_after_Wednesday := Buddy_cards_on_Tuesday + Buddy_cards_bought_on_Wednesday
def Buddy_cards_bought_on_Thursday := Buddy_total_cards_on_Thursday - Buddy_cards_after_Wednesday

theorem ratio_of_cards_bought (h1 : Buddy_cards_on_Monday = 30) 
                              (h2 : Buddy_loss_fraction = 1/2)
                              (h3 : Buddy_cards_bought_on_Wednesday = 12)
                              (h4 : Buddy_total_cards_on_Thursday = 32) : 
                              Buddy_cards_bought_on_Thursday = (1/3) * Buddy_cards_on_Tuesday :=
by
  -- Proof steps (to be completed)
  sorry

end ratio_of_cards_bought_l385_385305


namespace bob_depth_l385_385053

noncomputable def bob_depth_lim
  (u : ℝ) (σ : ℝ) (ρ : ℝ) (g : ℝ) : ℝ :=
  let v := (1 : ℝ) / 2 * u ^ 2 / ((ρ / σ - 1) * g) in
  v

theorem bob_depth
  (u : ℝ := 23) (σ : ℝ := 100) (ρ : ℝ := 1000) (g : ℝ := 9.81) :
  filter.tendsto (bob_depth_lim u σ ρ g) (filter.cocompact ℝ) (𝓝 3) :=
by
  sorry

end bob_depth_l385_385053


namespace game_show_probability_l385_385023

theorem game_show_probability :
  let n := 4 in
  let p_correct := (1:ℚ) / 4 in
  let p_incorrect := (3:ℚ) / 4 in
  let win_probability := (Nat.choose n 3) * (p_correct ^ 3) * (p_incorrect ^ 1) + (Nat.choose n 4) * (p_correct ^ 4) in
  win_probability = 13 / 256 :=
by
  sorry

end game_show_probability_l385_385023


namespace find_smallest_subtract_l385_385659

-- Definitions for multiples
def is_mul_2 (n : ℕ) : Prop := 2 ∣ n
def is_mul_3 (n : ℕ) : Prop := 3 ∣ n
def is_mul_5 (n : ℕ) : Prop := 5 ∣ n

-- Statement of the problem
theorem find_smallest_subtract (x : ℕ) :
  (is_mul_2 (134 - x)) ∧ (is_mul_3 (134 - x)) ∧ (is_mul_5 (134 - x)) → x = 14 :=
by
  sorry

end find_smallest_subtract_l385_385659


namespace total_number_of_boys_l385_385736

-- Define the circular arrangement and the opposite positions
variable (n : ℕ)

theorem total_number_of_boys (h : (40 ≠ 10 ∧ (40 - 10) * 2 = n - 2)) : n = 62 := 
sorry

end total_number_of_boys_l385_385736


namespace range_of_φ_l385_385951

-- Define the function f
def f (x : ℝ) (φ : ℝ) : ℝ := 2 * Real.cos (3 * x + φ) + 1

-- Define the conditions
def conditions (φ : ℝ) : Prop := 
  ∀ x : ℝ, -π / 12 < x ∧ x < π / 6 → f x φ > 1 ∧ |φ| < π / 2

/-- Prove that the range of φ must be [-π / 4, 0] --/
theorem range_of_φ (φ : ℝ) :
  (conditions φ) → φ ≥ -π / 4 ∧ φ ≤ 0 :=
sorry

end range_of_φ_l385_385951


namespace cos_210_eq_neg_sqrt_3_div_2_l385_385829

theorem cos_210_eq_neg_sqrt_3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by
  -- leave proof as sorry
  sorry

end cos_210_eq_neg_sqrt_3_div_2_l385_385829


namespace largest_value_l385_385994

theorem largest_value (x : ℝ) (h : x = 1/4) :
  (∀ y ∈ {(x, x^2, 1/2 * x, 1/x, real.sqrt x)}, y ≤ 1/x) :=
by {
  intro y,
  intro h_y,
  cases h_y; simp [h],
  sorry
}

end largest_value_l385_385994


namespace blue_ball_higher_probability_l385_385017
open MeasureTheory ProbabilityTheory

-- Definitions of the probabilities
def prob_bin_0 := 1 / 4

def prob_bin_k (k : ℕ) (hk : k ≥ 1) : ℝ := (1 / 2)^k * (3 / 4)

-- Define the event that blue ball is in higher bin than yellow ball
def event_blue_higher (X Y : ℕ) : Prop := X > Y

-- Statement of the theorem
theorem blue_ball_higher_probability :
  let X := ProbabilityTheory.Probability.of_event prob_bin_0 prob_bin_k in
  let Y := ProbabilityTheory.Probability.of_event prob_bin_0 prob_bin_k in
  ProbabilityTheory.Probability (event_blue_higher X Y) = 3 / 8 :=
sorry

end blue_ball_higher_probability_l385_385017


namespace only_valid_pairs_l385_385499

-- Define conditions for a and n belonging to natural numbers
def satisfies_condition (a n : ℕ) : Prop :=
  (a+1)^n - a^n ∣ n

-- State the theorem to prove the only pairs (a, n) satisfying the condition are (a, 1)
theorem only_valid_pairs : ∀ (a n : ℕ), satisfies_condition a n → n = 1 ∧ a ∈ ℕ :=
by
  intro a n h
  sorry

end only_valid_pairs_l385_385499


namespace camilla_jellybeans_l385_385463

theorem camilla_jellybeans (b c : ℕ) (h1 : b = 3 * c) (h2 : b - 20 = 4 * (c - 20)) :
  b = 180 :=
by
  -- Proof steps would go here
  sorry

end camilla_jellybeans_l385_385463


namespace units_digit_of_consecutive_product_l385_385724

theorem units_digit_of_consecutive_product (n : ℕ) :
  let product := n * (n + 1) * (n + 2) * (n + 3) in product % 10 = 4 :=
sorry

end units_digit_of_consecutive_product_l385_385724


namespace units_digit_of_5_pow_12_plus_4_sq_l385_385375

theorem units_digit_of_5_pow_12_plus_4_sq : 
  let units_digit (n : ℕ) := n % 10,
      d5 := units_digit (5 ^ 12),
      d4 := units_digit (4 ^ 2)
  in units_digit (d5 + d4) = 1 :=
by
  sorry

end units_digit_of_5_pow_12_plus_4_sq_l385_385375


namespace number_of_terms_equal_one_l385_385329

noncomputable def first_non_zero_digit {n : ℕ} (h : n > 0) : ℕ :=
if d : (1 / Real.sqrt n).toString.dropWhile (λ c, c = '.') != [] then
  d.head.toNat
else
  0

theorem number_of_terms_equal_one :
  (∑ n in finset.range (10^6 + 1), if first_non_zero_digit (by linarith) n = 1 then 1 else 0) = 757576 :=
by sorry

end number_of_terms_equal_one_l385_385329


namespace cos_210_eq_neg_sqrt3_div_2_l385_385852

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385852


namespace cos_210_eq_neg_sqrt3_div_2_l385_385875

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385875


namespace percentage_increase_visitors_l385_385337

theorem percentage_increase_visitors
  (original_visitors : ℕ)
  (original_fee : ℝ := 1)
  (fee_reduction : ℝ := 0.25)
  (visitors_increase : ℝ := 0.20) :
  ((original_visitors + (visitors_increase * original_visitors)) / original_visitors - 1) * 100 = 20 := by
  sorry

end percentage_increase_visitors_l385_385337


namespace div_operation_example_l385_385366

theorem div_operation_example : ((180 / 6) / 3) = 10 := by
  sorry

end div_operation_example_l385_385366


namespace IQR_correct_l385_385116

def list_of_numbers : List ℝ := 
  [42, 24, 30, 28, 26, 19, 33, 35, 47, 55, 61, 27, 39, 46, 52, 20, 22, 37, 48, 60, 50, 44, 31, 54, 58]

noncomputable def Q1_Q3 (l : List ℝ) : ℝ × ℝ :=
  let sorted_list := l.qsort (≤)
  let median := sorted_list[sorted_list.length / 2]
  let first_half := sorted_list.take (sorted_list.length / 2)
  let second_half := sorted_list.drop (sorted_list.length / 2 + 1)
  let Q1 := (first_half[first_half.length / 2 - 1] + first_half[first_half.length / 2]) / 2
  let Q3 := (second_half[second_half.length / 2 - 1] + second_half[second_half.length / 2]) / 2
  (Q1, Q3)

noncomputable def IQR (l : List ℝ) : ℝ :=
  let (Q1, Q3) := Q1_Q3 l
  Q3 - Q1

theorem IQR_correct : IQR list_of_numbers = 23.5 := by
  sorry

end IQR_correct_l385_385116


namespace algebra_expression_solution_l385_385134

theorem algebra_expression_solution
  (m : ℝ)
  (h : m^2 + m - 1 = 0) :
  m^3 + 2 * m^2 - 2001 = -2000 := by
  sorry

end algebra_expression_solution_l385_385134


namespace intersection_A_B_l385_385559

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

def setA : Set ℝ := { x | Real.log x > 0 }
def setB : Set ℝ := { x | Real.exp x * Real.exp x < 3 }

theorem intersection_A_B : setA ∩ setB = { x | 1 < x ∧ x < log2 3 } :=
by
  sorry

end intersection_A_B_l385_385559


namespace alexei_loss_per_week_l385_385040

-- Definitions
def aleesia_loss_per_week : ℝ := 1.5
def aleesia_total_weeks : ℕ := 10
def total_loss : ℝ := 35
def alexei_total_weeks : ℕ := 8

-- The statement to prove
theorem alexei_loss_per_week :
  (total_loss - aleesia_loss_per_week * aleesia_total_weeks) / alexei_total_weeks = 2.5 := 
by sorry

end alexei_loss_per_week_l385_385040


namespace correct_statements_l385_385173

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x else log x / log 0.5

theorem correct_statements (a : ℝ) :
  (a ≤ 0 → f (f a) = -a) ∧
  (a ≥ 1 → f (f a) = 1 / a) :=
by
  sorry

end correct_statements_l385_385173


namespace savings_ratio_l385_385600

-- conditions translated into Lean 4 definitions
def josiah_daily_savings: ℝ := 0.25
def josiah_days: ℕ := 24
def leah_daily_savings: ℝ := 0.50
def leah_days: ℕ := 20
def megan_days: ℕ := 12
def total_savings: ℝ := 28

-- statement to prove the question with the given conditions
theorem savings_ratio:
  let josiah_total := josiah_daily_savings * josiah_days in
  let leah_total := leah_daily_savings * leah_days in
  let megan_total := total_savings - josiah_total - leah_total in
  let megan_daily_savings := megan_total / megan_days in
  (megan_daily_savings / leah_daily_savings) = 2 :=
by
  sorry

end savings_ratio_l385_385600


namespace sequence_general_term_l385_385242

theorem sequence_general_term (n : ℕ) (hn : 0 < n) : 
  ∃ (a_n : ℕ), a_n = 2 * Int.floor (Real.sqrt (n - 1)) + 1 :=
by
  sorry

end sequence_general_term_l385_385242


namespace value_of_S_l385_385075

def recursive_sequence : ℤ → ℚ
| 0       => 3
| (n + 1) => 1003 - n + (1 / 3) * (recursive_sequence n)

def S : ℚ :=
  let n := 1000
  recursive_sequence n

theorem value_of_S : S = 3006 := by
  sorry

end value_of_S_l385_385075


namespace triangle_angle_bac_eq_30_l385_385295

/-- Let \( ABC \) be a triangle in which \( \angle ABC = 60^\circ \).
Let \( I \) and \( O \) be the incentre and circumcentre of \( ABC \), respectively.
Let \( M \) be the midpoint of the arc \( BC \) of the circumcircle of \( ABC \), which does not contain the point \( A \).
Then \( \angle BAC = 30^\circ \) given that \( MB = OI \). -/
theorem triangle_angle_bac_eq_30 (ABC : Triangle)
  (∠ABC = 60°)
  (I : Incentre ABC) 
  (O : Circumcentre ABC) 
  (M : Midpoint_of_arc_BC_without_A) 
  (h : MB = OI) :
  ∠BAC = 30° :=
sorry

end triangle_angle_bac_eq_30_l385_385295


namespace projection_matrix_correct_l385_385119

open Matrix

noncomputable def matrixQ : Matrix (Fin 3) (Fin 3) ℝ :=
  λ i j, (1 : ℝ) / 3

theorem projection_matrix_correct (v : Fin 3 → ℝ) :
    (matrixQ.mulVec v) = ((v 0 + v 1 + v 2) / 3) • (λ i, 1 : Fin 3 → ℝ) := by
  sorry

end projection_matrix_correct_l385_385119


namespace cos_210_eq_neg_sqrt3_div_2_l385_385850

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385850


namespace find_number_of_roses_l385_385771

theorem find_number_of_roses : ∃ a : ℕ, 300 ≤ a ∧ a ≤ 400 ∧ a % 21 = 13 ∧ a % 15 = 7 :=
by
  -- Existential quantifier for the number 'a'
  use 307
  
  -- Proof of the conditions for 'a'
  split
  -- Proof that 300 ≤ 307 ∧ 307 ≤ 400
  exact ⟨by linarith, by linarith⟩
  split
  -- Proof that 307 % 21 = 13
  exact by norm_num
  -- Proof that 307 % 15 = 7 (because -8 % 15 = 7)
  exact by norm_num

end find_number_of_roses_l385_385771


namespace intersection_point_unique_m_l385_385574

theorem intersection_point_unique_m (m : ℕ) (h1 : m > 0)
  (x y : ℤ) (h2 : 13 * x + 11 * y = 700) (h3 : y = m * x - 1) : m = 6 :=
by
  sorry

end intersection_point_unique_m_l385_385574


namespace correct_statements_l385_385731

variable (model : Type) (r_A r_B R2 : ℝ) (totalProducts defectiveProducts : ℕ)
variable (selectionEvents : Set (Set ℕ))

-- Conditions
def narrow_band_better_fit : Prop :=
  ∀ (residuals : model → ℝ), (∀ d d' ∈ residuals, |d| < |d'|) → True

def stronger_correlation_abs : Prop :=
  r_A = 0.97 ∧ r_B = -0.99 ∧ |r_A| < |r_B|

def worse_fit_smaller_R2 : Prop :=
  R2 ≥ 0 ∧ R2 < 1 ∧ R2 ≤ 0.5

def probability_one_defective : Prop :=
  totalProducts = 10 ∧ defectiveProducts = 3 ∧
  (∀ (selection : Set ℕ), selection ∈ selectionEvents →
    selection.card = 2 → selection.inter {1, 2, 3}.card = 1 →
    (selectionEvents.prob selection = 7 / 15))

-- Theorem
theorem correct_statements : narrow_band_better_fit model ∧ 
                              worse_fit_smaller_R2 R2 ∧ 
                              probability_one_defective totalProducts defectiveProducts selectionEvents := by
  sorry

end correct_statements_l385_385731


namespace vector_addition_l385_385900

theorem vector_addition :
  (⟨5, -9⟩ : ℕ × ℤ) + (⟨-8, 14⟩ : ℕ × ℤ) = ⟨-3, 5⟩ :=
sorry

end vector_addition_l385_385900


namespace complement_M_union_N_in_I_l385_385187

open Set

def I := {x ∈ (range 11).toSet | true}
def M := {1, 2, 3}.toSet
def N := {2, 4, 6, 8, 10}.toSet

theorem complement_M_union_N_in_I : (I \ (M ∪ N)) = ({0, 5, 7, 9}.toSet) := sorry

end complement_M_union_N_in_I_l385_385187


namespace donovan_lap_time_is_45_l385_385079

-- Definitions based on the conditions
def circular_track_length : ℕ := 600
def michael_lap_time : ℕ := 40
def michael_laps_to_pass_donovan : ℕ := 9

-- The theorem to prove
theorem donovan_lap_time_is_45 : ∃ D : ℕ, 8 * D = michael_laps_to_pass_donovan * michael_lap_time ∧ D = 45 := by
  sorry

end donovan_lap_time_is_45_l385_385079


namespace david_in_ninth_place_l385_385473

theorem david_in_ninth_place (pos : ℕ → ℕ) (h1 : pos Rand = pos Hikmet + 3)
  (h2 : pos Jack = pos Marta - 2)
  (h3 : pos Marta = 5)
  (h4 : pos Todd = pos Rand + 2)
  (h5 : pos David = pos Marta + 3)
  (h6 : pos Hikmet = pos Todd - 1)
  (h7 : pos Vanessa = pos Jack + 1) :
  pos David = 9 := 
sorry

end david_in_ninth_place_l385_385473


namespace Kim_morning_routine_time_l385_385602

def total_employees : ℕ := 9
def senior_employees : ℕ := 3
def overtime_employees : ℕ := 4
def regular_employees : ℕ := total_employees - senior_employees
def non_overtime_employees : ℕ := total_employees - overtime_employees

def coffee_time : ℕ := 5
def status_update_time (regular senior : ℕ) : ℕ := (regular * 2) + (senior * 3)
def payroll_update_time (overtime non_overtime : ℕ) : ℕ := (overtime * 3) + (non_overtime * 1)
def email_time : ℕ := 10
def task_allocation_time : ℕ := 7

def total_morning_routine_time : ℕ :=
  coffee_time +
  status_update_time regular_employees senior_employees +
  payroll_update_time overtime_employees non_overtime_employees +
  email_time +
  task_allocation_time

theorem Kim_morning_routine_time : total_morning_routine_time = 60 := by
  sorry

end Kim_morning_routine_time_l385_385602


namespace AC_length_is_17_l385_385587

theorem AC_length_is_17
  (A B C M : Type)
  [IsoscelesTriangle A B C]
  (h_AB_BC : AB = BC)
  (h_AM : AM = 7)
  (h_MB : MB = 3)
  (h_angle_BMC : ∠BMC = 60) 
  : AC = 17 := by
  sorry

end AC_length_is_17_l385_385587


namespace seats_per_bus_l385_385352

theorem seats_per_bus (students buses : ℕ) (h1 : students = 14) (h2 : buses = 7) : students / buses = 2 := by
  sorry

end seats_per_bus_l385_385352


namespace min_value_of_sum_of_squares_l385_385289

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : x^2 + y^2 + z^2 ≥ 4.8 :=
sorry

end min_value_of_sum_of_squares_l385_385289


namespace at_least_two_pass_written_test_expectation_number_of_admission_advantage_l385_385711

noncomputable def probability_of_passing_written_test_A : ℝ := 0.4
noncomputable def probability_of_passing_written_test_B : ℝ := 0.8
noncomputable def probability_of_passing_written_test_C : ℝ := 0.5

noncomputable def probability_of_passing_interview_A : ℝ := 0.8
noncomputable def probability_of_passing_interview_B : ℝ := 0.4
noncomputable def probability_of_passing_interview_C : ℝ := 0.64

theorem at_least_two_pass_written_test :
  (probability_of_passing_written_test_A * probability_of_passing_written_test_B * (1 - probability_of_passing_written_test_C) +
  probability_of_passing_written_test_A * (1 - probability_of_passing_written_test_B) * probability_of_passing_written_test_C +
  (1 - probability_of_passing_written_test_A) * probability_of_passing_written_test_B * probability_of_passing_written_test_C +
  probability_of_passing_written_test_A * probability_of_passing_written_test_B * probability_of_passing_written_test_C = 0.6) :=
sorry

theorem expectation_number_of_admission_advantage :
  (3 * (probability_of_passing_written_test_A * probability_of_passing_interview_A) +
  3 * (probability_of_passing_written_test_B * probability_of_passing_interview_B) +
  3 * (probability_of_passing_written_test_C * probability_of_passing_interview_C) = 0.96) :=
sorry

end at_least_two_pass_written_test_expectation_number_of_admission_advantage_l385_385711


namespace feuerbach_theorem_l385_385000

theorem feuerbach_theorem (ABC : Triangle) : 
  touches (nine_point_circle ABC) (incircle ABC) ∧ 
  (∀ A_triangle, A_triangle ∈ excircles ABC → 
    touches (nine_point_circle ABC) A_triangle) := sorry

end feuerbach_theorem_l385_385000


namespace cosine_210_l385_385841

noncomputable def Q := rotate 210 (1, 0)
noncomputable def E := foot Q (1, 0) (-1, 0)

theorem cosine_210 : cos 210 = - (sqrt 3/2) :=
sorry

end cosine_210_l385_385841


namespace factorize_difference_of_squares_l385_385110

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_difference_of_squares_l385_385110


namespace count_irrationals_in_list_l385_385801

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

theorem count_irrationals_in_list :
  let lst := [22 / 7, 3.14159, real.sqrt 7, -8, real.cbrt 2, 0.6, 0, real.sqrt 36, real.pi / 3] in
  list.countp is_irrational lst = 3 :=
by sorry

end count_irrationals_in_list_l385_385801


namespace tangent_length_l385_385466

noncomputable def point := ℝ × ℝ

def distance (p1 p2 : point) :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def O := (0, 0) : point
def A := (4, 5) : point
def B := (8, 10) : point
def C := (7, 17) : point

def OA := distance O A
def OB := distance O B

theorem tangent_length :
  let Tangent_Length := real.sqrt (OA * OB)
  Tangent_Length = 2 * real.sqrt 41 :=
by
  sorry

end tangent_length_l385_385466


namespace x_100_equals_2_power_397_l385_385152

-- Define the sequences
noncomputable def a_n (n : ℕ) : ℕ := 2^(n-1)
noncomputable def b_n (n : ℕ) : ℕ := 5*n - 3

-- Define the merged sequence x_n
noncomputable def x_n (k : ℕ) : ℕ := 2^(4*k - 3)

-- Prove x_100 is 2^397
theorem x_100_equals_2_power_397 : x_n 100 = 2^397 := by
  unfold x_n
  show 2^(4*100 - 3) = 2^397
  rfl

end x_100_equals_2_power_397_l385_385152


namespace range_of_y_over_x_l385_385925

theorem range_of_y_over_x (x y : ℝ) (h : (x - 2) ^ 2 + y ^ 2 = 3) :
  ∃ k : ℝ, (y / x = k) ∧ k ∈ (- real.sqrt 3, real.sqrt 3) :=
sorry

end range_of_y_over_x_l385_385925


namespace value_of_y_l385_385217

theorem value_of_y (x y : ℝ) (h1 : x^(2 * y) = 9) (h2 : x = 3) : y = 1 := by
  sorry

end value_of_y_l385_385217


namespace vector_subtraction_l385_385975

variable (a b : ℝ × ℝ)

def vector_calc (a b : ℝ × ℝ) : ℝ × ℝ :=
  (2 * a.1 - b.1, 2 * a.2 - b.2)

theorem vector_subtraction :
  a = (2, 4) → b = (-1, 1) → vector_calc a b = (5, 7) := by
  intros ha hb
  simp [vector_calc]
  rw [ha, hb]
  simp
  sorry

end vector_subtraction_l385_385975


namespace S_range_l385_385135

theorem S_range (x : ℝ) (y : ℝ) (S : ℝ) 
  (h1 : y = 2 * x - 1) 
  (h2 : 0 ≤ x) 
  (h3 : x ≤ 1 / 2) 
  (h4 : S = x * y) : 
  -1 / 8 ≤ S ∧ S ≤ 0 := 
sorry

end S_range_l385_385135


namespace find_k_value_l385_385192

-- Definitions for the vectors a and b
def vector_a : ℝ × ℝ × ℝ := (0, 1, -1)
def vector_b : ℝ × ℝ × ℝ := (1, 0, 2)

-- Definition for the dot product of two 3D vectors
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Condition: k * a + b is perpendicular to a - b
def perp_condition (k : ℝ) : Prop :=
  let vector_ka_plus_b := (k * vector_a.1 + vector_b.1, k * vector_a.2 + vector_b.2, k * vector_a.3 + vector_b.3)
  let vector_a_minus_b := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2, vector_a.3 - vector_b.3)
  dot_product vector_ka_plus_b vector_a_minus_b = 0

-- The mathematic problem statement in Lean 4
theorem find_k_value : ∃ k : ℝ, perp_condition k ∧ k = 7 / 4 :=
by 
  use 7 / 4
  split
  · sorry -- Here would be the proof that perp_condition 7/4 is true
  · rfl

end find_k_value_l385_385192


namespace question_1_question_2_l385_385299

variable (x a : ℝ)

def p (a : ℝ) := ∀ x : ℝ, (x^2 - 4*a*x + 3*a^2 < 0)
def q := ∀ x : ℝ, (x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0)

theorem question_1 (h_a : a = 1) (hq : q) :
  p a ∧ q → (2 < x ∧ x < 3) := 
sorry

theorem question_2 (h_q : q) :
  (q is sufficient_but_not_necessary_condition p) → (1 < a ∧ a ≤ 2) :=
sorry

end question_1_question_2_l385_385299


namespace domain_shift_l385_385221

theorem domain_shift (f : ℝ → ℝ) (dom_f : ∀ x, 1 ≤ x ∧ x ≤ 4 → f x = f x) :
  ∀ x, -1 ≤ x ∧ x ≤ 2 ↔ (1 ≤ x + 2 ∧ x + 2 ≤ 4) :=
by
  sorry

end domain_shift_l385_385221


namespace mean_inequalities_l385_385279

noncomputable def arith_mean (a : List ℝ) : ℝ := 
  (a.foldr (· + ·) 0) / a.length

noncomputable def geom_mean (a : List ℝ) : ℝ := 
  Real.exp ((a.foldr (λ x y => Real.log x + y) 0) / a.length)

noncomputable def harm_mean (a : List ℝ) : ℝ := 
  a.length / (a.foldr (λ x y => 1 / x + y) 0)

def is_positive (a : List ℝ) : Prop := 
  ∀ x ∈ a, x > 0

def bounds (a : List ℝ) (m g h : ℝ) : Prop := 
  let α := List.minimum a
  let β := List.maximum a
  α ≤ h ∧ h ≤ g ∧ g ≤ m ∧ m ≤ β

theorem mean_inequalities (a : List ℝ) (h g m : ℝ) (h_assoc: h = harm_mean a) (g_assoc: g = geom_mean a) (m_assoc: m = arith_mean a) :
  is_positive a → bounds a m g h :=
  
sorry

end mean_inequalities_l385_385279


namespace dot_product_v_w_l385_385065

-- Define the vectors v and w
def v := (ℝ × ℝ × ℝ) 
def w := (ℝ × ℝ × ℝ) 

-- Populate the vectors with the given values
def v := (-5, 2, -3)
def w := (7, -4, 6)

-- Assert the dot product result
theorem dot_product_v_w : ((-5) * 7 + 2 * (-4) + (-3) * 6) = -61 :=
by
  sorry

end dot_product_v_w_l385_385065


namespace factorize_x_squared_minus_one_l385_385104

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_x_squared_minus_one_l385_385104


namespace distance_AB_polar_l385_385607

open Real

/-- The distance between points A and B in polar coordinates, given that θ₁ - θ₂ = π. -/
theorem distance_AB_polar (A B : ℝ × ℝ) (r1 r2 : ℝ) (θ1 θ2 : ℝ) (hA : A = (r1, θ1)) (hB : B = (r2, θ2)) (hθ : θ1 - θ2 = π) :
  dist (r1 * cos θ1, r1 * sin θ1) (r2 * cos θ2, r2 * sin θ2) = r1 + r2 :=
sorry

end distance_AB_polar_l385_385607


namespace ed_has_2_dogs_l385_385483

-- Ed's pets
variable (D : ℕ) -- number of dogs

-- Conditions
axiom cats_eq_3 : 3 = 3
axiom fish_eq_twice_other_pets : 2 * (D + 3)
axiom total_pets_eq_15 : D + 3 + 2 * (D + 3) = 15

-- Theorem to prove
theorem ed_has_2_dogs : D = 2 :=
sorry

end ed_has_2_dogs_l385_385483


namespace audi_cross_time_l385_385362

open_locale classical

variables (x v t : ℝ)
variable (t_audi_cross : ℝ)

def distance (speed time : ℝ) : ℝ := speed * time

theorem audi_cross_time (x v : ℝ)
  (h1 : ∀ t, distance v t = x + v * t)  -- distance = initial distance + speed * time
  (h2 : ∀ t, distance v t = 2 * (x + v * t)) -- BMW distance = twice the Audi distance
  (h3 : 17 ≤ t ∧ t ≤ 18) -- time is between 17:00 and 18:00
  : t_audi_cross = 17.25 ∨ t_audi_cross = 17.75 :=
sorry

end audi_cross_time_l385_385362


namespace problem_I_problem_II_problem_III_l385_385183

-- Condition definitions
def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 1)
def g (a : ℝ) (t : ℝ) (x : ℝ) : ℝ := 2 * log a (2 * x + t)
def F (a : ℝ) (t : ℝ) (x : ℝ) : ℝ := a^(log a (x + 1)) + t * x^2 - 2 * t + 1

-- Problem I
theorem problem_I (a : ℝ) (t : ℝ) (x : ℝ) (ha : a > 0) (ha1 : a ≠ 1) (hsol : f a 3 - g a t 3 = 0) : t = -4 := by 
  sorry

-- Problem II
theorem problem_II (a : ℝ) (x : ℝ) (ha : 0 < a) (ha1 : a < 1) (ht : t = 1) :
  f a x ≤ g a 1 x → (-1 / 2 < x ∧ x ≤ 0) := by
  sorry

-- Problem III
theorem problem_III (a : ℝ) (t : ℝ) (x : ℝ) (ha : a > 0) 
  (ha1 : a ≠ 1) (ht : t < 0 ∨ t > 0) (hroot : ∃ x, x ∈ Icc (-1 : ℝ) 3 ∧ F a t x = 0) :
  (t ≤ -5/7) ∨ (t ≥ (2 + real.sqrt 2) / 4) := by 
  sorry

end problem_I_problem_II_problem_III_l385_385183


namespace inequality_solution_minimum_value_l385_385746

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := abs (x + 3) - abs (x - 4)

-- Proof 1: Prove that the solution to f(x) > 3 is \{ x | x > 2 \}
theorem inequality_solution {x : ℝ} : f(x) > 3 ↔ x > 2 := sorry

-- Proof 2: Prove that the minimum value of f(x) is 0
theorem minimum_value : ∀ x, f(x) ≥ 0 := sorry

end inequality_solution_minimum_value_l385_385746


namespace roger_paid_fraction_23_percent_l385_385084

variable (A : ℝ)

-- Define the costs of the movie ticket and soda
def movie_cost : ℝ := 0.20 * (A - soda_cost A)
def soda_cost : ℝ := 0.05 * (A - movie_cost A)

-- Define the total cost Roger paid as a fraction of A
def total_cost_fraction : ℝ := (movie_cost A + soda_cost A) / A

-- The theorem we need to prove
theorem roger_paid_fraction_23_percent (hA : 0 < A) :
  abs (total_cost_fraction A - 0.23) < 0.01 :=
by
  -- Proof skipped
  sorry

end roger_paid_fraction_23_percent_l385_385084


namespace parabola_properties_l385_385633

theorem parabola_properties (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c < 0) :
  (∀ x, a * x^2 + b * x + c >= a * (x^2)) ∧
  (c < 0) ∧ 
  (-b / (2 * a) < 0) :=
by
  sorry

end parabola_properties_l385_385633


namespace area_of_right_triangle_l385_385200

theorem area_of_right_triangle (a b : ℝ) (h : a > 0 ∧ b > 0) : 
  (let γ := real.pi / 2 in
   let S := 1 / 2 * (a * b) / real.sin γ in
   S = 1 / 2 * a * b) :=
by
  sorry

end area_of_right_triangle_l385_385200


namespace train_length_proof_l385_385437

noncomputable def train_speed_kmh : ℝ := 50
noncomputable def crossing_time_s : ℝ := 9
noncomputable def length_of_train_m : ℝ := 125

theorem train_length_proof:
  ∀ (speed_kmh: ℝ) (time_s: ℝ), 
  speed_kmh = train_speed_kmh →
  time_s = crossing_time_s →
  (speed_kmh * (1000 / 3600) * time_s) = length_of_train_m :=
by intros speed_kmh time_s h_speed_kmh h_time_s
   -- Proof omitted
   sorry

end train_length_proof_l385_385437


namespace three_lines_concurrent_l385_385763

variables {P : Type} [MetricSpace P] [AffineSpace P]

-- Suppose we have points vertices forming the triangular prism
variables (A B C A1 B1 C1 : P)
variables (C0 A0 B0 : P) -- Intersection points of diagonals of quadrilaterals

-- Define the condition: convex polyhedron structure and intersection points
def convex_polyhedron_with_concurrence :=
  let quadrilateral_faces := [(A, A1, B1, B), (B, B1, C1, C), (C, C1, A1, A)] in
  let intersection_points := [C0, A0, B0] in
  ∀ (tri_face : P) (opp_quad_face_inter : P) (l : List P),
    tri_face = [A, B, C] ∧
    opp_quad_face_inter = [A1, B1, C1] ∧
    l = quadrilateral_faces → 
    (A1, A0), (B1, B0), (C1, C0) lines_concurrent.

-- Theorem statement: Lines A1A0, B1B0, and C1C0 are concurrent
theorem three_lines_concurrent (A B C A1 B1 C1 C0 A0 B0: P) 
  (h_convex : convex_polyhedron_with_concurrence A B C A1 B1 C1 C0 A0 B0) :
  lines_concurrent (A1, A0) (B1, B0) (C1, C0) :=
sorry

end three_lines_concurrent_l385_385763


namespace sum_first_ten_terms_arithmetic_l385_385267

def arithmetic_sequence_sum (a₁ : ℤ) (n : ℕ) (d : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem sum_first_ten_terms_arithmetic (a₁ a₂ a₆ d : ℤ) 
  (h1 : a₁ = -2) 
  (h2 : a₂ + a₆ = 2) 
  (common_diff : d = 1) :
  arithmetic_sequence_sum a₁ 10 d = 25 :=
by
  rw [h1, common_diff]
  sorry

end sum_first_ten_terms_arithmetic_l385_385267


namespace condition_a_condition_b_condition_c_l385_385562

-- Definitions for conditions
variable {ι : Type*} (f₁ f₂ f₃ f₄ : ι → ℝ) (x : ι)

-- First part: Condition to prove second equation is a consequence of first
theorem condition_a :
  (∀ x, f₁ x * f₄ x = f₂ x * f₃ x) →
  ((f₂ x ≠ 0) ∧ (f₂ x + f₄ x ≠ 0)) →
  (f₁ x * (f₂ x + f₄ x) = f₂ x * (f₁ x + f₃ x)) :=
sorry

-- Second part: Condition to prove first equation is a consequence of second
theorem condition_b :
  (∀ x, f₁ x * (f₂ x + f₄ x) = f₂ x * (f₁ x + f₃ x)) →
  ((f₄ x ≠ 0) ∧ (f₂ x ≠ 0)) →
  (f₁ x * f₄ x = f₂ x * f₃ x) :=
sorry

-- Third part: Condition for equivalence of the equations
theorem condition_c :
  (∀ x, (f₁ x * f₄ x = f₂ x * f₃ x) ∧ (x ∉ {x | f₂ x + f₄ x = 0})) ↔
  (∀ x, (f₁ x * (f₂ x + f₄ x) = f₂ x * (f₁ x + f₃ x)) ∧ (x ∉ {x | f₄ x = 0})) :=
sorry

end condition_a_condition_b_condition_c_l385_385562


namespace min_value_proof_l385_385543

-- Define the necessary conditions
variables {x y : ℝ}

-- The main definition of the problem
def minValueOfExpr (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 2) : ℝ :=
  1 / x + 3 / y

-- The statement that needs to be proved in Lean 4
theorem min_value_proof (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 2) :
  minValueOfExpr x y hx hy hxy = 2 + Real.sqrt 3 := 
sorry

end min_value_proof_l385_385543


namespace N_inside_or_on_triangle_ABC_l385_385799

variables {A B C D M P Q N : Type}
variables [inhabited A] [inhabited B] [inhabited C] [inhabited D]
variables [inhabited M] [inhabited P] [inhabited Q] [inhabited N]

def midpoint (X Y : Type) [inhabited X] [inhabited Y] : Type := sorry

axiom BC_parallel_AD : B ∥ D
axiom M_midpoint_CD : midpoint C D = M
axiom P_midpoint_MA : midpoint M A = P
axiom Q_midpoint_MB : midpoint M B = Q
axiom N_intersection_DP_CQ : ¬(N ∈ (DP ∩ CQ))

theorem N_inside_or_on_triangle_ABC 
  (BC_parallel_AD : B ∥ D) 
  (M_midpoint_CD : midpoint C D = M) 
  (P_midpoint_MA : midpoint M A = P) 
  (Q_midpoint_MB : midpoint M B = Q) 
  (N_intersection_DP_CQ : ¬(N ∈ (DP ∩ CQ))) : 
  N ∈ triangle ABC := 
sorry

end N_inside_or_on_triangle_ABC_l385_385799


namespace passengers_decreased_l385_385359

def initial_count : ℕ := 35
def got_off : ℕ := 18
def got_on : ℕ := 15

theorem passengers_decreased : initial_count - got_off + got_on = 32 → initial_count - 32 = 3 := by
  intro h
  rw [h]
  apply rfl

end passengers_decreased_l385_385359


namespace vector_calc_l385_385978

-- Definitions of the vectors a and b
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

-- Statement to prove that 2a - b = (5, 7)
theorem vector_calc : 2 • a - b = (5, 7) :=
by {
  -- Proof will be filled here
  sorry
}

end vector_calc_l385_385978


namespace max_shirt_price_l385_385442

theorem max_shirt_price (total_budget : ℝ) (entrance_fee : ℝ) (num_shirts : ℝ) 
  (discount_rate : ℝ) (tax_rate : ℝ) (max_price : ℝ) 
  (budget_after_fee : total_budget - entrance_fee = 195)
  (shirt_discount : num_shirts > 15 → discounted_price = num_shirts * max_price * (1 - discount_rate))
  (price_with_tax : discounted_price * (1 + tax_rate) ≤ 195) : 
  max_price ≤ 10 := 
sorry

end max_shirt_price_l385_385442


namespace hyperbola_point_x_coordinate_l385_385052

-- Given a and b are positive real numbers, and y1 is the given ordinate of the point
variables (a b y1 : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_y1 : ℝ)

-- Define the x-coordinate of the point on the hyperbola corresponding to the given y1
def hyperbola_x (a b y1 : ℝ) : ℝ := (a / b) * real.sqrt (b^2 + y1^2)

-- The hyperbola equation
def hyperbola (a b x y : ℝ) : Prop := (x / a)^2 - (y / b)^2 = 1

-- The point P on the hyperbola and its x-coordinate
variables (P : ℝ × ℝ) (hP : hyperbola a b P.1 P.2)

theorem hyperbola_point_x_coordinate : P.2 = y1 → P.1 = hyperbola_x a b y1 :=
sorry

end hyperbola_point_x_coordinate_l385_385052


namespace program_computation_l385_385639

def final_value (S : ℕ) (i : ℕ) : ℕ :=
  if i > 10 then S
  else final_value (3 * S) (i + 1)

theorem program_computation :
  final_value 1 1 = 3^10 :=
  sorry

end program_computation_l385_385639


namespace gcd_111_148_l385_385493

theorem gcd_111_148 : Nat.gcd 111 148 = 37 :=
by
  sorry

end gcd_111_148_l385_385493


namespace find_angle_between_vectors_l385_385492

noncomputable def vector_a : ℝ × ℝ × ℝ := (1, 1, -4)
noncomputable def vector_b : ℝ × ℝ × ℝ := (1, -2, 2)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def cos_angle (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

theorem find_angle_between_vectors :
  let φ := real.acos (cos_angle vector_a vector_b) * 180 / real.pi in
  φ = 135 := 
begin
  sorry
end

end find_angle_between_vectors_l385_385492


namespace cosine_210_l385_385844

noncomputable def Q := rotate 210 (1, 0)
noncomputable def E := foot Q (1, 0) (-1, 0)

theorem cosine_210 : cos 210 = - (sqrt 3/2) :=
sorry

end cosine_210_l385_385844


namespace find_cos_of_tan_add_pi_four_and_quadrant_l385_385535

open Real

noncomputable def tan_add_pi_four := λ x : ℝ, tan (x + π / 4) = 2
noncomputable def angle_in_third_quadrant := λ x : ℝ, π < x ∧ x < 3 * π / 2
noncomputable def cos_x (x : ℝ) : ℝ := - (3 * sqrt 10 / 10)

theorem find_cos_of_tan_add_pi_four_and_quadrant (x : ℝ) (h_tan : tan_add_pi_four x) (h_quad : angle_in_third_quadrant x) : 
  cos x = cos_x x :=
by
  sorry

end find_cos_of_tan_add_pi_four_and_quadrant_l385_385535


namespace largest_of_eight_consecutive_summing_to_5400_l385_385698

theorem largest_of_eight_consecutive_summing_to_5400 :
  ∃ (n : ℕ), (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) = 5400)
  → (n+7 = 678) :=
by 
  sorry

end largest_of_eight_consecutive_summing_to_5400_l385_385698


namespace count_permutations_divisible_by_2_l385_385205

-- Definition of the original set of digits
def original_digits : Multiset ℕ := {2, 3, 1, 1, 5, 7, 1, 5, 2}

-- Condition defining even criteria (necessary for divisibility by 2)
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Theorem statement
theorem count_permutations_divisible_by_2 :
  (original_digits.count 2 = 2) →
  (original_digits.count 1 = 3) →
  (original_digits.count 5 = 2) →
  (original_digits.count 3 = 1) →
  (original_digits.count 7 = 1) →
  let even_digits := {x // Multiset.mem x original_digits ∧ is_even x} in
  even_digits = {⟨2, _⟩, ⟨2, _⟩} →
  let remaining_digits := {2, 3, 1, 1, 5, 7, 1, 5} in
  Multiset.card remaining_digits = 8 →
  (Multiset.card (Multiset.of_list [⟨2, sorry⟩, ⟨3, sorry⟩, ⟨1, sorry⟩, ⟨1, sorry⟩, ⟨5, sorry⟩, ⟨7, sorry⟩, ⟨1, sorry⟩, ⟨5, sorry⟩])) = 8 →
  Multiset.count 1 remaining_digits = 3 →
  Multiset.count 2 remaining_digits = 1 →
  Multiset.count 5 remaining_digits = 2 →
  Multiset.count 3 remaining_digits = 1 →
  Multiset.count 7 remaining_digits = 1 →
  multichoose remaining_digits ∅ 8 3 2 1 1 1 = 3360 :=
by sorry

end count_permutations_divisible_by_2_l385_385205


namespace same_points_among_teams_l385_385585

theorem same_points_among_teams :
  ∀ (n : Nat), n = 28 → 
  ∀ (G D N : Nat), G = 378 → D >= 284 → N <= 94 →
  (∃ (team_scores : Fin n → Int), ∀ (i j : Fin n), i ≠ j → team_scores i = team_scores j) := by
sorry

end same_points_among_teams_l385_385585


namespace unique_point_on_line_l385_385643

theorem unique_point_on_line (z : ℂ) (h1 : z ≠ 0) (h2 : ¬ (∃ (θ : ℝ), z = θ)) :
  ∃! z : ℂ, (z ≠ 0) ∧ ¬ (∃ (θ : ℝ), z = θ) ∧ (1 + z^23) / z^64 ∈ ℝ :=
sorry

end unique_point_on_line_l385_385643


namespace age_product_difference_is_nine_l385_385451

namespace ArnoldDanny

def current_age := 4
def product_today (A : ℕ) := A * A
def product_next_year (A : ℕ) := (A + 1) * (A + 1)
def difference (A : ℕ) := product_next_year A - product_today A

theorem age_product_difference_is_nine :
  difference current_age = 9 :=
by
  sorry

end ArnoldDanny

end age_product_difference_is_nine_l385_385451


namespace mailman_junk_mail_l385_385776

variable (junk_mail_per_house : ℕ) (houses_per_block : ℕ)

theorem mailman_junk_mail (h1 : junk_mail_per_house = 2) (h2 : houses_per_block = 7) :
  junk_mail_per_house * houses_per_block = 14 :=
by
  sorry

end mailman_junk_mail_l385_385776


namespace shift_function_equiv_l385_385324

noncomputable def shifted_function : (ℝ → ℝ) :=
  λ x, 2 * sin (2 * x + π / 6)

noncomputable def expected_function : (ℝ → ℝ) :=
  λ x, 2 * sin (2 * x - π / 3)

theorem shift_function_equiv :
  ∀ x, shifted_function (x + π / 4) = expected_function x :=
by
  sorry

end shift_function_equiv_l385_385324


namespace product_evaluation_l385_385281

def f (x : ℕ) : ℕ := x^2 + 3 * x + 2

theorem product_evaluation :
  (∏ n in (Finset.range 2019).map (Finset.natEmbedding), (1 - | (2 : ℤ) / (f n) |)) = 337 / 1010 := sorry

end product_evaluation_l385_385281


namespace non_zero_digits_in_decimal_l385_385208

theorem non_zero_digits_in_decimal (n d : ℕ) (h : n = 180) (h_d : d = 2^4 * 5^6) : 
  let ratio := (n : ℚ) / d,
      decimal_digits := to_digits 10 ratio.to_decimal in
  count_nonzero_digits_right_of_point decimal_digits = 2 := by
  sorry

-- Auxiliary definitions
def to_digits (base : ℕ) (n : ℚ) : list ℕ := sorry
def count_nonzero_digits_right_of_point (ds : list ℕ) : ℕ := sorry

end non_zero_digits_in_decimal_l385_385208


namespace inequality_true_l385_385210

theorem inequality_true (a b : ℝ) (h : a^2 + b^2 > 1) : |a| + |b| > 1 :=
sorry

end inequality_true_l385_385210


namespace largest_no_substring_multiple_9_unique_l385_385921

def is_multiple_of_9 (n : ℕ) : Prop :=
  n % 9 = 0

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

def no_substring_multiple_of_9 (n : ℕ) : Prop :=
  ∀ m ∈ (n.to_digits 10).tails.filter_map list.head?',
  ¬ is_multiple_of_9 m ∘ digit_sum

def largest_N_no_substring_multiple_9 : ℕ :=
  88888888

theorem largest_no_substring_multiple_9_unique :
  ∀ N : ℕ, no_substring_multiple_of_9 N → N ≤ largest_N_no_substring_multiple_9 :=
sorry

end largest_no_substring_multiple_9_unique_l385_385921


namespace parabola_equation_line_AB_fixed_point_min_area_AMBN_l385_385165

-- Prove that the equation of the parabola is y^2 = 4x given the focus (1,0) for y^2 = 2px
theorem parabola_equation (p : ℝ) (h : p > 0) (foc : (1, 0) = (1, 2*p*1/4)):
  (∀ x y: ℝ, y^2 = 4*x ↔ y^2 = 2*p*x) := sorry

-- Prove that line AB passes through fixed point T(2,0) given conditions
theorem line_AB_fixed_point (A B : ℝ × ℝ) (hA : A.2^2 = 4*A.1) 
    (hB : B.2^2 = 4*B.1) (h : A.1*B.1 + A.2*B.2 = -4) :
  ∃ T : ℝ × ℝ, T = (2, 0) := sorry

-- Prove that minimum value of area Quadrilateral AMBN is 48
theorem min_area_AMBN (T : ℝ × ℝ) (A B M N : ℝ × ℝ)
    (hT : T = (2, 0)) (hA : A.2^2 = 4*A.1) (hB : B.2^2 = 4*B.1)
    (hM : M.2^2 = 4*M.1) (hN : N.2^2 = 4*N.1)
    (line_AB : A.1 * B.1 + A.2 * B.2 = -4) :
  ∀ (m : ℝ), T.2 = -(1/m)*T.1 + 2 → 
  ((1+m^2) * (1+1/m^2)) * ((m^2 + 2) * (1/m^2 + 2)) = 256 → 
  8 * 48 = 48 := sorry

end parabola_equation_line_AB_fixed_point_min_area_AMBN_l385_385165


namespace zeros_of_f_range_of_f_l385_385174

-- Define the function f
def f (x : ℝ) : ℝ := -9^x + 3^(x+1) - 2

-- Proof statement for the zeros of the function
theorem zeros_of_f :
  (f 0 = 0 ∧ f (Real.log 2 / Real.log 3) = 0) :=
by {
  -- The proof goes here
  sorry
}

-- Proof statement for the range of the function when x ∈ [0, 1]
theorem range_of_f :
  ∀ x, 0 ≤ x ∧ x ≤ 1 → -2 ≤ f x ∧ f x ≤ 25/4 :=
by {
  -- The proof goes here
  sorry
}

end zeros_of_f_range_of_f_l385_385174


namespace range_of_a_l385_385957

theorem range_of_a (a : ℝ) :
  (∃ x ∈ set.Ioo 4 6, deriv (λ x, (1 / 2) * x^2 - (a + 2) * x + 2 * a * real.log x + 1) x = 0) →
  4 < a ∧ a < 6 :=
begin
  -- Proof will be filled in here
  sorry
end

end range_of_a_l385_385957


namespace cos_210_eq_neg_sqrt3_div_2_l385_385870

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385870


namespace find_m_l385_385942

noncomputable def curve_C (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ
noncomputable def line_l (ρ θ m : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 6) = m
noncomputable def common_point_condition (m : ℝ) : Prop := (Real.abs (1 - 2 * m)) / 2 = 1

theorem find_m (m : ℝ) (h1 : ∃ ρ θ, curve_C ρ θ ∧ line_l ρ θ m) :
  m = -1/2 ∨ m = 3/2 :=
by
  sorry

end find_m_l385_385942


namespace magnitude_sum_of_unit_vectors_l385_385542

variables (a b : ℝ^3) (h_dot : a • b = 1/3)

noncomputable def is_unit_vector (v : ℝ^3) : Prop :=
  ∥v∥ = 1
  
theorem magnitude_sum_of_unit_vectors (ha : is_unit_vector a) (hb : is_unit_vector b) :
  ∥a + b∥ = (2 * real.sqrt 6) / 3 := 
sorry

end magnitude_sum_of_unit_vectors_l385_385542


namespace factorize_expression_l385_385894

-- Define the variables
variables (a b : ℝ)

-- State the theorem to prove the factorization
theorem factorize_expression : a^2 - 2 * a * b = a * (a - 2 * b) :=
by 
  -- Proof goes here
  sorry

end factorize_expression_l385_385894


namespace cos_210_eq_neg_sqrt3_div_2_l385_385853

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385853


namespace extra_fruits_l385_385351

theorem extra_fruits (r g s : Nat) (hr : r = 42) (hg : g = 7) (hs : s = 9) : r + g - s = 40 :=
by
  sorry

end extra_fruits_l385_385351


namespace original_sequence_polynomial_of_degree_3_l385_385139

def is_polynomial_of_degree (u : ℕ → ℤ) (n : ℕ) :=
  ∃ a b c d : ℤ, u n = a * n^3 + b * n^2 + c * n + d

def fourth_difference_is_zero (u : ℕ → ℤ) :=
  ∀ n : ℕ, (u (n + 4) - 4 * u (n + 3) + 6 * u (n + 2) - 4 * u (n + 1) + u n) = 0

theorem original_sequence_polynomial_of_degree_3 (u : ℕ → ℤ)
  (h : fourth_difference_is_zero u) : 
  ∃ (a b c d : ℤ), ∀ n : ℕ, u n = a * n^3 + b * n^2 + c * n + d := sorry

end original_sequence_polynomial_of_degree_3_l385_385139


namespace cos_210_eq_neg_sqrt3_div_2_l385_385866

theorem cos_210_eq_neg_sqrt3_div_2 : real.cos (210 * real.pi / 180) = - (real.sqrt 3 / 2) :=
by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385866


namespace mean_second_set_l385_385224

theorem mean_second_set (x : ℝ) (h : (28 + x + 42 + 78 + 104) / 5 = 62) :
  (48 + 62 + 98 + 124 + x) / 5 = 78 :=
sorry

end mean_second_set_l385_385224


namespace train_length_l385_385435

theorem train_length 
  (speed_kmh : ℝ) (time_s : ℝ)
  (h_speed : speed_kmh = 50)
  (h_time : time_s = 9) : 
  let speed_ms := (speed_kmh * 1000) / 3600 in
  let length_m := speed_ms * time_s in
  length_m = 125 :=
by
  -- Speed of the train in m/s
  have speed_ms_def : speed_ms = (speed_kmh * 1000) / 3600 := by rfl
  -- Calculation of the length of the train
  have length_m_def : length_m = speed_ms * time_s := by rfl
  -- Substituting values
  rw [h_speed, h_time, speed_ms_def, length_m_def]
  -- Converting speed to m/s
  have speed_calc : (50 * 1000) / 3600 = 125 / 9 := sorry
  -- Final calculation
  -- Simplifying the final expression
  rw [speed_calc]
  have final_calc : (125 / 9) * 9 = 125 := by sorry
  exact final_calc

end train_length_l385_385435


namespace value_of_a_plus_d_l385_385214

variable (a b c d : ℝ)

theorem value_of_a_plus_d (h1 : a + b = 4) (h2 : b + c = 7) (h3 : c + d = 5) : a + d = 4 :=
sorry

end value_of_a_plus_d_l385_385214


namespace train_length_proof_l385_385436

noncomputable def train_speed_kmh : ℝ := 50
noncomputable def crossing_time_s : ℝ := 9
noncomputable def length_of_train_m : ℝ := 125

theorem train_length_proof:
  ∀ (speed_kmh: ℝ) (time_s: ℝ), 
  speed_kmh = train_speed_kmh →
  time_s = crossing_time_s →
  (speed_kmh * (1000 / 3600) * time_s) = length_of_train_m :=
by intros speed_kmh time_s h_speed_kmh h_time_s
   -- Proof omitted
   sorry

end train_length_proof_l385_385436


namespace sum_a_b_max_power_l385_385458

theorem sum_a_b_max_power (a b : ℕ) (h_pos : 0 < a) (h_b_gt_1 : 1 < b) (h_lt_600 : a ^ b < 600) : a + b = 26 :=
sorry

end sum_a_b_max_power_l385_385458


namespace factorize_x_squared_minus_1_l385_385096

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by
  -- Proof goes here
  sorry

end factorize_x_squared_minus_1_l385_385096


namespace d_n_2_d_n_3_l385_385584

def d (n k : ℕ) : ℕ :=
  if k = 0 then 1
  else if n = 1 then 0
  else (0:ℕ) -- Placeholder to demonstrate that we need a recurrence relation, not strictly necessary here for the statement.

theorem d_n_2 (n : ℕ) (hn : n ≥ 2) : 
  d n 2 = (n^2 - 3*n + 2) / 2 := 
by 
  sorry

theorem d_n_3 (n : ℕ) (hn : n ≥ 3) : 
  d n 3 = (n^3 - 7*n + 6) / 6 := 
by 
  sorry

end d_n_2_d_n_3_l385_385584


namespace angle_between_vectors_l385_385145

variables {V : Type} [inner_product_space ℝ V]

def magnitude (v : V) : ℝ := real.sqrt (inner_product_space.norm_sq v)
def angle_between (u v : V) : ℝ := real.arccos ((inner_product_space.inner u v) / (magnitude u * magnitude v))

theorem angle_between_vectors (a b : V) 
  (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : magnitude b = 2 * magnitude a)
  (h2 : inner_product_space.inner a (a - b) = 0) 
  : angle_between a b = real.pi / 3 :=
by
  sorry

end angle_between_vectors_l385_385145


namespace area_of_triangle_AFB_l385_385244

/-!
In a triangle ABC, the medians AD and CE have lengths 18 and 27 respectively, and AB = 24.
Extend CE to intersect the circumcircle of ABC at F.
The area of triangle AFB is m * sqrt n, where m and n are positive integers and n is not divisible by the square of any prime.
Prove that m + n = 63.
-/

theorem area_of_triangle_AFB (A B C D E F : Point) (m n : ℕ) 
  (hAD : median_length A D = 18)
  (hCE : median_length C E = 27)
  (hAB : dist A B = 24)
  (hCircumcircle : is_circumcircle A B C) 
  (hIntersect : extends CE to F on circumcircle)
  (hArea : area A F B = m * sqrt n)
  (hPosInt : m > 0 ∧ n > 0)
  (hSquarefree : ¬ ∃ p : ℕ, prime p ∧ p^2 ∣ n) :
  m + n = 63 := 
sorry

end area_of_triangle_AFB_l385_385244


namespace min_positive_value_l385_385987

theorem min_positive_value (c d : ℤ) (h : c > d) : 
  ∃ x : ℝ, x = (c + 2 * d) / (c - d) + (c - d) / (c + 2 * d) ∧ x = 2 :=
by {
  sorry
}

end min_positive_value_l385_385987


namespace problem_1_problem_2_l385_385608

variables (A : Type) (n m : ℕ) 
  (A_i : Finset (Finset A))
  (h1 : A_i.card = m)
  (h2 : ∀ (i j : Fin m), i ≠ j → A_i i ∩ A_i j = ∅)
  (h3 : ∀ i : Fin m, (A_i i).card = A_i i.card)

noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k

theorem problem_1 :
  ∑ i in Finset.range m, 1 / (C n (A_i i).card) ≤ 1 :=
sorry

theorem problem_2 :
  ∑ i in Finset.range m, C n (A_i i).card ≥ m * m :=
sorry

end problem_1_problem_2_l385_385608


namespace trinomial_ne_binomial_l385_385825

theorem trinomial_ne_binomial (a b c A B : ℝ) (h : a ≠ 0) : 
  ¬ ∀ x : ℝ, ax^2 + bx + c = Ax + B :=
by
  sorry

end trinomial_ne_binomial_l385_385825


namespace second_high_school_kids_proof_l385_385898

noncomputable def number_of_kids_from_second_high_school
  (denied_first : ℕ)
  (riverside_total : ℕ)
  (denied_second : ℕ)
  (mountaintop_total : ℕ)
  (total_allowed : ℕ)
  : ℕ :=
if H : riverside_total >= 0 ∧ denied_first = (20 * riverside_total) / 100 ∧
        mountaintop_total >= 0 ∧ denied_second = mountaintop_total / 2 ∧
        total_allowed = (riverside_total - denied_first) + (mountaintop_total - denied_second) + ?(kids_second_allowed) then
      ?(kids_second_allowed / 0.30)
else
  0

theorem second_high_school_kids_proof
  (denied_first : ℕ)
  (riverside_total : ℕ)
  (denied_second : ℕ)
  (mountaintop_total : ℕ)
  (total_allowed : ℕ)
  (kids_second_allowed : ℕ)
  (Hdk : denied_first = (20 * riverside_total) / 100)
  (Hmk : denied_second = mountaintop_total / 2)
  (Htotal : total_allowed = (riverside_total - denied_first) + (mountaintop_total - denied_second) + kids_second_allowed)
  (Hallowed : kids_second_allowed = 27)
  : number_of_kids_from_second_high_school denied_first riverside_total denied_second mountaintop_total total_allowed = 90 :=
by
  rw [number_of_kids_from_second_high_school]
  simp [Hdk, Hmk, Htotal, Hallowed]
  sorry

end second_high_school_kids_proof_l385_385898


namespace square_free_even_integers_count_l385_385206

-- Define what it means to be square-free
def square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ∣ n → m = 1

-- Define the predicate for even integers greater than 2 and less than 200
def valid_integer (n : ℕ) : Prop :=
  n > 2 ∧ n < 200 ∧ n % 2 = 0

-- Define the count of such integers
noncomputable def count_square_free_even_integers : ℕ :=
  (Finset.range 200).filter (λ n => valid_integer n ∧ square_free n).card

-- The main statement to prove
theorem square_free_even_integers_count : count_square_free_even_integers = 28 :=
sorry

end square_free_even_integers_count_l385_385206


namespace factorize_expression_l385_385895

-- Define the variables
variables (a b : ℝ)

-- State the theorem to prove the factorization
theorem factorize_expression : a^2 - 2 * a * b = a * (a - 2 * b) :=
by 
  -- Proof goes here
  sorry

end factorize_expression_l385_385895


namespace Andy_final_position_l385_385450

theorem Andy_final_position : 
  let start_position : (ℤ × ℤ) := (10, -10)
  let initial_direction : ℕ := 0  -- 0: East, 1: North, 2: West, 3: South
  let move_distance (n : ℕ) : ℤ := 2 * n
  let move (pos : (ℤ × ℤ)) (dir : ℕ) (dist : ℤ) : (ℤ × ℤ) := 
    match dir % 4 with
    | 0 => (pos.1 + dist, pos.2)
    | 1 => (pos.1, pos.2 + dist)
    | 2 => (pos.1 - dist, pos.2)
    | 3 => (pos.1, pos.2 - dist)
    | _ => pos
  let rec final_position (pos : (ℤ × ℤ)) (turns : ℕ) : (ℤ × ℤ) :=
    if turns = 0 then pos 
    else 
      let new_dir := turns % 4
      let new_dist := move_distance turns
      let new_pos := move pos new_dir new_dist
      final_position new_pos (turns - 1)
  in
    final_position start_position 2022 = (12, 4038) :=
sorry

end Andy_final_position_l385_385450


namespace yearly_production_target_l385_385402

-- Definitions for the conditions
def p_current : ℕ := 100
def p_add : ℕ := 50

-- The theorem to be proven
theorem yearly_production_target : (p_current + p_add) * 12 = 1800 := by
  sorry  -- Proof is omitted

end yearly_production_target_l385_385402


namespace inequality_proof_l385_385138

theorem inequality_proof
  (n : ℕ) 
  (n_ge_2 : n ≥ 2)
  (a : ℕ → ℕ)
  (a_increasing : ∀ i j, i < j → a i < a j)
  (a_sum_le_one : (∑ i in Finset.range n, 1 / a i) ≤ 1)
  (x : ℝ) :
  ( (∑ i in Finset.range n, 1 / ((a i : ℝ) ^ 2 + x^2)) ^ 2) 
  ≤ 
  (1 / 2) * (1 / (a 0 * (a 0 - 1) + x^2)) := 
sorry

end inequality_proof_l385_385138


namespace proof_problem_l385_385961

theorem proof_problem
  (k : ℝ) (k_pos : k > 0)
  (x1 x2 x3 : ℝ)
  (hx : x1 < x2 ∧ x2 < x3)
  (H1 : x1 + x3 = 2 * x2)
  (H2 : x2 = π)
  (H3 : ∀ x, sin x = k * x - k * π) :
  tan (x2 - x3) / (x1 - x2) = 1 :=
sorry

end proof_problem_l385_385961


namespace probability_same_color_is_27_over_100_l385_385984

def num_sides_die1 := 20
def num_sides_die2 := 20

def maroon_die1 := 5
def teal_die1 := 6
def cyan_die1 := 7
def sparkly_die1 := 1
def silver_die1 := 1

def maroon_die2 := 4
def teal_die2 := 6
def cyan_die2 := 7
def sparkly_die2 := 1
def silver_die2 := 2

noncomputable def probability_same_color : ℚ :=
  (maroon_die1 * maroon_die2 + teal_die1 * teal_die2 + cyan_die1 * cyan_die2 + sparkly_die1 * sparkly_die2 + silver_die1 * silver_die2) /
  (num_sides_die1 * num_sides_die2)

theorem probability_same_color_is_27_over_100 :
  probability_same_color = 27 / 100 := 
sorry

end probability_same_color_is_27_over_100_l385_385984


namespace dave_pages_l385_385472

noncomputable def total_cards (new_cards old_cards : ℕ) : ℕ :=
  new_cards + old_cards

noncomputable def pages_to_use (total_cards cards_per_page : ℕ) : ℕ :=
  total_cards / cards_per_page

theorem dave_pages (new_cards old_cards cards_per_page : ℕ) :
  new_cards = 3 → old_cards = 13 → cards_per_page = 8 →
  pages_to_use (total_cards new_cards old_cards) cards_per_page = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  have h : total_cards 3 13 = 16 := rfl
  rw [h]
  have h4 : pages_to_use 16 8 = 2 := rfl
  rw [h4]
  exact h4
  sorry

end dave_pages_l385_385472


namespace polynomial_bound_integer_solutions_l385_385282

theorem polynomial_bound_integer_solutions (f : Polynomial ℝ) (hc : ∃ c > 0, ∀ x : ℝ, f.degree ≥ 1 → ∃ n₀ : ℤ, ∀ (p : Polynomial ℝ), p.degree ≥ n₀ ∧ p.leading_coeff = 1 → 
  Multiset.card (Multiset.filter (λ x, |eval x (eval p f)| ≤ c) (Multiset.range (p.degree.to_nat + 1))) ≤ p.degree) : Prop :=
sorry

end polynomial_bound_integer_solutions_l385_385282


namespace cost_per_mile_eq_l385_385195

theorem cost_per_mile_eq :
  ( ∀ x : ℝ, (65 + 0.40 * 325 = x * 325) → x = 0.60 ) :=
by
  intros x h
  have eq1 : 65 + 0.40 * 325 = 195 := by sorry
  rw [eq1] at h
  have eq2 : 195 = 325 * x := h
  field_simp at eq2
  exact eq2

end cost_per_mile_eq_l385_385195


namespace contrapositive_proof_l385_385336

theorem contrapositive_proof (a b : ℝ) : 
  (a > b → a - 1 > b - 1) ↔ (a - 1 ≤ b - 1 → a ≤ b) :=
sorry

end contrapositive_proof_l385_385336


namespace dot_product_is_ten_l385_385561

-- Define the vectors
def a : ℝ × ℝ := (1, -2)
def b (m : ℝ) : ℝ × ℝ := (2, m)

-- Define the condition that the vectors are parallel
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 / v2.1 = v1.2 / v2.2

-- The main theorem statement
theorem dot_product_is_ten (m : ℝ) (h : parallel a (b m)) : 
  a.1 * (b m).1 + a.2 * (b m).2 = 10 := by
  sorry

end dot_product_is_ten_l385_385561


namespace product_of_numbers_l385_385669

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 9) (h2 : x^2 + y^2 = 157) : x * y = 22 := 
by 
  sorry

end product_of_numbers_l385_385669


namespace value_of_N_l385_385439

theorem value_of_N (N : ℕ): 6 < (N : ℝ) / 4 ∧ (N : ℝ) / 4 < 7.5 ↔ N = 25 ∨ N = 26 ∨ N = 27 ∨ N = 28 ∨ N = 29 := 
by
  sorry

end value_of_N_l385_385439


namespace quadrilateral_parallelogram_if_bisecting_diagonals_l385_385042

theorem quadrilateral_parallelogram_if_bisecting_diagonals
  (Q : Type)
  [quadrilateral Q]
  (bisect_diagonals : ∀ d1 d2 : Q.1 → Q.1, (d1 ∈ Q) ∧ (d2 ∈ Q) ∧ (midpoint d1 = midpoint d2)) :
  parallelogram Q :=
sorry

end quadrilateral_parallelogram_if_bisecting_diagonals_l385_385042


namespace average_sitting_time_l385_385752

theorem average_sitting_time (number_of_students : ℕ) (number_of_seats : ℕ) (total_travel_time : ℕ) 
  (h1 : number_of_students = 6) 
  (h2 : number_of_seats = 4) 
  (h3 : total_travel_time = 192) :
  (number_of_seats * total_travel_time) / number_of_students = 128 :=
by
  sorry

end average_sitting_time_l385_385752


namespace factorial_fraction_l385_385061

theorem factorial_fraction :
  (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_fraction_l385_385061


namespace valid_three_digit_numbers_count_l385_385444

-- Define the set of digits
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}

-- Define the overall condition to check three-digit numbers
def is_valid_number (n : ℕ) : Prop :=
  let d1 := n / 100 in
  let d2 := (n % 100) / 10 in
  let d3 := n % 10 in
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 ∧
  d1 ≠ 0 ∧
  d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧
  d1 + d2 + d3 = 9

-- Define the number of valid three-digit numbers
def count_valid_numbers : ℕ :=
  (Finset.range 900).filter is_valid_number).card

-- The theorem stating the count of valid three-digit numbers
theorem valid_three_digit_numbers_count : count_valid_numbers = 16 :=
by
  sorry

end valid_three_digit_numbers_count_l385_385444


namespace cos_210_eq_neg_sqrt_3_div_2_l385_385830

theorem cos_210_eq_neg_sqrt_3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by
  -- leave proof as sorry
  sorry

end cos_210_eq_neg_sqrt_3_div_2_l385_385830


namespace translation_vector_condition_l385_385671

theorem translation_vector_condition (m n : ℝ) :
  (∀ x : ℝ, 2 * (x - m) + n = 2 * x + 5) → n = 2 * m + 5 :=
by
  intro h
  -- proof can be filled here
  sorry

end translation_vector_condition_l385_385671


namespace primes_infinite_l385_385316

theorem primes_infinite : ∀ (S : Set ℕ), (∀ p, p ∈ S → Nat.Prime p) → (∃ a, a ∉ S ∧ Nat.Prime a) :=
by
  sorry

end primes_infinite_l385_385316


namespace average_speed_is_39_01_l385_385755

noncomputable def total_distance : ℝ := 3
noncomputable def speed1 : ℝ := 80
noncomputable def speed2 : ℝ := 24
noncomputable def speed3 : ℝ := 44

noncomputable def time1 : ℝ := 1 / speed1
noncomputable def time2 : ℝ := 1 / speed2
noncomputable def time3 : ℝ := 1 / speed3

noncomputable def total_time : ℝ := time1 + time2 + time3

theorem average_speed_is_39_01 :
  (total_distance / total_time) ≈ 39.01 :=
by
  sorry

end average_speed_is_39_01_l385_385755


namespace min_value_of_a_l385_385929

theorem min_value_of_a 
  (a b x1 x2 : ℕ) 
  (h1 : a = b - 2005) 
  (h2 : (x1 + x2) = a) 
  (h3 : (x1 * x2) = b) 
  (h4 : x1 > 0 ∧ x2 > 0) : 
  a ≥ 95 :=
sorry

end min_value_of_a_l385_385929


namespace Shawna_situps_l385_385323

theorem Shawna_situps :
  ∀ (goal_per_day : ℕ) (total_days : ℕ) (tuesday_situps : ℕ) (wednesday_situps : ℕ),
  goal_per_day = 30 →
  total_days = 3 →
  tuesday_situps = 19 →
  wednesday_situps = 59 →
  (goal_per_day * total_days) - (tuesday_situps + wednesday_situps) = 12 :=
by
  intros goal_per_day total_days tuesday_situps wednesday_situps
  sorry

end Shawna_situps_l385_385323


namespace cos_210_eq_neg_sqrt3_div_2_l385_385860

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385860


namespace simplify_and_evaluate_l385_385655

theorem simplify_and_evaluate (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) (h3 : x ≠ 1) :
  (x = -1) → ( (x-1) / (x^2 - 2*x + 1) / ((x^2 + x - 1) / (x-1) - (x + 1)) - 1 / (x - 2) = -2 / 3 ) :=
by 
  intro hx
  rw [hx]
  sorry

end simplify_and_evaluate_l385_385655


namespace find_books_second_shop_l385_385322

def total_books (books_first_shop books_second_shop : ℕ) : ℕ :=
  books_first_shop + books_second_shop

def total_cost (cost_first_shop cost_second_shop : ℕ) : ℕ :=
  cost_first_shop + cost_second_shop

def average_price (total_cost total_books : ℕ) : ℕ :=
  total_cost / total_books

theorem find_books_second_shop : 
  ∀ (books_first_shop cost_first_shop cost_second_shop : ℕ),
    books_first_shop = 65 →
    cost_first_shop = 1480 →
    cost_second_shop = 920 →
    average_price (total_cost cost_first_shop cost_second_shop) (total_books books_first_shop (2400 / 20 - 65)) = 20 →
    2400 / 20 - 65 = 55 := 
by sorry

end find_books_second_shop_l385_385322


namespace vector_dot_product_condition_l385_385589

theorem vector_dot_product_condition
  (A B C D M : EuclideanGeometry.Point)
  (h_triangle : EuclideanGeometry.is_right_triangle A B C)
  (h_angle : ∠B = 90)
  (h_AB : EuclideanGeometry.distance A B = 3)
  (h_BC : EuclideanGeometry.distance B C = 3)
  (h_BD_DC : 2 • EuclideanGeometry.Vector B D = EuclideanGeometry.Vector B C)
  (h_midpoint_M : EuclideanGeometry.is_midpoint M A D) :
  ∀ v_AM v_BC : EuclideanGeometry.Vector,
    v_AM = (1/2 : ℝ) • (EuclideanGeometry.Vector A B + 1/3 • EuclideanGeometry.Vector B C) →
    v_BC = EuclideanGeometry.Vector B C →
    EuclideanGeometry.dot_product v_AM v_BC = 3 :=
sorry

end vector_dot_product_condition_l385_385589


namespace ratio_of_areas_between_inscribed_circles_l385_385430

theorem ratio_of_areas_between_inscribed_circles (s : ℝ) (hs : s > 0) :
  let rL := s / 2
  let rS := s / (2 * Real.sqrt 2)
  let areaL := Real.pi * rL^2
  let areaS := Real.pi * rS^2
  areaS / areaL = 1 / 2 :=
by
  unfold rL rS areaL areaS
  sorry

end ratio_of_areas_between_inscribed_circles_l385_385430


namespace cos_210_eq_neg_sqrt3_div_2_l385_385864

theorem cos_210_eq_neg_sqrt3_div_2 : real.cos (210 * real.pi / 180) = - (real.sqrt 3 / 2) :=
by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385864


namespace second_term_of_geo_series_l385_385046

theorem second_term_of_geo_series
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h_r : r = -1 / 3)
  (h_S : S = 25)
  (h_sum : S = a / (1 - r)) :
  (a * r) = -100 / 9 :=
by
  -- Definitions and conditions here are provided
  have hr : r = -1 / 3 := by exact h_r
  have hS : S = 25 := by exact h_S
  have hsum : S = a / (1 - r) := by exact h_sum
  -- The proof of (a * r) = -100 / 9 goes here
  sorry

end second_term_of_geo_series_l385_385046


namespace square_area_inscribed_in_circle_l385_385795

-- Define the conditions
def area_of_circle (r : ℝ) : ℝ := π * r^2
def inscribed_square_area (r : ℝ) : ℝ := (2 * r / Real.sqrt 2)^2

-- Given the conditions, prove the question
theorem square_area_inscribed_in_circle (h : area_of_circle 5 = 25 * π) : inscribed_square_area 5 = 50 :=
by
  -- given the conditions, we need to derive the required proof
  sorry

end square_area_inscribed_in_circle_l385_385795


namespace max_value_of_g_max_g_value_l385_385880

noncomputable def g : ℕ → ℕ
| n := if n ≤ 12 
       then n^2 + 20
       else g (n - 7)

theorem max_value_of_g : ∀ n, g n ≤ 164 :=
by sorry

theorem max_g_value : ∃ n, g n = 164 :=
by sorry

end max_value_of_g_max_g_value_l385_385880


namespace sum_of_possible_values_l385_385753

theorem sum_of_possible_values (A B : ℕ) 
  (hA1 : A < 10) (hA2 : 0 < A) (hB1 : B < 10) (hB2 : 0 < B)
  (h1 : 3 / 12 < A / 12) (h2 : A / 12 < 7 / 12)
  (h3 : 1 / 10 < 1 / B) (h4 : 1 / B < 1 / 3) :
  3 + 6 = 9 :=
by
  sorry

end sum_of_possible_values_l385_385753


namespace large_seat_capacity_l385_385330

-- Definition of conditions
def num_large_seats : ℕ := 7
def total_capacity_large_seats : ℕ := 84

-- Theorem to prove
theorem large_seat_capacity : total_capacity_large_seats / num_large_seats = 12 :=
by
  sorry

end large_seat_capacity_l385_385330


namespace grasshopper_jump_l385_385677

-- Definitions for the distances jumped
variables (G F M : ℕ)

-- Conditions given in the problem
def condition1 : Prop := G = F + 19
def condition2 : Prop := M = F - 12
def condition3 : Prop := M = 8

-- The theorem statement
theorem grasshopper_jump : condition1 G F ∧ condition2 F M ∧ condition3 M → G = 39 :=
by
  sorry

end grasshopper_jump_l385_385677


namespace ellipse_equation_and_m_range_l385_385945

noncomputable def ellipse_eq (x y a b : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def parabola_focus : ℝ × ℝ := (Real.sqrt 2, 0)

theorem ellipse_equation_and_m_range 
  (a b : ℝ) 
  (ellipse_shared_focus : ∃ c, ellipse_eq c 0 a b ∧ c = Real.sqrt 2)
  (quadrilateral_area : 2 * a * b = 4 * Real.sqrt 3) 
  (k m : ℝ) 
  (k_ne_zero : k ≠ 0) 
  (triangle_isosceles : (∃ A M N : ℝ × ℝ, A = (0, -b) 
                         ∧ ellipse_eq A.1 A.2 a b
                         ∧ ellipse_eq M.1 M.2 a b 
                         ∧ ellipse_eq N.1 N.2 a b
                         ∧ M ≠ N 
                         ∧ ∃ P : ℝ × ℝ, P = (M.1 + N.1) / 2, (M.2 + N.2) / 2
                         ∧ (P.1 = -(3*m*k) / (3*k^2 + 1))
                         ∧ (P.2 = m / (3*k^2 + 1))
                         ∧ ∃ x₁ x₂ : ℝ, x₁ ∈ set.range (λ x, (x, k*x + m))
                         ∧ ((k * x₁ + m + 1) / (3 * k * m) = 1/k)
                         ∧ (triangle A M N x₁ x₂ P is_isosceles)))
  : (a = Real.sqrt 3) ∧ (b = 1) ∧ (∀ m, 1/2 < m ∧ m < 2) := by
  sorry

end ellipse_equation_and_m_range_l385_385945


namespace cos_210_eq_neg_sqrt3_div_2_l385_385849

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385849


namespace sum_first_ten_terms_arithmetic_l385_385268

def arithmetic_sequence_sum (a₁ : ℤ) (n : ℕ) (d : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem sum_first_ten_terms_arithmetic (a₁ a₂ a₆ d : ℤ) 
  (h1 : a₁ = -2) 
  (h2 : a₂ + a₆ = 2) 
  (common_diff : d = 1) :
  arithmetic_sequence_sum a₁ 10 d = 25 :=
by
  rw [h1, common_diff]
  sorry

end sum_first_ten_terms_arithmetic_l385_385268


namespace proof_problem_l385_385151

noncomputable def find_slope (m : ℝ) : ℝ := m / 2
noncomputable def midpoint (x1 y1 x2 y2 : ℝ) : (ℝ, ℝ) :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Given conditions
def direction_vector (m : ℝ) := (2, m)
def point_M := (1, 0)
def point_N (m : ℝ) := (2 * m - 1, m)
def line_l_slope := -1
def intercept_y (a : ℝ) := a / 2

-- Translated Proof Problem
theorem proof_problem (m : ℝ) (A : ℝ × ℝ) :
  find_slope m = line_l_slope →
  A = midpoint (1 : ℝ) 0 (2 * m - 1) m →
  A = (-2, -1) ∧ ((∃ a : ℝ, (A.1 / a + A.2 / intercept_y a = 1) ∧ a = -4) ∨ (A.1 - 2 * A.2 = 0)) :=
by
  sorry

end proof_problem_l385_385151


namespace option_A_option_C_option_D_l385_385379

noncomputable def ratio_12_11 := (12 : ℝ) / 11
noncomputable def ratio_11_10 := (11 : ℝ) / 10

theorem option_A : ratio_12_11^11 > ratio_11_10^10 := sorry

theorem option_C : ratio_12_11^10 > ratio_11_10^9 := sorry

theorem option_D : ratio_11_10^12 > ratio_12_11^13 := sorry

end option_A_option_C_option_D_l385_385379


namespace coprime_solution_l385_385908

theorem coprime_solution (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_eq : 5 * a + 7 * b = 29 * (6 * a + 5 * b)) : a = 3 ∧ b = 2 :=
sorry

end coprime_solution_l385_385908


namespace number_of_perfect_squares_or_evens_l385_385427

theorem number_of_perfect_squares_or_evens (s: Set (ℕ×ℕ)) 
  (h: s = { n | ∃ k, k ∈ finset.range (135) ∧ (n = 100 + k) ∧ (100 ≤ n ∧ n ≤ 234) ∧ (even k ∨ ∃ m, m * m = n)}) : 
  s.card = 71 := 
by 
  -- sorry is used to skip the actual proof
  sorry

end number_of_perfect_squares_or_evens_l385_385427


namespace number_of_a_values_l385_385612

def d1 (a : ℤ) : ℤ := a^2 + 3^a + a * 3^((a + 1) / 2)
def d2 (a : ℤ) : ℤ := a^2 + 3^a - a * 3^((a + 1) / 2)

theorem number_of_a_values (h : 1 ≤ a ∧ a ≤ 300) :
  ∃ n, n = 43 ∧ (d1 a) * (d2 a) % 7 = 0 :=
sorry

end number_of_a_values_l385_385612


namespace sum_max_min_a_l385_385181

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x^2

theorem sum_max_min_a
  (e : ℝ) (he : Real.exp 1 = e)
  (a : ℝ)
  (g : ℝ → ℝ) (hg : ∀ x, g x = a - x^2)
  (h : ℝ → ℝ) (hh : ∀ x, h x = 2 * Real.log x)
  (interv : ∀ (x : ℝ), 1 / e ≤ x ∧ x ≤ e) 
  (symm : ∃ x ∈ Icc (1 / e : ℝ) e, g x = - h x) :
  (∃ (xmin xmax : ℝ), xmin = 1 ∧ xmax = e^2 - 2) →
  (xmin + xmax = e^2 - 1) :=
begin
  sorry  -- Proof not required, only the statement.
end

end sum_max_min_a_l385_385181


namespace largest_expression_l385_385997

noncomputable def x : ℝ := 1 / 4

def expr_A : ℝ := x
def expr_B : ℝ := x^2
def expr_C : ℝ := (1 / 2) * x
def expr_D : ℝ := 1 / x
def expr_E : ℝ := Real.sqrt x

theorem largest_expression :
  expr_D > expr_A ∧ expr_D > expr_B ∧ expr_D > expr_C ∧ expr_D > expr_E := by
  sorry

end largest_expression_l385_385997


namespace cos_210_eq_neg_sqrt3_div_2_l385_385854

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385854


namespace find_number_of_roses_l385_385772

theorem find_number_of_roses : ∃ a : ℕ, 300 ≤ a ∧ a ≤ 400 ∧ a % 21 = 13 ∧ a % 15 = 7 :=
by
  -- Existential quantifier for the number 'a'
  use 307
  
  -- Proof of the conditions for 'a'
  split
  -- Proof that 300 ≤ 307 ∧ 307 ≤ 400
  exact ⟨by linarith, by linarith⟩
  split
  -- Proof that 307 % 21 = 13
  exact by norm_num
  -- Proof that 307 % 15 = 7 (because -8 % 15 = 7)
  exact by norm_num

end find_number_of_roses_l385_385772


namespace complex_number_solution_l385_385220

theorem complex_number_solution
  (z : ℂ)
  (h : i * (z - 1) = 1 + i) :
  z = 2 - i :=
sorry

end complex_number_solution_l385_385220


namespace vector_subtraction_l385_385974

variable (a b : ℝ × ℝ)

def vector_calc (a b : ℝ × ℝ) : ℝ × ℝ :=
  (2 * a.1 - b.1, 2 * a.2 - b.2)

theorem vector_subtraction :
  a = (2, 4) → b = (-1, 1) → vector_calc a b = (5, 7) := by
  intros ha hb
  simp [vector_calc]
  rw [ha, hb]
  simp
  sorry

end vector_subtraction_l385_385974


namespace cos_210_eq_neg_sqrt3_div_2_l385_385867

theorem cos_210_eq_neg_sqrt3_div_2 : real.cos (210 * real.pi / 180) = - (real.sqrt 3 / 2) :=
by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385867


namespace focal_length_of_hyperbola_is_four_l385_385159

noncomputable def hyperbola_focal_length (m : ℝ) (hm : m > 0) : ℝ :=
  let a := Real.sqrt m in
  let b := 1 in
  2 * Real.sqrt (a ^ 2 + b ^ 2)

theorem focal_length_of_hyperbola_is_four (m : ℝ) (hm : m > 0)
  (h_asymptote : ∃ (k : ℝ), k ≠ 0 ∧ k * sqrt 3 + m = 0) : hyperbola_focal_length m hm = 4 :=
sorry

end focal_length_of_hyperbola_is_four_l385_385159


namespace correct_starting_position_l385_385404

variable a b c d e f : Type
variable children : List (Type)
variable elim_order : List (Type)

-- Conditions
def standing_in_circle : Prop :=
  children = [a, b, c, d, e, f]

def ninth_elimination (start : Type) : List (Type) :=
  -- Define the elimination process according to the ninth count
  -- This would be the logic of how every ninth child is eliminated
  sorry

def last_remaining_child (start : Type) (order : List (Type)) : Type :=
  -- Define how the last remaining child is found
  sorry

-- Given conditions
def conditions : Prop :=
  standing_in_circle ∧ 
  (∀ start, last_remaining_child start (ninth_elimination start) = c)

-- The goal is to prove that the correct starting position is f
theorem correct_starting_position (conditions : Prop) : 
  (last_remaining_child f (ninth_elimination f) = c) :=
  sorry

end correct_starting_position_l385_385404


namespace sum_series_l385_385083

theorem sum_series (n : ℕ) (h : n > 0) : 
  (finset.range n).sum (λ k, (k + 1 : ℚ) / 4 ^ (k + 1)) = (4 / 3) * (1 - 1 / 4 ^ n) :=
by
  sorry

end sum_series_l385_385083


namespace min_value_reciprocal_l385_385154

theorem min_value_reciprocal (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1) :
  (1 / a + 1 / b) ≥ 4 :=
by
  sorry

end min_value_reciprocal_l385_385154


namespace cos_210_eq_neg_sqrt3_div_2_l385_385856

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385856


namespace triangle_angle_sqrt_inequality_l385_385745

-- Problem 1
theorem triangle_angle : ∀ (A B C : ℝ), A + B + C = 180 -> (A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60) :=
by
  sorry

-- Problem 2
theorem sqrt_inequality (n : ℝ) (h : n ≥ 0) : 
  sqrt (n + 2) - sqrt (n + 1) ≤ sqrt (n + 1) - sqrt n :=
by
  sorry

end triangle_angle_sqrt_inequality_l385_385745


namespace problem_l385_385338

theorem problem 
  (k a b c : ℝ)
  (h1 : (3 : ℝ)^2 - 7 * 3 + k = 0)
  (h2 : (a : ℝ)^2 - 7 * a + k = 0)
  (h3 : (b : ℝ)^2 - 8 * b + (k + 1) = 0)
  (h4 : (c : ℝ)^2 - 8 * c + (k + 1) = 0) :
  a + b * c = 17 := sorry

end problem_l385_385338


namespace hallie_hourly_wage_is_10_l385_385979

noncomputable def hallie_hourly_wage : ℕ :=
  let monday_hours : ℕ := 7
  let tuesday_hours : ℕ := 5
  let wednesday_hours : ℕ := 7
  let monday_tips : ℕ := 18
  let tuesday_tips : ℕ := 12
  let wednesday_tips : ℕ := 20
  let total_hours : ℕ := monday_hours + tuesday_hours + wednesday_hours
  let total_tips : ℕ := monday_tips + tuesday_tips + wednesday_tips
  let total_earnings : ℕ := 240
  let hourly_wage := ((total_earnings - total_tips) / total_hours : ℕ)
  hourly_wage

theorem hallie_hourly_wage_is_10 : hallie_hourly_wage = 10 :=
by 
  let monday_hours : ℕ := 7
  let tuesday_hours : ℕ := 5
  let wednesday_hours : ℕ := 7
  let monday_tips : ℕ := 18
  let tuesday_tips : ℕ := 12
  let wednesday_tips : ℕ := 20
  let total_hours : ℕ := monday_hours + tuesday_hours + wednesday_hours
  let total_tips : ℕ := monday_tips + tuesday_tips + wednesday_tips
  let total_earnings : ℕ := 240
  have equation : 19 * 10 + 50 = 240 := by norm_num
  show hallie_hourly_wage = 10 from
  have hourly_wage := ((total_earnings - total_tips) / total_hours : ℕ) 
  calc hourly_wage
      = (total_earnings - total_tips) / total_hours : rfl
  ... = 190 / 19 : by norm_num
  ... = 10 : by norm_num

#eval hallie_hourly_wage_is_10 -- should return true

end hallie_hourly_wage_is_10_l385_385979


namespace common_tangent_circles_l385_385581

noncomputable def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

noncomputable def circle2 (x y a : ℝ) : Prop :=
  (x + 4)^2 + (y - a)^2 = 25

theorem common_tangent_circles :
  ∃ a : ℝ, (∃ x y : ℝ, circle1 x y ∧ circle2 x y a) → a = 2 * Real.sqrt 5 ∨ a = -2 * Real.sqrt 5 :=
begin
  sorry
end

end common_tangent_circles_l385_385581


namespace find_smallest_r_disjoint_set_l385_385294

theorem find_smallest_r_disjoint_set 
  (A : Set ℕ := {a | ∃ k : ℕ, a = 3 + 10 * k ∨ a = 6 + 26 * k ∨ a = 5 + 29 * k})
  : ∃ (r b : ℕ), (∀ k : ℕ, b + r * k ∉ A) ∧ r = 290 :=
by
  sorry

end find_smallest_r_disjoint_set_l385_385294


namespace linear_function_comparison_linear_function_increase_linear_function_exceeds_100_first_l385_385964

theorem linear_function_comparison (x : ℝ) :
  let y1 := 50 + 2 * x,
      y2 := 5 * x in
  (y1 = y2 ↔ x = 50 / 3) ∧ (y1 > y2 ↔ x < 50 / 3) ∧ (y1 < y2 ↔ x > 50 / 3) :=
sorry

theorem linear_function_increase (x : ℝ) :
  let y1 := 50 + 2 * x,
      y2 := 5 * x in
  (∀ x, (50 + 2 * (x + 1)) = y1 + 2) ∧ (∀ x, (5 * (x + 1)) = y2 + 5) :=
sorry

theorem linear_function_exceeds_100_first :
  let y1 := 50 + 2 * (25 : ℝ),
      y2 := 5 * (20 : ℝ) in
  ∃ x1 x2 : ℝ, y1 = 100 ∧ y2 = 100 ∧ x2 = 20 ∧ x1 = 25 ∧ x2 < x1 :=
sorry

end linear_function_comparison_linear_function_increase_linear_function_exceeds_100_first_l385_385964


namespace product_of_valid_a_values_l385_385491

noncomputable def valid_a_values (a : ℝ) : Prop :=
  ∃ x1 x2 : ℤ, (x1 * x2 = -8 * ↑a) ∧ (x1 + x2 = -2 * ↑a) ∧ (x1 ≠ x2)

theorem product_of_valid_a_values :
  ∏ a in {a | valid_a_values a}.to_finset, a = 506.25 :=
sorry

end product_of_valid_a_values_l385_385491


namespace symmetry_P_over_xOz_l385_385593

-- Definition for the point P and the plane xOz
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def P : Point3D := { x := 2, y := 3, z := 4 }

def symmetry_over_xOz_plane (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

theorem symmetry_P_over_xOz : symmetry_over_xOz_plane P = { x := 2, y := -3, z := 4 } :=
by
  -- The proof is omitted.
  sorry

end symmetry_P_over_xOz_l385_385593


namespace ratio_Sally_to_Mater_l385_385624

def cost_Lightning_McQueen : ℕ := 140000
def cost_Mater : ℕ := 0.1 * cost_Lightning_McQueen
def cost_Sally_McQueen : ℕ := 42000

theorem ratio_Sally_to_Mater : cost_Sally_McQueen / cost_Mater = 3 := by
  sorry

end ratio_Sally_to_Mater_l385_385624


namespace vectors_orthogonal_l385_385969

theorem vectors_orthogonal (a b : ℝ^n) (ha : a ≠ 0) (hb : b ≠ 0) (h : ∥a + b∥ = ∥a - b∥) : 
  a.dot b = 0 := 
by 
  sorry

end vectors_orthogonal_l385_385969


namespace sum_g_eq_inv_m_cubed_l385_385509

def g (m n : ℕ) : ℝ :=
  ∑' i , (1 / (m + i) ^ n)

theorem sum_g_eq_inv_m_cubed (m : ℕ) (h : m ≥ 2) : 
  ∑' n in (Finset.Ico 2 (Nat.succ Nat.succ)), g m n = 1 / m^3 :=
sorry

end sum_g_eq_inv_m_cubed_l385_385509


namespace find_perfect_square_l385_385725

theorem find_perfect_square :
  ∃ n ∈ {10, 11, 12, 13, 14}, is_integral_domain.Ring (n:ℕ), 
  ∃ k:ℕ, (n + 1) = 3 * k^2 :=
by
  sorry

end find_perfect_square_l385_385725


namespace tom_killed_enemies_l385_385712

-- Define the number of points per enemy
def points_per_enemy : ℝ := 10

-- Define the bonus threshold and bonus factor
def bonus_threshold : ℝ := 100
def bonus_factor : ℝ := 1.5

-- Define the total score achieved by Tom
def total_score : ℝ := 2250

-- Define the number of enemies killed by Tom
variable (E : ℝ)

-- The proof goal
theorem tom_killed_enemies 
  (h1 : E ≥ bonus_threshold)
  (h2 : bonus_factor * points_per_enemy * E = total_score) : 
  E = 150 :=
sorry

end tom_killed_enemies_l385_385712


namespace limit_sum_b_n_l385_385168

noncomputable def a_n (n : ℕ) : ℤ :=
  if n = 0 then 0 else (-1 / 2 * n^2 - 1 / 2 * n - (-1 / 2 * (n - 1)^2 - 1 / 2 * (n - 1))).toInt

noncomputable def S_n (n : ℕ) : ℤ := (-1 / 2 * n^2 - 1 / 2 * n).toInt

noncomputable def b_n (n : ℕ) : ℝ := (2 : ℝ)^(a_n n)

theorem limit_sum_b_n : 
  (lim at_top (λ n, ∑ i in Finset.range n, b_n (i + 1)) = 1) :=
sorry

end limit_sum_b_n_l385_385168


namespace part1_part2_part3_l385_385363

variable (prob_A : ℚ) (prob_B : ℚ)
variables (prob_miss_A : ℚ) (prob_miss_B : ℚ)

theorem part1 :
  prob_A = 2 / 3 →
  prob_B = 3 / 4 →
  prob_A * prob_B = 1 / 2 :=
by
  intros h1 h2
  sorry

theorem part2 :
  prob_A = 2 / 3 →
  prob_miss_A = 1 / 3 →
  (prob_A ^ 3 * prob_miss_A + prob_miss_A * prob_A ^ 3) = 16 / 81 :=
by
  intros h1 h2
  sorry

theorem part3 :
  prob_B = 3 / 4 →
  prob_miss_B = 1 / 4 →
  (prob_B ^ 2 * prob_miss_B ^ 2 + prob_miss_B * prob_B * prob_miss_B ^ 2) = 3 / 64 :=
by
  intros h1 h2
  sorry

end part1_part2_part3_l385_385363


namespace speed_of_river_l385_385026

-- Definitions of the conditions
def rowing_speed_still_water := 9 -- kmph in still water
def total_time := 1 -- hour for a round trip
def total_distance := 8.84 -- km

-- Distance to the place the man rows to
def d := total_distance / 2

-- Problem statement in Lean 4
theorem speed_of_river (v : ℝ) : 
  rowing_speed_still_water = 9 ∧
  total_time = 1 ∧
  total_distance = 8.84 →
  (4.42 / (rowing_speed_still_water + v) + 4.42 / (rowing_speed_still_water - v) = 1) →
  v = 1.2 := 
by
  sorry

end speed_of_river_l385_385026


namespace ginger_total_water_l385_385916

def hours_worked : Nat := 8
def cups_per_bottle : Nat := 2
def bottles_drank_per_hour : Nat := 1
def bottles_for_plants : Nat := 5

theorem ginger_total_water : 
  (hours_worked * cups_per_bottle * bottles_drank_per_hour) + (bottles_for_plants * cups_per_bottle) = 26 :=
by
  sorry

end ginger_total_water_l385_385916


namespace dvaneft_percentage_bounds_l385_385416

theorem dvaneft_percentage_bounds (x y z : ℝ) (n m : ℕ) 
  (h1 : x * n + y * m = z * (m + n))
  (h2 : 3 * x * n = y * m)
  (h3_1 : 10 ≤ y - x)
  (h3_2 : y - x ≤ 18)
  (h4_1 : 18 ≤ z)
  (h4_2 : z ≤ 42)
  : (15 ≤ (n:ℝ) / (2 * (n + m)) * 100) ∧ ((n:ℝ) / (2 * (n + m)) * 100 ≤ 25) :=
by
  sorry

end dvaneft_percentage_bounds_l385_385416


namespace problem_proof_l385_385990

theorem problem_proof (x : ℝ) (h : x + 1/x = 3) : (x - 3) ^ 2 + 36 / (x - 3) ^ 2 = 12 :=
sorry

end problem_proof_l385_385990


namespace total_pages_in_book_l385_385702

theorem total_pages_in_book : 
  ∀ (n : ℕ), (∑ i in finset.range n, if i < 10 then 1 else if i < 100 then 2 else 3) = 990 → n = 366 := by
  intro n h
  sorry

end total_pages_in_book_l385_385702


namespace options_correct_l385_385937

-- Assume the domains and functions are appropriately defined
axiom f : ℝ → ℝ
axiom g : ℝ → ℝ

-- Conditions from the given problem
axiom domain_f : ∀ x : ℝ, f x ∈ ℝ  -- The domain of f(x) is ℝ
axiom fg_add_one : ∀ x : ℝ, g x + f x = 1  -- g(x) + f(x) = 1
axiom odd_g : ∀ x : ℝ, g (x + 1) = -g (-x + 1)  -- g(x + 1) is odd
axiom odd_f : ∀ x : ℝ, f (2 - x) = -f (2 + x)  -- f(2 - x) is odd

-- Theorem statement confirming the correct options
theorem options_correct : 
  g 0 = -1 ∧ g 1 = 0 ∧ g 2 = 1 ∧ g 3 ≠ 0 :=
by
  sorry

end options_correct_l385_385937


namespace number_of_subsets_of_C_l385_385968

def set_A : Set ℕ := {1, 2, 3, 4, 5}
def set_B : Set ℕ := {1, 3, 5, 7, 9}
def set_C : Set ℕ := set_A ∩ set_B
def num_subsets (S : Set α) : ℕ := 2 ^ S.toFinset.card

theorem number_of_subsets_of_C : num_subsets set_C = 8 :=
by
  sorry

end number_of_subsets_of_C_l385_385968


namespace solution_set_of_inequality_l385_385692

theorem solution_set_of_inequality (x : ℝ) : ((x - 1) * (2 - x) ≥ 0) ↔ (1 ≤ x ∧ x ≤ 2) :=
sorry

end solution_set_of_inequality_l385_385692


namespace sin_B_in_triangle_ABC_l385_385595

noncomputable def triangle := Type*
variables {A B C : triangle}
variables (AC BC : ℝ)

-- The given conditions
def right_angle_at_A (A B C : triangle) : Prop := true -- Placeholder for \angle A = 90^\circ
def length_AC : ℝ := 4
def length_BC : ℝ := real.sqrt 41

-- The theorem we want to prove
theorem sin_B_in_triangle_ABC 
  (h_right : right_angle_at_A A B C)
  (h_AC : AC = length_AC)
  (h_BC : BC = length_BC) :
  real.sin (real.atan (AC / BC)) = 4 / real.sqrt 41 :=
sorry

end sin_B_in_triangle_ABC_l385_385595


namespace correct_option_B_l385_385043

/-- Representation of lines and planes, and their relationships -/
variables (l m : Line) (α β : Plane)

/-- Definition of perpendicular and parallel relationships -/
def perp (x y : Plane) : Prop := sorry
def parallel (x y : Plane) : Prop := sorry
def subset (l : Line) (p : Plane) : Prop := sorry

/-- The main theorem to be proved -/
theorem correct_option_B (l : Line) (α β : Plane) :
  (perp l β) ∧ (parallel α β) → (perp l α) :=
sorry

end correct_option_B_l385_385043


namespace triangle_foci_angle_90_l385_385296

noncomputable def ellipse_point (x y : ℝ) : Prop :=
  (x^2 / 100 + y^2 / 36 = 1)

def foci1 : ℝ × ℝ := (-8, 0)

def foci2 : ℝ × ℝ := (8, 0)

def triangle_area (P : ℝ × ℝ) : Prop :=
  let ⟨x, y⟩ := P in
  36 = 0.5 * abs ((x * (0 - 0)) + (-8 * (0 - y)) + (8 * (y - 0)))

def angle_condition (P : ℝ × ℝ) : Prop :=
  let ⟨x, y⟩ := P in
  (foci1.1 - x) * (foci2.1 - x) + (foci1.2 - y) * (foci2.2 - y) = 90

theorem triangle_foci_angle_90 (P : ℝ × ℝ) (h1 : ellipse_point P.1 P.2)
  (h2 : triangle_area P) : angle_condition P :=
sorry

end triangle_foci_angle_90_l385_385296


namespace sin_arcsin_plus_arctan_l385_385899

theorem sin_arcsin_plus_arctan :
  let a := Real.arcsin (4/5)
  let b := Real.arctan 1
  Real.sin (a + b) = (7 * Real.sqrt 2) / 10 := by
  sorry

end sin_arcsin_plus_arctan_l385_385899


namespace midpoint_of_KB_l385_385395

variables {A B C P X Y L K : Type}
variables [circle_geom : has_circle_geometry P C X Y]
variables [triangle_geom : has_triangle_geometry ABC A B P Y A C P X]
variables [lines_and_intersections : has_perpendicular_and_intersections r s A Y P X A B L K]

def angle_A_eq_45 (ABC : triangle) : Prop :=
  angle A = 45

def is_midpoint_of (L : Point) (K : Point) (B : Point) : Prop := 
  distance L K = distance L B

theorem midpoint_of_KB 
    (h_triangle : is_acute_triangle ABC)
    (h_angle_A : angle_A_eq_45 ABC)
    (h_conditions : circle_geom ∧ triangle_geom ∧ lines_and_intersections)
    : is_midpoint_of L K B :=
sorry

end midpoint_of_KB_l385_385395


namespace factorize_x_squared_minus_one_l385_385102

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_x_squared_minus_one_l385_385102


namespace log_x_16_l385_385572

theorem log_x_16 (x : ℝ) (h : log 8 (5 * x) = 3) : log x 16 = 36 / (1 - 9 * log 5 2) :=
by
  sorry

end log_x_16_l385_385572


namespace problem_1_problem_2_l385_385558

-- Definitions for sets A and B
def A (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 6
def B (x : ℝ) (m : ℝ) : Prop := (x - 1 + m) * (x - 1 - m) ≤ 0

-- Problem (1): What is A ∩ B when m = 3
theorem problem_1 : ∀ (x : ℝ), A x → B x 3 → (-1 ≤ x ∧ x ≤ 4) := by
  intro x hA hB
  sorry

-- Problem (2): What is the range of m if A ⊆ B and m > 0
theorem problem_2 (m : ℝ) : m > 0 → (∀ x, A x → B x m) → (m ≥ 5) := by
  intros hm hAB
  sorry

end problem_1_problem_2_l385_385558


namespace solution_set_of_inequalities_l385_385696

-- Define the conditions of the inequality system
def inequality1 (x : ℝ) : Prop := x - 2 ≥ -5
def inequality2 (x : ℝ) : Prop := 3 * x < x + 2

-- The statement to prove the solution set of the inequalities
theorem solution_set_of_inequalities :
  { x : ℝ | inequality1 x ∧ inequality2 x } = { x : ℝ | -3 ≤ x ∧ x < 1 } :=
  sorry

end solution_set_of_inequalities_l385_385696


namespace largest_expression_l385_385996

noncomputable def x : ℝ := 1 / 4

def expr_A : ℝ := x
def expr_B : ℝ := x^2
def expr_C : ℝ := (1 / 2) * x
def expr_D : ℝ := 1 / x
def expr_E : ℝ := Real.sqrt x

theorem largest_expression :
  expr_D > expr_A ∧ expr_D > expr_B ∧ expr_D > expr_C ∧ expr_D > expr_E := by
  sorry

end largest_expression_l385_385996


namespace k_range_l385_385148

def y_increasing (k : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → k * x₁ + 1 < k * x₂ + 1
def y_max_min (k : ℝ) : Prop := (∃ x : ℝ, 0 ≤ x ∧ x ≤ k ∧ (x^2 - 2 * x + 3 = 2)) ∧ (∃ x : ℝ, 0 ≤ x ∧ x ≤ k ∧ (x^2 - 2 * x + 3 = 3))

theorem k_range (k : ℝ) (hk : (¬ (0 < k ∧ y_max_min k) ∧ (0 < k ∨ y_max_min k))) : 
  (0 < k ∧ k < 1) ∨ (k > 2) :=
sorry

end k_range_l385_385148


namespace calculate_total_earnings_l385_385255

-- Define the pay rates for each task
def pay_A := 40
def pay_B := 50
def pay_C := 60

-- Define the hours required to complete each task
def hours_per_task := 2

-- Define the weekly work schedule (hours per day for each task)
def hours_per_day : List (Int × Int × Int) :=
  [(5, 5, 0),  -- Monday: 10 hours (Tasks A and B)
   (4, 0, 4),  -- Tuesday: 8 hours (Tasks A and C)
   (0, 6, 0),  -- Wednesday: 6 hours (Task B only)
   (2, 2, 2),  -- Thursday: 12 hours (Tasks A, B, and C)
   (0, 0, 4),  -- Friday: 4 hours (Task C only)
   (5, 5, 0),  -- Saturday: 10 hours (Tasks A and B)
   (0, 0, 0)]  -- Sunday: No work

-- Calculate the total number of each task completed during the week
def total_tasks (tasks : List (Int × Int × Int)) : (Int × Int × Int) :=
  tasks.foldl (fun (acc : Int × Int × Int) (x : Int × Int × Int) =>
    (acc.1 + x.1, acc.2 + x.2, acc.3 + x.3)) (0, 0, 0)

-- Define the main theorem to prove
theorem calculate_total_earnings : total_tasks hours_per_day =
    (9, 10, 6) → (9 * pay_A + 10 * pay_B + 6 * pay_C) = 1220 :=
by
  -- Assume the total task counts directly as inputs per the problem setup
  intro h_total_tasks
  calc
    (9 * pay_A + 10 * pay_B + 6 * pay_C)
        = (9 * 40 + 10 * 50 + 6 * 60) := by sorry
    ... = 1220 := by sorry

end calculate_total_earnings_l385_385255


namespace sum_first_10_terms_arithmetic_seq_l385_385275

theorem sum_first_10_terms_arithmetic_seq :
  ∀ (a : ℕ → ℤ), (a 1 = -2) → (a 2 + a 6 = 2) →
  (∃ S₁₀, S₁₀ = 10 * a 1 + (10 * (10 - 1)) / 2 * (a 2 - a 1) / (2 - 1) ∧ S₁₀ = 25) :=
begin
  assume a,
  assume h1 : a 1 = -2,
  assume h2 : a 2 + a 6 = 2,
  sorry
end

end sum_first_10_terms_arithmetic_seq_l385_385275


namespace part1_part2_l385_385554

def f (x : ℝ) : ℝ := |x - 1|
def g (x : ℝ) : ℝ := -x^2 + 6 * x - 5

theorem part1 : {x : ℝ | g x ≥ f x} = Set.Icc 1 4 := sorry

theorem part2 (x : ℝ) (hx : x ∈ Set.Icc 1 4) : g x - f x ≤ 9 / 4 := 
begin
  use 5 / 2,
  simp,
  sorry
end

example : g (5 / 2) - f (5 / 2) = 9 / 4 := by simp

end part1_part2_l385_385554


namespace sequence_n_l385_385113

theorem sequence_n (a : ℕ → ℕ) (h : ∀ n : ℕ, 0 < n → (n^2 + 1) * a n = n * (a (n^2) + 1)) :
  ∀ n : ℕ, 0 < n → a n = n := 
by
  sorry

end sequence_n_l385_385113


namespace book_price_increase_l385_385350

theorem book_price_increase (P : ℝ) (x : ℝ) :
  (P * (1 + x / 100)^2 = P * 1.3225) → x = 15 :=
by
  sorry

end book_price_increase_l385_385350


namespace possible_to_transform_20_twos_to_21_twos_l385_385348

def double_digit (n : Nat) : Nat × Nat :=
  let product := n * 2
  if product >= 10 then (product - 10, 1) else (product, 0)

def transform (digits : List Nat) : List Nat :=
  match digits with
  | []        => []
  | (x :: xs) => 
    let (doubled, carry) := double_digit x
    (doubled + carry) :: xs

theorem possible_to_transform_20_twos_to_21_twos :
  ∃ (steps : Nat), let initial := List.replicate 20 2
                       final := List.replicate 21 2
                   in ∀ steps ≤ steps, transform (List.replicate (20 - steps) 2 ++ List.replicate steps 1) = final :=
sorry

end possible_to_transform_20_twos_to_21_twos_l385_385348


namespace eliminate_duplicates_3n_2m1_l385_385891

theorem eliminate_duplicates_3n_2m1 :
  ∀ k: ℤ, ∃ n m: ℤ, 3 * n ≠ 2 * m + 1 ↔ 2 * m + 1 = 12 * k + 1 ∨ 2 * m + 1 = 12 * k + 5 :=
by
  sorry

end eliminate_duplicates_3n_2m1_l385_385891


namespace count_of_statements_implies_expr_l385_385468

-- Define the propositions p, q, and r
variable {p q r : Prop}

-- Define the four statements
def stmt1 := p ∧ q ∧ r
def stmt2 := ¬ p ∧ q ∧ ¬ r
def stmt3 := p ∧ ¬ q ∧ ¬ r
def stmt4 := ¬ p ∧ q ∧ r

-- Define the expression to be tested for implication
def expr := ¬ ((p → ¬ q) → r)

-- Define a function to check if a given statement implies expr
def implies_expr (s : Prop) := s → expr

-- The theorem to be proven
theorem count_of_statements_implies_expr :
  (implies_expr stmt1 ↔ true) ∧
  (implies_expr stmt2 ↔ false) ∧
  (implies_expr stmt3 ↔ false) ∧
  (implies_expr stmt4 ↔ true) →
  (∀ f : (stmt1 → expr), ∀ t : ((¬ stmt2 → expr) ∧ (stmt3 → expr) ∧ (¬ stmt4 → expr)), true) ∧
    2 = number_of_statements_that_imply (stmt1 ∧ ¬stmt2 ∧ stmt3 ∧ ¬stmt4) expr := by
sorry

end count_of_statements_implies_expr_l385_385468


namespace additional_cats_l385_385777

theorem additional_cats {M R C : ℕ} (h1 : 20 * R = M) (h2 : 4 + 2 * C = 10) : C = 3 := 
  sorry

end additional_cats_l385_385777


namespace semicircle_perimeter_l385_385006

theorem semicircle_perimeter (r : ℝ) (h : r = 7) : 2 * r + π * r ≈ 36 := by
  rw [h]
  -- Perimeter = Diameter + Half Circumference
  -- Diameter = 2 * 7
  -- Half Circumference = π * 7
  -- Perimeter = 2 * 7 + π * 7
  -- Perimeter ≈ 36 using π ≈ 3.14
  sorry

end semicircle_perimeter_l385_385006


namespace problem_statement_l385_385479

def sqrt_sum_eq (n : ℕ) : ℝ :=
  ∑ i in finset.range n, 1 / real.sqrt (i + 1 + real.sqrt ((i + 1)^2 - 4))

theorem problem_statement :
  ∃ (p q r : ℕ), p + q * real.sqrt r = sqrt_sum_eq 10000 ∧ (p > 0 ∧ q > 0 ∧ r > 0) ∧ (∀ k : ℕ, k^2 ∣ r → k = 1) ∧ (p + q + r = 5002) := sorry

end problem_statement_l385_385479


namespace faye_candy_count_l385_385396

theorem faye_candy_count :
  let initial_candy := 47
  let candy_ate := 25
  let candy_given := 40
  initial_candy - candy_ate + candy_given = 62 :=
by
  let initial_candy := 47
  let candy_ate := 25
  let candy_given := 40
  sorry

end faye_candy_count_l385_385396


namespace vector_subtraction_l385_385970

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

-- Define the expression to be proven
def expression : ℝ × ℝ := ((2 * a.1 - b.1), (2 * a.2 - b.2))

-- The theorem statement
theorem vector_subtraction : expression = (5, 7) :=
by
  -- The proof will be provided here
  sorry

end vector_subtraction_l385_385970


namespace surface_area_of_circumscribed_sphere_l385_385537

-- Define the basic structures and conditions
variables {V : Type*} [euclidean_space 4 V]

noncomputable def equilateral_triangle (a b c : V) : Prop :=
  dist a b = 4 ∧ dist b c = 4 ∧ dist c a = 4

noncomputable def circumscribed_sphere_radius (P A B C : V) : ℝ :=
  let h := 4 / real.sqrt 3 in
  let r := 4 / real.sqrt 3 in
  real.sqrt (r^2 + (h/2)^2)

noncomputable def surface_area_of_sphere (r : ℝ) : ℝ := 4 * real.pi * r^2

-- Main problem statement
theorem surface_area_of_circumscribed_sphere (P A B C O : V) :
  (PC_prop : dist P C = 2 * circumscribed_sphere_radius P A B C) →
  (equilateral_triangle ABC : equilateral_triangle A B C) →
  (tetrahedron_volume P A B C : volume P A B C = 16 / 3) →
  surface_area_of_sphere (circumscribed_sphere_radius P A B C) = 80 * real.pi / 3 :=
sorry

end surface_area_of_circumscribed_sphere_l385_385537


namespace proof_problem_l385_385531

theorem proof_problem (a b c d : ℝ) (h1 : a + b = 1) (h2 : c * d = 1) : 
  (a * c + b * d) * (a * d + b * c) ≥ 1 := 
by 
  sorry

end proof_problem_l385_385531


namespace beyonce_total_songs_l385_385810

theorem beyonce_total_songs :
  let singles := 12
  let albums := 4
  let songs_per_album := 18 + 14
  let total_album_songs := albums * songs_per_album
  let total_songs := total_album_songs + singles
  total_songs = 140 := by
  let singles := 12
  let albums := 4
  let songs_per_album := 18 + 14
  let total_album_songs := albums * songs_per_album
  let total_songs := total_album_songs + singles
  sorry

end beyonce_total_songs_l385_385810


namespace reggie_marbles_bet_l385_385318

theorem reggie_marbles_bet 
  (initial_marbles : ℕ) (final_marbles : ℕ) (games_played : ℕ) (games_lost : ℕ) (bet_per_game : ℕ)
  (h_initial : initial_marbles = 100) 
  (h_final : final_marbles = 90) 
  (h_games : games_played = 9) 
  (h_losses : games_lost = 1) : 
  bet_per_game = 13 :=
by
  sorry

end reggie_marbles_bet_l385_385318


namespace coin_payment_possible_l385_385342

theorem coin_payment_possible :
  ∃ (denominations : Finset ℕ), 
  denominations.card = 12 ∧
  (∀ (amount : ℕ), 1 ≤ amount ∧ amount ≤ 6543 → 
                    ∃ (coins : Multiset ℕ), 
                    coins.card ≤ 8 ∧ 
                    coins.sum = amount ∧ 
                    (∀ coin ∈ coins, coin ∈ denominations)) :=
sorry

end coin_payment_possible_l385_385342


namespace largest_from_selections_l385_385074

noncomputable def is_repeating (x : ℚ) : Prop := ∃ (a b : ℤ), x = a + (b / (10 ^ n - 1)) for some n > 0
noncomputable def as_decimal (x : ℚ) (digits : ℕ) : ℚ := (int_part x * (10 ^ digits) + frac_part x * (10 ^ digits))

theorem largest_from_selections :
  let A := 7.215666666...
  let B := 7.215
  let C := 7.2156156...
  let D := 7.21562156...
  let E := 7.21566 in
  max A (max B (max C (max D E))) = E := by sorry

end largest_from_selections_l385_385074


namespace rabbit_stores_60_items_l385_385583

theorem rabbit_stores_60_items :
  ∀ (h_r h_d h_f x : ℕ),
  4 * h_r = x →
  5 * h_d = x →
  7 * h_f = x →
  h_d = h_r - 3 →
  h_f = h_d + 2 →
  x = 60 :=
begin
  intros,
  sorry
end

end rabbit_stores_60_items_l385_385583


namespace largest_odd_integer_satisfying_inequality_l385_385495

theorem largest_odd_integer_satisfying_inequality : 
  ∃ (x : ℤ), (x % 2 = 1) ∧ (1 / 4 < x / 6) ∧ (x / 6 < 7 / 9) ∧ (∀ y : ℤ, (y % 2 = 1) ∧ (1 / 4 < y / 6) ∧ (y / 6 < 7 / 9) → y ≤ x) :=
sorry

end largest_odd_integer_satisfying_inequality_l385_385495


namespace cubic_inequality_l385_385480

theorem cubic_inequality :
  {x : ℝ | x^3 - 12*x^2 + 47*x - 60 < 0} = {x : ℝ | 3 < x ∧ x < 5} :=
by
  sorry

end cubic_inequality_l385_385480


namespace sum_subset_eq_100_exists_l385_385354

theorem sum_subset_eq_100_exists (nums : Fin 100 → ℕ) 
  (h1 : ∀ i, nums i < 100) 
  (h2 : (Finset.univ.sum nums) = 200) : 
  ∃ (s : Finset (Fin 100)), s.sum nums = 100 :=
begin
  sorry
end

end sum_subset_eq_100_exists_l385_385354


namespace total_roses_l385_385766

theorem total_roses (a : ℕ) (x y k : ℕ) (h1 : 300 ≤ a) (h2 : a ≤ 400)
  (h3 : a = 21 * x + 13) (h4 : a = 15 * y - 8) (h5 : a + 8 = 105 * k) :
  a = 307 :=
sorry

end total_roses_l385_385766


namespace vector_angle_pi_over_six_l385_385231

variables {α : Type*} [inner_product_space ℝ α]

def angle_between_vectors (a b : α) : ℝ :=
real.arccos (inner_product_space.inner a b / (∥a∥ * ∥b∥))

theorem vector_angle_pi_over_six 
  (a b : α) 
  (ha : ∥a∥ = real.sqrt 3) 
  (hb : ∥b∥ = 2) 
  (h_perp : ⟪a, a - b⟫ = 0) : 
  angle_between_vectors a b = π / 6 := 
sorry

end vector_angle_pi_over_six_l385_385231


namespace coordinates_of_B_l385_385234

-- Define the points O, A, B and C
structure Point where
  x : ℝ
  y : ℝ

def O : Point := { x := 0, y := 0 }
def A : Point := { x := 4, y := 3 }
def B : Point := { x := 7, y := -1 }
def C : Point := sorry -- To be defined when leveraging point in the fourth quadrant condition

-- Define the distance function between two points
def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)

-- Define the conditions
axiom condition1 : O = { x := 0, y := 0 }
axiom condition2 : A = { x := 4, y := 3 }
axiom condition3 : C.y < 0 -- C in the fourth quadrant implies y is negative
axiom condition4 : distance O A = distance A B ∧ distance A B = distance B C ∧ distance B C = distance C O -- Square condition

-- Define the theorem to prove
theorem coordinates_of_B : B = { x := 7, y := -1 } :=
  sorry

end coordinates_of_B_l385_385234


namespace minimum_distance_value_l385_385163

-- Definitions of the points and the parabola
def F : ℝ × ℝ := (0, 4)
def P : ℝ × ℝ := (2, 3)
def parabola (p : ℝ) (h : p > 0) : set (ℝ × ℝ) := {M | M.1^2 = 2 * p * M.2}

-- The conditions
axiom hF : ∃ p : ℝ, p > 0 ∧ parabola p hF.contains F
axiom hP : P = (2, 3)

-- The theorem to be proven
theorem minimum_distance_value (M : ℝ × ℝ) :
  (∃ M ∈ parabola (classical.some hF.fst) (classical.some_spec hF.fst).1, 
  M.1^2 = 16 * M.2) →
    |dist M F| + |dist M P| = 7 :=
sorry

end minimum_distance_value_l385_385163


namespace positive_integer_count_l385_385605

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem positive_integer_count (k : ℕ) 
  (p : Fin k → ℕ) 
  (hp : ∀ i, is_prime (p i)) 
  (α : Fin k → ℕ) 
  (hα : (∏ i, α i) = ∏ i, p i) :
  (∑ _ : Unit, 1) = k^k := 
sorry

end positive_integer_count_l385_385605


namespace y_intercept_of_tangent_line_l385_385218

def point (x y : ℝ) : Prop := true

def circle_eq (x y : ℝ) : ℝ := x^2 + y^2 + 4*x - 2*y + 3

theorem y_intercept_of_tangent_line :
  ∃ m b : ℝ,
  (∀ x : ℝ, circle_eq x (m*x + b) = 0 → m * m = 1) ∧
  (∃ P: ℝ × ℝ, P = (-1, 0)) ∧
  ∀ b : ℝ, (∃ m : ℝ, m = 1 ∧ (∃ P: ℝ × ℝ, P = (-1, 0)) ∧ b = 1) := 
sorry

end y_intercept_of_tangent_line_l385_385218


namespace cos_210_eq_neg_sqrt_3_div_2_l385_385827

theorem cos_210_eq_neg_sqrt_3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by
  -- leave proof as sorry
  sorry

end cos_210_eq_neg_sqrt_3_div_2_l385_385827


namespace exists_f_l385_385939

open Real

noncomputable def f (C : ℝ) : ℝ → ℝ :=
  λ x, x - cos x + C

theorem exists_f (C : ℝ) : ∃ f : ℝ → ℝ, (∀ x, has_deriv_at f (1 + sin x) x) ∧ (∀ x, f x = x - cos x + C) :=
begin
  use f C,
  split,
  { intro x,
    convert has_deriv_at.add (has_deriv_at_id x) (has_deriv_at.neg (has_deriv_at_cos x)),
    { rw add_neg_eq_sub,
      refl },
    { rw sub_neg_eq_add },
    { rw has_deriv_at_cos,
      refl }
  },
  { intro x,
    refl }
end

end exists_f_l385_385939


namespace irrationals_count_correct_l385_385803

noncomputable def countIrrationals (l : List ℝ) : Nat :=
  l.countp (λ x, ¬ ∃ (a b : ℚ), b ≠ 0 ∧ x = a / b)

theorem irrationals_count_correct :
  countIrrationals [22 / 7, 3.14159, real.sqrt 7, -8, real.cbrt 2, 0.6, 0, real.sqrt 36, real.pi / 3] = 3 := 
by 
  sorry

end irrationals_count_correct_l385_385803


namespace tan_y_in_terms_of_a_and_b_l385_385986

variable {a b : ℝ}
variable (y : ℝ)

theorem tan_y_in_terms_of_a_and_b (h1 : a > b) (h2 : b > 0) (h3 : 0 < y ∧ y < π / 2)
  (h4 : sin y = 3 * a * b / Real.sqrt (a^6 + 3 * a^3 * b^3 + b^6)) :
  tan y = 3 * a * b / Real.sqrt (a^6 + 3 * a^3 * b^3 + b^6 - 9 * a^2 * b^2) :=
sorry

end tan_y_in_terms_of_a_and_b_l385_385986


namespace increase_statistics_if_add_75_l385_385811

theorem increase_statistics_if_add_75 :
  let scores := [40, 45, 50, 50, 55, 55, 60, 60, 64, 68, 68, 70, 72, 75, 75]
  let new_score := 75
  let new_scores := scores ++ [new_score]
  median(new_scores) > median(scores) ∧
  mean(new_scores) > mean(scores) ∧
  mode(new_scores) > mode(scores) := by
    sorry

end increase_statistics_if_add_75_l385_385811


namespace chores_per_week_l385_385312

theorem chores_per_week :
  ∀ (cookie_per_chore : ℕ) 
    (total_money : ℕ) 
    (cost_per_pack : ℕ) 
    (cookies_per_pack : ℕ) 
    (weeks : ℕ)
    (chores_per_week : ℕ),
  cookie_per_chore = 3 →
  total_money = 15 →
  cost_per_pack = 3 →
  cookies_per_pack = 24 →
  weeks = 10 →
  chores_per_week = (total_money / cost_per_pack * cookies_per_pack / weeks) / cookie_per_chore →
  chores_per_week = 4 :=
by
  intros cookie_per_chore total_money cost_per_pack cookies_per_pack weeks chores_per_week
  intros h1 h2 h3 h4 h5 h6
  sorry

end chores_per_week_l385_385312


namespace perimeter_of_equilateral_triangle_7cm_l385_385122

/-
Theorem: Given a figure with 3 vertices and the distance between any two vertices is 7 cm,
the perimeter of the figure is 21 cm.
-/

theorem perimeter_of_equilateral_triangle_7cm : 
  ∀ (v1 v2 v3 : ℝ) (d : ℝ), 
    d = 7 → 
    dist v1 v2 = d → dist v2 v3 = d → dist v3 v1 = d → 
    3 * d = 21 :=
by 
  intros v1 v2 v3 d h1 h2 h3 h4
  rw [h1]
  simp
  sorry

end perimeter_of_equilateral_triangle_7cm_l385_385122


namespace min_value_n_l385_385216

theorem min_value_n (n : ℕ) (h1 : 4 ∣ 60 * n) (h2 : 8 ∣ 60 * n) : n = 1 := 
  sorry

end min_value_n_l385_385216


namespace complement_union_M_N_l385_385188

def U := {x : ℕ | 1 ≤ x ∧ x < 9}
def M := {1, 3, 5, 7}
def N := {5, 6, 7}
def union_M_N := M ∪ N
def complement (S U : Set ℕ) := {x ∈ U | x ∉ S}

theorem complement_union_M_N : complement union_M_N U = {2, 4, 8} :=
  sorry

end complement_union_M_N_l385_385188


namespace ray_two_digit_number_l385_385004

theorem ray_two_digit_number (a b n : ℕ) (ha : 1 ≤ a) (hb : b ≤ 9) (hn : n = 10 * a + b) (h1 : n = 4 * (a + b) + 3) (h2 : n + 18 = 10 * b + a) : n = 35 := by
  sorry

end ray_two_digit_number_l385_385004


namespace smallest_prime_20_less_than_square_l385_385374

open Nat

theorem smallest_prime_20_less_than_square : ∃ (p : ℕ), Prime p ∧ (∃ (n : ℕ), p = n^2 - 20) ∧ p = 5 := by
  sorry

end smallest_prime_20_less_than_square_l385_385374


namespace remainder_is_23_l385_385998

def number_remainder (n : ℤ) : ℤ :=
  n % 36

theorem remainder_is_23 (n : ℤ) (h1 : n % 4 = 3) (h2 : n % 9 = 5) :
  number_remainder n = 23 :=
by
  sorry

end remainder_is_23_l385_385998


namespace find_students_in_group_A_l385_385709

-- Definitions based on conditions
def students_in_group_B : ℕ := 80
def forgot_homework_group_A (A : ℕ) : ℕ := Nat.floor (0.20 * A)
def forgot_homework_group_B : ℕ := Nat.floor (0.15 * students_in_group_B)
def total_students (A : ℕ) : ℕ := A + students_in_group_B
def total_forgot_homework (A : ℕ) : ℕ := forgot_homework_group_A A + forgot_homework_group_B
def forgot_homework_percentage (A : ℕ) : ℕ := Nat.floor (0.16 * total_students A)

-- Theorem statement
theorem find_students_in_group_A (A : ℕ) :
  total_forgot_homework A = forgot_homework_percentage A → A = 20 :=
by
  sorry

end find_students_in_group_A_l385_385709


namespace function_always_passes_through_point_l385_385136

theorem function_always_passes_through_point (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
    ∃ y, y = a^(-4 + 4) + 3 ∧ (-4, y) = (-4, 4) :=
by
  use 4
  sorry

end function_always_passes_through_point_l385_385136


namespace find_derivative_at_minus_one_third_l385_385517

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * f' (-1 / 3) * x

theorem find_derivative_at_minus_one_third :
  let f' (x : ℝ) : ℝ := deriv f x in
  f' (-1 / 3) = 2 / 3 :=
by
  sorry

end find_derivative_at_minus_one_third_l385_385517


namespace fraction_of_employees_with_pagers_l385_385455

theorem fraction_of_employees_with_pagers
  (E : ℕ) -- Total number of employees
  (cellphones : ℕ := (2/3 : ℚ) * E) -- Number of employees with cell phones
  (neither : ℕ := (1/3 : ℚ) * E) -- Number of employees with neither
  (both : ℕ := 0.4 * E) -- Number of employees with both
  (pagers : ℕ) :
  (pagers/E : ℚ) = 0.8 := 
sorry

end fraction_of_employees_with_pagers_l385_385455


namespace balance_balls_l385_385307

theorem balance_balls (R O G B : ℝ) (h₁ : 4 * R = 8 * G) (h₂ : 3 * O = 6 * G) (h₃ : 8 * G = 6 * B) :
  3 * R + 2 * O + 4 * B = (46 / 3) * G :=
by
  -- Using the given conditions to derive intermediate results (included in the detailed proof, not part of the statement)
  sorry

end balance_balls_l385_385307


namespace complement_A_B_eq_singleton_three_l385_385528

open Set

variable (A : Set ℕ) (B : Set ℕ) (a : ℕ)

theorem complement_A_B_eq_singleton_three (hA : A = {2, 3, 4})
    (hB : B = {a + 2, a}) (h_inter : A ∩ B = B) : A \ B = {3} :=
  sorry

end complement_A_B_eq_singleton_three_l385_385528


namespace min_value_parabola_MF_MP_equals_7_l385_385160

-- Define the conditions
def point (x y : ℝ) := (x, y)
def focus : (ℝ × ℝ) := point 0 4
def parabola (p : ℝ) (x y : ℝ) := x^2 = 2 * p * y
def point_on_parabola (p : ℝ) (x y : ℝ) := parabola p x y
def directrix (p : ℝ) : ℝ := -p / 2
def distance (a b : ℝ × ℝ) : ℝ := real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- Define the problem
def min_MF_MP (p : ℝ) (P focus : ℝ × ℝ) (x y : ℝ) (h : parabola p x y) : ℝ :=
  let M := point x y in
  let F := focus in
  distance M F + distance M P

-- State the theorem
theorem min_value_parabola_MF_MP_equals_7 :  
  ∃ M : point ℝ ℝ, ∃ p : ℝ, p > 0 ∧ focus = point 0 4 ∧ point_on_parabola p M.1 M.2 ∧
  P = point 2 3 ∧ min_MF_MP p P focus M.1 M.2 (point_on_parabola p M.1 M.2) = 7 :=
sorry

end min_value_parabola_MF_MP_equals_7_l385_385160


namespace smallest_n_with_314_l385_385907

noncomputable def contains_314 (n : ℕ) (m : ℕ) : Prop :=
  let frac := (m : ℚ) / (n : ℚ) in
  let dec_str := frac.to_decimal_string in
  "314".isIn dec_str

theorem smallest_n_with_314 :
  ∃ m n : ℕ,
    Nat.coprime m n ∧
    m < n ∧
    contains_314 n m ∧
    ∀ n' (h' : n' < n), ¬ ∃ m',
      Nat.coprime m' n' ∧
      m' < n' ∧
      contains_314 n' m' :=
begin
  sorry
end

end smallest_n_with_314_l385_385907


namespace find_number_of_roses_l385_385773

theorem find_number_of_roses : ∃ a : ℕ, 300 ≤ a ∧ a ≤ 400 ∧ a % 21 = 13 ∧ a % 15 = 7 :=
by
  -- Existential quantifier for the number 'a'
  use 307
  
  -- Proof of the conditions for 'a'
  split
  -- Proof that 300 ≤ 307 ∧ 307 ≤ 400
  exact ⟨by linarith, by linarith⟩
  split
  -- Proof that 307 % 21 = 13
  exact by norm_num
  -- Proof that 307 % 15 = 7 (because -8 % 15 = 7)
  exact by norm_num

end find_number_of_roses_l385_385773


namespace line_PQ_fixed_point_l385_385538

theorem line_PQ_fixed_point :
  ∃ (P Q : ℝ × ℝ), (A : ℝ × ℝ) = (1, 2) →
  (on_parabola_C : P.1 = (P.2)^2 / 4 ∧ Q.1 = (Q.2)^2 / 4) →
  (perpendicular : (P.1 - 1, P.2 - 2) • (Q.1 - 1, Q.2 - 2) = 0) →
  ∃ (t : ℝ), (∀ (s : ℝ), x = t * y + s) →
  (line_passes_fixed : ∃ (s : ℝ), s = 2 * t + 5 → P = Q → s - 3 = 2 * (t + 1) 
                            ∨ s - 3 = -2 * (t + 1)) ∧ x = 5  ∧ y = -2.
sorry

end line_PQ_fixed_point_l385_385538


namespace find_total_roses_l385_385769

open Nat

theorem find_total_roses 
  (a : ℕ)
  (h1 : 300 ≤ a)
  (h2 : a ≤ 400)
  (h3 : a % 21 = 13)
  (h4 : a % 15 = 7) : 
  a = 307 := 
sorry

end find_total_roses_l385_385769


namespace triangle_XYZ_XY_length_l385_385248

theorem triangle_XYZ_XY_length (X Y Z : Type) [Realized X Y Z] (XY XZ YZ : ℝ) 
(h1 : ∠XYZ = π / 2) (h2 : YZ = 13) (h3 : tan Z = 3 * cos Y) :
  XY = 2 * sqrt 338 / 3 :=
sorry

end triangle_XYZ_XY_length_l385_385248


namespace train_b_speed_l385_385364

/-- Given:
    1. Length of train A: 150 m
    2. Length of train B: 150 m
    3. Speed of train A: 54 km/hr
    4. Time taken to cross train B: 12 seconds
    Prove: The speed of train B is 36 km/hr
-/
theorem train_b_speed (l_A l_B : ℕ) (V_A : ℕ) (t : ℕ) (h1 : l_A = 150) (h2 : l_B = 150) (h3 : V_A = 54) (h4 : t = 12) :
  ∃ V_B : ℕ, V_B = 36 := sorry

end train_b_speed_l385_385364


namespace minimum_t_l385_385596

-- Definitions:
variables {A B C E F : Type}
variables [OrderedField A] [OrderedField B] [OrderedField C] [OrderedField E] [OrderedField F]

-- Defining the triangle and midpoints
def is_midpoint (P : Type) (X Y Z : Type) {a b : A} (Mid : X) := 
  Mid = (a + b) / 2

-- Define the condition relating the sides of the triangle
def side_relation (AB AC : A) : Prop :=
  3 * AB = 2 * AC

-- Define main theorem statement
theorem minimum_t {AB AC BE CF : A} (h1 : side_relation AB AC) 
  (h2 : BE < (t : A) * CF) : t ≥ (7 : A) / 8 :=
sorry

end minimum_t_l385_385596


namespace not_integer_20_diff_l385_385077

theorem not_integer_20_diff (a b : ℝ) (hne : a ≠ b) 
  (no_roots1 : ∀ x, x^2 + 20 * a * x + 10 * b ≠ 0) 
  (no_roots2 : ∀ x, x^2 + 20 * b * x + 10 * a ≠ 0) : 
  ¬ (∃ k : ℤ, 20 * (b - a) = k) :=
by
  sorry

end not_integer_20_diff_l385_385077


namespace quadratic_solution_l385_385687

theorem quadratic_solution
  (a c : ℝ) (h : a ≠ 0) (h_passes_through : ∃ b, b = c - 9 * a) :
  ∀ (x : ℝ), (ax^2 - 2 * a * x + c = 0) ↔ (x = -1) ∨ (x = 3) :=
by
  sorry

end quadratic_solution_l385_385687


namespace emily_spent_20_dollars_l385_385486

/-- Let X be the amount Emily spent on Friday. --/
variables (X : ℝ)

/-- Emily spent twice the amount on Saturday. --/
def saturday_spent := 2 * X

/-- Emily spent three times the amount on Sunday. --/
def sunday_spent := 3 * X

/-- The total amount spent over the three days is $120. --/
axiom total_spent : X + saturday_spent X + sunday_spent X = 120

/-- Prove that X = 20. --/
theorem emily_spent_20_dollars : X = 20 :=
sorry

end emily_spent_20_dollars_l385_385486


namespace areas_equal_l385_385452

universe u
noncomputable theory

open_locale classical

-- Definitions for points and properties
variables {Point : Type u} [AffineSpace Point ℝ] -- Setting up a type for points in an affine space over real numbers
variables {A B C D E F K M : Point} -- Declaring the points used in the problem
variables (midpoint : Point → Point → Point)
variables (parallel : Point → Point → Point → Point → Prop)
variables (line_segment : Point → Point → Set Point)
variables (area : Set Point → ℝ)

-- Conditions from the problem
variable (is_square : Set Point → Prop)
variable [is_square (∅ : Set Point)] -- Placeholder for the actual square A, B, C, D points
variable (hE : E = midpoint A B)
variable (hF : F = midpoint A D)
variable (hCE_BF_intersect_at_K : ∃ (K : Point), K ∈ (line_segment C E) ∩ (line_segment B F))
variable (hM_on_EC : M ∈ (line_segment E C))
variable (BM_parallel_KD : parallel B M K D)

-- Required proof statement
theorem areas_equal 
  (htriangle_KFD : Set Point) (htrapezoid_KBMD : Set Point)
  (h_area_KFD : area htriangle_KFD) 
  (h_area_KBMD : area htrapezoid_KBMD) :
  area htriangle_KFD = area htrapezoid_KBMD := sorry -- Proof omitted

end areas_equal_l385_385452


namespace ginger_total_water_l385_385914

def hours_worked : Nat := 8
def cups_per_bottle : Nat := 2
def bottles_drank_per_hour : Nat := 1
def bottles_for_plants : Nat := 5

theorem ginger_total_water : 
  (hours_worked * cups_per_bottle * bottles_drank_per_hour) + (bottles_for_plants * cups_per_bottle) = 26 :=
by
  sorry

end ginger_total_water_l385_385914


namespace compute_value_l385_385035

theorem compute_value : (7^2 - 6^2)^3 = 2197 := by
  sorry

end compute_value_l385_385035


namespace length_chord_AB_standard_equation_circle_M_l385_385920

-- Define the point P_0 and circle centered at the origin with radius 2√2
def P₀ : ℝ × ℝ := (-1, 2)
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 8

-- Define the chord AB and the angle of inclination α
def α : ℝ := 135

-- Define the conditions for P_0, the inclination α, and being a bisector
def P₀_in_circle : Prop := circle_eq (P₀.1) (P₀.2)
def is_bisector (a b : ℝ) : Prop := true -- Placeholder, as the full geometric definition is complex

-- Problem 1: Length of AB when α = 135°
theorem length_chord_AB : 
  α = 135 → ∃ l : ℝ, l = sqrt 30 := 
by sorry

-- Define point C and conditions for circle M passing through C and tangent to AB at P0
def C : ℝ × ℝ := (3, 0)
def circle_M (x y : ℝ) : Prop := (x - 1/4)^2 + (y + 1/2)^2 = 125 / 16

-- Problem 2: Standard equation of circle M
theorem standard_equation_circle_M :
  ∃ x y : ℝ, circle_M x y ∧ circle_eq x y ∧ is_bisector x y :=
by sorry

end length_chord_AB_standard_equation_circle_M_l385_385920


namespace rowing_time_to_place_and_back_l385_385781

def rowing_time
  (V_p : ℝ) -- Speed of the person in still water
  (V_c : ℝ) -- Speed of the current
  (D : ℝ)  -- Distance to the place
  : ℝ :=
  let V_downstream := V_p + V_c
  let V_upstream := V_p - V_c
  let T_downstream := D / V_downstream
  let T_upstream := D / V_upstream
  T_downstream + T_upstream

theorem rowing_time_to_place_and_back :
  rowing_time 10 2 48 = 10 :=
by
  sorry

end rowing_time_to_place_and_back_l385_385781


namespace right_triangle_exists_l385_385383

theorem right_triangle_exists (a b c : ℝ) (h1 : a = 2 ∧ b = 3 ∧ c = 4 ∨
                                       a = sqrt 2 ∧ b = sqrt 3 ∧ c = 5 ∨
                                       a = 2 ∧ b = 2 ∧ c = 2 * sqrt 3 ∨
                                       a = 1 ∧ b = 2 ∧ c = sqrt 3) :
                                       (∃ a b c, a = 1 ∧ b = 2 ∧ c = sqrt 3 ∧ a^2 + b^2 = c^2) :=
by
  sorry

end right_triangle_exists_l385_385383


namespace morgan_change_l385_385627

-- Define the costs of the items and the amount paid
def hamburger_cost : ℕ := 4
def onion_rings_cost : ℕ := 2
def smoothie_cost : ℕ := 3
def amount_paid : ℕ := 20

-- Define total cost
def total_cost := hamburger_cost + onion_rings_cost + smoothie_cost

-- Define the change received
def change_received := amount_paid - total_cost

-- Statement of the problem in Lean 4
theorem morgan_change : change_received = 11 := by
  -- include proof steps here
  sorry

end morgan_change_l385_385627


namespace correct_statement_l385_385384

def StatementA : Prop := ¬ random_event ("The sun rises in the east, and the moon sets in the west")

def StatementB : Prop := ¬ certain_event (buying_lottery_tickets 100 0.01)

def StatementC : Prop := ¬ correct_geometric_principle (adjacent_interior_angles_are_complementary)

def StatementD : Prop := possible_event (coin_flip_heads_up 100 50)

theorem correct_statement : StatementD :=
by 
  sorry


end correct_statement_l385_385384


namespace spherical_coordinates_change_l385_385030

def spherical_to_rectangular (ρ θ ϕ : ℝ) : (ℝ × ℝ × ℝ) :=
  (ρ * sin ϕ * cos θ, ρ * sin ϕ * sin θ, ρ * cos ϕ)

def convert_coordinates (x y z : ℝ) := (x, y, -z)

theorem spherical_coordinates_change :
  let initial_spherical := (5, 5 * π / 6, π / 3)
  let (x, y, z) := spherical_to_rectangular 5 (5 * π / 6) (π / 3)
  let new_rectangular := convert_coordinates x y z
  let new_spherical := (5, 5 * π / 6, 2 * π / 3)
  spherical_to_rectangular new_spherical.1 new_spherical.2 new_spherical.3 = new_rectangular :=
by {
  sorry
}

end spherical_coordinates_change_l385_385030


namespace relationship_among_abc_l385_385518

noncomputable def f : ℝ → ℝ := λ x, Real.exp (-Real.abs x)

def a := f (Real.log 3 / Real.log 0.5)
def b := f (Real.log 5 / Real.log 2)
def c := f 0

theorem relationship_among_abc : b < a ∧ a < c := 
by
  sorry

end relationship_among_abc_l385_385518


namespace factorize_x_squared_minus_one_l385_385095

theorem factorize_x_squared_minus_one (x : Real) : (x^2 - 1) = (x + 1) * (x - 1) :=
sorry

end factorize_x_squared_minus_one_l385_385095


namespace length_of_PS_l385_385594

theorem length_of_PS 
  (P R S Q : Type) 
  [has_add P] [has_le P] [has_sub P]
  (PR PQ QR : ℝ)
  (PR_length : PR = 72)
  (PQ_length : PQ = 32)
  (QR_length : QR = 64)
  (angle_bisector_S : ∀ (PS SR : ℝ), PS / SR = PQ / QR): 
  ∃ (PS : ℝ), PS = 24 := by
  sorry

end length_of_PS_l385_385594


namespace train_passes_jogger_l385_385775

noncomputable def speed_of_jogger_kmph := 9
noncomputable def speed_of_train_kmph := 45
noncomputable def jogger_lead_m := 270
noncomputable def train_length_m := 120

noncomputable def speed_of_jogger_mps := speed_of_jogger_kmph * (1000 / 3600) -- converting km/hr to m/s
noncomputable def speed_of_train_mps := speed_of_train_kmph * (1000 / 3600) -- converting km/hr to m/s
noncomputable def relative_speed_mps := speed_of_train_mps - speed_of_jogger_mps
noncomputable def total_distance_m := jogger_lead_m + train_length_m
noncomputable def time_to_pass_jogger := total_distance_m / relative_speed_mps

theorem train_passes_jogger : time_to_pass_jogger = 39 :=
  by
    -- Proof steps would be provided here
    sorry

end train_passes_jogger_l385_385775


namespace find_other_number_l385_385394

theorem find_other_number (B : ℕ) (HCF : ℕ) (LCM : ℕ) (A : ℕ) 
  (h1 : A = 24) 
  (h2 : HCF = 16) 
  (h3 : LCM = 312) 
  (h4 : HCF * LCM = A * B) :
  B = 208 :=
by
  sorry

end find_other_number_l385_385394


namespace lines_concurrent_l385_385719

   -- Definitions of congruence, equilateral, and parallelism
   structure Triangle (α : Type) [OrderedRing α] :=
     (A B C : α × α)
     (is_equilateral : ∀ (X Y Z : α × α), (dist X Y = dist Y Z) ∧ (dist Y Z = dist Z X) ∧ (dist Z X = dist X Y))
     (is_congruent : Triangle α → Prop)
     (is_parallel : (α × α) → (α × α) → Prop)

   -- Instances for specific triangles ABC and PQR
   variables {α : Type} [OrderedRing α]
   def ABC : Triangle α := {
     A := (0, 0),
     B := (1, 0),
     C := (1 / 2, sqrt(3) / 2),
     is_equilateral := sorry,  -- Equilateral property proof
     is_congruent := sorry,    -- Congruence property proof
     is_parallel := sorry      -- Parallel property proof
   }

   def PQR : Triangle α := {
     A := (0, sqrt(3) / 2),
     B := (1, sqrt(3) / 2),
     C := (1 / 2, 0),
     is_equilateral := sorry,  
     is_congruent := sorry,
     is_parallel := sorry
   }

   -- Midpoint definition
   def midpoint (p q : α × α) : α × α :=
     ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

   -- Problem statement in Lean
   theorem lines_concurrent (t1 t2 : Triangle α) (H : α × α)
     (hexagon : list (α × α))
     (A1 A2 A3 A4 A5 A6 : α × α)
     (h₁ : hexagon = [A1, A2, A3, A4, A5, A6])
     (h₂ : ∃ G, G = midpoint t1.A t2.A
              ∧ G = midpoint t1.B t2.B
              ∧ G = midpoint t1.C t2.C)
     :
     -- Conclusion: A1A4, A2A5, A3A6 intersect at a common point H
     ∃ (H : α × α), collinear A1 A4 H ∧ collinear A2 A5 H ∧ collinear A3 A6 H :=
   sorry -- Proof of concurrency
   
end lines_concurrent_l385_385719


namespace six_dice_not_same_probability_l385_385721

theorem six_dice_not_same_probability :
  let total_outcomes := 6^6
  let all_same := 6
  let probability_all_same := all_same / total_outcomes
  let probability_not_all_same := 1 - probability_all_same
  probability_not_all_same = 7775 / 7776 :=
by
  sorry

end six_dice_not_same_probability_l385_385721


namespace min_distance_between_bees_l385_385401

namespace BeeFlight

theorem min_distance_between_bees
  (v1 v2 a b x : ℝ) :
  let y := (λ x, real.sqrt ((x * v1 - a)^2 + (x * v2 - b)^2))
  in
  ∀ x, ∃ x₀, y x₀ = real.sqrt ((x₀ * v1 - a)^2 + (x₀ * v2 - b)^2) ∧
             x₀ = (a * v1 + b * v2) / (v1^2 + v2^2) ∧
             (∀ x, y x ≥ y x₀) :=
by
  sorry

end BeeFlight

end min_distance_between_bees_l385_385401


namespace tan_alpha_minus_pi_over_4_l385_385130

variable (α β : ℝ)

-- Given conditions
axiom h1 : Real.tan (α + β) = 2 / 5
axiom h2 : Real.tan β = 1 / 3

-- The goal to prove
theorem tan_alpha_minus_pi_over_4: 
  Real.tan (α - π / 4) = -8 / 9 := by
  sorry

end tan_alpha_minus_pi_over_4_l385_385130


namespace sum_of_reciprocal_of_S_seq_l385_385523

variables {a b n : ℕ}
variables {a_seq : ℕ → ℕ} {b_seq : ℕ → ℕ} {S_seq : ℕ → ℕ}

-- Define the arithmetic and geometric sequences
def a_seq (n : ℕ) : ℕ := 2 * n + 1
def b_seq (n : ℕ) : ℕ := 2 ^ (n - 1)
def S_seq (n : ℕ) : ℕ := n * (n + 2)

-- Conditions
axiom a1 : a_seq 1 = 3
axiom b1 : b_seq 1 = 1
axiom cond_S2 : b_seq 2 * S_seq 2 = 64
axiom cond_S3 : b_seq 3 * S_seq 3 = 960

-- Proof goal
theorem sum_of_reciprocal_of_S_seq (n : ℕ) : 
  (range n).map (λ k, 1 / S_seq (k + 1)).sum = (n * (3 * n + 5)) / (4 * (n + 1) * (n + 2)) :=
sorry

end sum_of_reciprocal_of_S_seq_l385_385523


namespace instantaneous_velocity_at_3_l385_385047

def s (t : ℝ) : ℝ := 1 - t + t^2

theorem instantaneous_velocity_at_3 :
  (derivative s 3) = 5 :=
sorry

end instantaneous_velocity_at_3_l385_385047


namespace staffing_ways_l385_385057

theorem staffing_ways (total_resumes : ℕ) (unsatisfactory_percentage : ℚ) (job_openings : ℕ)
  (resumes : ℕ := 30) (satisfactory_percentage : ℚ := 0.6) (openings : ℕ := 6)
  (expected_result : ℕ := 7257984) :
  total_resumes = resumes → 
  unsatisfactory_percentage = 1 - satisfactory_percentage →
  job_openings = openings →
  (let satisfactory_candidated := total_resumes * satisfactory_percentage.to_rat in
  let number_of_ways := satisfactory_candidated.to_nat *? (satisfactory_candidated.to_nat - 1) *?
                        (satisfactory_candidated.to_nat - 2) *?
                        (satisfactory_candidated.to_nat - 3) *?
                        (satisfactory_candidated.to_nat - 4) *?
                        (satisfactory_candidated.to_nat - 5))
  number_of_ways = expected_result) :=
begin
  intros hresumes hpercentage hopenings,
  simp only [hresumes, hpercentage, hopenings],
  sorry
end

end staffing_ways_l385_385057


namespace value_of_f_2x_l385_385988

theorem value_of_f_2x (f : ℝ → ℝ) (h : ∀ x : ℝ, f(x) = 3) (x : ℝ) : f(2*x) = 3 := by
  sorry

end value_of_f_2x_l385_385988


namespace cosine_210_l385_385846

noncomputable def Q := rotate 210 (1, 0)
noncomputable def E := foot Q (1, 0) (-1, 0)

theorem cosine_210 : cos 210 = - (sqrt 3/2) :=
sorry

end cosine_210_l385_385846


namespace orvin_balloons_l385_385311

def regular_price : ℕ := 2
def total_money_initial := 42 * regular_price
def pair_cost := regular_price + (regular_price / 2)
def pairs := total_money_initial / pair_cost
def balloons_from_sale := pairs * 2

def extra_money : ℕ := 18
def price_per_additional_balloon := 2 * regular_price
def additional_balloons := extra_money / price_per_additional_balloon
def greatest_number_of_balloons := balloons_from_sale + additional_balloons

theorem orvin_balloons (pairs balloons_from_sale additional_balloons greatest_number_of_balloons : ℕ) :
  pairs * 2 = 56 →
  additional_balloons = 4 →
  greatest_number_of_balloons = 60 :=
by
  sorry

end orvin_balloons_l385_385311


namespace correct_operation_l385_385382

variable (a b m : ℕ)

theorem correct_operation :
  (3 * a^2 * 2 * a^2 ≠ 5 * a^2) ∧
  ((2 * a^2)^3 = 8 * a^6) ∧
  (m^6 / m^3 ≠ m^2) ∧
  ((a + b)^2 ≠ a^2 + b^2) →
  ((2 * a^2)^3 = 8 * a^6) :=
by
  intros
  sorry

end correct_operation_l385_385382


namespace round_to_nearest_tenth_l385_385648

theorem round_to_nearest_tenth : 
  let x := 36.89753 
  let tenth_place := 8
  let hundredth_place := 9
  (hundredth_place > 5) → (Float.round (10 * x) / 10 = 36.9) := 
by
  intros x tenth_place hundredth_place h
  sorry

end round_to_nearest_tenth_l385_385648


namespace polynomial_remainder_l385_385614

def h (x : ℝ) : ℝ := x^3 + x^2 + x + 1

theorem polynomial_remainder (x : ℝ) :
  let hx := h x in
  let hx8 := h (x^8) in
  ∃ r, hx8 = hx * (some_polynomial x) + r ∧ r = 4 :=
sorry

end polynomial_remainder_l385_385614


namespace loss_is_15pc_when_sold_for_187_l385_385428

noncomputable def CP : ℝ := 220  -- From solution CP = 264 / 1.20

-- Define the conditions given in the problem
def sold_price_gain_20pc : ℝ := 264
def sold_price_loss : ℝ := 187

-- Define the calculation of loss percentage
def loss_percentage (CP SP : ℝ) : ℝ := ((CP - SP) / CP) * 100

-- Statement to prove the loss percentage is 15% when sold for Rs. 187
theorem loss_is_15pc_when_sold_for_187 : loss_percentage CP sold_price_loss = 15 :=
by
  -- Sorry is used to skip the proof
  sorry

end loss_is_15pc_when_sold_for_187_l385_385428


namespace h_two_l385_385573

noncomputable def h (x : ℝ) : ℝ := 4 * (x + 1) + 2

theorem h_two : h 2 = 14 :=
by
  -- We start with the given condition
  have H : ∀ (x : ℝ), h (3 * x - 4) = 4 * x + 6 := sorry
  -- Let's find the intermediate steps
  have x_val : 3 * (2 : ℝ) - 4 = 2 := sorry
  have h_2_eq : h (2 : ℝ) = 4 * (2 : ℝ) + 6 := H (2 / 3)
  -- Combine these steps
  show h 2 = 14 from h_2_eq

end h_two_l385_385573


namespace square_side_length_l385_385794

theorem square_side_length
  (P : ℕ) (A : ℕ) (s : ℕ)
  (h1 : P = 44)
  (h2 : A = 121)
  (h3 : P = 4 * s)
  (h4 : A = s * s) :
  s = 11 :=
sorry

end square_side_length_l385_385794


namespace domain_of_f_l385_385368

noncomputable def f (x : ℝ) : ℝ := log 3 (log 2 (log 7 (log 6 x)))

theorem domain_of_f :
  ∀ x : ℝ, x > 6 ^ (7 ^ 8) → ∃ y, f(x) = y :=
by
  intro x
  intro hx
  -- Continue with the proof steps typically, but we use sorry to indicate skipping the proof.
  sorry

end domain_of_f_l385_385368


namespace find_inverse_of_25_l385_385932

-- Define the inverses and the modulo
def inverse_mod (a m i : ℤ) : Prop :=
  (a * i) % m = 1

-- The given condition in the problem
def condition (m : ℤ) : Prop :=
  inverse_mod 5 m 39

-- The theorem we want to prove
theorem find_inverse_of_25 (m : ℤ) (h : condition m) : inverse_mod 25 m 8 :=
by
  sorry

end find_inverse_of_25_l385_385932


namespace nine_digit_permutations_divisible_by_2_l385_385203

theorem nine_digit_permutations_divisible_by_2 :
  let digits := [2, 3, 1, 1, 5, 7, 1, 5, 2] in
  (∃ n : ℕ, n = (1 * (8.factorial / (3.factorial * 2.factorial * 1.factorial * 1.factorial))) 
              ∧ digits.permutations.count (λ xs, xs.last = some 2) = n 
              ∧ with_last_digit (l : List ℕ) : Prop := l.last = some 2) := 3360 := 
sorry

end nine_digit_permutations_divisible_by_2_l385_385203


namespace exists_set_M_inter_plane_finite_nonempty_l385_385078

open Set

/-- Define the set M in R^3 as {(t^5, t^3, t) | t ∈ ℝ} -/
def M : Set (ℝ × ℝ × ℝ) := { p | ∃ t : ℝ, p = (t^5, t^3, t) }

/-- For every plane λ in ℝ^3 given by the equation ax + by + cz + d = 0, 
    the intersection M ∩ λ is finite and nonempty. -/
theorem exists_set_M_inter_plane_finite_nonempty :
  ∃ M : Set (ℝ × ℝ × ℝ), (∀ a b c d : ℝ, a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 →
    ∃ t : ℝ, ∃ p : ℝ × ℝ × ℝ, p = (t^5, t^3, t) ∧ a * (t^5) + b * (t^3) + c * t + d = 0
    ∧ (λ u, a * u.1 + b * u.2 + c * u.3 + d = 0) '' M ⊆ M ∧
    Finite ((λ u, a * u.1 + b * u.2 + c * u.3 + d = 0) '' M)) :=
begin
  use M,
  intros a b c d habc,
  obtain ⟨t, h⟩ := exists_real_root_of_odd_degree_poly (λ x, a * x^5 + b * x^3 + c * x + d),
  refine ⟨t, ⟨(t^5, t^3, t), rfl⟩, _, _, _⟩,
  { exact h, },
  { sorry, },
  { sorry, },
end

end exists_set_M_inter_plane_finite_nonempty_l385_385078


namespace solid_is_cylinder_l385_385701

def solid_views (v1 v2 v3 : String) : Prop := 
  -- This definition makes a placeholder for the views of the solid.
  sorry

def is_cylinder (s : String) : Prop := 
  s = "Cylinder"

theorem solid_is_cylinder (v1 v2 v3 : String) (h : solid_views v1 v2 v3) :
  ∃ s : String, is_cylinder s :=
sorry

end solid_is_cylinder_l385_385701


namespace percentage_defective_units_shipped_for_sale_l385_385241

theorem percentage_defective_units_shipped_for_sale (D : ℝ) (h1 : 0.08 * D) (h2 : 0.0032 * D) :
  (0.0032 / 0.08) * 100 = 4 := 
by
  sorry

end percentage_defective_units_shipped_for_sale_l385_385241


namespace eccentricity_ellipse_l385_385415

theorem eccentricity_ellipse (a b c : ℝ) (h : a > b > 0)
  (h1 : c = sqrt (a^2 - b^2))
  (collinear : ∃ k : ℝ, (k * 3, k * -1) = (-(2 * c * a^2) / (a^2 + b^2), (2 * c * b^2) / (a^2 + b^2))) :
  sqrt (1 - b^2 / a^2) = sqrt (6) / 3 :=
by
  sorry

end eccentricity_ellipse_l385_385415


namespace correct_transformation_l385_385213

theorem correct_transformation (x y : ℤ) (h : x = y) : x - 2 = y - 2 :=
by
  sorry

end correct_transformation_l385_385213


namespace answer_A_answer_D_l385_385339

def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

axiom ω_pos : ∀ (ω : ℝ), ω > 0
axiom T_bound : ∀ (T : ℝ), 2 * Real.pi / 3 < T ∧ T < 2 * Real.pi
axiom sym_line : ∀ (x : ℝ), x = Real.pi / 8

theorem answer_A (ω : ℝ) (T : ℝ) (h_limit : 2 * Real.pi / 3 < T ∧ T < 2 * Real.pi) (h_sym : x = Real.pi / 8) :
  ∃ ω, ω = 2 :=
sorry

theorem answer_D (x : ℝ) :
  ∀ (x : ℝ), ∃ k : ℤ, f 2 x = Math.sqrt 2 * x + Math.sqrt 2 / 2 := 
sorry

end answer_A_answer_D_l385_385339


namespace problem1_problem2_problem3_problem4_l385_385013

-- Proof Problem 1
theorem problem1 (a x : ℝ) (p : |x - a| < 4) (q : - x^2 + 5 * x - 6 > 0) :
  -1 ≤ a ∧ a ≤ 6 := sorry

-- Proof Problem 2
theorem problem2 (a : ℝ) (h : ∃ x₀ : ℝ, 2 ^ x₀ - 2 ≤ a^2 - 3 * a = false) :
  1 ≤ a ∧ a ≤ 2 := sorry

-- Proof Problem 3
theorem problem3 (a : ℝ) (h : ¬ ∀ x : ℝ, a * x^2 - 2 * a * x + 5 > 0) :
  (a < 0) ∨ (a ≥ 5) := sorry

-- Proof Problem 4
theorem problem4 :
  (A ∪ B = {②, ④, ⑤}) := sorry

end problem1_problem2_problem3_problem4_l385_385013


namespace factorize_x_squared_minus_one_l385_385091

theorem factorize_x_squared_minus_one (x : Real) : (x^2 - 1) = (x + 1) * (x - 1) :=
sorry

end factorize_x_squared_minus_one_l385_385091


namespace AM_less_than_BM_plus_CM_l385_385260

-- Define the isosceles triangle and circle with given properties
variables {O A B C M : Type*}
variables [Inhabited O] [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited M]

-- Conditions in the problem
variable {circle_O : ∀ {P Q R : Type*}, ∃ (O : Type*), inscribed_in_circle P Q R O}
variables (ABC_isosceles : isosceles_triangle A B C) (AB_eq_AC : AB = AC)
variable (M_midpoint_arc : midpoint_of_arc_not_containing_A M B C A)

-- The proof statement we need to prove
theorem AM_less_than_BM_plus_CM (A B C M : Type*) 
  [triangle_geom ABC_isosceles]
  [AB_eq_AC : AB = AC]
  [M_midpoint_arc M B C A] :
  AM < BM + CM := 
sorry

end AM_less_than_BM_plus_CM_l385_385260


namespace correct_heroes_count_l385_385309

-- Define types for inhabitants
inductive Inhabitant
| Hero
| Villain

open Inhabitant

-- Four inhabitants in a circular seating arrangement
constant A : Inhabitant
constant B : Inhabitant
constant C : Inhabitant
constant D : Inhabitant

-- Conditions based on the problem
axiom condition1 : (A = Hero ∧ B = Villain ∨ A = Villain ∧ B = Hero)
axiom condition2 : (B = Hero ∧ C = Villain ∨ B = Villain ∧ C = Hero)
axiom condition3 : (C = Hero ∧ D = Villain ∨ C = Villain ∧ D = Hero)
axiom condition4 : (D = Hero ∧ A = Villain ∨ D = Villain ∧ A = Hero)

-- The question we need to prove is the number of Heroes present
def number_of_heroes : ℕ :=
  match (A, B, C, D) with
  | (Hero, Hero, Hero, Hero) => 4
  | (Hero, Hero, Hero, Villain) => 3
  | (Hero, Hero, Villain, Hero) => 3
  | (Hero, Hero, Villain, Villain) => 2
  | (Hero, Villain, Hero, Hero) => 3
  | (Hero, Villain, Hero, Villain) => 2
  | (Hero, Villain, Villain, Hero) => 2
  | (Hero, Villain, Villain, Villain) => 1
  | (Villain, Hero, Hero, Hero) => 3
  | (Villain, Hero, Hero, Villain) => 2
  | (Villain, Hero, Villain, Hero) => 2
  | (Villain, Hero, Villain, Villain) => 1
  | (Villain, Villain, Hero, Hero) => 2
  | (Villain, Villain, Hero, Villain) => 1
  | (Villain, Villain, Villain, Hero) => 1
  | (Villain, Villain, Villain, Villain) => 0

theorem correct_heroes_count : number_of_heroes = 2 := 
by
  sorry

end correct_heroes_count_l385_385309


namespace height_difference_l385_385980

-- Define the heights of Eiffel Tower and Burj Khalifa as constants
def eiffelTowerHeight : ℕ := 324
def burjKhalifaHeight : ℕ := 830

-- Define the statement that needs to be proven
theorem height_difference : burjKhalifaHeight - eiffelTowerHeight = 506 := by
  sorry

end height_difference_l385_385980


namespace tangent_lines_equal_intercepts_on_circle_l385_385904

theorem tangent_lines_equal_intercepts_on_circle :
  ∀ (L : ℝ → AffineLine ℝ^2),
  ((∃ k : ℝ, L k = {p : ℝ^2 | p.2 = k * p.1}) ∨
  (∃ c : ℝ, L c = {p : ℝ^2 | p.1 + p.2 + c = 0})) →
  (∀ (x y : ℝ), (x - 2)^2 + y^2 = 2 → (L 0).dist ⟨2, 0⟩ = real.sqrt 2) →
  ((L 0).equation = ⟨1, -1, 0⟩ ∨
   (L 1).equation = ⟨1, 1, -4⟩ ∨
   (L 2).equation = ⟨1, 1, 0⟩) :=
by 
  sorry

end tangent_lines_equal_intercepts_on_circle_l385_385904


namespace sum_first_ten_terms_arithmetic_l385_385269

def arithmetic_sequence_sum (a₁ : ℤ) (n : ℕ) (d : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem sum_first_ten_terms_arithmetic (a₁ a₂ a₆ d : ℤ) 
  (h1 : a₁ = -2) 
  (h2 : a₂ + a₆ = 2) 
  (common_diff : d = 1) :
  arithmetic_sequence_sum a₁ 10 d = 25 :=
by
  rw [h1, common_diff]
  sorry

end sum_first_ten_terms_arithmetic_l385_385269


namespace min_typeB_trucks_needed_l385_385021

theorem min_typeB_trucks_needed
  (typeA_capacity : ℕ)
  (typeB_capacity : ℕ)
  (total_goods : ℕ)
  (num_typeA : ℕ)
  (required_typeB : ℕ) :
  typeA_capacity = 20 →
  typeB_capacity = 15 →
  total_goods = 300 →
  num_typeA = 7 →
  (15 * required_typeB + 7 * 20) ≥ 300 :=
begin
  intros hA hB hT hN,
  rw hA at *,
  rw hB at *,
  rw hT at *,
  rw hN at *,
  sorry
end

end min_typeB_trucks_needed_l385_385021


namespace find_b_for_system_l385_385580

theorem find_b_for_system (x y b : ℝ) (h1 : x = 1) (h2 : x + by = 0) (h3 : x + y = -1) : b = 1 / 2 := 
by 
  sorry

end find_b_for_system_l385_385580


namespace correct_statements_l385_385730

variable (model : Type) (r_A r_B R2 : ℝ) (totalProducts defectiveProducts : ℕ)
variable (selectionEvents : Set (Set ℕ))

-- Conditions
def narrow_band_better_fit : Prop :=
  ∀ (residuals : model → ℝ), (∀ d d' ∈ residuals, |d| < |d'|) → True

def stronger_correlation_abs : Prop :=
  r_A = 0.97 ∧ r_B = -0.99 ∧ |r_A| < |r_B|

def worse_fit_smaller_R2 : Prop :=
  R2 ≥ 0 ∧ R2 < 1 ∧ R2 ≤ 0.5

def probability_one_defective : Prop :=
  totalProducts = 10 ∧ defectiveProducts = 3 ∧
  (∀ (selection : Set ℕ), selection ∈ selectionEvents →
    selection.card = 2 → selection.inter {1, 2, 3}.card = 1 →
    (selectionEvents.prob selection = 7 / 15))

-- Theorem
theorem correct_statements : narrow_band_better_fit model ∧ 
                              worse_fit_smaller_R2 R2 ∧ 
                              probability_one_defective totalProducts defectiveProducts selectionEvents := by
  sorry

end correct_statements_l385_385730


namespace exists_fixed_circle_l385_385603

theorem exists_fixed_circle (A B C M P Q N : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space M] 
  [metric_space P] [metric_space Q] [metric_space N]
  (h1 : CA = CB) (h2 : ∠ACB = 120) (h3 : midpoint M A B) (h4 : variable_point P circumcircle_ABC) 
  (h5 : point_on_segment Q C P) (h6 : QP = 2 * QC)
  (h7 : line_through P ⊥ AB intersects MQ at N) :
  ∃ (fixed_circle : Type) [metric_space fixed_circle], circle_centered_at C fixed_circle radius CA ∧ N ∈ fixed_circle :=
sorry

end exists_fixed_circle_l385_385603


namespace sum_c_d_eq_24_l385_385817

theorem sum_c_d_eq_24 (c d : ℕ) (h_pos_c : c > 0) (h_pos_d : d > 1) (h_max_power : c^d < 500 ∧ ∀ ⦃x y : ℕ⦄, x^y < 500 → x^y ≤ c^d) : c + d = 24 :=
sorry

end sum_c_d_eq_24_l385_385817


namespace bumper_car_rides_l385_385050

-- Define the conditions
def rides_on_ferris_wheel : ℕ := 7
def cost_per_ride : ℕ := 5
def total_tickets : ℕ := 50

-- Formulate the statement to be proved
theorem bumper_car_rides : ∃ n : ℕ, 
  total_tickets = (rides_on_ferris_wheel * cost_per_ride) + (n * cost_per_ride) ∧ n = 3 :=
sorry

end bumper_car_rides_l385_385050


namespace vector_magnitude_problem_l385_385164

variables (a b : ℝ^3) (θ : ℝ)

noncomputable def vector_a_magnitude : Prop := ∥a∥ = 1
noncomputable def vector_b_magnitude : Prop := ∥b∥ = 2
noncomputable def angle_between_vectors : Prop := θ = real.pi / 3

theorem vector_magnitude_problem 
  (h1 : vector_a_magnitude a) 
  (h2 : vector_b_magnitude b) 
  (h3 : angle_between_vectors θ) :
  ∥(3 : ℝ) • a + b∥ = real.sqrt 19 := 
sorry

end vector_magnitude_problem_l385_385164


namespace net_increase_investment_l385_385389

noncomputable def initial_investment : ℝ := 100
noncomputable def first_year_increase_percent : ℝ := 0.80
noncomputable def second_year_decrease_percent : ℝ := 0.30

theorem net_increase_investment :
  let first_year_end_wealth := initial_investment * (1 + first_year_increase_percent) in
  let second_year_end_wealth := first_year_end_wealth * (1 - second_year_decrease_percent) in
  second_year_end_wealth - initial_investment = 26 :=
by
  sorry

end net_increase_investment_l385_385389


namespace probability_greater_120_l385_385556

noncomputable def normal_distribution (mean : ℝ) (σ : ℝ) : ProbabilityDistribution :=
sorry -- Placeholder for normal distribution definition

variable {X : ℝ → ProbabilityDistribution}

axiom normal_X : X = normal_distribution 100 σ

axiom probability_80_to_120 : ∀ {p : ℝ → ℝ}, (pX : Probability) (80 < X ≤ 120) = 3 / 4

theorem probability_greater_120 :
  (probability {x : ℝ | x > 120} X) = 1 / 8 :=
begin
  sorry
end

end probability_greater_120_l385_385556


namespace tangent_line_at_origin_l385_385672

noncomputable def tangent_line_eq (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ :=
  λ x, f a + (f' a) * (x - a)
  where
    f' : for x in ℝ, derives (f x) = f' x
   .

theorem tangent_line_at_origin (f : ℝ → ℝ) (hf : ∀ x : ℝ, f x = x^3) :
  tangent_line_eq f 0 = λ x, (0 : ℝ) :=
by
  sorry

end tangent_line_at_origin_l385_385672


namespace options_correct_l385_385938

-- Assume the domains and functions are appropriately defined
axiom f : ℝ → ℝ
axiom g : ℝ → ℝ

-- Conditions from the given problem
axiom domain_f : ∀ x : ℝ, f x ∈ ℝ  -- The domain of f(x) is ℝ
axiom fg_add_one : ∀ x : ℝ, g x + f x = 1  -- g(x) + f(x) = 1
axiom odd_g : ∀ x : ℝ, g (x + 1) = -g (-x + 1)  -- g(x + 1) is odd
axiom odd_f : ∀ x : ℝ, f (2 - x) = -f (2 + x)  -- f(2 - x) is odd

-- Theorem statement confirming the correct options
theorem options_correct : 
  g 0 = -1 ∧ g 1 = 0 ∧ g 2 = 1 ∧ g 3 ≠ 0 :=
by
  sorry

end options_correct_l385_385938


namespace find_mistaken_divisor_l385_385232

-- Define the conditions
def remainder : ℕ := 0
def quotient_correct : ℕ := 32
def divisor_correct : ℕ := 21
def quotient_mistaken : ℕ := 56
def dividend : ℕ := quotient_correct * divisor_correct + remainder

-- Prove the mistaken divisor
theorem find_mistaken_divisor : ∃ x : ℕ, dividend = quotient_mistaken * x + remainder ∧ x = 12 :=
by
  -- We leave this as an exercise to the prover
  sorry

end find_mistaken_divisor_l385_385232


namespace cos_210_eq_neg_sqrt3_div_2_l385_385851

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385851


namespace length_QJ_incircle_l385_385247

theorem length_QJ_incircle
  (P Q R J D E F : Type)
  (PQ PR QR : ℝ)
  (hPQ : PQ = 11)
  (hPR : PR = 17)
  (hQR : QR = 10)
  (incircle_touches : ∀ (x : Type), x ∈ {D, E, F} ⊆ {QR, RP, PQ})
  (incenter_J : incenter PQR J)
  : ∃ QJ : ℝ, QJ = 3.41 :=
by
  sorry

end length_QJ_incircle_l385_385247


namespace find_k_value_l385_385625

theorem find_k_value :
  (∃ p q : ℝ → ℝ,
    (∀ x, p x = 3 * x + 5) ∧
    (∃ k : ℝ, (∀ x, q x = k * x + 3) ∧
      (p (-4) = -7) ∧ (q (-4) = -7) ∧ k = 2.5)) :=
by
  sorry

end find_k_value_l385_385625


namespace speed_difference_is_36_l385_385441

open Real

noncomputable def alex_speed : ℝ := 8 / (40 / 60)
noncomputable def jordan_speed : ℝ := 12 / (15 / 60)
noncomputable def speed_difference : ℝ := jordan_speed - alex_speed

theorem speed_difference_is_36 : speed_difference = 36 := by
  have hs1 : alex_speed = 8 / (40 / 60) := rfl
  have hs2 : jordan_speed = 12 / (15 / 60) := rfl
  have hd : speed_difference = jordan_speed - alex_speed := rfl
  rw [hs1, hs2] at hd
  simp [alex_speed, jordan_speed, speed_difference] at hd
  sorry

end speed_difference_is_36_l385_385441


namespace area_difference_of_circles_l385_385460

theorem area_difference_of_circles : 
  let r1 := 30
  let r2 := 15
  let pi := Real.pi
  900 * pi - 225 * pi = 675 * pi := by
  sorry

end area_difference_of_circles_l385_385460


namespace problem_l385_385935

-- Given conditions
variable {R : Type} [field R]
variable (g f : R → R)

-- Definitions from conditions
def condition1 : Prop := ∀ x, g(x) + f(x) = 1
def condition2 : Prop := ∀ x, g(x) = -g(-x + 2)
def condition3 : Prop := ∀ x, f(x) = -f(4 - x)

-- Statements to prove
def statement_g0  : Prop := g 0 = -1
def statement_g1  : Prop := g 1 = 0
def statement_g2  : Prop := g 2 = 1
def statement_g3  : Prop := g 3 = 2

-- Lean theorem statements
theorem problem (h1 : condition1 g f) (h2 : condition2 g) (h3 : condition3 f) :
  (statement_g0 g) ∧ (statement_g1 g) ∧ (statement_g2 g) ∧ ¬(statement_g3 g) := sorry

end problem_l385_385935


namespace regular_hexagon_points_on_line_l385_385310

theorem regular_hexagon_points_on_line 
  (ABCDEF : Type)
  [hexagon : RegularHexagon ABCDEF] 
  (A B C D E F : hexagon.Point)
  (M N : hexagon.Point)
  {λ : ℝ} 
  (h₁ : λ = AM/AC)
  (h₂ : λ = CN/CE)
  (h₃ : collinear B M N) :
  λ = (√3)/3 := 
sorry

end regular_hexagon_points_on_line_l385_385310


namespace ratio_fourth_to_third_l385_385304

theorem ratio_fourth_to_third (third_graders fifth_graders fourth_graders : ℕ) (H1 : third_graders = 20) (H2 : fifth_graders = third_graders / 2) (H3 : third_graders + fifth_graders + fourth_graders = 70) : fourth_graders / third_graders = 2 := by
  sorry

end ratio_fourth_to_third_l385_385304


namespace circle_hyperbola_intersection_l385_385007

def hyperbola_equation (x y a : ℝ) : Prop := x^2 - y^2 = a^2
def circle_equation (x y c d r : ℝ) : Prop := (x - c)^2 + (y - d)^2 = r^2

theorem circle_hyperbola_intersection (a r : ℝ) (P Q R S : ℝ × ℝ):
  (∃ c d: ℝ, 
    circle_equation P.1 P.2 c d r ∧ 
    circle_equation Q.1 Q.2 c d r ∧ 
    circle_equation R.1 R.2 c d r ∧ 
    circle_equation S.1 S.2 c d r ∧ 
    hyperbola_equation P.1 P.2 a ∧ 
    hyperbola_equation Q.1 Q.2 a ∧ 
    hyperbola_equation R.1 R.2 a ∧ 
    hyperbola_equation S.1 S.2 a
  ) →
  (P.1^2 + P.2^2) + (Q.1^2 + Q.2^2) + (R.1^2 + R.2^2) + (S.1^2 + S.2^2) = 4 * r^2 :=
by
  sorry

end circle_hyperbola_intersection_l385_385007


namespace smallest_possible_sum_l385_385604

noncomputable def is_good (a b : ℕ) : Prop := a >= b / 2 + 7

noncomputable def is_satisfiable (a b : ℕ) (S T : Finset ℕ) : Prop :=
  is_good a b ∧ a ∈ S ∧ b ∈ T

noncomputable def is_unacceptable (R : Finset ℕ) (S T : Finset ℕ) : Prop :=
  if h := ∃a ∈ R, ∃b ∈ T, is_satisfiable a b S T then
    ∃ (b_count: ℕ → ℕ), (∀ b ∈ T, b_count b = 1) ∧
    ∃ (r_count: ℕ → ℕ), (∀ r ∈ R, r_count r = 1) ∧
    (∑ b in T.filter (λ b => ∃ a ∈ R, is_satisfiable a b S T), 1 ) < R.card 
  else 
    false

noncomputable def problem_statement (S T : Finset ℕ) : Prop :=
  S = {14, 20, 16, 32, 23, 31} ∧
  (∀ R ⊆ S, ¬is_unacceptable R S T) ∧
  T.card >= 20 ∧
  T.sum = 210

theorem smallest_possible_sum (S T : Finset ℕ) : problem_statement S T :=
begin
  sorry
end

end smallest_possible_sum_l385_385604


namespace susan_gave_robert_cats_l385_385328

theorem susan_gave_robert_cats :
  ∃ x : ℕ, (21 - x) = (3 + x) + 14 ∧ x = 2 :=
by {
  existsi 2,
  split,
  { linarith },
  { simp },
  sorry
}

end susan_gave_robert_cats_l385_385328


namespace factorization_proof_l385_385085

def factorization_problem (x : ℝ) : Prop := (x^2 - 1)^2 - 6 * (x^2 - 1) + 9 = (x - 2)^2 * (x + 2)^2

theorem factorization_proof (x : ℝ) : factorization_problem x :=
by
  -- The proof is omitted.
  sorry

end factorization_proof_l385_385085


namespace remainder_of_disjoint_subsets_mod_1000_l385_385611

open Function

def setT : Finset ℕ := {i | 1 <= i ∧ i <= 12}.toFinset

def numberOfDisjointSubsets (T : Finset ℕ) : ℕ := (3 ^ T.card - 2 * 2 ^ T.card + 1) / 2

def remainderWhenDividedBy1000 (n : ℕ) : ℕ := n % 1000

theorem remainder_of_disjoint_subsets_mod_1000 :
  remainderWhenDividedBy1000 (numberOfDisjointSubsets setT) = 625 :=
  by
  sorry

end remainder_of_disjoint_subsets_mod_1000_l385_385611


namespace main_theorem_l385_385140

-- r sequence definition
def r_seq (n : ℕ) (r : ℕ → ℕ) : Prop :=
  ∀ n ≥ 2, r n = r 1 * r 2 * ... * r (n - 1) + 1

-- Given a sequence of natural numbers a, with the condition on their reciprocals sum
def sum_reciprocal_lt_one (n : ℕ) (a : ℕ → ℕ) : Prop :=
  (∑ i in finset.range n, 1 / (a i)) < 1

-- The sequence of natural numbers a and r with the required properties
variables (n : ℕ) (a r : ℕ → ℕ)

theorem main_theorem :
  n ≥ 2 →
  r_seq n r →
  sum_reciprocal_lt_one n a →
  (∑ i in finset.range n, 1 / (a i)) ≤ 1 - ∑ i in finset.range n, 1 / (r i) := sorry

end main_theorem_l385_385140


namespace nancy_weight_l385_385505

theorem nancy_weight (w : ℕ) (h : (60 * w) / 100 = 54) : w = 90 :=
by
  sorry

end nancy_weight_l385_385505


namespace R_has_n_real_roots_l385_385258

-- Define P(x) as a polynomial of degree n with specified roots
noncomputable def P (x : ℂ) (n : ℕ) : ℂ := 
  ∏ k in finset.range n, (x - (complex.I - k))

-- Define R(x) and S(x) such that P(x) = R(x) + iS(x) and they have real coefficients
noncomputable def R (x : ℝ) := ∑ n in finset.range n, (some_real_polynomial n x)
noncomputable def S (x : ℝ) := ∑ n in finset.range n, (some_real_polynomial n x)

-- The main theorem to prove
theorem R_has_n_real_roots (n : ℕ) (P R : polynomial ℝ) (S : polynomial ℝ) 
  (hP : P = R + (complex.I • S))
  (hdegP : P.nat_degree = n)
  (hrootsP : ∀ k : ℕ, k < n → is_root P (complex.I - k)) :
  ∃ (roots : fin n → ℝ), ∀ k, is_root R (roots k) :=
sorry

end R_has_n_real_roots_l385_385258


namespace factorize_difference_of_squares_l385_385089

variable (x : ℝ)

theorem factorize_difference_of_squares :
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorize_difference_of_squares_l385_385089


namespace equilibrium_angles_l385_385051

def degrees_to_radians (d m s : Float) : Float :=
  d * Float.pi / 180 + m * Float.pi / (180 * 60) + s * Float.pi / (180 * 3600)

noncomputable def alpha := degrees_to_radians 52 45 20
noncomputable def beta := degrees_to_radians 162 44 48
noncomputable def gamma := degrees_to_radians 144 29 52

def a : Float := 2488
def b : Float := 927
def c : Float := 1815

theorem equilibrium_angles :
  a * Float.cos alpha + b * Float.cos beta + c * Float.cos gamma = 0 ∧ 
  a * Float.sin alpha + b * Float.sin beta + c * Float.sin gamma = 0 :=
by
  sorry

end equilibrium_angles_l385_385051


namespace evaluate_expression_l385_385892

theorem evaluate_expression (b : ℕ) (h : b = 2) : (b^3 * b^4) - 10 = 118 :=
by
  rw h
  norm_num
  sorry

end evaluate_expression_l385_385892


namespace largest_square_area_l385_385048

theorem largest_square_area (a b c : ℝ)
  (h1 : a^2 + b^2 = c^2)
  (h2 : a^2 + b^2 + c^2 = 450) :
  c^2 = 225 :=
by
  sorry

end largest_square_area_l385_385048


namespace big_container_capacity_l385_385754

theorem big_container_capacity (C : ℝ) (h1 : 0.35 * C + 48 = 0.75 * C) :
  C = 120 := 
begin
  sorry
end

end big_container_capacity_l385_385754


namespace georges_total_stickers_l385_385879

theorem georges_total_stickers :
  let Bob := 12 in
  let Tom := 3 * Bob in
  let Dan := 2 * Tom in
  let George := 5 * Dan in
  let total := Bob + Tom + Dan + George in
  let distributed := 100 in
  let total_after_distribution := total + distributed in
  let each := total_after_distribution / 4 in
  George + each = 505 := by
  sorry

end georges_total_stickers_l385_385879


namespace larger_number_is_84_l385_385678

theorem larger_number_is_84 (x y : ℕ) (HCF LCM : ℕ)
  (h_hcf : HCF = 84)
  (h_lcm : LCM = 21)
  (h_ratio : x * 4 = y)
  (h_product : x * y = HCF * LCM) :
  y = 84 :=
by
  sorry

end larger_number_is_84_l385_385678


namespace falling_factorial_sum_l385_385606

open BigOperators

noncomputable def falling_factorial (x : ℕ) : ℕ → ℕ
| 0 := 1
| (n+1) := x * falling_factorial n

theorem falling_factorial_sum (x y : ℕ) (n : ℕ) :
  falling_factorial (x + y) n = ∑ k in finset.range (n + 1), nat.choose n k * falling_factorial x k * falling_factorial y (n - k) :=
sorry

end falling_factorial_sum_l385_385606


namespace smallest_n_for_rms_integer_l385_385512

theorem smallest_n_for_rms_integer :
  let rms (n : ℕ) : Real := Real.sqrt ((∑ i in Finset.range (n+1), (i+1)^2 : ℚ) / n)
  in (∀ n > 1, ∃ k : ℕ, rms n = k) → n = 337 :=
by
  sorry

end smallest_n_for_rms_integer_l385_385512


namespace prime_only_one_solution_l385_385112

theorem prime_only_one_solution (p : ℕ) (hp : Nat.Prime p) : 
  (∃ k : ℕ, 2 * p^4 - p^2 + 16 = k^2) → p = 3 := 
by 
  sorry

end prime_only_one_solution_l385_385112


namespace road_coloring_possible_l385_385673

-- Definitions of the problem
def City : Type := sorry
def Republic := Set City
def Federation := Republic × Republic

variable (A B : Republic)
variable (roads : Set (City × City))
variable (colors : Fin 10)

-- Define the condition that each road connects a city in A to a city in B
def connects_cities (road : City × City) : Prop := 
  (road.1 ∈ A ∧ road.2 ∈ B) ∨ (road.1 ∈ B ∧ road.2 ∈ A)

-- Define the condition that from any city, there are at most 10 roads
def max_10_roads (c : City) : Prop :=
  roads.count (λ road, road.1 = c ∨ road.2 = c) ≤ 10

-- Theorem statement with conditions and question
theorem road_coloring_possible :
  (∀ road : City × City, road ∈ roads → connects_cities road) →
  (∀ c : City, max_10_roads c) →
  ∃ (coloring : roads → colors),
    ∀ (c : City) (r1 r2 : City × City),
      r1 ∈ roads → r2 ∈ roads → (r1.1 = c ∨ r1.2 = c) →
      (r2.1 = c ∨ r2.2 = c) → r1 ≠ r2 → coloring r1 ≠ coloring r2 :=
sorry

end road_coloring_possible_l385_385673


namespace cos_210_eq_neg_sqrt3_div2_l385_385837

theorem cos_210_eq_neg_sqrt3_div2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div2_l385_385837


namespace part1_part2_l385_385550

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 4 * x + a + 3
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := b * x + 5 - 2 * b

theorem part1 (a : ℝ) : (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f x a = 0) ↔ -8 ≤ a ∧ a ≤ 0 :=
sorry

theorem part2 (b : ℝ) : (∀ x1 : ℝ, 1 ≤ x1 ∧ x1 ≤ 4 → ∃ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 4 ∧ f x2 3 = g x1 b) ↔ -1 ≤ b ∧ b ≤ 1/2 :=
sorry

end part1_part2_l385_385550


namespace no_common_points_iff_parallel_l385_385008

-- Definitions based on conditions:
def line (a : Type) : Prop := sorry
def plane (M : Type) : Prop := sorry
def no_common_points (a : Type) (M : Type) : Prop := sorry
def parallel (a : Type) (M : Type) : Prop := sorry

-- Theorem stating the relationship is necessary and sufficient
theorem no_common_points_iff_parallel (a M : Type) :
  no_common_points a M ↔ parallel a M := sorry

end no_common_points_iff_parallel_l385_385008


namespace inequality_solution_f_is_odd_l385_385742

-- Problem 1 
theorem inequality_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (a > 1 ∧ ∀ x : ℝ, a^(2*x - 1) > a^(x + 2) → x > 3) ∧
  (0 < a ∧ a < 1 ∧ ∀ x : ℝ, a^(2*x - 1) > a^(x + 2) → x < 3) :=
sorry

-- Problem 2
def f (x : ℝ) : ℝ :=
if x > 0 then sqrt x + 1 
else if x = 0 then 0 
else -sqrt (-x) - 1

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
sorry

end inequality_solution_f_is_odd_l385_385742


namespace area_GPQ_div_ABC_l385_385250

-- Define a triangle ABC with medians AD, BE, and CF intersecting at centroid G
variable (A B C D E F G P Q : Point)

-- Given conditions
variable hABC_triangle : is_triangle A B C
variable hMedians_intersect_centroid : ∀ (X : Point), is_median X G
variable hMidpoint_P_AF : midpoint P A F
variable hMidpoint_Q_BC : midpoint Q B C
variable hArea_ratio_GPQ_ABC : area (triangle G P Q) = (1 / 9 : ℝ) * area (triangle A B C)

-- Statement to prove
theorem area_GPQ_div_ABC (hABC_triangle : is_triangle A B C)
    (hMedians_intersect_centroid : ∀ (X : Point), is_median X G)
    (hMidpoint_P_AF : midpoint P A F)
    (hMidpoint_Q_BC : midpoint Q B C) :
    area (triangle G P Q) = (1 / 9 : ℝ) * area (triangle A B C) :=
sorry  -- The proof is omitted

end area_GPQ_div_ABC_l385_385250


namespace angle_equiv_330_neg390_l385_385445

theorem angle_equiv_330_neg390 : ∃ k : ℤ, 330 = -390 + 360 * k :=
by
  sorry

end angle_equiv_330_neg390_l385_385445


namespace cos_210_eq_neg_sqrt3_div_2_l385_385858

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385858


namespace orange_roses_count_l385_385592

variables (roses_red roses_pink roses_yellow roses_total : ℕ)
variable (x : ℕ) -- total number of orange roses

-- Given conditions
def conditions := 
  roses_red = 12 ∧ 
  roses_pink = 18 ∧ 
  roses_yellow = 20 ∧ 
  (0.5 * roses_red).toNat + (0.5 * roses_pink).toNat + (0.25 * roses_yellow).toNat + (0.25 * x).toNat = 22

-- Prove that the number of orange roses is 8
theorem orange_roses_count : conditions roses_red roses_pink roses_yellow 22 x → x = 8 :=
by
  intros h
  sorry

end orange_roses_count_l385_385592


namespace man_speed_with_current_l385_385418

-- Define the conditions
def current_speed : ℕ := 3
def man_speed_against_current : ℕ := 14

-- Define the man's speed in still water (v) based on the given speed against the current
def man_speed_in_still_water : ℕ := man_speed_against_current + current_speed

-- Prove that the man's speed with the current is 20 kmph
theorem man_speed_with_current : man_speed_in_still_water + current_speed = 20 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end man_speed_with_current_l385_385418


namespace number_of_shoes_lost_l385_385302

-- Definitions for the problem conditions
def original_pairs : ℕ := 20
def pairs_left : ℕ := 15
def shoes_per_pair : ℕ := 2

-- Translating the conditions to individual shoe counts
def original_shoes : ℕ := original_pairs * shoes_per_pair
def remaining_shoes : ℕ := pairs_left * shoes_per_pair

-- Statement of the proof problem
theorem number_of_shoes_lost : original_shoes - remaining_shoes = 10 := by
  sorry

end number_of_shoes_lost_l385_385302


namespace mixture_ratios_equal_quantities_l385_385002

-- Define the given conditions
def ratio_p_milk_water := (5, 4)
def ratio_q_milk_water := (2, 7)

-- Define what we're trying to prove: the ratio p : q such that the resulting mixture has equal milk and water
theorem mixture_ratios_equal_quantities 
  (P Q : ℝ) 
  (h1 : 5 * P + 2 * Q = 4 * P + 7 * Q) :
  P / Q = 5 :=
  sorry

end mixture_ratios_equal_quantities_l385_385002


namespace shopkeeper_standard_weight_l385_385429

theorem shopkeeper_standard_weight
    (cost_price : ℝ)
    (actual_weight_used : ℝ)
    (profit_percentage : ℝ)
    (standard_weight : ℝ)
    (H1 : actual_weight_used = 800)
    (H2 : profit_percentage = 25) :
    standard_weight = 1000 :=
by 
    sorry

end shopkeeper_standard_weight_l385_385429


namespace sequence_general_term_l385_385966

noncomputable def a_n (n : ℕ) : ℝ :=
  if n = 0 then 4 else 4 * (-1 / 3)^(n - 1) 

theorem sequence_general_term (n : ℕ) (hn : n ≥ 1) 
  (hrec : ∀ n, 3 * a_n (n + 1) + a_n n = 0)
  (hinit : a_n 2 = -4 / 3) :
  a_n n = 4 * (-1 / 3)^(n - 1) := by
  sorry

end sequence_general_term_l385_385966


namespace min_value_expression_l385_385497

def find_min_value (x : ℝ) : ℝ := (sin x) ^ 8 + (cos x) ^ 8 + 1 / (sin x) ^ 6 + (cos x) ^ 6 + 1

theorem min_value_expression : ∀ x : ℝ, (sin x)^2 + (cos x)^2 = 1 → 
  find_min_value x = 1 / 2 :=
sorry

end min_value_expression_l385_385497


namespace pairs_satisfy_ineq_l385_385011

theorem pairs_satisfy_ineq (x y : ℝ) :
  |Real.sin x - Real.sin y| + Real.sin x * Real.sin y ≤ 0 ↔
  ∃ n m : ℤ, x = n * Real.pi ∧ y = m * Real.pi := 
sorry

end pairs_satisfy_ineq_l385_385011


namespace total_surface_area_of_tower_l385_385747

-- Definition of given cube volumes
def cube_volumes : List ℕ := [1, 8, 27, 64, 125, 216, 343, 512]

-- Function to calculate the side length of a cube from its volume
def cube_side_length (v : ℕ) : ℕ :=
  Nat.cbrt v

-- Function to calculate the surface area of a cube given its side length
def surface_area (s : ℕ) : ℕ :=
  6 * s^2

-- Function to calculate the adjusted surface area for the cubes in the tower
def adjusted_surface_area (volumes : List ℕ) : ℕ :=
  let side_lengths := volumes.map cube_side_length
  let areas := side_lengths.map surface_area
  let overlap_adjustments := side_lengths.drop 1 -- Ignore the first length for adjustment
                            |>.map (λ s => s^2)
  areas.head! + -- Add the surface area of the first cube
  (areas.drop 1).zipWith (λ a o => a - o) overlap_adjustments |>.sum

-- The proof goal: the total surface area of the tower
theorem total_surface_area_of_tower : adjusted_surface_area cube_volumes = 1021 :=
by
  sorry

end total_surface_area_of_tower_l385_385747


namespace sum_of_largest_odd_divisors_111_to_219_l385_385293

def largest_odd_divisor (n : ℕ) : ℕ :=
  if n % 2 = 1 then n else largest_odd_divisor (n / 2)

def sum_largest_odd_divisors (start end : ℕ) : ℕ :=
  let rec helper (current end acc : ℕ) :=
    if current > end then acc
    else helper (current + 1) end (acc + largest_odd_divisor current)
  helper start end 0

theorem sum_of_largest_odd_divisors_111_to_219 :
  sum_largest_odd_divisors 111 219 = 12045 :=
by
  sorry

end sum_of_largest_odd_divisors_111_to_219_l385_385293


namespace larger_number_of_product_56_and_sum_15_l385_385718

theorem larger_number_of_product_56_and_sum_15 (x y : ℕ) (h1 : x * y = 56) (h2 : x + y = 15) : max x y = 8 := 
by
  sorry

end larger_number_of_product_56_and_sum_15_l385_385718


namespace sum_possible_students_l385_385032

theorem sum_possible_students (s : ℕ) (h_range : 100 ≤ s ∧ s ≤ 250)
  (h_sections : ∃ k : ℕ, s - 1 = 7 * k) :
  ∑ x in finset.filter (λ x, x % 7 = 1) (finset.range 251), if 100 ≤ x ∧ x ≤ 250 then x else 0 = 3696 :=
by sorry

end sum_possible_students_l385_385032


namespace corrected_mean_is_40_point_6_l385_385393

theorem corrected_mean_is_40_point_6 
  (mean_original : ℚ) (num_observations : ℕ) (wrong_observation : ℚ) (correct_observation : ℚ) :
  num_observations = 50 → mean_original = 40 → wrong_observation = 15 → correct_observation = 45 →
  ((mean_original * num_observations + (correct_observation - wrong_observation)) / num_observations = 40.6 : Prop) :=
by intros; sorry

end corrected_mean_is_40_point_6_l385_385393


namespace Ming_initial_ladybugs_l385_385321

-- Define the conditions
def Sami_spiders : Nat := 3
def Hunter_ants : Nat := 12
def insects_remaining : Nat := 21
def ladybugs_flew_away : Nat := 2

-- Formalize the proof problem
theorem Ming_initial_ladybugs : Sami_spiders + Hunter_ants + (insects_remaining + ladybugs_flew_away) - (Sami_spiders + Hunter_ants) = 8 := by
  sorry

end Ming_initial_ladybugs_l385_385321


namespace problem_l385_385936

-- Given conditions
variable {R : Type} [field R]
variable (g f : R → R)

-- Definitions from conditions
def condition1 : Prop := ∀ x, g(x) + f(x) = 1
def condition2 : Prop := ∀ x, g(x) = -g(-x + 2)
def condition3 : Prop := ∀ x, f(x) = -f(4 - x)

-- Statements to prove
def statement_g0  : Prop := g 0 = -1
def statement_g1  : Prop := g 1 = 0
def statement_g2  : Prop := g 2 = 1
def statement_g3  : Prop := g 3 = 2

-- Lean theorem statements
theorem problem (h1 : condition1 g f) (h2 : condition2 g) (h3 : condition3 f) :
  (statement_g0 g) ∧ (statement_g1 g) ∧ (statement_g2 g) ∧ ¬(statement_g3 g) := sorry

end problem_l385_385936


namespace max_n_sum_of_squares_l385_385720

theorem max_n_sum_of_squares (n : ℕ) (k : ℕ → ℕ) (h_distinct : ∀ i j, i ≠ j → k i ≠ k j) (h_sum : (∑ i in finset.range n, (k i)^2) = 2500) : 
  n ≤ 18 :=
sorry

end max_n_sum_of_squares_l385_385720


namespace sum_first_10_terms_arithmetic_sequence_l385_385264

-- Define the first term and the sum of the second and sixth terms as given conditions
def a1 : ℤ := -2
def condition_a2_a6 (a2 a6 : ℤ) : Prop := a2 + a6 = 2

-- Define the general term 'a_n' of the arithmetic sequence
def a_n (a1 d : ℤ) (n : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum 'S_n' of the first 'n' terms of the arithmetic sequence
def S_n (a1 d : ℤ) (n : ℤ) : ℤ := n * (a1 + ((n - 1) * d) / 2)

-- The theorem statement to prove that S_10 = 25 given the conditions
theorem sum_first_10_terms_arithmetic_sequence 
  (d a2 a6 : ℤ) 
  (h1 : a2 = a_n a1 d 2) 
  (h2 : a6 = a_n a1 d 6)
  (h3 : condition_a2_a6 a2 a6) : 
  S_n a1 d 10 = 25 := 
by
  sorry

end sum_first_10_terms_arithmetic_sequence_l385_385264


namespace percentage_speaking_both_langs_l385_385457

def diplomats_total : ℕ := 100
def diplomats_french : ℕ := 22
def diplomats_not_russian : ℕ := 32
def diplomats_neither : ℕ := 20

theorem percentage_speaking_both_langs
  (h1 : 20% diplomats_total = diplomats_neither)
  (h2 : diplomats_total - diplomats_not_russian = 68)
  (h3 : diplomats_total ≠ 0) :
  (22 + 68 - 80) / diplomats_total * 100 = 10 :=
by
  sorry

end percentage_speaking_both_langs_l385_385457


namespace bird_families_migration_l385_385386

theorem bird_families_migration 
  (total_families : ℕ)
  (africa_families : ℕ)
  (asia_families : ℕ)
  (south_america_families : ℕ)
  (africa_days : ℕ)
  (asia_days : ℕ)
  (south_america_days : ℕ)
  (migrated_families : ℕ)
  (remaining_families : ℕ)
  (total_migration_time : ℕ)
  (H1 : total_families = 200)
  (H2 : africa_families = 60)
  (H3 : asia_families = 95)
  (H4 : south_america_families = 30)
  (H5 : africa_days = 7)
  (H6 : asia_days = 14)
  (H7 : south_america_days = 10)
  (H8 : migrated_families = africa_families + asia_families + south_america_families)
  (H9 : remaining_families = total_families - migrated_families)
  (H10 : total_migration_time = 
          africa_families * africa_days + 
          asia_families * asia_days + 
          south_america_families * south_america_days) :
  remaining_families = 15 ∧ total_migration_time = 2050 :=
by
  sorry

end bird_families_migration_l385_385386


namespace spending_example_l385_385484

theorem spending_example (X : ℝ) (h₁ : X + 2 * X + 3 * X = 120) : X = 20 := by
  sorry

end spending_example_l385_385484


namespace sum_p_q_r_l385_385278

noncomputable def a : ℝ := Real.sqrt (9 / 25)
noncomputable def b : ℝ := -Real.sqrt (((3 + Real.sqrt 2) ^ 2) / 14)

theorem sum_p_q_r :
  (a^2 = 9 / 25) ∧
  (0 < a) ∧
  (b^2 = (3 + Real.sqrt 2) ^ 2 / 14) ∧
  (b < 0) ∧
  ((a - b) ^ 3 = 88 * Real.sqrt 2 / 12750) →
  88 + 2 + 12750 = 12840 := by
  intro h
  cases h
  sortrying (sorry : ((a - b) ^ 3 = 88 * Real.sqrt 2 / 12750))
  rfl

end sum_p_q_r_l385_385278


namespace correct_calculation_l385_385376

theorem correct_calculation :
  (-2 * a * b^2)^3 = -8 * a^3 * b^6 :=
by sorry

end correct_calculation_l385_385376


namespace train_car_count_l385_385482

theorem train_car_count (counted : ℕ) (time_counted : ℝ) (total_time : ℝ) :
  counted = 8 → time_counted = 12 → total_time = 210 → ∃ n, n = 140 := 
by 
  intros h1 h2 h3
  use 140
  sorry

end train_car_count_l385_385482


namespace max_min_PQ_distance_l385_385515

def incircle_eq (x y : ℝ) : Prop := (x - 1) ^ 2 + (y - 1) ^ 2 = 1

def circumcircle_eq (x z : ℝ) : Prop := (x - 1) ^ 2 + (z - 1) ^ 2 = 2

def distance_squared (P Q : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := P;
  let (x2, y2, z2) := Q;
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 + (z1 - z2) ^ 2

theorem max_min_PQ_distance :
  ∀ (α β : ℝ),
    incircle_eq (1 + Float.cos α) (1 + Float.sin α) →
    circumcircle_eq (1 + Float.sqrt 2 * Float.cos β) (1 + Float.sqrt 2 * Float.sin β) →
    let P := (1 + Float.cos α, 1 + Float.sin α, 0);
    let Q := (1 + Float.sqrt 2 * Float.cos β, 0, 1 + Float.sqrt 2 * Float.sin β);
    let d := distance_squared P Q;
    d ≤ (5 + 2 * Float.sqrt 6) ∧ d ≥ (5 - 2 * Float.sqrt 6) :=
sorry

end max_min_PQ_distance_l385_385515


namespace greatest_perfect_power_sum_l385_385815

def sum_c_d_less_500 : ℕ :=
  let c := 22
  let d := 2
  c + d

theorem greatest_perfect_power_sum :
  ∃ c d : ℕ, 0 < c ∧ 1 < d ∧ c^d < 500 ∧
  ∀ x y : ℕ, 0 < x ∧ 1 < y ∧ x^y < 500 → x^y ≤ c^d ∧ (c + d = 24) :=
by
  sorry

end greatest_perfect_power_sum_l385_385815


namespace correct_conclusion_l385_385985

theorem correct_conclusion 
  (negation_not_correct : ¬ (∀ x : ℝ, x^2 + x + 1 > 0) ↔ ∃ x : ℝ, x^2 + x + 1 ≤ 0)
  (converse_correct : (∀ a b : ℝ, a * b = 0 → (a = 0 ∨ b = 0)) ↔ (∀ a b : ℝ, a * b ≠ 0 → (a ≠ 0 ∧ b ≠ 0)))
  (regression_incorrect : ∀ x : ℕ, ¬ (∃ y : ℕ, y = 3 + 2 * x ∧ x = 2 ∧ y = 7)) 
  (probability_incorrect : (∀ a b : ℝ, a ∈ set.Icc 0 1 ∧ b ∈ set.Icc 0 1 → a^2 + b^2 < (1/4)) ∨ (4 / π ≠ (1 / 16))) : 
  (∀ (stmt_correct : Prop), stmt_correct ↔ (converse_correct)).
Proof
sory

end correct_conclusion_l385_385985


namespace distance_covered_is_correct_l385_385027

-- Definition of the given conditions
def boat_speed_in_still_water : ℝ := 15  -- in kmph
def current_speed : ℝ := 3  -- in kmph
def time_taken_seconds : ℝ := 11.999040076793857  -- in seconds

-- Conversion constants
def km_to_m (x : ℝ) : ℝ := x * 1000
def hour_to_s (x : ℝ) : ℝ := x * 3600

-- Theorem statement
theorem distance_covered_is_correct :
  let effective_speed_kmph := boat_speed_in_still_water + current_speed
  let effective_speed_mps := (km_to_m effective_speed_kmph) / hour_to_s 1
  let distance_covered := effective_speed_mps * time_taken_seconds
  distance_covered ≈ 59.995200383969285 := by sorry

end distance_covered_is_correct_l385_385027


namespace range_of_a_l385_385176

noncomputable def f (a x : ℝ) : ℝ := (1 / 3) * x^3 + x^2 + a * x

noncomputable def g (x : ℝ) : ℝ := 1 / (Real.exp x)

theorem range_of_a (a : ℝ) :
  (∃ (x1 x2 : ℝ), x1 ∈ Set.Icc (1 / 2) 2 ∧ x2 ∈ Set.Icc (1 / 2) 2 ∧
    (f a x1)' ≤ g x2) ↔ a ≤ Real.sqrt Real.exp 1 / Real.exp 1 - 5 / 4 :=
by
  sorry

end range_of_a_l385_385176


namespace c_pays_l385_385790

-- Defining the problem condition
variables (cost_price_A : ℝ) (profit_rate_A : ℝ) (profit_rate_B : ℝ)

-- These are the given conditions from the problem
def condition1 := cost_price_A = 150
def condition2 := profit_rate_A = 0.20
def condition3 := profit_rate_B = 0.25

-- We need to prove that the selling price for C is 225
theorem c_pays (h₁ : condition1) (h₂ : condition2) (h₃ : condition3) : 
  let selling_price_A := cost_price_A + profit_rate_A * cost_price_A in
  let cost_price_B := selling_price_A in
  let profit_B := profit_rate_B * cost_price_B in
  let selling_price_B := cost_price_B + profit_B in
  selling_price_B = 225 :=
by
  sorry

end c_pays_l385_385790


namespace actual_price_of_good_l385_385735

theorem actual_price_of_good (P : ℝ) 
  (hp : 0.684 * P = 6500) : P = 9502.92 :=
by 
  sorry

end actual_price_of_good_l385_385735


namespace cos_210_eq_neg_sqrt3_div2_l385_385838

theorem cos_210_eq_neg_sqrt3_div2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div2_l385_385838


namespace find_multiplier_l385_385024

theorem find_multiplier (A N : ℕ) (h : A = 64) : (2 * (A + 8) * N - 2 * N * (A - 8)) / 2 = A → N = 8 :=
by
  intros h1
  rw h at h1
  sorry

end find_multiplier_l385_385024


namespace henry_initial_action_figures_l385_385199

theorem henry_initial_action_figures (total_needed : ℕ) (cost_per_figure : ℕ) (amount_needed : ℕ) :
  total_needed = 8 → cost_per_figure = 6 → amount_needed = 30 → 
  let figures_to_buy := amount_needed / cost_per_figure in
  let initial_figures := total_needed - figures_to_buy in
  initial_figures = 3 :=
by
  intro h1 h2 h3
  let figures_to_buy := amount_needed / cost_per_figure
  let initial_figures := total_needed - figures_to_buy
  have h4 : figures_to_buy = 5 := by rw [h2, h3]; norm_num
  have h5 : initial_figures = 3 := by rw [h1, h4]; norm_num
  exact h5

end henry_initial_action_figures_l385_385199


namespace nancy_weight_l385_385506

theorem nancy_weight (w : ℕ) (h : (60 * w) / 100 = 54) : w = 90 :=
by
  sorry

end nancy_weight_l385_385506


namespace correct_conclusions_count_l385_385377

theorem correct_conclusions_count :
  let stmt1 := ¬(0 = (∃ (n : ℤ), (n < 0)))
      stmt2 := ¬(∃ (q : ℚ), 7 < q ∧ q < 9 ∧ (∀ (r : ℚ), 7 < r ∧ r < 9 → r = q))
      stmt3 := ∀ (a b : ℕ), a + b = 0 → a = -b
      stmt4 := ∀ (a b c : ℚ), c = a - b → ¬(c < b)
      stmt5 := ∀ (n : ℚ), (0 < n ∧ abs n < 1) → (1 < n)
      stmt6 := ∀ (q : ℚ), ¬((q > 0 ∨ q < 0) ∨ q = 0)
  in
  (stmt3 ∧ stmt4) ∧ ¬stmt1 ∧ ¬stmt2 ∧ ¬stmt5 ∧ ¬stmt6 → true := 
by 
  sorry

end correct_conclusions_count_l385_385377


namespace water_removal_l385_385706

noncomputable def total_water_removed_liters 
  (length_main_pool width_main_pool length_sunbathing_area width_sunbathing_area : ℝ)
  (height_main_pool_remove height_sunbathing_area_level conversion_factor : ℝ) : ℝ :=
let volume_main_pool := length_main_pool * width_main_pool * height_main_pool_remove in
let volume_sunbathing_area := length_sunbathing_area * width_sunbathing_area * height_sunbathing_area_level in
let total_volume_cubic_feet := volume_main_pool + volume_sunbathing_area in
total_volume_cubic_feet * conversion_factor

theorem water_removal 
  (length_main_pool width_main_pool length_sunbathing_area width_sunbathing_area : ℝ)
  (height_main_pool_remove height_sunbathing_area_level conversion_factor : ℝ)
  (conversion_10_inches : 10 / 12 = height_main_pool_remove)
  (conversion_4_inches : 4 / 12 = height_sunbathing_area_level)
  (conversion_factor_value : conversion_factor = 28.3168)
  : total_water_removed_liters length_main_pool width_main_pool 
                              length_sunbathing_area 
                              width_sunbathing_area 
                              height_main_pool_remove 
                              height_sunbathing_area_level 
                              conversion_factor ≈ 93200 := 
by
  sorry

end water_removal_l385_385706


namespace frog_position_after_20_jumps_l385_385049

/-- 
The frog starts at position 1, and for each subsequent nth jump,
it jumps n steps in a clockwise circular manner among 6 positions.
Prove that after 20 jumps, the frog returns to position 1.
-/
theorem frog_position_after_20_jumps : 
  let sum_of_steps := (20 * 21) / 2 in
  sum_of_steps % 6 = 0 := 
by
  let sum_of_steps := (20 * 21) / 2
  have h : sum_of_steps = 210 := by simp [sum_of_steps]
  have h_mod : 210 % 6 = 0 := by norm_num
  exact h_mod

end frog_position_after_20_jumps_l385_385049


namespace problem_part1_l385_385010

theorem problem_part1 :
  (-((1 : ℚ) / 3))^(-2) + real.cbrt (-8) + abs (real.sqrt 3 - 2) - ((1 / (real.sqrt 2023 - 1)))^0 + 2 * real.sin (real.pi / 3) = 8 := by
  sorry

end problem_part1_l385_385010


namespace max_2x_plus_y_value_l385_385919

open Real

def on_ellipse (P : ℝ × ℝ) : Prop := 
  (P.1^2 / 4 + P.2^2 = 1)

def max_value_2x_plus_y (P : ℝ × ℝ) (h : on_ellipse P) : ℝ := 
  2 * P.1 + P.2

theorem max_2x_plus_y_value (P : ℝ × ℝ) (h : on_ellipse P):
  ∃ (m : ℝ), max_value_2x_plus_y P h = m ∧ m = sqrt 17 :=
sorry

end max_2x_plus_y_value_l385_385919


namespace correctness_of_statements_l385_385076

theorem correctness_of_statements :
  (¬(∀ x : ℝ, x^2 + x + 1 > 0) ↔ ∃ x : ℝ, x^2 + x + 1 ≤ 0) = false ∧
  (¬(ab = 0 → a = 0 ∨ b = 0) ↔ (ab ≠ 0 → a ≠ 0 ∧ b ≠ 0)) = true ∧
  (∀ f : ℝ → ℝ, (∀ x, f(x - 1) = f(1 - (x - 1)) → f x = -f (-x))) = true ∧
  (∀ f : ℝ → ℝ, (∀ x, f(x + 1) = f(1 - x) → ∀ x, f(x + 1) = f(1 - x) ↔ x ≠ 1)) = false :=
by sorry

end correctness_of_statements_l385_385076


namespace problem1_f_x_linear_problem2_f_x_l385_385743

-- Problem 1 statement: Prove f(x) = 2x + 7 given conditions
theorem problem1_f_x_linear (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 2 * x + 7)
  (h2 : ∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17) : 
  ∀ x, f x = 2 * x + 7 :=
by sorry

-- Problem 2 statement: Prove f(x) = 2x - 1/x given conditions
theorem problem2_f_x (f : ℝ → ℝ) 
  (h1 : ∀ x, 2 * f x + f (1 / x) = 3 * x) : 
  ∀ x, f x = 2 * x - 1 / x :=
by sorry

end problem1_f_x_linear_problem2_f_x_l385_385743


namespace factorize_x_squared_minus_1_l385_385100

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by
  -- Proof goes here
  sorry

end factorize_x_squared_minus_1_l385_385100


namespace min_value_parabola_MF_MP_equals_7_l385_385161

-- Define the conditions
def point (x y : ℝ) := (x, y)
def focus : (ℝ × ℝ) := point 0 4
def parabola (p : ℝ) (x y : ℝ) := x^2 = 2 * p * y
def point_on_parabola (p : ℝ) (x y : ℝ) := parabola p x y
def directrix (p : ℝ) : ℝ := -p / 2
def distance (a b : ℝ × ℝ) : ℝ := real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- Define the problem
def min_MF_MP (p : ℝ) (P focus : ℝ × ℝ) (x y : ℝ) (h : parabola p x y) : ℝ :=
  let M := point x y in
  let F := focus in
  distance M F + distance M P

-- State the theorem
theorem min_value_parabola_MF_MP_equals_7 :  
  ∃ M : point ℝ ℝ, ∃ p : ℝ, p > 0 ∧ focus = point 0 4 ∧ point_on_parabola p M.1 M.2 ∧
  P = point 2 3 ∧ min_MF_MP p P focus M.1 M.2 (point_on_parabola p M.1 M.2) = 7 :=
sorry

end min_value_parabola_MF_MP_equals_7_l385_385161


namespace hotel_breakfast_problem_l385_385774

noncomputable def total_rolls : ℕ := 12
noncomputable def guests : ℕ := 3
noncomputable def roll_types : ℕ := 4
noncomputable def rolls_per_type : ℕ := 3
noncomputable def rolls_per_guest : ℕ := 4

theorem hotel_breakfast_problem :
  let m := 72 in
  let n := 1925 in
  m + n = 1997 :=
by
  -- the proof steps would be here
  sorry

end hotel_breakfast_problem_l385_385774


namespace percentage_for_x_plus_y_l385_385571

theorem percentage_for_x_plus_y (x y : Real) (P : Real) 
  (h1 : 0.60 * (x - y) = (P / 100) * (x + y)) 
  (h2 : y = 0.5 * x) : 
  P = 20 := 
by 
  sorry

end percentage_for_x_plus_y_l385_385571


namespace cosine_210_l385_385843

noncomputable def Q := rotate 210 (1, 0)
noncomputable def E := foot Q (1, 0) (-1, 0)

theorem cosine_210 : cos 210 = - (sqrt 3/2) :=
sorry

end cosine_210_l385_385843


namespace sum_first_10_terms_arithmetic_sequence_l385_385265

-- Define the first term and the sum of the second and sixth terms as given conditions
def a1 : ℤ := -2
def condition_a2_a6 (a2 a6 : ℤ) : Prop := a2 + a6 = 2

-- Define the general term 'a_n' of the arithmetic sequence
def a_n (a1 d : ℤ) (n : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum 'S_n' of the first 'n' terms of the arithmetic sequence
def S_n (a1 d : ℤ) (n : ℤ) : ℤ := n * (a1 + ((n - 1) * d) / 2)

-- The theorem statement to prove that S_10 = 25 given the conditions
theorem sum_first_10_terms_arithmetic_sequence 
  (d a2 a6 : ℤ) 
  (h1 : a2 = a_n a1 d 2) 
  (h2 : a6 = a_n a1 d 6)
  (h3 : condition_a2_a6 a2 a6) : 
  S_n a1 d 10 = 25 := 
by
  sorry

end sum_first_10_terms_arithmetic_sequence_l385_385265


namespace inequality_proof_l385_385510

variables {n : ℕ} (r s t u v : ℕ → ℝ)
variables (hv : ∀ i, 0 < r i ∧ 0 < s i ∧ 0 < t i ∧ 0 < u i ∧ 0 < v i) 

noncomputable def R (n : ℕ) (r : ℕ → ℝ) := (∑ i in finset.range n, r i) / n
noncomputable def S (n : ℕ) (s : ℕ → ℝ) := (∑ i in finset.range n, s i) / n
noncomputable def T (n : ℕ) (t : ℕ → ℝ) := (∑ i in finset.range n, t i) / n
noncomputable def U (n : ℕ) (u : ℕ → ℝ) := (∑ i in finset.range n, u i) / n
noncomputable def V (n : ℕ) (v : ℕ → ℝ) := (∑ i in finset.range n, v i) / n

noncomputable def product_quotients (r s t u v : ℕ → ℝ) (n : ℕ) :=
  ∏ i in finset.range n, (r i * s i * t i * u i * v i + 1) / (r i * s i * t i * u i * v i - 1)

noncomputable def quotient_mean (R S T U V : ℝ) :=
  (R * S * T * U * V + 1) / (R * S * T * U * V - 1)

theorem inequality_proof (n : ℕ) (r s t u v : ℕ → ℝ)
  (hv : ∀ i, 1 < r i ∧ 1 < s i ∧ 1 < t i ∧ 1 < u i ∧ 1 < v i) :
  product_quotients r s t u v n ≥ (quotient_mean (R n r) (S n s) (T n t) (U n u) (V n v)) ^ n :=
by {
  sorry
}

end inequality_proof_l385_385510


namespace best_model_l385_385019

-- Define the data points for the sales volume
def sales_volume (x : ℕ) : ℕ :=
  match x with
  | 1 => 100
  | 2 => 200
  | 3 => 400
  | 4 => 790
  | _ => 0  -- We only care about x = 1, 2, 3, 4 for now

-- Define the candidate function models
def model_A (x : ℕ) : ℕ := 100
def model_B (x : ℕ) : ℕ := 50 * x^2 - 50 * x + 100
def model_C (x : ℕ) : ℕ := 50 * 2^x
def model_D (x : ℕ) : ℕ := 100 * Nat.log2 x + 100

-- The proof problem is to show that model_C best fits the data points.
theorem best_model : (∀ x, x ∈ [1, 2, 3, 4] -> model_C x = sales_volume x) ∧
                     (∀ x, x ∈ [1, 2, 3, 4] -> abs (model_C x - sales_volume x) 
                     ≤ abs (model_A x - sales_volume x)) ∧
                     (∀ x, x ∈ [1, 2, 3, 4] -> abs (model_C x - sales_volume x) 
                     ≤ abs (model_B x - sales_volume x)) ∧
                     (∀ x, x ∈ [1, 2, 3, 4] -> abs (model_C x - sales_volume x)
                     ≤ abs (model_D x - sales_volume x)) :=
by
  sorry

end best_model_l385_385019


namespace unripe_oranges_harvest_l385_385198

theorem unripe_oranges_harvest (daily_unripe_oranges: ℕ) (days: ℕ) (harvest_rate: daily_unripe_oranges = 65) (duration: days = 6) :
  daily_unripe_oranges * days = 390 :=
by
  rw [harvest_rate, duration]
  rfl

end unripe_oranges_harvest_l385_385198


namespace g_inv_f17_l385_385989

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom f_inv_g : ∀ x : ℝ, f_inv (g x) = x^4 - 1
axiom g_has_inv : g_inv = function.inverseable g

theorem g_inv_f17 : g_inv (f 17) = real.root 18 (4 : ℝ) :=
  sorry

end g_inv_f17_l385_385989


namespace B_is_midpoint_of_PQ_l385_385190

noncomputable def midpoint_of_PQ
  (ABC : Type)
  [triangle ABC]
  (incircle: ABC → Circle)
  (D E F : ABC)
  (tangent_D : TangentPoint incircle ABC BC D)
  (tangent_E : TangentPoint incircle ABC CA E)
  (tangent_F : TangentPoint incircle ABC AB F)
  (P Q : ABC)
  (perp_F_to_EF : PerpendicularFrom F E F P)
  (perp_D_to_ED : PerpendicularFrom D E D Q)
  (B : ABC) : Prop :=
  midpoint B P Q

theorem B_is_midpoint_of_PQ
  (ABC : Type)
  [triangle ABC]
  (incircle: ABC → Circle)
  (D E F : ABC)
  (tangent_D : TangentPoint incircle ABC BC D)
  (tangent_E : TangentPoint incircle ABC CA E)
  (tangent_F : TangentPoint incircle ABC AB F)
  (P Q : ABC)
  (perp_F_to_EF : PerpendicularFrom F E F P)
  (perp_D_to_ED : PerpendicularFrom D E D Q)
  (B : ABC) :
  midpoint_of_PQ ABC incircle D E F tangent_D tangent_E tangent_F P Q perp_F_to_EF perp_D_to_ED B :=
  sorry

end B_is_midpoint_of_PQ_l385_385190


namespace range_of_b_decreasing_l385_385211

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := - (1 / 2) * x ^ 2 + b * Real.log(x + 2)

theorem range_of_b_decreasing :
  (∀ x : ℝ, -1 < x → f x b ≤ f x (b + ε) → ε < 0) → (b ∈ Icc (-∞) (-1)) :=
sorry

end range_of_b_decreasing_l385_385211


namespace factorize_x_squared_minus_one_l385_385093

theorem factorize_x_squared_minus_one (x : Real) : (x^2 - 1) = (x + 1) * (x - 1) :=
sorry

end factorize_x_squared_minus_one_l385_385093


namespace distribution_percent_l385_385756

noncomputable def percent_less_than_mean_plus_sd (m d : ℝ) (f : ℝ → ℝ) : ℝ :=
  if h : 64 / 100 then 82 / 100 else 0

theorem distribution_percent (m d : ℝ) (f : ℝ → ℝ) 
  (h_sym : ∀ x, f (m + x) = f (m - x)) 
  (h64 : ∫ x in m - d..m + d, f x = 64 / 100) : 
  percent_less_than_mean_plus_sd m d f = 82 / 100 :=
by
  sorry

end distribution_percent_l385_385756


namespace zoe_overall_accuracy_rate_l385_385636

theorem zoe_overall_accuracy_rate 
  (total_problems x : ℕ)
  (chloe_ind_accuracy zoe_ind_accuracy overall_accuracy : ℚ)
  (h₁ : chloe_ind_accuracy = 0.80)
  (h₂ : overall_accuracy = 0.88)
  (h₃ : zoe_ind_accuracy = 0.90) :
  let y := 0.96 in
  let Z := ((0.45 : ℚ) + 0.48) in
  Z * 100 = 93 := 
by
  sorry

end zoe_overall_accuracy_rate_l385_385636


namespace min_value_l385_385539

noncomputable def min_res (a b c : ℝ) : ℝ := 
  if h : (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a^2 + 4 * b^2 + 9 * c^2 = 4 * b + 12 * c - 2) 
  then (1 / a + 2 / b + 3 / c) 
  else 0

theorem min_value (a b c : ℝ) : (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a^2 + 4 * b^2 + 9 * c^2 = 4 * b + 12 * c - 2) 
    → min_res a b c = 6 := 
sorry

end min_value_l385_385539


namespace trajectory_of_center_l385_385146

-- The main problem statement with conditions and the desired equation
theorem trajectory_of_center {A B C M : Point} (hA_on_circle : A ∈ circle (0, 0) 16)
  (hB_on_circle : B ∈ circle (0, 0) 16) (hAB : distance A B = 6)
  (hCircle_M_through_C : C ∈ circle (center_of_diametric_circle A B, distance_between_center_and_C A B C)):
  let C := (1, -1) in
  let center_of_diametric_circle := midpoint A B in
  let distance_between_center_and_C := λ A B C, distance center_of_diametric_circle C = 3,
    set_of (λ M, (M.1 - 1)^2 + (M.2 + 1)^2 = 9).nonempty :=
sorry

end trajectory_of_center_l385_385146


namespace vec_magnitude_proof_l385_385564

noncomputable def vec_a : ℝ × ℝ := (2, 0)
noncomputable def vec_b_norm : ℝ := 1 -- This denotes |vec_b| = 1
noncomputable def angle_rad : ℝ := real.pi / 3 -- This is 60 degrees in radians

theorem vec_magnitude_proof (vec_b : ℝ × ℝ)
  (h1 : real.sqrt (vec_b.1 ^ 2 + vec_b.2 ^ 2) = vec_b_norm)
  (h2 : (2 * vec_b.1 + 4 * vec_b.2) = 4 * (real.cos angle_rad)) :
  real.sqrt ((vec_a.1 + 2 * vec_b.1) ^ 2 + (vec_a.2 + 2 * vec_b.2) ^ 2) = 2 * real.sqrt 3 := 
by 
  sorry

end vec_magnitude_proof_l385_385564


namespace train_length_l385_385434

theorem train_length 
  (speed_kmh : ℝ) (time_s : ℝ)
  (h_speed : speed_kmh = 50)
  (h_time : time_s = 9) : 
  let speed_ms := (speed_kmh * 1000) / 3600 in
  let length_m := speed_ms * time_s in
  length_m = 125 :=
by
  -- Speed of the train in m/s
  have speed_ms_def : speed_ms = (speed_kmh * 1000) / 3600 := by rfl
  -- Calculation of the length of the train
  have length_m_def : length_m = speed_ms * time_s := by rfl
  -- Substituting values
  rw [h_speed, h_time, speed_ms_def, length_m_def]
  -- Converting speed to m/s
  have speed_calc : (50 * 1000) / 3600 = 125 / 9 := sorry
  -- Final calculation
  -- Simplifying the final expression
  rw [speed_calc]
  have final_calc : (125 / 9) * 9 = 125 := by sorry
  exact final_calc

end train_length_l385_385434


namespace product_of_two_numbers_l385_385343

noncomputable def leastCommonMultiple (a b : ℕ) : ℕ :=
  Nat.lcm a b

noncomputable def greatestCommonDivisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

theorem product_of_two_numbers (a b : ℕ) :
  leastCommonMultiple a b = 36 ∧ greatestCommonDivisor a b = 6 → a * b = 216 := by
  sorry

end product_of_two_numbers_l385_385343


namespace diagonals_bisect_each_other_l385_385654

-- Define a structure for a quadrilateral with its vertices
structure Quadrilateral :=
(A B C D : Point)

-- Define a structure for midpoints of quadrilateral sides
structure Midpoints (Q : Quadrilateral) :=
(A1 : midpoint Q.A Q.B)
(B1 : midpoint Q.B Q.C)
(C1 : midpoint Q.C Q.D)
(D1 : midpoint Q.D Q.A)

-- The main theorem
theorem diagonals_bisect_each_other (Q : Quadrilateral) (M : Midpoints Q) : 
  bisects (line M.A1 M.C1) (line M.B1 M.D1) :=
sorry

end diagonals_bisect_each_other_l385_385654


namespace find_d_l385_385681

noncomputable def direction_vector_solution : ℝ × ℝ :=
  let v := (2 : ℝ, -1 : ℝ)
  let d := (5 / Real.sqrt 29, 2 / Real.sqrt 29)
  d

-- Statement of the theorem
theorem find_d (x y t : ℝ) (h1 : y = (2 * x + 3) / 5)
  (h2 : ∃ v : ℝ × ℝ, (x, y) = (v.1 + t * (5 / Real.sqrt 29), v.2 + t * (2 / Real.sqrt 29)))
  (h3 : ∀ x, x ≥ 2 → (Real.sqrt ((x - 2) ^ 2 + ((2 * x + 3) / 5 + 1) ^ 2)) = t) :
  direction_vector_solution = (5 / Real.sqrt 29, 2 / Real.sqrt 29) := 
by 
  sorry

end find_d_l385_385681


namespace polynomial_value_l385_385615

noncomputable def p (x : ℝ) : ℝ :=
  (x - 1) * (x - 2) * (x - 3) * (x - 4) + 24 * x

theorem polynomial_value :
  (p 1 = 24) ∧ (p 2 = 48) ∧ (p 3 = 72) ∧ (p 4 = 96) →
  p 0 + p 5 = 168 := 
by
  sorry

end polynomial_value_l385_385615


namespace bus_stoppage_time_l385_385391

theorem bus_stoppage_time
  (speed_without_stoppages : ℝ)
  (speed_with_stoppages : ℝ)
  (H1 : speed_without_stoppages = 60)
  (H2 : speed_with_stoppages = 40)
  : (60 * (1 - speed_with_stoppages / speed_without_stoppages) * 60 = 20) :=
begin
  sorry
end

end bus_stoppage_time_l385_385391


namespace factorize_x_squared_minus_one_l385_385092

theorem factorize_x_squared_minus_one (x : Real) : (x^2 - 1) = (x + 1) * (x - 1) :=
sorry

end factorize_x_squared_minus_one_l385_385092


namespace faster_speed_l385_385029

variable (v : ℝ)
variable (distance fasterDistance speed time : ℝ)
variable (h_distance : distance = 24)
variable (h_speed : speed = 4)
variable (h_fasterDistance : fasterDistance = distance + 6)
variable (h_time : time = distance / speed)

theorem faster_speed (h : 6 = fasterDistance / v) : v = 5 :=
by
  sorry

end faster_speed_l385_385029


namespace max_value_of_f_l385_385496

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

theorem max_value_of_f : ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ 2 := 
by
  sorry

end max_value_of_f_l385_385496


namespace compare_sqrts_l385_385464

theorem compare_sqrts : sqrt (8) - sqrt (6) < sqrt (7) - sqrt (5) := by
  have h8 := Real.sqrt_pos.mpr (by linarith)
  have h6 := Real.sqrt_pos.mpr (by linarith)
  have h7 := Real.sqrt_pos.mpr (by linarith)
  have h5 := Real.sqrt_pos.mpr (by linarith)
  calc
    sqrt (8) - sqrt (6) < sqrt (7) - sqrt (5) : sorry

end compare_sqrts_l385_385464


namespace y_equals_x_l385_385575

theorem y_equals_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x = 1 + 1 / y) (h2 : y = 1 + 1 / x) : y = x :=
sorry

end y_equals_x_l385_385575


namespace average_student_headcount_l385_385367

theorem average_student_headcount :
  let a := 10700
  let b := 11300
  let c := 11200
  (a + b + c) / 3 = 11067 :=
by
  let a := 10700
  let b := 11300
  let c := 11200
  have h : (a + b + c : ℝ) / 3 = 11066.6666666667 := by sorry
  have round_eq : Real.round (11066.6666666667) = 11067 := by sorry
  show Real.round ((a + b + c) / 3) = 11067 from by
    rw [h, round_eq]
    exact Real.round_coe,
  sorry

end average_student_headcount_l385_385367


namespace change_received_correct_l385_385628

-- Define the prices of items and the amount paid
def price_hamburger : ℕ := 4
def price_onion_rings : ℕ := 2
def price_smoothie : ℕ := 3
def amount_paid : ℕ := 20

-- Define the total cost and the change received
def total_cost : ℕ := price_hamburger + price_onion_rings + price_smoothie
def change_received : ℕ := amount_paid - total_cost

-- Theorem stating the change received
theorem change_received_correct : change_received = 11 := by
  sorry

end change_received_correct_l385_385628


namespace block_subset_weight_l385_385734

theorem block_subset_weight (n : ℕ) (weights : fin n → ℝ) (h_all_at_least_1 : ∀ i, 1 ≤ weights i)
  (h_total_weight : ∑ i, weights i = 2 * n) (r : ℝ) (h_r : 0 ≤ r ∧ r ≤ 2 * n - 2) :
  ∃ (I : finset (fin n)), r ≤ ∑ i in I, weights i ∧ ∑ i in I, weights i ≤ r + 2 :=
  sorry

end block_subset_weight_l385_385734


namespace carli_charlie_flute_ratio_l385_385059

theorem carli_charlie_flute_ratio :
  let charlie_flutes := 1
  let charlie_horns := 2
  let charlie_harps := 1
  let carli_horns := charlie_horns / 2
  let total_instruments := 7
  ∃ (carli_flutes : ℕ), 
    (charlie_flutes + charlie_horns + charlie_harps + carli_flutes + carli_horns = total_instruments) ∧ 
    (carli_flutes / charlie_flutes = 2) :=
by
  sorry

end carli_charlie_flute_ratio_l385_385059


namespace product_not_100_l385_385207

theorem product_not_100 (A B C D E : ℚ × ℚ) (hA : A = (10, 10)) (hB : B = (20, -5)) (hC : C = (-4, -25)) (hD : D = (50, 2)) (hE : E = (5/2, 40)) :
  A.1 * A.2 = 100 ∧
  B.1 * B.2 ≠ 100 ∧
  C.1 * C.2 = 100 ∧
  D.1 * D.2 = 100 ∧
  E.1 * E.2 = 100 :=
begin
  simp [hA, hB, hC, hD, hE],
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
end

end product_not_100_l385_385207


namespace math_problem_is_n_eq_5_l385_385610

noncomputable def problem_statement (n : ℕ) (a : ℕ → ℕ) :=
  (∀ x : ℕ, (1 + 2 * x)^n = ∑ i in Finset.range(n + 1), a i * x^i) ∧
  (a 3 = 2 * a 2)

theorem math_problem_is_n_eq_5 : ∃ n a, problem_statement n a ∧ n = 5 :=
  sorry

end math_problem_is_n_eq_5_l385_385610


namespace no_such_positive_integer_l385_385901

theorem no_such_positive_integer (n : ℕ) (d : ℕ → ℕ)
  (h₁ : ∃ d1 d2 d3 d4 d5, d 1 = d1 ∧ d 2 = d2 ∧ d 3 = d3 ∧ d 4 = d4 ∧ d 5 = d5) 
  (h₂ : 1 ≤ d 1 ∧ d 1 < d 2 ∧ d 2 < d 3 ∧ d 3 < d 4 ∧ d 4 < d 5)
  (h₃ : ∀ i, 1 ≤ i → i ≤ 5 → d i ∣ n)
  (h₄ : ∀ i, 1 ≤ i → i ≤ 5 → ∀ j, i ≠ j → d i ≠ d j)
  (h₅ : ∃ x, 1 + (d 2)^2 + (d 3)^2 + (d 4)^2 + (d 5)^2 = x^2) :
  false :=
sorry

end no_such_positive_integer_l385_385901


namespace factor_determination_l385_385569

-- Define the problem statement
theorem factor_determination (d : ℝ) :
  (λ x : ℝ, d * x^3 + 25 * x^2 - 5 * d * x + 45) (-5) = 0 → d = 6.7 :=
by
  intro h
  sorry

end factor_determination_l385_385569


namespace rectangular_prism_diagonal_l385_385169

theorem rectangular_prism_diagonal 
  (a b c : ℝ)
  (h1 : 2 * a * b + 2 * b * c + 2 * c * a = 11)
  (h2 : 4 * (a + b + c) = 24) :
  (a^2 + b^2 + c^2 = 25) :=
by {
  -- Sorry to skip the proof steps
  sorry
}

end rectangular_prism_diagonal_l385_385169


namespace sum_c_d_eq_24_l385_385816

theorem sum_c_d_eq_24 (c d : ℕ) (h_pos_c : c > 0) (h_pos_d : d > 1) (h_max_power : c^d < 500 ∧ ∀ ⦃x y : ℕ⦄, x^y < 500 → x^y ≤ c^d) : c + d = 24 :=
sorry

end sum_c_d_eq_24_l385_385816


namespace range_of_a_l385_385952

noncomputable def f (x a : ℝ) : ℝ :=
  (1/2) * x^2 - (a + 2) * x + 2 * a * Real.log x + 1

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ set.Ioo 4 6, deriv (λ x, f x a) x = 0) ↔ a ∈ set.Ioo 4 6 :=
by
  sorry

end range_of_a_l385_385952


namespace number_of_b_values_l385_385885

noncomputable def valid_b_values : ℤ → Prop :=
λ b, 
  let discriminant := b^2 - 20 in
  ∀ x : ℤ, x^2 + b * x + 5 ≤ 0 → 
    (let root1 := (-(b) - Real.sqrt discriminant) / 2,
         root2 := (-(b) + Real.sqrt discriminant) / 2 in
     (floor root1 = x ∨ floor root2 = x) ∧ (ceil root1 = x ∨ ceil root2 = x))

theorem number_of_b_values :
  ∃! (b: ℤ), (b = 10 ∨ b = -10) ∧ (valid_b_values b) :=
by sorry

end number_of_b_values_l385_385885


namespace c_pays_l385_385789

-- Defining the problem condition
variables (cost_price_A : ℝ) (profit_rate_A : ℝ) (profit_rate_B : ℝ)

-- These are the given conditions from the problem
def condition1 := cost_price_A = 150
def condition2 := profit_rate_A = 0.20
def condition3 := profit_rate_B = 0.25

-- We need to prove that the selling price for C is 225
theorem c_pays (h₁ : condition1) (h₂ : condition2) (h₃ : condition3) : 
  let selling_price_A := cost_price_A + profit_rate_A * cost_price_A in
  let cost_price_B := selling_price_A in
  let profit_B := profit_rate_B * cost_price_B in
  let selling_price_B := cost_price_B + profit_B in
  selling_price_B = 225 :=
by
  sorry

end c_pays_l385_385789


namespace laborers_absent_l385_385762

theorem laborers_absent (x : ℕ) : 
  let total_planned_labor_days := 15 * 9
  let actual_labor_days := (15 - x) * 15
  (total_planned_labor_days = actual_labor_days) → x = 6 := 
by
  intros
  let total_planned_labor_days : ℕ := 15 * 9
  let actual_labor_days : ℕ := (15 - x) * 15
  have h : total_planned_labor_days = actual_labor_days := by assumption
  let eq1 : 135 = 225 - 15 * x := h
  sorry

end laborers_absent_l385_385762


namespace min_positive_period_monotone_increasing_interval_range_in_interval_l385_385167

open Real

noncomputable def f (x : ℝ) : ℝ := sin(2 * x)^2 + sqrt 3 * sin(2 * x) * sin(2 * x + π / 2)

theorem min_positive_period (ω : ℝ) (h : ω > 0) : ω = 2 :=
by sorry

theorem monotone_increasing_interval (k : ℤ) : 
  ∀ x, (k * π / 2 - π / 12 ≤ x) ∧ (x ≤ k * π / 2 + π / 6) ↔ 
       (f (x - 0)) ≤ (f (x + 0)) :=
by sorry

theorem range_in_interval : 
  ∃ l u, l = 0 ∧ u = 3 / 2 ∧ (∀ y, 0 ≤ y ∧ y ≤ π / 3 → (f y) ≥ l ∧ (f y) ≤ u) :=
by sorry

end min_positive_period_monotone_increasing_interval_range_in_interval_l385_385167


namespace min_XP_PY_l385_385191

theorem min_XP_PY (O A B P X Y : Point) :
  (OnRay O A X) ∧ (OnRay O B Y) ∧ (Between P OA OB) ∧ (OnRay X P Y) ∧ (P ≠ O) ∧ (O ≠ A) ∧ (O ≠ B) -> 
  XP * PY = minimal_value → OX = OY :=
by
  sorry

end min_XP_PY_l385_385191


namespace cost_per_mile_eq_l385_385196

theorem cost_per_mile_eq :
  ( ∀ x : ℝ, (65 + 0.40 * 325 = x * 325) → x = 0.60 ) :=
by
  intros x h
  have eq1 : 65 + 0.40 * 325 = 195 := by sorry
  rw [eq1] at h
  have eq2 : 195 = 325 * x := h
  field_simp at eq2
  exact eq2

end cost_per_mile_eq_l385_385196


namespace invention_reasoning_l385_385566

-- Define your propositions or conditions
def invention (x y : Type) (f : x → y) := true

-- The main theorem stating the conclusion
theorem invention_reasoning : 
  (∃ (x y : Type) (f : x → y), invention x y f) → 
  (Submarine := Fish) →
  (reasoning = Analogy) :=
by
  sorry

end invention_reasoning_l385_385566


namespace correct_proposition_converse_l385_385876

-- Defining the proposition 1
def converse_statement (x y : ℝ) : Prop := (x + y = 0) → (x = -y)

-- Proof problem.
theorem correct_proposition_converse : ∀ (x y : ℝ), converse_statement x y :=
by
  intro x y
  unfold converse_statement
  intro h
  rw ← h
  sorry

end correct_proposition_converse_l385_385876


namespace union_sets_l385_385219

def setA : Set ℝ := { x | abs (x - 1) < 3 }
def setB : Set ℝ := { x | x^2 - 4 * x < 0 }

theorem union_sets :
  setA ∪ setB = { x : ℝ | -2 < x ∧ x < 4 } :=
sorry

end union_sets_l385_385219


namespace inequality_sqrt_sum_le_e_l385_385640

open Real

theorem inequality_sqrt_sum_le_e (a : ℕ → ℝ) (n : ℕ) (h : ∀ k, 1 ≤ k → k ≤ n → 0 < a k) :
  ∑ k in Finset.range(n + 1), real.root (∏ i in Finset.range(k+1), a i) (k+1) ≤ 
  exp 1 * ∑ k in Finset.range(n + 1), a k :=
by sorry

end inequality_sqrt_sum_le_e_l385_385640


namespace bike_price_l385_385254

theorem bike_price (x : ℝ) (h : 0.20 * x = 240) : x = 1200 :=
by
  sorry

end bike_price_l385_385254


namespace min_value_x_plus_2_div_x_minus_2_l385_385511

theorem min_value_x_plus_2_div_x_minus_2 (x : ℝ) (h : x > 2) : 
  ∃ m, m = 2 + 2 * Real.sqrt 2 ∧ x + 2/(x-2) ≥ m :=
by sorry

end min_value_x_plus_2_div_x_minus_2_l385_385511


namespace factorial_divides_product_of_powers_l385_385651

theorem factorial_divides_product_of_powers (n : ℕ) :
  (factorial n) ∣ (∏ k in finset.range n, (2^n - 2^k)) :=
sorry

end factorial_divides_product_of_powers_l385_385651


namespace inequality_hold_for_positive_reals_l385_385653

theorem inequality_hold_for_positive_reals :
  ∀ (a : ℕ → ℝ) (M P : ℕ), (0 < M) → (0 < P) → 
  (∀ i, 0 < a i) → 
  M * (∑ i in Finset.range M, (a i) ^ P) 
  ≤ (∑ i in Finset.range M, (a i) ^ (P + 1)) * (∑ i in Finset.range M, (a i) ^ (-1)) := 
by
  intros a M P hM hP ha
  sorry

end inequality_hold_for_positive_reals_l385_385653


namespace eccentricity_is_half_ratio_of_areas_l385_385454

variable (a b c : ℝ)
variable (x y x1 y1 x2 y2 : ℝ)
variable (k : ℝ)
variable (F G D E O A B : (ℝ × ℝ))
variable (S1 S2 : ℝ)

# Check the conditions
def conditions :=
  a > b ∧ b > 0 ∧ -- a > b > 0
  let e := c / a in
  b = sqrt 3 * c ∧
  a^2 = b^2 + c^2 ∧ -- Implicitly a^2 = 4c^2
  e = 1/2 ∧
  -- Constraints for the midpoint G
  let G' := (-(4 * c * k^2) / (4 * k^2 + 3), 3 * c * k / (4 * k^2 + 3)) in
  G = G' ∧
  -- More implicit definitions follow (same as in step 5 of the original translation)
  S1 = abs((G.1 - D.1) * (G.2 - F.2) - (G.1 - F.1) * (G.2 - D.2)) / 2 ∧
  S2 = abs((O.1 - E.1) * (O.2 - D.2) - (O.1 - D.1) * (O.2 - E.2)) / 2 ∧
  k ≠ 0 -- Assume k is not zero since the line A B should not be vertical

theorem eccentricity_is_half (h : conditions a b c x y x1 y1 x2 y2 k F G D E O A B S1 S2) : c / a = 1 / 2 :=
by
  sorry

theorem ratio_of_areas (h : conditions a b c x y x1 y1 x2 y2 k F G D E O A B S1 S2) : S1 / S2 > 9 :=
by
  sorry

end eccentricity_is_half_ratio_of_areas_l385_385454


namespace prism_circumsphere_surface_area_l385_385141

-- We define a triangle with given side lengths and midpoints
structure Triangle where
  A1 A2 A3 : Type 
  side_lengths : A1 → A2 → A3 → ℕ → ℕ → ℕ
  A1A2 : side_lengths A1 A2 _ = 8
  A2A3 : side_lengths A2 A3 _ = 10
  A3A1 : side_lengths A3 A1 _ = 12

-- We define midpoints B, C, D and the triangular prism structure
structure Prism where
  triangle : Triangle
  B C D : Type 
  is_midpoint : (B → C → D) → Bool

-- Circumsphere of the given prism
structure Circumsphere where
  prism : Prism
  radius : ℝ
  surface_area : ℝ

-- The statement to be proved
theorem prism_circumsphere_surface_area (prism : Prism) : 
  prism.Circumsphere.surface_area = (77 * Real.pi) / 2 :=
sorry

end prism_circumsphere_surface_area_l385_385141


namespace possible_values_of_D_plus_E_l385_385887

theorem possible_values_of_D_plus_E 
  (D E : ℕ) 
  (hD : 0 ≤ D ∧ D ≤ 9) 
  (hE : 0 ≤ E ∧ E ≤ 9) 
  (hdiv : (D + 8 + 6 + 4 + E + 7 + 2) % 9 = 0) : 
  D + E = 0 ∨ D + E = 9 ∨ D + E = 18 := 
sorry

end possible_values_of_D_plus_E_l385_385887


namespace parabola_directrix_l385_385185

-- Define the conditions
def parabola (p : ℝ) : Prop := ∀ (x y : ℝ), y^2 = 2 * p * x
def directrix (x : ℝ) : Prop := x = -2

-- The statement to prove
theorem parabola_directrix (p : ℝ) (h : directrix (-2)): parabola p ∧ ¬ (p = 4):=

begin
  sorry
end

end parabola_directrix_l385_385185


namespace find_principal_l385_385034

-- Define the conditions
variables (P R : ℝ)

-- Define the original and increased interest rates over 15 years
def original_interest := P * R * 15 / 100
def increased_interest := P * (R + 8) * 15 / 100

-- State the theorem to prove
theorem find_principal (h : increased_interest = original_interest + 2750) : P = 2291.67 :=
by 
  -- The proof will go here
  sorry

end find_principal_l385_385034


namespace opposite_of_neg_three_l385_385685

theorem opposite_of_neg_three : -(-3) = 3 := by
  sorry

end opposite_of_neg_three_l385_385685


namespace triangle_angle_B_is_60_l385_385189

theorem triangle_angle_B_is_60
  (a b c : ℝ)
  (A B C : ℝ)
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (triangle_ABC_conds : ∀ x : ℝ, x = a + b + c ∧ A + B + C = 180)
  (equation_condition : c / (a + b) + a / (b + c) = 1) :
  B = 60 :=
begin
  sorry
end

end triangle_angle_B_is_60_l385_385189


namespace ellipse_equation_area_l385_385928

-- Define the foci and points on the ellipse
def F1 := (-2 : ℝ, 0 : ℝ)
def F2 := (2 : ℝ, 0 : ℝ)
def P := (2 : ℝ, 5/3 : ℝ)

-- Define the variables and conditions
variables {a b : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : F1.1 = -2 ∧ F1.2 = 0) (h4 : F2.1 = 2 ∧ F2.2 = 0)
          (h_pass : (2 : ℝ, 5/3 : ℝ) = P)

-- equation of the ellipse that needs to be proved
noncomputable def ell_eq (x y : ℝ) := x^2 / 9 + y^2 / 5 = 1

-- Define the area calculation
def distance (A B : (ℝ × ℝ)) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
def area_triangle (d1 d2 : ℝ) := (1/2) * d1 * d2 * real.sin (real.pi / 3)

-- proving the equation of the ellipse and the area of the triangle
theorem ellipse_equation_area (h_ellipse : ell_eq 2 (5 / 3))
    (h_angle : ∃ P, ell_eq P.1 P.2 ∧ ∠ (F1, P, F2) = real.pi / 3):
  (9 := a^2 ∧ 5 := b^2 ∧ 
   distance (F1, F2) = 4 ∧
   distance (F1, P) * distance (F2, P) = 20 / 3 ∧
   area_triangle (distance (F1, P)) (distance (F2, P)) = 5 * real.sqrt 3 / 3) :=
sorry

end ellipse_equation_area_l385_385928


namespace number_of_scalene_triangles_with_perimeter_less_than_seventeen_l385_385905

-- Define the problem conditions and the target proof

def is_valid_scalene_triangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a + b > c ∧ b + c > a ∧ c + a > b

def perimeter_less_than (p : ℕ) (a b c : ℕ) : Prop :=
  a + b + c < p

theorem number_of_scalene_triangles_with_perimeter_less_than_seventeen :
  { (a, b, c) : ℕ × ℕ × ℕ // is_valid_scalene_triangle a b c ∧ perimeter_less_than 17 a b c }.card = 14 :=
sorry

end number_of_scalene_triangles_with_perimeter_less_than_seventeen_l385_385905


namespace bugs_meet_point_l385_385714

theorem bugs_meet_point
  (P Q R S : Type*)
  (d_PQ d_QR d_RP : ℝ) 
  (eq_d_PQ : d_PQ = 7) 
  (eq_d_QR : d_QR = 8) 
  (eq_d_RP : d_RP = 9)
  (crawl_distance : ℝ)
  (half_perimeter : crawl_distance = (d_PQ + d_QR + d_RP) / 2)
  (meet_distance : crawl_distance = 12)
  (start_P : P = S)
  (start_Q : d_PQ + 5 = crawl_distance) :
  ∃ S, (d_PQ + 5 = crawl_distance) :=
by
  sorry

end bugs_meet_point_l385_385714


namespace range_of_y_l385_385991

theorem range_of_y (y : ℝ) (h1: 1 / y < 3) (h2: 1 / y > -4) : y > 1 / 3 :=
by
  sorry

end range_of_y_l385_385991


namespace distance_between_locations_l385_385780

theorem distance_between_locations
  (d_AC d_BC : ℚ)
  (d : ℚ)
  (meet_C : d_AC + d_BC = d)
  (travel_A_B : 150 + 150 + 540 = 840)
  (distance_ratio : 840 / 540 = 14 / 9)
  (distance_ratios : d_AC / d_BC = 14 / 9)
  (C_D : 540 = 5 * d / 23) :
  d = 2484 :=
by
  sorry

end distance_between_locations_l385_385780


namespace woodblocks_per_log_l385_385403

theorem woodblocks_per_log (x: ℕ) :
  (16 * x = 80) → x = 5 :=
by
  intro h,
  have h1 : 16 * x = 80 := h,
  sorry

end woodblocks_per_log_l385_385403


namespace definite_integral_solution_l385_385064

theorem definite_integral_solution :
  ∫ x in 0..1, (sqrt (1 - x^2) + 3 * x) = (Real.pi / 4) + (3 / 2) :=
by
  sorry

end definite_integral_solution_l385_385064


namespace vector_subtraction_l385_385973

variable (a b : ℝ × ℝ)

def vector_calc (a b : ℝ × ℝ) : ℝ × ℝ :=
  (2 * a.1 - b.1, 2 * a.2 - b.2)

theorem vector_subtraction :
  a = (2, 4) → b = (-1, 1) → vector_calc a b = (5, 7) := by
  intros ha hb
  simp [vector_calc]
  rw [ha, hb]
  simp
  sorry

end vector_subtraction_l385_385973


namespace concurrency_of_AP_BQ_CR_l385_385038

theorem concurrency_of_AP_BQ_CR (
  {A B C D E F P Q R G: Type*}
  [triangle ABC]
  [acute_angled_triangle ABC]
  [altitudes A D, B E, C F]
  [perpendicular A P E F, B Q F D, C R D E]
  [feet A P E F, B Q F D, C R D E]):
  concurrent AP BQ CR :=
sorry

end concurrency_of_AP_BQ_CR_l385_385038


namespace find_p_and_q_solution_set_l385_385353

theorem find_p_and_q (p q : ℝ) (h : ∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - p * x - q < 0) : 
  p = 5 ∧ q = -6 :=
sorry

theorem solution_set (p q : ℝ) (h_p : p = 5) (h_q : q = -6) : 
  ∀ x : ℝ, q * x^2 - p * x - 1 > 0 ↔ - (1 / 2) < x ∧ x < - (1 / 3) :=
sorry

end find_p_and_q_solution_set_l385_385353


namespace first_reduction_percentage_l385_385431

theorem first_reduction_percentage 
  (P : ℝ)  -- original price
  (x : ℝ)  -- first day reduction percentage
  (h : P > 0) -- price assumption
  (h2 : 0 ≤ x ∧ x ≤ 100) -- percentage assumption
  (cond : P * (1 - x / 100) * 0.86 = 0.774 * P) : 
  x = 10 := 
sorry

end first_reduction_percentage_l385_385431


namespace opposite_of_neg_three_l385_385683

def opposite (x : Int) : Int := -x

theorem opposite_of_neg_three : opposite (-3) = 3 := by
  -- To be proven using Lean
  sorry

end opposite_of_neg_three_l385_385683


namespace triangle_count_correct_l385_385069

def satisfies_equation (x y : ℕ) : Prop := 37 * x + y = 1853

def area_is_integer (x1 y1 x2 y2 : ℕ) : Prop :=
  let area := (1853 * (x1 - x2)) in area % 2 = 0 ∧ area ≠ 0

def valid_triangle (x1 y1 x2 y2 : ℕ) : Prop :=
  satisfies_equation x1 y1 ∧ satisfies_equation x2 y2 ∧ area_is_integer x1 y1 x2 y2

def num_valid_triangles : ℕ := 625

theorem triangle_count_correct:
  ∃ count : ℕ, count = 625 ∧ (count = nat.card {t : Σ' (x1 y1 x2 y2 : ℕ), valid_triangle x1 y1 x2 y2}) :=
by {
  use 625,
  split,
  rfl,
  sorry
}

end triangle_count_correct_l385_385069


namespace f_x_plus_1_even_f_x_plus_3_odd_l385_385617

variable (R : Type) [CommRing R]

variable (f : R → R)

-- Conditions
axiom condition1 : ∀ x : R, f (1 + x) = f (1 - x)
axiom condition2 : ∀ x : R, f (x - 2) + f (-x) = 0

-- Prove that f(x + 1) is an even function
theorem f_x_plus_1_even (x : R) : f (x + 1) = f (-(x + 1)) :=
by sorry

-- Prove that f(x + 3) is an odd function
theorem f_x_plus_3_odd (x : R) : f (x + 3) = - f (-(x + 3)) :=
by sorry

end f_x_plus_1_even_f_x_plus_3_odd_l385_385617


namespace opposite_of_neg_three_l385_385686

theorem opposite_of_neg_three : -(-3) = 3 := by
  sorry

end opposite_of_neg_three_l385_385686


namespace constant_term_expansion_l385_385335

noncomputable def binomial_constant_term : ℤ :=
let x := 1 in
let expr := x^2 + (1 / x^2 : ℚ) - (2 : ℚ) in
∑ i in range 7, (binomial 6 i) * (-1)^i * x^(6 - 2 * i)

theorem constant_term_expansion : binomial_constant_term = -20 := by
  have h1 : binomial_constant_term = 6.choose 3 * (-1)^3 := sorry,
  have h2 : 6.choose 3 = 20 := by norm_num,
  calc
    binomial_constant_term
        = 6.choose 3 * (-1)^3 : by rw [h1]
    ... = 20 * (-1) : by rw [h2]
    ... = -20 : by norm_num

end constant_term_expansion_l385_385335


namespace sum_first_10_terms_arithmetic_seq_l385_385272

theorem sum_first_10_terms_arithmetic_seq :
  ∀ (a : ℕ → ℤ), (a 1 = -2) → (a 2 + a 6 = 2) →
  (∃ S₁₀, S₁₀ = 10 * a 1 + (10 * (10 - 1)) / 2 * (a 2 - a 1) / (2 - 1) ∧ S₁₀ = 25) :=
begin
  assume a,
  assume h1 : a 1 = -2,
  assume h2 : a 2 + a 6 = 2,
  sorry
end

end sum_first_10_terms_arithmetic_seq_l385_385272


namespace max_value_of_expression_l385_385184

theorem max_value_of_expression (p q r s : ℕ) (h_perm : {p, q, r, s} = {1, 3, 5, 7}) :
  pq + qr + rs + sp ≤ 64 :=
by
  sorry

end max_value_of_expression_l385_385184


namespace vector_subtraction_l385_385971

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

-- Define the expression to be proven
def expression : ℝ × ℝ := ((2 * a.1 - b.1), (2 * a.2 - b.2))

-- The theorem statement
theorem vector_subtraction : expression = (5, 7) :=
by
  -- The proof will be provided here
  sorry

end vector_subtraction_l385_385971


namespace complement_of_angle_is_acute_l385_385446

theorem complement_of_angle_is_acute (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < 90) : 0 < 90 - θ ∧ 90 - θ < 90 :=
by sorry

end complement_of_angle_is_acute_l385_385446


namespace star_composition_l385_385507

def star (a b : ℝ) (h : a ≠ b) : ℝ :=
  (a + b) / (a - b)

theorem star_composition :
  let h1 : 1 ≠ 2 := by norm_num,
      h2 : -3 ≠ 4 := by norm_num
  in star (star 1 2 h1) 4 h2 = -1 / 7 :=
sorry

end star_composition_l385_385507


namespace tennis_tournament_l385_385750

theorem tennis_tournament
  (players : Fin 12 → Type) 
  (played : ∀ i j, i ≠ j → (players i → players j) ∨ (players j → players i))
  (not_lost_all : ∀ i, ∃ j, i ≠ j ∧ (players i → players j)) :
  ∃ (A B C : Fin 12), (players A → players B) ∧ (players B → players C) ∧ (players C → players A) :=
by
  sorry

end tennis_tournament_l385_385750


namespace officer_roles_assignment_l385_385526

theorem officer_roles_assignment :
  let Set := {Alice, Bob, Carol, Dave}
  let Officers := {President, Secretary, Treasurer}
  ∃ (f : Set → Officers), function.bijective f :=
  number_of_ways(Set.card, Officers.card) = 24 :=
sorry

end officer_roles_assignment_l385_385526


namespace number_of_valid_4_digit_numbers_l385_385201

/-- 
Prove that the number of 4-digit positive integers which have four different digits, 
the leading digit is not zero, the integer is a multiple of 5, 7 is the largest digit, 
and the first and last digits are the same is 30. 
-/
theorem number_of_valid_4_digit_numbers : 
  (∃ (count : ℕ), count = 30 ∧ ∀ (n : ℕ),
    1000 ≤ n ∧ n < 10000 ∧ 
    (∀ (d1 d2 d3 d4 : ℕ), 
      n = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ 
      (d1 ≠ d2) ∧ (d1 ≠ d3) ∧ (d1 ≠ d4) ∧ (d2 ≠ d3) ∧ (d2 ≠ d4) ∧ (d3 ≠ d4) ∧ 
      d1 ≠ 0 ∧ (n % 5 = 0) ∧ (max d1 (max d2 (max d3 d4)) = 7) ∧ (d1 = d4))
      ↔ count = 30) :=
begin
  existsi (30 : ℕ),
  split,
  { refl },
  { intros n,
    split,
    { rintro ⟨d1, d2, d3, d4, h_n, h_diff_digits, h_leading_not_zero, h_multiple_of_5, h_max_digit_7, h_first_last_same⟩,
      sorry },
    {  sorry }
  }
end

end number_of_valid_4_digit_numbers_l385_385201


namespace predicted_grandson_height_l385_385028

theorem predicted_grandson_height (teacher_height grandfather_height father_height son_height: ℝ)
  (h_teacher : teacher_height = 176) 
  (h_grandfather : grandfather_height = 173)
  (h_father : father_height = 170)
  (h_son : son_height = 182)
  (linear_relation : ∃ a b, ∀ x, son_height = a * father_height + b) :
  (predicted_height grandson_height : ℝ) := 185 := sorry

end predicted_grandson_height_l385_385028


namespace true_discount_l385_385664

theorem true_discount (BD PV TD : ℝ) (h1 : BD = 36) (h2 : PV = 180) :
  TD = 30 :=
by
  sorry

end true_discount_l385_385664


namespace nine_digit_permutations_divisible_by_2_l385_385202

theorem nine_digit_permutations_divisible_by_2 :
  let digits := [2, 3, 1, 1, 5, 7, 1, 5, 2] in
  (∃ n : ℕ, n = (1 * (8.factorial / (3.factorial * 2.factorial * 1.factorial * 1.factorial))) 
              ∧ digits.permutations.count (λ xs, xs.last = some 2) = n 
              ∧ with_last_digit (l : List ℕ) : Prop := l.last = some 2) := 3360 := 
sorry

end nine_digit_permutations_divisible_by_2_l385_385202


namespace triangle_angles_and_area_l385_385249

variables (a b c : ℝ) (A B C : ℝ)

-- Conditions
def condition1 := (b - a) * (Real.sin B + Real.sin A) = c * (Real.sqrt 3 * Real.sin B - Real.sin C)
def law_of_sines := a / Real.sin A = b / Real.sin B
def law_of_sines2 := b / Real.sin B = c / Real.sin C
def law_of_cosines := b^2 + c^2 - a^2 = Real.sqrt 3 * b * c
def cosA := (b^2 + c^2 - a^2) / (2 * b * c) = Real.sqrt 3 / 2
def A_value := A = Real.pi / 6

noncomputable def area (a c B : ℝ) : ℝ := 1/2 * a * c * Real.sin B

-- Primary statement that proves the solutions
theorem triangle_angles_and_area 
  (h1 : condition1)
  (h2 : law_of_sines)
  (h3 : law_of_sines2) 
  (h4 : law_of_cosines) :
    A = Real.pi / 6 
    ∧ (A = Real.pi / 6 → (a = 2 → (B = Real.pi / 4 → area a (Real.sqrt 2 + Real.sqrt 6) B = Real.sqrt 3 + 1)))
    ∧ (A = Real.pi / 6 → (a = 2 → (c = Real.sqrt 3 * b → area 2 (2 * Real.sqrt 3) (Real.pi / 6) = Real.sqrt 3))) :=
by {
  sorry
}

end triangle_angles_and_area_l385_385249


namespace triangular_number_30_eq_465_perimeter_dots_30_eq_88_l385_385054

-- Definition of the 30th triangular number
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

-- Definition of the perimeter dots for the triangular number
def perimeter_dots (n : ℕ) : ℕ := n + 2 * (n - 1)

-- Theorem to prove the 30th triangular number is 465
theorem triangular_number_30_eq_465 : triangular_number 30 = 465 := by
  sorry

-- Theorem to prove the perimeter dots for the 30th triangular number is 88
theorem perimeter_dots_30_eq_88 : perimeter_dots 30 = 88 := by
  sorry

end triangular_number_30_eq_465_perimeter_dots_30_eq_88_l385_385054


namespace bus_fare_max_profit_passenger_count_change_l385_385741

noncomputable def demand (p : ℝ) : ℝ := 3000 - 20 * p
noncomputable def train_fare : ℝ := 10
noncomputable def train_capacity : ℝ := 1000
noncomputable def bus_cost (y : ℝ) : ℝ := y + 5

theorem bus_fare_max_profit : 
  ∃ (p_bus : ℝ), 
  p_bus = 50.5 ∧ 
  p_bus * (demand p_bus - train_capacity) - bus_cost (demand p_bus - train_capacity) = 
  p_bus * (demand p_bus - train_capacity) - (demand p_bus - train_capacity + 5) := 
sorry

theorem passenger_count_change :
  (demand train_fare - train_capacity) + train_capacity - demand 75.5 = 500 :=
sorry

end bus_fare_max_profit_passenger_count_change_l385_385741


namespace Nancy_weighs_90_pounds_l385_385503

theorem Nancy_weighs_90_pounds (W : ℝ) (h1 : 0.60 * W = 54) : W = 90 :=
by
  sorry

end Nancy_weighs_90_pounds_l385_385503


namespace find_m_l385_385557

def a_n (n m : ℝ) : ℝ := (1 / 3) * n^3 - (5 / 4) * n^2 + 3 + m

theorem find_m (m : ℝ) :
  (∃ n : ℝ, 1 ≤ n ∧ (∀ k : ℝ, 1 ≤ k → a_n k m ≥ a_n n m) ∧ a_n n m = 1) → m = 1 / 3 := 
by
  sorry

end find_m_l385_385557


namespace cos_210_eq_neg_sqrt3_div_2_l385_385861

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l385_385861


namespace correct_statements_are_ACD_l385_385729

theorem correct_statements_are_ACD :
  (∀ (width : ℝ), narrower_width_implies_better_fit width) ∧
  (rA rB : ℝ) (h_rA : rA = 0.97) (h_rB : rB = -0.99)
    → ¬ (stronger_A_implies_correlation rA rB) ∧
  (∀ (R2 : ℝ), smaller_R2_implies_worse_fit R2) ∧
  (num_products : ℕ) (defective_products : ℕ) (selected_products : ℕ)
    (h_num : num_products = 10) (h_def : defective_products = 3) (h_sel : selected_products = 2),
    probability_exactly_one_defective num_products defective_products selected_products = 7 / 15 :=
by
  sorry  -- Proof is not required

end correct_statements_are_ACD_l385_385729


namespace dartboard_even_score_probability_l385_385409

theorem dartboard_even_score_probability :
  let inner_radius := 4
      outer_radius := 8
      inner_values := [3, 4, 3]
      outer_values := [4, 3, 3]
      inner_area := (Real.pi * inner_radius^2) / 3
      outer_area := (Real.pi * (outer_radius^2 - inner_radius^2)) / 3
      even_regions := [4, 4, 3]
      odd_regions := [3, 3, 3]
      total_area := Real.pi * outer_radius^2
      probability_even := (inner_area + 2 * outer_area) / total_area
      probability_odd := (2 * inner_area + outer_area) / total_area
  in (probability_even^2 + probability_odd^2 = 37 / 72) := by
  sorry

end dartboard_even_score_probability_l385_385409


namespace factorize_x_squared_minus_1_l385_385098

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by
  -- Proof goes here
  sorry

end factorize_x_squared_minus_1_l385_385098


namespace other_root_of_quadratic_l385_385578

theorem other_root_of_quadratic (a : ℝ) :
  (∀ x, x^2 + a * x - 2 = 0 → x = -1) → ∃ m, x = m ∧ m = 2 :=
by
  sorry

end other_root_of_quadratic_l385_385578


namespace cos_210_eq_neg_sqrt_3_div_2_l385_385832

theorem cos_210_eq_neg_sqrt_3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by
  -- leave proof as sorry
  sorry

end cos_210_eq_neg_sqrt_3_div_2_l385_385832


namespace factorial_fraction_l385_385062

theorem factorial_fraction : (fact 8 + fact 9) / fact 7 = 80 :=
by
  sorry

end factorial_fraction_l385_385062


namespace problem_correct_statements_l385_385545

noncomputable def e1 (a b : ℝ) (h : a > b ∧ b > 0) : ℝ := (Real.sqrt (a^2 - b^2)) / a
noncomputable def e2 (a b : ℝ) (h : a > b ∧ b > 0) : ℝ := (Real.sqrt (a^2 + b^2)) / a

theorem problem_correct_statements (a b : ℝ) (h : a > b ∧ b > 0):
  (let e1 := e1 a b h in
  let e2 := e2 a b h in
  (e1 * e2 < 1) ∧ 
  (e1^2 + e2^2 = 2) ∧ 
  (e1 + e2 < 2)) :=
  sorry

end problem_correct_statements_l385_385545


namespace cosh_le_exp_sqr_l385_385128

open Real

theorem cosh_le_exp_sqr {x k : ℝ} : (∀ x : ℝ, cosh x ≤ exp (k * x^2)) ↔ k ≥ 1/2 :=
sorry

end cosh_le_exp_sqr_l385_385128


namespace vector_BA_complex_number_l385_385534

theorem vector_BA_complex_number :
  let OA := Complex.ofReal 2 - 3 * Complex.I
  let OB := -3 * Complex.ofReal + 2 * Complex.I
  let BA := OA - OB
in BA = 5 - 5 * Complex.I :=
by
  let OA := Complex.ofReal 2 - 3 * Complex.I
  let OB := -3 * Complex.ofReal + 2 * Complex.I
  let BA := OA - OB
  show BA = 5 - 5 * Complex.I
  sorry

end vector_BA_complex_number_l385_385534


namespace product_evaluation_l385_385893

theorem product_evaluation :
  (∏ n in Finset.range 99, (n * (n + 3)) / ((n + 1) * (n + 1))) = (51 / 50) :=
by sorry

end product_evaluation_l385_385893


namespace points_equal_and_concyclic_l385_385805

variables {A B_0 B_1 C_0 C_1 P_0 Q_0 P_1 Q_1 P_2 : Type*}
variables [Elliptical (B_0 B_1 : A)]

-- Given conditions
def ellipse_intersects (B_0 B_1 : A) (C_0 C_1 : A) (A B : A) : Prop :=
  intersects (ellipse B_0 B_1) (line_segment A B)

variable (ell_cond : ellipse_intersects B_0 B_1 C_0 C_1 A B_1)
variable (on_AB0 : point_on_ray P_0 A B_0)
variable (on_C1B0 : point_on_ray Q_0 C_1 B_0)
variable (eq_B0P0_Q0 : eq_distance B_0 P_0 B_0 Q_0)
variable (on_B1A : point_on_ray P_1 B_1 A)
variable (eq_C1Q0_P1 : eq_distance C_1 Q_0 C_1 P_1)
variable (on_B1C0 : point_on_ray Q_1 B_1 C_0)
variable (eq_B1P1_Q1 : eq_distance B_1 P_1 B_1 Q_1)
variable (on_AB0_P2 : point_on_ray P_2 A B_0)
variable (eq_C0Q1_P2 : eq_distance C_0 Q_1 C_0 P_2)

-- Statement to prove P_0 = P_2 and P_0, Q_0, Q_1, P_1 are concyclic
theorem points_equal_and_concyclic :
  P_0 = P_2 ∧ concyclic P_0 Q_0 Q_1 P_1 :=
begin
  sorry
end

end points_equal_and_concyclic_l385_385805


namespace break_even_min_production_quantity_l385_385690

open Real

noncomputable def total_cost (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2
noncomputable def total_revenue (x : ℝ) : ℝ := 25 * x

theorem break_even_min_production_quantity :
  ∃ x ∈ Ioo 0 240, (total_revenue x = total_cost x) ∧ (∀ y ∈ Ioo 0 240, total_revenue y ≥ total_cost y → y ≥ 150) :=
sorry

end break_even_min_production_quantity_l385_385690


namespace tangent_line_proof_minimum_a_proof_l385_385548

noncomputable def f (x : ℝ) := 2 * Real.log x - 3 * x^2 - 11 * x

def tangent_equation_correct : Prop :=
  let y := f 1
  let slope := (2 / 1 - 6 * 1 - 11)
  (slope = -15) ∧ (y = -14) ∧ (∀ x y, y = -15 * (x - 1) + -14 ↔ 15 * x + y - 1 = 0)

def minimum_a_correct : Prop :=
  ∃ a : ℤ, 
    (∀ x, f x ≤ (a - 3) * x^2 + (2 * a - 13) * x - 2) ↔ (a = 2)

theorem tangent_line_proof : tangent_equation_correct := sorry

theorem minimum_a_proof : minimum_a_correct := sorry

end tangent_line_proof_minimum_a_proof_l385_385548


namespace sum_of_all_possible_values_of_g1_l385_385284

noncomputable def g (x : ℝ) : ℝ := 6069 * x

theorem sum_of_all_possible_values_of_g1 (g : ℝ → ℝ) (h1 : ¬is_constant g) 
  (h2 : ∀ x ≠ 0, g(x - 1) + g(x) + g(x + 1) = (g(x) ^ 2) / (2024 * x)) : 
  g 1 = 6069 := 
sorry

end sum_of_all_possible_values_of_g1_l385_385284


namespace measuring_spoon_set_price_l385_385197

theorem measuring_spoon_set_price
  (cookies_sold : ℕ) (price_per_cookie : ℝ)
  (cupcakes_sold : ℕ) (price_per_cupcake : ℝ)
  (sets_bought : ℕ) (money_left : ℝ) :
  cookies_sold = 40 → price_per_cookie = 0.8 →
  cupcakes_sold = 30 → price_per_cupcake = 2 →
  sets_bought = 2 → money_left = 79 →
  let total_earnings := (cookies_sold * price_per_cookie) + (cupcakes_sold * price_per_cupcake)
      total_spent := total_earnings - money_left
      price_per_set := total_spent / sets_bought
  in
  price_per_set = 6.5 :=
by
  intros h1 h2 h3 h4 h5 h6
  simp [h1, h2, h3, h4, h5, h6]
  sorry

end measuring_spoon_set_price_l385_385197


namespace factorize_expr_l385_385897

theorem factorize_expr (a b : ℝ) : a^2 - 2 * a * b = a * (a - 2 * b) := 
by 
  sorry

end factorize_expr_l385_385897


namespace natural_number_solutions_l385_385500

theorem natural_number_solutions :
  {x y z : ℕ // x ≤ y ∧ y ≤ z ∧ x * y + y * z + z * x = 80 }.to_finset.card = 6 := by
  sorry

end natural_number_solutions_l385_385500


namespace altitudes_product_le_six_volume_l385_385641

theorem altitudes_product_le_six_volume 
  (T : Type) [tetrahedron T] (m1 m2 m3 : ℝ) (V : ℝ)
  (h1 : altitude T m1) (h2 : altitude T m2) (h3 : altitude T m3) (vol : volume T V) : 
  6 * V ≥ m1 * m2 * m3 ∧
  (6 * V = m1 * m2 * m3 ↔ right_angled_tetrahedron T) :=
by
  sorry

end altitudes_product_le_six_volume_l385_385641


namespace vector_calc_l385_385976

-- Definitions of the vectors a and b
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

-- Statement to prove that 2a - b = (5, 7)
theorem vector_calc : 2 • a - b = (5, 7) :=
by {
  -- Proof will be filled here
  sorry
}

end vector_calc_l385_385976


namespace second_triangle_weight_approximation_l385_385432

-- Define the basic properties of the first right triangle
def triangle1_base : ℝ := 3
def triangle1_hypotenuse : ℝ := 5
def triangle1_weight : ℝ := 12

-- Define the basic properties of the second right triangle
def triangle2_base : ℝ := 5
def triangle2_hypotenuse : ℝ := 7

-- Given the conditions, we want to prove the weight of the second triangle
theorem second_triangle_weight_approximation :
  let h1 := Real.sqrt (triangle1_hypotenuse^2 - triangle1_base^2),
      area1 := (triangle1_base * h1) / 2,
      h2 := Real.sqrt (triangle2_hypotenuse^2 - triangle2_base^2),
      area2 := (triangle2_base * h2) / 2,
      weight2 := (area2 * triangle1_weight) / area1
  in weight2 ≈ 24.5 := sorry

end second_triangle_weight_approximation_l385_385432


namespace sum_of_series_l385_385597

theorem sum_of_series (a b c : ℕ) (h1 : a = 3) (h2 : b = 11) (h3 : c = 10) :
  ∀ n : ℕ, n > 0 → 
  ( (finset.range n).sum (λ k, (k+1)*((k+2)^2)) = (n * (n + 1) / 12) * (a * n^2 + b * n + c) ) :=
by
  intro n hn
  rw [h1, h2, h3]
  sorry

end sum_of_series_l385_385597


namespace verify_equal_lengths_l385_385365

theorem verify_equal_lengths (D F1 F2: Point) (B C: Segment) (h: D F2 = B C) : D F1 = B C := 
sorry

end verify_equal_lengths_l385_385365


namespace eccentricity_range_l385_385962

open Real

namespace Hyperbola

variables {a b e : ℝ}

-- condition: given the hyperbola (x^2)/(a^2) - (y^2)/(b^2) = 1 with a > 0 and b > 0
def hyperbola (x y : ℝ) : Prop := (x^2) / (a^2) - (y^2) / (b^2) = 1
def positive_a : Prop := a > 0
def positive_b : Prop := b > 0

-- condition: a line is drawn through the left focus F and intersects the hyperbola at points A and B
def line_through_left_focus_intersects (F A B : ℝ × ℝ) : Prop := sorry

-- condition: |AB| = 4b
def ab_distance_eq_4b (A B : ℝ × ℝ) : Prop := dist A B = 4 * b

-- condition: there are exactly two such lines
def exactly_two_such_lines : Prop := sorry

-- prove that the range of eccentricity e is (1, √5 / 2) ∪ (√5, +∞)
theorem eccentricity_range :
  positive_a →
  positive_b →
  (∃ (A B : ℝ × ℝ), line_through_left_focus_intersects F A B ∧ ab_distance_eq_4b A B ∧ exactly_two_such_lines) →
  (1 < e ∧ e < sqrt 5 / 2) ∨ (sqrt 5 < e) :=
sorry

end Hyperbola

end eccentricity_range_l385_385962


namespace find_equation_of_circle_M_find_equation_of_line_l_l385_385514

noncomputable def A := (-Real.sqrt 3, 0)
noncomputable def B := (Real.sqrt 3, 0)
noncomputable def C := (0, -3)

def circle_M (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*y - 3 = 0

def line_l (E F : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (E.1 = k * E.2 ∧ F.1 = k * F.2) ∧ (Real.dist E F = Real.sqrt 15)

theorem find_equation_of_circle_M :
  circle_M A.1 A.2 ∧ circle_M B.1 B.2 ∧ circle_M C.1 C.2 :=
by sorry

theorem find_equation_of_line_l (E F : ℝ × ℝ) (h_intersect : circle_M E.1 E.2 ∧ circle_M F.1 F.2) :
  line_l E F → ∃ k : ℝ, k = Real.sqrt 3 ∨ k = -Real.sqrt 3 :=
by sorry

end find_equation_of_circle_M_find_equation_of_line_l_l385_385514


namespace problem_statement_l385_385287

variable (p q r s : ℝ) (ω : ℂ)

theorem problem_statement (hp : p ≠ -1) (hq : q ≠ -1) (hr : r ≠ -1) (hs : s ≠ -1) 
  (hω : ω ^ 4 = 1) (hω_ne : ω ≠ 1)
  (h_eq : (1 / (p + ω) + 1 / (q + ω) + 1 / (r + ω) + 1 / (s + ω)) = 3 / ω^2) :
  1 / (p + 1) + 1 / (q + 1) + 1 / (r + 1) + 1 / (s + 1) = 3 := 
by sorry

end problem_statement_l385_385287


namespace max_min_value_of_a_l385_385015

theorem max_min_value_of_a 
  (a b c d : ℝ) 
  (h1 : a + b + c + d = 3) 
  (h2 : a^2 + 2 * b^2 + 3 * c^2 + 6 * d^2 = 5) : 
  1 ≤ a ∧ a ≤ 2 := 
sorry

end max_min_value_of_a_l385_385015


namespace choose_subset_of_men_l385_385793

open Classical

theorem choose_subset_of_men (n : ℕ) :
  ∃ P ⊆ (Finset.powersetLen (Finset.univ : Finset (Fin (Nat.choose(2*n, n)))) (n+1)),
    (∀ x ∈ P, ∀ y ∈ P, x ≠ y → knows x y) ∨ (∀ x ∈ P, ∀ y ∈ P, x ≠ y → ¬ knows x y) :=
sorry

end choose_subset_of_men_l385_385793


namespace distance_between_pulley_centers_l385_385449

theorem distance_between_pulley_centers (R1 R2 CD : ℝ) (R1_pos : R1 = 10) (R2_pos : R2 = 6) (CD_pos : CD = 30) :
  ∃ AB : ℝ, AB = 2 * Real.sqrt 229 :=
by
  sorry

end distance_between_pulley_centers_l385_385449


namespace james_vacuuming_hours_l385_385252

/-- James spends some hours vacuuming and 3 times as long on the rest of his chores. 
    He spends 12 hours on his chores in total. -/
theorem james_vacuuming_hours (V : ℝ) (h : V + 3 * V = 12) : V = 3 := 
sorry

end james_vacuuming_hours_l385_385252


namespace harold_betty_choose_3_common_l385_385631

open BigOperators

-- Definition of combinations
def comb (n k : ℕ) : ℕ := nat.choose n k

-- The total number of ways each can choose 6 books from 12 books
def total_choices : ℕ := comb 12 6 * comb 12 6

-- The number of successful outcomes where exactly 3 books are the same
def successful_outcomes : ℕ := comb 12 3 * comb 9 3 * comb 9 3

-- The probability Harold and Betty choose exactly 3 books in common
def probability (total successful : ℕ) : ℚ := successful / total

theorem harold_betty_choose_3_common :
  probability total_choices successful_outcomes = 220 / 1215 :=
by
  rw [total_choices, successful_outcomes, probability]
  exact sorry

end harold_betty_choose_3_common_l385_385631


namespace number_of_bouquets_l385_385761

theorem number_of_bouquets : ∃ n, n = 9 ∧ ∀ x y : ℕ, 3 * x + 2 * y = 50 → (x < 17) ∧ (x % 2 = 0 → y = (50 - 3 * x) / 2) :=
by
  sorry

end number_of_bouquets_l385_385761


namespace sum_first_ten_terms_arithmetic_l385_385270

def arithmetic_sequence_sum (a₁ : ℤ) (n : ℕ) (d : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem sum_first_ten_terms_arithmetic (a₁ a₂ a₆ d : ℤ) 
  (h1 : a₁ = -2) 
  (h2 : a₂ + a₆ = 2) 
  (common_diff : d = 1) :
  arithmetic_sequence_sum a₁ 10 d = 25 :=
by
  rw [h1, common_diff]
  sorry

end sum_first_ten_terms_arithmetic_l385_385270


namespace sum_first_ten_terms_arithmetic_l385_385271

def arithmetic_sequence_sum (a₁ : ℤ) (n : ℕ) (d : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem sum_first_ten_terms_arithmetic (a₁ a₂ a₆ d : ℤ) 
  (h1 : a₁ = -2) 
  (h2 : a₂ + a₆ = 2) 
  (common_diff : d = 1) :
  arithmetic_sequence_sum a₁ 10 d = 25 :=
by
  rw [h1, common_diff]
  sorry

end sum_first_ten_terms_arithmetic_l385_385271


namespace encoding_inequality_unique_word_recovery_l385_385238

noncomputable def valid_encodings (k : Char → ℕ) : ℕ :=
if k 'Ш' = 0 ∧ k 'Е' = 0 ∧ k 'С' = 0 ∧ k 'Т' = 0 ∧ k 'Ь' = 0 ∧ k 'О' ∈ finset.range 10 then 10 else 0

theorem encoding_inequality (k : Char → ℕ) :
  (∀ c, (c = 'Е' ∨ c = 'О' ∨ c = 'С' ∨ c = 'Т' ∨ c = 'Ш' ∨ c = 'Ь') → k c ∈ finset.range 10) →
  k 'С' + k 'Т' + k 'О' ≥ k 'Ш' + k 'Е' + 2 * k 'С' + 2 * k 'Т' + k 'Ь' + k 'О' →
  valid_encodings k = 10 :=
by { intros, sorry }

theorem unique_word_recovery (k : Char → ℕ) :
  (∀ c, (c = 'Е' ∨ c = 'О' ∨ c = 'С' ∨ c = 'Т' ∨ c = 'Ш' ∨ c = 'Ь') → k c ∈ finset.range 10) →
  ¬(∀ (w1 w2 : List Char), (Sum (List.map k w1) = Sum (List.map k w2) → w1 = w2)) :=
by { intros, sorry }

end encoding_inequality_unique_word_recovery_l385_385238


namespace triangle_cos_sum_eq_one_l385_385245

theorem triangle_cos_sum_eq_one (a b c A B C : ℝ) (h1 : b^2 = a * c) 
    (h2 : sin A ≠ 0) (h3 : sin B ≠ 0) (h4 : sin C ≠ 0)
    (h5 : a / sin A = b / sin B) (h6 : b / sin B = c / sin C) 
    (h7 : sin B ^ 2 = sin A * sin C):
    cos (A - C) + cos B + cos (2 * B) = 1 := by
  sorry

end triangle_cos_sum_eq_one_l385_385245


namespace cards_drawn_to_product_even_l385_385080

theorem cards_drawn_to_product_even :
  ∃ n, (∀ (cards_drawn : Finset ℕ), 
    (cards_drawn ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}) ∧
    (cards_drawn.card = n) → 
    ¬ (∀ c ∈ cards_drawn, c % 2 = 1)
  ) ∧ n = 8 :=
by
  sorry

end cards_drawn_to_product_even_l385_385080


namespace sum_of_digits_of_third_smallest_multiple_l385_385261

def is_divisible_by_all (n : ℕ) : Prop :=
  ∀ m ∈ {1, 2, 3, 4, 5, 6, 7, 8}, m ∣ n

def third_smallest_multiple (n : ℕ) : ℕ :=
  3 * n

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem sum_of_digits_of_third_smallest_multiple :
  let m := 840 in
  sum_of_digits (third_smallest_multiple m) = 9 :=
by
  let m := 840
  show sum_of_digits (third_smallest_multiple m) = 9
  sorry

end sum_of_digits_of_third_smallest_multiple_l385_385261


namespace water_left_ratio_l385_385707

theorem water_left_ratio (h1: 2 * (30 / 10) = 6)
                        (h2: 2 * (30 / 10) = 6)
                        (h3: 4 * (60 / 10) = 24)
                        (water_left: ℕ)
                        (total_water_collected: ℕ) 
                        (h4: water_left = 18)
                        (h5: total_water_collected = 36) : 
  water_left * 2 = total_water_collected :=
by
  sorry

end water_left_ratio_l385_385707


namespace ratio_AXD_AXZY_l385_385520

-- Let's define the conditions first
variables (A B C D X Y Z : Type)
variables (S_AXD S_AXZY S_ABCX S_YBCX S_XYZ S_YBCZ : ℝ)

-- Conditions based on given problem
axiom AXD_to_ABCX : S_AXD / S_ABCX = 1 / 2
axiom AXY_to_YBCX : S_AXY / S_YBCX = 1 / 2
axiom XYZ_to_YBCZ : S_XYZ / S_YBCZ = 1 / 2
axiom areas_relation1 : S_YBCX = S_YBCZ + S_XYZ
axiom areas_relation2 : S_ABCX = S_AXY + S_YBCX
axiom areas_relation3 : S_AXD = 2 * S_AXD / S_ABCX * S_ABCX  -- Based on ratio
axiom S_AXZY_def : S_AXZY = S_ABCX + S_XYZ

-- The final proof statement
theorem ratio_AXD_AXZY : S_AXD / S_AXZY = 9 / 10 :=
  sorry

end ratio_AXD_AXZY_l385_385520


namespace factorize_x_squared_minus_one_l385_385105

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_x_squared_minus_one_l385_385105


namespace intersection_volume_of_pyramids_l385_385349

variables {A B C D E F : Point} -- Vertices of the lower base of the hexagon
variables {M N P Q R S : Point} -- Midpoints of the sides of the upper base of the hexagon
variables {O O1 : Point} -- Centers of the lower and upper bases respectively
variables {V : ℝ} -- Volume of the hexagonal prism

-- Define the regular hexagonal prism structure
def regular_hexagonal_prism (A B C D E F M N P Q R S O O1 : Point) : Prop :=
  is_hexagon A B C D E F ∧
  are_midpoints [M N P Q R S] [A B C D E F] ∧ 
  is_center O [A B C D E F] ∧
  is_center O1 [M N P Q R S]

-- The main proof statement
theorem intersection_volume_of_pyramids (prism : regular_hexagonal_prism A B C D E F M N P Q R S O O1) :
  volume (intersection (pyramid O1 [A, B, C, D, E, F]) (pyramid O [M, N, P, Q, R, S])) = (1 / 14) * V :=
sorry

end intersection_volume_of_pyramids_l385_385349


namespace smallest_part_2340_division_l385_385751

theorem smallest_part_2340_division :
  ∃ (A B C : ℕ), (A + B + C = 2340) ∧ 
                 (A / 5 = B / 7) ∧ 
                 (B / 7 = C / 11) ∧ 
                 (A = 510) :=
by 
  sorry

end smallest_part_2340_division_l385_385751


namespace max_points_tetrahedron_l385_385708

-- Define the problem domain and constraints.
noncomputable def points_in_space (P : ℕ) := P = 300

noncomputable def plane_splits_points (n : ℕ) (A B C D : Finset (Fin n)) :=
  (A.size = (300 / 2)) ∧
  (B.size = (300 / 2)) ∧
  (C.size = (300 / 2)) ∧
  (D.size = (300 / 2))

-- Define the goal: maximum number of points in the tetrahedron.
noncomputable def max_points_in_tetrahedron (A B C D : Finset (Fin n)) := 100

-- The theorem to be proven.
theorem max_points_tetrahedron (n : ℕ) (A B C D : Finset (Fin n)) :
  points_in_space 300 →
  plane_splits_points n A B C D →
  max_points_in_tetrahedron A B C D =
  100 :=
by {
  sorry -- The proof will go here.
}

end max_points_tetrahedron_l385_385708


namespace virus_count_after_5_hours_l385_385759

noncomputable def k : ℝ := Real.log 2

noncomputable def virus_growth (t : ℝ) : ℝ :=
  Real.exp (k * t)

lemma virus_doubles_in_half_hour : virus_growth 0.5 = 2 :=
by
  unfold virus_growth k
  rw [Real.exp_mul, ← Real.log_exp, Real.log_mul, Real.log_two, mul_comm, ← Real.exp_lt_iff_le_log_iff],
  -- Prove the rest or add sorry to skip the steps

theorem virus_count_after_5_hours : virus_growth 5 = 32 :=
by
  unfold virus_growth k
  rw [Real.exp_mul, Real.log_mul, Real.log_two, mul_comm, ← Real.exp_ln_eq, Real.exp_nat_eq],
  exact Real.nat_cast 32

end virus_count_after_5_hours_l385_385759


namespace cos_210_eq_neg_sqrt_3_div_2_l385_385833

theorem cos_210_eq_neg_sqrt_3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by
  -- leave proof as sorry
  sorry

end cos_210_eq_neg_sqrt_3_div_2_l385_385833


namespace find_replacement_percentage_l385_385637

noncomputable def final_percentage_replacement_alcohol_solution (a₁ p₁ p₂ x : ℝ) : Prop :=
  let d := 0.4 -- gallons
  let final_solution := 1 -- gallon
  let initial_pure_alcohol := a₁ * p₁ / 100
  let remaining_pure_alcohol := initial_pure_alcohol - (d * p₁ / 100)
  let added_pure_alcohol := d * x / 100
  remaining_pure_alcohol + added_pure_alcohol = final_solution * p₂ / 100

theorem find_replacement_percentage :
  final_percentage_replacement_alcohol_solution 1 75 65 50 :=
by
  sorry

end find_replacement_percentage_l385_385637


namespace smallest_n_for_f_greater_than_21_l385_385280

def f (n : ℕ) : ℕ := Inf {k : ℕ | n^2 ∣ k!}

theorem smallest_n_for_f_greater_than_21 :
  ∃ (n : ℕ), (21 ∣ n) ∧ f(n) > 21 ∧ ∀ m, (21 ∣ m) → m < n → f(m) ≤ 21 :=
begin
  use 84,
  split,
  { sorry },  -- Proof that 21 divides 84
  split,
  { sorry },  -- Proof that f(84) > 21
  { sorry },  -- Proof that for any m which is a multiple of 21 and less than 84, f(m) ≤ 21
end

end smallest_n_for_f_greater_than_21_l385_385280


namespace complex_number_quadrant_l385_385944

theorem complex_number_quadrant :
  let z := (4 * Complex.i) / (1 + Complex.i) in
  (z.re > 0 ∧ z.im > 0) :=
by
  let z := (4 * Complex.i) / (1 + Complex.i)
  sorry

end complex_number_quadrant_l385_385944


namespace probability_all_white_balls_l385_385018

-- Definitions
def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7
def balls_drawn : ℕ := 7

-- Lean theorem statement
theorem probability_all_white_balls :
  (Nat.choose white_balls balls_drawn : ℚ) / (Nat.choose total_balls balls_drawn) = 8 / 6435 :=
sorry

end probability_all_white_balls_l385_385018


namespace max_min_product_l385_385638

-- Define the conditions of the problem
variable {a b c : ℝ} (h_ellipse : a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 = b^2 + c^2)
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in x^2 / a^2 + y^2 / b^2 = 1

variable (P : ℝ × ℝ) (hP : is_on_ellipse P)
variable (F1 F2 : ℝ × ℝ) (dist_F1_F2 : dist F1 F2 = 2 * c)
variable (dist_PF1 : ℝ) (dist_PF2 : ℝ)

-- Prove the statement
theorem max_min_product (h1 : dist_PF1 + dist_PF2 = 2 * a)
  : ∃ d, d = c^2 ∧
    ∀ (dist_PF1' dist_PF2' : ℝ),
      dist_PF1' + dist_PF2' = 2 * a → 
      (a - c ≤ dist_PF1' ∧ dist_PF1' ≤ a + c → 
        (dist_PF1' * dist_PF2' ≤ a^2 ∧ dist_PF1' * dist_PF2' ≥ b^2)) := sorry

end max_min_product_l385_385638


namespace possible_values_expression_l385_385157

theorem possible_values_expression (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d > 0) :
  ∃ v ∈ ({5, 1, -1} : set ℝ), 
  v = (a / |a|) + (b / |b|) + (c / |c|) + ((a * b * c) / |a * b * c|) + (d / |d|) :=
by
  sorry

end possible_values_expression_l385_385157


namespace product_probability_divisible_by_12_l385_385058

def eight_sided_dice : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

def is_divisible_by_12 (n : ℕ) : Prop := n % 12 = 0

def probability_divisible_by_12 (rolls : List ℕ) : ℚ := 
  if rolls.length = 8 then 
    let counts := List.countp (λ n, n∈ eight_sided_dice) rolls 
    -- Skipping detailed calculations of probability
    149 / 256
  else 
    0

theorem product_probability_divisible_by_12 :
  probability_divisible_by_12 [a, b, c, d, e, f, g, h means ∀ a, b, c, d, e, f, g, h ∈ eight_sided_dice := sorry

end product_probability_divisible_by_12_l385_385058


namespace proof_problem_l385_385297

-- Define the functions f and g
def f (x : ℝ) : ℝ := (2 * x^3 + 4 * x^2 + 5 * x + 7) / (x^2 + 2 * x + 3)
def g (x : ℝ) : ℝ := x + 2

-- Prove that f(g(-1)) + g(f(-1)) = 7
theorem proof_problem : f (g (-1)) + g (f (-1)) = 7 := by sorry

end proof_problem_l385_385297


namespace greatest_perfect_power_sum_l385_385813

def sum_c_d_less_500 : ℕ :=
  let c := 22
  let d := 2
  c + d

theorem greatest_perfect_power_sum :
  ∃ c d : ℕ, 0 < c ∧ 1 < d ∧ c^d < 500 ∧
  ∀ x y : ℕ, 0 < x ∧ 1 < y ∧ x^y < 500 → x^y ≤ c^d ∧ (c + d = 24) :=
by
  sorry

end greatest_perfect_power_sum_l385_385813


namespace polynomial_degree_l385_385461

theorem polynomial_degree : 
  degree ((3 * (X ^ 5) + 2 * (X ^ 4) - (X ^ 2) + 5) * (5 * (X ^ 8) - 4 * (X ^ 5) + X - 20) - (X ^ 3 + 2) ^ 6) = 18 :=
by sorry

end polynomial_degree_l385_385461


namespace arc_length_of_sector_l385_385662

theorem arc_length_of_sector (r : ℝ) (θ : ℝ) (h : r = Real.pi ∧ θ = 120) : 
  r * θ / 180 * Real.pi = 2 * Real.pi * Real.pi / 3 :=
by
  sorry

end arc_length_of_sector_l385_385662


namespace evaluate_at_minus_two_l385_385502

noncomputable def evaluate_expression (x : ℝ) : ℝ :=
  (x - 3) / (3 * x^2 - 6 * x) * (x + 2 - 5 / (x - 2))

theorem evaluate_at_minus_two : (x : ℝ) 
  (hx : x * (x^2 - 4) = 0 ∧ x = -2) :
  evaluate_expression x = -1 / 6 := 
by
  cases hx with hcond hval
  simp only [evaluate_expression] at *
  sorry

end evaluate_at_minus_two_l385_385502


namespace eccentricity_of_ellipse_l385_385529

-- Let F1, F2 be the foci of ellipse C, and P be a point on C such that 
-- |PF1|, |F1F2|, and |PF2| form an arithmetic sequence, 
-- prove that the eccentricity e of C is 1/2.
theorem eccentricity_of_ellipse (C : Type)
  (F1 F2 : C)
  (P : C)
  (hF1F2 : dist F1 F2 = 2 * c)
  (hP : P ∈ set_of (λ p : C, dist p F1 + dist p F2 = 2 * a))
  (hArithSeq : 2 * dist F1 F2 = dist P F1 + dist P F2) :
  let e := c / a in e = 1 / 2 :=
by
  sorry

end eccentricity_of_ellipse_l385_385529


namespace jack_more_emails_morning_than_afternoon_l385_385251

def emails_afternoon := 3
def emails_morning := 5

theorem jack_more_emails_morning_than_afternoon :
  emails_morning - emails_afternoon = 2 :=
by
  sorry

end jack_more_emails_morning_than_afternoon_l385_385251


namespace exists_rectangle_discrepancy_ge_six_l385_385586

theorem exists_rectangle_discrepancy_ge_six :
  ∀ (n : ℕ) (N : ℕ), n = 10 ^ 2019 → N = 10 ^ 4038 → 
  ∃ (R : set (ℝ × ℝ)), (∃ x1 y1 x2 y2 : ℝ, 0 ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ n ∧ 0 ≤ y1 ∧ y1 ≤ y2 ∧ y2 ≤ n ∧ 
  R = {p : ℝ × ℝ | x1 ≤ p.1 ∧ p.1 ≤ x2 ∧ y1 ≤ p.2 ∧ p.2 ≤ y2}) →
  ∃ (pts : set (ℝ × ℝ)), (card pts = N) →
  let area R := (x2 - x1) * (y2- y1) in
  let num_points_in_R := card (R ∩ pts) in
  |num_points_in_R - N * area R| ≥ 6 := 
begin 
  sorry
end

end exists_rectangle_discrepancy_ge_six_l385_385586


namespace question1_if_quadratic_has_real_roots_then_range_of_m_question2_sufficient_condition_for_quadratic_has_real_roots_combined_proof_l385_385527

-- Definitions for the first part of the problem
def quadratic_has_real_roots (m : ℝ) : Prop :=
  let Δ := m^2 - 4 * (m^2 - 2*m + 1)
  Δ ≥ 0

def range_of_m (m : ℝ) : Prop :=
  (2/3) ≤ m ∧ m ≤ 2

-- Proof statement for the first part
theorem question1_if_quadratic_has_real_roots_then_range_of_m (m : ℝ) :
  quadratic_has_real_roots m → range_of_m m :=
sorry

-- Definitions for the second part of the problem
def sufficient_condition (a m : ℝ) : Prop :=
  1 - 2*a < m ∧ m < a + 1

theorem question2_sufficient_condition_for_quadratic_has_real_roots (a : ℝ) :
  (∀ m, sufficient_condition a m → quadratic_has_real_roots m) → (a ≤ 1/6) :=
sorry

-- Combined Answer: Prove real roots and sufficient condition relationship
theorem combined_proof (a : ℝ) (m : ℝ) :
  quadratic_has_real_roots m → range_of_m m ∧ (sufficient_condition a m → a ≤ 1/6) :=
begin
  intro h,
  split,
  { exact question1_if_quadratic_has_real_roots_then_range_of_m m h },
  { intro s,
    exact question2_sufficient_condition_for_quadratic_has_real_roots a s },
end

end question1_if_quadratic_has_real_roots_then_range_of_m_question2_sufficient_condition_for_quadratic_has_real_roots_combined_proof_l385_385527


namespace growth_operation_two_operations_growth_operation_four_operations_l385_385910

noncomputable def growth_operation_perimeter (initial_side_length : ℕ) (growth_operations : ℕ) := 
  initial_side_length * 3 * (4/3 : ℚ)^(growth_operations + 1)

theorem growth_operation_two_operations :
  growth_operation_perimeter 9 2 = 48 := by sorry

theorem growth_operation_four_operations :
  growth_operation_perimeter 9 4 = 256 / 3 := by sorry

end growth_operation_two_operations_growth_operation_four_operations_l385_385910


namespace find_omega2019_value_l385_385680

noncomputable def omega_n (n : ℕ) : ℝ := (2 * n - 1) * Real.pi / 2

theorem find_omega2019_value :
  omega_n 2019 = 4037 * Real.pi / 2 :=
by
  sorry

end find_omega2019_value_l385_385680


namespace range_of_m_l385_385172

noncomputable def f (x m : ℝ) : ℝ :=
  x^3 - 2 * Real.exp(1) * x^2 + m * x - Real.log x

theorem range_of_m (m : ℝ) :
  (∀ x > 0, f(x, m) > x) → (m > Real.exp(1)^2 + 1 / Real.exp(1) + 1) :=
  sorry

end range_of_m_l385_385172


namespace total_roses_l385_385765

theorem total_roses (a : ℕ) (x y k : ℕ) (h1 : 300 ≤ a) (h2 : a ≤ 400)
  (h3 : a = 21 * x + 13) (h4 : a = 15 * y - 8) (h5 : a + 8 = 105 * k) :
  a = 307 :=
sorry

end total_roses_l385_385765


namespace triangle_is_isosceles_l385_385016

-- Define a structure to represent a triangle with vertices and incircle radii for two smaller triangles
structure TriangleWithEqualInscribedCircles :=
  (A B C : Point)
  (D : Point) -- Point where angle bisector of ∠BAC intersects BC
  (r1 r2 : Real) -- Radii of inscribed circles of triangles ABD and ACD
  (angle_bisector_BAC : Angle)
  (incircle_radii_equal : r1 = r2)

-- Define the final theorem
theorem triangle_is_isosceles {ABC : TriangleWithEqualInscribedCircles}
  (h : ABC.incircle_radii_equal) : 
  dist ABC.A ABC.B = dist ABC.A ABC.C := sorry

end triangle_is_isosceles_l385_385016


namespace next_work_together_in_l385_385081

-- Definitions of working days for each member
def elena_works_every := 5
def felix_works_every := 8
def george_works_every := 9
def hanna_works_every := 11

-- Definition of the least common multiple function
def lcm (a b : Nat) : Nat :=
a * b / Nat.gcd a b

-- Compute the LCM of four numbers
def lcm4 (a b c d : Nat) : Nat :=
lcm (lcm a b) (lcm c d)

-- Theorem stating the next day all four team members will work together
theorem next_work_together_in : lcm4 elena_works_every felix_works_every george_works_every hanna_works_every = 3960 :=
by
  sorry

end next_work_together_in_l385_385081


namespace problem1_problem2_l385_385462

theorem problem1 :
  (-1 : ℤ) ^ 2021 + abs (-real.sqrt 3) + real.cbrt 8 - real.sqrt 16 = -3 + real.sqrt 3 :=
by
  sorry

theorem problem2 :
  (-1 : ℤ) ^ 2 - real.cbrt 27 + abs (1 - real.sqrt 2) = -5 + real.sqrt 2 :=
by
  sorry

end problem1_problem2_l385_385462


namespace solution_set_quadratic_inequality_l385_385888

theorem solution_set_quadratic_inequality :
  { x : ℝ | x^2 + 3 * x - 4 < 0 } = { x : ℝ | -4 < x ∧ x < 1 } :=
sorry

end solution_set_quadratic_inequality_l385_385888


namespace log_implication_exponentiation_not_necessary_sufficient_but_not_necessary_l385_385398

theorem log_implication (x y : ℝ) (h : Real.log x > Real.log y) : 
  10^x > 10^y :=
by sorry

theorem exponentiation_not_necessary (x y : ℝ) (h : 10^x > 10^y) : 
  ¬(Real.log x ≤ Real.log y) := 
by sorry

theorem sufficient_but_not_necessary (x y : ℝ) : 
  (Real.log x > Real.log y → 10^x > 10^y) ∧ 
  (∀ h : 10^x > 10^y, ¬(Real.log x ≤ Real.log y)) :=
by sorry

end log_implication_exponentiation_not_necessary_sufficient_but_not_necessary_l385_385398


namespace scientist_took_absent_mindedness_pills_l385_385740

-- Definitions for the conditions
variables {Ω : Type*} [measurable_space Ω] {P : measure Ω}

def R : event Ω := sorry --Event that the Scientist took pills for absent-mindedness
def A : event Ω := sorry --Event that knee pain stopped
def B : event Ω := sorry --Event that absent-mindedness disappeared

-- Given conditions
def P_R : ℝ := 1/2
def P_A_given_R : ℝ := 0.8
def P_B_given_R : ℝ := 0.05

def P_R_complement : ℝ := 1/2
def P_A_given_R_complement : ℝ := 0.9
def P_B_given_R_complement : ℝ := 0.02

-- Joint probabilities
def P_R_A_B : ℝ := P_R * P_A_given_R * P_B_given_R
def P_R_complement_A_B : ℝ := P_R_complement * P_A_given_R_complement * P_B_given_R_complement

-- Event that both A and B happen
def P_A_B : ℝ := P_R_A_B + P_R_complement_A_B

-- Required conditional probability
noncomputable def P_R_given_A_B : ℝ := P_R_A_B / P_A_B

-- Theorem we want to prove
theorem scientist_took_absent_mindedness_pills :
  P_R_given_A_B = 0.69 :=
sorry

end scientist_took_absent_mindedness_pills_l385_385740


namespace correct_sqrt_evaluation_l385_385381

theorem correct_sqrt_evaluation:
  2 * Real.sqrt 2 - Real.sqrt 2 = Real.sqrt 2 :=
by 
  sorry

end correct_sqrt_evaluation_l385_385381


namespace part1_part2_l385_385652

theorem part1 (a b c : ℤ) (h1 : |a| < 10^6) (h2 : |b| < 10^6) (h3 : |c| < 10^6) (nz : ¬(a = 0 ∧ b = 0 ∧ c = 0)) :
  |(a : ℝ) + b * Real.sqrt 2 + c * Real.sqrt 3| > 10^(-21) :=
sorry

theorem part2 : ∃ a b c : ℤ, |(a : ℝ) + b * Real.sqrt 2 + c * Real.sqrt 3| < 10^(-11) :=
sorry

end part1_part2_l385_385652


namespace quadratic_has_real_solutions_iff_l385_385073

theorem quadratic_has_real_solutions_iff (m : ℝ) :
  ∃ x y : ℝ, (y = m * x + 3) ∧ (y = (3 * m - 2) * x ^ 2 + 5) ↔ 
  (m ≤ 12 - 8 * Real.sqrt 2) ∨ (m ≥ 12 + 8 * Real.sqrt 2) :=
by
  sorry

end quadratic_has_real_solutions_iff_l385_385073


namespace train_length_is_correct_l385_385796

noncomputable def convert_speed (speed_kmh : ℕ) : ℝ :=
  (speed_kmh : ℝ) * 5 / 18

noncomputable def relative_speed (train_speed_kmh man's_speed_kmh : ℕ) : ℝ :=
  convert_speed train_speed_kmh + convert_speed man's_speed_kmh

noncomputable def length_of_train (train_speed_kmh man's_speed_kmh : ℕ) (time_seconds : ℝ) : ℝ := 
  relative_speed train_speed_kmh man's_speed_kmh * time_seconds

theorem train_length_is_correct :
  length_of_train 60 6 29.997600191984645 = 550 :=
by
  sorry

end train_length_is_correct_l385_385796


namespace group_formation_count_l385_385357

def total_students := 5
def all_student_heights := {1, 2, 3, 4, 5}

theorem group_formation_count : 
  ∃ (A B : Finset ℕ), A.nonempty ∧ B.nonempty ∧ A ∩ B = ∅ ∧ 
    (∀ a ∈ A, ∀ b ∈ B, a < b) ∧ all_student_heights = A ∪ B ∧ 
    (A.card + B.card = total_students) ∧ 
    (A.card + B.card = 2^total_students - 1) := sorry

end group_formation_count_l385_385357


namespace hyperbola_focal_product_l385_385150

-- Define the hyperbola with given equation and point P conditions
def Hyperbola (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1 }

-- Define properties of vectors related to foci
def perpendicular (v1 v2 : ℝ × ℝ) := (v1.1 * v2.1 + v1.2 * v2.2 = 0)

-- Define the point-focus distance product condition
noncomputable def focalProduct (P F1 F2 : ℝ × ℝ) := (Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2)) * (Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2))

theorem hyperbola_focal_product :
  ∀ (a b : ℝ) (F1 F2 P : ℝ × ℝ),
  Hyperbola a b P ∧ perpendicular (P - F1) (P - F2) ∧
  -- Assuming a parabola property ties F1 with a specific value
  ((P.1 - F1.1)^2 + (P.2 - F1.2)^2 = 4 * (Real.sqrt  ((P.1 - F2.1)^2 + (P.2 - F2.2)^2))) →
  focalProduct P F1 F2 = 14 := by
  sorry

end hyperbola_focal_product_l385_385150


namespace spending_example_l385_385485

theorem spending_example (X : ℝ) (h₁ : X + 2 * X + 3 * X = 120) : X = 20 := by
  sorry

end spending_example_l385_385485


namespace part_a_part_b_part_c_part_d_part_e_part_f_l385_385620

variables (a b c p : ℝ)
variables (G O I I_a : Point)
variables (R r r_a : ℝ)
variable (h_sum : a + b + c = 2 * p)

-- Proof (a)
theorem part_a :
  a^2 + b^2 + c^2 = 2 * p^2 - 2 * r^2 - 8 * R * r := sorry

-- Proof (b)
theorem part_b :
  dist O G ^ 2 = R^2 - (1 / 9) * (a^2 + b^2 + c^2) := sorry

-- Proof (c)
theorem part_c :
  dist I G ^ 2 = (1 / 9) * (p^2 + 5 * r^2 - 16 * R * r) := sorry

-- Proof (d)
theorem part_d :
  dist O I ^ 2 = R^2 - 2 * R * r := sorry

-- Proof (e)
theorem part_e :
  dist O I_a ^ 2 = R^2 + 2 * R * r_a := sorry

-- Proof (f)
theorem part_f :
  dist I I_a ^ 2 = 4 * R * (r_a - r) := sorry

end part_a_part_b_part_c_part_d_part_e_part_f_l385_385620


namespace closest_integer_to_500_sum_l385_385115

theorem closest_integer_to_500_sum : 
  let S := 500 * (∑ n in finset.range 15000 \ finset.range 2, 1 / (n^2 - 4)) in
  abs (S - 135) <= 0.5 :=
by
  sorry

end closest_integer_to_500_sum_l385_385115


namespace quadratic_roots_sum_squares_l385_385156

theorem quadratic_roots_sum_squares {a b : ℝ} 
  (h₁ : a + b = -1) 
  (h₂ : a * b = -5) : 
  2 * a^2 + a + b^2 = 16 :=
by sorry

end quadratic_roots_sum_squares_l385_385156


namespace tangent_ellipse_hyperbola_l385_385044

theorem tangent_ellipse_hyperbola {x y m : ℝ} 
  (h1 : 4 * x^2 + y^2 = 4) 
  (h2 : x^2 - m * (y - 2)^2 = 4) : 
  m = 1 / 3 := 
begin
  sorry
end

end tangent_ellipse_hyperbola_l385_385044


namespace cos_210_eq_neg_sqrt_3_div_2_l385_385828

theorem cos_210_eq_neg_sqrt_3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by
  -- leave proof as sorry
  sorry

end cos_210_eq_neg_sqrt_3_div_2_l385_385828


namespace initial_red_marbles_l385_385417

theorem initial_red_marbles (R : ℕ) (blue_marbles_initial : ℕ) (red_marbles_removed : ℕ) :
  blue_marbles_initial = 30 →
  red_marbles_removed = 3 →
  (R - red_marbles_removed) + (blue_marbles_initial - 4 * red_marbles_removed) = 35 →
  R = 20 :=
by
  intros h_blue h_red h_total
  sorry

end initial_red_marbles_l385_385417


namespace g_45_l385_385676

noncomputable def g : ℝ → ℝ := sorry

axiom g_functional (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x * y) = g x / y
axiom g_30 : g 30 = 30

theorem g_45 : g 45 = 20 := by
  -- proof to be completed
  sorry

end g_45_l385_385676


namespace sin_2_bac_is_1_l385_385426

theorem sin_2_bac_is_1
  (AB AC : ℝ)
  (ABC_is_isosceles_right : AB = AC ∧ AB = 2)
  (BCD_is_right : ∀ BC CD BD : ℝ, ∠BCD = 90 ∧ BC = 2 * Real.sqrt 2)
  (equal_perimeters : ∀ P₁ P₂ : ℝ, P₁ = 4 + 2 * Real.sqrt 2 ∧ P₂ = 4 + 2 * Real.sqrt 2) :
  let ∠BAC := 45 in
  Real.sin (2 * ∠BAC) = 1 :=
by
  intros
  sorry

end sin_2_bac_is_1_l385_385426


namespace circle_problem_is_solved_l385_385826

def circle_problem_pqr : ℕ :=
  let n := 3 / 2;
  let p := 3;
  let q := 1;
  let r := 4;
  p + q + r

theorem circle_problem_is_solved : circle_problem_pqr = 8 :=
by {
  -- Additional context of conditions can be added here if necessary
  sorry
}

end circle_problem_is_solved_l385_385826


namespace residue_of_11_1201_mod_19_l385_385370

theorem residue_of_11_1201_mod_19 :
    (11^1201) % 19 = 1 := by
sorries

end residue_of_11_1201_mod_19_l385_385370


namespace factorize_difference_of_squares_l385_385108

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_difference_of_squares_l385_385108


namespace max_factors_of_2_in_J_max_value_of_M_max_value_of_M_achieved_l385_385072

def J (a : ℕ) : ℕ := 10^5 + 2^a

def M (a : ℕ) : ℕ :=
  nat.factorization (J a) 2

theorem max_factors_of_2_in_J (a : ℕ) : M a ≤ 5 :=
by sorry

theorem max_value_of_M : ∀ a : ℕ, M a ≤ 5 :=
by sorry

theorem max_value_of_M_achieved : ∃ a : ℕ, M a = 5 :=
by sorry

end max_factors_of_2_in_J_max_value_of_M_max_value_of_M_achieved_l385_385072


namespace schoolchildren_number_l385_385739

theorem schoolchildren_number (n m S : ℕ) 
  (h1 : S = 22 * n + 3)
  (h2 : S = (n - 1) * m)
  (h3 : n ≤ 18)
  (h4 : m ≤ 36) : 
  S = 135 := 
sorry

end schoolchildren_number_l385_385739


namespace divisibility_problem_l385_385882

theorem divisibility_problem (a b k : ℕ) :
  (a = 7 * k^2 ∧ b = 7 * k) ∨ (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) →
  a * b^2 + b + 7 ∣ a^2 * b + a + b := by
  intro h
  cases h
  case inl h1 =>
    rw [h1.1, h1.2]
    sorry
  case inr h2 =>
    cases h2
    case inl h21 =>
      rw [h21.1, h21.2]
      sorry
    case inr h22 =>
      rw [h22.1, h22.2]
      sorry

end divisibility_problem_l385_385882


namespace probability_odd_function_l385_385525

-- Definitions of the given functions
def f1 (x : ℝ) : ℝ := x
def f2 (x : ℝ) : ℝ := Real.sin x
def f3 (x : ℝ) : ℝ := Real.cos x
def f4 (x : ℝ) : ℝ := Real.log (x + Real.sqrt (1 + x^2))

-- The main statement to be proved
theorem probability_odd_function : 
  ∃ p : ℚ, p = 1 / 2 ∧ (∃ f g ∈ {f1, f2, f3, f4}, ¬ (f = g) ∧ (is_odd (f * g))) :=
sorry

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

end probability_odd_function_l385_385525
